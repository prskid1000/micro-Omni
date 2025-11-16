
import argparse, json, os, torch, json as js
from torch import nn
from torch.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from omni.vision_encoder import ViTTiny
from omni.training_utils import set_seed, get_lr_scheduler, clip_gradients, SimpleLogger, validate_loss, check_gradient_explosion, cleanup_old_checkpoints
from tqdm import tqdm

class ImgCapDataset(Dataset):
    def __init__(self, manifest, image_root, img_size=224):
        self.manifest_path = manifest
        self.root = image_root
        self.tf = transforms.Compose([
            transforms.Resize((img_size,img_size)),
            transforms.ToTensor()
        ])
        # Build item offset index for JSON array
        # JSON arrays are structured, so we'll parse and store offsets to each object
        self.item_offsets = []
        self.item_lengths = []  # Store approximate lengths for seeking
        try:
            with open(manifest, 'rb') as f:
                content = f.read()
                # Find array start
                start_pos = content.find(b'[')
                if start_pos == -1:
                    raise ValueError("JSON manifest must be an array")
                # Parse JSON to find object boundaries (handle commas and whitespace)
                pos = start_pos + 1
                depth = 0
                obj_start = None
                in_string = False
                escape_next = False
                while pos < len(content):
                    byte = content[pos:pos+1]
                    if escape_next:
                        escape_next = False
                    elif byte == b'\\':
                        escape_next = True
                    elif byte == b'"' and not escape_next:
                        in_string = not in_string
                    elif not in_string:
                        if byte == b'{':
                            if depth == 0:
                                obj_start = pos
                            depth += 1
                        elif byte == b'}':
                            depth -= 1
                            if depth == 0 and obj_start is not None:
                                # Found complete object
                                self.item_offsets.append(obj_start)
                                self.item_lengths.append(pos - obj_start + 1)
                                obj_start = None
                    pos += 1
        except Exception:
            # If parsing fails, fall back to loading entire JSON
            pass
        
        # Fallback: if parsing failed, load entire JSON (for malformed JSON)
        if not self.item_offsets:
            self.items = js.load(open(manifest, 'r', encoding='utf-8'))
            self._use_fallback = True
        else:
            self._use_fallback = False
    
    def __len__(self):
        if self._use_fallback:
            return len(self.items)
        return len(self.item_offsets)
    
    def __getitem__(self, i):
        if self._use_fallback:
            it = self.items[i]
        else:
            # Read specific JSON object using offset
            with open(self.manifest_path, 'rb') as f:
                f.seek(self.item_offsets[i])
                # Read the object, handling potential trailing comma/whitespace
                obj_bytes = f.read(self.item_lengths[i])
                # Try to parse, if it fails due to trailing comma, read a bit more
                try:
                    it = js.loads(obj_bytes.decode('utf-8'))
                except:
                    # Read until next object or end of array
                    chunk = obj_bytes
                    while True:
                        next_byte = f.read(1)
                        if not next_byte or next_byte in [b'{', b']']:
                            break
                        chunk += next_byte
                    # Try to find and parse the JSON object in the chunk
                    chunk_str = chunk.decode('utf-8')
                    # Find first { and matching }
                    start = chunk_str.find('{')
                    if start != -1:
                        depth = 0
                        end = start
                        for j, char in enumerate(chunk_str[start:], start):
                            if char == '{':
                                depth += 1
                            elif char == '}':
                                depth -= 1
                                if depth == 0:
                                    end = j + 1
                                    break
                        it = js.loads(chunk_str[start:end])
                    else:
                        # Fallback: load entire JSON
                        self.items = js.load(open(self.manifest_path, 'r', encoding='utf-8'))
                        self._use_fallback = True
                        it = self.items[i]
        img = Image.open(os.path.join(self.root, it["image"])).convert("RGB")
        return self.tf(img), it["caption"]

def main(cfg):
    # Set random seed for reproducibility
    seed = cfg.get("seed", 42)
    set_seed(seed)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(cfg["save_dir"], exist_ok=True)
    ds = ImgCapDataset(cfg["train_manifest"], cfg["image_root"], cfg["img_size"])
    dl = DataLoader(ds, batch_size=cfg.get("batch_size", 8), shuffle=True, num_workers=cfg.get("num_workers", 2), drop_last=cfg.get("drop_last", True))
    # torch.compile() support (optional, PyTorch 2.0+)
    use_compile = cfg.get("use_compile", False)
    
    vit = ViTTiny(cfg["img_size"], cfg["patch"], cfg["d_model"], cfg["n_layers"], cfg["n_heads"], cfg["d_ff"], cfg["dropout"], compile_model=use_compile).to(device)
    
    # Use contrastive learning (CLIP-style) for proper vision-language alignment
    # Project image CLS token to embedding space for contrastive learning
    embed_dim = cfg.get("embed_dim", cfg["d_model"])  # Embedding dimension for contrastive learning
    img_proj = nn.Linear(cfg["d_model"], embed_dim).to(device)
    text_proj = nn.Linear(cfg["d_model"], embed_dim).to(device)  # For text embeddings (if using text encoder)
    
    # Simple text encoder using word embeddings (can be replaced with proper tokenizer)
    # For now, we'll use a simple approach: encode captions as bag-of-words
    vocab_size = cfg.get("vocab_size", 10000)  # Vocabulary size for caption encoding
    text_embed = nn.Embedding(vocab_size, cfg["d_model"]).to(device)
    
    opt = torch.optim.AdamW(list(vit.parameters())+list(img_proj.parameters())+list(text_proj.parameters())+list(text_embed.parameters()), 
                           lr=cfg["lr"], weight_decay=cfg["wd"])
    
    # Contrastive loss (InfoNCE)
    temperature = cfg.get("temperature", 0.07)  # Temperature for contrastive loss
    
    # Learning rate scheduler with warmup
    warmup_steps = cfg.get("warmup_steps", 500)
    max_steps = cfg.get("max_steps", 5000)
    scheduler = get_lr_scheduler(opt, warmup_steps, max_steps)
    
    # Gradient clipping
    max_grad_norm = cfg.get("max_grad_norm", 1.0)
    
    # Gradient accumulation
    accumulation_steps = cfg.get("gradient_accumulation_steps", 1)
    
    # Mixed precision training (AMP)
    use_amp = cfg.get("use_amp", True) and device == "cuda"
    scaler = GradScaler('cuda') if use_amp else None
    if use_amp:
        print("Mixed precision training (AMP) enabled")
    if accumulation_steps > 1:
        print(f"Gradient accumulation: {accumulation_steps} steps")
    
    # Simple vocabulary for caption encoding (create from training data)
    print("Building vocabulary from captions...")
    # Iterate through dataset to collect captions (works with lazy loading)
    all_captions = []
    for i in range(len(ds)):
        _, caption = ds[i]
        all_captions.append(caption)
    # Create simple word-based vocabulary
    word_to_idx = {}
    word_freq = {}
    for cap in all_captions:
        words = cap.lower().split()
        for word in words:
            word_freq[word] = word_freq.get(word, 0) + 1
    
    # Keep top N most frequent words
    sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
    vocab_size = min(vocab_size, len(sorted_words))
    word_to_idx = {word: idx+1 for idx, (word, _) in enumerate(sorted_words[:vocab_size-1])}  # +1 for padding=0
    word_to_idx["<UNK>"] = vocab_size - 1
    print(f"Vocabulary size: {len(word_to_idx)}")
    
    def encode_caption(caption):
        """Encode caption to bag-of-words representation"""
        words = caption.lower().split()
        # Simple bag-of-words: sum embeddings
        word_ids = [word_to_idx.get(word, word_to_idx["<UNK>"]) for word in words]
        if len(word_ids) == 0:
            word_ids = [word_to_idx["<UNK>"]]
        # Average pooling
        word_embeds = text_embed(torch.tensor(word_ids, device=device))
        return word_embeds.mean(dim=0)  # Average pooling

    # Split dataset for validation
    val_split = cfg.get("val_split", 0.1)  # 10% for validation
    total_size = len(ds)
    val_size = int(total_size * val_split)
    train_size = total_size - val_size
    train_ds, val_ds = torch.utils.data.random_split(ds, [train_size, val_size], generator=torch.Generator().manual_seed(seed))
    train_dl = DataLoader(train_ds, batch_size=cfg.get("batch_size", 8), shuffle=True, num_workers=cfg.get("num_workers", 2), drop_last=cfg.get("drop_last", True))
    val_dl = DataLoader(val_ds, batch_size=cfg.get("batch_size", 8), shuffle=False, num_workers=cfg.get("num_workers", 2), drop_last=False)
    
    # Initialize logger
    logger = SimpleLogger("Vision")
    
    step=0
    vit.train()
    img_proj.train()
    text_proj.train()
    text_embed.train()
    max_epochs = cfg.get("max_epochs", 9999)
    print_freq = cfg.get("print_freq", 100)
    checkpoint_freq = cfg.get("checkpoint_freq", 500)  # Save checkpoint every N steps
    val_freq = cfg.get("val_freq", 200)  # Validate every N steps
    
    # Resume from checkpoint if available
    resume_from = None
    if os.path.exists(cfg["save_dir"]):
        checkpoint_files = [f for f in os.listdir(cfg["save_dir"]) if f.startswith("vision_step_") and f.endswith(".pt")]
        if checkpoint_files:
            # Extract step numbers and find latest
            step_numbers = []
            for f in checkpoint_files:
                try:
                    step_num = int(f.replace("vision_step_", "").replace(".pt", ""))
                    step_numbers.append((step_num, f))
                except:
                    continue
            if step_numbers:
                step_numbers.sort(key=lambda x: x[0], reverse=True)
                resume_from = os.path.join(cfg["save_dir"], step_numbers[0][1])
                step = step_numbers[0][0]
                logger.info(f"Found checkpoint at step {step}, resuming from: {resume_from}")
                
                # Load full checkpoint state
                checkpoint = torch.load(resume_from, map_location=device)
                if isinstance(checkpoint, dict) and "vit" in checkpoint:
                    vit.load_state_dict(checkpoint["vit"])
                    if "img_proj" in checkpoint:
                        img_proj.load_state_dict(checkpoint["img_proj"])
                    if "text_proj" in checkpoint:
                        text_proj.load_state_dict(checkpoint["text_proj"])
                    if "text_embed" in checkpoint:
                        text_embed.load_state_dict(checkpoint["text_embed"])
                    if "optimizer" in checkpoint:
                        opt.load_state_dict(checkpoint["optimizer"])
                    if "scheduler" in checkpoint:
                        scheduler.load_state_dict(checkpoint["scheduler"])
                    if "scaler" in checkpoint and scaler is not None:
                        scaler.load_state_dict(checkpoint["scaler"])
                    if "step" in checkpoint:
                        step = checkpoint["step"]
                    logger.info(f"Resumed from step {step}")
                else:
                    # Legacy checkpoint format
                    if "vit" in checkpoint:
                        vit.load_state_dict(checkpoint["vit"])
                    logger.info(f"Loaded model weights from checkpoint (legacy format)")
    
    logger.training_start(cfg["max_steps"], train_size, val_size)
    
    # Skip to the correct epoch/step if resuming
    start_epoch = 0
    steps_per_epoch = len(train_dl)
    initial_step = step
    if step > 0:
        start_epoch = step // steps_per_epoch
        logger.info(f"Resuming from epoch {start_epoch}, step {step}")
    
    for epoch in range(start_epoch, max_epochs):
        for batch_idx, (img, cap) in enumerate(tqdm(train_dl, desc=f"epoch{epoch}")):
            # Skip batches if resuming mid-epoch
            if epoch == start_epoch and initial_step > 0:
                current_batch_step = epoch * steps_per_epoch + batch_idx
                if current_batch_step < initial_step:
                    continue
            img = img.to(device)
            B = img.shape[0]
            
            # Mark step begin for CUDAGraphs optimization
            if device == "cuda":
                torch.compiler.cudagraph_mark_step_begin()
            
            # Encode images and captions
            if use_amp:
                with autocast(device_type='cuda'):
                    cls, _ = vit(img)  # (B,1,d)
                    img_emb = img_proj(cls.squeeze(1))  # (B, embed_dim)
                    img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)  # L2 normalize
                    
                    # Encode captions
                    text_embs = torch.stack([encode_caption(c) for c in cap]).to(device)  # (B, d_model)
                    text_emb = text_proj(text_embs)  # (B, embed_dim)
                    text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)  # L2 normalize
                    
                    # Contrastive loss (InfoNCE)
                    # Similarity matrix: (B, B)
                    logits = torch.matmul(img_emb, text_emb.t()) / temperature  # (B, B)
                    labels = torch.arange(B, device=device)  # Positive pairs are on diagonal
                    loss = nn.CrossEntropyLoss()(logits, labels)
                    # Free intermediate tensors
                    del cls, img_emb, text_embs, text_emb, logits
            else:
                cls, _ = vit(img)  # (B,1,d)
                img_emb = img_proj(cls.squeeze(1))  # (B, embed_dim)
                img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)  # L2 normalize
                
                # Encode captions
                text_embs = torch.stack([encode_caption(c) for c in cap]).to(device)  # (B, d_model)
                text_emb = text_proj(text_embs)  # (B, embed_dim)
                text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)  # L2 normalize
                
                # Contrastive loss (InfoNCE)
                logits = torch.matmul(img_emb, text_emb.t()) / temperature  # (B, B)
                labels = torch.arange(B, device=device)  # Positive pairs are on diagonal
                loss = nn.CrossEntropyLoss()(logits, labels)
                # Free intermediate tensors
                del cls, img_emb, text_embs, text_emb, logits
            
            # Forward pass with gradient accumulation
            loss_scaled = loss / accumulation_steps  # Scale loss for accumulation
            
            if use_amp:
                scaler.scale(loss_scaled).backward()
            else:
                loss_scaled.backward()
            
            # Detach loss for logging (free computation graph)
            loss_val = loss.detach()
            del loss
            
            # Gradient accumulation: only step optimizer every N steps
            if (step + 1) % accumulation_steps == 0:
                # Unscale before checking gradients
                if use_amp:
                    scaler.unscale_(opt)
            
                # Validate loss value (unscaled)
                unscaled_loss = loss_val * accumulation_steps
                try:
                    validate_loss(unscaled_loss, min_loss=-1e6, max_loss=1e6)
                except RuntimeError as e:
                    logger.error(f"Step {step}: {e}")
                    logger.error("Skipping this batch due to invalid loss")
                    opt.zero_grad()
                    if use_amp:
                        scaler.update()
                    continue
                
                # Check for gradient explosion before clipping (after unscaling if AMP)
                try:
                    grad_norm_vit, is_exploded_vit = check_gradient_explosion(vit, max_grad_norm=100.0, raise_on_error=False)
                    grad_norm_proj, is_exploded_proj = check_gradient_explosion(img_proj, max_grad_norm=100.0, raise_on_error=False)
                    if is_exploded_vit or is_exploded_proj:
                        logger.error(f"Step {step}: Gradient explosion detected (vit={grad_norm_vit:.2f}, proj={grad_norm_proj:.2f}). Skipping this batch.")
                        opt.zero_grad()  # Clear gradients
                        if use_amp:
                            scaler.update()  # Update scaler even though we skipped (unscale was called)
                        continue
                except RuntimeError as e:
                    logger.error(f"Step {step}: {e}")
                    opt.zero_grad()  # Clear gradients
                    if use_amp:
                        scaler.update()  # Update scaler even though we skipped (unscale was called)
                    continue
                
                # Gradient clipping (already unscaled if using AMP)
                if use_amp:
                    clip_gradients(vit, max_grad_norm)
                    clip_gradients(img_proj, max_grad_norm)
                    clip_gradients(text_proj, max_grad_norm)
                    clip_gradients(text_embed, max_grad_norm)
                    scaler.step(opt)
                    scaler.update()
                else:
                    clip_gradients(vit, max_grad_norm)
                    clip_gradients(img_proj, max_grad_norm)
                    clip_gradients(text_proj, max_grad_norm)
                    clip_gradients(text_embed, max_grad_norm)
                    opt.step()
                scheduler.step()
                opt.zero_grad()  # Clear gradients after stepping
            else:
                # Not accumulation step - just validate loss
                unscaled_loss = loss_val * accumulation_steps
                try:
                    validate_loss(unscaled_loss, min_loss=-1e6, max_loss=1e6)
                except RuntimeError as e:
                    logger.error(f"Step {step}: {e}")
                    logger.error("Skipping this batch due to invalid loss")
                    continue
            step+=1
            if step % print_freq == 0:
                current_lr = scheduler.get_last_lr()[0]
                unscaled_loss = loss_val * accumulation_steps
                logger.train_step(step, float(unscaled_loss), current_lr, epoch)
            
            # Periodic checkpointing
            if step % checkpoint_freq == 0 and step > 0:
                checkpoint_path = os.path.join(cfg["save_dir"], f"vision_step_{step}.pt")
                checkpoint_data = {
                    "vit": vit.state_dict(),
                    "img_proj": img_proj.state_dict(),
                    "text_proj": text_proj.state_dict(),
                    "text_embed": text_embed.state_dict(),
                    "optimizer": opt.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "step": step
                }
                if scaler is not None:
                    checkpoint_data["scaler"] = scaler.state_dict()
                torch.save(checkpoint_data, checkpoint_path)
                logger.checkpoint(step, checkpoint_path)
                # Clean up old checkpoints (keep only last one)
                cleanup_old_checkpoints(cfg["save_dir"], "vision_step_", keep_last_n=1)
            
            # Validation
            if step % val_freq == 0 and step > 0:
                vit.eval()
                img_proj.eval()
                text_proj.eval()
                text_embed.eval()
                val_loss_sum = 0.0
                val_count = 0
                with torch.no_grad():
                    for val_img, val_cap in val_dl:
                        val_img = val_img.to(device)
                        val_B = val_img.shape[0]
                        if use_amp:
                            with autocast(device_type='cuda'):
                                val_cls, _ = vit(val_img)
                                val_img_emb = img_proj(val_cls.squeeze(1))
                                val_img_emb = val_img_emb / val_img_emb.norm(dim=-1, keepdim=True)
                                
                                val_text_embs = torch.stack([encode_caption(c) for c in val_cap]).to(device)
                                val_text_emb = text_proj(val_text_embs)
                                val_text_emb = val_text_emb / val_text_emb.norm(dim=-1, keepdim=True)
                                
                                val_logits = torch.matmul(val_img_emb, val_text_emb.t()) / temperature
                                val_labels = torch.arange(val_B, device=device)
                                val_loss = nn.CrossEntropyLoss()(val_logits, val_labels)
                        else:
                            val_cls, _ = vit(val_img)
                            val_img_emb = img_proj(val_cls.squeeze(1))
                            val_img_emb = val_img_emb / val_img_emb.norm(dim=-1, keepdim=True)
                            
                            val_text_embs = torch.stack([encode_caption(c) for c in val_cap]).to(device)
                            val_text_emb = text_proj(val_text_embs)
                            val_text_emb = val_text_emb / val_text_emb.norm(dim=-1, keepdim=True)
                            
                            val_logits = torch.matmul(val_img_emb, val_text_emb.t()) / temperature
                            val_labels = torch.arange(val_B, device=device)
                            val_loss = nn.CrossEntropyLoss()(val_logits, val_labels)
                        
                        # Validate validation loss
                        try:
                            validate_loss(val_loss, min_loss=-1e6, max_loss=1e6)
                            val_loss_sum += float(val_loss.detach())
                            val_count += 1
                            # Free validation tensors
                            del val_cls, val_img_emb, val_text_embs, val_text_emb, val_logits, val_loss
                        except RuntimeError as e:
                            logger.warning(f"Step {step}: Invalid validation loss: {e}")
                            # Continue with other validation batches
                        if val_count >= 10:  # Limit validation batches
                            break
                
                avg_val_loss = val_loss_sum / val_count
                logger.val_step(step, avg_val_loss, epoch)
                
                vit.train()
                img_proj.train()
                text_proj.train()
                text_embed.train()
            
            if step >= cfg["max_steps"]:
                final_path = os.path.join(cfg["save_dir"], "vision.pt")
                checkpoint_data = {
                    "vit": vit.state_dict(),
                    "img_proj": img_proj.state_dict(),
                    "text_proj": text_proj.state_dict(),
                    "text_embed": text_embed.state_dict(),
                    "optimizer": opt.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "step": step
                }
                if scaler is not None:
                    checkpoint_data["scaler"] = scaler.state_dict()
                torch.save(checkpoint_data, final_path)
                logger.info(f"Final model saved to {cfg['save_dir']}")
                logger.training_end(step)
                return
        
        # Final validation at end of epoch
        vit.eval()
        img_proj.eval()
        text_proj.eval()
        text_embed.eval()
        val_loss_sum = 0.0
        val_count = 0
        with torch.no_grad():
            for val_img, val_cap in val_dl:
                val_img = val_img.to(device)
                val_B = val_img.shape[0]
                if use_amp:
                    with autocast(device_type='cuda'):
                        val_cls, _ = vit(val_img)
                        val_img_emb = img_proj(val_cls.squeeze(1))
                        val_img_emb = val_img_emb / val_img_emb.norm(dim=-1, keepdim=True)
                        
                        val_text_embs = torch.stack([encode_caption(c) for c in val_cap]).to(device)
                        val_text_emb = text_proj(val_text_embs)
                        val_text_emb = val_text_emb / val_text_emb.norm(dim=-1, keepdim=True)
                        
                        val_logits = torch.matmul(val_img_emb, val_text_emb.t()) / temperature
                        val_labels = torch.arange(val_B, device=device)
                        val_loss = nn.CrossEntropyLoss()(val_logits, val_labels)
                else:
                    val_cls, _ = vit(val_img)
                    val_img_emb = img_proj(val_cls.squeeze(1))
                    val_img_emb = val_img_emb / val_img_emb.norm(dim=-1, keepdim=True)
                    
                    val_text_embs = torch.stack([encode_caption(c) for c in val_cap]).to(device)
                    val_text_emb = text_proj(val_text_embs)
                    val_text_emb = val_text_emb / val_text_emb.norm(dim=-1, keepdim=True)
                    
                    val_logits = torch.matmul(val_img_emb, val_text_emb.t()) / temperature
                    val_labels = torch.arange(val_B, device=device)
                    val_loss = nn.CrossEntropyLoss()(val_logits, val_labels)
                
                # Validate validation loss
                try:
                    validate_loss(val_loss, min_loss=-1e6, max_loss=1e6)
                    val_loss_sum += float(val_loss.detach())
                    val_count += 1
                    # Free validation tensors
                    del val_cls, val_img_emb, val_text_embs, val_text_emb, val_logits, val_loss
                except RuntimeError as e:
                    logger.warning(f"Epoch {epoch}: Invalid validation loss: {e}")
                    # Continue with other validation batches
        
        avg_val_loss = val_loss_sum / max(val_count, 1)
        logger.epoch_end(epoch, train_loss=None, val_loss=avg_val_loss)
        vit.train()
        img_proj.train()
        text_proj.train()
        text_embed.train()
        
        # Save at end of epoch if max_steps not reached
        if step < cfg["max_steps"]:
            final_path = os.path.join(cfg["save_dir"], "vision.pt")
            checkpoint_data = {
                "vit": vit.state_dict(),
                "head": head.state_dict(),
                "optimizer": opt.state_dict(),
                "scheduler": scheduler.state_dict(),
                "step": step
            }
            if scaler is not None:
                checkpoint_data["scaler"] = scaler.state_dict()
            torch.save(checkpoint_data, final_path)
            logger.info(f"Model saved to {cfg['save_dir']} at end of epoch {epoch}, step {step}")
            return

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--config", required=True)
    cfg = json.load(open(ap.parse_args().config))
    main(cfg)
