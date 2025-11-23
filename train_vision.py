
import argparse, json, os, torch
from torch import nn
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from omni.vision_encoder import ViTTiny
from omni.thinker import ThinkerLM
from omni.tokenizer import BPETokenizer
from omni.utils import (
    set_seed, get_lr_scheduler, clip_gradients, SimpleLogger, validate_loss, 
    check_gradient_explosion, ImgCapDataset,
    load_checkpoint, setup_resume_data_loading, calculate_resume_position,
    ValidationSkipSamplesContext, find_checkpoint, save_training_metadata, load_training_metadata
)
from tqdm import tqdm


def main(cfg):
    # Set random seed for reproducibility
    seed = cfg.get("seed", 42)
    set_seed(seed)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    save_dir = cfg.get("save_dir", "checkpoints/vision_tiny")
    os.makedirs(save_dir, exist_ok=True)
    train_manifest = cfg.get("train_manifest", "data/images/production_annotations.json")
    image_root = cfg.get("image_root", "data/images")
    
    model_name = "vision"
    metadata = load_training_metadata(save_dir, model_name)
    
    # torch.compile() support (optional, PyTorch 2.0+)
    use_compile = cfg.get("use_compile", False)
    
    d_model = cfg.get("d_model", 128)
    vit = ViTTiny(cfg.get("img_size", 224), cfg.get("patch", 16), d_model, cfg.get("n_layers", 4), cfg.get("n_heads", 2), cfg.get("d_ff", 512), cfg.get("dropout", 0.1), compile_model=use_compile).to(device)
    
    # Use contrastive learning (CLIP-style) for proper vision-language alignment
    # Project image CLS token to embedding space for contrastive learning
    embed_dim = cfg.get("embed_dim", d_model)  # Embedding dimension for contrastive learning
    img_proj = nn.Linear(d_model, embed_dim).to(device)
    
    # Configurable: Use Thinker model or simple tokenizer+embedding for text encoding
    use_thinker_for_text = cfg.get("use_thinker_for_text", True)
    thinker_ckpt_dir = cfg.get("thinker_ckpt", "checkpoints/thinker_tiny")
    thinker_cfg = cfg.get("thinker", {})
    ctx_len = cfg.get("ctx_len", 512)
    vocab_size = cfg.get("vocab_size", 32000)
    
    think = None
    text_embed = None
    
    if use_thinker_for_text:
        # Use Thinker model for text encoding (frozen) - better contextual embeddings
        print("Using Thinker model for text encoding (recommended)")
        thinker_d_model = cfg.get("thinker_d_model", 256)
        text_proj = nn.Linear(thinker_d_model, embed_dim).to(device)
        
        # Load Thinker model architecture
        think = ThinkerLM(
            thinker_cfg.get("vocab_size", 32000),
            thinker_cfg.get("n_layers", 4),
            thinker_cfg.get("d_model", 256),
            thinker_cfg.get("n_heads", 4),
            thinker_cfg.get("d_ff", 1024),
            thinker_cfg.get("dropout", 0.1),
            thinker_cfg.get("rope_theta", 10000),
            ctx_len,
            use_gqa=thinker_cfg.get("use_gqa", False),
            use_swiglu=thinker_cfg.get("use_swiglu", True),
            use_moe=thinker_cfg.get("use_moe", False),
            num_experts=thinker_cfg.get("num_experts", 8),
            num_experts_per_tok=thinker_cfg.get("num_experts_per_tok", 2),
            compile_model=False  # Don't compile Thinker for vision training
        ).to(device)
        
        # Load trained Thinker if available
        thinker_path, thinker_ckpt = find_checkpoint(thinker_ckpt_dir, "thinker.pt", "thinker_step_", device)
        if thinker_ckpt is not None:
            if isinstance(thinker_ckpt, dict):
                if "model" in thinker_ckpt:
                    think.load_state_dict(thinker_ckpt["model"])
                elif "thinker" in thinker_ckpt:
                    think.load_state_dict(thinker_ckpt["thinker"])
                else:
                    think.load_state_dict(thinker_ckpt)
            else:
                think.load_state_dict(thinker_ckpt)
            print(f"✓ Loaded trained Thinker from {thinker_path}")
        else:
            print("⚠ Warning: Thinker checkpoint not found, using untrained Thinker")
        
        # Freeze Thinker - we only use it for text encoding, not training
        for param in think.parameters():
            param.requires_grad = False
        think.eval()
        print("✓ Thinker model frozen (used only for text encoding)")
    else:
        # Use simple tokenizer + embedding layer (lighter, faster, but less contextual)
        print("Using tokenizer + embedding layer for text encoding (lighter option)")
        text_proj = nn.Linear(d_model, embed_dim).to(device)
        # text_embed will be created after tokenizer is loaded
    
    # Load or train tokenizer
    tok_model_path = os.path.join(thinker_ckpt_dir, "tokenizer.model")
    if os.path.exists(tok_model_path):
        print(f"Loading tokenizer from {tok_model_path}")
        tok = BPETokenizer(tok_model_path)
        vocab_size = tok.sp.get_piece_size()
        print(f"Tokenizer vocab size: {vocab_size}")
    else:
        # Train tokenizer from captions if not found
        print(f"Tokenizer not found at {tok_model_path}, training new tokenizer from captions...")
        os.makedirs(thinker_ckpt_dir, exist_ok=True)
        temp_caption_file = os.path.join(save_dir, ".temp_captions.txt")
        with open(train_manifest, 'r', encoding='utf-8') as f:
            manifest_items = json.load(f)
        with open(temp_caption_file, 'w', encoding='utf-8') as f:
            for item in manifest_items:
                caption = item.get("caption", "").strip()
                if caption:
                    f.write(caption + "\n")
        print(f"  Training tokenizer on {len(manifest_items):,} captions...")
        BPETokenizer.train_new(temp_caption_file, tok_model_path, vocab_size=thinker_cfg.get("vocab_size", vocab_size))
        tok = BPETokenizer(tok_model_path)
        vocab_size = tok.sp.get_piece_size()
        print(f"✓ Tokenizer trained and saved to {tok_model_path}")
        # Clean up temp file
        try:
            os.remove(temp_caption_file)
        except:
            pass
    
    # Create token embedding layer if not using Thinker
    if not use_thinker_for_text:
        text_embed = nn.Embedding(vocab_size, d_model).to(device)
    
    # Optimizer: include text_embed if using simple mode
    opt_params = list(vit.parameters()) + list(img_proj.parameters()) + list(text_proj.parameters())
    if text_embed is not None:
        opt_params += list(text_embed.parameters())
    opt = torch.optim.AdamW(opt_params, lr=cfg.get("lr", 3e-4), weight_decay=cfg.get("wd", 0.01))
    
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
    
    def encode_caption(caption):
        """Encode caption using tokenizer and either Thinker model or simple embedding"""
        # Tokenize caption
        token_ids = tok.encode(caption)
        # Truncate to context length
        token_ids = token_ids[:ctx_len-1]  # -1 for BOS token
        # Add BOS token (typically token ID 1)
        token_ids = [1] + token_ids  # BOS=1
        if len(token_ids) == 0:
            token_ids = [1]  # At least BOS token
        # Convert to tensor
        token_tensor = torch.tensor(token_ids, device=device, dtype=torch.long)
        
        if use_thinker_for_text:
            # Use Thinker model for contextual embeddings (better quality)
            token_tensor = token_tensor.unsqueeze(0)  # (1, T)
            with torch.no_grad():
                # Use Thinker to get contextual embeddings
                text_emb = think(idx=token_tensor)  # (1, T, thinker_d_model)
            # Average pooling over sequence to get single embedding
            return text_emb.squeeze(0).mean(dim=0)  # (thinker_d_model,)
        else:
            # Use simple token embeddings (lighter, faster)
            token_embeds = text_embed(token_tensor)  # (T, d_model)
            # Average pooling over sequence to get single embedding
            return token_embeds.mean(dim=0)  # (d_model,)

    # Split dataset for validation
    val_split = cfg.get("val_split", 0.1)  # 10% for validation
    
    train_ds = ImgCapDataset(
        train_manifest, 
        image_root, 
        cfg.get("img_size", 224),
        shuffle_buffer_size=cfg.get("shuffle_buffer_size", 10000),
        seed=seed,
        skip_samples=0
    )
    train_ds._val_split = val_split
    train_ds._val_mode = False  # Training mode
    
    val_ds = ImgCapDataset(
        train_manifest, 
        image_root, 
        cfg.get("img_size", 224),
        shuffle_buffer_size=0,  # No shuffling for validation
        seed=seed,  # Same seed for consistent hash-based split
        skip_samples=0
    )
    val_ds._val_split = val_split
    val_ds._val_mode = True  # Validation mode
    
    # Approximate sizes for logging (will count if needed)
    try:
        total_size = train_ds.get_length()
        train_size = int(total_size * (1 - val_split))
        val_size = total_size - train_size
    except:
        train_size = val_size = None  # Unknown size
    
    # Note: shuffle=False for IterableDataset (shuffling handled internally)
    train_dl = DataLoader(train_ds, batch_size=cfg.get("batch_size", 8), shuffle=False, num_workers=cfg.get("num_workers", 2), drop_last=cfg.get("drop_last", True))
    val_dl = DataLoader(val_ds, batch_size=cfg.get("batch_size", 8), shuffle=False, num_workers=cfg.get("num_workers", 2), drop_last=cfg.get("drop_last", True))
    
    # Initialize logger
    logger = SimpleLogger("Vision")
    
    step=0
    vit.train()
    img_proj.train()
    text_proj.train()
    # Thinker is frozen (eval mode)
    max_epochs = cfg.get("max_epochs", 9999)
    print_freq = cfg.get("print_freq", 100)
    checkpoint_freq = cfg.get("checkpoint_freq", 500)  # Save checkpoint every N steps
    val_freq = cfg.get("val_freq", 200)  # Validate every N steps
    
    # Resume from checkpoint if available
    step = 0
    # Resume from checkpoint if available
    step = 0
    step, metadata = load_checkpoint(
        save_dir, 
        model_name, 
        device, 
        logger,
        state_dict_loaders={
            "vit": (vit, vit.load_state_dict),
            "img_proj": (img_proj, img_proj.load_state_dict),
            "text_proj": (text_proj, text_proj.load_state_dict),
            "optimizer": (opt, opt.load_state_dict),
            "scheduler": (scheduler, scheduler.load_state_dict),
            "scaler": (scaler, scaler.load_state_dict) if scaler is not None else None
        }
    )
    # Handle scaler separately if needed
    # Handle scaler separately if needed
    if step > 0 and scaler is not None:
        model_path = os.path.join(save_dir, f"{model_name}.pt")
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=device)
            if isinstance(checkpoint, dict) and "scaler" in checkpoint:
                scaler.load_state_dict(checkpoint["scaler"])
    
    # Update skip_samples for dataset if resuming
    batch_size = cfg.get("batch_size", 8)
    new_train_dl = setup_resume_data_loading(
        train_ds, step, batch_size, logger,
        train_dl_kwargs={
            "num_workers": cfg.get("num_workers", 2),
            "drop_last": cfg.get("drop_last", True)
        }
    )
    if new_train_dl is not None:
        train_dl = new_train_dl
    
    logger.training_start(cfg["max_steps"], train_size, val_size)
    
    # Calculate steps per epoch and determine starting epoch/position
    # For IterableDataset, we can't use len() directly, so calculate from dataset size
    batch_size = cfg.get("batch_size", 8)
    drop_last = cfg.get("drop_last", True)
    if train_size is not None:
        steps_per_epoch = train_size // batch_size
        if not drop_last and train_size % batch_size != 0:
            steps_per_epoch += 1
    else:
        # Fallback: use a large number if size is unknown (for progress bar)
        # The actual training will work fine, just progress bar won't be accurate
        steps_per_epoch = 1000000  # Large placeholder
    initial_step = step
    start_epoch, start_batch_idx = calculate_resume_position(step, steps_per_epoch)
    if step > 0:
        logger.info(f"Resuming from step {step} (epoch {start_epoch}, batch {start_batch_idx}/{steps_per_epoch})")
    
    for epoch in range(start_epoch, max_epochs):
        # Recreate DataLoader for each epoch since IterableDatasets are exhausted after one iteration
        # skip_samples is automatically reset to 0 by the dataset after first iteration
        if epoch > start_epoch:
            train_dl = DataLoader(train_ds, batch_size=cfg.get("batch_size", 8), shuffle=False, 
                                 num_workers=cfg.get("num_workers", 2), 
                                 drop_last=cfg.get("drop_last", True))
        
        # Create progress bar with correct starting position when resuming mid-epoch
        remaining_epochs = max_epochs - epoch - 1
        pbar_desc = f"epoch{epoch}/{max_epochs-1} (remaining:{remaining_epochs}) step{step}"
        if epoch == start_epoch and start_batch_idx > 0:
            pbar = tqdm(train_dl, desc=pbar_desc, initial=start_batch_idx, total=steps_per_epoch)
        else:
            pbar = tqdm(train_dl, desc=pbar_desc, total=steps_per_epoch)
        
        # Start enumeration from the correct position when resuming mid-epoch
        enum_start = start_batch_idx if (epoch == start_epoch and start_batch_idx > 0) else 0
        for batch_idx, (img, cap) in enumerate(pbar, start=enum_start):
            # Skip batches if resuming mid-epoch
            # batch_idx already represents the position in the epoch when enum_start > 0
            if epoch == start_epoch and initial_step > 0:
                current_batch_step = epoch * steps_per_epoch + batch_idx
                if current_batch_step < initial_step:
                    continue
            
            # Update progress bar description
            remaining_epochs = max_epochs - epoch - 1
            pbar.set_description(f"epoch{epoch}/{max_epochs-1} (remaining:{remaining_epochs}) step{step} batch{batch_idx}")
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
                
                # Gradient clipping first (already unscaled if using AMP)
                # Clip gradients to prevent explosion, then check if still too high
                try:
                    grad_norm_vit_before = clip_gradients(vit, max_grad_norm)
                    grad_norm_img_proj_before = clip_gradients(img_proj, max_grad_norm)
                    grad_norm_text_proj_before = clip_gradients(text_proj, max_grad_norm)
                    
                    # Check for gradient explosion AFTER clipping
                    # Use a higher threshold (10x max_grad_norm) since we've already clipped
                    # This allows clipping to fix most cases, only skip if truly exploded
                    explosion_threshold = max(100.0, max_grad_norm * 10)
                    grad_norm_vit_after, is_exploded_vit = check_gradient_explosion(vit, max_grad_norm=explosion_threshold, raise_on_error=False)
                    grad_norm_proj_after, is_exploded_proj = check_gradient_explosion(img_proj, max_grad_norm=explosion_threshold, raise_on_error=False)
                    grad_norm_text_proj_after, is_exploded_text_proj = check_gradient_explosion(text_proj, max_grad_norm=explosion_threshold, raise_on_error=False)
                    
                    if is_exploded_vit or is_exploded_proj or is_exploded_text_proj:
                        logger.error(f"Step {step}: Gradient explosion detected after clipping (vit: {grad_norm_vit_before:.2f}->{grad_norm_vit_after:.2f}, img_proj: {grad_norm_img_proj_before:.2f}->{grad_norm_proj_after:.2f}, text_proj: {grad_norm_text_proj_before:.2f}->{grad_norm_text_proj_after:.2f}). Skipping this batch.")
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
                
                # Optimizer step (gradients already clipped)
                if use_amp:
                    scaler.step(opt)
                    scaler.update()
                else:
                    opt.step()
                scheduler.step()
                opt.zero_grad()  # Clear gradients after stepping
                step += 1  # Increment step counter only when optimizer step occurs
            else:
                # Not accumulation step - just validate loss
                unscaled_loss = loss_val * accumulation_steps
                try:
                    validate_loss(unscaled_loss, min_loss=-1e6, max_loss=1e6)
                except RuntimeError as e:
                    logger.error(f"Step {step}: {e}")
                    logger.error("Skipping this batch due to invalid loss")
                    continue
            if step % print_freq == 0:
                current_lr = scheduler.get_last_lr()[0]
                unscaled_loss = loss_val * accumulation_steps
                logger.train_step(step, float(unscaled_loss), current_lr, epoch)
            
            # Periodic checkpointing
            # Periodic checkpointing
            if step % checkpoint_freq == 0 and step > 0:
                # Save model weights only (overwrite existing file)
                model_path = os.path.join(save_dir, f"{model_name}.pt")
                checkpoint_data = {
                    "vit": vit.state_dict(),
                    "img_proj": img_proj.state_dict(),
                    "text_proj": text_proj.state_dict(),
                    "optimizer": opt.state_dict(),
                    "scheduler": scheduler.state_dict(),
                }
                if scaler is not None:
                    checkpoint_data["scaler"] = scaler.state_dict()
                torch.save(checkpoint_data, model_path)
                
                # Save training metadata
                training_metadata = {
                    "step": step,
                    "epoch": epoch
                }
                save_training_metadata(save_dir, model_name, training_metadata)
                logger.checkpoint(step, model_path)
            
            # Validation
            if step % val_freq == 0 and step > 0:
                with ValidationSkipSamplesContext(train_ds):
                    vit.eval()
                    img_proj.eval()
                    text_proj.eval()
                    # Thinker is already in eval mode (frozen)
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
                    # Thinker remains in eval mode (frozen)
            
            if step >= cfg["max_steps"]:
                final_path = os.path.join(save_dir, f"{model_name}.pt")
                checkpoint_data = {
                    "vit": vit.state_dict(),
                    "img_proj": img_proj.state_dict(),
                    "text_proj": text_proj.state_dict(),
                    "optimizer": opt.state_dict(),
                    "scheduler": scheduler.state_dict(),
                }
                if scaler is not None:
                    checkpoint_data["scaler"] = scaler.state_dict()
                torch.save(checkpoint_data, final_path)
                
                # Save final training metadata
                training_metadata = {
                    "step": step,
                    "epoch": epoch
                }
                save_training_metadata(save_dir, model_name, training_metadata)
                logger.info(f"Final model saved to {cfg['save_dir']}")
                logger.training_end(step)
                return
        
        # Final validation at end of epoch
        with ValidationSkipSamplesContext(train_ds):
            vit.eval()
            img_proj.eval()
            text_proj.eval()
            # Thinker is already in eval mode (frozen)
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
            # Thinker remains in eval mode (frozen)
        
        # Save at end of epoch (checkpoint for resuming)
        # Save at end of epoch (checkpoint for resuming)
        final_path = os.path.join(cfg["save_dir"], f"{model_name}.pt")
        checkpoint_data = {
            "vit": vit.state_dict(),
            "img_proj": img_proj.state_dict(),
            "text_proj": text_proj.state_dict(),
            "optimizer": opt.state_dict(),
            "scheduler": scheduler.state_dict(),
        }
        if scaler is not None:
            checkpoint_data["scaler"] = scaler.state_dict()
        torch.save(checkpoint_data, final_path)
        
        # Save training metadata
        training_metadata = {
            "step": step,
            "epoch": epoch
        }
        save_training_metadata(save_dir, model_name, training_metadata)
        logger.info(f"Model saved to {cfg['save_dir']} at end of epoch {epoch}, step {step}")
        
        # Check if we've reached max_steps after epoch completion
        if step >= cfg["max_steps"]:
            logger.info(f"Reached max_steps={cfg['max_steps']}. Training complete.")
            logger.training_end(step)
            return
        
        # Continue to next epoch
        start_batch_idx = 0  # Reset batch index for new epoch

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--config", required=True)
    cfg = json.load(open(ap.parse_args().config))
    main(cfg)
