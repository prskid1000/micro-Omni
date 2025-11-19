
import argparse, json, os, torch
from torch import nn
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from omni.vision_encoder import ViTTiny
from omni.training_utils import (
    set_seed, get_lr_scheduler, clip_gradients, SimpleLogger, validate_loss, 
    check_gradient_explosion, cleanup_old_checkpoints, ImgCapDataset,
    load_checkpoint, setup_resume_data_loading, calculate_resume_position,
    ValidationSkipSamplesContext
)
from tqdm import tqdm


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
    # Check for existing vocabulary checkpoint (resumable)
    vocab_checkpoint_path = os.path.join(cfg["save_dir"], "vocab_build_checkpoint.json")
    word_freq = {}
    start_idx = 0
    
    if os.path.exists(vocab_checkpoint_path):
        print(f"Found vocabulary checkpoint, resuming from checkpoint...")
        try:
            import json
            checkpoint_data = json.load(open(vocab_checkpoint_path, 'r'))
            word_freq = checkpoint_data.get("word_freq", {})
            start_idx = checkpoint_data.get("last_processed_idx", 0)
            print(f"  Resuming from caption {start_idx:,}")
        except Exception as e:
            print(f"  Warning: Could not load checkpoint: {e}, starting from beginning")
            word_freq = {}
            start_idx = 0
    
    if start_idx == 0:
        print("Building vocabulary from captions (processing in chunks, resumable)...")
    
    # Build vocabulary incrementally without loading all captions into memory
    total_captions = len(ds)
    checkpoint_freq = 10000  # Save checkpoint every N captions
    
    for i in range(start_idx, total_captions):
        _, caption = ds[i]
        words = caption.lower().split()
        for word in words:
            word_freq[word] = word_freq.get(word, 0) + 1
        
        # Save checkpoint periodically (resumable)
        if (i + 1) % checkpoint_freq == 0:
            try:
                import json
                checkpoint_data = {
                    "word_freq": word_freq,
                    "last_processed_idx": i + 1
                }
                json.dump(checkpoint_data, open(vocab_checkpoint_path, 'w'))
                print(f"  Checkpoint saved: processed {i+1:,}/{total_captions:,} captions...")
            except Exception as e:
                print(f"  Warning: Could not save checkpoint: {e}")
        
        # Progress indicator for large datasets
        if (i + 1) % 10000 == 0:
            print(f"  Processed {i+1:,}/{total_captions:,} captions...")
    
    # Final checkpoint save
    try:
        import json
        checkpoint_data = {
            "word_freq": word_freq,
            "last_processed_idx": total_captions
        }
        json.dump(checkpoint_data, open(vocab_checkpoint_path, 'w'))
    except:
        pass
    
    # Create simple word-based vocabulary
    word_to_idx = {}
    
    # Keep top N most frequent words
    sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
    vocab_size = min(vocab_size, len(sorted_words))
    word_to_idx = {word: idx+1 for idx, (word, _) in enumerate(sorted_words[:vocab_size-1])}  # +1 for padding=0
    word_to_idx["<UNK>"] = vocab_size - 1
    print(f"Vocabulary size: {len(word_to_idx)}")
    
    # Clean up checkpoint after successful completion
    if os.path.exists(vocab_checkpoint_path):
        try:
            os.remove(vocab_checkpoint_path)
        except:
            pass
    
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
    step = 0
    step, resume_from = load_checkpoint(
        cfg["save_dir"], 
        "vision_step_", 
        device, 
        logger,
        state_dict_loaders={
            "vit": (vit, vit.load_state_dict),
            "img_proj": (img_proj, img_proj.load_state_dict),
            "text_proj": (text_proj, text_proj.load_state_dict),
            "text_embed": (text_embed, text_embed.load_state_dict),
            "optimizer": (opt, opt.load_state_dict),
            "scheduler": (scheduler, scheduler.load_state_dict),
            "scaler": (scaler, scaler.load_state_dict) if scaler is not None else None
        }
    )
    # Handle scaler separately if needed
    if step > 0 and resume_from and scaler is not None:
        checkpoint = torch.load(resume_from, map_location=device)
        if isinstance(checkpoint, dict) and "scaler" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler"])
    
    # Update skip_samples for dataset if resuming
    batch_size = cfg.get("batch_size", 8)
    new_train_dl = setup_resume_data_loading(
        train_ds, step, batch_size, logger,
        train_dl_kwargs={
            "shuffle": True,
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
        # Create progress bar with correct starting position when resuming mid-epoch
        if epoch == start_epoch and start_batch_idx > 0:
            pbar = tqdm(train_dl, desc=f"epoch{epoch} step{step}", initial=start_batch_idx, total=steps_per_epoch)
        else:
            pbar = tqdm(train_dl, desc=f"epoch{epoch} step{step}", total=steps_per_epoch)
        
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
            pbar.set_description(f"epoch{epoch} step{step} batch{batch_idx}")
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
                with ValidationSkipSamplesContext(train_ds):
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
        with ValidationSkipSamplesContext(train_ds):
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
            logger.info(f"Model saved to {cfg['save_dir']} at end of epoch {epoch}, step {step}")
            return

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--config", required=True)
    cfg = json.load(open(ap.parse_args().config))
    main(cfg)
