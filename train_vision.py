import argparse, json, os, torch
from torch import nn
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from omni.vision_encoder import ViTTiny, TransformerTextEncoder
from omni.thinker import ThinkerLM
from omni.tokenizer import BPETokenizer
from omni.utils import (
    set_seed, get_lr_scheduler, clip_gradients, SimpleLogger, validate_loss, 
    ImgCapDataset, EMA,
    load_checkpoint, setup_resume_data_loading, calculate_resume_position,
    ValidationSkipSamplesContext, find_checkpoint, save_training_metadata, load_training_metadata,
    TrainingMonitor, setup_cuda, LearnableTemperature, ProjectionHead
)
from tqdm import tqdm


def main(cfg):
    # Set random seed for reproducibility
    seed = cfg.get("seed", 42)
    set_seed(seed)
    
    device = setup_cuda()
    save_dir = cfg.get("save_dir", "checkpoints/vision_tiny")
    os.makedirs(save_dir, exist_ok=True)
    train_manifest = cfg.get("train_manifest", "data/images/production_annotations.json")
    image_root = cfg.get("image_root", "data/images")
    
    # Initialize logger early
    logger = SimpleLogger("Vision")
    
    model_name = "vision"
    metadata = load_training_metadata(save_dir, model_name)
    
    # torch.compile() support (optional, PyTorch 2.0+)
    use_compile = cfg.get("use_compile", False)
    
    d_model = cfg.get("d_model", 768)  # ViT-Base: 768
    vit = ViTTiny(cfg.get("img_size", 224), cfg.get("patch", 16), d_model, cfg.get("n_layers", 12), cfg.get("n_heads", 12), cfg.get("d_ff", 3072), cfg.get("dropout", 0.1), compile_model=use_compile).to(device)
    
    # Use contrastive learning (CLIP-style) for proper vision-language alignment
    # Project image CLS token to embedding space for contrastive learning
    embed_dim = cfg.get("embed_dim", 512)  # CLIP standard: 512
    img_proj = ProjectionHead(d_model, d_model, embed_dim).to(device)
    
    # Configurable: Use Thinker model or simple tokenizer+embedding for text encoding
    use_thinker_for_text = cfg.get("use_thinker_for_text", True)
    thinker_ckpt_dir = cfg.get("thinker_ckpt", "checkpoints/thinker_tiny")
    thinker_cfg = cfg.get("thinker", {})
    ctx_len = cfg.get("ctx_len", 512)
    vocab_size = cfg.get("vocab_size", 32000)
    
    # Use learned attention pooling for best quality
    text_pooling = "attention"
    print(f"Text encoder pooling method: {text_pooling} (learned attention pooling)")
    
    think = None
    text_encoder = None
    
    if use_thinker_for_text:
        # Use Thinker model for text encoding (frozen) - better contextual embeddings
        print("Using Thinker model for text encoding (recommended)")
        thinker_d_model = cfg.get("thinker_d_model", 256)
        text_proj = ProjectionHead(thinker_d_model, thinker_d_model, embed_dim).to(device)
        
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
            compile_model=False,  # Don't compile Thinker for vision training
            use_spiking=thinker_cfg.get("use_spiking", False),
            use_ltc=thinker_cfg.get("use_ltc", False),
            window_size=thinker_cfg.get("window_size", 0)
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
        # Use TransformerTextEncoder (CLIP-style) for proper text encoding
        print(f"Using TransformerTextEncoder for text encoding")
        text_proj = ProjectionHead(d_model, d_model, embed_dim).to(device)
        # text_encoder will be created after tokenizer is loaded
    
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
    
    # Create text encoder if not using Thinker
    if not use_thinker_for_text:
        text_encoder = TransformerTextEncoder(
            vocab_size, 
            d_model = cfg.get("text_d_model", d_model),
            n_layers=cfg.get("text_n_layers", 6),
            n_heads=cfg.get("text_n_heads", 8),
            d_ff=cfg.get("text_d_ff", 2048),
            max_len=cfg.get("text_max_len", 77),  # CLIP standard
            dropout=cfg.get("dropout", 0.1)
        ).to(device)
        print(f"✓ Created TransformerTextEncoder with vocab_size={vocab_size}, d_model={d_model}, n_layers={cfg.get('text_n_layers', 6)}, max_len={cfg.get('text_max_len', 77)}")
    
    # Initialize projections with Xavier uniform (better than normal(0.01))
    for module in [img_proj, text_proj]:
        for m in module.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    print("✓ Initialized projection weights with Xavier uniform")
    
    # Contrastive loss (InfoNCE) with learnable temperature
    temperature = LearnableTemperature(init_value=cfg.get("temperature", 0.07)).to(device)
    
    # Optimizer: include text_encoder, temperature, and projection heads
    opt_params = list(vit.parameters()) + list(img_proj.parameters()) + list(text_proj.parameters()) + list(temperature.parameters())
    if text_encoder is not None:
        opt_params += list(text_encoder.parameters())
        print(f"✓ Optimizer includes text_encoder parameters")
    # CLIP optimizer settings
    opt = torch.optim.AdamW(opt_params, lr=cfg.get("lr", 5e-4), weight_decay=cfg.get("wd", 0.2), betas=(0.9, 0.98), fused=device=="cuda")
    
    # EMA for improved model quality (optional)
    use_ema = cfg.get("use_ema", False)
    ema_decay = cfg.get("ema_decay", 0.999)
    ema = None
    if use_ema:
        # EMA tracks all trainable models
        class EMAWrapper(nn.Module):
            def __init__(self):
                super().__init__()
                self.vit = vit
                self.img_proj = img_proj
                self.text_proj = text_proj
                self.temperature = temperature
                if text_encoder is not None:
                    self.text_encoder = text_encoder
        ema_model = EMAWrapper()
        ema = EMA(ema_model, decay=ema_decay, device=device)
        logger.info(f"✓ EMA enabled with decay={ema_decay}")
    
    
    # Learning rate scheduler with warmup (CLIP-style: longer warmup, cosine decay)
    warmup_steps = cfg.get("warmup_steps", 2000)
    max_steps = cfg.get("max_steps", 5000)
    scheduler = get_lr_scheduler(opt, warmup_steps, max_steps)
    
    # Training monitor (handles LR spikes, early stopping, etc.)
    monitor = TrainingMonitor(cfg)
    
    # Gradient clipping
    max_grad_norm = cfg.get("max_grad_norm", 1.0)
    
    # Gradient accumulation
    accumulation_steps = cfg.get("gradient_accumulation_steps", 1)
    
    # Validation loss threshold for reloading
    val_loss_threshold = cfg.get("val_loss_threshold", float('inf'))
    
    # Mixed precision training (AMP)
    use_amp = cfg.get("use_amp", True) and device == "cuda"
    scaler = GradScaler('cuda') if use_amp else None
    if use_amp:
        print("Mixed precision training (AMP) enabled")
    if accumulation_steps > 1:
        print(f"Gradient accumulation: {accumulation_steps} steps")
    
    def encode_caption(caption):
        """Encode caption using tokenizer and either Thinker model or TransformerTextEncoder"""
        # Handle two possible caption representations:
        # - A raw Python string (tokenize here)
        # - A 1D torch.Tensor or list of token ids (already tokenized by the dataset worker)
        if torch.is_tensor(caption):
            ids = caption.tolist()
        elif isinstance(caption, (list, tuple)):
            ids = list(caption)
        else:
            # Tokenize caption string
            ids = tok.encode(str(caption))

        # If dataset provided padded sequence (PAD=0), trim trailing PADs to restore original length
        while len(ids) > 1 and ids[-1] == 0:
            ids.pop()

        # Ensure BOS/CLS token at start. If already present (1), avoid duplicating.
        if len(ids) == 0 or ids[0] != 1:
            ids = [1] + ids

        # Truncate to context length
        ids = ids[:ctx_len]

        token_tensor = torch.tensor(ids, device=device, dtype=torch.long)

        if use_thinker_for_text:
            token_tensor = token_tensor.unsqueeze(0)  # (1, T)
            with torch.no_grad():
                text_emb = think(idx=token_tensor)  # (1, T, thinker_d_model)
            # Mean pooling across tokens
            return text_emb.squeeze(0).mean(dim=0)  # (thinker_d_model,)
        else:
            # Use TransformerTextEncoder (CLIP-style)
            with (torch.no_grad() if not text_encoder.training else torch.enable_grad()):
                text_emb = text_encoder(token_tensor, return_cls=True)  # (d_model,)
            return text_emb

    # Split dataset for validation
    val_split = cfg.get("val_split", 0.1)  # 10% for validation
    
    train_ds = ImgCapDataset(
        train_manifest,
        image_root,
        tok,
        ctx_len,
        img_size=cfg.get("img_size", 224),
        shuffle_buffer_size=cfg.get("shuffle_buffer_size", 10000),
        seed=seed,
        skip_samples=0,
        augment=cfg.get("use_augmentation", False)
    )
    train_ds._val_split = val_split
    train_ds._val_mode = False  # Training mode
    
    val_ds = ImgCapDataset(
        train_manifest,
        image_root,
        tok,
        ctx_len,
        img_size=cfg.get("img_size", 224),
        shuffle_buffer_size=cfg.get("shuffle_buffer_size", 100),  # Shuffle validation for different batches each time
        seed=seed,  # Same seed for consistent hash-based split
        skip_samples=0,
        augment=cfg.get("use_augmentation", False)
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
    train_dl = DataLoader(train_ds, batch_size=cfg.get("batch_size", 8), shuffle=False, num_workers=cfg.get("num_workers", 2), drop_last=cfg.get("drop_last", True), pin_memory=True)
    val_dl = DataLoader(val_ds, batch_size=cfg.get("batch_size", 8), shuffle=False, num_workers=cfg.get("num_workers", 2), drop_last=cfg.get("drop_last", True), pin_memory=True)
    
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
            "temperature": (temperature, temperature.load_state_dict),
            "text_encoder": (text_encoder, text_encoder.load_state_dict) if text_encoder is not None else None,
            "text_embed": (text_encoder, text_encoder.load_state_dict) if text_encoder is not None else None,  # Backward compatibility
            "optimizer": (opt, opt.load_state_dict),
            "scheduler": (scheduler, scheduler.load_state_dict),
            "scaler": (scaler, scaler.load_state_dict) if scaler is not None else None,
            "ema": (ema, ema.load_state_dict) if ema is not None else None,
            "monitor": (monitor, monitor.load_state_dict)
        }
    )
    
    # Track validation loss for reload logic
    last_checkpoint_val_loss = metadata.get("last_checkpoint_val_loss", None) if metadata else None
    most_recent_val_loss = last_checkpoint_val_loss
    consecutive_reloads = 0  # Track consecutive reloads due to validation loss spikes
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
    
    epoch = start_epoch
    while epoch < max_epochs:
        reload_needed = False
        # Recreate DataLoader for each epoch since IterableDatasets are exhausted after one iteration
        # skip_samples is automatically reset to 0 by the dataset after first iteration
        if epoch > start_epoch:
            train_dl = DataLoader(train_ds, batch_size=cfg.get("batch_size", 8), shuffle=False,
                                 num_workers=cfg.get("num_workers", 2),
                                 drop_last=cfg.get("drop_last", True), pin_memory=True)
        
        # Create progress bar with correct starting position when resuming mid-epoch
        remaining_epochs = max_epochs - epoch - 1
        pbar_desc = f"epoch{epoch}/{max_epochs-1} (remaining:{remaining_epochs}) step{step}"
        if epoch == start_epoch and start_batch_idx > 0:
            pbar = tqdm(train_dl, desc=pbar_desc, initial=start_batch_idx, total=steps_per_epoch)
        else:
            pbar = tqdm(train_dl, desc=pbar_desc, total=steps_per_epoch)
        
        # Start enumeration from the correct position when resuming mid-epoch
        enum_start = start_batch_idx if (epoch == start_epoch and start_batch_idx > 0) else 0
        batch_step = 0  # Count every batch processed, for accumulation and logging
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
            
            # Check for NaN inputs before forward pass
            if torch.isnan(img).any():
                logger.error(f"Step {step}: NaN in input images, skipping batch")
                continue
            
            # Encode images and captions
            if use_amp:
                with autocast(device_type='cuda'):
                    cls, _ = vit(img)  # (B,1,d)
                    img_emb = img_proj(cls.squeeze(1))  # (B, embed_dim)
                    img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)  # L2 normalize
                    
                    # Encode captions (batch if possible for speed)
                    if torch.is_tensor(cap) and cap.dim() == 2:
                        token_batch = cap.to(device)  # (B, T)
                        if use_thinker_for_text and think is not None:
                            with torch.no_grad():
                                # Get contextual hidden embeddings (B, T, thinker_d_model)
                                token_embs = think(idx=token_batch, return_embeddings=True)
                            text_embs = token_embs.mean(dim=1)  # (B, thinker_d_model)
                        else:
                            # TransformerTextEncoder supports batched input and returns pooled (B, d_model)
                            text_embs = text_encoder(token_batch, return_cls=True)
                    else:
                        # Fallback: handle strings or 1D tensors per-sample
                        text_embs = torch.stack([encode_caption(c) for c in cap]).to(device)  # (B, d_model)
                    text_emb = text_proj(text_embs)  # (B, embed_dim)
                    text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)  # L2 normalize
                    
                    # Contrastive loss (InfoNCE)
                    # Similarity matrix: (B, B)
                    temp = temperature()
                    logits_i2t = torch.matmul(img_emb, text_emb.t()) / temp  # Image-to-Text (B, B)
                    logits_t2i = torch.matmul(text_emb, img_emb.t()) / temp  # Text-to-Image (B, B)
                    labels = torch.arange(B, device=device)  # Positive pairs are on diagonal
                    loss_i2t = nn.CrossEntropyLoss(label_smoothing=cfg.get("label_smoothing", 0.0))(logits_i2t, labels)
                    loss_t2i = nn.CrossEntropyLoss(label_smoothing=cfg.get("label_smoothing", 0.0))(logits_t2i, labels)
                    loss = (loss_i2t + loss_t2i) / 2  # Symmetric loss
                    # Free intermediate tensors
                    del cls, img_emb, text_embs, text_emb, logits_i2t, logits_t2i
            else:
                cls, _ = vit(img)  # (B,1,d)
                img_emb = img_proj(cls.squeeze(1))  # (B, embed_dim)
                img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)  # L2 normalize

                # Encode captions (batch if possible for speed)
                if torch.is_tensor(cap) and cap.dim() == 2:
                    token_batch = cap.to(device)  # (B, T)
                    if use_thinker_for_text and think is not None:
                        with torch.no_grad():
                            token_embs = think(idx=token_batch, return_embeddings=True)
                        text_embs = token_embs.mean(dim=1)
                    else:
                        text_embs = text_encoder(token_batch, return_cls=True)
                else:
                    text_embs = torch.stack([encode_caption(c) for c in cap]).to(device)
                text_emb = text_proj(text_embs)
                text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)  # L2 normalize

                # Contrastive loss (InfoNCE)
                temp = temperature()
                logits_i2t = torch.matmul(img_emb, text_emb.t()) / temp  # Image-to-Text (B, B)
                logits_t2i = torch.matmul(text_emb, img_emb.t()) / temp  # Text-to-Image (B, B)
                labels = torch.arange(B, device=device)  # Positive pairs are on diagonal
                loss_i2t = nn.CrossEntropyLoss(label_smoothing=cfg.get("label_smoothing", 0.0))(logits_i2t, labels)
                loss_t2i = nn.CrossEntropyLoss(label_smoothing=cfg.get("label_smoothing", 0.0))(logits_t2i, labels)
                loss = (loss_i2t + loss_t2i) / 2  # Symmetric loss
                # Free intermediate tensors
                del cls, img_emb, text_embs, text_emb, logits_i2t, logits_t2i
            
            # Forward pass with gradient accumulation
            loss_scaled = loss / accumulation_steps  # Scale loss for accumulation

            if use_amp:
                scaler.scale(loss_scaled).backward()
            else:
                loss_scaled.backward()

            # Detach loss for logging (free computation graph)
            loss_val = loss.detach()
            del loss

            batch_step += 1  # Count every batch

            # Only step optimizer every N batches
            if batch_step % accumulation_steps == 0:
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
                    opt.zero_grad(set_to_none=True)
                    if use_amp:
                        scaler.update()
                    continue
                
                # Gradient clipping (already unscaled if using AMP)
                try:
                    grad_norm_vit = clip_gradients(vit, max_grad_norm)
                    grad_norm_img_proj = clip_gradients(img_proj, max_grad_norm)
                    grad_norm_text_proj = clip_gradients(text_proj, max_grad_norm)
                    grad_norm_temp = clip_gradients(temperature, max_grad_norm)

                    all_grad_norms = [grad_norm_vit, grad_norm_img_proj, grad_norm_text_proj, grad_norm_temp]
                    if text_encoder is not None:
                        grad_norm_text_enc = clip_gradients(text_encoder, max_grad_norm)
                        all_grad_norms.append(grad_norm_text_enc)
                    max_grad = max(all_grad_norms)
                    explosion_threshold = max(100.0, max_grad_norm * 10)
                    if max_grad > explosion_threshold:
                        logger.error(f"Step {step}: Gradient explosion detected (max: {max_grad:.2f}). Skipping batch.")
                        opt.zero_grad(set_to_none=True)
                        if use_amp:
                            scaler.update()
                        continue
                except RuntimeError as e:
                    logger.error(f"Step {step}: {e}")
                    opt.zero_grad(set_to_none=True)
                    if use_amp:
                        scaler.update()
                    continue
                
                # Optimizer step (gradients already clipped)
                if use_amp:
                    scaler.step(opt)
                    scaler.update()
                else:
                    opt.step()
                scheduler.step()
                
                # Update EMA after optimizer step
                if ema is not None:
                    ema.update()
                
                # Update training monitor (countdown if active)
                monitor.step(opt, logger)
                
                opt.zero_grad(set_to_none=True)  # Clear gradients after stepping
                step += 1  # This is the "effective" step for logging

            # Use batch_step for all frequency checks
            if batch_step % print_freq == 0:
                current_lr = scheduler.get_last_lr()[0]
                unscaled_loss = loss_val * accumulation_steps
                logger.train_step(step, float(unscaled_loss), current_lr, epoch)

            # Periodic checkpointing
            if batch_step % checkpoint_freq == 0 and batch_step > 0:
                # Save model weights only (overwrite existing file)
                model_path = os.path.join(save_dir, f"{model_name}.pt")
                checkpoint_data = {
                    "vit": vit.state_dict(),
                    "img_proj": img_proj.state_dict(),
                    "text_proj": text_proj.state_dict(),
                    "optimizer": opt.state_dict(),
                    "scheduler": scheduler.state_dict(),
                }
                if text_encoder is not None:
                    checkpoint_data["text_encoder"] = text_encoder.state_dict()
                if scaler is not None:
                    checkpoint_data["scaler"] = scaler.state_dict()
                if ema is not None:
                    checkpoint_data["ema"] = ema.state_dict()
                checkpoint_data["monitor"] = monitor.get_state_dict()
                torch.save(checkpoint_data, model_path)
                
                # Save training metadata
                training_metadata = {
                    "step": step,
                    "epoch": epoch,
                    "last_checkpoint_val_loss": most_recent_val_loss if most_recent_val_loss is not None else last_checkpoint_val_loss,
                }
                save_training_metadata(save_dir, model_name, training_metadata)
                logger.checkpoint(step, model_path)
                
                # Update last_checkpoint_val_loss
                if most_recent_val_loss is not None:
                    last_checkpoint_val_loss = most_recent_val_loss
            
            # Validation
            if step % val_freq == 0 and step > 0:
                with ValidationSkipSamplesContext(train_ds):
                    # Apply EMA weights for validation if enabled
                    if ema is not None:
                        ema.apply_shadow()
                    
                    vit.eval()
                    img_proj.eval()
                    text_proj.eval()
                    # Thinker is already in eval mode (frozen)
                    val_loss_sum = 0.0
                    val_count = 0
                    val_batches = cfg.get("val_batches", 100)  # None = full validation
                    with torch.no_grad():
                        for val_img, val_cap in val_dl:
                            val_img = val_img.to(device)
                            val_B = val_img.shape[0]
                            if use_amp:
                                with autocast(device_type='cuda'):
                                    val_cls, _ = vit(val_img)
                                    val_img_emb = img_proj(val_cls.squeeze(1))
                                    val_img_emb = val_img_emb / val_img_emb.norm(dim=-1, keepdim=True)
                                    
                                    # Batch-process validation captions when possible
                                    if torch.is_tensor(val_cap) and val_cap.dim() == 2:
                                        val_token_batch = val_cap.to(device)
                                        if use_thinker_for_text and think is not None:
                                            val_token_embs = think(idx=val_token_batch, return_embeddings=True)
                                            val_text_embs = val_token_embs.mean(dim=1)
                                        else:
                                            val_text_embs = text_encoder(val_token_batch, return_cls=True)
                                    else:
                                        val_text_embs = torch.stack([encode_caption(c) for c in val_cap]).to(device)
                                    val_text_emb = text_proj(val_text_embs)
                                    val_text_emb = val_text_emb / val_text_emb.norm(dim=-1, keepdim=True)
                                    
                                    val_logits = torch.matmul(val_img_emb, val_text_emb.t()) / temperature()
                                    val_labels = torch.arange(val_B, device=device)
                                    val_loss = nn.CrossEntropyLoss(label_smoothing=cfg.get("label_smoothing", 0.0))(val_logits, val_labels)
                            else:
                                val_cls, _ = vit(val_img)
                                val_img_emb = img_proj(val_cls.squeeze(1))
                                val_img_emb = val_img_emb / val_img_emb.norm(dim=-1, keepdim=True)

                                val_text_embs = torch.stack([encode_caption(c) for c in val_cap]).to(device)
                                val_text_emb = text_proj(val_text_embs)
                                val_text_emb = val_text_emb / val_text_emb.norm(dim=-1, keepdim=True)

                                val_logits = torch.matmul(val_img_emb, val_text_emb.t()) / temperature()
                                val_labels = torch.arange(val_B, device=device)
                                val_loss = nn.CrossEntropyLoss(label_smoothing=cfg.get("label_smoothing", 0.0))(val_logits, val_labels)
                            
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
                            
                            if val_batches is not None and val_count >= val_batches:
                                break
                    
                    avg_val_loss = val_loss_sum / max(val_count, 1)
                    logger.val_step(step, avg_val_loss, epoch)
                    
                    # Restore original weights after validation
                    if ema is not None:
                        ema.restore()
                    
                    # Check for LR spike trigger / early stopping
                    monitor.on_val_end(avg_val_loss, opt, {"vit": vit, "img_proj": img_proj, "text_proj": text_proj}, logger)
                    if monitor.should_stop:
                        break
                    
                    # Check for loss spike
                    if last_checkpoint_val_loss is not None and val_loss_threshold < float('inf'):
                        if avg_val_loss > last_checkpoint_val_loss + val_loss_threshold:
                             logger.warning(f"Validation loss spiked! {avg_val_loss:.4f} > {last_checkpoint_val_loss:.4f} + {val_loss_threshold}. Reloading from last checkpoint...")
                             consecutive_reloads += 1
                             if consecutive_reloads >= 2:
                                 logger.error(f"Training stopped: Validation loss spiked {consecutive_reloads} times consecutively.")
                                 logger.error("This indicates the model is not learning effectively. Consider:")
                                 logger.error("  - Reducing learning rate")
                                 logger.error("  - Adjusting val_loss_threshold")
                                 logger.error("  - Checking data quality")
                                 logger.training_end(step)
                                 return
                             reload_needed = True
                             break
                        else:
                            consecutive_reloads = 0  # Reset counter on successful validation
                    most_recent_val_loss = avg_val_loss
                    
                    vit.train()
                    img_proj.train()
                    text_proj.train()
                    # Thinker remains in eval mode (frozen)
            
            if reload_needed:
                break
            
            if step >= cfg["max_steps"]:
                final_path = os.path.join(save_dir, f"{model_name}.pt")
                checkpoint_data = {
                    "vit": vit.state_dict(),
                    "img_proj": img_proj.state_dict(),
                    "text_proj": text_proj.state_dict(),
                    "optimizer": opt.state_dict(),
                    "scheduler": scheduler.state_dict(),
                }
                if text_encoder is not None:
                    checkpoint_data["text_encoder"] = text_encoder.state_dict()
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
        
        if reload_needed:
            # Reload from last checkpoint
            step, metadata = load_checkpoint(
                save_dir, 
                model_name, 
                device, 
                logger,
                state_dict_loaders={
                    "vit": (vit, vit.load_state_dict),
                    "img_proj": (img_proj, img_proj.load_state_dict),
                    "text_proj": (text_proj, text_proj.load_state_dict),
                    "text_encoder": (text_encoder, text_encoder.load_state_dict) if text_encoder is not None else None,
                    "text_embed": (text_encoder, text_encoder.load_state_dict) if text_encoder is not None else None,  # Backward compatibility
                    "optimizer": (opt, opt.load_state_dict),
                    "scheduler": (scheduler, scheduler.load_state_dict),
                    "scaler": (scaler, scaler.load_state_dict) if scaler is not None else None
                }
            )
            last_checkpoint_val_loss = metadata.get("last_checkpoint_val_loss", None) if metadata else None
            most_recent_val_loss = last_checkpoint_val_loss
            
            # Recalculate positions
            start_epoch, start_batch_idx = calculate_resume_position(step, steps_per_epoch)
            epoch = start_epoch
            
            # Reset dataloader
            train_dl = setup_resume_data_loading(
                train_ds, step, batch_size, logger,
                train_dl_kwargs={
                    "num_workers": cfg.get("num_workers", 2),
                    "drop_last": cfg.get("drop_last", True)
                }
            )
            continue
        
        # Final validation at end of epoch
        with ValidationSkipSamplesContext(train_ds):
            vit.eval()
            img_proj.eval()
            text_proj.eval()
            # Thinker is already in eval mode (frozen)
            val_loss_sum = 0.0
            val_count = 0
            val_batches = cfg.get("val_batches_epoch_end", None)  # None = full validation at epoch end
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
                            
                            val_logits = torch.matmul(val_img_emb, val_text_emb.t()) / temperature()
                            val_labels = torch.arange(val_B, device=device)
                            val_loss = nn.CrossEntropyLoss(label_smoothing=cfg.get("label_smoothing", 0.0))(val_logits, val_labels)
                    else:
                        val_cls, _ = vit(val_img)
                        val_img_emb = img_proj(val_cls.squeeze(1))
                        val_img_emb = val_img_emb / val_img_emb.norm(dim=-1, keepdim=True)

                        # Batch-process validation captions when possible
                        if torch.is_tensor(val_cap) and val_cap.dim() == 2:
                            val_token_batch = val_cap.to(device)
                            if use_thinker_for_text and think is not None:
                                val_token_embs = think(idx=val_token_batch, return_embeddings=True)
                                val_text_embs = val_token_embs.mean(dim=1)
                            else:
                                val_text_embs = text_encoder(val_token_batch, return_cls=True)
                        else:
                            val_text_embs = torch.stack([encode_caption(c) for c in val_cap]).to(device)
                        val_text_emb = text_proj(val_text_embs)
                        val_text_emb = val_text_emb / val_text_emb.norm(dim=-1, keepdim=True)

                        val_logits = torch.matmul(val_img_emb, val_text_emb.t()) / temperature()
                        val_labels = torch.arange(val_B, device=device)
                        val_loss = nn.CrossEntropyLoss(label_smoothing=cfg.get("label_smoothing", 0.0))(val_logits, val_labels)
                    
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
                    
                    if val_batches is not None and val_count >= val_batches:
                        break
            
            avg_val_loss = val_loss_sum / max(val_count, 1)
            logger.epoch_end(epoch, train_loss=None, val_loss=avg_val_loss)
            
            # Check for loss spike
            if last_checkpoint_val_loss is not None and val_loss_threshold < float('inf'):
                if avg_val_loss > last_checkpoint_val_loss + val_loss_threshold:
                     logger.warning(f"Validation loss spiked! {avg_val_loss:.4f} > {last_checkpoint_val_loss:.4f} + {val_loss_threshold}. Reloading from last checkpoint...")
                     reload_needed = True
            most_recent_val_loss = avg_val_loss
            
            vit.train()
            img_proj.train()
            text_proj.train()
            # Thinker remains in eval mode (frozen)
        
        if reload_needed:
            # Reload from last checkpoint
            step, metadata = load_checkpoint(
                save_dir, 
                model_name, 
                device, 
                logger,
                state_dict_loaders={
                    "vit": (vit, vit.load_state_dict),
                    "img_proj": (img_proj, img_proj.load_state_dict),
                    "text_proj": (text_proj, text_proj.load_state_dict),
                    "text_encoder": (text_encoder, text_encoder.load_state_dict) if text_encoder is not None else None,
                    "text_embed": (text_encoder, text_encoder.load_state_dict) if text_encoder is not None else None,  # Backward compatibility
                    "optimizer": (opt, opt.load_state_dict),
                    "scheduler": (scheduler, scheduler.load_state_dict),
                    "scaler": (scaler, scaler.load_state_dict) if scaler is not None else None
                }
            )
            last_checkpoint_val_loss = metadata.get("last_checkpoint_val_loss", None) if metadata else None
            most_recent_val_loss = last_checkpoint_val_loss
            
            # Recalculate positions
            start_epoch, start_batch_idx = calculate_resume_position(step, steps_per_epoch)
            epoch = start_epoch
            
            # Reset dataloader
            train_dl = setup_resume_data_loading(
                train_ds, step, batch_size, logger,
                train_dl_kwargs={
                    "num_workers": cfg.get("num_workers", 2),
                    "drop_last": cfg.get("drop_last", True)
                }
            )
            continue
        
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
        if text_encoder is not None:
            checkpoint_data["text_encoder"] = text_encoder.state_dict()
        if scaler is not None:
            checkpoint_data["scaler"] = scaler.state_dict()
        torch.save(checkpoint_data, final_path)
        
        # Save training metadata
        training_metadata = {
            "step": step,
            "epoch": epoch,
            "last_checkpoint_val_loss": most_recent_val_loss if most_recent_val_loss is not None else last_checkpoint_val_loss,
        }
        save_training_metadata(save_dir, model_name, training_metadata)
        logger.info(f"Model saved to {cfg['save_dir']} at end of epoch {epoch}, step {step}")
        
        # Update last_checkpoint_val_loss
        if most_recent_val_loss is not None:
            last_checkpoint_val_loss = most_recent_val_loss
        
        # Check if we've reached max_steps after epoch completion
        if step >= cfg["max_steps"]:
            logger.info(f"Reached max_steps={cfg['max_steps']}. Training complete.")
            logger.training_end(step)
            return
        
        # Continue to next epoch
        start_batch_idx = 0  # Reset batch index for new epoch
        epoch += 1

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--config", required=True)
    cfg = json.load(open(ap.parse_args().config))
    main(cfg)
