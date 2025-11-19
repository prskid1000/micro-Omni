
import argparse, json, os, torch
from torch import nn
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import torchaudio
from tqdm import tqdm

from omni.thinker import ThinkerLM
from omni.audio_encoder import AudioEncoderTiny
from omni.vision_encoder import ViTTiny
from omni.training_utils import (
    set_seed, get_lr_scheduler, clip_gradients, SimpleLogger, validate_loss, 
    check_gradient_explosion, reload_from_last_checkpoint, cleanup_old_checkpoints, MixDataset,
    load_checkpoint, setup_resume_data_loading, calculate_resume_position,
    ValidationSkipSamplesContext
)

def mix_collate_fn(batch):
    """Custom collate function that handles missing keys"""
    result = {}
    # Get all possible keys from the batch
    all_keys = set()
    for item in batch:
        all_keys.update(item.keys())
    
    # For each key, collect values (use None for missing)
    for key in all_keys:
        values = []
        for item in batch:
            if key in item:
                values.append(item[key])
            else:
                values.append(None)
        result[key] = values if None not in values else values
    return result

def main(cfg):
    # Set random seed for reproducibility
    seed = cfg.get("seed", 42)
    set_seed(seed)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(cfg["save_dir"], exist_ok=True)
    ds = MixDataset(cfg["sft_mix"]["text_path"], cfg["sft_mix"]["image_manifest"], cfg["sft_mix"]["image_root"], cfg["sft_mix"]["asr_csv"], cfg["ctx_len"])
    print(f"Dataset size: {len(ds)}")
    dl = DataLoader(ds, batch_size=cfg.get("batch_size", 2), shuffle=True, num_workers=cfg.get("num_workers", 2), drop_last=cfg.get("drop_last", True), collate_fn=mix_collate_fn)
    print(f"DataLoader created, starting training...")

    # load components
    # torch.compile() support (optional, PyTorch 2.0+)
    use_compile = cfg.get("use_compile", False)
    
    thinker_cfg = cfg.get("thinker", {})
    think = ThinkerLM(
        thinker_cfg.get("vocab_size", 5000),
        thinker_cfg.get("n_layers", 4),
        thinker_cfg.get("d_model", 256),
        thinker_cfg.get("n_heads", 4),
        thinker_cfg.get("d_ff", 1024),
        thinker_cfg.get("dropout", 0.1),
        thinker_cfg.get("rope_theta", 10000),
        cfg["ctx_len"],
        use_gqa=thinker_cfg.get("use_gqa", False),
        use_swiglu=thinker_cfg.get("use_swiglu", True),
        use_moe=thinker_cfg.get("use_moe", False),
        num_experts=thinker_cfg.get("num_experts", 8),
        num_experts_per_tok=thinker_cfg.get("num_experts_per_tok", 2),
        compile_model=use_compile
    ).to(device)
    if os.path.exists(os.path.join(cfg["thinker_ckpt"], "thinker.pt")):
        think.load_state_dict(torch.load(os.path.join(cfg["thinker_ckpt"], "thinker.pt"), map_location=device))

    # Load audio encoder config from checkpoint or use defaults
    audio_cfg_path = "configs/audio_enc_tiny.json"
    if os.path.exists(audio_cfg_path):
        audio_cfg = json.load(open(audio_cfg_path))
        downsample_factor = audio_cfg.get("downsample_time", 8)
        aud = AudioEncoderTiny(
            d=audio_cfg.get("d_model", 192),
            heads=audio_cfg.get("n_heads", 3),
            ff=audio_cfg.get("d_ff", 768),
            layers=audio_cfg.get("n_layers", 4),
            dropout=audio_cfg.get("dropout", 0.1),
            downsample_factor=downsample_factor
        ).to(device)
    else:
        aud = AudioEncoderTiny().to(device)
    if os.path.exists(os.path.join(cfg["audio_ckpt"], "audio_enc.pt")):
        aud.load_state_dict(torch.load(os.path.join(cfg["audio_ckpt"], "audio_enc.pt"), map_location=device)["enc"])

    # Load vision encoder config from checkpoint or use defaults
    vision_cfg_path = "configs/vision_tiny.json"
    if os.path.exists(vision_cfg_path):
        vision_cfg = json.load(open(vision_cfg_path))
        vis = ViTTiny(
            img_size=vision_cfg.get("img_size", 224),
            patch=vision_cfg.get("patch", 16),
            d=vision_cfg.get("d_model", 128),
            layers=vision_cfg.get("n_layers", 4),
            heads=vision_cfg.get("n_heads", 2),
            ff=vision_cfg.get("d_ff", 512),
            dropout=vision_cfg.get("dropout", 0.1)
        ).to(device)
    else:
        vis = ViTTiny().to(device)
    if os.path.exists(os.path.join(cfg["vision_ckpt"], "vision.pt")):
        vis.load_state_dict(torch.load(os.path.join(cfg["vision_ckpt"], "vision.pt"), map_location=device)["vit"])

    # simple projectors - use actual model dimensions
    audio_dim = audio_cfg.get("d_model", 192) if os.path.exists(audio_cfg_path) else 384
    vision_dim = vision_cfg.get("d_model", 128) if os.path.exists(vision_cfg_path) else 192
    thinker_d_model = thinker_cfg.get("d_model", 256)
    proj_a = torch.nn.Linear(audio_dim, thinker_d_model).to(device)
    proj_v = torch.nn.Linear(vision_dim, thinker_d_model).to(device)
    opt = torch.optim.AdamW(list(think.parameters())+list(proj_a.parameters())+list(proj_v.parameters()), lr=cfg["lr"], weight_decay=cfg["wd"])
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=0)
    
    # Learning rate scheduler with warmup
    warmup_steps = cfg.get("warmup_steps", 200)
    max_steps = cfg.get("max_steps", 2000)
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

    mel_spec = torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_fft=1024, hop_length=160, win_length=400, n_mels=128).to(device)
    tok_model = os.path.join(cfg["thinker_ckpt"], "tokenizer.model")
    from omni.tokenizer import BPETokenizer
    tok = BPETokenizer(tok_model)

    def pack_text(prompt, answer, ctx):
        ids = [1] + tok.encode(prompt + " " + answer)
        ids = ids[:ctx]
        x = torch.tensor(ids + [0]*(ctx-len(ids)), dtype=torch.long)
        y = x.clone(); y[:-1]=x[1:]; y[-1]=0
        return x,y

    max_epochs = cfg.get("max_epochs", 9999)
    print_freq = cfg.get("print_freq", 20)
    prompt = cfg.get("prompt", "You are an omni assistant.")
    
    # Split dataset for validation
    val_split = cfg.get("val_split", 0.1)  # 10% for validation
    total_size = len(ds)
    val_size = int(total_size * val_split)
    train_size = total_size - val_size
    train_ds, val_ds = torch.utils.data.random_split(ds, [train_size, val_size], generator=torch.Generator().manual_seed(seed))
    train_dl = DataLoader(train_ds, batch_size=cfg.get("batch_size", 2), shuffle=True, num_workers=cfg.get("num_workers", 2), drop_last=cfg.get("drop_last", True), collate_fn=mix_collate_fn)
    val_dl = DataLoader(val_ds, batch_size=cfg.get("batch_size", 2), shuffle=False, num_workers=cfg.get("num_workers", 2), drop_last=False, collate_fn=mix_collate_fn)
    
    print(f"Starting training: max_epochs={max_epochs}, max_steps={cfg['max_steps']}, batch_size={cfg.get('batch_size', 2)}")
    print(f"Train samples: {train_size}, Val samples: {val_size}")
    
    def process_batch(data, is_training=True, use_amp_flag=False):
        """Process a batch of multimodal data efficiently"""
        B = len(data["text"])
        
        # Batch process images
        img_embeddings = []
        img_indices = []
        if "image" in data and isinstance(data["image"], list):
            valid_images = []
            valid_indices = []
            for b in range(B):
                if data["image"][b] is not None and os.path.exists(data["image"][b]):
                    valid_images.append(data["image"][b])
                    valid_indices.append(b)
            
            if valid_images:
                # Load and process all images at once
                img_tensors = []
                for img_path in valid_images:
                    img = Image.open(img_path).convert("RGB")
                    img = transforms.Resize((224,224))(img)
                    img = transforms.ToTensor()(img)
                    img_tensors.append(img)
                
                if img_tensors:
                    img_batch = torch.stack(img_tensors).to(device)  # (N, 3, 224, 224)
                    with torch.set_grad_enabled(is_training):
                        if use_amp_flag and is_training:
                            with autocast(device_type='cuda'):
                                cls_batch, _ = vis(img_batch)  # (N, 1, d_vision)
                                img_emb_batch = proj_v(cls_batch)  # (N, 1, thinker_d_model)
                        else:
                            cls_batch, _ = vis(img_batch)  # (N, 1, d_vision)
                            img_emb_batch = proj_v(cls_batch)  # (N, 1, thinker_d_model)
                    
                    # Store embeddings for valid indices
                    for idx, emb in zip(valid_indices, img_emb_batch):
                        # emb is (1, thinker_d_model), need to add batch dimension: (1, 1, thinker_d_model)
                        img_embeddings.append((idx, emb.unsqueeze(0)))
        
        # Batch process audio
        audio_embeddings = []
        audio_indices = []
        if "audio" in data and isinstance(data["audio"], list):
            valid_audios = []
            valid_indices = []
            for b in range(B):
                if data["audio"][b] is not None and os.path.exists(data["audio"][b]):
                    valid_audios.append(data["audio"][b])
                    valid_indices.append(b)
            
            if valid_audios:
                # Process audio files
                mel_list = []
                for audio_path in valid_audios:
                    wav, _ = torchaudio.load(audio_path)
                    wav = wav.to(device)
                    mel = mel_spec(wav)[0].T.unsqueeze(0)  # (1, T, 128)
                    mel_list.append(mel)
                
                if mel_list:
                    # Pad mels to same length for batching
                    max_mel_len = max(m.shape[1] for m in mel_list)
                    mel_batch = []
                    for m in mel_list:
                        pad_len = max_mel_len - m.shape[1]
                        if pad_len > 0:
                            m = torch.cat([m, torch.zeros(1, pad_len, m.shape[2], device=device)], dim=1)
                        mel_batch.append(m.squeeze(0))
                    mel_batch = torch.stack(mel_batch)  # (N, T, 128)
                    
                    with torch.set_grad_enabled(is_training):
                        if use_amp_flag and is_training:
                            with autocast(device_type='cuda'):
                                audio_emb_batch = aud(mel_batch)  # (N, T', d_audio)
                                audio_emb_batch = proj_a(audio_emb_batch)  # (N, T', thinker_d_model)
                        else:
                            audio_emb_batch = aud(mel_batch)  # (N, T', d_audio)
                            audio_emb_batch = proj_a(audio_emb_batch)  # (N, T', thinker_d_model)
                    
                    # Limit audio length and store
                    max_audio_tokens = cfg["ctx_len"] // 4
                    for idx, emb in zip(valid_indices, audio_emb_batch):
                        emb_trimmed = emb[:max_audio_tokens, :].unsqueeze(0)  # (1, T_trimmed, thinker_d_model)
                        audio_embeddings.append((idx, emb_trimmed))
        
        # Process text for all samples
        text_embeddings = []
        for b in range(B):
            ans = data["text"][b]
            x_ids, y_ids = pack_text(prompt, ans, cfg["ctx_len"])
            x_ids, y_ids = x_ids.to(device), y_ids.to(device)
            text_emb = think.tok_emb(x_ids.unsqueeze(0))  # (1, T_text, thinker_d_model)
            text_embeddings.append((b, text_emb, y_ids))
        
        # Combine embeddings for each sample
        batch_embeddings = []
        batch_targets = []
        for b in range(B):
            multimodal_emb_list = []
            
            # Add image if present
            for idx, emb in img_embeddings:
                if idx == b:
                    multimodal_emb_list.append(emb)
            
            # Add audio if present
            for idx, emb in audio_embeddings:
                if idx == b:
                    multimodal_emb_list.append(emb)
            
            # Get text embedding
            text_emb = None
            y_ids = None
            for idx, emb, y in text_embeddings:
                if idx == b:
                    text_emb = emb  # (1, T_text, d_thinker)
                    y_ids = y  # (T_text,)
                    break
            
            # Calculate remaining context for text
            multimodal_len = sum(emb.shape[1] for emb in multimodal_emb_list)
            max_text_len = cfg["ctx_len"] - multimodal_len - 1
            if max_text_len < 1:
                max_text_len = 1
            
            text_emb = text_emb[:, :max_text_len, :]  # (1, T_text, d_thinker)
            y_ids = y_ids[:max_text_len]  # (T_text,)
            
            # Combine all embeddings
            if multimodal_emb_list:
                # All embeddings should be (1, T, d_thinker)
                combined_emb = torch.cat(multimodal_emb_list + [text_emb], dim=1)  # (1, T_total, thinker_d_model)
                multimodal_padding = torch.zeros(multimodal_len, dtype=y_ids.dtype, device=device)
                y_ids = torch.cat([multimodal_padding, y_ids], dim=0)
            else:
                combined_emb = text_emb  # (1, T_text, d_thinker)
            
            batch_embeddings.append(combined_emb)
            batch_targets.append(y_ids)
        
        # Pad sequences to same length for batching
        max_len = max(emb.shape[1] for emb in batch_embeddings)
        padded_embeddings = []
        padded_targets = []
        attention_masks = []
        
        for emb, y in zip(batch_embeddings, batch_targets):
            seq_len = emb.shape[1]
            pad_len = max_len - seq_len
            
            # Pad embeddings
            if pad_len > 0:
                pad_emb = torch.zeros(1, pad_len, emb.shape[2], device=device)
                emb = torch.cat([emb, pad_emb], dim=1)
            
            # Pad targets
            if pad_len > 0:
                pad_target = torch.zeros(pad_len, dtype=y.dtype, device=device)
                y = torch.cat([y, pad_target], dim=0)
            
            # Create attention mask (1 for real tokens, 0 for padding)
            mask = torch.ones(seq_len, dtype=torch.float, device=device)
            if pad_len > 0:
                mask = torch.cat([mask, torch.zeros(pad_len, device=device)], dim=0)
            
            padded_embeddings.append(emb)
            padded_targets.append(y)
            attention_masks.append(mask)
        
        # Stack into batch
        batch_emb = torch.cat(padded_embeddings, dim=0)  # (B, T_max, thinker_d_model)
        batch_targets = torch.stack(padded_targets)  # (B, T_max)
        batch_mask = torch.stack(attention_masks)  # (B, T_max)
        
        return batch_emb, batch_targets, batch_mask
    
    # Initialize logger
    logger = SimpleLogger("OmniSFT")
    
    val_freq = cfg.get("val_freq", 100)  # Validate every N steps
    checkpoint_freq = cfg.get("checkpoint_freq", 500)  # Save checkpoint every N steps
    
    step = 0  # Global step counter (not per-epoch)
    
    # Resume from checkpoint if available
    step, resume_from = load_checkpoint(
        cfg["save_dir"], 
        "omni_step_", 
        device, 
        logger,
        state_dict_loaders={
            "thinker": (think, think.load_state_dict),
            "proj_a": (proj_a, proj_a.load_state_dict),
            "proj_v": (proj_v, proj_v.load_state_dict),
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
    batch_size = cfg.get("batch_size", 2)
    new_train_dl = setup_resume_data_loading(
        train_ds, step, batch_size, logger,
        train_dl_kwargs={
            "shuffle": True,
            "num_workers": cfg.get("num_workers", 2),
            "drop_last": cfg.get("drop_last", True),
            "collate_fn": mix_collate_fn
        }
    )
    if new_train_dl is not None:
        train_dl = new_train_dl
    
    logger.training_start(cfg["max_steps"], train_size, val_size)
    
    # Calculate steps per epoch and determine starting epoch/position
    # For IterableDataset, we can't use len() directly, so calculate from dataset size
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
        logger.epoch_start(epoch)
        think.train()
        proj_a.train()
        proj_v.train()
        
        # Create progress bar with correct starting position when resuming mid-epoch
        if epoch == start_epoch and start_batch_idx > 0:
            pbar = tqdm(train_dl, desc=f"epoch{epoch} step{step}", initial=start_batch_idx, total=steps_per_epoch)
        else:
            pbar = tqdm(train_dl, desc=f"epoch{epoch} step{step}", total=steps_per_epoch)
        
        # Start enumeration from the correct position when resuming mid-epoch
        enum_start = start_batch_idx if (epoch == start_epoch and start_batch_idx > 0) else 0
        for batch_idx, data in enumerate(pbar, start=enum_start):
            # Skip batches if resuming mid-epoch
            # batch_idx already represents the position in the epoch when enum_start > 0
            if epoch == start_epoch and initial_step > 0:
                current_batch_step = epoch * steps_per_epoch + batch_idx
                if current_batch_step < initial_step:
                    continue
            
            # Update progress bar description
            pbar.set_description(f"epoch{epoch} step{step} batch{batch_idx}")
            batch_emb, batch_targets, batch_mask = process_batch(data, is_training=True, use_amp_flag=use_amp)
            
            # Mark step begin for CUDAGraphs optimization
            if device == "cuda":
                torch.compiler.cudagraph_mark_step_begin()
            
            # Forward pass with mixed precision
            try:
                if use_amp:
                    with autocast():
                        logits = think(embeddings=batch_emb)  # (B, T, vocab)
                        # Calculate loss (mask out padding)
                        loss = loss_fn(logits.view(-1, logits.size(-1)), batch_targets.view(-1))
                        # Free logits after loss computation
                        del logits
                else:
                    logits = think(embeddings=batch_emb)  # (B, T, vocab)
                    # Calculate loss (mask out padding)
                    loss = loss_fn(logits.view(-1, logits.size(-1)), batch_targets.view(-1))
                    # Free logits after loss computation
                    del logits
            except RuntimeError as e:
                error_msg = str(e)
                if "NaN detected in attention probabilities after softmax" in error_msg or "Numerical instability" in error_msg:
                    logger.error(f"Step {step}: {e}")
                    logger.error("Reloading from last checkpoint...")
                    # Reload from last checkpoint
                    reloaded_step = reload_from_last_checkpoint(
                        cfg["save_dir"], "omni_step_", device, logger, think, opt, scheduler, scaler
                    )
                    if reloaded_step > 0:
                        step = reloaded_step
                        # Also reload proj_a and proj_v from checkpoint
                        checkpoint_files = [f for f in os.listdir(cfg["save_dir"]) if f.startswith("omni_step_") and f.endswith(".pt")]
                        if checkpoint_files:
                            step_numbers = []
                            for f in checkpoint_files:
                                try:
                                    step_num = int(f.replace("omni_step_", "").replace(".pt", ""))
                                    step_numbers.append((step_num, f))
                                except:
                                    continue
                            if step_numbers:
                                step_numbers.sort(key=lambda x: x[0], reverse=True)
                                last_checkpoint = os.path.join(cfg["save_dir"], step_numbers[0][1])
                                checkpoint = torch.load(last_checkpoint, map_location=device)
                                if isinstance(checkpoint, dict):
                                    if "proj_a" in checkpoint:
                                        proj_a.load_state_dict(checkpoint["proj_a"])
                                    if "proj_v" in checkpoint:
                                        proj_v.load_state_dict(checkpoint["proj_v"])
                        # Recalculate start_epoch, start_batch_idx and initial_step for resuming
                        start_epoch = step // steps_per_epoch
                        start_batch_idx = step % steps_per_epoch
                        initial_step = step
                        logger.info(f"Resuming from step {step} (epoch {start_epoch}, batch {start_batch_idx}/{steps_per_epoch})")
                    opt.zero_grad()
                    # Don't call scaler.update() here - no backward pass occurred, so no inf checks were recorded
                    continue
                else:
                    # Re-raise if it's a different error
                    raise
            
            # Scale loss for gradient accumulation
            loss_scaled = loss / accumulation_steps
            
            # Backward pass with gradient scaling
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
                    grad_norm_think, is_exploded_think = check_gradient_explosion(think, max_grad_norm=100.0, raise_on_error=False)
                    grad_norm_proj_a, is_exploded_proj_a = check_gradient_explosion(proj_a, max_grad_norm=100.0, raise_on_error=False)
                    grad_norm_proj_v, is_exploded_proj_v = check_gradient_explosion(proj_v, max_grad_norm=100.0, raise_on_error=False)
                    if is_exploded_think or is_exploded_proj_a or is_exploded_proj_v:
                        logger.error(f"Step {step}: Gradient explosion detected (think={grad_norm_think:.2f}, proj_a={grad_norm_proj_a:.2f}, proj_v={grad_norm_proj_v:.2f}). Skipping this batch.")
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
                    clip_gradients(think, max_grad_norm)
                    clip_gradients(proj_a, max_grad_norm)
                    clip_gradients(proj_v, max_grad_norm)
                    scaler.step(opt)
                    scaler.update()
                else:
                    clip_gradients(think, max_grad_norm)
                    clip_gradients(proj_a, max_grad_norm)
                    clip_gradients(proj_v, max_grad_norm)
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
            
            step += 1  # Increment global step counter
            
            if step % print_freq == 0:
                current_lr = scheduler.get_last_lr()[0]
                unscaled_loss = loss_val * accumulation_steps
                # Calculate perplexity for evaluation
                perplexity = torch.exp(unscaled_loss).item() if unscaled_loss.item() < 10 else float('inf')
                logger.train_step(step, float(unscaled_loss), current_lr, epoch)
                if step % (print_freq * 10) == 0:  # Log perplexity less frequently
                    logger.info(f"Perplexity: {perplexity:.2f}")
            
            # Validation
            if step % val_freq == 0 and step > 0:
                with ValidationSkipSamplesContext(train_ds):
                    think.eval()
                    proj_a.eval()
                    proj_v.eval()
                    val_loss_sum = 0.0
                    val_count = 0
                    
                    with torch.no_grad():
                        for val_data in val_dl:
                            val_emb, val_targets, val_mask = process_batch(val_data, is_training=False, use_amp_flag=use_amp)
                        try:
                            if use_amp:
                                with autocast(device_type='cuda'):
                                    val_logits = think(embeddings=val_emb)
                                    val_loss = loss_fn(val_logits.view(-1, val_logits.size(-1)), val_targets.view(-1))
                            else:
                                val_logits = think(embeddings=val_emb)
                                val_loss = loss_fn(val_logits.view(-1, val_logits.size(-1)), val_targets.view(-1))
                            
                            # Validate validation loss
                            validate_loss(val_loss, min_loss=-1e6, max_loss=1e6)
                            val_loss_sum += float(val_loss.detach())
                            val_count += 1
                            # Free validation tensors
                            del val_logits, val_loss
                            except RuntimeError as e:
                                error_msg = str(e)
                                if "NaN detected in attention probabilities after softmax" in error_msg or "Numerical instability" in error_msg:
                                    logger.error(f"Step {step}: {e}")
                                    logger.error("Reloading from last checkpoint...")
                                    # Reload from last checkpoint
                                    reloaded_step = reload_from_last_checkpoint(
                                        cfg["save_dir"], "omni_step_", device, logger, think, opt, scheduler, scaler
                                    )
                                    if reloaded_step > 0:
                                        step = reloaded_step
                                        # Also reload proj_a and proj_v from checkpoint
                                        checkpoint_files = [f for f in os.listdir(cfg["save_dir"]) if f.startswith("omni_step_") and f.endswith(".pt")]
                                        if checkpoint_files:
                                            step_numbers = []
                                            for f in checkpoint_files:
                                                try:
                                                    step_num = int(f.replace("omni_step_", "").replace(".pt", ""))
                                                    step_numbers.append((step_num, f))
                                                except:
                                                    continue
                                            if step_numbers:
                                                step_numbers.sort(key=lambda x: x[0], reverse=True)
                                                last_checkpoint = os.path.join(cfg["save_dir"], step_numbers[0][1])
                                                checkpoint = torch.load(last_checkpoint, map_location=device)
                                                if isinstance(checkpoint, dict):
                                                    if "proj_a" in checkpoint:
                                                        proj_a.load_state_dict(checkpoint["proj_a"])
                                                    if "proj_v" in checkpoint:
                                                        proj_v.load_state_dict(checkpoint["proj_v"])
                                        start_epoch, start_batch_idx = calculate_resume_position(step, steps_per_epoch)
                                        initial_step = step
                                        logger.info(f"Resuming from step {step} (epoch {start_epoch}, batch {start_batch_idx}/{steps_per_epoch})")
                                    think.train()
                                    proj_a.train()
                                    proj_v.train()
                                    break
                                else:
                                    logger.warning(f"Step {step}: Invalid validation loss: {e}")
                                    # Continue with other validation batches
                            if val_count >= 10:  # Limit validation batches
                                break
                    
                    avg_val_loss = val_loss_sum / val_count
                    logger.val_step(step, avg_val_loss, epoch)
                    
                    think.train()
                    proj_a.train()
                    proj_v.train()
            
            # Periodic checkpointing
            if step % checkpoint_freq == 0 and step > 0:
                checkpoint_path = os.path.join(cfg["save_dir"], f"omni_step_{step}.pt")
                os.makedirs(cfg["save_dir"], exist_ok=True)
                checkpoint_data = {
                    "thinker": think.state_dict(),
                    "proj_a": proj_a.state_dict(),
                    "proj_v": proj_v.state_dict(),
                    "optimizer": opt.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "step": step
                }
                if scaler is not None:
                    checkpoint_data["scaler"] = scaler.state_dict()
                torch.save(checkpoint_data, checkpoint_path)
                logger.checkpoint(step, checkpoint_path)
                # Clean up old checkpoints (keep only last one)
                cleanup_old_checkpoints(cfg["save_dir"], "omni_step_", keep_last_n=1)
            
            if step >= cfg["max_steps"]:
                os.makedirs(cfg["save_dir"], exist_ok=True)
                final_path = os.path.join(cfg["save_dir"], "omni.pt")
                checkpoint_data = {
                    "thinker": think.state_dict(),
                    "proj_a": proj_a.state_dict(),
                    "proj_v": proj_v.state_dict(),
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
            think.eval()
            proj_a.eval()
            proj_v.eval()
            val_loss_sum = 0.0
            val_count = 0
            with torch.no_grad():
            for val_data in val_dl:
                val_emb, val_targets, val_mask = process_batch(val_data, is_training=False, use_amp_flag=use_amp)
                try:
                    if use_amp:
                        with autocast():
                            val_logits = think(embeddings=val_emb)
                            val_loss = loss_fn(val_logits.view(-1, val_logits.size(-1)), val_targets.view(-1))
                    else:
                        val_logits = think(embeddings=val_emb)
                        val_loss = loss_fn(val_logits.view(-1, val_logits.size(-1)), val_targets.view(-1))
                    
                    # Validate validation loss
                    validate_loss(val_loss, min_loss=-1e6, max_loss=1e6)
                    val_loss_sum += float(val_loss.detach())
                    val_count += 1
                    # Free validation tensors
                    del val_logits, val_loss
                except RuntimeError as e:
                    error_msg = str(e)
                    if "NaN detected in attention probabilities after softmax" in error_msg or "Numerical instability" in error_msg:
                        logger.error(f"Epoch {epoch}: {e}")
                        logger.error("Reloading from last checkpoint...")
                        # Reload from last checkpoint
                        reloaded_step = reload_from_last_checkpoint(
                            cfg["save_dir"], "omni_step_", device, logger, think, opt, scheduler, scaler
                        )
                        if reloaded_step > 0:
                            step = reloaded_step
                            # Also reload proj_a and proj_v from checkpoint
                            checkpoint_files = [f for f in os.listdir(cfg["save_dir"]) if f.startswith("omni_step_") and f.endswith(".pt")]
                            if checkpoint_files:
                                step_numbers = []
                                for f in checkpoint_files:
                                    try:
                                        step_num = int(f.replace("omni_step_", "").replace(".pt", ""))
                                        step_numbers.append((step_num, f))
                                    except:
                                        continue
                                if step_numbers:
                                    step_numbers.sort(key=lambda x: x[0], reverse=True)
                                    last_checkpoint = os.path.join(cfg["save_dir"], step_numbers[0][1])
                                    checkpoint = torch.load(last_checkpoint, map_location=device)
                                    if isinstance(checkpoint, dict):
                                        if "proj_a" in checkpoint:
                                            proj_a.load_state_dict(checkpoint["proj_a"])
                                        if "proj_v" in checkpoint:
                                            proj_v.load_state_dict(checkpoint["proj_v"])
                            start_epoch = step // steps_per_epoch
                            start_batch_idx = step % steps_per_epoch
                            initial_step = step
                            logger.info(f"Resuming from step {step} (epoch {start_epoch}, batch {start_batch_idx}/{steps_per_epoch})")
                        think.train()
                        proj_a.train()
                        proj_v.train()
                        break
                    else:
                        logger.warning(f"Epoch {epoch}: Invalid validation loss: {e}")
                        # Continue with other validation batches
        
        avg_val_loss = val_loss_sum / max(val_count, 1)
        logger.epoch_end(epoch, train_loss=None, val_loss=avg_val_loss)
        
        # Restore skip_samples after validation
        if hasattr(underlying_ds, 'skip_samples'):
            underlying_ds.skip_samples = original_skip_samples
        
        # Save at end of epoch if max_steps not reached
        if step < cfg["max_steps"]:
            os.makedirs(cfg["save_dir"], exist_ok=True)
            final_path = os.path.join(cfg["save_dir"], "omni.pt")
            checkpoint_data = {
                "thinker": think.state_dict(),
                "proj_a": proj_a.state_dict(),
                "proj_v": proj_v.state_dict(),
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
