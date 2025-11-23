
import argparse, json, os, torch
from functools import partial
from torch import nn
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from omni.codec import RVQ
from omni.talker import TalkerTiny
from omni.utils import (
    set_seed, get_lr_scheduler, clip_gradients, SimpleLogger, validate_loss, 
    check_gradient_explosion, TTSDataset,
    load_checkpoint, setup_resume_data_loading, calculate_resume_position,
    ValidationSkipSamplesContext, collate_mel_fn, analyze_tts_dataset,
    save_training_metadata, load_training_metadata
)
from tqdm import tqdm

def main(cfg):
    # Set random seed for reproducibility
    seed = cfg.get("seed", 42)
    set_seed(seed)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    save_dir = cfg.get("save_dir", "checkpoints/talker_tiny")
    os.makedirs(save_dir, exist_ok=True)
    sr = cfg.get("sample_rate", 16000)
    n_mels = cfg.get("n_mels", 128)
    frame_ms = cfg.get("frame_ms", 80)
    
    print("Using TTSDataset (streaming, lower memory, sequential I/O)")
    
    # torch.compile() support (optional, PyTorch 2.0+)
    use_compile = cfg.get("use_compile", False)
    
    codebooks = cfg.get("codebooks", 2)
    codebook_size = cfg.get("codebook_size", 128)
    rvq = RVQ(codebooks, codebook_size, d=64, compile_model=use_compile).to(device)
    talker = TalkerTiny(
        cfg.get("d_model", 384), 
        cfg.get("n_layers", 8), 
        cfg.get("n_heads", 6), 
        cfg.get("d_ff", 1536), 
        codebooks, 
        codebook_size, 
        cfg.get("dropout", 0.1),
        use_gqa=cfg.get("use_gqa", False),
        use_swiglu=cfg.get("use_swiglu", True),
        rope_theta=cfg.get("rope_theta", 10000.0),
        compile_model=use_compile
    ).to(device)
    opt = torch.optim.AdamW(list(rvq.parameters())+list(talker.parameters()), lr=cfg.get("lr", 3e-4), weight_decay=cfg.get("wd", 0.01))
    # Use reduction='none' to get per-element losses, then we'll mask padding manually
    # Note: RVQ codes are 0-127, so we can't use ignore_index=0 (0 is a valid code)
    loss_fn = nn.CrossEntropyLoss(reduction='none')  # Get per-element losses for masking
    
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

    # Get TTS CSV path
    tts_csv = cfg.get("tts_csv", "data/audio/production_tts.csv")
    
    # Check if metadata exists (contains previously calculated values)
    model_name = "talker"
    metadata = load_training_metadata(save_dir, model_name)
    
    if metadata and not cfg.get("recalculate_dataset_stats", False):
        # Load calculated values from metadata (avoid recalculating)
        print("Loading dataset statistics from metadata...")
        max_mel_length_dynamic = metadata.get("max_mel_length", None)
        print(f"Mel length: {max_mel_length_dynamic} frames (from metadata)")
    else:
        # Auto-calculate max_mel_length from dataset using percentile (minimizes padding)
        print("Analyzing TTS dataset for mel length...")
        # Sample a subset for mel length calculation (optional, can be configured)
        sample_size = cfg.get("max_mel_length_sample_size", None)  # None = check all files
        # Percentile threshold for minimizing padding (default: 95% coverage)
        mel_percentile = cfg.get("max_mel_length_percentile", 95.0)
        max_mel_length_dynamic = analyze_tts_dataset(
            tts_csv, sr=sr, n_mels=n_mels, frame_ms=frame_ms, 
            sample_size=sample_size, mel_percentile=mel_percentile
        )
        frame_rate = sr / int(sr * frame_ms / 1000)  # frames/sec for talker
        print(f"Mel length at {mel_percentile}th percentile: {max_mel_length_dynamic} frames (~{max_mel_length_dynamic / frame_rate:.2f} seconds, covers {mel_percentile}% of data, minimizes padding)")
        print(f"  Note: ~{100 - mel_percentile:.1f}% of data will be truncated if longer (acceptable for outliers)")
        
        # Save calculated values to metadata (so we don't recalculate next time)
        training_metadata = {
            "step": 0,  # Will be updated when we save checkpoints
            "epoch": 0,
            "max_mel_length": max_mel_length_dynamic
        }
        save_training_metadata(save_dir, model_name, training_metadata)
        print("✓ Saved dataset statistics to metadata (will be reused on next run)")
    
    # Allow override from config, but default to auto-calculated value
    max_mel_length = cfg.get("max_mel_length", max_mel_length_dynamic)
    if max_mel_length != max_mel_length_dynamic:
        print(f"⚠ Warning: Config max_mel_length={max_mel_length} differs from dataset max length={max_mel_length_dynamic}")
        print(f"  Using config value: {max_mel_length}")
    else:
        print(f"✓ Using auto-calculated max_mel_length: {max_mel_length}")
    
    if use_compile:
        print(f"Using fixed max_mel_length={max_mel_length} for CUDA graphs compatibility")
    
    # Split dataset for validation
    val_split = cfg.get("val_split", 0.1)  # 10% for validation
    
    train_ds = TTSDataset(
        tts_csv, 
        sr=sr, 
        n_mels=n_mels, 
        frame_ms=frame_ms, 
        cfg=cfg,
        shuffle_buffer_size=cfg.get("shuffle_buffer_size", 10000),
        seed=seed,
        skip_samples=0
    )
    train_ds._val_split = val_split
    train_ds._val_mode = False  # Training mode
    
    val_ds = TTSDataset(
        tts_csv, 
        sr=sr, 
        n_mels=n_mels, 
        frame_ms=frame_ms, 
        cfg=cfg,
        shuffle_buffer_size=0,  # No shuffling for validation
        seed=seed,  # Same seed for consistent hash-based split
        skip_samples=0
    )
    val_ds._val_split = val_split
    val_ds._val_mode = True  # Validation mode
    
    # Create collate function with fixed max length using functools.partial (pickleable for Windows multiprocessing)
    collate_fn_with_max = partial(collate_mel_fn, max_mel_length=max_mel_length)
    
    # Approximate sizes for logging (will count if needed)
    try:
        total_size = train_ds.get_length()
        train_size = int(total_size * (1 - val_split))
        val_size = total_size - train_size
    except:
        train_size = val_size = None  # Unknown size
    
    # Note: shuffle=False for IterableDataset (shuffling handled internally)
    # Use module-level collate function for Windows multiprocessing compatibility
    train_dl = DataLoader(train_ds, batch_size=cfg.get("batch_size", 4), shuffle=False, num_workers=cfg.get("num_workers", 2), drop_last=cfg.get("drop_last", True), collate_fn=collate_fn_with_max)
    val_dl = DataLoader(val_ds, batch_size=cfg.get("batch_size", 4), shuffle=False, num_workers=cfg.get("num_workers", 2), drop_last=cfg.get("drop_last", True), collate_fn=collate_fn_with_max)
    
    # Initialize logger
    logger = SimpleLogger("Talker")
    
    step=0
    rvq.train()
    talker.train()
    max_epochs = cfg.get("max_epochs", 9999)
    print_freq = cfg.get("print_freq", 50)
    checkpoint_freq = cfg.get("checkpoint_freq", 500)  # Save checkpoint every N steps
    val_freq = cfg.get("val_freq", 200)  # Validate every N steps
    
    # Resume from checkpoint if available
    step = 0
    step, metadata = load_checkpoint(
        save_dir, 
        model_name, 
        device, 
        logger,
        state_dict_loaders={
            "rvq": (rvq, rvq.load_state_dict),
            "talker": (talker, talker.load_state_dict),
            "optimizer": (opt, opt.load_state_dict),
            "scheduler": (scheduler, scheduler.load_state_dict),
            "scaler": (scaler, scaler.load_state_dict) if scaler is not None else None
        }
    )
    # Load scaler from model file if needed
    if step > 0 and scaler is not None:
        model_path = os.path.join(save_dir, f"{model_name}.pt")
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=device)
            if isinstance(checkpoint, dict) and "scaler" in checkpoint:
                scaler.load_state_dict(checkpoint["scaler"])
    
    # Update skip_samples for dataset if resuming
    batch_size = cfg.get("batch_size", 4)
    new_train_dl = setup_resume_data_loading(
        train_ds, step, batch_size, logger,
        train_dl_kwargs={
            "num_workers": cfg.get("num_workers", 2),
            "drop_last": cfg.get("drop_last", True),
            "collate_fn": collate_fn_with_max
        }
    )
    if new_train_dl is not None:
        train_dl = new_train_dl
    
    max_steps = cfg.get("max_steps", 5000)
    logger.training_start(max_steps, train_size, val_size)
    
    # Calculate steps per epoch and determine starting epoch/position
    # For IterableDataset, we can't use len() directly, so calculate from dataset size
    batch_size = cfg.get("batch_size", 4)
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
            train_dl = DataLoader(train_ds, batch_size=cfg.get("batch_size", 4), shuffle=False, 
                                 num_workers=cfg.get("num_workers", 2), 
                                 drop_last=cfg.get("drop_last", True), 
                                 collate_fn=collate_fn_with_max)
        
        # Create progress bar with correct starting position when resuming mid-epoch
        remaining_epochs = max_epochs - epoch - 1
        pbar_desc = f"epoch{epoch}/{max_epochs-1} (remaining:{remaining_epochs}) step{step}"
        if epoch == start_epoch and start_batch_idx > 0:
            pbar = tqdm(train_dl, desc=pbar_desc, initial=start_batch_idx, total=steps_per_epoch)
        else:
            pbar = tqdm(train_dl, desc=pbar_desc, total=steps_per_epoch)
        
        # Start enumeration from the correct position when resuming mid-epoch
        enum_start = start_batch_idx if (epoch == start_epoch and start_batch_idx > 0) else 0
        for batch_idx, batch_data in enumerate(pbar, start=enum_start):
            # Skip batches if resuming mid-epoch
            # batch_idx already represents the position in the epoch when enum_start > 0
            if epoch == start_epoch and initial_step > 0:
                current_batch_step = epoch * steps_per_epoch + batch_idx
                if current_batch_step < initial_step:
                    continue
            
            # Unpack batch data (now includes mel_lengths)
            if isinstance(batch_data, tuple) and len(batch_data) == 2:
                mel, mel_lengths = batch_data
                mel_lengths = mel_lengths.to(device)
            else:
                # Fallback for old collate function that doesn't return lengths
                mel = batch_data
                # Estimate lengths from mel energy (fallback)
                mel_energy = mel.abs().sum(dim=-1)  # (B, T)
                threshold = mel_energy.max(dim=1, keepdim=True)[0] * 0.01
                mel_lengths = (mel_energy > threshold).sum(dim=1)  # (B,)
            
            # Update progress bar description
            remaining_epochs = max_epochs - epoch - 1
            pbar.set_description(f"epoch{epoch}/{max_epochs-1} (remaining:{remaining_epochs}) step{step} batch{batch_idx}")
            mel = mel.to(device)  # (B,T,128)
            
            # Mark step begin for CUDAGraphs optimization
            if device == "cuda":
                torch.compiler.cudagraph_mark_step_begin()
            
            # Create mask to exclude padding frames from loss
            # mel_lengths: (B,) - actual lengths before padding
            B, T = mel.shape[0], mel.shape[1]
            mask = torch.arange(T, device=device).unsqueeze(0) < mel_lengths.unsqueeze(1)  # (B, T)
            # Exclude first frame (BOS) from loss, only predict from frame 1 onwards
            mask = mask[:, 1:]  # (B, T-1) - mask for target frames
            
            # Forward pass with mixed precision
            if use_amp:
                with autocast(device_type='cuda'):
                    # Batch encode all frames at once (optimized)
                    idxs = rvq.encode(mel)  # (B,T,2) - encodes all frames in batch
                    # AR training: predict current codes from previous codes
                    prev = torch.roll(idxs, 1, dims=1); prev[:,0,:]=0
                    base_logit, res_logit = talker(prev)  # (B, T, codebook_size)
                    
                    # Targets: codes shifted by 1 (predict next code)
                    base_targets = idxs[:, 1:, 0]  # (B, T-1)
                    res_targets = idxs[:, 1:, 1]    # (B, T-1)
                    
                    # Flatten and apply mask
                    base_logit_flat = base_logit[:, :-1, :].reshape(-1, base_logit.size(-1))  # (B*(T-1), codebook_size)
                    res_logit_flat = res_logit[:, :-1, :].reshape(-1, res_logit.size(-1))
                    base_targets_flat = base_targets.reshape(-1)  # (B*(T-1),)
                    res_targets_flat = res_targets.reshape(-1)
                    mask_flat = mask.reshape(-1)  # (B*(T-1),)
                    
                    # Compute per-element losses
                    base_loss_per_elem = loss_fn(base_logit_flat, base_targets_flat)  # (B*(T-1),)
                    res_loss_per_elem = loss_fn(res_logit_flat, res_targets_flat)    # (B*(T-1),)
                    
                    # Apply mask: only compute loss on non-padding frames
                    valid_elements = mask_flat.float().sum().clamp(min=1)
                    base_loss = (base_loss_per_elem * mask_flat.float()).sum() / valid_elements
                    res_loss = (res_loss_per_elem * mask_flat.float()).sum() / valid_elements
                    
                    loss = base_loss + res_loss
                    # Free intermediate tensors
                    del base_logit, res_logit, prev, mask
            else:
                # Batch encode all frames at once (optimized)
                idxs = rvq.encode(mel)  # (B,T,2) - encodes all frames in batch
                # AR training: predict current codes from previous codes
                prev = torch.roll(idxs, 1, dims=1); prev[:,0,:]=0
                base_logit, res_logit = talker(prev)  # (B, T, codebook_size)
                
                # Targets: codes shifted by 1
                base_targets = idxs[:, 1:, 0]  # (B, T-1)
                res_targets = idxs[:, 1:, 1]    # (B, T-1)
                
                # Flatten and apply mask
                base_logit_flat = base_logit[:, :-1, :].reshape(-1, base_logit.size(-1))
                res_logit_flat = res_logit[:, :-1, :].reshape(-1, res_logit.size(-1))
                base_targets_flat = base_targets.reshape(-1)
                res_targets_flat = res_targets.reshape(-1)
                mask_flat = mask.reshape(-1)
                
                # Compute per-element losses
                base_loss_per_elem = loss_fn(base_logit_flat, base_targets_flat)  # (B*(T-1),)
                res_loss_per_elem = loss_fn(res_logit_flat, res_targets_flat)    # (B*(T-1),)
                
                # Apply mask: only compute loss on non-padding frames
                valid_elements = mask_flat.float().sum().clamp(min=1)
                base_loss = (base_loss_per_elem * mask_flat.float()).sum() / valid_elements
                res_loss = (res_loss_per_elem * mask_flat.float()).sum() / valid_elements
                
                loss = base_loss + res_loss
                # Free intermediate tensors
                del base_logit, res_logit, prev, mask
            
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
                
                # Gradient clipping first (already unscaled if using AMP)
                # Clip gradients to prevent explosion, then check if still too high
                try:
                    grad_norm_rvq_before = clip_gradients(rvq, max_grad_norm)
                    grad_norm_talker_before = clip_gradients(talker, max_grad_norm)
                    
                    # Check for gradient explosion AFTER clipping
                    # Use a higher threshold (10x max_grad_norm) since we've already clipped
                    # This allows clipping to fix most cases, only skip if truly exploded
                    explosion_threshold = max(100.0, max_grad_norm * 10)
                    grad_norm_rvq_after, is_exploded_rvq = check_gradient_explosion(rvq, max_grad_norm=explosion_threshold, raise_on_error=False)
                    grad_norm_talker_after, is_exploded_talker = check_gradient_explosion(talker, max_grad_norm=explosion_threshold, raise_on_error=False)
                    
                    if is_exploded_rvq or is_exploded_talker:
                        logger.error(f"Step {step}: Gradient explosion detected after clipping (rvq: {grad_norm_rvq_before:.2f}->{grad_norm_rvq_after:.2f}, talker: {grad_norm_talker_before:.2f}->{grad_norm_talker_after:.2f}). Skipping this batch.")
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
            
            # Periodic checkpointing - save only model file and metadata
            if step % checkpoint_freq == 0 and step > 0:
                # Save model weights only (overwrite existing file)
                model_path = os.path.join(save_dir, f"{model_name}.pt")
                model_data = {
                    "rvq": rvq.state_dict(),
                    "talker": talker.state_dict(),
                    "optimizer": opt.state_dict(),
                    "scheduler": scheduler.state_dict(),
                }
                if scaler is not None:
                    model_data["scaler"] = scaler.state_dict()
                torch.save(model_data, model_path)
                
                # Save training metadata (step, calculated values, etc.)
                training_metadata = {
                    "step": step,
                    "epoch": epoch,
                    "max_mel_length": max_mel_length_dynamic
                }
                save_training_metadata(save_dir, model_name, training_metadata)
                logger.checkpoint(step, model_path)
            
            # Validation
            if step % val_freq == 0 and step > 0:
                with ValidationSkipSamplesContext(train_ds):
                    rvq.eval()
                    talker.eval()
                    val_loss_sum = 0.0
                    val_count = 0
                    with torch.no_grad():
                        for val_batch_data in val_dl:
                            # Unpack validation batch (may include mel_lengths)
                            if isinstance(val_batch_data, tuple) and len(val_batch_data) == 2:
                                val_mel, val_mel_lengths = val_batch_data
                                val_mel_lengths = val_mel_lengths.to(device)
                            else:
                                val_mel = val_batch_data
                                # Fallback: estimate lengths
                                val_mel_energy = val_mel.abs().sum(dim=-1)
                                threshold = val_mel_energy.max(dim=1, keepdim=True)[0] * 0.01
                                val_mel_lengths = (val_mel_energy > threshold).sum(dim=1)
                            
                            # Create mask for validation
                            val_B, val_T = val_mel.shape[0], val_mel.shape[1]
                            val_mask = torch.arange(val_T, device=device).unsqueeze(0) < val_mel_lengths.unsqueeze(1)
                            val_mask = val_mask[:, 1:]  # (B, T-1)
                            
                            val_mel = val_mel.to(device)
                            if use_amp:
                                with autocast(device_type='cuda'):
                                    # Batch encode all frames at once (optimized)
                                    val_idxs = rvq.encode(val_mel)  # (B,T,2)
                                    val_prev = torch.roll(val_idxs, 1, dims=1); val_prev[:,0,:]=0
                                    val_base_logit, val_res_logit = talker(val_prev)
                                    
                                    # Apply masking like in training
                                    val_base_targets = val_idxs[:, 1:, 0]
                                    val_res_targets = val_idxs[:, 1:, 1]
                                    val_base_logit_flat = val_base_logit[:, :-1, :].reshape(-1, val_base_logit.size(-1))
                                    val_res_logit_flat = val_res_logit[:, :-1, :].reshape(-1, val_res_logit.size(-1))
                                    val_base_targets_flat = val_base_targets.reshape(-1)
                                    val_res_targets_flat = val_res_targets.reshape(-1)
                                    val_mask_flat = val_mask.reshape(-1)
                                    
                                    val_base_loss_per_elem = loss_fn(val_base_logit_flat, val_base_targets_flat)
                                    val_res_loss_per_elem = loss_fn(val_res_logit_flat, val_res_targets_flat)
                                    val_valid_elements = val_mask_flat.float().sum().clamp(min=1)
                                    val_base_loss = (val_base_loss_per_elem * val_mask_flat.float()).sum() / val_valid_elements
                                    val_res_loss = (val_res_loss_per_elem * val_mask_flat.float()).sum() / val_valid_elements
                                    val_loss = val_base_loss + val_res_loss
                            else:
                                # Batch encode all frames at once (optimized)
                                val_idxs = rvq.encode(val_mel)  # (B,T,2)
                                val_prev = torch.roll(val_idxs, 1, dims=1); val_prev[:,0,:]=0
                                val_base_logit, val_res_logit = talker(val_prev)
                                
                                # Apply masking like in training
                                val_base_targets = val_idxs[:, 1:, 0]
                                val_res_targets = val_idxs[:, 1:, 1]
                                val_base_logit_flat = val_base_logit[:, :-1, :].reshape(-1, val_base_logit.size(-1))
                                val_res_logit_flat = val_res_logit[:, :-1, :].reshape(-1, val_res_logit.size(-1))
                                val_base_targets_flat = val_base_targets.reshape(-1)
                                val_res_targets_flat = val_res_targets.reshape(-1)
                                val_mask_flat = val_mask.reshape(-1)
                                
                                val_base_loss_per_elem = loss_fn(val_base_logit_flat, val_base_targets_flat)
                                val_res_loss_per_elem = loss_fn(val_res_logit_flat, val_res_targets_flat)
                                val_valid_elements = val_mask_flat.float().sum().clamp(min=1)
                                val_base_loss = (val_base_loss_per_elem * val_mask_flat.float()).sum() / val_valid_elements
                                val_res_loss = (val_res_loss_per_elem * val_mask_flat.float()).sum() / val_valid_elements
                                val_loss = val_base_loss + val_res_loss
                            
                            # Validate validation loss
                            try:
                                validate_loss(val_loss, min_loss=-1e6, max_loss=1e6)
                                val_loss_sum += float(val_loss.detach())
                                val_count += 1
                            except RuntimeError as e:
                                logger.warning(f"Step {step}: Invalid validation loss: {e}")
                                # Continue with other validation batches
                            if val_count >= 10:  # Limit validation batches
                                break
                
                    avg_val_loss = val_loss_sum / val_count
                    logger.val_step(step, avg_val_loss, epoch)
                    
                    rvq.train()
                    talker.train()
            
            if step >= max_steps:
                # Save final model weights
                final_path = os.path.join(save_dir, f"{model_name}.pt")
                model_data = {
                    "rvq": rvq.state_dict(),
                    "talker": talker.state_dict(),
                    "optimizer": opt.state_dict(),
                    "scheduler": scheduler.state_dict(),
                }
                if scaler is not None:
                    model_data["scaler"] = scaler.state_dict()
                torch.save(model_data, final_path)
                
                # Save final training metadata
                training_metadata = {
                    "step": step,
                    "epoch": epoch,
                    "max_mel_length": max_mel_length_dynamic
                }
                save_training_metadata(save_dir, model_name, training_metadata)
                logger.info(f"Final model saved to {save_dir}")
                logger.training_end(step)
                return
        
        # Final validation at end of epoch
        with ValidationSkipSamplesContext(train_ds):
            rvq.eval()
            talker.eval()
            val_loss_sum = 0.0
            val_count = 0
            with torch.no_grad():
                for val_batch_data in val_dl:
                    # Unpack validation batch (may include mel_lengths)
                    if isinstance(val_batch_data, tuple) and len(val_batch_data) == 2:
                        val_mel, val_mel_lengths = val_batch_data
                        val_mel_lengths = val_mel_lengths.to(device)
                    else:
                        val_mel = val_batch_data
                        # Fallback: estimate lengths
                        val_mel_energy = val_mel.abs().sum(dim=-1)
                        threshold = val_mel_energy.max(dim=1, keepdim=True)[0] * 0.01
                        val_mel_lengths = (val_mel_energy > threshold).sum(dim=1)
                    
                    # Create mask for validation
                    val_B, val_T = val_mel.shape[0], val_mel.shape[1]
                    val_mask = torch.arange(val_T, device=device).unsqueeze(0) < val_mel_lengths.unsqueeze(1)
                    val_mask = val_mask[:, 1:]  # (B, T-1)
                    
                    val_mel = val_mel.to(device)
                    val_mel = val_mel.to(device)
                    if use_amp:
                        with autocast(device_type='cuda'):
                            # Batch encode all frames at once (optimized)
                            val_idxs = rvq.encode(val_mel)  # (B,T,2)
                            val_prev = torch.roll(val_idxs, 1, dims=1); val_prev[:,0,:]=0
                            val_base_logit, val_res_logit = talker(val_prev)
                            
                            # Apply masking like in training
                            val_base_targets = val_idxs[:, 1:, 0]
                            val_res_targets = val_idxs[:, 1:, 1]
                            val_base_logit_flat = val_base_logit[:, :-1, :].reshape(-1, val_base_logit.size(-1))
                            val_res_logit_flat = val_res_logit[:, :-1, :].reshape(-1, val_res_logit.size(-1))
                            val_base_targets_flat = val_base_targets.reshape(-1)
                            val_res_targets_flat = val_res_targets.reshape(-1)
                            val_mask_flat = val_mask.reshape(-1)
                            
                            val_base_loss_per_elem = loss_fn(val_base_logit_flat, val_base_targets_flat)
                            val_res_loss_per_elem = loss_fn(val_res_logit_flat, val_res_targets_flat)
                            val_valid_elements = val_mask_flat.float().sum().clamp(min=1)
                            val_base_loss = (val_base_loss_per_elem * val_mask_flat.float()).sum() / val_valid_elements
                            val_res_loss = (val_res_loss_per_elem * val_mask_flat.float()).sum() / val_valid_elements
                            val_loss = val_base_loss + val_res_loss
                    else:
                        # Batch encode all frames at once (optimized)
                        val_idxs = rvq.encode(val_mel)  # (B,T,2)
                        val_prev = torch.roll(val_idxs, 1, dims=1); val_prev[:,0,:]=0
                        val_base_logit, val_res_logit = talker(val_prev)
                        
                        # Apply masking like in training
                        val_base_targets = val_idxs[:, 1:, 0]
                        val_res_targets = val_idxs[:, 1:, 1]
                        val_base_logit_flat = val_base_logit[:, :-1, :].reshape(-1, val_base_logit.size(-1))
                        val_res_logit_flat = val_res_logit[:, :-1, :].reshape(-1, val_res_logit.size(-1))
                        val_base_targets_flat = val_base_targets.reshape(-1)
                        val_res_targets_flat = val_res_targets.reshape(-1)
                        val_mask_flat = val_mask.reshape(-1)
                        
                        val_base_loss_per_elem = loss_fn(val_base_logit_flat, val_base_targets_flat)
                        val_res_loss_per_elem = loss_fn(val_res_logit_flat, val_res_targets_flat)
                        val_valid_elements = val_mask_flat.float().sum().clamp(min=1)
                        val_base_loss = (val_base_loss_per_elem * val_mask_flat.float()).sum() / val_valid_elements
                        val_res_loss = (val_res_loss_per_elem * val_mask_flat.float()).sum() / val_valid_elements
                        val_loss = val_base_loss + val_res_loss
                    
                    # Validate validation loss
                    try:
                        validate_loss(val_loss, min_loss=-1e6, max_loss=1e6)
                        val_loss_sum += float(val_loss.detach())
                        val_count += 1
                        # Free validation tensors
                        del val_base_logit, val_res_logit, val_idxs, val_prev, val_loss
                    except RuntimeError as e:
                        logger.warning(f"Epoch {epoch}: Invalid validation loss: {e}")
                        # Continue with other validation batches
                        del val_base_logit, val_res_logit, val_idxs, val_prev, val_loss
        
            avg_val_loss = val_loss_sum / max(val_count, 1)
            logger.epoch_end(epoch, train_loss=None, val_loss=avg_val_loss)
            
            rvq.train()
            talker.train()
        
        # Save at end of epoch (checkpoint for resuming)
        final_path = os.path.join(save_dir, f"{model_name}.pt")
        model_data = {
            "rvq": rvq.state_dict(),
            "talker": talker.state_dict(),
            "optimizer": opt.state_dict(),
            "scheduler": scheduler.state_dict(),
        }
        if scaler is not None:
            model_data["scaler"] = scaler.state_dict()
        torch.save(model_data, final_path)
        
        # Save training metadata
        training_metadata = {
            "step": step,
            "epoch": epoch,
            "max_mel_length": max_mel_length_dynamic
        }
        save_training_metadata(save_dir, model_name, training_metadata)
        logger.info(f"Model saved to {save_dir} at end of epoch {epoch}, step {step}")
        
        # Check if we've reached max_steps after epoch completion
        if step >= max_steps:
            logger.info(f"Reached max_steps={max_steps}. Training complete.")
            logger.training_end(step)
            return
        
        # Continue to next epoch
        start_batch_idx = 0  # Reset batch index for new epoch

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--config", required=True)
    cfg = json.load(open(ap.parse_args().config))
    main(cfg)
