
import argparse, json, os, torch
from torch import nn
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from omni.audio_encoder import AudioEncoderTiny
from omni.utils import set_seed, get_lr_scheduler, clip_gradients, SimpleLogger, validate_loss, check_gradient_explosion, ASRDataset, load_checkpoint, setup_resume_data_loading, calculate_resume_position, cleanup_old_checkpoints, ValidationSkipSamplesContext, collate_mel_text_fn
from tqdm import tqdm

def main(cfg):
    # Set random seed for reproducibility
    seed = cfg.get("seed", 42)
    set_seed(seed)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(cfg["save_dir"], exist_ok=True)
    sr = cfg.get("sample_rate", 16000)
    n_mels = cfg.get("mel_bins", 128)
    downsample_factor = cfg.get("downsample_time", 8)  # 8x for 12.5 Hz (16000/160/8 = 12.5)
    # torch.compile() support (optional, PyTorch 2.0+)
    use_compile = cfg.get("use_compile", False)
    
    model = AudioEncoderTiny(
        cfg["d_model"], 
        cfg["n_heads"], 
        cfg["d_ff"], 
        cfg["n_layers"], 
        cfg["dropout"],
        downsample_factor=downsample_factor,
        compile_model=use_compile
    ).to(device)
    # Improved CTC head: proper character vocabulary
    # Build character vocabulary (printable ASCII + special tokens)
    # This will be used in the training loop for encoding text
    char_to_idx = {}
    for i in range(32, 127):  # Printable ASCII
        char_to_idx[chr(i)] = len(char_to_idx) + 1
    char_to_idx['\n'] = len(char_to_idx) + 1
    char_to_idx['\t'] = len(char_to_idx) + 1
    char_to_idx['<UNK>'] = len(char_to_idx) + 1
    vocab_size_ctc = len(char_to_idx) + 1  # +1 for blank token (0)
    
    # Use proper vocabulary size instead of toy 64
    vocab = cfg.get("ctc_vocab_size", vocab_size_ctc)  # Proper char vocab size
    head = nn.Linear(cfg["d_model"], vocab).to(device)
    ctc_loss = nn.CTCLoss(blank=0, zero_infinity=True)
    print(f"CTC vocabulary size: {vocab} (includes blank token)")
    print(f"Character vocabulary: {len(char_to_idx)} characters")
    opt = torch.optim.AdamW(list(model.parameters())+list(head.parameters()), lr=cfg["lr"], weight_decay=cfg["wd"])
    
    unk_idx = char_to_idx['<UNK>']
    max_text_len = cfg.get("max_text_len", 64)
    head.char_to_idx = char_to_idx
    head.unk_idx = unk_idx
    head.max_text_len = max_text_len

    def encode_text_batch(text_batch):
        targets = []
        lengths = []
        for t in text_batch:
            ids = [char_to_idx.get(c, unk_idx) for c in t[:max_text_len]]
            if not ids:
                ids = [unk_idx]
            targets.append(torch.tensor(ids, dtype=torch.long))
            lengths.append(len(ids))
        if targets:
            targets = torch.cat(targets)
        else:
            targets = torch.empty(0, dtype=torch.long)
        return targets, torch.tensor(lengths, dtype=torch.long)

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

    # Fixed maximum mel length for uniform batch sizes (required for CUDA graphs)
    # This ensures all batches have the same shape, preventing CUDA graphs errors
    max_mel_length = cfg.get("max_mel_length", 2048)  # Default: 2048 frames (~20s at 100Hz)
    if use_compile:
        print(f"Using fixed max_mel_length={max_mel_length} for CUDA graphs compatibility")
    
    # Create collate function with fixed max length
    def make_collate_fn(max_len):
        def collate(batch):
            return collate_mel_text_fn(batch, max_mel_length=max_len)
        return collate
    
    collate_fn_with_max = make_collate_fn(max_mel_length)
    
    # Split dataset for validation
    val_split = cfg.get("val_split", 0.1)  # 10% for validation
    
    train_ds = ASRDataset(
        cfg["train_csv"], 
        sr=sr, 
        n_mels=n_mels, 
        cfg=cfg,
        shuffle_buffer_size=cfg.get("shuffle_buffer_size", 10000),
        seed=seed,
        skip_samples=0  # Will be updated after checkpoint load
    )
    train_ds._val_split = val_split
    train_ds._val_mode = False  # Training mode
    
    val_ds = ASRDataset(
        cfg["train_csv"], 
        sr=sr, 
        n_mels=n_mels, 
        cfg=cfg,
        shuffle_buffer_size=0,  # No shuffling for validation
        seed=seed,  # Same seed for consistent hash-based split
        skip_samples=0  # Don't skip validation samples
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
    train_dl = DataLoader(train_ds, batch_size=cfg.get("batch_size", 4), shuffle=False, num_workers=cfg.get("num_workers", 2), drop_last=cfg.get("drop_last", True), collate_fn=collate_fn_with_max)
    val_dl = DataLoader(val_ds, batch_size=cfg.get("batch_size", 4), shuffle=False, num_workers=cfg.get("num_workers", 2), drop_last=False, collate_fn=collate_fn_with_max)
    
    # Initialize logger
    logger = SimpleLogger("AudioEncoder")
    
    step=0
    model.train()
    head.train()
    max_epochs = cfg.get("max_epochs", 9999)
    print_freq = cfg.get("print_freq", 100)
    checkpoint_freq = cfg.get("checkpoint_freq", 500)  # Save checkpoint every N steps
    val_freq = cfg.get("val_freq", 200)  # Validate every N steps
    
    # Resume from checkpoint if available
    step = 0
    step, resume_from = load_checkpoint(
        cfg["save_dir"], 
        "audio_enc_step_", 
        device, 
        logger,
        state_dict_loaders={
            "enc": (model, model.load_state_dict),
            "head": (head, head.load_state_dict),
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
    
    logger.training_start(cfg["max_steps"], train_size, val_size)
    
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
        # Create progress bar with correct starting position when resuming mid-epoch
        remaining_epochs = max_epochs - epoch - 1
        pbar_desc = f"epoch{epoch}/{max_epochs-1} (remaining:{remaining_epochs}) step{step}"
        if epoch == start_epoch and start_batch_idx > 0:
            pbar = tqdm(train_dl, desc=pbar_desc, initial=start_batch_idx, total=steps_per_epoch)
        else:
            pbar = tqdm(train_dl, desc=pbar_desc, total=steps_per_epoch)
        
        # Start enumeration from the correct position when resuming mid-epoch
        enum_start = start_batch_idx if (epoch == start_epoch and start_batch_idx > 0) else 0
        for batch_idx, (mel, text) in enumerate(pbar, start=enum_start):
            # Skip batches if resuming mid-epoch
            # batch_idx already represents the position in the epoch when enum_start > 0
            if epoch == start_epoch and initial_step > 0:
                current_batch_step = epoch * steps_per_epoch + batch_idx
                if current_batch_step < initial_step:
                    continue
            
            # Update progress bar description
            remaining_epochs = max_epochs - epoch - 1
            pbar.set_description(f"epoch{epoch}/{max_epochs-1} (remaining:{remaining_epochs}) step{step} batch{batch_idx}")
            mel = mel.to(device)
            
            tgt, tgt_lens = encode_text_batch(text)
            tgt = tgt.to(device)
            tgt_lens = tgt_lens.to(device)
            
            # Mark step begin for CUDAGraphs optimization
            if device == "cuda":
                torch.compiler.cudagraph_mark_step_begin()
            
            # Forward pass with mixed precision
            if use_amp:
                with autocast(device_type='cuda'):
                    x = model(mel)  # (B, T', d)
                    logit = head(x)  # (B,T',V)
                    log_prob = logit.log_softmax(-1).transpose(0,1)  # (T',B,V)
                    inp_lens = torch.full((log_prob.size(1),), log_prob.size(0), dtype=torch.long, device=log_prob.device)
                    loss = ctc_loss(log_prob, tgt, inp_lens, tgt_lens)
                    # Free intermediate tensors
                    del x, logit, log_prob
            else:
                x = model(mel)  # (B, T', d)
                logit = head(x)  # (B,T',V)
                log_prob = logit.log_softmax(-1).transpose(0,1)  # (T',B,V)
                inp_lens = torch.full((log_prob.size(1),), log_prob.size(0), dtype=torch.long, device=log_prob.device)
                loss = ctc_loss(log_prob, tgt, inp_lens, tgt_lens)
                # Free intermediate tensors
                del x, logit, log_prob
            
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
                    grad_norm_model_before = clip_gradients(model, max_grad_norm)
                    grad_norm_head_before = clip_gradients(head, max_grad_norm)
                    
                    # Check for gradient explosion AFTER clipping
                    # Use a higher threshold (10x max_grad_norm) since we've already clipped
                    # This allows clipping to fix most cases, only skip if truly exploded
                    explosion_threshold = max(100.0, max_grad_norm * 10)
                    grad_norm_model_after, is_exploded_model = check_gradient_explosion(model, max_grad_norm=explosion_threshold, raise_on_error=False)
                    grad_norm_head_after, is_exploded_head = check_gradient_explosion(head, max_grad_norm=explosion_threshold, raise_on_error=False)
                    
                    if is_exploded_model or is_exploded_head:
                        logger.error(f"Step {step}: Gradient explosion detected after clipping (model: {grad_norm_model_before:.2f}->{grad_norm_model_after:.2f}, head: {grad_norm_head_before:.2f}->{grad_norm_head_after:.2f}). Skipping this batch.")
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
            else:
                # Not accumulation step - just validate loss
                unscaled_loss = loss_val * accumulation_steps
                try:
                    validate_loss(unscaled_loss, min_loss=-1e6, max_loss=1e6)
                except RuntimeError as e:
                    logger.error(f"Step {step}: {e}")
                    logger.error("Skipping this batch due to invalid loss")
                    # Don't clear gradients here - we're accumulating
                    continue
            
            step+=1
            if step % print_freq == 0:
                current_lr = scheduler.get_last_lr()[0]
                unscaled_loss = loss_val * accumulation_steps
                logger.train_step(step, float(unscaled_loss), current_lr, epoch)
            
            # Periodic checkpointing
            if step % checkpoint_freq == 0 and step > 0:
                checkpoint_path = os.path.join(cfg["save_dir"], f"audio_enc_step_{step}.pt")
                checkpoint_data = {
                    "enc": model.state_dict(),
                    "head": head.state_dict(),
                    "optimizer": opt.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "step": step
                }
                if scaler is not None:
                    checkpoint_data["scaler"] = scaler.state_dict()
                torch.save(checkpoint_data, checkpoint_path)
                logger.checkpoint(step, checkpoint_path)
                # Clean up old checkpoints (keep only last one)
                cleanup_old_checkpoints(cfg["save_dir"], "audio_enc_step_", keep_last_n=1)
            
            # Validation
            if step % val_freq == 0 and step > 0:
                with ValidationSkipSamplesContext(train_ds):
                    model.eval()
                    head.eval()
                    val_loss_sum = 0.0
                    val_count = 0
                    batch_size = cfg.get("batch_size", 4)
                    with torch.no_grad():
                        for val_mel, val_text in val_dl:
                            # Skip batches that don't match training batch size (CUDA graphs require fixed batch sizes)
                            if use_compile and val_mel.size(0) != batch_size:
                                continue
                            val_mel = val_mel.to(device)
                            val_tgt, val_tgt_lens = encode_text_batch(val_text)
                            val_tgt = val_tgt.to(device)
                            val_tgt_lens = val_tgt_lens.to(device)
                            if use_amp:
                                with autocast(device_type='cuda'):
                                    val_x = model(val_mel)
                                    val_logit = head(val_x)
                                    val_log_prob = val_logit.log_softmax(-1).transpose(0,1)
                                    val_inp_lens = torch.full((val_log_prob.size(1),), val_log_prob.size(0), dtype=torch.long, device=val_log_prob.device)
                                    val_loss = ctc_loss(val_log_prob, val_tgt, val_inp_lens, val_tgt_lens)
                            else:
                                val_x = model(val_mel)
                                val_logit = head(val_x)
                                val_log_prob = val_logit.log_softmax(-1).transpose(0,1)
                                val_inp_lens = torch.full((val_log_prob.size(1),), val_log_prob.size(0), dtype=torch.long, device=val_log_prob.device)
                                val_loss = ctc_loss(val_log_prob, val_tgt, val_inp_lens, val_tgt_lens)
                            
                            # Validate validation loss
                            try:
                                validate_loss(val_loss, min_loss=-1e6, max_loss=1e6)
                                val_loss_sum += float(val_loss.detach())
                                val_count += 1
                                # Free validation tensors
                                del val_x, val_logit, val_log_prob, val_loss
                            except RuntimeError as e:
                                logger.warning(f"Step {step}: Invalid validation loss: {e}")
                                # Continue with other validation batches
                            if val_count >= 10:  # Limit validation batches
                                break
                    
                    avg_val_loss = val_loss_sum / val_count
                    logger.val_step(step, avg_val_loss, epoch)
                    
                    model.train()
                    head.train()
            
            if step >= cfg["max_steps"]:
                final_path = os.path.join(cfg["save_dir"], "audio_enc.pt")
                checkpoint_data = {
                    "enc": model.state_dict(),
                    "head": head.state_dict(),
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
            model.eval()
            head.eval()
            val_loss_sum = 0.0
            val_count = 0
            batch_size = cfg.get("batch_size", 4)
            with torch.no_grad():
                for val_mel, val_text in val_dl:
                    # Skip batches that don't match training batch size (CUDA graphs require fixed batch sizes)
                    if use_compile and val_mel.size(0) != batch_size:
                        continue
                    val_mel = val_mel.to(device)
                    val_tgt, val_tgt_lens = encode_text_batch(val_text)
                    val_tgt = val_tgt.to(device)
                    val_tgt_lens = val_tgt_lens.to(device)
                    if use_amp:
                        with autocast(device_type='cuda'):
                            val_x = model(val_mel)
                            val_logit = head(val_x)
                            val_log_prob = val_logit.log_softmax(-1).transpose(0,1)
                            val_inp_lens = torch.full((val_log_prob.size(1),), val_log_prob.size(0), dtype=torch.long, device=val_log_prob.device)
                            val_loss = ctc_loss(val_log_prob, val_tgt, val_inp_lens, val_tgt_lens)
                    else:
                        val_x = model(val_mel)
                        val_logit = head(val_x)
                        val_log_prob = val_logit.log_softmax(-1).transpose(0,1)
                        val_inp_lens = torch.full((val_log_prob.size(1),), val_log_prob.size(0), dtype=torch.long, device=val_log_prob.device)
                        val_loss = ctc_loss(val_log_prob, val_tgt, val_inp_lens, val_tgt_lens)
                    
                    # Validate validation loss
                    try:
                        validate_loss(val_loss, min_loss=-1e6, max_loss=1e6)
                        val_loss_sum += float(val_loss.detach())
                        val_count += 1
                        # Free validation tensors
                        del val_x, val_logit, val_log_prob, val_loss
                    except RuntimeError as e:
                        logger.warning(f"Epoch {epoch}: Invalid validation loss: {e}")
                        # Continue with other validation batches
                        del val_x, val_logit, val_log_prob, val_loss
        
            avg_val_loss = val_loss_sum / max(val_count, 1)
            logger.epoch_end(epoch, train_loss=None, val_loss=avg_val_loss)
            
            model.train()
            head.train()
        
        # Save at end of epoch (checkpoint for resuming)
        final_path = os.path.join(cfg["save_dir"], "audio_enc.pt")
        checkpoint_data = {
            "enc": model.state_dict(),
            "head": head.state_dict(),
            "optimizer": opt.state_dict(),
            "scheduler": scheduler.state_dict(),
            "step": step
        }
        if scaler is not None:
            checkpoint_data["scaler"] = scaler.state_dict()
        torch.save(checkpoint_data, final_path)
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
