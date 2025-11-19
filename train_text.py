
import argparse, json, torch, os
import tempfile
from torch import nn
from torch.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader
from omni.thinker import ThinkerLM
from omni.tokenizer import BPETokenizer
from omni.training_utils import set_seed, get_lr_scheduler, clip_gradients, SimpleLogger, validate_loss, check_gradient_explosion, reload_from_last_checkpoint, cleanup_old_checkpoints
from tqdm import tqdm

class TextDataset(Dataset):
    def __init__(self, path, tokenizer, ctx):
        self.path = path
        self.tok = tokenizer; self.ctx = ctx
        # Build line offset index (stores file positions, not content)
        # Read in binary mode and manually track offsets
        self.line_offsets = []
        with open(path, 'rb') as f:
            offset = 0
            while True:
                line_start = offset
                line_bytes = f.readline()
                if not line_bytes:
                    break
                # Decode to check if line is non-empty
                try:
                    decoded = line_bytes.decode('utf-8')
                    if decoded.strip():
                        self.line_offsets.append(line_start)
                except UnicodeDecodeError:
                    # Skip invalid UTF-8 lines
                    pass
                # Manually track offset by adding line length
                offset += len(line_bytes)
    def __len__(self): return len(self.line_offsets)
    def __getitem__(self, i):
        # Read only the specific line using file offset
        with open(self.path, 'rb') as f:
            f.seek(self.line_offsets[i])
            line_bytes = f.readline()
            text = line_bytes.decode('utf-8').strip()
        ids = self.tok.encode(text)[:self.ctx-1]
        ids = [1] + ids  # BOS=1 (SentencePiece default)
        pad = [0] * (self.ctx - len(ids))
        x = torch.tensor(ids + pad, dtype=torch.long)
        y = x.clone(); y[:-1]=x[1:]; y[-1]=0
        return x, y

def stream_text_file(text_path: str, chunk_size_mb: int = 100) -> str:
    """
    Stream entire text file in chunks to a temporary file (memory efficient).
    Processes the entire corpus without loading it all into memory.
    Returns path to temporary file with all data.
    """
    temp_file = tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', delete=False, suffix='.txt')
    temp_path = temp_file.name
    temp_file.close()
    
    lines_read = 0
    
    try:
        with open(text_path, 'rb') as infile, open(temp_path, 'w', encoding='utf-8') as outfile:
            # Read file in chunks to avoid loading entire file into memory
            while True:
                line_bytes = infile.readline()
                if not line_bytes:
                    break
                
                try:
                    line = line_bytes.decode('utf-8')
                    if line.strip():
                        outfile.write(line)
                        lines_read += 1
                        
                        # Progress indicator for large files
                        if lines_read % 100000 == 0:
                            print(f"  Streaming corpus: {lines_read:,} lines processed...")
                except UnicodeDecodeError:
                    pass  # Skip invalid UTF-8 lines
    except Exception as e:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise
    
    print(f"  Streamed entire corpus: {lines_read:,} lines")
    return temp_path

def main(cfg):
    # Set random seed for reproducibility
    seed = cfg.get("seed", 42)
    set_seed(seed)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(cfg["save_dir"], exist_ok=True)
    spm_model = os.path.join(cfg["save_dir"], "tokenizer.model")
    if not os.path.exists(spm_model):
        print(f"Creating tokenizer from {cfg['train_text']} (streaming entire corpus in chunks)...")
        # Stream entire corpus in chunks instead of loading entire file
        temp_streamed = None
        try:
            temp_streamed = stream_text_file(cfg["train_text"], chunk_size_mb=100)
            print(f"  Training tokenizer on entire corpus...")
            BPETokenizer.train_new(temp_streamed, spm_model, vocab_size=cfg["vocab_size"])
            print(f"✓ Tokenizer created: {spm_model}")
        finally:
            if temp_streamed and os.path.exists(temp_streamed):
                os.remove(temp_streamed)
    tok = BPETokenizer(spm_model)
    ds = TextDataset(cfg["train_text"], tok, cfg["ctx_len"])
    dl = DataLoader(ds, batch_size=cfg.get("batch_size", 8), shuffle=True, num_workers=cfg.get("num_workers", 2), drop_last=cfg.get("drop_last", True))
    
    # torch.compile() support (optional, PyTorch 2.0+)
    use_compile = cfg.get("use_compile", False)
    
    model = ThinkerLM(
        cfg["vocab_size"], 
        cfg["n_layers"], 
        cfg["d_model"], 
        cfg["n_heads"], 
        cfg["d_ff"], 
        cfg["dropout"], 
        cfg["rope_theta"], 
        cfg["ctx_len"],
        use_gqa=cfg.get("use_gqa", False),
        use_swiglu=cfg.get("use_swiglu", True),
        use_moe=cfg.get("use_moe", False),
        num_experts=cfg.get("num_experts", 8),
        num_experts_per_tok=cfg.get("num_experts_per_tok", 2),
        compile_model=use_compile
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    loss_fn = nn.CrossEntropyLoss(ignore_index=0)
    
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

    # Split dataset for validation
    val_split = cfg.get("val_split", 0.1)  # 10% for validation
    total_size = len(ds)
    val_size = int(total_size * val_split)
    train_size = total_size - val_size
    train_ds, val_ds = torch.utils.data.random_split(ds, [train_size, val_size], generator=torch.Generator().manual_seed(seed))
    train_dl = DataLoader(train_ds, batch_size=cfg.get("batch_size", 8), shuffle=True, num_workers=cfg.get("num_workers", 2), drop_last=cfg.get("drop_last", True))
    val_dl = DataLoader(val_ds, batch_size=cfg.get("batch_size", 8), shuffle=False, num_workers=cfg.get("num_workers", 2), drop_last=False)
    
    # Initialize logger
    logger = SimpleLogger("Thinker")
    
    step=0
    model.train()
    max_epochs = cfg.get("max_epochs", 9999)
    print_freq = cfg.get("print_freq", 1)
    checkpoint_freq = cfg.get("checkpoint_freq", 500)  # Save checkpoint every N steps
    val_freq = cfg.get("val_freq", 200)  # Validate every N steps
    
    # Resume from checkpoint if available
    resume_from = None
    checkpoint_files = [f for f in os.listdir(cfg["save_dir"]) if f.startswith("thinker_step_") and f.endswith(".pt")]
    if checkpoint_files:
        # Extract step numbers and find latest
        step_numbers = []
        for f in checkpoint_files:
            try:
                step_num = int(f.replace("thinker_step_", "").replace(".pt", ""))
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
            if isinstance(checkpoint, dict) and "model" in checkpoint:
                model.load_state_dict(checkpoint["model"])
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
                # Legacy checkpoint format (just model weights)
                model.load_state_dict(checkpoint)
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
        logger.epoch_start(epoch)
        for batch_idx, (x,y) in enumerate(tqdm(train_dl, desc=f"epoch{epoch}")):
            # Skip batches if resuming mid-epoch
            if epoch == start_epoch and initial_step > 0:
                current_batch_step = epoch * steps_per_epoch + batch_idx
                if current_batch_step < initial_step:
                    continue
            x,y = x.to(device), y.to(device)
            
            # Mark step begin for CUDAGraphs optimization
            if device == "cuda":
                torch.compiler.cudagraph_mark_step_begin()
            
            # Forward pass with mixed precision
            try:
                if use_amp:
                    with autocast(device_type='cuda'):
                        logits = model(x)  # (B,T,V)
                        loss = loss_fn(logits.view(-1, logits.size(-1)), y.view(-1))
                else:
                    logits = model(x)  # (B,T,V)
                    loss = loss_fn(logits.view(-1, logits.size(-1)), y.view(-1))
                
                # Free logits after loss computation (loss keeps its own graph)
                del logits
            except (RuntimeError, Exception) as e:
                error_msg = str(e)
                error_type = type(e).__name__
                # Check for Triton/Inductor compilation errors
                if ("Triton compilation failed" in error_msg or 
                    "failed to translate module to LLVM IR" in error_msg or 
                    "InductorError" in error_type or
                    "could not find LLVM intrinsic" in error_msg):
                    logger.error(f"Step {step}: Triton compilation error detected")
                    logger.error(f"Error type: {error_type}")
                    logger.error("This is often caused by GPU compatibility issues (e.g., RTX 50 series with compute capability 12.0).")
                    logger.error("Solution: Set 'use_compile': false in your training config to disable torch.compile()")
                    raise  # Re-raise to stop training
                if "NaN detected in attention probabilities after softmax" in error_msg or "Numerical instability" in error_msg:
                    logger.error(f"Step {step}: {e}")
                    logger.error("Reloading from last checkpoint...")
                    # Reload from last checkpoint
                    reloaded_step = reload_from_last_checkpoint(
                        cfg["save_dir"], "thinker_step_", device, logger, model, opt, scheduler, scaler
                    )
                    if reloaded_step > 0:
                        step = reloaded_step
                        # Recalculate start_epoch and initial_step for resuming
                        start_epoch = step // steps_per_epoch
                        initial_step = step
                        logger.info(f"Resuming from step {step}, epoch {start_epoch}")
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
                    grad_norm, is_exploded = check_gradient_explosion(model, max_grad_norm=100.0, raise_on_error=False)
                    if is_exploded:
                        logger.error(f"Step {step}: Gradient explosion detected (grad_norm={grad_norm:.2f}). Skipping this batch.")
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
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    scaler.step(opt)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    opt.step()
                scheduler.step()
                
                # Check if weights became NaN after optimizer step
                has_nan, has_inf, nan_count, inf_count = model.check_weights_stability()
                if has_nan or has_inf:
                    logger.error(f"Step {step}: Model weights corrupted after optimizer step (NaN={nan_count}, Inf={inf_count})")
                    logger.error("This indicates numerical instability. Consider:")
                    logger.error("  - Reducing learning rate")
                    logger.error("  - Using gradient clipping")
                    logger.error("  - Disabling mixed precision training")
                    logger.error("  - Checking for gradient explosion")
                    # Try to recover by loading from last checkpoint if available
                    checkpoint_files = [f for f in os.listdir(cfg["save_dir"]) if f.startswith("thinker_step_") and f.endswith(".pt")]
                    if checkpoint_files:
                        step_numbers = []
                        for f in checkpoint_files:
                            try:
                                step_num = int(f.replace("thinker_step_", "").replace(".pt", ""))
                                step_numbers.append((step_num, f))
                            except:
                                continue
                        if step_numbers:
                            step_numbers.sort(key=lambda x: x[0], reverse=True)
                            last_checkpoint = os.path.join(cfg["save_dir"], step_numbers[0][1])
                            logger.error(f"Attempting to recover from checkpoint: {last_checkpoint}")
                            checkpoint = torch.load(last_checkpoint, map_location=device)
                            if isinstance(checkpoint, dict) and "model" in checkpoint:
                                model.load_state_dict(checkpoint["model"])
                                if "optimizer" in checkpoint:
                                    opt.load_state_dict(checkpoint["optimizer"])
                                if "scheduler" in checkpoint:
                                    scheduler.load_state_dict(checkpoint["scheduler"])
                                if "scaler" in checkpoint and scaler is not None:
                                    scaler.load_state_dict(checkpoint["scaler"])
                                logger.info("Recovered from checkpoint. Continuing training...")
                            else:
                                model.load_state_dict(checkpoint)
                                logger.info("Recovered model weights from checkpoint. Continuing training...")
                    opt.zero_grad()
                    continue
                
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
                # Calculate perplexity for evaluation
                perplexity = torch.exp(unscaled_loss).item() if unscaled_loss.item() < 10 else float('inf')
                logger.train_step(step, float(unscaled_loss), current_lr, epoch)
                if step % (print_freq * 10) == 0:  # Log perplexity less frequently
                    logger.info(f"Perplexity: {perplexity:.2f}")
            
            # Periodic checkpointing
            if step % checkpoint_freq == 0 and step > 0:
                checkpoint_path = os.path.join(cfg["save_dir"], f"thinker_step_{step}.pt")
                checkpoint_data = {
                    "model": model.state_dict(),
                    "optimizer": opt.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "step": step
                }
                if scaler is not None:
                    checkpoint_data["scaler"] = scaler.state_dict()
                torch.save(checkpoint_data, checkpoint_path)
                logger.checkpoint(step, checkpoint_path)
                # Clean up old checkpoints (keep only last one)
                cleanup_old_checkpoints(cfg["save_dir"], "thinker_step_", keep_last_n=1)
            
            # Validation
            if step % val_freq == 0 and step > 0:
                # Check model weights for NaN/Inf before validation
                has_nan, has_inf, nan_count, inf_count = model.check_weights_stability()
                if has_nan or has_inf:
                    logger.error(f"Step {step}: Model weights corrupted (NaN={nan_count}, Inf={inf_count}). Skipping validation.")
                    logger.error("This indicates numerical instability during training. Consider:")
                    logger.error("  - Reducing learning rate")
                    logger.error("  - Using gradient clipping")
                    logger.error("  - Disabling mixed precision training")
                    logger.error("  - Checking for gradient explosion")
                    # Continue training but skip validation
                    model.train()
                    continue
                
                model.eval()
                val_loss_sum = 0.0
                val_count = 0
                with torch.no_grad():
                    for val_x, val_y in val_dl:
                        val_x, val_y = val_x.to(device), val_y.to(device)
                        try:
                            if use_amp:
                                with autocast(device_type='cuda'):
                                    val_logits = model(val_x)
                                    val_loss = loss_fn(val_logits.view(-1, val_logits.size(-1)), val_y.view(-1))
                            else:
                                val_logits = model(val_x)
                                val_loss = loss_fn(val_logits.view(-1, val_logits.size(-1)), val_y.view(-1))
                            
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
                                    cfg["save_dir"], "thinker_step_", device, logger, model, opt, scheduler, scaler
                                )
                                if reloaded_step > 0:
                                    step = reloaded_step
                                    start_epoch = step // steps_per_epoch
                                    initial_step = step
                                    logger.info(f"Resuming from step {step}, epoch {start_epoch}")
                                model.train()
                                break
                            else:
                                logger.warning(f"Step {step}: Validation error: {e}")
                                # Continue with other validation batches
                                break  # Break on NaN/Inf to avoid repeated errors
                        
                        if val_count >= 20:  # Limit validation batches
                            break
                
                avg_val_loss = val_loss_sum / val_count
                logger.val_step(step, avg_val_loss, epoch)
                
                model.train()
            
            if step >= max_steps:
                final_path = os.path.join(cfg["save_dir"], "thinker.pt")
                checkpoint_data = {
                    "model": model.state_dict(),
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
        # Check model weights for NaN/Inf before validation
        has_nan, has_inf, nan_count, inf_count = model.check_weights_stability()
        if has_nan or has_inf:
            logger.error(f"Epoch {epoch}: Model weights corrupted (NaN={nan_count}, Inf={inf_count}). Skipping validation.")
            logger.error("This indicates numerical instability during training. Consider:")
            logger.error("  - Reducing learning rate")
            logger.error("  - Using gradient clipping")
            logger.error("  - Disabling mixed precision training")
            logger.error("  - Checking for gradient explosion")
            model.train()
            avg_val_loss = float('inf')
        else:
            model.eval()
            val_loss_sum = 0.0
            val_count = 0
            with torch.no_grad():
                for val_x, val_y in val_dl:
                    val_x, val_y = val_x.to(device), val_y.to(device)
                    try:
                        if use_amp:
                            with autocast(device_type='cuda'):
                                val_logits = model(val_x)
                                val_loss = loss_fn(val_logits.view(-1, val_logits.size(-1)), val_y.view(-1))
                        else:
                            val_logits = model(val_x)
                            val_loss = loss_fn(val_logits.view(-1, val_logits.size(-1)), val_y.view(-1))
                        
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
                                cfg["save_dir"], "thinker_step_", device, logger, model, opt, scheduler, scaler
                            )
                            if reloaded_step > 0:
                                step = reloaded_step
                                start_epoch = step // steps_per_epoch
                                initial_step = step
                                logger.info(f"Resuming from step {step}, epoch {start_epoch}")
                            model.train()
                            break
                        else:
                            logger.warning(f"Epoch {epoch}: Validation error: {e}")
                            # Break on NaN/Inf to avoid repeated errors
                            break
            
            avg_val_loss = val_loss_sum / max(val_count, 1)
        
        logger.epoch_end(epoch, train_loss=None, val_loss=avg_val_loss)
        model.train()
        
        # Save at end of epoch if max_steps not reached
        if step < cfg["max_steps"]:
            final_path = os.path.join(cfg["save_dir"], "thinker.pt")
            checkpoint_data = {
                "model": model.state_dict(),
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
