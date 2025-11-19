
import argparse, json, os, csv, torch, pickle
from torch import nn
from torch.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader
import torchaudio
from omni.codec import RVQ
from omni.talker import TalkerTiny
from omni.training_utils import set_seed, get_lr_scheduler, clip_gradients, SimpleLogger, validate_loss, check_gradient_explosion, cleanup_old_checkpoints
from tqdm import tqdm

def collate_mel_fn(batch):
    """Collate function for variable-length mel spectrograms"""
    # Pad sequences to same length
    max_len = max(m.shape[0] for m in batch)
    n_mels = batch[0].shape[1]
    padded = []
    for m in batch:
        pad_len = max_len - m.shape[0]
        if pad_len > 0:
            m = torch.cat([m, torch.zeros(pad_len, n_mels)], dim=0)
        padded.append(m)
    return torch.stack(padded)

class TTSDataset(Dataset):
    def __init__(self, csv_path, sr=16000, n_mels=128, frame_ms=80, cfg=None):
        self.csv_path = csv_path
        self.sr = sr
        # Fix: win_length must be <= n_fft, and hop_length should be reasonable
        hop_length = int(sr * frame_ms / 1000)  # e.g., 16000 * 0.08 = 1280 samples
        win_length = min(1024, hop_length * 4)  # Ensure win_length <= n_fft
        self.melspec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sr, 
            n_fft=1024, 
            hop_length=hop_length, 
            win_length=win_length, 
            n_mels=n_mels
        )
        self.frame = int(sr*0.08)
        # Build or load cached row offset index (speeds up resuming)
        offset_cache_path = f"{csv_path}.row_offsets.pkl"
        file_mtime = os.path.getmtime(csv_path)
        cache_valid = False
        
        if os.path.exists(offset_cache_path):
            try:
                with open(offset_cache_path, 'rb') as cache_file:
                    cached_data = pickle.load(cache_file)
                    cached_mtime = cached_data.get('mtime', 0)
                    cached_offsets = cached_data.get('offsets', [])
                    cached_fieldnames = cached_data.get('fieldnames', None)
                    
                    # Check if cache is valid (file hasn't changed)
                    if abs(cached_mtime - file_mtime) < 1.0:  # Within 1 second tolerance
                        self.row_offsets = cached_offsets
                        self.fieldnames = cached_fieldnames
                        cache_valid = True
            except Exception:
                pass  # Cache invalid, will rebuild
        
        if not cache_valid:
            # Build row offset index (stores file positions, not content)
            self.row_offsets = []
            self.fieldnames = None
            with open(csv_path, 'rb') as f:
                # Read header
                header_line = f.readline()
                if not header_line:
                    raise ValueError("CSV file is empty")
                self.fieldnames = header_line.decode('utf-8').strip().split(',')
                # Build offset index for data rows
                offset = f.tell()
                while True:
                    line_start = offset
                    line_bytes = f.readline()
                    if not line_bytes:
                        break
                    try:
                        decoded = line_bytes.decode('utf-8').strip()
                        if decoded:
                            self.row_offsets.append(line_start)
                    except UnicodeDecodeError:
                        pass
                    offset += len(line_bytes)
            
            # Cache the offset index for future use
            try:
                with open(offset_cache_path, 'wb') as cache_file:
                    pickle.dump({'mtime': file_mtime, 'offsets': self.row_offsets, 'fieldnames': self.fieldnames}, cache_file)
            except Exception:
                pass  # Cache write failed, but continue anyway

    def __len__(self): return len(self.row_offsets)
    def __getitem__(self, i):
        # Read only the specific row using file offset
        with open(self.csv_path, 'rb') as f:
            f.seek(self.row_offsets[i])
            line_bytes = f.readline()
            line = line_bytes.decode('utf-8').strip()
        # Parse CSV row properly (handles quoted fields)
        import io
        reader = csv.DictReader(io.StringIO(line), fieldnames=self.fieldnames)
        row = next(reader)
        text, path = row["text"], row["wav"]
        wav, sr = torchaudio.load(path); assert sr==self.sr
        mel = self.melspec(wav)[0].T  # (T, n_mels)
        return mel

def main(cfg):
    # Set random seed for reproducibility
    seed = cfg.get("seed", 42)
    set_seed(seed)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(cfg["save_dir"], exist_ok=True)
    sr = cfg.get("sample_rate", 16000)
    n_mels = cfg.get("n_mels", 128)
    frame_ms = cfg.get("frame_ms", 80)
    ds = TTSDataset(cfg["tts_csv"], sr=sr, n_mels=n_mels, frame_ms=frame_ms, cfg=cfg)
    # Use module-level collate function for Windows multiprocessing compatibility
    dl = DataLoader(ds, batch_size=cfg.get("batch_size", 4), shuffle=True, num_workers=cfg.get("num_workers", 2), drop_last=cfg.get("drop_last", True), collate_fn=collate_mel_fn)
    
    # torch.compile() support (optional, PyTorch 2.0+)
    use_compile = cfg.get("use_compile", False)
    
    rvq = RVQ(cfg["codebooks"], cfg["codebook_size"], d=64, compile_model=use_compile).to(device)
    talker = TalkerTiny(
        cfg["d_model"], 
        cfg["n_layers"], 
        cfg["n_heads"], 
        cfg["d_ff"], 
        cfg["codebooks"], 
        cfg["codebook_size"], 
        cfg["dropout"],
        use_gqa=cfg.get("use_gqa", False),
        use_swiglu=cfg.get("use_swiglu", True),
        rope_theta=cfg.get("rope_theta", 10000.0),
        compile_model=use_compile
    ).to(device)
    opt = torch.optim.AdamW(list(rvq.parameters())+list(talker.parameters()), lr=cfg["lr"], weight_decay=cfg["wd"])
    loss_fn = nn.CrossEntropyLoss()
    
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
    train_dl = DataLoader(train_ds, batch_size=cfg.get("batch_size", 4), shuffle=True, num_workers=cfg.get("num_workers", 2), drop_last=cfg.get("drop_last", True), collate_fn=collate_mel_fn)
    val_dl = DataLoader(val_ds, batch_size=cfg.get("batch_size", 4), shuffle=False, num_workers=cfg.get("num_workers", 2), drop_last=False, collate_fn=collate_mel_fn)
    
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
    resume_from = None
    if os.path.exists(cfg["save_dir"]):
        checkpoint_files = [f for f in os.listdir(cfg["save_dir"]) if f.startswith("talker_step_") and f.endswith(".pt")]
        if checkpoint_files:
            # Extract step numbers and find latest
            step_numbers = []
            for f in checkpoint_files:
                try:
                    step_num = int(f.replace("talker_step_", "").replace(".pt", ""))
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
                if isinstance(checkpoint, dict) and "rvq" in checkpoint:
                    rvq.load_state_dict(checkpoint["rvq"])
                    if "talker" in checkpoint:
                        talker.load_state_dict(checkpoint["talker"])
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
                    if "rvq" in checkpoint:
                        rvq.load_state_dict(checkpoint["rvq"])
                    if "talker" in checkpoint:
                        talker.load_state_dict(checkpoint["talker"])
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
        for batch_idx, mel in enumerate(tqdm(train_dl, desc=f"epoch{epoch}")):
            # Skip batches if resuming mid-epoch
            if epoch == start_epoch and initial_step > 0:
                current_batch_step = epoch * steps_per_epoch + batch_idx
                if current_batch_step < initial_step:
                    continue
            mel = mel.to(device)  # (B,T,128)
            
            # Mark step begin for CUDAGraphs optimization
            if device == "cuda":
                torch.compiler.cudagraph_mark_step_begin()
            
            # Forward pass with mixed precision
            if use_amp:
                with autocast(device_type='cuda'):
                    # Batch encode all frames at once (optimized)
                    idxs = rvq.encode(mel)  # (B,T,2) - encodes all frames in batch
                    # AR training: predict current codes from previous codes
                    prev = torch.roll(idxs, 1, dims=1); prev[:,0,:]=0
                    base_logit, res_logit = talker(prev)
                    loss = loss_fn(base_logit.reshape(-1, base_logit.size(-1)), idxs[:,:,0].reshape(-1)) + \
                           loss_fn(res_logit.reshape(-1, res_logit.size(-1)),  idxs[:,:,1].reshape(-1))
                    # Free intermediate tensors
                    del base_logit, res_logit, prev
            else:
                # Batch encode all frames at once (optimized)
                idxs = rvq.encode(mel)  # (B,T,2) - encodes all frames in batch
                # AR training: predict current codes from previous codes
                prev = torch.roll(idxs, 1, dims=1); prev[:,0,:]=0
                base_logit, res_logit = talker(prev)
                loss = loss_fn(base_logit.reshape(-1, base_logit.size(-1)), idxs[:,:,0].reshape(-1)) + \
                       loss_fn(res_logit.reshape(-1, res_logit.size(-1)),  idxs[:,:,1].reshape(-1))
                # Free intermediate tensors
                del base_logit, res_logit, prev
            
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
                    grad_norm_rvq, is_exploded_rvq = check_gradient_explosion(rvq, max_grad_norm=100.0, raise_on_error=False)
                    grad_norm_talker, is_exploded_talker = check_gradient_explosion(talker, max_grad_norm=100.0, raise_on_error=False)
                    if is_exploded_rvq or is_exploded_talker:
                        logger.error(f"Step {step}: Gradient explosion detected (rvq={grad_norm_rvq:.2f}, talker={grad_norm_talker:.2f}). Skipping this batch.")
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
                    clip_gradients(rvq, max_grad_norm)
                    clip_gradients(talker, max_grad_norm)
                    scaler.step(opt)
                    scaler.update()
                else:
                    clip_gradients(rvq, max_grad_norm)
                    clip_gradients(talker, max_grad_norm)
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
                checkpoint_path = os.path.join(cfg["save_dir"], f"talker_step_{step}.pt")
                checkpoint_data = {
                    "rvq": rvq.state_dict(),
                    "talker": talker.state_dict(),
                    "optimizer": opt.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "step": step
                }
                if scaler is not None:
                    checkpoint_data["scaler"] = scaler.state_dict()
                torch.save(checkpoint_data, checkpoint_path)
                logger.checkpoint(step, checkpoint_path)
                # Clean up old checkpoints (keep only last one)
                cleanup_old_checkpoints(cfg["save_dir"], "talker_step_", keep_last_n=1)
            
            # Validation
            if step % val_freq == 0 and step > 0:
                rvq.eval()
                talker.eval()
                val_loss_sum = 0.0
                val_count = 0
                with torch.no_grad():
                    for val_mel in val_dl:
                        val_mel = val_mel.to(device)
                        if use_amp:
                            with autocast():
                                # Batch encode all frames at once (optimized)
                                val_idxs = rvq.encode(val_mel)  # (B,T,2)
                                val_prev = torch.roll(val_idxs, 1, dims=1); val_prev[:,0,:]=0
                                val_base_logit, val_res_logit = talker(val_prev)
                                val_loss = loss_fn(val_base_logit.reshape(-1, val_base_logit.size(-1)), val_idxs[:,:,0].reshape(-1)) + \
                                           loss_fn(val_res_logit.reshape(-1, val_res_logit.size(-1)), val_idxs[:,:,1].reshape(-1))
                        else:
                            # Batch encode all frames at once (optimized)
                            val_idxs = rvq.encode(val_mel)  # (B,T,2)
                            val_prev = torch.roll(val_idxs, 1, dims=1); val_prev[:,0,:]=0
                            val_base_logit, val_res_logit = talker(val_prev)
                            val_loss = loss_fn(val_base_logit.reshape(-1, val_base_logit.size(-1)), val_idxs[:,:,0].reshape(-1)) + \
                                       loss_fn(val_res_logit.reshape(-1, val_res_logit.size(-1)), val_idxs[:,:,1].reshape(-1))
                        
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
            
            if step >= cfg["max_steps"]:
                final_path = os.path.join(cfg["save_dir"], "talker.pt")
                checkpoint_data = {
                    "rvq": rvq.state_dict(),
                    "talker": talker.state_dict(),
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
        rvq.eval()
        talker.eval()
        val_loss_sum = 0.0
        val_count = 0
        with torch.no_grad():
            for val_mel in val_dl:
                val_mel = val_mel.to(device)
                if use_amp:
                    with autocast():
                        # Batch encode all frames at once (optimized)
                        val_idxs = rvq.encode(val_mel)  # (B,T,2)
                        val_prev = torch.roll(val_idxs, 1, dims=1); val_prev[:,0,:]=0
                        val_base_logit, val_res_logit = talker(val_prev)
                        val_loss = loss_fn(val_base_logit.reshape(-1, val_base_logit.size(-1)), val_idxs[:,:,0].reshape(-1)) + \
                                   loss_fn(val_res_logit.reshape(-1, val_res_logit.size(-1)), val_idxs[:,:,1].reshape(-1))
                else:
                    # Batch encode all frames at once (optimized)
                    val_idxs = rvq.encode(val_mel)  # (B,T,2)
                    val_prev = torch.roll(val_idxs, 1, dims=1); val_prev[:,0,:]=0
                    val_base_logit, val_res_logit = talker(val_prev)
                    val_loss = loss_fn(val_base_logit.reshape(-1, val_base_logit.size(-1)), val_idxs[:,:,0].reshape(-1)) + \
                               loss_fn(val_res_logit.reshape(-1, val_res_logit.size(-1)), val_idxs[:,:,1].reshape(-1))
                
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
        
        # Save at end of epoch if max_steps not reached
        if step < cfg["max_steps"]:
            final_path = os.path.join(cfg["save_dir"], "talker.pt")
            checkpoint_data = {
                "rvq": rvq.state_dict(),
                "talker": talker.state_dict(),
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
