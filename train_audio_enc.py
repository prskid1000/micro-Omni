
import argparse, json, os, torch, torchaudio, csv
from torch import nn
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader
from omni.audio_encoder import AudioEncoderTiny
from omni.training_utils import set_seed, get_lr_scheduler, clip_gradients, SimpleLogger, validate_loss, check_gradient_explosion
from tqdm import tqdm

def collate_fn(batch):
    mels, texts = zip(*batch)
    max_len = max(m.shape[0] for m in mels)
    n_mels = mels[0].shape[1]
    padded_mels = []
    for m in mels:
        pad_len = max_len - m.shape[0]
        if pad_len > 0:
            m = torch.cat([m, torch.zeros(pad_len, n_mels)], dim=0)
        padded_mels.append(m)
    return torch.stack(padded_mels), list(texts)

class ASRDataset(Dataset):
    def __init__(self, csv_path, sr=16000, n_mels=128, cfg=None):
        self.rows = []
        with open(csv_path, 'r', encoding='utf-8') as f:
            rd = csv.DictReader(f)
            for r in rd: self.rows.append(r)
        self.sr = sr
        self.melspec = torchaudio.transforms.MelSpectrogram(sample_rate=sr, n_fft=1024, hop_length=160, win_length=400, n_mels=n_mels)

    def __len__(self): return len(self.rows)
    def __getitem__(self, i):
        path, text = self.rows[i]["wav"], self.rows[i]["text"]
        wav, sr = torchaudio.load(path); assert sr==self.sr
        mel = self.melspec(wav)[0].T  # (T, n_mels)
        return mel, text

def main(cfg):
    # Set random seed for reproducibility
    seed = cfg.get("seed", 42)
    set_seed(seed)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(cfg["save_dir"], exist_ok=True)
    sr = cfg.get("sample_rate", 16000)
    n_mels = cfg.get("mel_bins", 128)
    ds = ASRDataset(cfg["train_csv"], sr=sr, n_mels=n_mels, cfg=cfg)
    dl = DataLoader(ds, batch_size=cfg.get("batch_size", 4), shuffle=True, num_workers=cfg.get("num_workers", 2), drop_last=cfg.get("drop_last", True), collate_fn=collate_fn)
    downsample_factor = cfg.get("downsample_time", 8)  # 8x for 12.5 Hz (16000/160/8 = 12.5)
    model = AudioEncoderTiny(
        cfg["d_model"], 
        cfg["n_heads"], 
        cfg["d_ff"], 
        cfg["n_layers"], 
        cfg["dropout"],
        downsample_factor=downsample_factor
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
    scaler = GradScaler() if use_amp else None
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
    train_dl = DataLoader(train_ds, batch_size=cfg.get("batch_size", 4), shuffle=True, num_workers=cfg.get("num_workers", 2), drop_last=cfg.get("drop_last", True), collate_fn=collate_fn)
    val_dl = DataLoader(val_ds, batch_size=cfg.get("batch_size", 4), shuffle=False, num_workers=cfg.get("num_workers", 2), drop_last=False, collate_fn=collate_fn)
    
    # Initialize logger
    logger = SimpleLogger("AudioEncoder")
    
    step=0
    model.train()
    head.train()
    max_epochs = cfg.get("max_epochs", 9999)
    print_freq = cfg.get("print_freq", 100)
    checkpoint_freq = cfg.get("checkpoint_freq", 500)  # Save checkpoint every N steps
    val_freq = cfg.get("val_freq", 200)  # Validate every N steps
    best_val_loss = float('inf')
    
    # Resume from checkpoint if available
    resume_from = None
    if os.path.exists(cfg["save_dir"]):
        checkpoint_files = [f for f in os.listdir(cfg["save_dir"]) if f.startswith("audio_enc_step_") and f.endswith(".pt")]
        if checkpoint_files:
            # Extract step numbers and find latest
            step_numbers = []
            for f in checkpoint_files:
                try:
                    step_num = int(f.replace("audio_enc_step_", "").replace(".pt", ""))
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
                if isinstance(checkpoint, dict) and "enc" in checkpoint:
                    model.load_state_dict(checkpoint["enc"])
                    if "head" in checkpoint:
                        head.load_state_dict(checkpoint["head"])
                    if "optimizer" in checkpoint:
                        opt.load_state_dict(checkpoint["optimizer"])
                    if "scheduler" in checkpoint:
                        scheduler.load_state_dict(checkpoint["scheduler"])
                    if "scaler" in checkpoint and scaler is not None:
                        scaler.load_state_dict(checkpoint["scaler"])
                    if "step" in checkpoint:
                        step = checkpoint["step"]
                    if "best_val_loss" in checkpoint:
                        best_val_loss = checkpoint["best_val_loss"]
                    logger.info(f"Resumed from step {step}, best_val_loss={best_val_loss:.4f}")
                else:
                    # Legacy checkpoint format
                    if "enc" in checkpoint:
                        model.load_state_dict(checkpoint["enc"])
                    if "head" in checkpoint:
                        head.load_state_dict(checkpoint["head"])
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
        for batch_idx, (mel, text) in enumerate(tqdm(train_dl, desc=f"epoch{epoch}")):
            # Skip batches if resuming mid-epoch
            if epoch == start_epoch and initial_step > 0:
                current_batch_step = epoch * steps_per_epoch + batch_idx
                if current_batch_step < initial_step:
                    continue
            mel = mel.to(device)
            
            tgt, tgt_lens = encode_text_batch(text)
            tgt = tgt.to(device)
            tgt_lens = tgt_lens.to(device)
            
            # Forward pass with mixed precision
            if use_amp:
                with autocast():
                    x = model(mel)  # (B, T', d)
                    logit = head(x)  # (B,T',V)
                    log_prob = logit.log_softmax(-1).transpose(0,1)  # (T',B,V)
                    inp_lens = torch.full((log_prob.size(1),), log_prob.size(0), dtype=torch.long, device=log_prob.device)
                    loss = ctc_loss(log_prob, tgt, inp_lens, tgt_lens)
            else:
                x = model(mel)  # (B, T', d)
                logit = head(x)  # (B,T',V)
                log_prob = logit.log_softmax(-1).transpose(0,1)  # (T',B,V)
                inp_lens = torch.full((log_prob.size(1),), log_prob.size(0), dtype=torch.long, device=log_prob.device)
                loss = ctc_loss(log_prob, tgt, inp_lens, tgt_lens)
            
            # Scale loss for gradient accumulation
            loss = loss / accumulation_steps
            
            # Backward pass with gradient scaling
            if use_amp:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Gradient accumulation: only step optimizer every N steps
            if (step + 1) % accumulation_steps == 0:
                # Unscale before checking gradients
                if use_amp:
                    scaler.unscale_(opt)
                
                # Validate loss value (unscaled)
                unscaled_loss = loss * accumulation_steps
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
                    grad_norm_model, is_exploded_model = check_gradient_explosion(model, max_grad_norm=100.0, raise_on_error=False)
                    grad_norm_head, is_exploded_head = check_gradient_explosion(head, max_grad_norm=100.0, raise_on_error=False)
                    if is_exploded_model or is_exploded_head:
                        logger.error(f"Step {step}: Gradient explosion detected (model={grad_norm_model:.2f}, head={grad_norm_head:.2f}). Skipping this batch.")
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
                    clip_gradients(model, max_grad_norm)
                    clip_gradients(head, max_grad_norm)
                    scaler.step(opt)
                    scaler.update()
                else:
                    clip_gradients(model, max_grad_norm)
                    clip_gradients(head, max_grad_norm)
                    opt.step()
                scheduler.step()
                opt.zero_grad()  # Clear gradients after stepping
            else:
                # Not accumulation step - just validate loss
                unscaled_loss = loss * accumulation_steps
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
                unscaled_loss = loss * accumulation_steps
                logger.train_step(step, float(unscaled_loss.detach()), current_lr, epoch)
            
            # Periodic checkpointing
            if step % checkpoint_freq == 0 and step > 0:
                checkpoint_path = os.path.join(cfg["save_dir"], f"audio_enc_step_{step}.pt")
                checkpoint_data = {
                    "enc": model.state_dict(),
                    "head": head.state_dict(),
                    "optimizer": opt.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "step": step,
                    "best_val_loss": best_val_loss
                }
                if scaler is not None:
                    checkpoint_data["scaler"] = scaler.state_dict()
                torch.save(checkpoint_data, checkpoint_path)
                logger.checkpoint(step, checkpoint_path)
            
            # Validation
            if step % val_freq == 0 and step > 0:
                model.eval()
                head.eval()
                val_loss_sum = 0.0
                val_count = 0
                with torch.no_grad():
                    for val_mel, val_text in val_dl:
                        val_mel = val_mel.to(device)
                        val_tgt, val_tgt_lens = encode_text_batch(val_text)
                        val_tgt = val_tgt.to(device)
                        val_tgt_lens = val_tgt_lens.to(device)
                        if use_amp:
                            with autocast():
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
                        except RuntimeError as e:
                            logger.warning(f"Step {step}: Invalid validation loss: {e}")
                            # Continue with other validation batches
                        if val_count >= 10:  # Limit validation batches
                            break
                
                avg_val_loss = val_loss_sum / val_count
                logger.val_step(step, avg_val_loss, epoch)
                
                # Save best model
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    best_path = os.path.join(cfg["save_dir"], "audio_enc_best.pt")
                    checkpoint_data = {
                        "enc": model.state_dict(),
                        "head": head.state_dict(),
                        "optimizer": opt.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "step": step,
                        "best_val_loss": best_val_loss
                    }
                    if scaler is not None:
                        checkpoint_data["scaler"] = scaler.state_dict()
                    torch.save(checkpoint_data, best_path)
                    logger.checkpoint(step, best_path, is_best=True)
                
                model.train()
                head.train()
            
            if step >= cfg["max_steps"]:
                final_path = os.path.join(cfg["save_dir"], "audio_enc.pt")
                checkpoint_data = {
                    "enc": model.state_dict(),
                    "head": head.state_dict(),
                    "optimizer": opt.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "step": step,
                    "best_val_loss": best_val_loss
                }
                if scaler is not None:
                    checkpoint_data["scaler"] = scaler.state_dict()
                torch.save(checkpoint_data, final_path)
                logger.info(f"Final model saved to {cfg['save_dir']}")
                logger.training_end(step)
                return
        
        # Final validation at end of epoch
        model.eval()
        head.eval()
        val_loss_sum = 0.0
        val_count = 0
        with torch.no_grad():
            for val_mel, val_text in val_dl:
                val_mel = val_mel.to(device)
                val_tgt, val_tgt_lens = encode_text_batch(val_text)
                val_tgt = val_tgt.to(device)
                val_tgt_lens = val_tgt_lens.to(device)
                if use_amp:
                    with autocast():
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
                except RuntimeError as e:
                    logger.warning(f"Epoch {epoch}: Invalid validation loss: {e}")
                    # Continue with other validation batches
        
        avg_val_loss = val_loss_sum / max(val_count, 1)
        logger.epoch_end(epoch, train_loss=None, val_loss=avg_val_loss)
        model.train()
        head.train()
        
        # Save at end of epoch if max_steps not reached
        if step < cfg["max_steps"]:
            final_path = os.path.join(cfg["save_dir"], "audio_enc.pt")
            checkpoint_data = {
                "enc": model.state_dict(),
                "head": head.state_dict(),
                "optimizer": opt.state_dict(),
                "scheduler": scheduler.state_dict(),
                "step": step,
                "best_val_loss": best_val_loss
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
