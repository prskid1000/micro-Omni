
import argparse, json, os, csv, torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torchaudio
from omni.codec import RVQ
from omni.talker import TalkerTiny
from omni.training_utils import set_seed, get_lr_scheduler, clip_gradients, SimpleLogger, validate_loss, check_gradient_explosion
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
        self.rows = []
        with open(csv_path, 'r', encoding='utf-8') as f:
            rd = csv.DictReader(f)
            for r in rd: self.rows.append(r)
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

    def __len__(self): return len(self.rows)
    def __getitem__(self, i):
        text, path = self.rows[i]["text"], self.rows[i]["wav"]
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
    rvq = RVQ(cfg["codebooks"], cfg["codebook_size"], d=64).to(device)
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
        rope_theta=cfg.get("rope_theta", 10000.0)
    ).to(device)
    opt = torch.optim.AdamW(list(rvq.parameters())+list(talker.parameters()), lr=cfg["lr"], weight_decay=cfg["wd"])
    loss_fn = nn.CrossEntropyLoss()
    
    # Learning rate scheduler with warmup
    warmup_steps = cfg.get("warmup_steps", 500)
    max_steps = cfg.get("max_steps", 5000)
    scheduler = get_lr_scheduler(opt, warmup_steps, max_steps)
    
    # Gradient clipping
    max_grad_norm = cfg.get("max_grad_norm", 1.0)

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
    best_val_loss = float('inf')
    
    logger.training_start(cfg["max_steps"], train_size, val_size)
    
    for epoch in range(max_epochs):
        for mel in tqdm(train_dl, desc=f"epoch{epoch}"):
            mel = mel.to(device)  # (B,T,128)
            # Batch encode all frames at once (optimized)
            idxs = rvq.encode(mel)  # (B,T,2) - encodes all frames in batch
            # AR training: predict current codes from previous codes
            prev = torch.roll(idxs, 1, dims=1); prev[:,0,:]=0
            base_logit, res_logit = talker(prev)
            loss = loss_fn(base_logit.reshape(-1, base_logit.size(-1)), idxs[:,:,0].reshape(-1)) + \
                   loss_fn(res_logit.reshape(-1, res_logit.size(-1)),  idxs[:,:,1].reshape(-1))
            
            # Validate loss value
            try:
                validate_loss(loss, min_loss=-1e6, max_loss=1e6)
            except RuntimeError as e:
                logger.error(f"Step {step}: {e}")
                logger.error("Skipping this batch due to invalid loss")
                continue
            
            opt.zero_grad()
            loss.backward()
            
            # Check for gradient explosion before clipping
            try:
                grad_norm_rvq, is_exploded_rvq = check_gradient_explosion(rvq, max_grad_norm=100.0, raise_on_error=False)
                grad_norm_talker, is_exploded_talker = check_gradient_explosion(talker, max_grad_norm=100.0, raise_on_error=False)
                if is_exploded_rvq or is_exploded_talker:
                    logger.error(f"Step {step}: Gradient explosion detected (rvq={grad_norm_rvq:.2f}, talker={grad_norm_talker:.2f}). Skipping this batch.")
                    opt.zero_grad()  # Clear gradients
                    continue
            except RuntimeError as e:
                logger.error(f"Step {step}: {e}")
                opt.zero_grad()  # Clear gradients
                continue
            
            # Gradient clipping
            clip_gradients(rvq, max_grad_norm)
            clip_gradients(talker, max_grad_norm)
            
            opt.step()
            scheduler.step()
            step+=1
            if step % print_freq == 0:
                current_lr = scheduler.get_last_lr()[0]
                logger.train_step(step, float(loss), current_lr, epoch)
            
            # Periodic checkpointing
            if step % checkpoint_freq == 0 and step > 0:
                checkpoint_path = os.path.join(cfg["save_dir"], f"talker_step_{step}.pt")
                torch.save({"rvq": rvq.state_dict(), "talker": talker.state_dict()}, checkpoint_path)
                logger.checkpoint(step, checkpoint_path)
            
            # Validation
            if step % val_freq == 0 and step > 0:
                rvq.eval()
                talker.eval()
                val_loss_sum = 0.0
                val_count = 0
                with torch.no_grad():
                    for val_mel in val_dl:
                        val_mel = val_mel.to(device)
                        # Batch encode all frames at once (optimized)
                        val_idxs = rvq.encode(val_mel)  # (B,T,2)
                        val_prev = torch.roll(val_idxs, 1, dims=1); val_prev[:,0,:]=0
                        val_base_logit, val_res_logit = talker(val_prev)
                        val_loss = loss_fn(val_base_logit.reshape(-1, val_base_logit.size(-1)), val_idxs[:,:,0].reshape(-1)) + \
                                   loss_fn(val_res_logit.reshape(-1, val_res_logit.size(-1)), val_idxs[:,:,1].reshape(-1))
                        
                        # Validate validation loss
                        try:
                            validate_loss(val_loss, min_loss=-1e6, max_loss=1e6)
                            val_loss_sum += float(val_loss)
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
                    best_path = os.path.join(cfg["save_dir"], "talker_best.pt")
                    torch.save({"rvq": rvq.state_dict(), "talker": talker.state_dict()}, best_path)
                    logger.checkpoint(step, best_path, is_best=True)
                
                rvq.train()
                talker.train()
            
            if step >= cfg["max_steps"]:
                torch.save({"rvq": rvq.state_dict(), "talker": talker.state_dict()}, os.path.join(cfg["save_dir"], "talker.pt"))
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
                # Batch encode all frames at once (optimized)
                val_idxs = rvq.encode(val_mel)  # (B,T,2)
                val_prev = torch.roll(val_idxs, 1, dims=1); val_prev[:,0,:]=0
                val_base_logit, val_res_logit = talker(val_prev)
                val_loss = loss_fn(val_base_logit.reshape(-1, val_base_logit.size(-1)), val_idxs[:,:,0].reshape(-1)) + \
                           loss_fn(val_res_logit.reshape(-1, val_res_logit.size(-1)), val_idxs[:,:,1].reshape(-1))
                
                # Validate validation loss
                try:
                    validate_loss(val_loss, min_loss=-1e6, max_loss=1e6)
                    val_loss_sum += float(val_loss)
                    val_count += 1
                except RuntimeError as e:
                    logger.warning(f"Epoch {epoch}: Invalid validation loss: {e}")
                    # Continue with other validation batches
        
        avg_val_loss = val_loss_sum / max(val_count, 1)
        logger.epoch_end(epoch, train_loss=None, val_loss=avg_val_loss)
        rvq.train()
        talker.train()
        
        # Save at end of epoch if max_steps not reached
        if step < cfg["max_steps"]:
            torch.save({"rvq": rvq.state_dict(), "talker": talker.state_dict()}, os.path.join(cfg["save_dir"], "talker.pt"))
            logger.info(f"Model saved to {cfg['save_dir']} at end of epoch {epoch}, step {step}")
            return

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--config", required=True)
    cfg = json.load(open(ap.parse_args().config))
    main(cfg)
