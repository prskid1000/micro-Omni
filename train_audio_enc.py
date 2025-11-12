
import argparse, json, os, torch, torchaudio, csv
from torch import nn
from torch.utils.data import Dataset, DataLoader
from omni.audio_encoder import AudioEncoderTiny
from omni.training_utils import set_seed, get_lr_scheduler, clip_gradients, SimpleLogger
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
    # simple CTC head on top
    vocab = cfg.get("ctc_vocab_size", 64)  # toy char vocab
    head = nn.Linear(cfg["d_model"], vocab).to(device)
    ctc_loss = nn.CTCLoss(blank=0, zero_infinity=True)
    opt = torch.optim.AdamW(list(model.parameters())+list(head.parameters()), lr=cfg["lr"], weight_decay=cfg["wd"])
    
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
    
    logger.training_start(cfg["max_steps"], train_size, val_size)
    
    for epoch in range(max_epochs):
        for mel, text in tqdm(train_dl, desc=f"epoch{epoch}"):
            mel = mel.to(device)
            x = model(mel)  # (B, T', d)
            logit = head(x)  # (B,T',V)
            # fabricate tiny targets: map chars to 1..63
            max_text_len = cfg.get("max_text_len", 16)
            tgt = []
            for t in text:
                ids = [ (ord(c)%63)+1 for c in t[:max_text_len] ]  # small cap
                tgt.append(torch.tensor(ids, dtype=torch.long))
            tgt_lens = torch.tensor([len(t) for t in tgt], dtype=torch.long)
            tgt = torch.cat(tgt)
            log_prob = logit.log_softmax(-1).transpose(0,1)  # (T',B,V)
            inp_lens = torch.full((log_prob.size(1),), log_prob.size(0), dtype=torch.long)
            loss = ctc_loss(log_prob, tgt, inp_lens, tgt_lens)
            opt.zero_grad()
            loss.backward()
            
            # Gradient clipping
            clip_gradients(model, max_grad_norm)
            clip_gradients(head, max_grad_norm)
            
            opt.step()
            scheduler.step()
            step+=1
            if step % print_freq == 0:
                current_lr = scheduler.get_last_lr()[0]
                logger.train_step(step, float(loss), current_lr, epoch)
            
            # Periodic checkpointing
            if step % checkpoint_freq == 0 and step > 0:
                checkpoint_path = os.path.join(cfg["save_dir"], f"audio_enc_step_{step}.pt")
                torch.save({"enc": model.state_dict(), "head": head.state_dict()}, checkpoint_path)
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
                        val_x = model(val_mel)
                        val_logit = head(val_x)
                        val_tgt = []
                        for t in val_text:
                            ids = [ (ord(c)%63)+1 for c in t[:max_text_len] ]
                            val_tgt.append(torch.tensor(ids, dtype=torch.long))
                        val_tgt_lens = torch.tensor([len(t) for t in val_tgt], dtype=torch.long)
                        val_tgt = torch.cat(val_tgt)
                        val_log_prob = val_logit.log_softmax(-1).transpose(0,1)
                        val_inp_lens = torch.full((val_log_prob.size(1),), val_log_prob.size(0), dtype=torch.long)
                        val_loss = ctc_loss(val_log_prob, val_tgt, val_inp_lens, val_tgt_lens)
                        val_loss_sum += float(val_loss)
                        val_count += 1
                        if val_count >= 10:  # Limit validation batches
                            break
                
                avg_val_loss = val_loss_sum / val_count
                logger.val_step(step, avg_val_loss, epoch)
                
                # Save best model
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    best_path = os.path.join(cfg["save_dir"], "audio_enc_best.pt")
                    torch.save({"enc": model.state_dict(), "head": head.state_dict()}, best_path)
                    logger.checkpoint(step, best_path, is_best=True)
                
                model.train()
                head.train()
            
            if step >= cfg["max_steps"]:
                torch.save({"enc": model.state_dict(), "head": head.state_dict()}, os.path.join(cfg["save_dir"], "audio_enc.pt"))
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
                val_x = model(val_mel)
                val_logit = head(val_x)
                val_tgt = []
                for t in val_text:
                    ids = [ (ord(c)%63)+1 for c in t[:max_text_len] ]
                    val_tgt.append(torch.tensor(ids, dtype=torch.long))
                val_tgt_lens = torch.tensor([len(t) for t in val_tgt], dtype=torch.long)
                val_tgt = torch.cat(val_tgt)
                val_log_prob = val_logit.log_softmax(-1).transpose(0,1)
                val_inp_lens = torch.full((val_log_prob.size(1),), val_log_prob.size(0), dtype=torch.long)
                val_loss = ctc_loss(val_log_prob, val_tgt, val_inp_lens, val_tgt_lens)
                val_loss_sum += float(val_loss)
                val_count += 1
        
        avg_val_loss = val_loss_sum / max(val_count, 1)
        logger.epoch_end(epoch, train_loss=None, val_loss=avg_val_loss)
        model.train()
        head.train()
        
        # Save at end of epoch if max_steps not reached
        if step < cfg["max_steps"]:
            torch.save({"enc": model.state_dict(), "head": head.state_dict()}, os.path.join(cfg["save_dir"], "audio_enc.pt"))
            logger.info(f"Model saved to {cfg['save_dir']} at end of epoch {epoch}, step {step}")
            return

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--config", required=True)
    cfg = json.load(open(ap.parse_args().config))
    main(cfg)
