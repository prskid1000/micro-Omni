
import argparse, json, os, csv, torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torchaudio
from omni.codec import RVQ
from omni.talker import TalkerTiny
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
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(cfg["save_dir"], exist_ok=True)
    sr = cfg.get("sample_rate", 16000)
    n_mels = cfg.get("n_mels", 128)
    frame_ms = cfg.get("frame_ms", 80)
    ds = TTSDataset(cfg["tts_csv"], sr=sr, n_mels=n_mels, frame_ms=frame_ms, cfg=cfg)
    # Use module-level collate function for Windows multiprocessing compatibility
    dl = DataLoader(ds, batch_size=cfg.get("batch_size", 4), shuffle=True, num_workers=cfg.get("num_workers", 2), drop_last=cfg.get("drop_last", True), collate_fn=collate_mel_fn)
    rvq = RVQ(cfg["codebooks"], cfg["codebook_size"], d=64).to(device)
    talker = TalkerTiny(cfg["d_model"], cfg["n_layers"], cfg["n_heads"], cfg["d_ff"], cfg["codebooks"], cfg["codebook_size"], cfg["dropout"]).to(device)
    opt = torch.optim.AdamW(list(rvq.parameters())+list(talker.parameters()), lr=cfg["lr"], weight_decay=cfg["wd"])
    loss_fn = nn.CrossEntropyLoss()

    step=0
    rvq.train()
    talker.train()
    max_epochs = cfg.get("max_epochs", 9999)
    print_freq = cfg.get("print_freq", 50)
    for epoch in range(max_epochs):
        for mel in tqdm(dl, desc=f"epoch{epoch}"):
            mel = mel.to(device)  # (B,T,128)
            B,T,_ = mel.shape
            idxs = []
            for t in range(T):
                idx = rvq.encode(mel[:,t,:])  # (B,2)
                idxs.append(idx)
            idxs = torch.stack(idxs, dim=1)  # (B,T,2)
            # AR training: predict current codes from previous codes
            prev = torch.roll(idxs, 1, dims=1); prev[:,0,:]=0
            base_logit, res_logit = talker(prev)
            loss = loss_fn(base_logit.reshape(-1, base_logit.size(-1)), idxs[:,:,0].reshape(-1)) + \
                   loss_fn(res_logit.reshape(-1, res_logit.size(-1)),  idxs[:,:,1].reshape(-1))
            opt.zero_grad(); loss.backward(); opt.step()
            step+=1
            if step % print_freq == 0:
                print("step", step, "loss", float(loss))
            if step >= cfg["max_steps"]:
                torch.save({"rvq": rvq.state_dict(), "talker": talker.state_dict()}, os.path.join(cfg["save_dir"], "talker.pt"))
                print("Saved to", cfg["save_dir"]); return
        # Save at end of epoch if max_steps not reached
        if step < cfg["max_steps"]:
            torch.save({"rvq": rvq.state_dict(), "talker": talker.state_dict()}, os.path.join(cfg["save_dir"], "talker.pt"))
            print("Saved to", cfg["save_dir"], "at end of epoch", epoch, "step", step); return

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--config", required=True)
    cfg = json.load(open(ap.parse_args().config))
    main(cfg)
