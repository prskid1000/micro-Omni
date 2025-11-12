
import argparse, json, os, torch, torchaudio, csv
from torch import nn
from torch.utils.data import Dataset, DataLoader
from omni.audio_encoder import AudioEncoderTiny
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
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(cfg["save_dir"], exist_ok=True)
    sr = cfg.get("sample_rate", 16000)
    n_mels = cfg.get("mel_bins", 128)
    ds = ASRDataset(cfg["train_csv"], sr=sr, n_mels=n_mels, cfg=cfg)
    dl = DataLoader(ds, batch_size=cfg.get("batch_size", 4), shuffle=True, num_workers=cfg.get("num_workers", 2), drop_last=cfg.get("drop_last", True), collate_fn=collate_fn)
    model = AudioEncoderTiny(cfg["d_model"], cfg["n_heads"], cfg["d_ff"], cfg["n_layers"], cfg["dropout"]).to(device)
    # simple CTC head on top
    vocab = cfg.get("ctc_vocab_size", 64)  # toy char vocab
    head = nn.Linear(cfg["d_model"], vocab).to(device)
    ctc_loss = nn.CTCLoss(blank=0, zero_infinity=True)
    opt = torch.optim.AdamW(list(model.parameters())+list(head.parameters()), lr=cfg["lr"], weight_decay=cfg["wd"])

    step=0
    max_epochs = cfg.get("max_epochs", 9999)
    print_freq = cfg.get("print_freq", 100)
    for epoch in range(max_epochs):
        for mel, text in tqdm(dl):
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
            opt.zero_grad(); loss.backward(); opt.step()
            step+=1
            if step % print_freq == 0: print("step", step, "ctc_loss", float(loss))
            if step >= cfg["max_steps"]:
                torch.save({"enc": model.state_dict(), "head": head.state_dict()}, os.path.join(cfg["save_dir"], "audio_enc.pt"))
                print("Saved", cfg["save_dir"]); return
        # Save at end of epoch if max_steps not reached
        if step < cfg["max_steps"]:
            torch.save({"enc": model.state_dict(), "head": head.state_dict()}, os.path.join(cfg["save_dir"], "audio_enc.pt"))
            print("Saved", cfg["save_dir"], "at end of epoch", epoch, "step", step); return

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--config", required=True)
    cfg = json.load(open(ap.parse_args().config))
    main(cfg)
