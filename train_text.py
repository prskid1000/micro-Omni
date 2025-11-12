
import argparse, json, torch, os
from torch import nn
from torch.utils.data import Dataset, DataLoader
from omni.thinker import ThinkerLM
from omni.tokenizer import BPETokenizer
from tqdm import tqdm

class TextDataset(Dataset):
    def __init__(self, path, tokenizer, ctx):
        with open(path, 'r', encoding='utf-8') as f:
            self.lines = [l.strip() for l in f if l.strip()]
        self.tok = tokenizer; self.ctx = ctx
    def __len__(self): return len(self.lines)
    def __getitem__(self, i):
        ids = self.tok.encode(self.lines[i])[:self.ctx-1]
        ids = [1] + ids  # BOS=1 (SentencePiece default)
        pad = [0] * (self.ctx - len(ids))
        x = torch.tensor(ids + pad, dtype=torch.long)
        y = x.clone(); y[:-1]=x[1:]; y[-1]=0
        return x, y

def main(cfg):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(cfg["save_dir"], exist_ok=True)
    spm_model = os.path.join(cfg["save_dir"], "tokenizer.model")
    if not os.path.exists(spm_model):
        BPETokenizer.train_new(cfg["train_text"], spm_model, vocab_size=cfg["vocab_size"])
    tok = BPETokenizer(spm_model)
    ds = TextDataset(cfg["train_text"], tok, cfg["ctx_len"])
    dl = DataLoader(ds, batch_size=cfg.get("batch_size", 8), shuffle=True, num_workers=cfg.get("num_workers", 2), drop_last=cfg.get("drop_last", True))
    model = ThinkerLM(cfg["vocab_size"], cfg["n_layers"], cfg["d_model"], cfg["n_heads"], cfg["d_ff"], cfg["dropout"], cfg["rope_theta"], cfg["ctx_len"]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    loss_fn = nn.CrossEntropyLoss(ignore_index=0)

    step=0
    model.train()
    max_epochs = cfg.get("max_epochs", 9999)
    print_freq = cfg.get("print_freq", 1)
    for epoch in range(max_epochs):
        for x,y in tqdm(dl, desc=f"epoch{epoch}"):
            x,y = x.to(device), y.to(device)
            logits = model(x)  # (B,T,V)
            loss = loss_fn(logits.view(-1, logits.size(-1)), y.view(-1))
            opt.zero_grad(); loss.backward(); opt.step()
            step+=1
            if step % print_freq == 0:
                print("step", step, "loss", float(loss))
            if step >= cfg["max_steps"]:
                torch.save(model.state_dict(), os.path.join(cfg["save_dir"], "thinker.pt"))
                print("Saved to", cfg["save_dir"]); return
        # Save at end of epoch if max_steps not reached
        if step < cfg["max_steps"]:
            torch.save(model.state_dict(), os.path.join(cfg["save_dir"], "thinker.pt"))
            print("Saved to", cfg["save_dir"], "at end of epoch", epoch, "step", step); return

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--config", required=True)
    cfg = json.load(open(ap.parse_args().config))
    main(cfg)
