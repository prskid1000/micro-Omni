
import argparse, json, os, torch, json as js, random
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from omni.vision_encoder import ViTTiny
from tqdm import tqdm

class ImgCapDataset(Dataset):
    def __init__(self, manifest, image_root, img_size=224):
        self.items = js.load(open(manifest, 'r', encoding='utf-8'))
        self.root = image_root
        self.tf = transforms.Compose([
            transforms.Resize((img_size,img_size)),
            transforms.ToTensor()
        ])
    def __len__(self): return len(self.items)
    def __getitem__(self, i):
        it = self.items[i]
        img = Image.open(os.path.join(self.root, it["image"])).convert("RGB")
        return self.tf(img), it["caption"]

def main(cfg):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(cfg["save_dir"], exist_ok=True)
    ds = ImgCapDataset(cfg["train_manifest"], cfg["image_root"], cfg["img_size"])
    dl = DataLoader(ds, batch_size=cfg.get("batch_size", 8), shuffle=True, num_workers=cfg.get("num_workers", 2), drop_last=cfg.get("drop_last", True))
    vit = ViTTiny(cfg["img_size"], cfg["patch"], cfg["d_model"], cfg["n_layers"], cfg["n_heads"], cfg["d_ff"], cfg["dropout"]).to(device)
    head_output_size = cfg.get("head_output_size", 64)
    head = nn.Linear(cfg["d_model"], head_output_size).to(device)  # predict bag-of-words toy target
    opt = torch.optim.AdamW(list(vit.parameters())+list(head.parameters()), lr=cfg["lr"], weight_decay=cfg["wd"])
    loss_fn = nn.CrossEntropyLoss()

    words = ["red","blue","square","background","big","small","roughly","pixel"]

    step=0
    max_epochs = cfg.get("max_epochs", 9999)
    print_freq = cfg.get("print_freq", 100)
    for epoch in range(max_epochs):
        for img, cap in tqdm(dl):
            img = img.to(device)
            cls,_ = vit(img)  # (B,1,d)
            logit = head(cls.squeeze(1))  # (B,64)
            # make a toy label: predict 'red'(0) if 'red' in caption else 'blue'(1)
            labels = torch.tensor([0 if "red" in c else 1 for c in cap], dtype=torch.long, device=device)
            loss = loss_fn(logit, labels)
            opt.zero_grad(); loss.backward(); opt.step()
            step+=1
            if step % print_freq == 0: print("step", step, "loss", float(loss))
            if step >= cfg["max_steps"]:
                torch.save({"vit": vit.state_dict(), "head": head.state_dict()}, os.path.join(cfg["save_dir"], "vision.pt"))
                print("Saved", cfg["save_dir"]); return
        # Save at end of epoch if max_steps not reached
        if step < cfg["max_steps"]:
            torch.save({"vit": vit.state_dict(), "head": head.state_dict()}, os.path.join(cfg["save_dir"], "vision.pt"))
            print("Saved", cfg["save_dir"], "at end of epoch", epoch, "step", step); return

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--config", required=True)
    cfg = json.load(open(ap.parse_args().config))
    main(cfg)
