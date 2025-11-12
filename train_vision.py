
import argparse, json, os, torch, json as js
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from omni.vision_encoder import ViTTiny
from omni.training_utils import set_seed, get_lr_scheduler, clip_gradients, SimpleLogger
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
    # Set random seed for reproducibility
    seed = cfg.get("seed", 42)
    set_seed(seed)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(cfg["save_dir"], exist_ok=True)
    ds = ImgCapDataset(cfg["train_manifest"], cfg["image_root"], cfg["img_size"])
    dl = DataLoader(ds, batch_size=cfg.get("batch_size", 8), shuffle=True, num_workers=cfg.get("num_workers", 2), drop_last=cfg.get("drop_last", True))
    vit = ViTTiny(cfg["img_size"], cfg["patch"], cfg["d_model"], cfg["n_layers"], cfg["n_heads"], cfg["d_ff"], cfg["dropout"]).to(device)
    head_output_size = cfg.get("head_output_size", 64)
    head = nn.Linear(cfg["d_model"], head_output_size).to(device)  # predict bag-of-words toy target
    opt = torch.optim.AdamW(list(vit.parameters())+list(head.parameters()), lr=cfg["lr"], weight_decay=cfg["wd"])
    loss_fn = nn.CrossEntropyLoss()
    
    # Learning rate scheduler with warmup
    warmup_steps = cfg.get("warmup_steps", 500)
    max_steps = cfg.get("max_steps", 5000)
    scheduler = get_lr_scheduler(opt, warmup_steps, max_steps)
    
    # Gradient clipping
    max_grad_norm = cfg.get("max_grad_norm", 1.0)

    words = ["red","blue","square","background","big","small","roughly","pixel"]

    # Split dataset for validation
    val_split = cfg.get("val_split", 0.1)  # 10% for validation
    total_size = len(ds)
    val_size = int(total_size * val_split)
    train_size = total_size - val_size
    train_ds, val_ds = torch.utils.data.random_split(ds, [train_size, val_size], generator=torch.Generator().manual_seed(seed))
    train_dl = DataLoader(train_ds, batch_size=cfg.get("batch_size", 8), shuffle=True, num_workers=cfg.get("num_workers", 2), drop_last=cfg.get("drop_last", True))
    val_dl = DataLoader(val_ds, batch_size=cfg.get("batch_size", 8), shuffle=False, num_workers=cfg.get("num_workers", 2), drop_last=False)
    
    # Initialize logger
    logger = SimpleLogger("Vision")
    
    step=0
    vit.train()
    head.train()
    max_epochs = cfg.get("max_epochs", 9999)
    print_freq = cfg.get("print_freq", 100)
    checkpoint_freq = cfg.get("checkpoint_freq", 500)  # Save checkpoint every N steps
    val_freq = cfg.get("val_freq", 200)  # Validate every N steps
    best_val_loss = float('inf')
    
    logger.training_start(cfg["max_steps"], train_size, val_size)
    
    for epoch in range(max_epochs):
        for img, cap in tqdm(train_dl, desc=f"epoch{epoch}"):
            img = img.to(device)
            cls,_ = vit(img)  # (B,1,d)
            logit = head(cls.squeeze(1))  # (B,64)
            # make a toy label: predict 'red'(0) if 'red' in caption else 'blue'(1)
            labels = torch.tensor([0 if "red" in c else 1 for c in cap], dtype=torch.long, device=device)
            loss = loss_fn(logit, labels)
            opt.zero_grad()
            loss.backward()
            
            # Gradient clipping
            clip_gradients(vit, max_grad_norm)
            clip_gradients(head, max_grad_norm)
            
            opt.step()
            scheduler.step()
            step+=1
            if step % print_freq == 0:
                current_lr = scheduler.get_last_lr()[0]
                logger.train_step(step, float(loss), current_lr, epoch)
            
            # Periodic checkpointing
            if step % checkpoint_freq == 0 and step > 0:
                checkpoint_path = os.path.join(cfg["save_dir"], f"vision_step_{step}.pt")
                torch.save({"vit": vit.state_dict(), "head": head.state_dict()}, checkpoint_path)
                logger.checkpoint(step, checkpoint_path)
            
            # Validation
            if step % val_freq == 0 and step > 0:
                vit.eval()
                head.eval()
                val_loss_sum = 0.0
                val_count = 0
                with torch.no_grad():
                    for val_img, val_cap in val_dl:
                        val_img = val_img.to(device)
                        val_cls, _ = vit(val_img)
                        val_logit = head(val_cls.squeeze(1))
                        val_labels = torch.tensor([0 if "red" in c else 1 for c in val_cap], dtype=torch.long, device=device)
                        val_loss = loss_fn(val_logit, val_labels)
                        val_loss_sum += float(val_loss)
                        val_count += 1
                        if val_count >= 10:  # Limit validation batches
                            break
                
                avg_val_loss = val_loss_sum / val_count
                logger.val_step(step, avg_val_loss, epoch)
                
                # Save best model
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    best_path = os.path.join(cfg["save_dir"], "vision_best.pt")
                    torch.save({"vit": vit.state_dict(), "head": head.state_dict()}, best_path)
                    logger.checkpoint(step, best_path, is_best=True)
                
                vit.train()
                head.train()
            
            if step >= cfg["max_steps"]:
                torch.save({"vit": vit.state_dict(), "head": head.state_dict()}, os.path.join(cfg["save_dir"], "vision.pt"))
                logger.info(f"Final model saved to {cfg['save_dir']}")
                logger.training_end(step)
                return
        
        # Final validation at end of epoch
        vit.eval()
        head.eval()
        val_loss_sum = 0.0
        val_count = 0
        with torch.no_grad():
            for val_img, val_cap in val_dl:
                val_img = val_img.to(device)
                val_cls, _ = vit(val_img)
                val_logit = head(val_cls.squeeze(1))
                val_labels = torch.tensor([0 if "red" in c else 1 for c in val_cap], dtype=torch.long, device=device)
                val_loss = loss_fn(val_logit, val_labels)
                val_loss_sum += float(val_loss)
                val_count += 1
        
        avg_val_loss = val_loss_sum / max(val_count, 1)
        logger.epoch_end(epoch, train_loss=None, val_loss=avg_val_loss)
        vit.train()
        head.train()
        
        # Save at end of epoch if max_steps not reached
        if step < cfg["max_steps"]:
            torch.save({"vit": vit.state_dict(), "head": head.state_dict()}, os.path.join(cfg["save_dir"], "vision.pt"))
            logger.info(f"Model saved to {cfg['save_dir']} at end of epoch {epoch}, step {step}")
            return

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--config", required=True)
    cfg = json.load(open(ap.parse_args().config))
    main(cfg)
