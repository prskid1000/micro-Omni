
import argparse, json, torch, os, random
from torch import nn
from torch.utils.data import Dataset, DataLoader
from omni.thinker import ThinkerLM
from omni.tokenizer import BPETokenizer
from omni.training_utils import set_seed, get_lr_scheduler, clip_gradients, SimpleLogger
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
    # Set random seed for reproducibility
    seed = cfg.get("seed", 42)
    set_seed(seed)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(cfg["save_dir"], exist_ok=True)
    spm_model = os.path.join(cfg["save_dir"], "tokenizer.model")
    if not os.path.exists(spm_model):
        BPETokenizer.train_new(cfg["train_text"], spm_model, vocab_size=cfg["vocab_size"])
    tok = BPETokenizer(spm_model)
    ds = TextDataset(cfg["train_text"], tok, cfg["ctx_len"])
    dl = DataLoader(ds, batch_size=cfg.get("batch_size", 8), shuffle=True, num_workers=cfg.get("num_workers", 2), drop_last=cfg.get("drop_last", True))
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
        num_experts_per_tok=cfg.get("num_experts_per_tok", 2)
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    loss_fn = nn.CrossEntropyLoss(ignore_index=0)
    
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
    best_val_loss = float('inf')
    
    logger.training_start(cfg["max_steps"], train_size, val_size)
    
    for epoch in range(max_epochs):
        logger.epoch_start(epoch)
        for x,y in tqdm(train_dl, desc=f"epoch{epoch}"):
            x,y = x.to(device), y.to(device)
            logits = model(x)  # (B,T,V)
            loss = loss_fn(logits.view(-1, logits.size(-1)), y.view(-1))
            opt.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            
            opt.step()
            scheduler.step()
            step+=1
            
            if step % print_freq == 0:
                current_lr = scheduler.get_last_lr()[0]
                logger.train_step(step, float(loss), current_lr, epoch)
            
            # Periodic checkpointing
            if step % checkpoint_freq == 0 and step > 0:
                checkpoint_path = os.path.join(cfg["save_dir"], f"thinker_step_{step}.pt")
                torch.save(model.state_dict(), checkpoint_path)
                logger.checkpoint(step, checkpoint_path)
            
            # Validation
            if step % val_freq == 0 and step > 0:
                model.eval()
                val_loss_sum = 0.0
                val_count = 0
                with torch.no_grad():
                    for val_x, val_y in val_dl:
                        val_x, val_y = val_x.to(device), val_y.to(device)
                        val_logits = model(val_x)
                        val_loss = loss_fn(val_logits.view(-1, val_logits.size(-1)), val_y.view(-1))
                        val_loss_sum += float(val_loss)
                        val_count += 1
                        if val_count >= 20:  # Limit validation batches
                            break
                
                avg_val_loss = val_loss_sum / val_count
                logger.val_step(step, avg_val_loss, epoch)
                
                # Save best model
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    best_path = os.path.join(cfg["save_dir"], "thinker_best.pt")
                    torch.save(model.state_dict(), best_path)
                    logger.checkpoint(step, best_path, is_best=True)
                
                model.train()
            
            if step >= max_steps:
                torch.save(model.state_dict(), os.path.join(cfg["save_dir"], "thinker.pt"))
                logger.info(f"Final model saved to {cfg['save_dir']}")
                logger.training_end(step)
                return
        
        # Final validation at end of epoch
        model.eval()
        val_loss_sum = 0.0
        val_count = 0
        with torch.no_grad():
            for val_x, val_y in val_dl:
                val_x, val_y = val_x.to(device), val_y.to(device)
                val_logits = model(val_x)
                val_loss = loss_fn(val_logits.view(-1, val_logits.size(-1)), val_y.view(-1))
                val_loss_sum += float(val_loss)
                val_count += 1
        
        avg_val_loss = val_loss_sum / max(val_count, 1)
        logger.epoch_end(epoch, train_loss=None, val_loss=avg_val_loss)
        model.train()
        
        # Save at end of epoch if max_steps not reached
        if step < cfg["max_steps"]:
            torch.save(model.state_dict(), os.path.join(cfg["save_dir"], "thinker.pt"))
            logger.info(f"Model saved to {cfg['save_dir']} at end of epoch {epoch}, step {step}")
            return

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--config", required=True)
    cfg = json.load(open(ap.parse_args().config))
    main(cfg)
