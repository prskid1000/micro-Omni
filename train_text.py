
import argparse, json, torch, os, random
from torch import nn
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader
from omni.thinker import ThinkerLM
from omni.tokenizer import BPETokenizer
from omni.training_utils import set_seed, get_lr_scheduler, clip_gradients, SimpleLogger, validate_loss, check_gradient_explosion
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
    
    # Mixed precision training (AMP)
    use_amp = cfg.get("use_amp", True) and device == "cuda"
    scaler = GradScaler() if use_amp else None
    if use_amp:
        print("Mixed precision training (AMP) enabled")

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
    
    # Resume from checkpoint if available
    resume_from = None
    checkpoint_files = [f for f in os.listdir(cfg["save_dir"]) if f.startswith("thinker_step_") and f.endswith(".pt")]
    if checkpoint_files:
        # Extract step numbers and find latest
        step_numbers = []
        for f in checkpoint_files:
            try:
                step_num = int(f.replace("thinker_step_", "").replace(".pt", ""))
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
            if isinstance(checkpoint, dict) and "model" in checkpoint:
                model.load_state_dict(checkpoint["model"])
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
                # Legacy checkpoint format (just model weights)
                model.load_state_dict(checkpoint)
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
        logger.epoch_start(epoch)
        for batch_idx, (x,y) in enumerate(tqdm(train_dl, desc=f"epoch{epoch}")):
            # Skip batches if resuming mid-epoch
            if epoch == start_epoch and initial_step > 0:
                current_batch_step = epoch * steps_per_epoch + batch_idx
                if current_batch_step < initial_step:
                    continue
            x,y = x.to(device), y.to(device)
            
            # Forward pass with mixed precision
            opt.zero_grad()
            if use_amp:
                with autocast():
                    logits = model(x)  # (B,T,V)
                    loss = loss_fn(logits.view(-1, logits.size(-1)), y.view(-1))
            else:
                logits = model(x)  # (B,T,V)
                loss = loss_fn(logits.view(-1, logits.size(-1)), y.view(-1))
            
            # Validate loss value
            try:
                validate_loss(loss, min_loss=-1e6, max_loss=1e6)
            except RuntimeError as e:
                logger.error(f"Step {step}: {e}")
                logger.error("Skipping this batch due to invalid loss")
                continue
            
            # Backward pass with gradient scaling
            if use_amp:
                scaler.scale(loss).backward()
                # Unscale before checking gradients (gradients are in scaled space until unscaled)
                scaler.unscale_(opt)
            else:
                loss.backward()
            
            # Check for gradient explosion before clipping (after unscaling if AMP)
            try:
                grad_norm, is_exploded = check_gradient_explosion(model, max_grad_norm=100.0, raise_on_error=False)
                if is_exploded:
                    logger.error(f"Step {step}: Gradient explosion detected (grad_norm={grad_norm:.2f}). Skipping this batch.")
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
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                scaler.step(opt)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                opt.step()
            scheduler.step()
            step+=1
            
            if step % print_freq == 0:
                current_lr = scheduler.get_last_lr()[0]
                logger.train_step(step, float(loss.detach()), current_lr, epoch)
            
            # Periodic checkpointing
            if step % checkpoint_freq == 0 and step > 0:
                checkpoint_path = os.path.join(cfg["save_dir"], f"thinker_step_{step}.pt")
                checkpoint_data = {
                    "model": model.state_dict(),
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
                val_loss_sum = 0.0
                val_count = 0
                with torch.no_grad():
                    for val_x, val_y in val_dl:
                        val_x, val_y = val_x.to(device), val_y.to(device)
                        if use_amp:
                            with autocast():
                                val_logits = model(val_x)
                                val_loss = loss_fn(val_logits.view(-1, val_logits.size(-1)), val_y.view(-1))
                        else:
                            val_logits = model(val_x)
                            val_loss = loss_fn(val_logits.view(-1, val_logits.size(-1)), val_y.view(-1))
                        
                        # Validate validation loss
                        try:
                            validate_loss(val_loss, min_loss=-1e6, max_loss=1e6)
                            val_loss_sum += float(val_loss.detach())
                            val_count += 1
                        except RuntimeError as e:
                            logger.warning(f"Step {step}: Invalid validation loss: {e}")
                            # Continue with other validation batches
                        
                        if val_count >= 20:  # Limit validation batches
                            break
                
                avg_val_loss = val_loss_sum / val_count
                logger.val_step(step, avg_val_loss, epoch)
                
                # Save best model
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    best_path = os.path.join(cfg["save_dir"], "thinker_best.pt")
                    checkpoint_data = {
                        "model": model.state_dict(),
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
            
            if step >= max_steps:
                final_path = os.path.join(cfg["save_dir"], "thinker.pt")
                checkpoint_data = {
                    "model": model.state_dict(),
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
        val_loss_sum = 0.0
        val_count = 0
        with torch.no_grad():
            for val_x, val_y in val_dl:
                val_x, val_y = val_x.to(device), val_y.to(device)
                if use_amp:
                    with autocast():
                        val_logits = model(val_x)
                        val_loss = loss_fn(val_logits.view(-1, val_logits.size(-1)), val_y.view(-1))
                else:
                    val_logits = model(val_x)
                    val_loss = loss_fn(val_logits.view(-1, val_logits.size(-1)), val_y.view(-1))
                
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
        
        # Save at end of epoch if max_steps not reached
        if step < cfg["max_steps"]:
            final_path = os.path.join(cfg["save_dir"], "thinker.pt")
            checkpoint_data = {
                "model": model.state_dict(),
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
