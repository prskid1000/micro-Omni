
"""
Train OCR (Optical Character Recognition) model for extracting text from images.

Architecture:
- Vision Encoder (ViT): Processes image
- Text Decoder: Autoregressively generates text from visual features
- Training: Teacher forcing with cross-entropy loss
"""

import argparse
import json
import os
import csv
import torch
from torch import nn
from torch.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from omni.ocr_model import OCRModel
from omni.training_utils import set_seed, get_lr_scheduler, clip_gradients, SimpleLogger, validate_loss, check_gradient_explosion, cleanup_old_checkpoints
from tqdm import tqdm


def collate_ocr_fn(batch):
    """Collate function for OCR dataset"""
    images, texts = zip(*batch)
    images = torch.stack(images)
    
    # Pad texts to same length
    max_text_len = max(len(t) for t in texts)
    padded_texts = []
    for t in texts:
        pad_len = max_text_len - len(t)
        if pad_len > 0:
            t = t + [0] * pad_len  # Pad with 0 (blank/PAD token)
        padded_texts.append(t)
    
    return images, torch.tensor(padded_texts, dtype=torch.long)


class OCRDataset(Dataset):
    """Dataset for OCR training - images with text labels"""
    def __init__(self, csv_path, image_root, img_size=224, cfg=None):
        self.csv_path = csv_path
        self.image_root = image_root
        self.img_size = img_size
        
        # Image transforms
        self.tf = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor()
        ])
        
        # Build character vocabulary
        self.char_to_idx = {}
        self.idx_to_char = {}
        self._build_vocab(csv_path)
        
        # Build row offset index for efficient random access
        self.row_offsets = []
        self.fieldnames = None
        with open(csv_path, 'rb') as f:
            header_line = f.readline()
            if not header_line:
                raise ValueError("CSV file is empty")
            self.fieldnames = header_line.decode('utf-8').strip().split(',')
            offset = f.tell()
            while True:
                line_start = offset
                line_bytes = f.readline()
                if not line_bytes:
                    break
                try:
                    decoded = line_bytes.decode('utf-8').strip()
                    if decoded:
                        self.row_offsets.append(line_start)
                except UnicodeDecodeError:
                    pass
                offset += len(line_bytes)
    
    def _build_vocab(self, csv_path):
        """Build character vocabulary from all texts in CSV (resumable, processes in chunks)"""
        # Check for existing vocabulary checkpoint (resumable)
        checkpoint_path = f"{csv_path}.vocab_checkpoint.json"
        chars = set()
        start_row = 0
        
        if os.path.exists(checkpoint_path):
            print("Found vocabulary checkpoint, resuming from checkpoint...")
            try:
                checkpoint_data = json.load(open(checkpoint_path, 'r'))
                chars = set(checkpoint_data.get("chars", []))
                start_row = checkpoint_data.get("last_processed_row", 0)
                print(f"  Resuming from row {start_row:,}")
            except Exception as e:
                print(f"  Warning: Could not load checkpoint: {e}, starting from beginning")
                chars = set()
                start_row = 0
        
        if start_row == 0:
            print("Building character vocabulary from OCR dataset (processing in chunks, resumable)...")
        
        # Build row offset index first (if not already done)
        row_offsets = []
        with open(csv_path, 'rb') as f:
            header = f.readline()
            fieldnames = header.decode('utf-8').strip().split(',')
            offset = f.tell()
            while True:
                line_start = offset
                line_bytes = f.readline()
                if not line_bytes:
                    break
                try:
                    decoded = line_bytes.decode('utf-8').strip()
                    if decoded:
                        row_offsets.append(line_start)
                except UnicodeDecodeError:
                    pass
                offset += len(line_bytes)
        
        # Process rows in chunks (resumable)
        checkpoint_freq = 10000  # Save checkpoint every N rows
        total_rows = len(row_offsets)
        
        with open(csv_path, 'rb') as f:
            header = f.readline()
            fieldnames = header.decode('utf-8').strip().split(',')
            
            for row_idx, row_start in enumerate(row_offsets[start_row:], start_row):
                f.seek(row_start)
                line_bytes = f.readline()
                try:
                    line = line_bytes.decode('utf-8').strip()
                    import io
                    reader = csv.DictReader(io.StringIO(line), fieldnames=fieldnames)
                    row = next(reader)
                    text = row.get("text", "") or row.get("label", "")
                    chars.update(text)
                except:
                    pass
                
                # Save checkpoint periodically (resumable)
                if (row_idx + 1) % checkpoint_freq == 0:
                    try:
                        checkpoint_data = {
                            "chars": list(chars),
                            "last_processed_row": row_idx + 1
                        }
                        json.dump(checkpoint_data, open(checkpoint_path, 'w'))
                        print(f"  Checkpoint saved: processed {row_idx+1:,}/{total_rows:,} rows...")
                    except Exception as e:
                        print(f"  Warning: Could not save checkpoint: {e}")
                
                # Progress indicator
                if (row_idx + 1) % 10000 == 0:
                    print(f"  Processed {row_idx+1:,}/{total_rows:,} rows...")
        
        # Final checkpoint save
        try:
            checkpoint_data = {
                "chars": list(chars),
                "last_processed_row": total_rows
            }
            json.dump(checkpoint_data, open(checkpoint_path, 'w'))
        except:
            pass
        
        # Build vocabulary (0 = PAD/BLANK, 1 = BOS, 2 = EOS)
        self.char_to_idx = {'<PAD>': 0, '<BOS>': 1, '<EOS>': 2, '<UNK>': 3}
        self.idx_to_char = {0: '<PAD>', 1: '<BOS>', 2: '<EOS>', 3: '<UNK>'}
        
        # Add all characters
        for char in sorted(chars):
            if char not in self.char_to_idx:
                idx = len(self.char_to_idx)
                self.char_to_idx[char] = idx
                self.idx_to_char[idx] = char
        
        print(f"Vocabulary size: {len(self.char_to_idx)} characters")
        
        # Clean up checkpoint after successful completion
        if os.path.exists(checkpoint_path):
            try:
                os.remove(checkpoint_path)
            except:
                pass
    
    def __len__(self):
        return len(self.row_offsets)
    
    def __getitem__(self, i):
        """Load image and text"""
        # Read specific row using file offset
        with open(self.csv_path, 'rb') as f:
            f.seek(self.row_offsets[i])
            line_bytes = f.readline()
            line = line_bytes.decode('utf-8').strip()
        
        # Parse CSV row
        import io
        reader = csv.DictReader(io.StringIO(line), fieldnames=self.fieldnames)
        row = next(reader)
        
        # Get image path and text
        if "image" in row:
            img_path = row["image"]
        elif "img" in row:
            img_path = row["img"]
        else:
            raise ValueError(f"CSV must have 'image' or 'img' column. Found: {self.fieldnames}")
        
        text = row.get("text", "") or row.get("label", "") or row.get("text_label", "")
        
        # Load image
        full_img_path = os.path.join(self.image_root, img_path) if not os.path.isabs(img_path) else img_path
        img = Image.open(full_img_path).convert("RGB")
        img_tensor = self.tf(img)
        
        # Encode text to character IDs
        text_ids = [self.char_to_idx.get(c, self.char_to_idx['<UNK>']) for c in text]
        if not text_ids:
            text_ids = [self.char_to_idx['<UNK>']]
        
        # Add BOS and EOS
        text_ids = [self.char_to_idx['<BOS>']] + text_ids + [self.char_to_idx['<EOS>']]
        
        return img_tensor, text_ids


def main(cfg):
    # Set random seed
    seed = cfg.get("seed", 42)
    set_seed(seed)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(cfg["save_dir"], exist_ok=True)
    
    # Load dataset
    csv_path = cfg.get("train_csv", "data/ocr/ocr_train.csv")
    image_root = cfg.get("image_root", "data/ocr")
    
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"OCR CSV not found. Expected: {csv_path}")
    
    ds = OCRDataset(csv_path, image_root, cfg.get("img_size", 224), cfg=cfg)
    vocab_size = len(ds.char_to_idx)
    print(f"Character vocabulary size: {vocab_size}")
    
    dl = DataLoader(
        ds,
        batch_size=cfg.get("batch_size", 4),
        shuffle=True,
        num_workers=cfg.get("num_workers", 2),
        drop_last=cfg.get("drop_last", True),
        collate_fn=collate_ocr_fn
    )
    
    # Initialize model
    use_compile = cfg.get("use_compile", False)
    model = OCRModel(
        img_size=cfg.get("img_size", 224),
        patch=cfg.get("patch", 16),
        vision_d_model=cfg.get("vision_d_model", 128),
        vision_layers=cfg.get("vision_layers", 4),
        vision_heads=cfg.get("vision_heads", 2),
        vision_d_ff=cfg.get("vision_d_ff", 512),
        decoder_d_model=cfg.get("decoder_d_model", 256),
        decoder_layers=cfg.get("decoder_layers", 4),
        decoder_heads=cfg.get("decoder_heads", 4),
        decoder_d_ff=cfg.get("decoder_d_ff", 1024),
        vocab_size=vocab_size,
        dropout=cfg.get("dropout", 0.1),
        compile_model=use_compile
    ).to(device)
    
    # Optimizer
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.get("lr", 3e-4),
        weight_decay=cfg.get("wd", 0.01)
    )
    
    # Loss function (ignore PAD token)
    loss_fn = nn.CrossEntropyLoss(ignore_index=0)
    
    # Learning rate scheduler
    warmup_steps = cfg.get("warmup_steps", 500)
    max_steps = cfg.get("max_steps", 10000)
    scheduler = get_lr_scheduler(opt, warmup_steps, max_steps)
    
    # Gradient clipping
    max_grad_norm = cfg.get("max_grad_norm", 1.0)
    
    # Gradient accumulation
    accumulation_steps = cfg.get("gradient_accumulation_steps", 1)
    
    # Mixed precision
    use_amp = cfg.get("use_amp", True) and device == "cuda"
    scaler = GradScaler('cuda') if use_amp else None
    if use_amp:
        print("Mixed precision training (AMP) enabled")
    if accumulation_steps > 1:
        print(f"Gradient accumulation: {accumulation_steps} steps")
    
    # Validation split
    val_split = cfg.get("val_split", 0.1)
    total_size = len(ds)
    val_size = int(total_size * val_split)
    train_size = total_size - val_size
    train_ds, val_ds = torch.utils.data.random_split(
        ds, [train_size, val_size],
        generator=torch.Generator().manual_seed(seed)
    )
    train_dl = DataLoader(
        train_ds,
        batch_size=cfg.get("batch_size", 4),
        shuffle=True,
        num_workers=cfg.get("num_workers", 2),
        drop_last=True,
        collate_fn=collate_ocr_fn
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=cfg.get("batch_size", 4),
        shuffle=False,
        num_workers=cfg.get("num_workers", 2),
        drop_last=False,
        collate_fn=collate_ocr_fn
    )
    
    # Logger
    logger = SimpleLogger("OCR")
    
    # Resume from checkpoint
    step = 0
    start_epoch = 0
    resume_from = None
    if os.path.exists(cfg["save_dir"]):
        checkpoint_files = [f for f in os.listdir(cfg["save_dir"]) if f.startswith("ocr_step_") and f.endswith(".pt")]
        if checkpoint_files:
            step_numbers = []
            for f in checkpoint_files:
                try:
                    step_num = int(f.replace("ocr_step_", "").replace(".pt", ""))
                    step_numbers.append((step_num, f))
                except:
                    continue
            if step_numbers:
                step_numbers.sort(key=lambda x: x[0], reverse=True)
                resume_from = os.path.join(cfg["save_dir"], step_numbers[0][1])
                step = step_numbers[0][0]
                logger.info(f"Found checkpoint at step {step}, resuming from: {resume_from}")
                
                checkpoint = torch.load(resume_from, map_location=device)
                if "model" in checkpoint:
                    model.load_state_dict(checkpoint["model"])
                if "optimizer" in checkpoint:
                    opt.load_state_dict(checkpoint["optimizer"])
                if "scheduler" in checkpoint:
                    scheduler.load_state_dict(checkpoint["scheduler"])
                if "scaler" in checkpoint and scaler is not None:
                    scaler.load_state_dict(checkpoint["scaler"])
                if "step" in checkpoint:
                    step = checkpoint["step"]
                if "char_to_idx" in checkpoint:
                    ds.char_to_idx = checkpoint["char_to_idx"]
                    ds.idx_to_char = checkpoint["idx_to_char"]
                logger.info(f"Resumed from step {step}")
    
    logger.training_start(max_steps, train_size, val_size)
    
    steps_per_epoch = len(train_dl)
    initial_step = step
    if step > 0:
        start_epoch = step // steps_per_epoch
    
    max_epochs = cfg.get("max_epochs", 9999)
    print_freq = cfg.get("print_freq", 50)
    checkpoint_freq = cfg.get("checkpoint_freq", 1000)
    val_freq = cfg.get("val_freq", 500)
    
    model.train()
    
    for epoch in range(start_epoch, max_epochs):
        for batch_idx, (images, text_ids) in enumerate(tqdm(train_dl, desc=f"epoch{epoch}")):
            # Skip batches if resuming
            if epoch == start_epoch and initial_step > 0:
                current_batch_step = epoch * steps_per_epoch + batch_idx
                if current_batch_step < initial_step:
                    continue
            
            images = images.to(device)  # (B, 3, H, W)
            text_ids = text_ids.to(device)  # (B, T)
            
            # Teacher forcing: shift by one for next token prediction
            input_ids = text_ids[:, :-1]  # (B, T-1)
            target_ids = text_ids[:, 1:]  # (B, T-1)
            
            if use_amp:
                with autocast(device_type='cuda'):
                    logits = model(images, input_ids)  # (B, T-1, vocab_size)
                    loss = loss_fn(logits.reshape(-1, logits.size(-1)), target_ids.reshape(-1))
            else:
                logits = model(images, input_ids)
                loss = loss_fn(logits.reshape(-1, logits.size(-1)), target_ids.reshape(-1))
            
            # Backward pass
            loss_scaled = loss / accumulation_steps
            if use_amp:
                scaler.scale(loss_scaled).backward()
            else:
                loss_scaled.backward()
            
            loss_val = loss.detach()
            del loss, logits
            
            # Gradient accumulation
            if (step + 1) % accumulation_steps == 0:
                if use_amp:
                    scaler.unscale_(opt)
                
                # Validate loss
                unscaled_loss = loss_val * accumulation_steps
                try:
                    validate_loss(unscaled_loss, min_loss=-1e6, max_loss=1e6)
                except RuntimeError as e:
                    logger.error(f"Step {step}: {e}. Skipping batch.")
                    opt.zero_grad()
                    if use_amp:
                        scaler.update()
                    continue
                
                # Check gradient explosion
                try:
                    grad_norm, is_exploded = check_gradient_explosion(model, max_grad_norm=100.0, raise_on_error=False)
                    if is_exploded:
                        logger.error(f"Step {step}: Gradient explosion (norm={grad_norm:.2f}). Skipping batch.")
                        opt.zero_grad()
                        if use_amp:
                            scaler.update()
                        continue
                except RuntimeError as e:
                    logger.error(f"Step {step}: {e}")
                    opt.zero_grad()
                    if use_amp:
                        scaler.update()
                    continue
                
                # Gradient clipping and step
                if use_amp:
                    clip_gradients(model, max_grad_norm)
                    scaler.step(opt)
                    scaler.update()
                else:
                    clip_gradients(model, max_grad_norm)
                    opt.step()
                
                scheduler.step()
                opt.zero_grad()
            
            step += 1
            
            # Logging
            if step % print_freq == 0:
                current_lr = scheduler.get_last_lr()[0]
                unscaled_loss = loss_val * accumulation_steps
                logger.train_step(step, float(unscaled_loss), current_lr, epoch)
            
            # Validation
            if step % val_freq == 0:
                model.eval()
                val_loss_sum = 0.0
                val_count = 0
                
                with torch.no_grad():
                    for val_images, val_text_ids in val_dl:
                        if val_count >= 10:  # Limit validation batches
                            break
                        
                        val_images = val_images.to(device)
                        val_text_ids = val_text_ids.to(device)
                        val_input_ids = val_text_ids[:, :-1]
                        val_target_ids = val_text_ids[:, 1:]
                        
                        if use_amp:
                            with autocast(device_type='cuda'):
                                val_logits = model(val_images, val_input_ids)
                                val_loss = loss_fn(val_logits.reshape(-1, val_logits.size(-1)), val_target_ids.reshape(-1))
                        else:
                            val_logits = model(val_images, val_input_ids)
                            val_loss = loss_fn(val_logits.reshape(-1, val_logits.size(-1)), val_target_ids.reshape(-1))
                        
                        try:
                            validate_loss(val_loss, min_loss=-1e6, max_loss=1e6)
                            val_loss_sum += float(val_loss.detach())
                            val_count += 1
                        except RuntimeError:
                            pass
                
                if val_count > 0:
                    avg_val_loss = val_loss_sum / val_count
                    logger.val_step(step, avg_val_loss, epoch)
                
                model.train()
            
            # Checkpointing
            if step % checkpoint_freq == 0:
                checkpoint_path = os.path.join(cfg["save_dir"], f"ocr_step_{step}.pt")
                torch.save({
                    "model": model.state_dict(),
                    "optimizer": opt.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "scaler": scaler.state_dict() if scaler else None,
                    "step": step,
                    "char_to_idx": ds.char_to_idx,
                    "idx_to_char": ds.idx_to_char,
                    "config": cfg
                }, checkpoint_path)
                
                # Save final checkpoint
                final_path = os.path.join(cfg["save_dir"], "ocr.pt")
                torch.save({
                    "model": model.state_dict(),
                    "char_to_idx": ds.char_to_idx,
                    "idx_to_char": ds.idx_to_char,
                    "config": cfg
                }, final_path)
                
                logger.info(f"Saved checkpoint: {checkpoint_path}")
                cleanup_old_checkpoints(cfg["save_dir"], "ocr_step_", keep_last=3)
            
            if step >= max_steps:
                logger.info(f"Reached max_steps ({max_steps}), stopping training")
                break
        
        if step >= max_steps:
            break
    
    # Save final model
    final_path = os.path.join(cfg["save_dir"], "ocr.pt")
    torch.save({
        "model": model.state_dict(),
        "char_to_idx": ds.char_to_idx,
        "idx_to_char": ds.idx_to_char,
        "config": cfg
    }, final_path)
    logger.info(f"Training complete! Final model saved to: {final_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train OCR model")
    parser.add_argument("--config", type=str, default="configs/ocr_tiny.json",
                       help="Path to config file")
    args = parser.parse_args()
    
    if os.path.exists(args.config):
        with open(args.config, 'r') as f:
            cfg = json.load(f)
    else:
        # Default config
        cfg = {
            "save_dir": "checkpoints/ocr_tiny",
            "train_csv": "data/ocr/ocr_train.csv",
            "image_root": "data/ocr",
            "img_size": 224,
            "patch": 16,
            "vision_d_model": 128,
            "vision_layers": 4,
            "vision_heads": 2,
            "vision_d_ff": 512,
            "decoder_d_model": 256,
            "decoder_layers": 4,
            "decoder_heads": 4,
            "decoder_d_ff": 1024,
            "dropout": 0.1,
            "batch_size": 4,
            "num_workers": 2,
            "drop_last": True,
            "lr": 3e-4,
            "wd": 0.01,
            "warmup_steps": 500,
            "max_steps": 10000,
            "max_epochs": 9999,
            "gradient_accumulation_steps": 1,
            "max_grad_norm": 1.0,
            "use_amp": True,
            "val_split": 0.1,
            "print_freq": 50,
            "checkpoint_freq": 1000,
            "val_freq": 500,
            "seed": 42
        }
        print(f"Config file not found, using defaults. Creating: {args.config}")
        os.makedirs(os.path.dirname(args.config), exist_ok=True)
        with open(args.config, 'w') as f:
            json.dump(cfg, f, indent=2)
    
    main(cfg)

