
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
import torch
from functools import partial
from torch import nn
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from omni.ocr_model import OCRModel
from omni.utils import (
    set_seed, get_lr_scheduler, clip_gradients, SimpleLogger, validate_loss, 
    check_gradient_explosion, OCRDataset,
    load_checkpoint, setup_resume_data_loading, calculate_resume_position,
    ValidationSkipSamplesContext, analyze_ocr_dataset,
    save_training_metadata, load_training_metadata
)
from tqdm import tqdm


def collate_ocr_fn(batch, max_text_length=None):
    """
    Collate function that pads all text sequences to a fixed maximum length.
    This ensures uniform batch sizes for CUDA graphs compilation.
    
    Args:
        batch: List of (image, text) tuples
        max_text_length: Fixed maximum length to pad to. If None, uses batch max (not recommended for CUDA graphs)
    """
    images, texts = zip(*batch)
    images = torch.stack(images)
    
    # Use fixed max length if provided, otherwise use batch max
    if max_text_length is not None:
        max_text_len = max_text_length
    else:
        max_text_len = max(len(t) for t in texts)
    
    padded_texts = []
    for t in texts:
        current_len = len(t)
        if current_len > max_text_len:
            # Truncate if longer than max (shouldn't happen with proper config)
            t = t[:max_text_len]
            current_len = max_text_len
        
        pad_len = max_text_len - current_len
        if pad_len > 0:
            t = t + [0] * pad_len  # Pad with 0 (blank/PAD token)
        padded_texts.append(t)
    
    return images, torch.tensor(padded_texts, dtype=torch.long)


def main(cfg):
    # Set random seed
    seed = cfg.get("seed", 42)
    set_seed(seed)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    save_dir = cfg.get("save_dir", "checkpoints/ocr_tiny")
    os.makedirs(save_dir, exist_ok=True)
    
    # Load dataset
    csv_path = cfg.get("train_csv", "data/ocr/ocr_train.csv")
    image_root = cfg.get("image_root", "data/ocr")
    
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"OCR CSV not found. Expected: {csv_path}")
    
    # Check if metadata exists (contains previously calculated values)
    model_name = "ocr"
    metadata = load_training_metadata(save_dir, model_name)
    
    if metadata and not cfg.get("recalculate_dataset_stats", False):
        # Load calculated values from metadata (avoid recalculating)
        print("Loading dataset statistics from metadata...")
        char_to_idx = metadata.get("char_to_idx", {})
        idx_to_char = metadata.get("idx_to_char", {})
        vocab_size_dynamic = metadata.get("vocab_size", None)
        max_text_length_dynamic = metadata.get("max_text_length", None)
        
        print(f"Character vocabulary size: {vocab_size_dynamic} (from metadata)")
        print(f"Text length: {max_text_length_dynamic} (from metadata)")
    else:
        # Analyze OCR dataset in a single pass: build vocabulary and calculate max text length
        print("Analyzing OCR dataset (vocabulary, text length)...")
        # Percentile threshold for minimizing padding (default: 95% coverage)
        text_percentile = cfg.get("max_text_length_percentile", 95.0)
        char_to_idx, idx_to_char, vocab_size_dynamic, max_text_length_dynamic = analyze_ocr_dataset(
            csv_path, text_percentile=text_percentile
        )
        
        print(f"Character vocabulary size: {vocab_size_dynamic}")
        print(f"Unique characters found: {len(char_to_idx) - 4}")  # Exclude <PAD>, <BOS>, <EOS>, <UNK>
        print(f"Text length at {text_percentile}th percentile: {max_text_length_dynamic} (covers {text_percentile}% of data, minimizes padding)")
        print(f"  Note: ~{100 - text_percentile:.1f}% of data will be truncated if longer (acceptable for outliers)")
        
        # Save calculated values to metadata (so we don't recalculate next time)
        training_metadata = {
            "step": 0,  # Will be updated when we save checkpoints
            "epoch": 0,
            "char_to_idx": char_to_idx,
            "idx_to_char": idx_to_char,
            "vocab_size": vocab_size_dynamic,
            "max_text_length": max_text_length_dynamic,
            "config": cfg
        }
        save_training_metadata(save_dir, model_name, training_metadata)
        print("✓ Saved dataset statistics to metadata (will be reused on next run)")
    
    # Allow override from config, but default to auto-calculated values
    vocab_size = vocab_size_dynamic  # OCR vocabulary is always dynamic, no override needed
    print(f"✓ Using dynamic vocabulary size: {vocab_size}")
    
    max_text_length = cfg.get("max_text_length", max_text_length_dynamic)
    if max_text_length != max_text_length_dynamic:
        print(f"⚠ Warning: Config max_text_length={max_text_length} differs from dataset max length={max_text_length_dynamic}")
        print(f"  Using config value: {max_text_length}")
    else:
        print(f"✓ Using auto-calculated max_text_length: {max_text_length}")
    
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
        use_gqa=cfg.get("use_gqa", False),
        use_swiglu=cfg.get("use_swiglu", True),
        use_flash=cfg.get("use_flash", True),
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
    
    train_ds = OCRDataset(
        csv_path, 
        image_root, 
        cfg.get("img_size", 224), 
        cfg=cfg,
        shuffle_buffer_size=cfg.get("shuffle_buffer_size", 10000),
        seed=seed,
        skip_samples=0,
        char_to_idx=char_to_idx,  # Use pre-built vocabulary (avoids rebuilding)
        idx_to_char=idx_to_char
    )
    train_ds._val_split = val_split
    train_ds._val_mode = False  # Training mode
    
    val_ds = OCRDataset(
        csv_path, 
        image_root, 
        cfg.get("img_size", 224), 
        cfg=cfg,
        shuffle_buffer_size=0,  # No shuffling for validation
        seed=seed,  # Same seed for consistent hash-based split
        skip_samples=0,
        char_to_idx=char_to_idx,  # Use pre-built vocabulary (avoids rebuilding)
        idx_to_char=idx_to_char
    )
    val_ds._val_split = val_split
    val_ds._val_mode = True  # Validation mode
    
    # max_text_length is already calculated above
    if use_compile:
        print(f"Using fixed max_text_length={max_text_length} for CUDA graphs compatibility")
    
    # Create collate function with fixed max length using functools.partial (pickleable for Windows multiprocessing)
    collate_fn_with_max = partial(collate_ocr_fn, max_text_length=max_text_length)
    
    # Approximate sizes for logging (will count if needed)
    try:
        total_size = train_ds.get_length()
        train_size = int(total_size * (1 - val_split))
        val_size = total_size - train_size
    except:
        train_size = val_size = None  # Unknown size
    
    # Note: shuffle=False for IterableDataset (shuffling handled internally)
    train_dl = DataLoader(
        train_ds,
        batch_size=cfg.get("batch_size", 4),
        shuffle=False,
        num_workers=cfg.get("num_workers", 2),
        drop_last=True,
        collate_fn=collate_fn_with_max
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=cfg.get("batch_size", 4),
        shuffle=False,
        num_workers=cfg.get("num_workers", 2),
        drop_last=cfg.get("drop_last", True),
        collate_fn=collate_fn_with_max
    )
    
    # Logger
    logger = SimpleLogger("OCR")
    
    # Resume from checkpoint
    step = 0
    step, metadata = load_checkpoint(
        save_dir, 
        model_name, 
        device, 
        logger,
        state_dict_loaders={
            "model": (model, model.load_state_dict),
            "optimizer": (opt, opt.load_state_dict),
            "scheduler": (scheduler, scheduler.load_state_dict),
            "scaler": (scaler, scaler.load_state_dict) if scaler is not None else None
        }
    )
    # Load scaler from model file if needed
    if step > 0 and scaler is not None:
        model_path = os.path.join(save_dir, f"{model_name}.pt")
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=device)
            if isinstance(checkpoint, dict) and "scaler" in checkpoint:
                scaler.load_state_dict(checkpoint["scaler"])
    # Load char_to_idx from metadata if available
    if step > 0 and metadata:
        if "char_to_idx" in metadata:
            train_ds.char_to_idx = metadata["char_to_idx"]
            train_ds.idx_to_char = metadata["idx_to_char"]
    
    # Update skip_samples for dataset if resuming
    batch_size = cfg.get("batch_size", 4)
    new_train_dl = setup_resume_data_loading(
        train_ds, step, batch_size, logger,
        train_dl_kwargs={
            "num_workers": cfg.get("num_workers", 2),
            "drop_last": True,
            "collate_fn": collate_fn_with_max
        }
    )
    if new_train_dl is not None:
        train_dl = new_train_dl
    
    logger.training_start(max_steps, train_size, val_size)
    
    # Calculate steps per epoch and determine starting epoch/position
    # For IterableDataset, we can't use len() directly, so calculate from dataset size
    batch_size = cfg.get("batch_size", 4)
    drop_last = True  # OCR uses drop_last=True
    if train_size is not None:
        steps_per_epoch = train_size // batch_size
        if not drop_last and train_size % batch_size != 0:
            steps_per_epoch += 1
    else:
        # Fallback: use a large number if size is unknown (for progress bar)
        # The actual training will work fine, just progress bar won't be accurate
        steps_per_epoch = 1000000  # Large placeholder
    initial_step = step
    start_epoch, start_batch_idx = calculate_resume_position(step, steps_per_epoch)
    if step > 0:
        logger.info(f"Resuming from step {step} (epoch {start_epoch}, batch {start_batch_idx}/{steps_per_epoch})")
    
    max_epochs = cfg.get("max_epochs", 9999)
    print_freq = cfg.get("print_freq", 50)
    checkpoint_freq = cfg.get("checkpoint_freq", 1000)
    val_freq = cfg.get("val_freq", 500)
    
    model.train()
    
    for epoch in range(start_epoch, max_epochs):
        # Recreate DataLoader for each epoch since IterableDatasets are exhausted after one iteration
        # skip_samples is automatically reset to 0 by the dataset after first iteration
        if epoch > start_epoch:
            train_dl = DataLoader(
                train_ds,
                batch_size=cfg.get("batch_size", 4),
                shuffle=False,
                num_workers=cfg.get("num_workers", 2),
                drop_last=True,
                collate_fn=collate_fn_with_max
            )
        
        # Create progress bar with correct starting position when resuming mid-epoch
        remaining_epochs = max_epochs - epoch - 1
        pbar_desc = f"epoch{epoch}/{max_epochs-1} (remaining:{remaining_epochs}) step{step}"
        if epoch == start_epoch and start_batch_idx > 0:
            pbar = tqdm(train_dl, desc=pbar_desc, initial=start_batch_idx, total=steps_per_epoch)
        else:
            pbar = tqdm(train_dl, desc=pbar_desc, total=steps_per_epoch)
        
        # Start enumeration from the correct position when resuming mid-epoch
        enum_start = start_batch_idx if (epoch == start_epoch and start_batch_idx > 0) else 0
        for batch_idx, (images, text_ids) in enumerate(pbar, start=enum_start):
            # Skip batches if resuming mid-epoch
            # batch_idx already represents the position in the epoch when enum_start > 0
            if epoch == start_epoch and initial_step > 0:
                current_batch_step = epoch * steps_per_epoch + batch_idx
                if current_batch_step < initial_step:
                    continue
            
            # Update progress bar description
            remaining_epochs = max_epochs - epoch - 1
            pbar.set_description(f"epoch{epoch}/{max_epochs-1} (remaining:{remaining_epochs}) step{step} batch{batch_idx}")
            
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
                
                # Gradient clipping first (already unscaled if using AMP)
                # Clip gradients to prevent explosion, then check if still too high
                try:
                    grad_norm_before = clip_gradients(model, max_grad_norm)
                    
                    # Check for gradient explosion AFTER clipping
                    # Use a higher threshold (10x max_grad_norm) since we've already clipped
                    # This allows clipping to fix most cases, only skip if truly exploded
                    explosion_threshold = max(100.0, max_grad_norm * 10)
                    grad_norm_after, is_exploded = check_gradient_explosion(model, max_grad_norm=explosion_threshold, raise_on_error=False)
                    
                    if is_exploded:
                        logger.error(f"Step {step}: Gradient explosion detected after clipping (norm: {grad_norm_before:.2f}->{grad_norm_after:.2f}). Skipping batch.")
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
                
                # Optimizer step (gradients already clipped)
                if use_amp:
                    scaler.step(opt)
                    scaler.update()
                else:
                    opt.step()
                
                scheduler.step()
                opt.zero_grad()
                step += 1  # Increment step counter only when optimizer step occurs
            
            # Logging
            if step % print_freq == 0:
                current_lr = scheduler.get_last_lr()[0]
                unscaled_loss = loss_val * accumulation_steps
                logger.train_step(step, float(unscaled_loss), current_lr, epoch)
            
            # Validation
            if step > 0 and step % val_freq == 0:
                with ValidationSkipSamplesContext(train_ds):
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
            
            # Checkpointing - save only model file and metadata
            if step > 0 and step % checkpoint_freq == 0:
                # Save model weights only (overwrite existing file)
                model_path = os.path.join(save_dir, f"{model_name}.pt")
                model_data = {
                    "model": model.state_dict(),
                    "optimizer": opt.state_dict(),
                    "scheduler": scheduler.state_dict(),
                }
                if scaler is not None:
                    model_data["scaler"] = scaler.state_dict()
                torch.save(model_data, model_path)
                
                # Save training metadata (step, calculated values, etc.)
                training_metadata = {
                    "step": step,
                    "epoch": epoch,
                    "char_to_idx": train_ds.char_to_idx,
                    "idx_to_char": train_ds.idx_to_char,
                    "vocab_size": vocab_size_dynamic,
                    "max_text_length": max_text_length_dynamic,
                    "config": cfg
                }
                save_training_metadata(save_dir, model_name, training_metadata)
                logger.checkpoint(step, model_path)
                
                # Save final checkpoint (same as periodic, but for compatibility)
                final_path = os.path.join(save_dir, f"{model_name}.pt")
                torch.save({
                    "model": model.state_dict(),
                    "char_to_idx": train_ds.char_to_idx,
                    "idx_to_char": train_ds.idx_to_char,
                    "config": cfg
                }, final_path)
                
                logger.info(f"Saved checkpoint: {checkpoint_path}")
                cleanup_old_checkpoints(save_dir, "ocr_step_", keep_last_n=1)
            
            if step >= max_steps:
                logger.info(f"Reached max_steps ({max_steps}), stopping training")
                break
        
        if step >= max_steps:
            break
    
    # Save final model
    final_path = os.path.join(save_dir, f"{model_name}.pt")
    model_data = {
        "model": model.state_dict(),
    }
    torch.save(model_data, final_path)
    
    # Save final training metadata
    training_metadata = {
        "step": step,
        "epoch": epoch,
        "char_to_idx": train_ds.char_to_idx,
        "idx_to_char": train_ds.idx_to_char,
        "vocab_size": vocab_size_dynamic,
        "max_text_length": max_text_length_dynamic,
        "config": cfg
    }
    save_training_metadata(save_dir, model_name, training_metadata)
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

