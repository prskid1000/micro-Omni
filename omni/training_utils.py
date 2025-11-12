"""
Training utilities for standard ML practices:
- Learning rate scheduling with warmup
- Gradient clipping
- Random seed setting
- Mixed precision training support
- Simple terminal logging
"""

import random
import math
import torch
import torch.nn as nn
from datetime import datetime

def set_seed(seed=42):
    """Set random seed for reproducibility"""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_lr_scheduler(optimizer, warmup_steps, max_steps, min_lr_ratio=0.1):
    """
    Create learning rate scheduler with linear warmup + cosine decay.
    
    Args:
        optimizer: PyTorch optimizer
        warmup_steps: Number of warmup steps
        max_steps: Total training steps
        min_lr_ratio: Minimum LR as ratio of initial LR (default: 0.1)
    
    Returns:
        LambdaLR scheduler
    """
    def lr_lambda(step):
        if step < warmup_steps:
            # Linear warmup: lr = base_lr * (step / warmup_steps)
            return step / max(warmup_steps, 1)
        else:
            # Cosine decay: lr = base_lr * (min_ratio + (1-min_ratio) * 0.5 * (1 + cos(progress * pi)))
            progress = (step - warmup_steps) / max((max_steps - warmup_steps), 1)
            return min_lr_ratio + (1 - min_lr_ratio) * 0.5 * (1 + torch.cos(torch.tensor(progress * math.pi)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

def clip_gradients(model, max_norm=1.0):
    """
    Clip gradients to prevent explosion.
    
    Args:
        model: PyTorch model
        max_norm: Maximum gradient norm (default: 1.0)
    
    Returns:
        Gradient norm before clipping
    """
    return torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

class SimpleLogger:
    """Simple terminal logger for training metrics"""
    
    def __init__(self, name="Training"):
        self.name = name
        self.start_time = datetime.now()
    
    def _format_time(self):
        """Format current time"""
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    def _format_message(self, level, message):
        """Format log message with timestamp"""
        timestamp = self._format_time()
        return f"[{timestamp}] [{self.name}] [{level}] {message}"
    
    def info(self, message):
        """Log info message"""
        print(self._format_message("INFO", message))
    
    def warning(self, message):
        """Log warning message"""
        print(self._format_message("WARN", message))
    
    def error(self, message):
        """Log error message"""
        print(self._format_message("ERROR", message))
    
    def train_step(self, step, loss, lr, epoch=None):
        """Log training step metrics"""
        epoch_str = f"epoch={epoch}, " if epoch is not None else ""
        msg = f"Step {step} | {epoch_str}train_loss={loss:.4f} | lr={lr:.6f}"
        print(self._format_message("TRAIN", msg))
    
    def val_step(self, step, val_loss, epoch=None):
        """Log validation step metrics"""
        epoch_str = f"epoch={epoch}, " if epoch is not None else ""
        msg = f"Step {step} | {epoch_str}val_loss={val_loss:.4f}"
        print(self._format_message("VAL", msg))
    
    def checkpoint(self, step, path, is_best=False):
        """Log checkpoint save"""
        best_str = " (BEST)" if is_best else ""
        msg = f"Checkpoint saved at step {step}{best_str}: {path}"
        print(self._format_message("CHECKPOINT", msg))
    
    def epoch_start(self, epoch):
        """Log epoch start"""
        msg = f"Starting epoch {epoch}"
        print(self._format_message("EPOCH", msg))
    
    def epoch_end(self, epoch, train_loss=None, val_loss=None):
        """Log epoch end"""
        parts = [f"Epoch {epoch} completed"]
        if train_loss is not None:
            parts.append(f"train_loss={train_loss:.4f}")
        if val_loss is not None:
            parts.append(f"val_loss={val_loss:.4f}")
        msg = " | ".join(parts)
        print(self._format_message("EPOCH", msg))
    
    def training_start(self, total_steps, train_samples, val_samples=None):
        """Log training start"""
        msg = f"Starting training | max_steps={total_steps} | train_samples={train_samples}"
        if val_samples is not None:
            msg += f" | val_samples={val_samples}"
        print(self._format_message("START", msg))
    
    def training_end(self, total_steps):
        """Log training end"""
        elapsed = datetime.now() - self.start_time
        msg = f"Training completed | total_steps={total_steps} | elapsed={elapsed}"
        print(self._format_message("END", msg))

