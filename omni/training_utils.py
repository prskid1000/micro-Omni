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

def check_numerical_stability(tensor, name="tensor", raise_on_error=True):
    """
    Check for NaN and Inf values in a tensor.
    
    Args:
        tensor: PyTorch tensor to check
        name: Name of tensor for error messages
        raise_on_error: If True, raise exception on NaN/Inf; if False, return bool
    
    Returns:
        bool: True if stable (no NaN/Inf), False otherwise
    
    Raises:
        RuntimeError: If raise_on_error=True and NaN/Inf detected
    """
    has_nan = torch.isnan(tensor).any().item()
    has_inf = torch.isinf(tensor).any().item()
    
    if has_nan or has_inf:
        nan_count = torch.isnan(tensor).sum().item() if has_nan else 0
        inf_count = torch.isinf(tensor).sum().item() if has_inf else 0
        error_msg = f"Numerical instability detected in {name}: NaN={nan_count}, Inf={inf_count}"
        
        if raise_on_error:
            raise RuntimeError(error_msg)
        return False
    
    return True

def validate_loss(loss, min_loss=-1e6, max_loss=1e6, raise_on_error=True):
    """
    Validate loss value is within reasonable bounds and not NaN/Inf.
    
    Args:
        loss: Loss tensor or scalar
        min_loss: Minimum acceptable loss value (default: -1e6)
        max_loss: Maximum acceptable loss value (default: 1e6)
        raise_on_error: If True, raise exception on invalid loss; if False, return bool
    
    Returns:
        bool: True if valid, False otherwise
    
    Raises:
        RuntimeError: If raise_on_error=True and loss is invalid
    """
    if isinstance(loss, torch.Tensor):
        loss_val = loss.detach().item()
    else:
        loss_val = float(loss)
    
    # Check for NaN/Inf
    if not (min_loss <= loss_val <= max_loss) or not (loss_val == loss_val):  # NaN check
        error_msg = f"Invalid loss value: {loss_val} (expected range: [{min_loss}, {max_loss}])"
        if raise_on_error:
            raise RuntimeError(error_msg)
        return False
    
    return True

def check_gradient_explosion(model, max_grad_norm=100.0, raise_on_error=True):
    """
    Check for gradient explosion by computing gradient norm.
    
    Args:
        model: PyTorch model
        max_grad_norm: Maximum acceptable gradient norm (default: 100.0)
        raise_on_error: If True, raise exception on explosion; if False, return bool
    
    Returns:
        tuple: (grad_norm, is_exploded) where grad_norm is the gradient norm and 
               is_exploded is True if gradient norm exceeds max_grad_norm
    
    Raises:
        RuntimeError: If raise_on_error=True and gradient explosion detected
    """
    # Compute gradient norm
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float('inf'))
    grad_norm_val = grad_norm.item()
    
    is_exploded = grad_norm_val > max_grad_norm or not (grad_norm_val == grad_norm_val)  # NaN check
    
    if is_exploded:
        error_msg = f"Gradient explosion detected: grad_norm={grad_norm_val:.2f} (max={max_grad_norm})"
        if raise_on_error:
            raise RuntimeError(error_msg)
        return grad_norm_val, True
    
    return grad_norm_val, False

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
    
    def metric(self, step, metric_name, value, epoch=None):
        """Log custom metric"""
        epoch_str = f"epoch={epoch}, " if epoch is not None else ""
        msg = f"Step {step} | {epoch_str}{metric_name}={value:.4f}"
        print(self._format_message("METRIC", msg))

def calculate_wer(reference, hypothesis):
    """
    Calculate Word Error Rate (WER) between reference and hypothesis texts.
    
    Args:
        reference: Reference text (ground truth)
        hypothesis: Hypothesis text (prediction)
    
    Returns:
        wer: Word Error Rate (0.0 to 1.0+)
    """
    ref_words = reference.lower().split()
    hyp_words = hypothesis.lower().split()
    
    # Dynamic programming for edit distance
    n, m = len(ref_words), len(hyp_words)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    
    # Initialize
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    
    # Fill DP table
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref_words[i-1] == hyp_words[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
    
    # WER = edit_distance / num_reference_words
    wer = dp[n][m] / max(n, 1)
    return wer

def calculate_perplexity(loss):
    """
    Calculate perplexity from cross-entropy loss.
    
    Args:
        loss: Cross-entropy loss value
    
    Returns:
        perplexity: Perplexity value (exp(loss))
    """
    if isinstance(loss, torch.Tensor):
        loss_val = loss.detach().item()
    else:
        loss_val = float(loss)
    
    # Clamp to prevent overflow
    loss_val = min(loss_val, 10.0)  # exp(10) ≈ 22026
    return math.exp(loss_val)

