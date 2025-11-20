
import math
import os
import random
import csv
import json
import glob
import torch
from torch import nn
from typing import Tuple
from datetime import datetime
from torch.utils.data import IterableDataset, Dataset
import torchaudio
from PIL import Image
from torchvision import transforms
from einops import rearrange

# Try to use recommended torchcodec API for audio loading
# This fixes the deprecation warning about torchaudio.load()
def load_audio(path):
    """
    Load audio file using recommended API when available,
    falling back to torchaudio.load() for compatibility.
    
    Args:
        path: Path to audio file
    
    Returns:
        tuple: (audio_tensor, sample_rate)
    """
    # Try torchaudio.load_with_torchcodec first (if available in future versions)
    if hasattr(torchaudio, 'load_with_torchcodec'):
        return torchaudio.load_with_torchcodec(path)
    
    # Try torchcodec.decoders.AudioDecoder (recommended API)
    try:
        from torchcodec.decoders import AudioDecoder
        decoder = AudioDecoder(path)
        decoded = decoder()
        return decoded.audio, decoded.sample_rate
    except (ImportError, AttributeError, TypeError):
        # Fall back to standard torchaudio.load()
        return torchaudio.load(path)

# Model utilities
def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """Apply RMS normalization to input tensor."""
    return (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)) * weight

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""
    def __init__(self, d: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return rms_norm(x, self.weight)

class RoPE(nn.Module):
    """
    Rotary Positional Embeddings (RoPE).
    TM-RoPE-lite for multimodal: we simply continue positions
    across modalities and allow 2D factorization for vision/audio if desired (kept simple here).
    """
    def __init__(self, d_head: int, theta: float = 10000.0) -> None:
        super().__init__()
        self.d = d_head
        self.theta = theta

    def forward(self, q: torch.Tensor, k: torch.Tensor, pos: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply rotary positional embeddings to queries and keys.
        
        Args:
            q: (B, H, T, D) query tensor
            k: (B, H, T, D) key tensor
            pos: (T,) position indices
        
        Returns:
            Tuple of (q_with_rope, k_with_rope)
        """
        device = q.device
        d = self.d
        T = pos.shape[0]
        inv = 1.0 / (self.theta ** (torch.arange(0, d, 2, device=device).float() / d))
        freqs = torch.einsum('t,f->tf', pos.float(), inv)  # (T, d/2)
        emb = torch.cat([freqs, freqs], dim=-1)  # (T, d)
        cos = emb.cos()[None, None, :, :]
        sin = emb.sin()[None, None, :, :]
        q1 = (q * cos) + (rotate_half(q) * sin)
        k1 = (k * cos) + (rotate_half(k) * sin)
        return q1, k1

def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Helper function for RoPE: rotate half the hidden dimensions."""
    x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
    return torch.cat([-x2, x1], dim=-1)

def make_positions(T: int, device: torch.device) -> torch.Tensor:
    """Create position indices from 0 to T-1."""
    return torch.arange(T, device=device).long()

# Checkpoint utilities
def find_checkpoint(checkpoint_dir, standard_name, step_prefix, device="cpu"):
    """
    Find checkpoint file, trying standard name first, then latest step checkpoint.
    
    This is useful for inference/export when you want to use the final checkpoint
    if available, but fall back to the latest step checkpoint if training was interrupted.
    
    Args:
        checkpoint_dir: Directory to search in
        standard_name: Standard checkpoint filename (e.g., "thinker.pt")
        step_prefix: Prefix for step checkpoints (e.g., "thinker_step_")
        device: Device to load checkpoint on
    
    Returns:
        tuple: (checkpoint_path, checkpoint_data) or (None, None) if not found
    """
    if not checkpoint_dir or not os.path.exists(checkpoint_dir):
        return None, None
    
    # Try standard checkpoint first
    standard_path = os.path.join(checkpoint_dir, standard_name)
    if os.path.exists(standard_path):
        try:
            checkpoint = torch.load(standard_path, map_location=device)
            return standard_path, checkpoint
        except Exception as e:
            print(f"Warning: Could not load {standard_path}: {e}")
    
    # Try to find latest step checkpoint
    if step_prefix:
        try:
            checkpoint_files = [f for f in os.listdir(checkpoint_dir) 
                              if f.startswith(step_prefix) and f.endswith(".pt")]
            if checkpoint_files:
                # Extract step numbers and find latest
                step_numbers = []
                for f in checkpoint_files:
                    try:
                        step_num = int(f.replace(step_prefix, "").replace(".pt", ""))
                        step_numbers.append((step_num, f))
                    except:
                        continue
                
                if step_numbers:
                    step_numbers.sort(key=lambda x: x[0], reverse=True)
                    latest_path = os.path.join(checkpoint_dir, step_numbers[0][1])
                    checkpoint = torch.load(latest_path, map_location=device)
                    print(f"Using step checkpoint: {step_numbers[0][1]} (step {step_numbers[0][0]})")
                    return latest_path, checkpoint
        except Exception as e:
            print(f"Warning: Could not search for step checkpoints: {e}")
    
    return None, None

# Training utilities
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

def cleanup_old_checkpoints(save_dir, checkpoint_prefix, keep_last_n=1):
    """
    Keep only the last N step checkpoints and delete older ones.
    Always preserves *_best.pt and final checkpoints (without "step_" in name).
    
    Args:
        save_dir: Directory containing checkpoints
        checkpoint_prefix: Prefix for checkpoint files (e.g., "thinker_step_", "audio_enc_step_")
        keep_last_n: Number of most recent checkpoints to keep (default: 1)
    """
    if not os.path.exists(save_dir):
        return
    
    # Find all step checkpoints
    pattern = os.path.join(save_dir, f"{checkpoint_prefix}*.pt")
    checkpoint_files = glob.glob(pattern)
    
    # Extract step numbers
    step_checkpoints = []
    for f in checkpoint_files:
        basename = os.path.basename(f)
        # Skip best and final checkpoints
        if "best" in basename:
            continue
        try:
            step_num = int(basename.replace(checkpoint_prefix, "").replace(".pt", ""))
            step_checkpoints.append((step_num, f))
        except:
            continue
    
    # Sort by step number (newest first)
    step_checkpoints.sort(key=lambda x: x[0], reverse=True)
    
    # Delete old checkpoints (keep only the last N)
    deleted_count = 0
    for step_num, checkpoint_path in step_checkpoints[keep_last_n:]:
        try:
            os.remove(checkpoint_path)
            deleted_count += 1
        except Exception as e:
            print(f"  Warning: Could not delete {checkpoint_path}: {e}")
    
    if deleted_count > 0:
        print(f"  Cleaned up {deleted_count} old checkpoint(s), keeping last {min(keep_last_n, len(step_checkpoints))}")

def reload_from_last_checkpoint(save_dir, checkpoint_prefix, device, logger, model, opt=None, scheduler=None, scaler=None):
    """
    Reload from the last saved checkpoint in the save directory.
    
    Args:
        save_dir: Directory containing checkpoints
        checkpoint_prefix: Prefix for checkpoint files (e.g., "thinker_step_", "omni_step_")
        device: Device to load checkpoint on
        logger: Logger instance for logging
        model: Model to load state into
        opt: Optimizer to load state into (optional)
        scheduler: Scheduler to load state into (optional)
        scaler: GradScaler to load state into (optional)
    
    Returns:
        int: step from checkpoint, or 0 if no checkpoint found
    """
    if not os.path.exists(save_dir):
        logger.error(f"Save directory does not exist: {save_dir}")
        return 0
    
    checkpoint_files = [f for f in os.listdir(save_dir) if f.startswith(checkpoint_prefix) and f.endswith(".pt")]
    if not checkpoint_files:
        logger.error(f"No checkpoint files found with prefix '{checkpoint_prefix}' in {save_dir}")
        return 0
    
    # Extract step numbers and find latest
    step_numbers = []
    for f in checkpoint_files:
        try:
            step_num = int(f.replace(checkpoint_prefix, "").replace(".pt", ""))
            step_numbers.append((step_num, f))
        except:
            continue
    
    if not step_numbers:
        logger.error(f"Could not parse step numbers from checkpoint files")
        return 0
    
    step_numbers.sort(key=lambda x: x[0], reverse=True)
    last_checkpoint = os.path.join(save_dir, step_numbers[0][1])
    step = step_numbers[0][0]
    
    logger.error(f"NaN detected in attention. Reloading from checkpoint: {last_checkpoint}")
    
    try:
        checkpoint = torch.load(last_checkpoint, map_location=device)
        
        # Load model state
        if isinstance(checkpoint, dict):
            # Try different possible keys for model state
            if "model" in checkpoint:
                model.load_state_dict(checkpoint["model"])
            elif "thinker" in checkpoint:
                model.load_state_dict(checkpoint["thinker"])
            else:
                # Assume it's a model state dict
                model.load_state_dict(checkpoint)
            
            # Load optimizer state
            if opt is not None and "optimizer" in checkpoint:
                opt.load_state_dict(checkpoint["optimizer"])
            
            # Load scheduler state
            if scheduler is not None and "scheduler" in checkpoint:
                scheduler.load_state_dict(checkpoint["scheduler"])
            
            # Load scaler state
            if scaler is not None and "scaler" in checkpoint:
                scaler.load_state_dict(checkpoint["scaler"])
            
            # Get step
            loaded_step = checkpoint.get("step", step)
            
            logger.info(f"Successfully reloaded from step {loaded_step}")
            return loaded_step
        else:
            # Legacy format - just model weights
            model.load_state_dict(checkpoint)
            logger.info(f"Loaded model weights from checkpoint (legacy format)")
            return step
    except Exception as e:
        logger.error(f"Failed to reload from checkpoint: {e}")
        return 0

def load_checkpoint(save_dir, checkpoint_prefix, device, logger, state_dict_loaders=None):
    """
    Load checkpoint from save directory and return step number.
    
    Args:
        save_dir: Directory containing checkpoints
        checkpoint_prefix: Prefix for checkpoint files (e.g., "thinker_step_", "talker_step_")
        device: Device to load checkpoint on
        logger: Logger instance for logging
        state_dict_loaders: Dict mapping checkpoint keys to (object, load_func) tuples.
                          Example: {"model": (model, model.load_state_dict), 
                                   "optimizer": (opt, opt.load_state_dict)}
    
    Returns:
        tuple: (step, checkpoint_path) or (0, None) if no checkpoint found
    """
    if not os.path.exists(save_dir):
        return 0, None
    
    checkpoint_files = [f for f in os.listdir(save_dir) if f.startswith(checkpoint_prefix) and f.endswith(".pt")]
    if not checkpoint_files:
        return 0, None
    
    # Extract step numbers and find latest
    step_numbers = []
    for f in checkpoint_files:
        try:
            step_num = int(f.replace(checkpoint_prefix, "").replace(".pt", ""))
            step_numbers.append((step_num, f))
        except:
            continue
    
    if not step_numbers:
        return 0, None
    
    step_numbers.sort(key=lambda x: x[0], reverse=True)
    resume_from = os.path.join(save_dir, step_numbers[0][1])
    step = step_numbers[0][0]
    logger.info(f"Found checkpoint at step {step}, resuming from: {resume_from}")
    
    try:
        checkpoint = torch.load(resume_from, map_location=device)
        
        if isinstance(checkpoint, dict):
            # Load state dicts using provided loaders
            if state_dict_loaders:
                for key, (obj, load_func) in state_dict_loaders.items():
                    if key in checkpoint:
                        load_func(checkpoint[key])
            
            # Get step from checkpoint if available
            if "step" in checkpoint:
                step = checkpoint["step"]
            
            logger.info(f"Resumed from step {step}")
        else:
            # Legacy checkpoint format - try to load as model if only one loader provided
            if state_dict_loaders and len(state_dict_loaders) == 1:
                key = list(state_dict_loaders.keys())[0]
                obj, load_func = state_dict_loaders[key]
                load_func(checkpoint)
            logger.info(f"Loaded model weights from checkpoint (legacy format)")
        
        return step, resume_from
    except Exception as e:
        logger.error(f"Failed to load checkpoint: {e}")
        return 0, None

def setup_resume_data_loading(train_ds, step, batch_size, logger, train_dl_kwargs):
    """
    Setup skip_samples for dataset when resuming training.
    
    Args:
        train_ds: Training dataset (may be SubsetDataset from random_split)
        step: Current step number (0 if not resuming)
        batch_size: Batch size for calculating skip_samples
        logger: Logger instance
        train_dl_kwargs: Dict of kwargs to recreate DataLoader (batch_size, num_workers, etc.)
            Note: Do not include 'shuffle' - IterableDatasets handle shuffling internally
    
    Returns:
        DataLoader: Recreated DataLoader with skip_samples applied
    """
    if step > 0:
        # Calculate approximate samples to skip: step * batch_size
        skip_samples = step * batch_size
        # Access underlying dataset from SubsetDataset (created by random_split)
        underlying_ds = train_ds.dataset if hasattr(train_ds, 'dataset') else train_ds
        underlying_ds.skip_samples = skip_samples
        logger.info(f"Dataset: will skip approximately {skip_samples} samples when resuming")
        
        # Recreate DataLoader so workers pick up the new skip_samples value
        from torch.utils.data import DataLoader
        train_dl_kwargs['batch_size'] = batch_size
        return DataLoader(train_ds, **train_dl_kwargs)
    
    return None  # No need to recreate if not resuming

def calculate_resume_position(step, steps_per_epoch):
    """
    Calculate starting epoch and batch index from global step.
    
    Args:
        step: Global step number
        steps_per_epoch: Number of steps per epoch
    
    Returns:
        tuple: (start_epoch, start_batch_idx)
    """
    if step > 0:
        start_epoch = step // steps_per_epoch
        start_batch_idx = step % steps_per_epoch
        return start_epoch, start_batch_idx
    return 0, 0

# Collate functions for DataLoader
def collate_mel_fn(batch, max_mel_length=None):
    """
    Collate function that pads all mel spectrograms to a fixed maximum length.
    This ensures uniform batch sizes for CUDA graphs compilation.
    
    Args:
        batch: List of mel spectrograms
        max_mel_length: Fixed maximum length to pad to. If None, uses batch max (not recommended for CUDA graphs)
    
    Returns:
        torch.Tensor: Stacked and padded mel spectrograms of shape (B, T, n_mels)
    """
    n_mels = batch[0].shape[1]
    
    # Use fixed max length if provided, otherwise use batch max
    if max_mel_length is not None:
        max_len = max_mel_length
    else:
        max_len = max(m.shape[0] for m in batch)
    
    padded = []
    for m in batch:
        current_len = m.shape[0]
        if current_len > max_len:
            # Truncate if longer than max (shouldn't happen with proper config)
            m = m[:max_len]
            current_len = max_len
        
        pad_len = max_len - current_len
        if pad_len > 0:
            m = torch.cat([m, torch.zeros(pad_len, n_mels)], dim=0)
        padded.append(m)
    return torch.stack(padded)

def collate_mel_text_fn(batch, max_mel_length=None):
    """
    Collate function that pads mel spectrograms and returns text list.
    This ensures uniform batch sizes for CUDA graphs compilation.
    
    Args:
        batch: List of (mel, text) tuples
        max_mel_length: Fixed maximum length to pad to. If None, uses batch max (not recommended for CUDA graphs)
    
    Returns:
        tuple: (padded_mels, texts) where padded_mels is (B, T, n_mels) and texts is a list
    """
    mels, texts = zip(*batch)
    n_mels = mels[0].shape[1]
    
    # Use fixed max length if provided, otherwise use batch max
    if max_mel_length is not None:
        max_len = max_mel_length
    else:
        max_len = max(m.shape[0] for m in mels)
    
    padded_mels = []
    for m in mels:
        current_len = m.shape[0]
        if current_len > max_len:
            # Truncate if longer than max (shouldn't happen with proper config)
            m = m[:max_len]
            current_len = max_len
        
        pad_len = max_len - current_len
        if pad_len > 0:
            m = torch.cat([m, torch.zeros(pad_len, n_mels)], dim=0)
        padded_mels.append(m)
    return torch.stack(padded_mels), list(texts)

def collate_mel_audio_fn(batch, max_mel_length=None, max_audio_length=None):
    """
    Collate function for mel spectrograms and audio pairs.
    Pads both mels and audios to fixed lengths for uniform batch sizes.
    
    Args:
        batch: List of (mel, audio) tuples
        max_mel_length: Fixed maximum mel length to pad to. If None, uses batch max
        max_audio_length: Fixed maximum audio length to pad to. If None, uses batch max
    
    Returns:
        tuple: (padded_mels, padded_audios) both as torch.Tensor
    """
    mels, audios = zip(*batch)
    
    # Pad mel spectrograms
    n_mels = mels[0].shape[1]
    if max_mel_length is not None:
        max_mel_len = max_mel_length
    else:
        max_mel_len = max(m.shape[0] for m in mels)
    
    padded_mels = []
    for m in mels:
        current_len = m.shape[0]
        if current_len > max_mel_len:
            m = m[:max_mel_len]
            current_len = max_mel_len
        pad_len = max_mel_len - current_len
        if pad_len > 0:
            m = torch.cat([m, torch.zeros(pad_len, n_mels)], dim=0)
        padded_mels.append(m)
    
    # Pad audio waveforms
    if max_audio_length is not None:
        max_audio_len = max_audio_length
    else:
        max_audio_len = max(a.shape[0] for a in audios)
    
    padded_audios = []
    for a in audios:
        current_len = a.shape[0]
        if current_len > max_audio_len:
            a = a[:max_audio_len]
            current_len = max_audio_len
        pad_len = max_audio_len - current_len
        if pad_len > 0:
            a = torch.cat([a, torch.zeros(pad_len)], dim=0)
        padded_audios.append(a)
    
    return torch.stack(padded_mels), torch.stack(padded_audios)

class ValidationSkipSamplesContext:
    """Context manager to temporarily reset skip_samples for validation."""
    def __init__(self, train_ds):
        self.train_ds = train_ds
        self.underlying_ds = train_ds.dataset if hasattr(train_ds, 'dataset') else train_ds
        self.original_skip_samples = None
    
    def __enter__(self):
        if hasattr(self.underlying_ds, 'skip_samples'):
            self.original_skip_samples = self.underlying_ds.skip_samples
            self.underlying_ds.skip_samples = 0
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if hasattr(self.underlying_ds, 'skip_samples') and self.original_skip_samples is not None:
            self.underlying_ds.skip_samples = self.original_skip_samples
        return False

# Dataset classes
class TextDataset(IterableDataset):
    """Streaming dataset: sequential I/O, low memory, efficient resuming."""
    def __init__(self, path, tokenizer, ctx, shuffle_buffer_size=10000, seed=None, skip_samples=0):
        self.path, self.tok, self.ctx = path, tokenizer, ctx
        self.shuffle_buffer_size, self.seed, self.skip_samples = shuffle_buffer_size, seed, skip_samples
        self._num_lines = None
    
    def get_length(self):
        """Count lines (expensive, cached after first call)"""
        if self._num_lines is None:
            self._num_lines = sum(1 for _ in open(self.path, 'r', encoding='utf-8', errors='ignore'))
        return self._num_lines
    
    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        num_workers = worker_info.num_workers if worker_info else 1
        worker_id = worker_info.id if worker_info else 0
        rng = random.Random(self.seed + worker_id if self.seed else None)
        buffer = []
        
        val_split = getattr(self, '_val_split', None)
        val_mode = getattr(self, '_val_mode', False)
        
        # Distribute skip_samples across workers
        worker_skip = (self.skip_samples // num_workers) + (1 if worker_id < (self.skip_samples % num_workers) else 0)
        skipped = 0
        
        with open(self.path, 'r', encoding='utf-8', errors='ignore', buffering=8192*1024) as f:
            for idx, line in enumerate(f):
                # Worker sharding
                if idx % num_workers != worker_id:
                    continue
                
                # Train/val split
                if val_split and (hash((idx, self.seed)) % 100 < val_split * 100) != val_mode:
                    continue
                
                text = line.strip()
                if not text:
                    continue
                
                # Skip samples when resuming
                if skipped < worker_skip:
                    skipped += 1
                    continue
                
                # Tokenize and create tensors
                ids = [1] + self.tok.encode(text)[:self.ctx-1]  # BOS=1
                x = torch.tensor(ids + [0] * (self.ctx - len(ids)), dtype=torch.long)
                y = torch.cat([x[1:], torch.tensor([0], dtype=torch.long)])  # Shift for next-token prediction
                
                # Buffer-based shuffling
                if self.shuffle_buffer_size > 0:
                    buffer.append((x, y))
                    if len(buffer) >= self.shuffle_buffer_size:
                        yield buffer.pop(rng.randint(0, len(buffer) - 1))
                else:
                    yield x, y
        
        # Yield remaining buffer items
        if buffer:
            rng.shuffle(buffer)
            yield from buffer
        
        # Reset skip_samples to 0 after first iteration (only use for resuming)
        # This allows subsequent epochs to start from the beginning
        if self.skip_samples > 0:
            print(f"Dataset exhausted: resetting skip_samples from {self.skip_samples} to 0 for next epoch")
            self.skip_samples = 0

class ASRDataset(IterableDataset):
    """Streaming dataset: sequential I/O, low memory, efficient resuming."""
    def __init__(self, csv_path, sr=16000, n_mels=128, cfg=None, shuffle_buffer_size=10000, seed=None, skip_samples=0):
        self.csv_path, self.sr = csv_path, sr
        self.shuffle_buffer_size, self.seed, self.skip_samples = shuffle_buffer_size, seed, skip_samples
        self.melspec = torchaudio.transforms.MelSpectrogram(sample_rate=sr, n_fft=1024, hop_length=160, win_length=400, n_mels=n_mels)
        self._num_rows = None
        # Error handling configuration
        self.warn_on_errors = cfg.get("warn_on_dataset_errors", False) if cfg else False
        self._error_counts = {"missing_file": 0, "load_error": 0, "empty_text": 0}
        self._first_error_logged = False
    
    def get_length(self):
        """Count CSV rows (expensive, cached after first call)"""
        if self._num_rows is None:
            import csv
            with open(self.csv_path, 'r', encoding='utf-8', errors='ignore') as f:
                reader = csv.DictReader(f)
                self._num_rows = sum(1 for _ in reader)
        return self._num_rows
    
    def get_error_stats(self):
        """Get statistics about skipped files due to errors."""
        return self._error_counts.copy()
    
    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        num_workers = worker_info.num_workers if worker_info else 1
        worker_id = worker_info.id if worker_info else 0
        rng = random.Random(self.seed + worker_id if self.seed else None)
        buffer = []
        
        val_split = getattr(self, '_val_split', None)
        val_mode = getattr(self, '_val_mode', False)
        
        # Reset error tracking for this iteration
        self._error_counts = {"missing_file": 0, "load_error": 0, "empty_text": 0}
        self._first_error_logged = False
        
        # Distribute skip_samples across workers
        worker_skip = (self.skip_samples // num_workers) + (1 if worker_id < (self.skip_samples % num_workers) else 0)
        skipped = 0
        
        with open(self.csv_path, 'r', encoding='utf-8', errors='ignore', buffering=8192*1024) as f:
            reader = csv.DictReader(f)
            for idx, row in enumerate(reader):
                # Worker sharding
                if idx % num_workers != worker_id:
                    continue
                
                # Train/val split
                if val_split and (hash((idx, self.seed)) % 100 < val_split * 100) != val_mode:
                    continue
                
                # Skip samples when resuming
                if skipped < worker_skip:
                    skipped += 1
                    continue
                
                # Get file path and text
                wav_path = row.get("wav", "").strip()
                text = row.get("text", "").strip()
                
                # Check if file path is provided
                if not wav_path:
                    if self.warn_on_errors and not self._first_error_logged:
                        print(f"Warning: Empty file path in row {idx} (worker {worker_id})")
                        self._first_error_logged = True
                    self._error_counts["load_error"] += 1
                    continue
                
                # Check if file exists (faster than trying to load)
                if not os.path.exists(wav_path):
                    if self.warn_on_errors and not self._first_error_logged:
                        print(f"Warning: File not found in row {idx} (worker {worker_id}): {wav_path}")
                        self._first_error_logged = True
                    self._error_counts["missing_file"] += 1
                    continue
                
                # Check if text is empty
                if not text:
                    if self.warn_on_errors and not self._first_error_logged:
                        print(f"Warning: Empty text in row {idx} (worker {worker_id}): {wav_path}")
                        self._first_error_logged = True
                    self._error_counts["empty_text"] += 1
                    # Still try to load audio, but text will be empty
                
                try:
                    wav, sr = load_audio(wav_path)
                    if sr != self.sr:
                        wav = torchaudio.transforms.Resample(sr, self.sr)(wav)
                    mel = self.melspec(wav)[0].T
                    
                    # Buffer-based shuffling
                    if self.shuffle_buffer_size > 0:
                        buffer.append((mel, text))
                        if len(buffer) >= self.shuffle_buffer_size:
                            yield buffer.pop(rng.randint(0, len(buffer) - 1))
                    else:
                        yield mel, text
                except Exception as e:
                    if self.warn_on_errors and not self._first_error_logged:
                        print(f"Warning: Error loading audio in row {idx} (worker {worker_id}): {wav_path}")
                        print(f"  Error: {str(e)}")
                        self._first_error_logged = True
                    self._error_counts["load_error"] += 1
                    continue
        
        # Yield remaining buffer items
        if buffer:
            rng.shuffle(buffer)
            yield from buffer
        
        # Reset skip_samples to 0 after first iteration (only use for resuming)
        # This allows subsequent epochs to start from the beginning
        if self.skip_samples > 0:
            print(f"Dataset exhausted: resetting skip_samples from {self.skip_samples} to 0 for next epoch")
            self.skip_samples = 0

class OCRDataset(IterableDataset):
    """Streaming dataset: sequential I/O, low memory, efficient resuming."""
    def __init__(self, csv_path, image_root, img_size=224, cfg=None, shuffle_buffer_size=10000, seed=None, skip_samples=0):
        self.csv_path, self.image_root, self.img_size = csv_path, image_root, img_size
        self.shuffle_buffer_size, self.seed, self.skip_samples = shuffle_buffer_size, seed, skip_samples
        self.tf = transforms.Compose([transforms.Resize((img_size, img_size)), transforms.ToTensor()])
        self.char_to_idx, self.idx_to_char = {}, {}
        self._num_rows = None
        self._build_vocab(csv_path)
    
    def get_length(self):
        """Count CSV rows (expensive, cached after first call)"""
        if self._num_rows is None:
            with open(self.csv_path, 'r', encoding='utf-8', errors='ignore') as f:
                reader = csv.DictReader(f)
                self._num_rows = sum(1 for _ in reader)
        return self._num_rows
    
    def _build_vocab(self, csv_path):
        """Build character vocabulary from CSV."""
        print("Building character vocabulary from OCR dataset...")
        chars = set()
        
        with open(csv_path, 'r', encoding='utf-8', errors='ignore') as f:
            reader = csv.DictReader(f)
            for row_idx, row in enumerate(reader):
                chars.update(row.get("text", "") or row.get("label", "") or row.get("text_label", ""))
                if (row_idx + 1) % 10000 == 0:
                    print(f"  Processed {row_idx+1:,} rows...")
        
        # Build vocabulary
        self.char_to_idx = {'<PAD>': 0, '<BOS>': 1, '<EOS>': 2, '<UNK>': 3}
        self.idx_to_char = {0: '<PAD>', 1: '<BOS>', 2: '<EOS>', 3: '<UNK>'}
        for char in sorted(chars):
            if char not in self.char_to_idx:
                idx = len(self.char_to_idx)
                self.char_to_idx[char] = idx
                self.idx_to_char[idx] = char
        
        print(f"Vocabulary size: {len(self.char_to_idx)} characters")
    
    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        num_workers = worker_info.num_workers if worker_info else 1
        worker_id = worker_info.id if worker_info else 0
        rng = random.Random(self.seed + worker_id if self.seed else None)
        buffer = []
        
        val_split = getattr(self, '_val_split', None)
        val_mode = getattr(self, '_val_mode', False)
        
        # Distribute skip_samples across workers
        worker_skip = (self.skip_samples // num_workers) + (1 if worker_id < (self.skip_samples % num_workers) else 0)
        skipped = 0
        
        with open(self.csv_path, 'r', encoding='utf-8', errors='ignore', buffering=8192*1024) as f:
            reader = csv.DictReader(f)
            for idx, row in enumerate(reader):
                # Worker sharding
                if idx % num_workers != worker_id:
                    continue
                
                # Train/val split
                if val_split and (hash((idx, self.seed)) % 100 < val_split * 100) != val_mode:
                    continue
                
                # Skip samples when resuming
                if skipped < worker_skip:
                    skipped += 1
                    continue
                
                img_path = row.get("image") or row.get("img")
                if not img_path:
                    continue
                text = row.get("text", "") or row.get("label", "") or row.get("text_label", "")
                
                try:
                    full_img_path = os.path.join(self.image_root, img_path) if not os.path.isabs(img_path) else img_path
                    img = Image.open(full_img_path).convert("RGB")
                    text_ids = [self.char_to_idx.get(c, self.char_to_idx['<UNK>']) for c in text] or [self.char_to_idx['<UNK>']]
                    result = (self.tf(img), [self.char_to_idx['<BOS>']] + text_ids + [self.char_to_idx['<EOS>']])
                    
                    # Buffer-based shuffling
                    if self.shuffle_buffer_size > 0:
                        buffer.append(result)
                        if len(buffer) >= self.shuffle_buffer_size:
                            yield buffer.pop(rng.randint(0, len(buffer) - 1))
                    else:
                        yield result
                except Exception:
                    continue
        
        # Yield remaining buffer items
        if buffer:
            rng.shuffle(buffer)
            yield from buffer
        
        # Reset skip_samples to 0 after first iteration (only use for resuming)
        # This allows subsequent epochs to start from the beginning
        if self.skip_samples > 0:
            print(f"Dataset exhausted: resetting skip_samples from {self.skip_samples} to 0 for next epoch")
            self.skip_samples = 0

class TTSDataset(IterableDataset):
    """Streaming dataset: sequential I/O, low memory, efficient resuming."""
    def __init__(self, csv_path, sr=16000, n_mels=128, frame_ms=80, cfg=None, shuffle_buffer_size=10000, seed=None, skip_samples=0):
        self.csv_path, self.sr = csv_path, sr
        self.shuffle_buffer_size, self.seed, self.skip_samples = shuffle_buffer_size, seed, skip_samples
        hop_length = int(sr * frame_ms / 1000)
        win_length = min(1024, hop_length * 4)
        self.melspec = torchaudio.transforms.MelSpectrogram(sample_rate=sr, n_fft=1024, hop_length=hop_length, win_length=win_length, n_mels=n_mels)
        self.frame = int(sr * 0.08)
        self._num_rows = None
    
    def get_length(self):
        """Count CSV rows (expensive, cached after first call)"""
        if self._num_rows is None:
            with open(self.csv_path, 'r', encoding='utf-8', errors='ignore') as f:
                reader = csv.DictReader(f)
                self._num_rows = sum(1 for _ in reader)
        return self._num_rows
    
    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        num_workers = worker_info.num_workers if worker_info else 1
        worker_id = worker_info.id if worker_info else 0
        rng = random.Random(self.seed + worker_id if self.seed else None)
        buffer = []
        
        val_split = getattr(self, '_val_split', None)
        val_mode = getattr(self, '_val_mode', False)
        
        # Distribute skip_samples across workers
        worker_skip = (self.skip_samples // num_workers) + (1 if worker_id < (self.skip_samples % num_workers) else 0)
        skipped = 0
        
        with open(self.csv_path, 'r', encoding='utf-8', errors='ignore', buffering=8192*1024) as f:
            reader = csv.DictReader(f)
            for idx, row in enumerate(reader):
                # Worker sharding
                if idx % num_workers != worker_id:
                    continue
                
                # Train/val split
                if val_split and (hash((idx, self.seed)) % 100 < val_split * 100) != val_mode:
                    continue
                
                # Skip samples when resuming
                if skipped < worker_skip:
                    skipped += 1
                    continue
                
                try:
                    wav, sr = load_audio(row["wav"])
                    if sr != self.sr:
                        wav = torchaudio.transforms.Resample(sr, self.sr)(wav)
                    mel = self.melspec(wav)[0].T
                    
                    # Buffer-based shuffling
                    if self.shuffle_buffer_size > 0:
                        buffer.append(mel)
                        if len(buffer) >= self.shuffle_buffer_size:
                            yield buffer.pop(rng.randint(0, len(buffer) - 1))
                    else:
                        yield mel
                except Exception:
                    continue
        
        # Yield remaining buffer items
        if buffer:
            rng.shuffle(buffer)
            yield from buffer
        
        # Reset skip_samples to 0 after first iteration (only use for resuming)
        # This allows subsequent epochs to start from the beginning
        if self.skip_samples > 0:
            print(f"Dataset exhausted: resetting skip_samples from {self.skip_samples} to 0 for next epoch")
            self.skip_samples = 0

class ImgCapDataset(IterableDataset):
    """Streaming dataset: sequential I/O, low memory, efficient resuming."""
    def __init__(self, manifest, image_root, img_size=224, shuffle_buffer_size=10000, seed=None, skip_samples=0):
        self.manifest_path, self.root = manifest, image_root
        self.shuffle_buffer_size, self.seed, self.skip_samples = shuffle_buffer_size, seed, skip_samples
        self.tf = transforms.Compose([transforms.Resize((img_size, img_size)), transforms.ToTensor()])
        self._num_items = None
    
    def get_length(self):
        """Count JSON manifest items (expensive, cached after first call)"""
        if self._num_items is None:
            with open(self.manifest_path, 'r', encoding='utf-8') as f:
                items = json.load(f)
                self._num_items = len(items)
        return self._num_items
    
    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        num_workers = worker_info.num_workers if worker_info else 1
        worker_id = worker_info.id if worker_info else 0
        rng = random.Random(self.seed + worker_id if self.seed else None)
        buffer = []
        
        val_split = getattr(self, '_val_split', None)
        val_mode = getattr(self, '_val_mode', False)
        
        # Distribute skip_samples across workers
        worker_skip = (self.skip_samples // num_workers) + (1 if worker_id < (self.skip_samples % num_workers) else 0)
        skipped = 0
        
        with open(self.manifest_path, 'r', encoding='utf-8') as f:
            items = json.load(f)
        
        for idx, it in enumerate(items):
            # Worker sharding
            if idx % num_workers != worker_id:
                continue
            
            # Train/val split
            if val_split and (hash((idx, self.seed)) % 100 < val_split * 100) != val_mode:
                continue
            
            # Skip samples when resuming
            if skipped < worker_skip:
                skipped += 1
                continue
            
            try:
                img = Image.open(os.path.join(self.root, it["image"])).convert("RGB")
                result = (self.tf(img), it["caption"])
                
                # Buffer-based shuffling
                if self.shuffle_buffer_size > 0:
                    buffer.append(result)
                    if len(buffer) >= self.shuffle_buffer_size:
                        yield buffer.pop(rng.randint(0, len(buffer) - 1))
                else:
                    yield result
            except Exception:
                continue
        
        # Yield remaining buffer items
        if buffer:
            rng.shuffle(buffer)
            yield from buffer
        
        # Reset skip_samples to 0 after first iteration (only use for resuming)
        # This allows subsequent epochs to start from the beginning
        if self.skip_samples > 0:
            print(f"Dataset exhausted: resetting skip_samples from {self.skip_samples} to 0 for next epoch")
            self.skip_samples = 0

class VocoderDataset(IterableDataset):
    """Streaming dataset: sequential I/O, low memory, efficient resuming."""
    def __init__(self, csv_path, sr=16000, n_mels=128, n_fft=1024, hop_length=256, cfg=None, shuffle_buffer_size=10000, seed=None, skip_samples=0):
        self.csv_path, self.sr, self.n_mels = csv_path, sr, n_mels
        self.shuffle_buffer_size, self.seed, self.skip_samples = shuffle_buffer_size, seed, skip_samples
        self.max_audio_length = cfg.get("max_audio_length", None) if cfg else None
        self.melspec = torchaudio.transforms.MelSpectrogram(sample_rate=sr, n_fft=n_fft, hop_length=hop_length, win_length=n_fft, n_mels=n_mels, fmin=0.0, fmax=sr / 2.0)
        self._num_rows = None
    
    def get_length(self):
        """Count CSV rows (expensive, cached after first call)"""
        if self._num_rows is None:
            with open(self.csv_path, 'r', encoding='utf-8', errors='ignore') as f:
                reader = csv.DictReader(f)
                self._num_rows = sum(1 for _ in reader)
        return self._num_rows
    
    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        num_workers = worker_info.num_workers if worker_info else 1
        worker_id = worker_info.id if worker_info else 0
        rng = random.Random(self.seed + worker_id if self.seed else None)
        buffer = []
        
        val_split = getattr(self, '_val_split', None)
        val_mode = getattr(self, '_val_mode', False)
        
        # Distribute skip_samples across workers
        worker_skip = (self.skip_samples // num_workers) + (1 if worker_id < (self.skip_samples % num_workers) else 0)
        skipped = 0
        
        with open(self.csv_path, 'r', encoding='utf-8', errors='ignore', buffering=8192*1024) as f:
            reader = csv.DictReader(f)
            for idx, row in enumerate(reader):
                # Worker sharding
                if idx % num_workers != worker_id:
                    continue
                
                # Train/val split
                if val_split and (hash((idx, self.seed)) % 100 < val_split * 100) != val_mode:
                    continue
                
                # Skip samples when resuming
                if skipped < worker_skip:
                    skipped += 1
                    continue
                
                path = row.get("wav") or row.get("audio")
                if not path:
                    continue
                
                try:
                    audio, sr = load_audio(path)
                    if sr != self.sr:
                        audio = torchaudio.transforms.Resample(sr, self.sr)(audio)
                    if audio.shape[0] > 1:
                        audio = audio.mean(dim=0, keepdim=True)
                    audio = audio.squeeze(0)
                    
                    if self.max_audio_length and audio.shape[0] > self.max_audio_length:
                        start_idx = torch.randint(0, audio.shape[0] - self.max_audio_length + 1, (1,)).item() if self.max_audio_length < audio.shape[0] else 0
                        audio = audio[start_idx:start_idx + self.max_audio_length]
                    
                    mel = self.melspec(audio.unsqueeze(0))[0].T
                    mel_min, mel_max = mel.min(), mel.max()
                    if mel_max > mel_min + 1e-6:
                        mel = (mel - mel_min) / (mel_max - mel_min + 1e-8)
                    
                    result = (mel, audio)
                    
                    # Buffer-based shuffling
                    if self.shuffle_buffer_size > 0:
                        buffer.append(result)
                        if len(buffer) >= self.shuffle_buffer_size:
                            yield buffer.pop(rng.randint(0, len(buffer) - 1))
                    else:
                        yield result
                except Exception:
                    continue
        
        # Yield remaining buffer items
        if buffer:
            rng.shuffle(buffer)
            yield from buffer
        
        # Reset skip_samples to 0 after first iteration (only use for resuming)
        # This allows subsequent epochs to start from the beginning
        if self.skip_samples > 0:
            print(f"Dataset exhausted: resetting skip_samples from {self.skip_samples} to 0 for next epoch")
            self.skip_samples = 0

class MixDataset(IterableDataset):
    """Streaming dataset: sequential I/O, low memory, efficient resuming."""
    def __init__(self, text_path, image_manifest, image_root, asr_csv, ctx=1024, shuffle_buffer_size=10000, seed=None, skip_samples=0):
        self.text_path, self.image_manifest_path, self.image_root, self.asr_csv_path, self.ctx = text_path, image_manifest, image_root, asr_csv, ctx
        self.shuffle_buffer_size, self.seed, self.skip_samples = shuffle_buffer_size, seed, skip_samples
        self.tf = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
        self._num_items = None
    
    def get_length(self):
        """Count max of text lines, images, and ASR rows (expensive, cached after first call)"""
        if self._num_items is None:
            # Count text lines
            text_count = sum(1 for _ in open(self.text_path, 'r', encoding='utf-8', errors='ignore'))
            # Count images
            with open(self.image_manifest_path, 'r', encoding='utf-8') as f:
                images = json.load(f)
                image_count = len(images)
            # Count ASR rows
            with open(self.asr_csv_path, 'r', encoding='utf-8', errors='ignore') as f:
                reader = csv.DictReader(f)
                asr_count = sum(1 for _ in reader)
            self._num_items = max(text_count, image_count, asr_count)
        return self._num_items
    
    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        num_workers = worker_info.num_workers if worker_info else 1
        worker_id = worker_info.id if worker_info else 0
        rng = random.Random(self.seed + worker_id if self.seed else None)
        buffer = []
        
        val_split = getattr(self, '_val_split', None)
        val_mode = getattr(self, '_val_mode', False)
        
        # Distribute skip_samples across workers
        worker_skip = (self.skip_samples // num_workers) + (1 if worker_id < (self.skip_samples % num_workers) else 0)
        skipped = 0
        
        # Load image manifest
        with open(self.image_manifest_path, 'r', encoding='utf-8') as f:
            images = json.load(f)
        
        # Create iterators for text and ASR
        text_lines = []
        with open(self.text_path, 'r', encoding='utf-8', errors='ignore') as f:
            text_lines = [line.strip() for line in f if line.strip()]
        
        asr_rows = []
        with open(self.asr_csv_path, 'r', encoding='utf-8', errors='ignore') as f:
            reader = csv.DictReader(f)
            asr_rows = list(reader)
        
        max_len = max(len(text_lines), len(images), len(asr_rows))
        
        for idx in range(max_len):
            # Worker sharding
            if idx % num_workers != worker_id:
                continue
            
            # Train/val split
            if val_split and (hash((idx, self.seed)) % 100 < val_split * 100) != val_mode:
                continue
            
            # Skip samples when resuming
            if skipped < worker_skip:
                skipped += 1
                continue
            
            it = {}
            
            # Load text
            if idx < len(text_lines):
                it["text"] = text_lines[idx]
            else:
                it["text"] = "Describe the image or audio."
            
            # Load image
            if idx < len(images):
                img_item = images[idx]
                img_path = os.path.join(self.image_root, img_item["image"])
                it["image"], it["caption"] = img_path, img_item["caption"]
            
            # Load ASR
            if idx < len(asr_rows):
                row = asr_rows[idx]
                it["audio"], it["trans"] = row["wav"], row["text"]
            
            # Buffer-based shuffling
            if self.shuffle_buffer_size > 0:
                buffer.append(it)
                if len(buffer) >= self.shuffle_buffer_size:
                    yield buffer.pop(rng.randint(0, len(buffer) - 1))
            else:
                yield it
        
        # Yield remaining buffer items
        if buffer:
            rng.shuffle(buffer)
            yield from buffer
        
        # Reset skip_samples to 0 after first iteration (only use for resuming)
        # This allows subsequent epochs to start from the beginning
        if self.skip_samples > 0:
            print(f"Dataset exhausted: resetting skip_samples from {self.skip_samples} to 0 for next epoch")
            self.skip_samples = 0
