import math
import os
import random
import json
import hashlib
import tempfile
from datetime import datetime
from typing import Any

import torch


class EMA:
    """Exponential Moving Average of model parameters."""

    def __init__(self, model, decay=0.999, device=None):
        self.model = model
        self.decay = decay
        self.device = device if device else next(model.parameters()).device
        self.shadow = {}
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone().to(self.device)

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name].clone()

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name].clone()
        self.backup = {}

    def state_dict(self):
        return {"decay": self.decay, "shadow": self.shadow}

    def load_state_dict(self, state_dict):
        self.decay = state_dict["decay"]
        self.shadow = state_dict["shadow"]


def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_cuda():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.fp32_precision = "tf32"
        torch.backends.cudnn.conv.fp32_precision = "tf32"
    return device


class LRSpike:
    def __init__(self, spike_multiplier=5.0, spike_duration=50, consecutive_increases=2):
        self.spike_multiplier = spike_multiplier
        self.spike_duration = spike_duration
        self.consecutive_increases = consecutive_increases
        self.val_loss_history = []
        self.consecutive_increase_count = 0
        self.spike_active = False
        self.spike_steps_remaining = 0
        self.original_lrs = []

    def check_and_spike(self, current_val_loss, optimizer, logger=None):
        self.val_loss_history.append(current_val_loss)
        if len(self.val_loss_history) >= 2:
            if current_val_loss > self.val_loss_history[-2]:
                self.consecutive_increase_count += 1
                if logger:
                    logger.info(
                        f"Validation loss increased: {self.val_loss_history[-2]:.4f} -> {current_val_loss:.4f} "
                        f"({self.consecutive_increase_count}/{self.consecutive_increases})"
                    )
            else:
                self.consecutive_increase_count = 0
        if self.consecutive_increase_count >= self.consecutive_increases and not self.spike_active:
            self.spike_active = True
            self.spike_steps_remaining = self.spike_duration
            self.original_lrs = [param_group["lr"] for param_group in optimizer.param_groups]
            for param_group in optimizer.param_groups:
                param_group["lr"] = param_group["lr"] * self.spike_multiplier
            if logger:
                logger.warning("LR SPIKE TRIGGERED! Validation loss increased consecutively.")
                logger.warning(
                    f"Spiking LR by {self.spike_multiplier}x for {self.spike_duration} steps: "
                    f"{self.original_lrs[0]:.2e} -> {optimizer.param_groups[0]['lr']:.2e}"
                )
            self.consecutive_increase_count = 0
            return True
        return False

    def step(self, optimizer, logger=None):
        if self.spike_active:
            self.spike_steps_remaining -= 1
            if self.spike_steps_remaining <= 0:
                for i, param_group in enumerate(optimizer.param_groups):
                    param_group["lr"] = self.original_lrs[i]
                if logger:
                    logger.info(f"LR spike ended. Restored LR to {optimizer.param_groups[0]['lr']:.2e}")
                self.spike_active = False
                self.spike_steps_remaining = 0

    def get_state_dict(self):
        return {
            "val_loss_history": self.val_loss_history,
            "consecutive_increase_count": self.consecutive_increase_count,
            "spike_active": self.spike_active,
            "spike_steps_remaining": self.spike_steps_remaining,
            "original_lrs": self.original_lrs,
        }

    def load_state_dict(self, state_dict):
        self.val_loss_history = state_dict.get("val_loss_history", [])
        self.consecutive_increase_count = state_dict.get("consecutive_increase_count", 0)
        self.spike_active = state_dict.get("spike_active", False)
        self.spike_steps_remaining = state_dict.get("spike_steps_remaining", 0)
        self.original_lrs = state_dict.get("original_lrs", [])


class TrainingMonitor:
    """Unified training monitor: LR spike + early stopping + val_loss threshold."""

    def __init__(self, cfg: dict):
        self.use_lr_spike = cfg.get("use_lr_spike", False)
        self.spike = (
            LRSpike(
                spike_multiplier=cfg.get("lr_spike_multiplier", 5.0),
                spike_duration=cfg.get("lr_spike_duration", 50),
                consecutive_increases=cfg.get("lr_spike_consecutive_increases", 3),
            )
            if self.use_lr_spike
            else None
        )
        self.val_loss_threshold = cfg.get("val_loss_threshold", float("inf"))
        self.consecutive_spike_limit = cfg.get("val_loss_spike_patience", 3)
        self.spike_count = 0
        self.reload_needed = False
        self.use_early_stopping = cfg.get("use_early_stopping", False)
        self.es_patience = cfg.get("early_stopping_patience", 5)
        self.es_min_delta = cfg.get("early_stopping_min_delta", 0.001)
        self.es_counter = 0
        self.should_stop = False
        self.best_val_loss = float("inf")
        self.best_state_dicts = {}
        self.last_checkpoint_val_loss = None

    def on_val_end(self, val_loss: float, optimizer, models: dict = None, logger=None) -> bool:
        improved = val_loss < self.best_val_loss - self.es_min_delta
        if improved:
            self.best_val_loss = val_loss
            self.es_counter = 0
            self.spike_count = 0
            if models:
                self.best_state_dicts = {n: {k: v.clone() for k, v in m.state_dict().items()} for n, m in models.items()}
        else:
            self.es_counter += 1
        self.reload_needed = False
        if self.last_checkpoint_val_loss is not None and self.val_loss_threshold < 999.0:
            if val_loss > self.last_checkpoint_val_loss + self.val_loss_threshold:
                self.spike_count += 1
                if logger:
                    logger.warning(f"Val loss spike ({self.spike_count}/{self.consecutive_spike_limit})")
                if self.spike_count >= self.consecutive_spike_limit:
                    self.reload_needed = True
                    self.spike_count = 0
                    if logger:
                        logger.warning("Sustained divergence detected. Reload needed.")
            else:
                self.spike_count = 0
        if self.spike is not None:
            self.spike.check_and_spike(val_loss, optimizer, logger)
        if self.use_early_stopping and not improved:
            if logger:
                logger.info(f"EarlyStopping: no improvement ({self.es_counter}/{self.es_patience})")
            if self.es_counter >= self.es_patience:
                self.should_stop = True
                if logger:
                    logger.info("EarlyStopping: patience exhausted. Stopping training.")
        return self.should_stop

    def update_checkpoint_loss(self, val_loss: float):
        self.last_checkpoint_val_loss = val_loss

    def step(self, optimizer, logger=None):
        if self.spike is not None:
            self.spike.step(optimizer, logger)

    def restore_best(self, models: dict, logger=None):
        if self.best_state_dicts:
            for name, model in models.items():
                if name in self.best_state_dicts:
                    model.load_state_dict(self.best_state_dicts[name])
            if logger:
                logger.info(f"Restored best weights (val_loss={self.best_val_loss:.4f})")

    def get_state_dict(self):
        state = {
            "best_val_loss": self.best_val_loss,
            "last_checkpoint_val_loss": self.last_checkpoint_val_loss,
            "es_counter": self.es_counter,
            "spike_count": self.spike_count,
            "should_stop": self.should_stop,
        }
        if self.spike is not None:
            state["lr_spike"] = self.spike.get_state_dict()
        return state

    def load_state_dict(self, state_dict):
        self.best_val_loss = state_dict.get("best_val_loss", float("inf"))
        self.last_checkpoint_val_loss = state_dict.get("last_checkpoint_val_loss", None)
        self.es_counter = state_dict.get("es_counter", 0)
        self.spike_count = state_dict.get("spike_count", 0)
        self.should_stop = state_dict.get("should_stop", False)
        if self.spike is not None and "lr_spike" in state_dict:
            self.spike.load_state_dict(state_dict["lr_spike"])


EarlyStopping = TrainingMonitor


def get_lr_scheduler(optimizer, warmup_steps, max_steps, min_lr_ratio=0.1):
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        current_step = min(step, max_steps)
        progress = (current_step - warmup_steps) / max((max_steps - warmup_steps), 1)
        return min_lr_ratio + (1 - min_lr_ratio) * 0.5 * (1 + torch.cos(torch.tensor(progress * math.pi)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def clip_gradients(model, max_norm=1.0):
    return torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)


def validate_loss(loss, min_loss=-1e6, max_loss=1e6, raise_on_error=True):
    loss_val = loss.detach().item() if isinstance(loss, torch.Tensor) else float(loss)
    if not (min_loss <= loss_val <= max_loss) or not (loss_val == loss_val):
        error_msg = f"Invalid loss value: {loss_val} (expected range: [{min_loss}, {max_loss}])"
        if raise_on_error:
            raise RuntimeError(error_msg)
        return False
    return True


def check_gradient_explosion(model, max_grad_norm=100.0, raise_on_error=True):
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf"))
    grad_norm_val = grad_norm.item()
    is_exploded = grad_norm_val > max_grad_norm or not (grad_norm_val == grad_norm_val)
    if is_exploded:
        error_msg = f"Gradient explosion detected: grad_norm={grad_norm_val:.2f} (max={max_grad_norm})"
        if raise_on_error:
            raise RuntimeError(error_msg)
        return grad_norm_val, True
    return grad_norm_val, False


class SimpleLogger:
    """Simple terminal logger for training metrics."""

    def __init__(self, name="Training"):
        self.name = name
        self.start_time = datetime.now()

    def _write(self, text: str) -> None:
        try:
            from tqdm import tqdm  # type: ignore

            tqdm.write(text)
        except Exception:
            print(text)

    def _format_time(self):
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def _format_message(self, level, message):
        return f"[{self._format_time()}] [{self.name}] [{level}] {message}"

    def info(self, message):
        self._write(self._format_message("INFO", message))

    def warning(self, message):
        self._write(self._format_message("WARN", message))

    def error(self, message):
        self._write(self._format_message("ERROR", message))

    def train_step(self, step, loss, lr, epoch=None):
        epoch_str = f"epoch={epoch}, " if epoch is not None else ""
        self._write(self._format_message("TRAIN", f"Step {step} | {epoch_str}train_loss={loss:.4f} | lr={lr:.6f}"))

    def val_step(self, step, val_loss, epoch=None):
        epoch_str = f"epoch={epoch}, " if epoch is not None else ""
        self._write(self._format_message("VAL", f"Step {step} | {epoch_str}val_loss={val_loss:.4f}"))

    def checkpoint(self, step, path, is_best=False):
        best_str = " (BEST)" if is_best else ""
        self._write(self._format_message("CHECKPOINT", f"Checkpoint saved at step {step}{best_str}: {path}"))

    def epoch_start(self, epoch):
        self._write(self._format_message("EPOCH", f"Starting epoch {epoch}"))

    def epoch_end(self, epoch, train_loss=None, val_loss=None):
        parts = [f"Epoch {epoch} completed"]
        if train_loss is not None:
            parts.append(f"train_loss={train_loss:.4f}")
        if val_loss is not None:
            parts.append(f"val_loss={val_loss:.4f}")
        self._write(self._format_message("EPOCH", " | ".join(parts)))

    def training_start(self, total_steps, train_samples, val_samples=None):
        msg = f"Starting training | max_steps={total_steps} | train_samples={train_samples}"
        if val_samples is not None:
            msg += f" | val_samples={val_samples}"
        self._write(self._format_message("START", msg))

    def training_end(self, total_steps):
        elapsed = datetime.now() - self.start_time
        self._write(self._format_message("END", f"Training completed | total_steps={total_steps} | elapsed={elapsed}"))

    def metric(self, step, metric_name, value, epoch=None):
        epoch_str = f"epoch={epoch}, " if epoch is not None else ""
        self._write(self._format_message("METRIC", f"Step {step} | {epoch_str}{metric_name}={value:.4f}"))


def build_run_id(script: str, config_path: str | None = None, save_dir: str | None = None) -> str:
    """Build a stable run id so resumed runs can upsert same keys."""
    base = f"{script}|{config_path or ''}|{save_dir or ''}"
    return hashlib.sha1(base.encode("utf-8")).hexdigest()[:16]


def _record_key(record: dict[str, Any]) -> str:
    run_id = str(record.get("run_id", ""))
    phase = str(record.get("phase", ""))
    epoch = str(record.get("epoch", ""))
    batch = str(record.get("batch", ""))
    step = str(record.get("step", ""))
    split = str(record.get("split", ""))
    metric_name = str(record.get("metric_name", ""))
    return f"{run_id}|{phase}|{split}|{epoch}|{batch}|{step}|{metric_name}"


def append_or_upsert_jsonl(path: str, record: dict[str, Any]) -> None:
    """Append a record or replace existing duplicate key atomically."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    key = _record_key(record)
    records: list[dict[str, Any]] = []
    replaced = False

    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                if _record_key(obj) == key:
                    records.append(record)
                    replaced = True
                else:
                    records.append(obj)

    if not replaced:
        records.append(record)

    fd, tmp_path = tempfile.mkstemp(prefix="metrics_", suffix=".jsonl", dir=os.path.dirname(path))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            for obj in records:
                f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass


class MetricsLogger:
    """Structured metrics logger with resume-safe upsert writes."""

    def __init__(self, script: str, run_id: str, metrics_path: str, device: str | None = None):
        self.script = script
        self.run_id = run_id
        self.metrics_path = metrics_path
        self.device = device

    def emit(
        self,
        *,
        phase: str,
        split: str,
        metric_name: str,
        metric_value: float,
        epoch: int | None,
        batch: int | None,
        step: int | None,
        lr: float | None = None,
        loss: float | None = None,
        checkpoint: str | None = None,
        extra: dict[str, Any] | None = None,
    ) -> None:
        record = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "script": self.script,
            "phase": phase,
            "run_id": self.run_id,
            "epoch": epoch,
            "batch": batch,
            "step": step,
            "split": split,
            "metric_name": metric_name,
            "metric_value": float(metric_value),
            "lr": lr,
            "loss": loss,
            "checkpoint": checkpoint,
            "device": self.device,
            "extra": extra or {},
        }
        append_or_upsert_jsonl(self.metrics_path, record)

    def train_step(self, *, epoch: int, batch: int, step: int, loss: float, lr: float) -> None:
        self.emit(
            phase="train",
            split="train",
            metric_name="loss",
            metric_value=loss,
            epoch=epoch,
            batch=batch,
            step=step,
            lr=lr,
            loss=loss,
        )

    def val_step(self, *, epoch: int, batch: int | None, step: int, loss: float, metric_name: str = "val_loss") -> None:
        self.emit(
            phase="val",
            split="val",
            metric_name=metric_name,
            metric_value=loss,
            epoch=epoch,
            batch=batch,
            step=step,
            loss=loss,
        )

    def event(self, *, epoch: int | None, batch: int | None, step: int | None, name: str, value: float = 1.0, extra: dict[str, Any] | None = None) -> None:
        self.emit(
            phase="event",
            split="event",
            metric_name=name,
            metric_value=value,
            epoch=epoch,
            batch=batch,
            step=step,
            extra=extra,
        )

    def test_metrics(self, metrics: dict[str, Any], *, phase: str = "test", split: str = "test") -> None:
        for k, v in metrics.items():
            if isinstance(v, bool):
                self.emit(phase=phase, split=split, metric_name=k, metric_value=float(v), epoch=0, batch=0, step=0)
            elif isinstance(v, (int, float)):
                self.emit(phase=phase, split=split, metric_name=k, metric_value=float(v), epoch=0, batch=0, step=0)


def reload_from_last_checkpoint(save_dir, checkpoint_prefix, device, logger, model, opt=None, scheduler=None, scaler=None):
    if not os.path.exists(save_dir):
        logger.error(f"Save directory does not exist: {save_dir}")
        return 0
    checkpoint_files = [f for f in os.listdir(save_dir) if f.startswith(checkpoint_prefix) and f.endswith(".pt")]
    if not checkpoint_files:
        logger.error(f"No checkpoint files found with prefix '{checkpoint_prefix}' in {save_dir}")
        return 0
    step_numbers = []
    for f in checkpoint_files:
        try:
            step_num = int(f.replace(checkpoint_prefix, "").replace(".pt", ""))
            step_numbers.append((step_num, f))
        except Exception:
            continue
    if not step_numbers:
        logger.error("Could not parse step numbers from checkpoint files")
        return 0
    step_numbers.sort(key=lambda x: x[0], reverse=True)
    last_checkpoint = os.path.join(save_dir, step_numbers[0][1])
    step = step_numbers[0][0]
    logger.error(f"NaN detected in attention. Reloading from checkpoint: {last_checkpoint}")
    try:
        checkpoint = torch.load(last_checkpoint, map_location=device)
        if isinstance(checkpoint, dict):
            if "model" in checkpoint:
                model.load_state_dict(checkpoint["model"])
            elif "thinker" in checkpoint:
                model.load_state_dict(checkpoint["thinker"])
            else:
                model.load_state_dict(checkpoint)
            if opt is not None and "optimizer" in checkpoint:
                opt.load_state_dict(checkpoint["optimizer"])
            if scheduler is not None and "scheduler" in checkpoint:
                scheduler.load_state_dict(checkpoint["scheduler"])
            if scaler is not None and "scaler" in checkpoint:
                scaler.load_state_dict(checkpoint["scaler"])
            loaded_step = checkpoint.get("step", step)
            logger.info(f"Successfully reloaded from step {loaded_step}")
            return loaded_step
        model.load_state_dict(checkpoint)
        logger.info("Loaded model weights from checkpoint (legacy format)")
        return step
    except Exception as e:
        logger.error(f"Failed to reload from checkpoint: {e}")
        return 0


__all__ = [
    "set_seed",
    "setup_cuda",
    "get_lr_scheduler",
    "clip_gradients",
    "validate_loss",
    "check_gradient_explosion",
    "SimpleLogger",
    "TrainingMonitor",
    "EMA",
    "reload_from_last_checkpoint",
    "build_run_id",
    "append_or_upsert_jsonl",
    "MetricsLogger",
]
