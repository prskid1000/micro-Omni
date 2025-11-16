# Checkpoint File Structure

## Overview

All `.pt` checkpoint files in μOmni use PyTorch's `torch.save()` to store a **dictionary** containing the complete training state. This allows training to resume exactly where it left off.

## Checkpoint Structure

### General Format

All checkpoints are saved as dictionaries with the following structure:

```python
checkpoint = {
    # Model weights (required)
    "model": model.state_dict(),  # or "thinker", "vit", "enc", etc.
    
    # Training state (required for resume)
    "optimizer": optimizer.state_dict(),
    "scheduler": scheduler.state_dict(),
    "step": int,  # Current training step
    "best_val_loss": float,  # Best validation loss seen
    
    # AMP scaler (if using mixed precision)
    "scaler": scaler.state_dict(),  # Optional, only if use_amp=True
}
```

## Model-Specific Checkpoint Keys

### Thinker (`train_text.py`)

```python
checkpoint = {
    "model": thinker_model.state_dict(),
    "optimizer": optimizer.state_dict(),
    "scheduler": scheduler.state_dict(),
    "step": step,
    "best_val_loss": best_val_loss,
    "scaler": scaler.state_dict()  # if use_amp=True
}
```

**File locations:**
- Periodic: `checkpoints/thinker_tiny/thinker_step_{step}.pt`
- Best model: `checkpoints/thinker_tiny/thinker_best.pt`
- Final: `checkpoints/thinker_tiny/thinker.pt`

### Vision Encoder (`train_vision.py`)

```python
checkpoint = {
    "vit": vision_encoder.state_dict(),
    "img_proj": img_proj.state_dict(),
    "text_proj": text_proj.state_dict(),
    "text_embed": text_embed.state_dict(),
    "optimizer": optimizer.state_dict(),
    "scheduler": scheduler.state_dict(),
    "step": step,
    "best_val_loss": best_val_loss,
    "scaler": scaler.state_dict()  # if use_amp=True
}
```

**File locations:**
- Periodic: `checkpoints/vision_tiny/vision_step_{step}.pt`
- Best model: `checkpoints/vision_tiny/vision_best.pt`
- Final: `checkpoints/vision_tiny/vision.pt`

### Audio Encoder (`train_audio_enc.py`)

```python
checkpoint = {
    "enc": audio_encoder.state_dict(),
    "head": ctc_head.state_dict(),
    "optimizer": optimizer.state_dict(),
    "scheduler": scheduler.state_dict(),
    "step": step,
    "best_val_loss": best_val_loss,
    "scaler": scaler.state_dict()  # if use_amp=True
}
```

**File locations:**
- Periodic: `checkpoints/audio_enc_tiny/audio_enc_step_{step}.pt`
- Best model: `checkpoints/audio_enc_tiny/audio_enc_best.pt`
- Final: `checkpoints/audio_enc_tiny/audio_enc.pt`

### Talker (`train_talker.py`)

```python
checkpoint = {
    "talker": talker_model.state_dict(),
    "optimizer": optimizer.state_dict(),
    "scheduler": scheduler.state_dict(),
    "step": step,
    "best_val_loss": best_val_loss,
    "scaler": scaler.state_dict()  # if use_amp=True
}
```

**File locations:**
- Periodic: `checkpoints/talker_tiny/talker_step_{step}.pt`
- Best model: `checkpoints/talker_tiny/talker_best.pt`
- Final: `checkpoints/talker_tiny/talker.pt`

### Multimodal SFT (`sft_omni.py`)

```python
checkpoint = {
    "thinker": thinker_model.state_dict(),
    "proj_a": audio_projector.state_dict(),
    "proj_v": vision_projector.state_dict(),
    "optimizer": optimizer.state_dict(),
    "scheduler": scheduler.state_dict(),
    "step": step,
    "best_val_loss": best_val_loss,
    "scaler": scaler.state_dict()  # if use_amp=True
}
```

**File locations:**
- Periodic: `checkpoints/omni_sft_tiny/omni_step_{step}.pt`
- Best model: `checkpoints/omni_sft_tiny/omni_best.pt`
- Final: `checkpoints/omni_sft_tiny/omni.pt`

## How to Inspect a Checkpoint

```python
import torch

# Load checkpoint
checkpoint = torch.load("checkpoints/thinker_tiny/thinker_best.pt", map_location="cpu")

# Inspect structure
print("Checkpoint keys:", list(checkpoint.keys()))
# Output: ['model', 'optimizer', 'scheduler', 'step', 'best_val_loss', 'scaler']

# Check training progress
print(f"Training step: {checkpoint['step']}")
print(f"Best validation loss: {checkpoint['best_val_loss']:.4f}")

# Inspect model weights
model_state = checkpoint["model"]
print(f"Model parameters: {len(model_state)} layers")
print(f"First few keys: {list(model_state.keys())[:5]}")

# Inspect optimizer state
opt_state = checkpoint["optimizer"]
print(f"Optimizer state keys: {list(opt_state.keys())}")
# Typically: ['state', 'param_groups']
```

## Loading Checkpoints

### For Inference (Model Weights Only)

```python
import torch
from omni.thinker import ThinkerLM

# Load checkpoint
checkpoint = torch.load("checkpoints/thinker_tiny/thinker_best.pt", map_location="cpu")

# Extract model weights
if isinstance(checkpoint, dict) and "model" in checkpoint:
    model_state = checkpoint["model"]
else:
    # Legacy format (just model weights)
    model_state = checkpoint

# Load into model
model = ThinkerLM(...)
model.load_state_dict(model_state)
model.eval()
```

### For Resuming Training (Full State)

```python
import torch

# Load checkpoint
checkpoint = torch.load("checkpoints/thinker_tiny/thinker_step_1500.pt", map_location=device)

# Load model
model.load_state_dict(checkpoint["model"])

# Load optimizer
optimizer.load_state_dict(checkpoint["optimizer"])

# Load scheduler
scheduler.load_state_dict(checkpoint["scheduler"])

# Load scaler (if using AMP)
if "scaler" in checkpoint and scaler is not None:
    scaler.load_state_dict(checkpoint["scaler"])

# Restore training state
step = checkpoint["step"]
best_val_loss = checkpoint["best_val_loss"]

# Continue training from step 1500
```

## Checkpoint File Sizes

Approximate sizes for tiny models:

| Model | Model Weights | Full Checkpoint | Location |
|-------|--------------|-----------------|----------|
| Thinker | ~15 MB | ~45 MB | `checkpoints/thinker_tiny/` |
| Vision | ~2 MB | ~6 MB | `checkpoints/vision_tiny/` |
| Audio | ~5 MB | ~15 MB | `checkpoints/audio_enc_tiny/` |
| Talker | ~8 MB | ~24 MB | `checkpoints/talker_tiny/` |
| Multimodal SFT | ~20 MB | ~60 MB | `checkpoints/omni_sft_tiny/` |

**Note**: Full checkpoints are ~3x larger than model weights alone because they include optimizer state (momentum buffers, etc.) and scheduler state.

## When Checkpoints Are Saved

1. **Periodic Checkpoints**: Every `checkpoint_freq` steps
   - Format: `{model}_step_{step}.pt`
   - Example: `thinker_step_500.pt`, `thinker_step_1000.pt`
   - **Auto-cleanup**: Only the **last checkpoint** is kept (older ones deleted automatically)

2. **Best Model**: When validation loss improves
   - Format: `{model}_best.pt`
   - Example: `thinker_best.pt`
   - **Always kept**: Never deleted by cleanup

3. **Final Model**: At end of training
   - Format: `{model}.pt`
   - Example: `thinker.pt`
   - **Always kept**: Final checkpoint

### Checkpoint Cleanup (Storage Optimization)

**All training scripts now automatically clean up old checkpoints** to save storage space.

**How it works:**
```python
# After saving each periodic checkpoint
torch.save(checkpoint_data, checkpoint_path)
logger.checkpoint(step, checkpoint_path)

# Automatically clean up old checkpoints (keep only last 1)
cleanup_old_checkpoints(cfg["save_dir"], "thinker_step_", keep_last_n=1)
```

**What gets deleted:**
- ✅ Old periodic checkpoints (e.g., `thinker_step_500.pt` when `thinker_step_1000.pt` is saved)

**What's always preserved:**
- ✅ Latest periodic checkpoint (e.g., `thinker_step_1000.pt`)
- ✅ Best model (`thinker_best.pt`)
- ✅ Final model (`thinker.pt`)

**Storage savings:**
- **Before**: 2,150+ checkpoint files (~43 GB for Thinker alone!)
- **After**: 3 checkpoint files (~60 MB for Thinker)
- **Reduction**: 99.9% storage saved! 🎉

**Configuration:**
```python
# In omni/training_utils.py
cleanup_old_checkpoints(
    save_dir="checkpoints/thinker_tiny",
    checkpoint_prefix="thinker_step_",
    keep_last_n=1  # Keep only the last N checkpoints (default: 1)
)
```

**Example directory after cleanup:**
```
checkpoints/thinker_tiny/
├── thinker_step_1075000.pt    ← Latest (kept)
├── thinker_best.pt             ← Best (always kept)
├── thinker.pt                  ← Final (always kept)
└── tokenizer.model             ← Not a checkpoint
```

**Benefits:**
- 💾 **Massive storage savings** (99%+ reduction)
- 🔄 **Resume still works** (latest checkpoint always available)
- 🏆 **Best model preserved** (for deployment)
- 🔒 **Final model safe** (never deleted)
- ⚡ **Automatic** (no manual cleanup needed)

## Automatic Resume

Training scripts automatically:
1. Scan `save_dir` for `{model}_step_*.pt` files
2. Find the latest step number
3. Load the full checkpoint state
4. Resume from that exact step

**Example:**
- Training stops at step 1999
- Latest checkpoint: `thinker_step_1500.pt` (checkpoint_freq=500)
- Training resumes from step 1500
- Script skips batches until step 1999
- Continues from step 2000

## Legacy Format Support

For backward compatibility, scripts also support legacy checkpoints that only contain model weights:

```python
# Legacy format (just model weights)
checkpoint = model.state_dict()  # Direct state dict, not a dict

# New format (full state)
checkpoint = {
    "model": model.state_dict(),
    "optimizer": ...,
    # ...
}
```

The loading code checks for both formats automatically.

## Storage Location

All checkpoints are stored in the directory specified by `save_dir` in the config file:

```json
{
  "save_dir": "checkpoints/thinker_tiny"
}
```

The directory structure **with automatic cleanup enabled:**
```
checkpoints/
├── thinker_tiny/
│   ├── thinker_step_1075000.pt  ← Only latest periodic checkpoint
│   ├── thinker_best.pt           ← Best model (always kept)
│   ├── thinker.pt                ← Final model (always kept)
│   └── tokenizer.model           ← Tokenizer (not a checkpoint)
├── vision_tiny/
│   ├── vision_step_231300.pt     ← Only latest
│   ├── vision_best.pt            ← Best (kept)
│   └── vision.pt                 ← Final (kept)
├── audio_enc_tiny/
│   ├── audio_enc_step_64200.pt   ← Only latest
│   ├── audio_enc_best.pt         ← Best (kept)
│   └── audio_enc.pt              ← Final (kept)
└── omni_sft_tiny/
    ├── omni_step_1290000.pt      ← Only latest
    ├── omni_best.pt              ← Best (kept)
    └── omni.pt                   ← Final (kept)
```

**Total storage:** ~270 MB (vs 80 GB without cleanup!) ✅

## Key Points

1. **All state is in one file**: Model, optimizer, scheduler, scaler, step, best_val_loss
2. **Dictionary format**: Easy to inspect and modify
3. **Automatic resume**: Training scripts handle checkpoint detection and loading
4. **Backward compatible**: Supports both new (dict) and legacy (state_dict only) formats
5. **Efficient storage**: PyTorch uses efficient serialization (pickle-based)
6. **Automatic cleanup**: Only last periodic checkpoint kept, 99% storage saved ⭐
7. **Safe cleanup**: Best and final models always preserved
8. **No manual intervention**: Cleanup happens automatically during training

---

**See Also:**
- [Training Workflow](07_Training_Workflow.md) - How training and checkpointing works
- [Inference Guide](08_Inference_Guide.md) - How to load checkpoints for inference
- [Quick Reference](QUICK_REFERENCE.md) - Quick commands and parameters

