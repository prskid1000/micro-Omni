# Chapter 39: Running Training - Complete Walkthrough

[← Previous: Setup Environment](38-setup-environment.md) | [Back to Index](00-INDEX.md) | [Next: Inference Examples →](40-inference-examples.md)

---

## 🚀 Complete Training Pipeline

Step-by-step guide to training μOmni from scratch.

---

## 📝 Prerequisites

✅ Environment setup complete (Chapter 38)  
✅ Data prepared (Chapter 35)  
✅ GPU available (12GB+)  
✅ **(Optional) Run LR Finder** to discover optimal learning rate before training  
✅ **(Automatic) EMA enabled** by default in all configs for better stability  
✅ **(Automatic) Early stopping** prevents endless validation loops

---

## 🔍 Before Training: Discover Optimal Learning Rate

**Highly recommended:** Run the LR Finder before starting each training stage to automatically discover the optimal learning rate. This saves hours of trial-and-error:

```bash
# Find optimal LR for Thinker (Stage A)
python find_lr.py --config configs/thinker_tiny.json \
  --model_type thinker \
  --output_plot lr_finder_thinker.png

# Find optimal LR for Audio Encoder (Stage B)
python find_lr.py --config configs/audio_enc_tiny.json \
  --model_type audio_enc \
  --output_plot lr_finder_audio.png

# Find optimal LR for Vision Encoder (Stage C)
python find_lr.py --config configs/vision_tiny.json \
  --model_type vision \
  --output_plot lr_finder_vision.png

# Find optimal LR for Talker (Stage D)
python find_lr.py --config configs/talker_tiny.json \
  --model_type talker \
  --output_plot lr_finder_talker.png

# Find optimal LR for OCR (Optional)
python find_lr.py --config configs/ocr_tiny.json \
  --model_type ocr \
  --output_plot lr_finder_ocr.png
```

**What it does:**

- Uses Smith 2017 range test method
- Runs for 5-10 minutes per model
- Automatically suggests optimal learning rate
- Generates plot showing loss vs learning rate curve
- Saves you hours of hyperparameter tuning

**Example output:**

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Suggested Learning Rate: 2.00e-05
(Based on steepest descent at LR 1.78e-05)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Plot saved to: lr_finder_thinker.png
```

**Update config with suggested LR:**

```json
{
  "learning_rate": 2.0e-5  // Use the suggested value
}
```

See [Chapter 36: Optimization Techniques](36-optimization-techniques.md#learning-rate-finder) for more details.

---

## 🎯 Training Sequence

### Stage A: Thinker (Text-Only)

```bash
# Train language model
python train_text.py --config configs/thinker_tiny.json

# Expected time: 8-12 hours
# Expected result: Perplexity < 10
# Output: checkpoints/thinker_tiny/model.pt + model_metadata.json
```

**Monitor:**

```bash
# Watch GPU
watch -n 1 nvidia-smi

# Check logs
tail -f checkpoints/thinker_tiny/training.log
```

### Stage B: Audio Encoder

```bash
python train_audio_enc.py --config configs/audio_enc_tiny.json

# Time: 6-10 hours
# Target: WER < 20%
# Output: checkpoints/audio_enc_tiny/model.pt + model_metadata.json
```

**Configuration Note:**

- `max_mel_length` is **auto-calculated** from your dataset (95th percentile)
- No manual configuration needed - training script analyzes dataset automatically
- Can override with `max_mel_length` or adjust `max_mel_length_percentile` if needed
- Optional: Check your dataset using `omni.utils` analysis helpers (e.g. `calculate_max_mel_length_from_asr_csv`) instead of a standalone script.

### Stage C: Vision Encoder

```bash
python train_vision.py --config configs/vision_tiny.json

# Time: 4-8 hours
# Target: Low contrastive loss (good vision-language alignment)
# Output: checkpoints/vision_tiny/model.pt + model_metadata.json
```

**Configuration Note:**

- Uses trained tokenizer from Stage A (`thinker_ckpt/tokenizer.model`)
- Configurable text encoding: `use_thinker_for_text` (default: true)
  - `true`: Uses frozen Thinker model for contextual embeddings (recommended)
  - `false`: Uses simple tokenizer + embedding layer (lighter option)
- If tokenizer not found, trains new one from image captions

### Stage D: Talker + RVQ

```bash
python train_talker.py --config configs/talker_tiny.json

# Time: 10-15 hours
# Target: Intelligible speech
# Output: checkpoints/talker_tiny/model.pt + model_metadata.json
```

**Configuration Note:**

- `max_mel_length` is **auto-calculated** from your dataset (95th percentile)
- No manual configuration needed - training script analyzes dataset automatically
- Talker uses different frame rate (12.5 fps with frame_ms=80) than audio encoder (100 fps)
- Can override with `max_mel_length` or adjust `max_mel_length_percentile` if needed

### Stage E: Multimodal SFT

```bash
python sft_omni.py --config configs/omni_sft_tiny.json

# Time: 6-12 hours
# Target: Good multimodal Q&A
# Output: checkpoints/omni_sft_tiny/model.pt + model_metadata.json
```

### Optional: HiFi-GAN Vocoder Training

```bash
# Train neural vocoder for better speech quality (optional)
python train_vocoder.py --config configs/vocoder_tiny.json

# Time: 2-4 hours (on 12GB GPU with optimizations)
# Target: Natural-sounding speech
# Output: checkpoints/vocoder_tiny/model.pt + model_metadata.json
# Note: Falls back to Griffin-Lim if checkpoint not available
```

**When to train:**

- After Stage D (Talker) is complete
- If you want higher quality speech output
- Griffin-Lim works fine for basic TTS, but HiFi-GAN is better

**Performance Optimizations:**

- ✅ torch.compile() enabled (10-20% speedup)
- ✅ cuDNN benchmark mode (5-15% speedup for convolutions)
- ✅ channels_last memory format (10-30% performance boost)
- ✅ Cached discriminator features (20-25% faster, no redundant forward passes)
- ✅ Mixed precision training (FP16) - 50% memory savings, 2x faster
- ✅ Gradient accumulation for memory efficiency
- ✅ Total speedup: 3-4x faster training vs baseline

**Implementation Notes:**

- ✅ Generator correctly handles tensor dimensions (outputs `(B, T_audio)` for batches)
- ✅ Audio loading automatically falls back to `torchaudio.load()` if torchcodec unavailable
- ✅ Discriminator inputs properly shaped as `(B, 1, T)` automatically
- ✅ All shape handling verified and working correctly
- ✅ CNN-specific optimizations (cuDNN autotuner, channels_last format)
- ✅ Feature caching eliminates redundant discriminator forward passes

### Optional: OCR Training

```bash
# Train OCR model for text extraction from images (optional)
python train_ocr.py --config configs/ocr_tiny.json

# Time: 4-8 hours (on 12GB GPU)
# Target: Accurate text extraction from images
# Output: checkpoints/ocr_tiny/model.pt + model_metadata.json
# Note: Can be used with --ocr flag in inference
```

**Configuration Note:**

- `max_text_length` is **auto-calculated** from your dataset (95th percentile)
- No manual configuration needed - training script analyzes dataset automatically
- Can override with `max_text_length` or adjust `max_text_length_percentile` if needed
- Default: 256 characters
- Adjust based on your text lengths (short: 128, long: 512)

**When to train:**

- If you need text extraction from images
- For document processing, scene text recognition
- Can be combined with multimodal understanding in Stage E

---

## 📊 Monitoring Training

### Key Metrics to Watch

**Stage A (Thinker):**

- Loss decreasing steadily
- Perplexity < 10 (target)
- No NaN/Inf values

**Stage B (Audio):**

- CTC loss decreasing
- WER improving (lower is better)
- Target: WER < 20%

**Stage C (Vision):**

- Accuracy increasing
- Loss decreasing
- Target: Accuracy > 70%

**Stage D (Talker):**

- Reconstruction error < 0.05
- Speech codes perplexity < 15
- Generated speech intelligible

**Stage E (SFT):**

- All modalities improving
- Cross-modal accuracy increasing
- Target: >60% on mixed tasks

---

## 🛡️ Training Stability & Auto-Reload

**Modern Training Features:**

All μOmni training scripts include state-of-the-art stability and optimization features:

### 1. Exponential Moving Average (EMA)

**Enabled by default** in all config files for improved training stability and generalization:

```json
{
  "use_ema": true,
  "ema_decay": 0.999
}
```

**What it does:**

- Maintains smoothed "shadow weights" alongside primary weights
- Uses shadow weights for validation (better performance)
- Minimal overhead, significant benefits
- Research shows 0.5-2% improvement in validation metrics

**How it works:**

- Updates shadow weights after each optimizer step: `θ_ema ← 0.999 × θ_ema + 0.001 × θ`
- Automatically saves/loads EMA state in checkpoints
- No configuration needed - works out-of-the-box

See [Chapter 36: EMA](36-optimization-techniques.md#exponential-moving-average-ema) for technical details.

### 2. Early Stopping for Validation Spikes

**Automatic protection** against endless validation reload loops:

```python
# Automatically stops after 2 consecutive validation loss increases
# No configuration needed
```

**What happens:**

```
Step 1000 | Val Loss: 2.34 ✓ (save checkpoint)
Step 2000 | Val Loss: 2.89 ✗ (worse! reload, consecutive_reloads=1)
Step 2000 | Val Loss: 2.91 ✗ (worse! reload, consecutive_reloads=2)

ERROR: Training stopped after 2 consecutive validation loss increases.
This usually indicates:
- Learning rate too high
- Overfitting
- Need different hyperparameters

Solutions:
1. Reduce learning rate by 2-5x
2. Enable/increase regularization (dropout, weight_decay)
3. Add more training data
4. Check for data quality issues
```

**Benefits:**

- Prevents infinite reload loops
- Fails fast with clear error messages
- Suggests concrete solutions
- Saves hours of wasted compute

**Common causes and fixes:**

1. **LR too high:** Reduce by 2-5x or rerun LR Finder
2. **Overfitting:** Add regularization, more data
3. **Data issues:** Check dataset quality
4. **Batch size too small:** Increase batch_size or gradient_accumulation_steps

See [Chapter 36: Early Stopping](36-optimization-techniques.md#early-stopping-for-validation-spikes) for more details.

### 3. Legacy Validation Loss Spike Protection (Deprecated)

**Note:** The old `val_loss_threshold` feature is superseded by the new early stopping mechanism above. It is no longer needed as early stopping provides better behavior with helpful error messages.

~~**How it works:**~~

~~1. **Threshold Check:** During validation, the script checks:~~
   ~~`current_val_loss > last_checkpoint_val_loss + val_loss_threshold`~~
~~2. **Trigger:** If the condition is met (loss spiked), a reload is triggered.~~
~~3. **Action:**~~
   ~~- Reloads model, optimizer, scheduler, and scaler from the **last saved checkpoint**.~~
   ~~- Resets the data loader to the correct position.~~
   ~~- Resumes training, effectively discarding the unstable steps since the last checkpoint.~~

~~**Configuration:**~~

~~```json
{
  "val_loss_threshold": 0.05 // Default: infinity (disabled) if not set
}
```~~

~~- Set to a reasonable value (e.g., `0.05` or `0.1`) to enable.~~
~~- Prevents training divergence and saves time by automatically recovering from instability.~~

---

## 🛠️ Resuming Training

**Automatic Resuming:** All training scripts automatically detect and resume from the latest checkpoint. Simply rerun the training command:

```bash
# Training interrupted? Just rerun - it will auto-resume!
python train_text.py --config configs/thinker_tiny.json

# The script automatically:
# ✅ Finds `model.pt` and `model_metadata.json` in save_dir
# ✅ Loads all states (model, optimizer, scheduler, scaler)
# ✅ Loads training state (step, epoch, config) from metadata
# ✅ Skips already-processed samples via skip_samples
# ✅ Continues from correct epoch and batch position
# ✅ Shows accurate progress bar
```

**What happens during resume:**

1. Script checks `save_dir` for `model.pt` and `model_metadata.json`
2. Loads training metadata (step, epoch, config) from JSON file
3. Loads model weights, optimizer state, scheduler state, and scaler from `model.pt`
4. Calculates `skip_samples = step * batch_size` and sets on dataset
5. Recreates DataLoader so workers pick up the new `skip_samples` value
6. Calculates starting epoch and batch index based on metadata
7. Initializes progress bar at correct position
8. Training continues seamlessly from where it stopped

**Epoch completion:**

- Training continues through all epochs until `max_steps` is reached
- Model is saved at the end of each epoch for checkpointing
- Training only stops when `max_steps` is reached or manually interrupted
- DataLoader is recreated for each new epoch (IterableDatasets are exhausted after one iteration)
- `skip_samples` is automatically reset to 0 by the dataset after each iteration completes
- This ensures each epoch starts from the beginning of the dataset

**Dataset exhaustion handling:**
The training scripts handle three scenarios gracefully:

1. **Dataset exceeds max_steps:**

   - Training stops when `max_steps` is reached (checked after each epoch)
   - All available data is processed up to the step limit
   - Model is saved at the final checkpoint

2. **Dataset smaller than one epoch:**

   - Processes all available batches in the epoch
   - Dataset automatically resets `skip_samples` to 0 after iteration completes
   - Next epoch starts from the beginning of the dataset
   - Training continues until `max_steps` is reached

3. **Dataset smaller than total epochs:**
   - Each epoch processes all available data from the beginning
   - `skip_samples` automatically resets to 0 after each iteration
   - Training continues through all epochs until `max_steps` is reached
   - Useful for small datasets that need multiple passes

**Automatic skip_samples reset:**

- All `IterableDataset` classes automatically reset `skip_samples` to 0 after the first iteration completes
- This happens in the dataset's `__iter__` method when the generator is exhausted
- Ensures subsequent epochs always start from the beginning of the dataset
- Works correctly even if the dataset is exhausted mid-epoch

**Validation during resume:**

- Validation always processes the full validation set
- `skip_samples` is temporarily reset to 0 during validation
- Original `skip_samples` is restored after validation
- This ensures validation metrics are always computed on complete data

**No manual intervention needed** - resuming is fully automatic!

---

## 💡 Tips

✅ **Run LR Finder before each stage** to discover optimal learning rate  
✅ **EMA enabled by default** - better stability and generalization  
✅ **LR Spike enabled by default** - automatically recovers from training plateaus  
✅ **Early stopping prevents endless loops** - fails fast with helpful errors  
✅ **Run stages in parallel** (if multiple GPUs)  
✅ **Start with small data** to verify pipeline  
✅ **Monitor GPU memory** with `nvidia-smi`  
✅ **Save checkpoints frequently** (every 1000 steps)  
✅ **Use tmux/screen** for long training sessions  
✅ **Automatic resuming** - just rerun training command if interrupted  
✅ **Consistent utilities** - all scripts share common checkpoint/resume logic  
✅ **Trust early stopping** - if it triggers, reduce LR or add regularization

**About LR Spike:**
- Automatically detects when validation loss increases consecutively
- Temporarily boosts learning rate to help escape plateaus
- Fully configurable via config (or disable with `"use_lr_spike": false`)
- See [Chapter 36: Optimization Techniques](36-optimization-techniques.md#5-learning-rate-spike-for-plateau-recovery) for details

---

## 📦 After Training: Export for Deployment

Once all stages are complete, you can export your trained model for deployment:

```bash
# Export all components to safetensors
python export.py \
    --omni_ckpt checkpoints/omni_sft_tiny \
    --thinker_ckpt checkpoints/thinker_tiny \
    --audio_ckpt checkpoints/audio_enc_tiny \
    --vision_ckpt checkpoints/vision_tiny \
    --talker_ckpt checkpoints/talker_tiny \
    --output_dir merged_model
```

See [Chapter 46: Model Export and Deployment](46-model-export-deployment.md) for detailed instructions, or [Chapter 47: Quick Start Export](47-quick-start-export.md) for a quick reference.

---

[Continue to Chapter 40: Inference Examples →](40-inference-examples.md)

---
