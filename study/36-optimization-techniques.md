# Chapter 36: Optimization Techniques

[← Previous: Data Preparation](35-data-preparation.md) | [Back to Index](00-INDEX.md) | [Next: Debugging →](37-debugging-troubleshooting.md)

---

## ⚡ Performance Optimizations

Key techniques to speed up training and inference in μOmni.

---

## 🚀 Training Optimizations

### 1. Mixed Precision (FP16)

**What:** Use 16-bit floats instead of 32-bit  
**Benefit:** 2x faster, 50% less memory  
**Enabled by default** in μOmni

```python
# Automatically uses torch.cuda.amp.autocast
with torch.cuda.amp.autocast():
    output = model(input)
```

### 2. Gradient Accumulation

**What:** Accumulate gradients over multiple batches  
**Benefit:** Simulate larger batch sizes without OOM

```json
{
  "batch_size": 4,
  "gradient_accumulation_steps": 4
  // Effective batch size = 16
}
```

**Implementation Details:**

- Loss is scaled by `1 / accumulation_steps` before backward pass
- Gradients accumulate over `accumulation_steps` batches
- Optimizer step occurs only every `accumulation_steps` batches
- **Step counter increments only when optimizer step occurs** - ensures learning rate scheduler, validation, and checkpointing happen at correct intervals

### 3. Gradient Checkpointing

**What:** Trade compute for memory  
**Benefit:** Train larger models on same GPU

```python
model.gradient_checkpointing_enable()
```

### 4. Flash Attention

**What:** Memory-efficient attention implementation  
**Benefit:** 2-4x faster, less memory

```python
# Automatically used if available
pip install flash-attn
```

### 5. Learning Rate Spike for Plateau Recovery

**What:** Automatically boost LR when validation loss increases consecutively  
**Benefit:** Escape local minima without manual intervention  
**Status:** ✅ Enabled by default in all training scripts

**How It Works:**

1. Monitors validation loss at each validation step
2. Counts consecutive increases in validation loss
3. When threshold reached (default: 2 consecutive increases), spikes LR temporarily
4. LR = current_lr × multiplier (default: 5×) for duration (default: 50 steps)
5. Automatically restores LR to normal scheduler value after spike duration

**Example Scenario:**

```
Step 1000: val_loss = 0.500 ✓
Step 1200: val_loss = 0.520 ⚠️ (increase 1/2)
Step 1400: val_loss = 0.540 ⚠️ (increase 2/2) → SPIKE TRIGGERED!
           LR: 3e-4 → 1.5e-3 (5× boost)
Step 1401-1450: Training with spiked LR
Step 1451: LR restored to 3e-4
Step 1600: val_loss = 0.450 ✓ (escaped plateau!)
```

**Configuration:**

```json
{
  "use_lr_spike": true, // Enable/disable (default: true)
  "lr_spike_multiplier": 10.0, // LR boost factor (default: 5.0)
  "lr_spike_duration": 100, // Steps to maintain spike (default: 50)
  "lr_spike_consecutive_increases": 2 // Trigger threshold (default: 2)
}
```

**Presets:**

| Preset             | Multiplier | Duration | Trigger | Use Case                         |
| ------------------ | ---------- | -------- | ------- | -------------------------------- |
| Conservative       | 3.0        | 30       | 3       | Stable training, gentle recovery |
| Balanced (default) | 5.0        | 50       | 2       | General purpose                  |
| Aggressive         | 10.0       | 100      | 2       | Stubborn plateaus                |

**When to Use:**

- ✅ Long training runs (>1000 steps)
- ✅ Training plateaus or loss stagnates
- ✅ Validation loss becomes unstable
- ❌ Very unstable training (frequent NaN/inf)
- ❌ Already using very high LR (>1e-2)
- ❌ Noisy validation set

**Monitoring:**
Watch for log messages during training:

```
⚠️ Validation loss increased: 0.4800 -> 0.5000 (1/2)
⚠️ LR SPIKE TRIGGERED! Spiking LR by 5.0x for 50 steps
ℹ️ LR spike ended. Restored LR to 3.00e-04
```

**Implementation Details:**

- Fully checkpoint-compatible (state saved/loaded)
- Works with all LR schedulers (warmup, cosine decay)
- Compatible with gradient accumulation, AMP, EMA
- Available in: `train_text.py`, `train_vision.py`, `train_audio_enc.py`

**Disabling:**

```json
{
  "use_lr_spike": false
}
```

### 6. CUDA Graphs Compilation

**What:** Compile models with `torch.compile()` using CUDA graphs backend  
**Benefit:** 10-20% speedup, reduced overhead  
**Requirement:** Fixed-length padding for variable-length sequences

```json
{
  "use_compile": true,
  "max_mel_length_percentile": 95.0, // Optional: For audio training (default: 95.0)
  "max_text_length_percentile": 95.0 // Optional: For OCR training (default: 95.0)
  // max_mel_length and max_text_length are auto-calculated - no need to set manually
}
```

**Why Fixed Length?**

- CUDA graphs require uniform tensor shapes across batches
- Variable-length sequences cause "tensor size mismatch" errors
- Fixed padding ensures all batches have identical shapes

**Auto-Calculation:**

- `max_mel_length` and `max_text_length` are **automatically calculated** from your dataset
- Uses **95th percentile** by default to minimize padding while covering 95% of data
- Automatically rounds up to nearest 256 for better memory alignment
- ~5% of samples will be skipped if longer (outliers filtered during dataset iteration)

**Implementation:**

- Audio training: All mel spectrograms padded to auto-calculated `max_mel_length` (longer samples skipped)
- OCR training: All text sequences padded to auto-calculated `max_text_length` (longer samples skipped)
- Collate functions in `omni/utils.py` handle padding automatically:
  - `collate_mel_fn()` - For mel-only batches (talker training)
  - `collate_mel_text_fn()` - For mel+text batches (audio encoder training)
  - `collate_mel_audio_fn()` - For mel+audio batches (vocoder training)
- All collate functions support `max_mel_length` parameter for fixed-length padding

**Memory Trade-off:**

- Slightly more memory due to padding
- But enables CUDA graphs optimization (10-20% faster)
- Worth it for most use cases

**Configuring Percentiles:**

- **Higher percentile (99.0):** More coverage, more padding, fewer samples skipped
- **Lower percentile (90.0):** Less padding, more samples skipped, less memory
- **Default (95.0):** Good balance - covers 95% of data with minimal padding, skips 5% outliers

**Optional Manual Override:**

- You can manually set `max_mel_length` or `max_text_length` in config to override auto-calculation
- Useful if you know your dataset characteristics and want specific values

---

## 🎯 Inference Optimizations

### 1. KV Caching

**Essential for generation!** Reuses computed key-value pairs.

**Without KV cache:**

- Generate 100 tokens: ~30 seconds
- Recomputes attention for all previous tokens every step

**With KV cache:**

- Generate 100 tokens: ~3 seconds
- 10x speedup!

### 2. Batch Inference

Process multiple inputs simultaneously:

```python
responses = model.chat_batch([
    "Question 1?",
    "Question 2?",
    "Question 3?"
])
```

### 3. Quantization (INT8)

**What:** Convert weights to 8-bit integers  
**Benefit:** 4x smaller model, faster inference  
**Trade-off:** Slight quality degradation

```python
model = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)
```

---

## 💾 Memory Optimizations

### 1. Streaming Dataset Loading

**What:** Stream data directly from files without pre-loading into RAM  
**Benefit:** Reduces RAM usage by 90%+ for large datasets  
**Status:** ✅ Implemented in all μOmni training scripts (2024 optimization)

**Technical Approach:**

- All datasets use `IterableDataset` for true streaming
- Sequential I/O with large buffers (8MB) for efficiency
- Direct file iteration - no offset tracking or caching
- Worker sharding for multi-process data loading
- Buffer-based shuffling for randomization

**Optimized Datasets (All Training Scripts):**

| Dataset            | Files                | Optimization              | Memory Savings   |
| ------------------ | -------------------- | ------------------------- | ---------------- |
| **TextDataset**    | `train_text.py`      | Direct line streaming     | No pre-loading   |
| **ASRDataset**     | `train_audio_enc.py` | CSV row streaming         | No pre-loading   |
| **TTSDataset**     | `train_talker.py`    | CSV row streaming         | No pre-loading   |
| **OCRDataset**     | `train_ocr.py`       | CSV row streaming         | No pre-loading   |
| **ImgCapDataset**  | `train_vision.py`    | JSON item streaming       | No pre-loading   |
| **VocoderDataset** | `train_vocoder.py`   | CSV row streaming         | No pre-loading   |
| **MixDataset**     | `sft_omni.py`        | All three types streaming | Combined savings |

**Audio Loading:**

- ✅ **Automatic fallback:** `load_audio()` function automatically falls back to `torchaudio.load()` if torchcodec unavailable
- ✅ **No dependencies required:** Works with or without torchcodec package
- ✅ **Error handling:** Gracefully handles missing audio files (skips with continue)

**Implementation Details:**

**Text Files:**

```python
# Streaming: Reads line-by-line directly
def __iter__(self):
    with open(self.path, 'r', encoding='utf-8', errors='ignore', buffering=8192*1024) as f:
        for idx, line in enumerate(f):
            # Worker sharding, train/val split, skip_samples support
            text = line.strip()
            if text:
                # Process and yield immediately
                yield processed_text
```

**CSV Files:**

```python
# Streaming: Uses csv.DictReader directly
def __iter__(self):
    with open(self.csv_path, 'r', encoding='utf-8', errors='ignore', buffering=8192*1024) as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            # Worker sharding, train/val split, skip_samples support
            # Process and yield immediately
            yield processed_data
```

**JSON Files:**

```python
# Streaming: Loads JSON once, then iterates
def __iter__(self):
    with open(self.manifest_path, 'r', encoding='utf-8') as f:
        items = json.load(f)
    for idx, item in enumerate(items):
        # Worker sharding, train/val split, skip_samples support
        # Process and yield immediately
        yield processed_item
```

**Memory Savings Examples:**

- **10M line text file**: ~500MB → ~0MB (only current line in memory)
- **1M row CSV**: ~200MB → ~0MB (only current row in memory)
- **100K image JSON**: ~50MB → ~50MB (JSON loaded once, then streamed)
- **Combined (MixDataset)**: All three optimizations applied simultaneously

**Key Benefits:**

- ✅ RAM usage now typically **lower than VRAM** (was opposite before)
- ✅ Can train on systems with limited RAM (8GB+)
- ✅ No cache files needed - simpler and cleaner
- ✅ True streaming - data processed on-demand
- ✅ Efficient resuming via `skip_samples` parameter (automatically handled by `setup_resume_data_loading()`)
- ✅ **Automatic skip_samples reset** - All `IterableDataset` classes automatically reset `skip_samples` to 0 after each iteration completes
- ✅ Worker sharding for multi-process data loading
- ✅ Buffer-based shuffling for randomization (controlled by `shuffle_buffer_size` config)
- ✅ Automatic checkpoint detection and resuming (via `load_checkpoint()`)
- ✅ Proper validation on full dataset (via `ValidationSkipSamplesContext`)
- ✅ **Note:** IterableDatasets handle shuffling internally - do not use `shuffle` parameter in DataLoader
- ✅ **Dataset exhaustion handling** - Gracefully handles datasets smaller than one epoch or total epochs

### 2. Efficient Tokenizer Training

**What:** Train tokenizers on entire dataset efficiently  
**Benefit:** Train on full dataset with optimized settings

**Implementation:**

- **Plain text files:** Passed directly to SentencePiece (no streaming, no temp files)
- **CSV/JSON files:** Stream extraction to temp file in `data/.temp/` (streams row-by-row/item-by-item to extract text)
- Processes entire dataset (not just samples) efficiently
- Temporary files auto-cleaned after training
- **Always enables `train_extremely_large_corpus=True`:** Uses 64-bit indexing for maximum file size compatibility
- **BPE model type:** Faster than Unigram, good balance of speed and quality
- **Default speed optimization:** `input_sentence_size=10000000` (10M sentences) limits training data for faster training by default
- **Use all data:** Set `input_sentence_size=0` to use entire corpus (slower but uses more data)

**Memory Behavior:**

- **Plain text:** SentencePiece loads entire file into memory during training (no streaming)
- **CSV/JSON extraction:** Streams data extraction (avoids loading structured data into memory), but SentencePiece still loads the extracted temp file into memory
- **Temp files:** Only used for CSV/JSON text extraction, stored in `data/.temp/` and auto-cleaned

**Note:** SentencePiece loads the entire file into memory during training (whether original or extracted temp file). The `train_extremely_large_corpus` flag enables 64-bit indexing (instead of 32-bit) to handle files > 2GB, but doesn't reduce memory usage. Streaming is only used for extracting text from CSV/JSON structured data.

### 3. Resumable Preprocessing

**What:** All preprocessing operations can resume if interrupted  
**Benefit:** No need to restart from beginning if process is stopped

**Resumable Operations:**

- ✅ **Tokenizer training:** SentencePiece handles large files directly (no streaming needed for plain text)
- ✅ **CSV/JSON extraction:** Temp files created in `data/.temp/` (only when needed)
- ✅ **Vocabulary building:** Saves progress every 10K items (vision, OCR)
- ✅ **Token counting:** Saves progress every 10K samples (text, CSV, images)
- ✅ **Training loops:** Already resumable via checkpoints

**Checkpoint Locations:**

- Vocabulary building: `{save_dir}/vocab_build_checkpoint.json`
- Token counting: `{file_path}.token_count_checkpoint.json`
- OCR vocabulary: `{csv_path}.vocab_checkpoint.json`
- All checkpoints auto-cleaned after successful completion

**Related Optimizations:**

- Removed `gc.collect()` calls (Python's GC handles this automatically)
- Removed `torch.cuda.empty_cache()` calls (PyTorch manages CUDA memory efficiently)
- These manual calls were unnecessary and could actually hurt performance

### 4. DataLoader Workers

**What:** Parallel data loading processes  
**Trade-off:** More workers = faster loading but more RAM

```json
{
  "num_workers": 2 // Default: 2 workers
  // Reduce to 0 or 1 if RAM is limited
}
```

**Recommendation:**

- **High RAM (32GB+)**: `num_workers: 2-4`
- **Medium RAM (16GB)**: `num_workers: 1-2`
- **Low RAM (8GB)**: `num_workers: 0-1`

### 5. Batch Size Tuning

**What:** Adjust batch size based on available memory  
**Benefit:** Maximize GPU utilization without OOM

```json
{
  "batch_size": 4, // Start conservatively
  "gradient_accumulation_steps": 4 // Simulate batch_size=16
}
```

**Strategy:**

1. Start with `batch_size: 4` (conservative, works on most GPUs)
2. Increase gradually (8, 16, 32) until you hit OOM
3. Use gradient accumulation to simulate larger batches without OOM

---

## 🧪 Exponential Moving Average (EMA)

### What is EMA?

**Concept:** Maintain a moving average of model weights during training for improved stability and generalization.

```
EMA maintains two sets of weights:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Primary Weights (θ):
- Updated by gradient descent
- Used during training
- Can be noisy/unstable

Shadow Weights (θ_ema):
- Smoothed average of primary weights
- Updated with exponential decay
- More stable, better generalization
- Used for validation and inference

Update Rule:
θ_ema ← decay × θ_ema + (1 - decay) × θ

Where decay is typically 0.999
```

### Why Use EMA?

```
Benefits of EMA:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. TRAINING STABILITY:
   - Smooths out noisy gradients
   - Reduces oscillations
   - More stable loss curves
   ✓ Better convergence

2. BETTER GENERALIZATION:
   - Averages out random fluctuations
   - Less overfitting
   - Improved validation performance
   ✓ Better final model quality

3. NO EXTRA COST:
   - Minimal memory overhead (2x weights)
   - Negligible compute overhead
   - Easy to implement
   ✓ Free performance boost!

Research shows: 0.5-2% improvement in validation metrics
```

### EMA in μOmni

**Implementation:**

```python
# In omni/utils.py
class EMA:
    """Exponential Moving Average for model weights"""
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {name: p.clone().detach()
                       for name, p in model.named_parameters()
                       if p.requires_grad}

    def update(self, model):
        """Update shadow weights after optimizer step"""
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = (
                    self.decay * self.shadow[name] +
                    (1 - self.decay) * param.data
                )

    def apply_shadow(self, model):
        """Temporarily use EMA weights (for validation)"""
        self.backup = {name: p.clone()
                       for name, p in model.named_parameters()
                       if p.requires_grad}
        for name, param in model.named_parameters():
            if param.requires_grad:
                param.data = self.shadow[name]

    def restore(self, model):
        """Restore original weights (after validation)"""
        for name, param in model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
```

**Usage Pattern:**

```python
# 1. Initialize EMA (after optimizer)
if config.use_ema:
    ema = EMA(model, decay=config.ema_decay)  # decay=0.999

# 2. Update EMA after each optimizer step
optimizer.step()
if config.use_ema:
    ema.update(model)

# 3. Use EMA weights for validation
if config.use_ema:
    ema.apply_shadow(model)
val_loss = validate(model, val_dataloader)
if config.use_ema:
    ema.restore(model)

# 4. Save EMA state in checkpoints
torch.save({
    'model': model.state_dict(),
    'optimizer': optimizer.state_dict(),
    'ema': ema.state_dict() if config.use_ema else None,
    ...
}, checkpoint_path)
```

**Configuration:**

All config files have EMA enabled by default:

```json
{
  "use_ema": true,
  "ema_decay": 0.999
}
```

**When to Use:**

- ✅ **Always recommended** - minimal cost, consistent benefits
- ✅ **Especially useful** for small datasets (reduces overfitting)
- ✅ **Critical** for long training runs (smooths instability)

**Decay Parameter Guidelines:**

```
Common decay values:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

0.99:   Fast averaging, more responsive
0.999:  Balanced (μOmni default)
0.9999: Very slow averaging, maximum smoothing

Effective window (steps to ~63% influence):
- 0.99   → ~100 steps
- 0.999  → ~1000 steps
- 0.9999 → ~10000 steps

Rule of thumb: decay = 1 - 1/window_size
```

---

## 🔍 Learning Rate Finder

### What is LR Finder?

**Concept:** Automatically discover optimal learning rate before training using range test method.

```
Traditional approach (trial-and-error):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Try lr=1e-3 → Too high, diverges ❌
Try lr=1e-5 → Too low, slow ❌
Try lr=5e-4 → Better, but is it optimal? 🤔
Waste hours/days guessing...

LR Finder approach (systematic):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Run lr_finder for 5-10 minutes
Get optimal lr automatically ✓
Start training with confidence ✓
```

### Smith 2017 Range Test Method

**Algorithm:**

```
1. Start with very small lr (e.g., 1e-7)
2. Gradually increase lr exponentially
3. Track loss at each lr value
4. Stop when loss diverges
5. Find lr with steepest negative slope
6. Recommend: lr = steepest_slope_lr / 10
```

**Loss vs Learning Rate Curve:**

```
Loss
  │
  │                                    ╱─────
  │                                 ╱─
  │                              ╱─
  │                           ╱─  ← Diverging
  │                        ╱─
  │                     ╱─
  │                  ╱─
  │               ╱─  ← Steepest descent
  │            ╱─      (optimal region)
  │         ╱─
  │      ╱─
  │   ╱─  ← Too slow
  │╱─
  └──────────────────────────────────────────
    1e-7  1e-6  1e-5  1e-4  1e-3  1e-2  1e-1
                  Learning Rate

Optimal LR: Where loss decreases fastest
Suggested: ~1e-4 (at steepest negative slope)
```

### Using LR Finder in μOmni

**Quick Start:**

```bash
# Find optimal LR for Thinker training
python find_lr.py --config configs/thinker_tiny.json \
  --model_type thinker \
  --output_plot lr_finder_thinker.png

# For Vision Encoder
python find_lr.py --config configs/vision_tiny.json \
  --model_type vision \
  --output_plot lr_finder_vision.png

# For Audio Encoder
python find_lr.py --config configs/audio_enc_tiny.json \
  --model_type audio_enc \
  --output_plot lr_finder_audio.png

# For Talker
python find_lr.py --config configs/talker_tiny.json \
  --model_type talker \
  --output_plot lr_finder_talker.png

# For OCR
python find_lr.py --config configs/ocr_tiny.json \
  --model_type ocr \
  --output_plot lr_finder_ocr.png
```

**Example Output:**

```
Running LR Finder for thinker...
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Step 10/100 | LR: 1.00e-07 | Loss: 8.543
Step 20/100 | LR: 5.62e-07 | Loss: 8.124
Step 30/100 | LR: 3.16e-06 | Loss: 7.542
Step 40/100 | LR: 1.78e-05 | Loss: 6.231  ← Steepest descent
Step 50/100 | LR: 1.00e-04 | Loss: 5.142
Step 60/100 | LR: 5.62e-04 | Loss: 5.834
Step 70/100 | LR: 3.16e-03 | Loss: 9.234  ← Diverging
Stopping early (loss increased by >4x)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Suggested Learning Rate: 2.00e-05
(Based on steepest descent at LR 1.78e-05, divided by 10)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Plot saved to: lr_finder_thinker.png
```

**Interpreting Results:**

1. **Suggested LR:** Use this value in your config
2. **Plot:** Visual confirmation of optimal range
3. **Too Flat:** If curve is flat, increase `end_lr`
4. **Immediate Spike:** If loss spikes immediately, decrease `start_lr`

**Advanced Options:**

```bash
# Custom LR range
python find_lr.py --config configs/thinker_tiny.json \
  --model_type thinker \
  --start_lr 1e-8 \
  --end_lr 1.0 \
  --num_steps 200

# Use more data samples
python find_lr.py --config configs/vision_tiny.json \
  --model_type vision \
  --num_steps 500
```

**When to Use:**

- ✅ **Starting new training** - discover optimal LR
- ✅ **After dataset changes** - LR may need adjustment
- ✅ **After model architecture changes** - optimal LR changes
- ✅ **Training plateaus** - check if LR needs tuning

**Tips:**

```
Best Practices:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. Run on small subset of data (find_lr.py does this automatically)
2. Use the suggested LR or slightly lower (e.g., 0.5-0.8x suggested)
3. If training is unstable, try 0.5x suggested LR
4. If training is too slow, try 1.5x suggested LR
5. Rerun LR finder if you change model architecture significantly
```

---

## 🛑 Early Stopping for Validation Spikes

### The Problem: Endless Reload Loops

**Before Early Stopping:**

```
Training loop without early stopping:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Step 1000 | Val Loss: 2.34 ✓ (save checkpoint)
Step 2000 | Val Loss: 2.89 ✗ (worse! reload checkpoint)
Step 2000 | Val Loss: 2.91 ✗ (worse again! reload)
Step 2000 | Val Loss: 2.88 ✗ (still worse! reload)
Step 2000 | Val Loss: 2.90 ✗ (reload again...)
... infinite loop! ❌

Problem: No escape from validation spike pattern
User must manually kill training and adjust hyperparameters
```

**After Early Stopping:**

```
Training loop WITH early stopping:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

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

Training halted safely ✓
```

### Implementation

**How It Works:**

```python
# In all training scripts
consecutive_reloads = 0
MAX_CONSECUTIVE_RELOADS = 2

# After validation
if val_loss >= best_val_loss:
    # Validation got worse
    consecutive_reloads += 1
    print(f"Validation loss increased. Reloading... "
          f"(consecutive reloads: {consecutive_reloads}/{MAX_CONSECUTIVE_RELOADS})")

    if consecutive_reloads >= MAX_CONSECUTIVE_RELOADS:
        # Stop training with helpful message
        raise RuntimeError(
            f"Training stopped after {MAX_CONSECUTIVE_RELOADS} consecutive "
            "validation loss increases. This usually indicates:\n"
            "- Learning rate too high\n"
            "- Overfitting\n"
            "- Need different hyperparameters\n\n"
            "Solutions:\n"
            "1. Reduce learning rate by 2-5x\n"
            "2. Enable/increase regularization (dropout, weight_decay)\n"
            "3. Add more training data\n"
            "4. Check for data quality issues"
        )

    # Reload checkpoint
    load_checkpoint(model, optimizer, best_checkpoint_path)
else:
    # Validation improved
    consecutive_reloads = 0  # Reset counter
    best_val_loss = val_loss
    save_checkpoint(...)
```

**Why 2 Consecutive Spikes?**

```
Why not stop after 1 spike?
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Single spikes can be normal:
- Random validation batch variance
- Temporary learning plateau
- Natural training oscillations

Stopping after 1 spike = too aggressive ❌

Why not allow 3+ spikes?
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

2 consecutive spikes indicate real problem:
- Systematic issue with hyperparameters
- Unlikely to be random variance
- Model won't recover by itself

3+ spikes = wasting time/compute ❌

2 consecutive spikes = sweet spot ✓
- Allows temporary variance
- Catches systematic problems
- Saves time and compute
```

### When Early Stopping Triggers

**Common Scenarios:**

```
1. LEARNING RATE TOO HIGH
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Training Loss: Decreasing (looks good)
Validation Loss: Spiking repeatedly
→ Model overfitting to training data

Solution: Reduce LR by 2-5x, or use LR Finder

2. OVERFITTING
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Training Loss: Very low (< 0.5)
Validation Loss: High and increasing
→ Model memorizing training data

Solution: More data, regularization (dropout, weight_decay)

3. DATA QUALITY ISSUES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Both losses: Unstable/erratic
→ Corrupted data, wrong labels, etc.

Solution: Inspect data, clean dataset

4. BATCH SIZE TOO SMALL
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Validation Loss: High variance
→ Noisy gradient estimates

Solution: Increase batch_size or gradient_accumulation_steps
```

**Debugging Tips:**

```bash
# If early stopping triggers:

# 1. Check the training curve in TensorBoard
tensorboard --logdir=logs/

# 2. Inspect last few validation runs
# Look for sudden spikes vs gradual increase

# 3. Try reducing learning rate first (safest fix)
# In config.json, change:
"learning_rate": 3e-4  →  "learning_rate": 1e-4

# 4. If that doesn't help, add regularization
"weight_decay": 0.0  →  "weight_decay": 0.01

# 5. Use LR Finder to find optimal LR
python find_lr.py --config configs/your_config.json \
  --model_type your_model_type
```

### Benefits

```
Advantages of Early Stopping:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. SAVES TIME:
   - No infinite reload loops
   - Fast failure with clear error message
   ✓ Hours saved per failed run

2. CLEARER DEBUGGING:
   - Error message explains likely causes
   - Suggests concrete solutions
   ✓ Faster iteration

3. PREVENTS CONFUSION:
   - Users don't wonder "why is it stuck?"
   - Clear indication something is wrong
   ✓ Better user experience

4. AUTOMATIC:
   - No configuration needed
   - Works out-of-the-box
   ✓ Zero overhead
```

---

## 💡 Best Practices

✅ **Always use FP16** for training  
✅ **Enable KV caching** for generation  
✅ **Use Flash Attention** if available  
✅ **Gradient accumulation** for large batches  
✅ **Streaming datasets enabled** by default (no action needed)  
✅ **Direct file iteration** - no cache files needed  
✅ **Efficient tokenizer training** - plain text passed directly to SentencePiece  
✅ **Resumable preprocessing** - safe to interrupt and resume  
✅ **Monitor GPU memory** with `nvidia-smi`  
✅ **Monitor RAM usage** - should be much lower than VRAM now  
✅ **Reduce num_workers** if RAM is limited  
✅ **Use EMA** for better stability and generalization (enabled by default)  
✅ **Run LR Finder** before starting new training runs  
✅ **Trust early stopping** - it catches problems early and saves time

---

[Continue to Chapter 37: Debugging →](37-debugging-troubleshooting.md)

---
