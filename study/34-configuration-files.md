# Chapter 34: Configuration Files Guide

[← Previous: Code Structure](33-code-structure.md) | [Back to Index](00-INDEX.md) | [Next: Data Preparation →](35-data-preparation.md)

---

## 🎯 Understanding Configuration Files

μOmni uses JSON configuration files for all training stages. This chapter explains each parameter.

---

## 📊 Configuration Files Overview

```
configs/
├── thinker_tiny.json      # Stage A: Language model
├── audio_enc_tiny.json    # Stage B: Audio encoder
├── vision_tiny.json       # Stage C: Vision encoder
├── talker_tiny.json       # Stage D: Speech generation
├── vocoder_tiny.json      # Optional: HiFi-GAN neural vocoder
├── ocr_tiny.json          # Optional: OCR text extraction
└── omni_sft_tiny.json     # Stage E: Multimodal SFT
```

---

## 📝 Common Parameters

### Model Architecture

```json
{
  "d_model": 256, // Embedding dimension
  "n_layers": 4, // Transformer layers
  "n_heads": 4, // Attention heads
  "d_ff": 1024, // FFN hidden size (usually 4×d_model)
  "dropout": 0.1, // Dropout rate (0-1)
  "ctx_len": 512 // Context length (tokens)
}
```

### Training Hyperparameters

```json
{
  "batch_size": 16, // Examples per batch
  "num_epochs": 10, // Training epochs
  "learning_rate": 3e-4, // LR (0.0005)
  "warmup_steps": 1000, // LR warmup
  "max_grad_norm": 1.0, // Gradient clipping
  "weight_decay": 0.01, // L2 regularization
  "val_loss_threshold": 0.05 // Reload if val loss spikes > last_best + threshold
}
```

### Data & Checkpointing

```json
{
  "data_path": "data/text/corpus.txt",
  "save_every": 1000, // Save checkpoint frequency
  "eval_every": 500, // Evaluation frequency
  "checkpoint_dir": "checkpoints/thinker_tiny/",
  "shuffle_buffer_size": 10000 // Buffer size for streaming dataset shuffling
}
```

**Note on Shuffling:**

- All datasets use `IterableDataset` which handles shuffling internally
- `shuffle_buffer_size`: Controls randomization in streaming datasets (default: 10000)
- Larger values = more randomization but more memory
- Set to 0 to disable shuffling (validation datasets use 0)
- **Do not use `shuffle` parameter in DataLoader** - IterableDatasets don't support it

---

## 🤖 Automatic Config Updates Based on Dataset Size

**Recommended:** After downloading datasets, automatically update training parameters:

```bash
# Update all configs based on actual dataset sizes (default)
python scripts/update_configs_from_data.py

# Preview changes without modifying files
python scripts/update_configs_from_data.py --dry-run

# Update only specific configs
python scripts/update_configs_from_data.py --config thinker vision

# Update multiple specific configs
python scripts/update_configs_from_data.py --config audio_enc talker vocoder

# Dry run for specific configs
python scripts/update_configs_from_data.py --dry-run --config omni_sft

# Skip text tokenization and assume 8B tokens (fast mode for large datasets)
python scripts/update_configs_from_data.py --skip-text-tokenization --assume-text-tokens 8000000000
```

**Supported config names:**

- `thinker` - Text-only training (thinker_tiny.json)
- `audio_enc` - Audio encoder training (audio_enc_tiny.json)
- `vision` - Vision encoder training (vision_tiny.json)
- `talker` - Talker training (talker_tiny.json)
- `omni_sft` - Multimodal SFT training (omni_sft_tiny.json)
- `ocr` - OCR training (ocr_tiny.json)
- `vocoder` - Vocoder training (vocoder_tiny.json)

**What gets updated:**

- `max_steps`: Calculated using research-based formulas:
  - **Text/Multimodal SFT:** From token count, batch size, and context length
  - **Vision/Audio/Talker/OCR:** From sample count and batch size
- `max_epochs`: Based on dataset size (1-3 for very large, 5-10 for small)
- `warmup_steps`: 4% of total steps (research-based, typically 3-5%, capped at 10K)
- `batch_size`: Automatically adjusted based on model size (larger models = smaller batch sizes)
- `gradient_accumulation_steps`: Automatically adjusted to maintain effective batch size
- `val_freq`: Every 500-1000 steps or 10% of steps per epoch
- `checkpoint_freq`: Every 5000-10000 steps or 1 per epoch
- Data paths: Automatically updated to production files if they exist
- **Model dimensions:** Validated and fixed for compatibility (OCR vision_d_model, SFT thinker/audio/vision dimensions)
- **Data paths:** Validated to ensure all training data files exist

**Training step calculation methods:**

- **Text training (`train_text.py`):** Uses **tokens** for step calculation
  - Each sample is tokenized to `ctx_len` tokens
  - Steps = tokens / (batch_size × ctx_len)
  - Token count recommendations:
    - **Very large (>100M tokens):** 1-3 epochs
    - **Large (50M-100M tokens):** 2-4 epochs
    - **Medium (10M-50M tokens):** 3-6 epochs
    - **Small (<10M tokens):** 5-10 epochs
- **Vision training (`train_vision.py`):** Uses **samples** for step calculation
  - Contrastive learning (image-caption pairs)
  - Steps = samples / batch_size
  - Sample count recommendations:
    - **Very large (>1M samples):** 1-3 epochs
    - **Large (500K-1M samples):** 2-4 epochs
    - **Medium (100K-500K samples):** 3-6 epochs
    - **Small (<100K samples):** 5-10 epochs
- **Audio training (`train_audio_enc.py`):** Uses **samples** for step calculation
  - CTC loss (audio-transcription pairs)
  - Steps = samples / batch_size
  - Same sample count recommendations as vision
- **Talker training (`train_talker.py`):** Uses **samples** for step calculation
  - TTS generation (text-audio pairs)
  - Steps = samples / batch_size
  - Same sample count recommendations as vision
- **Vocoder training (`train_vocoder.py`):** Uses **samples** for step calculation
  - Mel-to-audio generation (mel spectrogram-audio pairs)
  - Steps = samples / batch_size
  - Same sample count recommendations as vision
- **OCR training (`train_ocr.py`):** Uses **samples** for step calculation
  - Image-text pairs
  - Steps = samples / batch_size
  - Same sample count recommendations as vision
- **Multimodal SFT (`sft_omni.py`):** Uses **tokens** for step calculation
  - Text-based training with multimodal embeddings
  - Steps = tokens / (batch_size × ctx_len)
  - Same token count recommendations as text

**Token counting (for reference):**

- The script counts tokens using the BPE tokenizer for reference:
  - **Text:** Tokens from text corpus
  - **Images:** Tokens from captions (not used for step calculation, only reference)
  - **Audio:** Tokens from transcriptions (not used for step calculation, only reference)
  - **OCR:** Tokens from extracted text (not used for step calculation, only reference)
- If no tokenizer exists, one will be created automatically from the data
- **Important:** For vision/audio/talker/OCR, token counts are shown for reference only. Step calculation uses sample counts.
- **Fast mode:** Use `--skip-text-tokenization --assume-text-tokens N` to skip tokenization and use an assumed token count (e.g., 8000000000 for 8B tokens).

**Memory-efficient processing:**

- Tokenizer training: Plain text passed directly to SentencePiece. CSV/JSON streams text extraction to temp file.
- Temp files: Only used for CSV/JSON text extraction, stored in `data/.temp/` and auto-cleaned
- Token counting: Streams files line-by-line with automatic resume support
- All operations are resumable - if interrupted, will continue from last checkpoint
- Checkpoints saved every 10K samples/lines for safe resumption
- Datasets: All use streaming `IterableDataset` (no cache files needed)

**Files checked:**

- Text: `data/text/production_corpus.txt` or `data/text/tiny_corpus.txt`
- Images: `data/images/production_annotations.json` or `data/images/annotations.json`
- Audio: `data/audio/production_asr.csv` or `data/audio/asr.csv`
- TTS: `data/audio/production_tts.csv` or `data/audio/tts.csv`
- OCR: `data/ocr/production_ocr.csv` or `data/ocr/ocr_train.csv`
- Vocoder: Uses same audio data as TTS/ASR (`data/audio/production_tts.csv` or `production_asr.csv`)

**Selective updates:**

- Use `--config` to update only specific configs (e.g., `--config thinker vision`)
- When updating specific configs, only those configs are processed (others are skipped)
- TTS data is automatically loaded if either `talker` or `vocoder` is selected
- All configs are updated by default if `--config` is not specified

**Model size integration:**

- The script calculates model size from config files using mathematical formulas
- Batch size and gradient accumulation are automatically adjusted based on model size:
  - **Very large models (>100M params):** Smaller batch size, more gradient accumulation
  - **Large models (50M-100M params):** Moderate batch size, some gradient accumulation
  - **Medium models (10M-50M params):** Normal batch size, minimal accumulation
  - **Small models (<10M params):** Larger batch size, no accumulation needed
- Effective batch size is maintained: `EBS = batch_size × gradient_accumulation_steps`
- This ensures optimal memory usage while maintaining training stability

**Research-based formulas:**

- **Effective Batch Size:** `EBS = Micro Batch Size × Gradient Accumulation × Data Parallel`
- **Text/Multimodal SFT:**
  - **Tokens per step:** `tokens_per_step = EBS × context_length`
  - **Steps per epoch:** `steps_per_epoch = training_tokens / tokens_per_step`
- **Vision/Audio/Talker/OCR:**
  - **Steps per epoch:** `steps_per_epoch = training_samples / EBS`
- **All training types:**
  - **Total steps:** `max_steps = steps_per_epoch × recommended_epochs`
  - **Warmup steps:** 4% of total steps (based on research showing 3-5% is optimal)

**Note:** The script only checks production and synthetic files, ignoring intermediate dataset files.

---

## 📊 Dataset Filtering (Percentile-Based Outlier Skipping)

**All datasets now use percentile-based filtering** to skip outliers instead of truncating data. This ensures clean, properly-sized samples while minimizing data loss.

### How Filtering Works

**During Dataset Iteration:**

1. Each dataset loads a sample (text line, audio file, image+caption, etc.)
2. Measures the sample's length (tokens, mel frames, characters, etc.)
3. Compares against percentile-based threshold (from config or auto-calculated)
4. If length exceeds threshold → **skip sample** (via `continue` in `__iter__`)
5. If length is within threshold → **yield sample** for training

**Benefits:**

- ✅ **No truncation** - preserves data integrity
- ✅ **Clean samples** - all samples fit within context/length limits
- ✅ **Minimal loss** - typically skips only 5% of data (at 95th percentile)
- ✅ **Memory efficient** - padding only up to percentile threshold, not max possible length
- ✅ **Error tracking** - `get_error_stats()` shows how many samples skipped

### ASR Dataset (Audio Encoder)

**Three filtering checks:**

1. **Text length:** `len(text) > cfg.get('max_text_len', 512)` → skip
2. **Mel length:** `mel.shape[0] > cfg.get('max_mel_length', auto-calculated)` → skip
3. **CTC validation:** `output_frames < text_length` → skip (prevents CTC alignment failures)
   - `output_frames = mel_frames // downsample_factor`
   - Ensures acoustic frames can accommodate text length

**Error tracking:** `exceeds_max_len`, `ctc_too_short`

**Config parameters:**
```json
{
  "max_mel_length_percentile": 95.0,  // Auto-calculate from dataset
  "max_text_len": 512  // Manual override
}
```

### TTS Dataset (Talker)

**One filtering check:**

1. **Mel length:** `mel.shape[0] > cfg.get('max_mel_length', auto-calculated)` → skip

**Error tracking:** `exceeds_max_len`

**Config parameters:**
```json
{
  "max_mel_length_percentile": 95.0  // Auto-calculate from dataset
}
```

### OCR Dataset

**One filtering check:**

1. **Text length:** `len(text) > cfg.get('max_text_length', auto-calculated)` → skip

**Error tracking:** `exceeds_max_len`

**Config parameters:**
```json
{
  "max_text_length_percentile": 95.0  // Auto-calculate from dataset
}
```

### Vocoder Dataset

**Two filtering checks:**

1. **Audio length:** `len(audio) > max_audio_length` → skip
2. **Mel length:** `mel.shape[0] > max_mel_length` → skip

**Error tracking:** `exceeds_max_len`

**Config parameters:**
```json
{
  "max_audio_length": 8192,  // Fixed for memory optimization
  "max_mel_length": 512  // Calculated from max_audio_length
}
```

### Text Dataset (Thinker, SFT)

**One filtering check:**

1. **Token length:** `len(tokens) > ctx_len` → skip

**Special features:**

- **Sentence splitting:** Splits text into sentences using regex `(?<=[.!?])\s+`
- **Better boundaries:** Provides semantic boundaries instead of arbitrary line breaks
- **Auto context length:** Calculates optimal `ctx_len` from 95th percentile of token lengths

**Error tracking:** `exceeds_max_len`

**Config parameters:**
```json
{
  "use_sentences": true,  // Enable sentence-based splitting
  "ctx_len_sample_size": 1000000,  // Samples for analysis
  "ctx_len_percentile": 95.0  // Percentile threshold
}
```

### Monitoring Filtered Samples

**All datasets support error statistics:**

```python
# After training epoch
stats = dataset.get_error_stats()
print(f"Samples processed: {stats['total_samples']}")
print(f"Samples skipped (exceeds_max_len): {stats['exceeds_max_len']}")
print(f"Samples skipped (ctc_too_short): {stats['ctc_too_short']}")  # ASR only
```

**Typical results (95th percentile):**
- ~95% of samples processed successfully
- ~5% skipped as outliers
- Minimal impact on training data coverage

### Adjusting Percentile Thresholds

**Higher percentile (e.g., 99.0):**
- ✅ More data coverage (99% of samples)
- ❌ More padding (inefficient memory usage)
- ❌ Slower training (larger batch tensors)

**Lower percentile (e.g., 90.0):**
- ✅ Less padding (efficient memory)
- ✅ Faster training (smaller batch tensors)
- ❌ Less data coverage (90% of samples)

**Recommended:** 95th percentile (default) - good balance between coverage and efficiency

---

## 🔧 CUDA Graphs Compatibility (Fixed-Length Padding)

**Important:** When using `use_compile: true` with CUDA graphs backend, all batches must have uniform tensor shapes. Variable-length sequences are automatically padded to fixed maximum lengths.

### Audio Training (`train_audio_enc.py`, `train_talker.py`)

```json
{
  "use_compile": true,
  "max_mel_length_percentile": 95.0 // Optional: Percentile for auto-calculation (default: 95.0)
  // max_mel_length is auto-calculated from dataset - no need to set manually
}
```

**Auto-Calculation:**

- `max_mel_length` is **automatically calculated** from your dataset during training
- Uses **95th percentile** by default to minimize padding while covering 95% of data
- Rounds up to nearest 256 for better memory alignment
- ~5% of samples will be skipped if longer (outliers filtered during dataset iteration)

**How it works:**

- Training script analyzes all audio files in your dataset
- Calculates mel spectrogram lengths for each file
- Uses 95th percentile to determine optimal `max_mel_length`
- Ensures minimal padding while covering most of your data

**Frame Rate Reference:**

- **Audio Encoder:** Frame rate = 16000 Hz / 160 hop_length = 100 frames/second
  - 60 seconds = 6000 frames
  - 20 seconds = 2000 frames
- **Talker:** Frame rate = 16000 Hz / 1280 hop_length = 12.5 frames/second (with frame_ms=80)
  - 60 seconds = 750 frames
  - 20 seconds = 250 frames

**Optional Override:**

- You can manually set `max_mel_length` in config to override auto-calculation
- You can adjust `max_mel_length_percentile` (e.g., 99.0 for more coverage, 90.0 for less padding)

**Memory Impact:**

- Each frame: ~128 mel bins × 4 bytes = 512 bytes
- Per sample: `max_mel_length × 512 bytes`
- Per batch: `batch_size × max_mel_length × 512 bytes`

**Check your dataset (optional):**

```bash
# (script removed) Use `omni.utils` helpers instead, e.g.:
# python -c "from omni import utils; print(utils.calculate_max_mel_length_from_asr_csv('data/audio/production_asr.csv'))"
```

### OCR Training (`train_ocr.py`)

```json
{
  "use_compile": true,
  "max_text_length_percentile": 95.0 // Optional: Percentile for auto-calculation (default: 95.0)
  // max_text_length is auto-calculated from dataset - no need to set manually
}
```

**Auto-Calculation:**

- `max_text_length` is **automatically calculated** from your dataset during training
- Uses **95th percentile** by default to minimize padding while covering 95% of data
- ~5% of samples will be skipped if longer (outliers filtered during dataset iteration)

**How it works:**

- Training script analyzes all text samples in your dataset
- Calculates text lengths for each sample
- Uses 95th percentile to determine optimal `max_text_length`
- Ensures minimal padding while covering most of your data

**Optional Override:**

- You can manually set `max_text_length` in config to override auto-calculation
- You can adjust `max_text_length_percentile` (e.g., 99.0 for more coverage, 90.0 for less padding)

**Why Fixed Length?**

- CUDA graphs require fixed tensor shapes for optimal performance
- Variable-length batches cause "tensor size mismatch" errors
- Fixed padding ensures all batches have identical shapes
- Enables 10-20% speedup with CUDA graphs compilation

**What Happens:**

- Sequences shorter than max: Padded with zeros
- Sequences longer than max: Skipped during dataset iteration (filtered before reaching collate function)
- All batches: Uniform shape = CUDA graphs compatible

---

## 💡 Tuning Tips

**For faster training:**

- Increase `batch_size` (if GPU allows)
- Reduce `num_epochs`
- Increase `learning_rate` slightly

**For better quality:**

- Increase `n_layers`, `d_model`
- More training data
- Lower `learning_rate`, more `num_epochs`

**Memory issues:**

- Decrease `batch_size`
- Reduce `ctx_len` or adjust percentile thresholds for auto-calculated max lengths
- Use gradient accumulation

**CUDA graphs compatibility:**

- `max_mel_length` and `max_text_length` are **auto-calculated** from your dataset
- Uses 95th percentile by default (configurable via `*_percentile` options)
- Automatically rounds up to nearest 256 for better memory alignment
- Adjust percentile if needed: higher (99.0) = more coverage/more padding, lower (90.0) = less padding/more skipping

---

## 📋 Example Configuration Files

### `configs/vocoder_tiny.json` (Optional - HiFi-GAN Neural Vocoder)

```json
{
  "save_dir": "checkpoints/vocoder_tiny",
  "train_csv": "data/audio/production_tts.csv",
  "sample_rate": 16000,
  "n_mels": 128,
  "n_fft": 1024,
  "hop_length": 256,
  "batch_size": 2,
  "num_workers": 1,
  "max_audio_length": 8192,
  "gradient_accumulation_steps": 4,
  "lr_g": 0.0002,
  "lr_d": 0.0002,
  "max_steps": 100000,
  "use_amp": true,
  "lambda_mel": 45.0,
  "lambda_fm": 2.0,
  "lambda_adv": 1.0
}
```

**Key Parameters:**

- `max_audio_length`: Limits audio to 8192 samples (~0.5s) for 12GB VRAM
- `gradient_accumulation_steps`: 4 (effective batch size = 2 × 4 = 8)
- `lr_g`, `lr_d`: Separate learning rates for generator and discriminators
- `lambda_mel`, `lambda_fm`, `lambda_adv`: Loss weights for training

**Implementation Notes:**

- ✅ Generator correctly outputs `(B, T_audio)` shape for batch processing
- ✅ Audio loading works with or without torchcodec (automatic fallback)
- ✅ All tensor shapes verified and working correctly

**Memory Optimization (12GB VRAM):**

- `batch_size`: 2 (reduce to 1 if OOM)
- `max_audio_length`: 8192 (~0.5s, reduce to 4096 if OOM)
- `gradient_accumulation_steps`: 4 (simulates batch_size=8)
- `use_amp`: true (FP16 saves ~50% memory)

---

### `configs/ocr_tiny.json` (Optional - OCR Model)

```json
{
  "save_dir": "checkpoints/ocr_tiny",
  "train_csv": "data/ocr/production_ocr.csv",
  "image_root": "data/ocr",
  "img_size": 224,
  "patch": 16,
  "vision_d_model": 512,
  "vision_layers": 4,
  "vision_heads": 8,
  "vision_d_ff": 2048,
  "decoder_d_model": 1024,
  "decoder_layers": 4,
  "decoder_heads": 16,
  "decoder_d_ff": 4096,
  "dropout": 0.1,
  "use_gqa": false,
  "use_swiglu": true,
  "use_flash": true,
  "rope_theta": 10000.0,
  "batch_size": 4,
  "num_workers": 2,
  "drop_last": true,
  "lr": 3e-4,
  "wd": 0.01,
  "warmup_steps": 500,
  "max_steps": 10000,
  "max_epochs": 9999,
  "gradient_accumulation_steps": 2,
  "max_grad_norm": 1.0,
  "use_amp": true,
  "val_split": 0.1,
  "print_freq": 50,
  "checkpoint_freq": 1000,
  "val_freq": 500,
  "seed": 42,
  "shuffle_buffer_size": 10000,
  "use_compile": false,
  "max_text_length_percentile": 95.0 // Auto-calculated, no need to set max_text_length
}
```

**Key Parameters:**

- `train_csv`: CSV file with `image,text` columns
- `image_root`: Root directory for images
- `max_text_length`: Auto-calculated from dataset (95th percentile), can be overridden
- `max_text_length_percentile`: Percentile for auto-calculation (default: 95.0)
- `vocab_size`: Dynamic (built from dataset characters)

**Architecture Notes:**

- ✅ ViT encoder extracts visual features from image patches
- ✅ Transformer decoder with cross-attention to image features
- ✅ RoPE for relative position encoding in text sequences
- ✅ Separate norm instances per layer (matches Thinker pattern)
- ✅ KV caching support for fast autoregressive generation

---

### `configs/vision_tiny.json` (Stage C - Vision Encoder)

```json
{
  "save_dir": "checkpoints/vision_tiny",
  "train_manifest": "data/images/annotations.json",
  "image_root": "data/images",
  "img_size": 224,
  "patch": 16,
  "d_model": 128,
  "n_layers": 4,
  "n_heads": 2,
  "d_ff": 512,
  "dropout": 0.1,
  "embed_dim": 128,
  "use_thinker_for_text": true,
  "thinker_ckpt": "checkpoints/thinker_tiny",
  "ctx_len": 512,
  "vocab_size": 32000,
  "thinker": {
    "vocab_size": 32000,
    "n_layers": 4,
    "d_model": 256,
    "n_heads": 4,
    "d_ff": 1024,
    "dropout": 0.1,
    "rope_theta": 10000,
    "use_gqa": false,
    "use_swiglu": true,
    "use_moe": false
  },
  "temperature": 0.07,
  "batch_size": 8,
  "lr": 3e-4,
  "max_steps": 199716,
  "max_epochs": 3
}
```

**Key Parameters:**

- `train_manifest`: Path to JSON file with image-caption pairs (format: `[{"image": "path", "caption": "text"}, ...]`)
- `image_root`: Root directory for image files
- `thinker_ckpt`: Directory containing trained tokenizer from Stage A (`tokenizer.model`) and optionally trained Thinker (`thinker.pt`)
- `use_thinker_for_text`: Whether to use Thinker model (true) or simple tokenizer+embedding (false) for text encoding
  - **`true` (recommended)**: Uses frozen Thinker model for contextual text embeddings - better quality, more aligned with Stage E
  - **`false`**: Uses simple tokenizer + embedding layer - lighter, faster, but less contextual
- `ctx_len`: Context length for text encoding (matches Thinker's context length)
- `vocab_size`: Vocabulary size (automatically detected from tokenizer if available)
- `embed_dim`: Embedding dimension for contrastive learning (default: same as `d_model`)
- `temperature`: Temperature parameter for contrastive loss (InfoNCE)
- `thinker`: Thinker model configuration (only used if `use_thinker_for_text: true`)

**Training Method:**

- **Contrastive Learning (CLIP-style)**: Aligns image embeddings with text caption embeddings
- Uses trained tokenizer from Stage A for consistent text encoding
- If tokenizer not found, trains new one from image captions
- **Loss**: Contrastive loss (InfoNCE) - encourages matching image-caption pairs to be similar

**Text Encoding Options:**

1. **With Thinker (`use_thinker_for_text: true`)** - Recommended:

   - Captions are tokenized using BPE tokenizer (from `thinker_ckpt/tokenizer.model`)
   - Token IDs are passed through frozen Thinker model to get contextual embeddings
   - Average pooling over sequence to get text representation
   - Projected to `embed_dim` for contrastive learning
   - **Benefits**: Better contextual understanding, aligned with Stage E processing
   - **Requires**: Trained Thinker from Stage A (optional, will use untrained if not found)

2. **Without Thinker (`use_thinker_for_text: false`)** - Lighter option:
   - Captions are tokenized using BPE tokenizer (from `thinker_ckpt/tokenizer.model`)
   - Token IDs are embedded using learnable embedding layer
   - Average pooling over sequence to get text representation
   - Projected to `embed_dim` for contrastive learning
   - **Benefits**: Lighter, faster, no dependency on Thinker checkpoint
   - **Trade-off**: Less contextual understanding than Thinker-based encoding

---

### `configs/ocr_tiny.json` (Optional - OCR Model)

```json
{
  "save_dir": "checkpoints/ocr_tiny",
  "train_csv": "data/ocr/production_ocr.csv",
  "image_root": "data/ocr",
  "img_size": 224,
  "patch": 16,
  "vision_d_model": 512,
  "vision_layers": 4,
  "vision_heads": 8,
  "vision_d_ff": 2048,
  "decoder_d_model": 1024,
  "decoder_layers": 4,
  "decoder_heads": 16,
  "decoder_d_ff": 4096,
  "dropout": 0.1,
  "batch_size": 4,
  "gradient_accumulation_steps": 2,
  "lr": 3e-4,
  "max_steps": 10000
}
```

**Key Parameters:**

- `vision_d_model`, `vision_layers`, `vision_heads`: Vision encoder (ViT) architecture
- `decoder_d_model`, `decoder_layers`, `decoder_heads`: Text decoder architecture
- `train_csv`: Path to OCR CSV file (format: `image,text`)
- `image_root`: Root directory for image files
- `max_text_length`: Auto-calculated from dataset (95th percentile), can be overridden
- `max_text_length_percentile`: Percentile for auto-calculation (default: 95.0)

**Architecture:**

- Vision Encoder: ViT-Tiny (processes image patches)
- Text Decoder: Autoregressive decoder (generates text from visual features)
- Training: Teacher forcing with cross-entropy loss

**CUDA Graphs Compatibility:**

- `max_text_length`: Auto-calculated when `use_compile: true` to ensure uniform batch sizes
- All text sequences are padded to this fixed length (samples exceeding threshold are skipped during dataset iteration)
- Prevents "tensor size mismatch" errors with CUDA graphs compilation

---

[Continue to Chapter 35: Data Preparation →](35-data-preparation.md)

---
