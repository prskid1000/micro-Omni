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
└── omni_sft_tiny.json     # Stage E: Multimodal SFT
```

---

## 📝 Common Parameters

### Model Architecture

```json
{
  "d_model": 256,        // Embedding dimension
  "n_layers": 4,         // Transformer layers
  "n_heads": 4,          // Attention heads
  "d_ff": 1024,          // FFN hidden size (usually 4×d_model)
  "dropout": 0.1,        // Dropout rate (0-1)
  "ctx_len": 512         // Context length (tokens)
}
```

### Training Hyperparameters

```json
{
  "batch_size": 16,          // Examples per batch
  "num_epochs": 10,          // Training epochs
  "learning_rate": 3e-4,     // LR (0.0003)
  "warmup_steps": 1000,      // LR warmup
  "max_grad_norm": 1.0,      // Gradient clipping
  "weight_decay": 0.01       // L2 regularization
}
```

### Data & Checkpointing

```json
{
  "data_path": "data/text/corpus.txt",
  "save_every": 1000,        // Save checkpoint frequency
  "eval_every": 500,         // Evaluation frequency
  "checkpoint_dir": "checkpoints/thinker_tiny/"
}
```

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
- **Fast mode:** Use `--skip-text-tokenization --assume-text-tokens N` to skip tokenization and use an assumed token count (e.g., 8000000000 for 8B tokens). Sample counts are still read from offset cache files when available.

**Memory-efficient processing:**
- Tokenizer training: Plain text passed directly to SentencePiece (no streaming). CSV/JSON streams text extraction to temp file.
- Temp files: Only used for CSV/JSON text extraction (streams extraction), stored in `data/.temp/` and auto-cleaned
- Token counting processes files line-by-line with automatic resume support
- All operations are resumable - if interrupted, will continue from last checkpoint
- Checkpoints saved every 10K samples/lines for safe resumption

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
- Reduce `ctx_len`
- Use gradient accumulation

**Automatic tuning:**
- Use `scripts/update_configs_from_data.py` to automatically set epochs/steps based on your dataset size and model architecture
- Automatically adjusts batch size and gradient accumulation based on model size
- Uses research-based formulas for optimal training configuration
- Ensures optimal training duration and memory usage without manual calculation

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

**Architecture:**
- Vision Encoder: ViT-Tiny (processes image patches)
- Text Decoder: Autoregressive decoder (generates text from visual features)
- Training: Teacher forcing with cross-entropy loss

---

[Continue to Chapter 35: Data Preparation →](35-data-preparation.md)

---
