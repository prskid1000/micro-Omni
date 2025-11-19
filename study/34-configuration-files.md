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
# Update all configs based on actual dataset sizes
python scripts/update_configs_from_data.py

# Preview changes without modifying files
python scripts/update_configs_from_data.py --dry-run
```

**What gets updated:**
- `max_steps`: Calculated from token count, batch size, and context length using research-based formulas
- `max_epochs`: Based on token count (1-3 for very large, 5-10 for small)
- `warmup_steps`: 4% of total steps (research-based, typically 3-5%, capped at 10K)
- `batch_size`: Automatically adjusted based on model size (larger models = smaller batch sizes)
- `gradient_accumulation_steps`: Automatically adjusted to maintain effective batch size
- `val_freq`: Every 500-1000 steps or 10% of steps per epoch
- `checkpoint_freq`: Every 5000-10000 steps or 1 per epoch
- Data paths: Automatically updated to production files if they exist

**Token count recommendations:**
- **Very large (>100M tokens):** 1-3 epochs
- **Large (50M-100M tokens):** 2-4 epochs
- **Medium (10M-50M tokens):** 3-6 epochs
- **Small (<10M tokens):** 5-10 epochs

**Note:** The script counts actual tokens using the BPE tokenizer:
- **Text:** Tokens from text corpus
- **Images:** Tokens from captions
- **Audio:** Tokens from transcriptions (ASR/TTS)
- **OCR:** Tokens from extracted text
- If no tokenizer exists, one will be created automatically from the data

**Files checked:**
- Text: `data/text/production_corpus.txt` or `data/text/tiny_corpus.txt`
- Images: `data/images/production_annotations.json` or `data/images/annotations.json`
- Audio: `data/audio/production_asr.csv` or `data/audio/asr.csv`
- TTS: `data/audio/production_tts.csv` or `data/audio/tts.csv`
- OCR: `data/ocr/production_ocr.csv` or `data/ocr/ocr_train.csv`
- Vocoder: Uses same audio data as TTS/ASR (no separate check)

**Token counting:**
- All training parameters are calculated based on **token counts**, not sample counts
- The script uses the BPE tokenizer to count actual tokens in:
  - Text files (line-by-line)
  - Image captions (from JSON manifest)
  - Audio transcriptions (from CSV files)
  - OCR text (from CSV files)
- If no tokenizer exists, one is automatically created from the text data
- Token counts are more accurate than sample counts for determining training duration

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
- **Tokens per step:** `tokens_per_step = EBS × context_length`
- **Steps per epoch:** `steps_per_epoch = training_tokens / tokens_per_step`
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
