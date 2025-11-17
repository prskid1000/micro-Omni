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
- `max_steps`: Calculated from dataset size and batch size
- `max_epochs`: Based on dataset size (1-3 for large, 5-10 for small)
- `warmup_steps`: 5-10% of total steps (capped at 10K)
- `val_freq`: Every 500-1000 steps or 10% of steps per epoch
- `checkpoint_freq`: Every 5000-10000 steps or 1 per epoch
- Data paths: Automatically updated to production files if they exist

**Dataset size recommendations:**
- **Very large (>1M samples):** 1-3 epochs
- **Large (500K-1M):** 2-4 epochs
- **Medium (100K-500K):** 3-6 epochs
- **Small (50K-100K):** 5-10 epochs
- **Very small (<50K):** 10-20 epochs

**Files checked:**
- Text: `data/text/production_corpus.txt` or `data/text/tiny_corpus.txt`
- Images: `data/images/production_annotations.json` or `data/images/annotations.json`
- Audio: `data/audio/production_asr.csv` or `data/audio/asr.csv`
- TTS: `data/audio/production_tts.csv` or `data/audio/tts.csv`

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
- Use `scripts/update_configs_from_data.py` to automatically set epochs/steps based on your dataset size
- Ensures optimal training duration without manual calculation

---

[Continue to Chapter 35: Data Preparation →](35-data-preparation.md)

---
