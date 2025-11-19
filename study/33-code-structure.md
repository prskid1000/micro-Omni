# Chapter 33: Codebase Structure Guide

[← Previous: Inference Pipeline](32-inference-pipeline.md) | [Back to Index](00-INDEX.md) | [Next: Configuration Files →](34-configuration-files.md)

---

## 📂 Directory Structure

```
μOmni/
├── omni/                      # Core modules
│   ├── __init__.py
│   ├── thinker.py            # Decoder-only LLM (20.32M params)
│   ├── audio_encoder.py      # AuT-Tiny (2.05M params)
│   ├── vision_encoder.py     # ViT-Tiny (914K params)
│   ├── talker.py             # Speech generator (2.24M params)
│   ├── codec.py              # RVQ + Griffin-Lim vocoder
│   ├── tokenizer.py          # BPE tokenizer wrapper
│   ├── utils.py              # RMSNorm, RoPE, helpers
│   └── training_utils.py     # Training helpers (checkpoint loading, resuming, validation)
│
├── configs/                   # JSON configurations
│   ├── thinker_tiny.json     # Thinker config
│   ├── audio_enc_tiny.json   # Audio encoder config
│   ├── vision_tiny.json      # Vision encoder config
│   ├── talker_tiny.json      # Talker config
│   └── omni_sft_tiny.json    # Multimodal SFT config
│
├── scripts/                   # Utility scripts
│   ├── check_setup.py        # Verify installation
│   ├── download_production_text.py  # Download text data
│   ├── download_production_audio.py # Download audio data
│   ├── download_production_image.py # Download image data
│   ├── download_production_ocr.py   # Download OCR data
│   ├── update_configs_from_data.py # Auto-update configs from data
│   └── make_synthetic_datasets.py   # Generate test data
│
├── train_text.py             # Stage A: Thinker pretraining
├── train_audio_enc.py        # Stage B: Audio encoder
├── train_vision.py           # Stage C: Vision encoder
├── train_talker.py           # Stage D: Talker + RVQ
├── train_vocoder.py          # Optional: HiFi-GAN vocoder
├── train_ocr.py              # Optional: OCR model
├── sft_omni.py              # Stage E: Multimodal SFT
│
├── infer_chat.py            # Inference interface
├── test_all_media.py        # Test multimodal inputs
│
├── data/                    # Training data (create)
│   ├── text/                # Text corpus files
│   ├── images/              # Image manifest files
│   ├── audio/               # Audio CSV files
│   └── ocr/                 # OCR CSV files
│
├── checkpoints/             # Model weights (create)
│   ├── thinker_tiny/
│   ├── audio_enc_tiny/
│   ├── vision_tiny/
│   ├── talker_tiny/
│   └── omni_sft_tiny/
│
├── examples/                # Sample inputs
│   ├── sample_image.png
│   ├── sample_audio.wav
│   └── sample_text.txt
│
├── study/                   # Documentation (this!)
│   ├── 00-INDEX.md
│   ├── 01-what-is-ai.md
│   └── ...
│
├── requirements.txt         # Python dependencies
└── README.md               # Main README
```

---

## 🔍 Key Files Explained

### Core Modules (`omni/`)

#### `thinker.py`
```python
class ThinkerLM(nn.Module):
    """
    Decoder-only transformer (GPT-style)
    - Accepts token IDs or embeddings
    - Causal attention with RoPE
    - KV caching for fast generation
    """
```

#### `audio_encoder.py`
```python
class AudioEncoderTiny(nn.Module):
    """
    Audio understanding encoder
    - Input: Mel spectrogram (T, 128)
    - Process: Conv downsample + Transformer
    - Output: Frame embeddings (T/8, 192)
    """
```

#### `vision_encoder.py`
```python
class ViTTiny(nn.Module):
    """
    Vision Transformer encoder
    - Input: Image (224×224×3)
    - Process: Patch embedding + Transformer
    - Output: CLS token (1, 128)
    """
```

#### `talker.py`
```python
class TalkerTiny(nn.Module):
    """
    Speech code generator
    - Input: Previous RVQ codes
    - Process: Transformer decoder
    - Output: Next frame codes (base + residual)
    """
```

#### `codec.py`
```python
class RVQ(nn.Module):
    """Residual Vector Quantization"""

class GriffinLimVocoder:
    """Classical vocoder (no training)"""
```

---

### Training Scripts

#### `train_text.py`
- **Stage A**: Thinker pretraining
- **Data**: Text corpus
- **Loss**: Cross-entropy (next-token)
- **Output**: `checkpoints/thinker_tiny/`

#### `train_audio_enc.py`
- **Stage B**: Audio encoder ASR
- **Data**: Audio + transcriptions
- **Loss**: CTC
- **Output**: `checkpoints/audio_enc_tiny/`

#### `train_vision.py`
- **Stage C**: Vision encoder
- **Data**: Images + captions
- **Loss**: Cross-entropy
- **Output**: `checkpoints/vision_tiny/`

#### `train_talker.py`
- **Stage D**: Talker + RVQ
- **Data**: Audio files
- **Loss**: Cross-entropy + MSE
- **Output**: `checkpoints/talker_tiny/`

#### `train_vocoder.py`
- **Optional**: HiFi-GAN vocoder training
- **Data**: Audio files (TTS/ASR CSV)
- **Loss**: Adversarial
- **Output**: `checkpoints/vocoder_tiny/`

#### `train_ocr.py`
- **Optional**: OCR model training
- **Data**: Images + text labels
- **Loss**: Cross-entropy
- **Output**: `checkpoints/ocr_tiny/`

#### `sft_omni.py`
- **Stage E**: Multimodal SFT
- **Data**: Mixed modalities (text, images, audio)
- **Loss**: Cross-entropy
- **Output**: `checkpoints/omni_sft_tiny/`

---

### Configuration Files

All configs are JSON:

```json
{
  "model_params": { ... },
  "training_params": { ... },
  "data_params": { ... }
}
```

See [Chapter 34: Configuration Files](34-configuration-files.md) for details.

---

## 💡 Code Navigation Tips

### Find Component Definition
```bash
# Thinker architecture
grep -n "class ThinkerLM" omni/thinker.py

# Attention implementation
grep -n "class Attention" omni/thinker.py

# RVQ codec
grep -n "class RVQ" omni/codec.py
```

### Find Training Loop
```bash
# Thinker training
grep -n "def train" train_text.py

# SFT training
grep -n "def train" sft_omni.py
```

### Find Inference Code
```bash
# Generation function
grep -n "def generate" infer_chat.py

# Multimodal processing
grep -n "multimodal" infer_chat.py
```

---

## 📊 File Dependencies

```
thinker.py
├── utils.py (RMSNorm, RoPE)
└── (no other omni deps)

audio_encoder.py
├── utils.py (RMSNorm)
└── (no other omni deps)

vision_encoder.py
├── utils.py (RMSNorm)
└── (no other omni deps)

talker.py
├── utils.py (RMSNorm, RoPE)
└── (no other omni deps)

codec.py
└── (standalone)

infer_chat.py
├── thinker.py
├── audio_encoder.py
├── vision_encoder.py
├── talker.py
├── codec.py
└── tokenizer.py
```

---

## 💾 Streaming Datasets

All training scripts use streaming `IterableDataset` implementations:

- **Text files**: Stream line-by-line directly
- **CSV files**: Use `csv.DictReader` for row-by-row streaming
- **JSON files**: Load once, then iterate through items

**Benefits:**
- ✅ No cache files needed - simpler and cleaner
- ✅ Minimal memory usage - only current item in memory
- ✅ Efficient resuming via `skip_samples` parameter
- ✅ Worker sharding for multi-process data loading
- ✅ Buffer-based shuffling for randomization

See [Chapter 36: Optimization Techniques](36-optimization-techniques.md) for details.

## 🔄 Common Training Utilities

All training scripts share common utilities from `omni/training_utils.py`:

### Checkpoint Management
- **`load_checkpoint()`**: Automatically finds and loads the latest checkpoint
  - Handles model, optimizer, scheduler, and scaler state dicts
  - Returns step number and checkpoint path
  - Supports legacy checkpoint formats

### Resuming Training
- **`setup_resume_data_loading()`**: Configures dataset `skip_samples` for resuming
  - Handles `SubsetDataset` wrappers from `random_split`
  - Recreates DataLoader with updated skip_samples
  - Works seamlessly with IterableDataset streaming

- **`calculate_resume_position()`**: Calculates epoch and batch position from global step
  - Returns `(start_epoch, start_batch_idx)` tuple
  - Used for progress bar initialization and epoch tracking

### Validation
- **`ValidationSkipSamplesContext`**: Context manager for validation loops
  - Temporarily resets `skip_samples` to 0 for validation
  - Ensures validation always processes full validation set
  - Automatically restores original `skip_samples` after validation

**Benefits:**
- ✅ Consistent resuming logic across all training scripts
- ✅ Automatic checkpoint detection (no `--resume` flag needed)
- ✅ Proper validation on full dataset regardless of training resumption
- ✅ Reduced code duplication and easier maintenance

---

## 💡 Key Takeaways

✅ **Modular structure** - each component independent  
✅ **Clear separation** - training vs inference  
✅ **Config-driven** - easy to modify parameters  
✅ **Self-contained** - minimal dependencies  
✅ **Streaming datasets** - efficient memory usage

---

[Continue to Chapter 34: Configuration Files →](34-configuration-files.md)

