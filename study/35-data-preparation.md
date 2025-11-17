# Chapter 35: Data Preparation Guide

[← Previous: Configuration Files](34-configuration-files.md) | [Back to Index](00-INDEX.md) | [Next: Optimization Techniques →](36-optimization-techniques.md)

---

## 🎯 Preparing Training Data

Each training stage requires specific data formats. This chapter explains data preparation for all stages using the production-grade download scripts.

---

## 📊 Data Requirements by Stage

### Stage A: Text Data (`train_text.py`)

**Format:** Plain text file, UTF-8 encoding, one line per sample

**Location:** `data/text/production_corpus.txt` (or individual dataset files)

**Example content:**
```
The cat sat on the mat.
Hello world, how are you?
Machine learning is fascinating.
This is a conversation example.
Math problem: Solve for x in 2x + 5 = 15.
```

**Config Key:** `train_text`

**Download:**
```bash
# Intelligent download (25-30GB, diverse knowledge)
python scripts/download_production_text.py --dataset all --combine

# Or download specific categories
python scripts/download_production_text.py --dataset scientific --combine
python scripts/download_production_text.py --dataset conversations --combine
```

**Features:**
- ✅ Fine-grained resumption (checkpoints during processing)
- ✅ Diverse knowledge: General, Conversations, Scientific, Tools
- ✅ Automatic size management (25-30GB target)
- ✅ Ready to use - no formatting needed

---

### Stage B: Audio Data - ASR (`train_audio_enc.py`)

**Format:** CSV file with header row

**Location:** `data/audio/production_asr.csv` (or individual dataset files)

**Example CSV:**
```csv
wav,text
data/audio/librispeech/train-clean-100/19/198/19-198-0000.flac,"CHAPTER I"
data/audio/librispeech/train-clean-100/19/198/19-198-0001.flac,"THE BOY WHO LIVED"
data/audio/commonvoice/clip1.wav,"hello world"
```

**Config Key:** `train_csv`

**Requirements:**
- Columns: `wav`, `text` (in that order)
- Audio files: 16kHz WAV/FLAC (torchaudio handles both)
- 1-10 second clips recommended
- Millions of samples for production training

**Download:**
```bash
# Intelligent download (25-30GB, diverse audio)
python scripts/download_production_audio.py --dataset all --combine

# Or download specific categories
python scripts/download_production_audio.py --dataset general --combine
python scripts/download_production_audio.py --dataset scientific --combine
```

**Features:**
- ✅ Fine-grained resumption (checkpoints by split/speaker)
- ✅ Diverse audio: General speech, Scientific talks, Environmental sounds
- ✅ Automatic size management (25-30GB target)
- ✅ Ready to use - no formatting needed

---

### Stage C: Image Data (`train_vision.py`)

**Format:** JSON array of objects

**Location:** `data/images/production_annotations.json` (or individual dataset files)

**Example JSON:**
```json
[
  {
    "image": "imagenet_subset/train/n01440764/n01440764_18.JPEG",
    "caption": "An image of n01440764",
    "category": "n01440764"
  },
  {
    "image": "food101/food-101/images/apple_pie/12345.jpg",
    "caption": "A photo of apple pie",
    "category": "apple_pie"
  }
]
```

**Config Keys:** `train_manifest`, `image_root`

**Requirements:**
- JSON array format (not nested `images` object)
- Each object must have: `image` (path), `caption` (text)
- Optional: `category` field (not used by training script)
- Images: JPG/PNG, will be resized to 224×224 during training
- Millions of images for production training

**Download:**
```bash
# Intelligent download (25-30GB, diverse images)
python scripts/download_production_image.py --dataset all --combine

# Or download specific categories
python scripts/download_production_image.py --dataset general --combine
python scripts/download_production_image.py --dataset nature --combine
```

**Features:**
- ✅ Fine-grained resumption (checkpoints by class)
- ✅ Diverse images: General, Scientific/Medical, Art, Nature, Domain-specific
- ✅ Automatic size management (25-30GB target)
- ✅ Ready to use - no formatting needed

---

### Stage D: Speech Data - TTS (`train_talker.py`)

**Format:** CSV file with header row (note: different column order from ASR!)

**Location:** `data/audio/production_tts.csv`

**Example CSV:**
```csv
text,wav
"hello world",data/audio/librispeech/train-clean-100/19/198/19-198-0000.flac
"how are you",data/audio/librispeech/train-clean-100/19/198/19-198-0001.flac
```

**Config Key:** `tts_csv`

**Requirements:**
- Columns: `text`, `wav` (in that order - **different from ASR!**)
- Audio files: 16kHz WAV/FLAC
- Clean speech, 1-10 seconds
- Millions of samples for production training

**Download:**
```bash
# TTS format is automatically created when combining ASR data
python scripts/download_production_audio.py --dataset all --combine
# Creates both: production_asr.csv (wav,text) AND production_tts.csv (text,wav)
```

**Features:**
- ✅ Automatically generated from ASR data
- ✅ Same audio files, just different CSV column order
- ✅ Ready to use - no formatting needed

---

### Stage E: Multimodal Data (`sft_omni.py`)

**Format:** Mixed - requires all three modalities

**Locations:**
- Text: `data/text/production_corpus.txt`
- Images: `data/images/production_annotations.json`
- Audio: `data/audio/production_asr.csv`

**Config Keys:** `text_path`, `image_manifest`, `image_root`, `asr_csv`

**Download:**
```bash
# Download all modalities with intelligent diversity balancing
python scripts/download_production_text.py --dataset all --combine
python scripts/download_production_image.py --dataset all --combine
python scripts/download_production_audio.py --dataset all --combine
```

**Features:**
- ✅ All formats match training script requirements
- ✅ Balanced diversity across modalities
- ✅ Ready to use - no formatting needed

---

## 🛠️ Production Download Scripts

### Quick Start (Recommended)

```bash
# Download all modalities with intelligent 25-30GB diversity balancing
python scripts/download_production_text.py --dataset all --combine --min-gb 25 --max-gb 30
python scripts/download_production_image.py --dataset all --combine --min-gb 25 --max-gb 30
python scripts/download_production_audio.py --dataset all --combine --min-gb 25 --max-gb 30
```

### Advanced Options

```bash
# Download specific categories only
python scripts/download_production_text.py --dataset scientific --combine
python scripts/download_production_image.py --dataset nature --combine
python scripts/download_production_audio.py --dataset general --combine

# Resume interrupted downloads (automatic fine-grained checkpoints)
python scripts/download_production_text.py --dataset all --combine
# If interrupted, just run again - will resume from exact position

# Reset and re-download everything
python scripts/download_production_text.py --dataset all --reset --combine
```

### Synthetic Data (For Testing)

```bash
# Generate small synthetic datasets for testing
python scripts/make_synthetic_datasets.py

# Creates:
# - data/text/tiny_corpus.txt (~2MB)
# - data/images/annotations.json (~20MB)
# - data/audio/asr.csv, tts.csv (~15MB)
```

---

## 📋 Format Verification

All download scripts output data in the **exact format** required by training scripts:

| Training Script | Config Key | Format | Download Script Output |
|----------------|------------|--------|----------------------|
| `train_text.py` | `train_text` | Plain text (`.txt`) | `data/text/production_corpus.txt` |
| `train_audio_enc.py` | `train_csv` | CSV: `wav,text` | `data/audio/production_asr.csv` |
| `train_talker.py` | `tts_csv` | CSV: `text,wav` | `data/audio/production_tts.csv` |
| `train_vision.py` | `train_manifest`<br>`image_root` | JSON array: `[{"image": "...", "caption": "..."}]` | `data/images/production_annotations.json` |

**✅ No additional formatting needed!** Data is ready to use directly.

---

## 🎯 Intelligent Download Features

### Diversity Balancing
When using `--dataset all`, scripts intelligently download from multiple categories:
- **Text**: General, Conversations, Scientific, Tools
- **Audio**: General speech, Scientific talks, Environmental sounds
- **Images**: General, Scientific/Medical, Art, Nature, Domain-specific

### Size Management
- Target: 25-30GB per modality (configurable with `--min-gb` and `--max-gb`)
- Automatically stops when reaching size limits
- Ensures balanced distribution across categories

### Fine-Grained Resumption
- Checkpoints saved during processing (by file, line, class, etc.)
- Resume from exact position if interrupted
- No need to restart from beginning

---

## 📝 Example Config Files

### `configs/thinker_tiny.json`
```json
{
  "train_text": "data/text/production_corpus.txt",
  "vocab_size": 32000,
  "ctx_len": 1024,
  ...
}
```

### `configs/audio_enc_tiny.json`
```json
{
  "train_csv": "data/audio/production_asr.csv",
  "sample_rate": 16000,
  "mel_bins": 128,
  ...
}
```

### `configs/talker_tiny.json`
```json
{
  "tts_csv": "data/audio/production_tts.csv",
  "sample_rate": 16000,
  "n_mels": 128,
  ...
}
```

### `configs/vision_tiny.json`
```json
{
  "train_manifest": "data/images/production_annotations.json",
  "image_root": "data/images",
  "img_size": 224,
  ...
}
```

---

## ⚙️ After Downloading: Update Training Configs

**Important:** After downloading datasets, update training parameters based on actual data size:

```bash
# Automatically update all config files based on dataset sizes
python scripts/update_configs_from_data.py

# Preview changes first (recommended)
python scripts/update_configs_from_data.py --dry-run
```

**What this does:**
- ✅ Counts samples in your production/synthetic datasets
- ✅ Calculates optimal `max_steps`, `max_epochs`, `warmup_steps`
- ✅ Adjusts validation and checkpoint frequencies
- ✅ Updates data paths to production files automatically
- ✅ Applies best practices based on dataset size

**Why it matters:**
- Large datasets (>1M samples) need fewer epochs (1-3)
- Small datasets (<50K samples) need more epochs (10-20)
- Proper warmup steps prevent training instability
- Correct max_steps ensures complete training without waste

**Example output:**
```
[1] Analyzing Text Dataset...
  Text samples: 2,500,000 (from data/text/production_corpus.txt)
  
[Stage A] Text-only Training (thinker_tiny.json)
  Samples: 2,500,000
  Steps/epoch: 312,500
  Recommended epochs: 2
  Max steps: 625,000
  Warmup steps: 31,250
```

See [Chapter 34: Configuration Files](34-configuration-files.md) for more details.

---

## 💡 Tips

✅ **Start with intelligent download:** `--dataset all --combine` for balanced diversity  
✅ **Production-grade:** Millions of samples, 25-30GB per modality  
✅ **Fine-grained resumption:** Safe to interrupt and resume  
✅ **No formatting needed:** Outputs are in final format ready for training  
✅ **Diverse knowledge:** Automatically balances across categories  
✅ **Synthetic data:** Use `make_synthetic_datasets.py` for quick testing  
✅ **Update configs:** Run `scripts/update_configs_from_data.py` after downloading to optimize training parameters

---

## 📚 Additional Resources

- **Detailed format specs:** See `docs/DATA_FORMAT_REQUIREMENTS.md`
- **Training script verification:** See `docs/TRAINING_SCRIPTS_VERIFICATION.md`
- **Configuration guide:** See [Chapter 34: Configuration Files](34-configuration-files.md)

---

[Continue to Chapter 36: Optimization Techniques →](36-optimization-techniques.md)

---
