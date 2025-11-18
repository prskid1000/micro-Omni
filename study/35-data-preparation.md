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
# ⚠️ Always use --combine to create production_corpus.txt (required for training)
# Download with sample limit (default: 1,000,000 per dataset, ~6M total combined)
python scripts/download_production_text.py --dataset all --combine

# Or specify custom sample limit per dataset
python scripts/download_production_text.py --dataset all --combine --max-samples 500000  # ~3M total

# Or download specific categories
python scripts/download_production_text.py --dataset scientific --combine
```

**Features:**
- ✅ **`--combine` creates `production_corpus.txt`** - the file that `train_text.py` uses
- ✅ Fine-grained resumption (checkpoints during processing)
- ✅ Diverse knowledge: General, Scientific
- ✅ Sample-based limits (default: 1,000,000 samples per dataset, combined totals: Text ~6M, Audio ~4M, Images ~4M)
- ✅ Ready to use - no formatting needed
- ✅ **ArXiv uses parallel downloads** (5 concurrent API requests) for faster downloads

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
# ⚠️ Always use --combine to create production_asr.csv and production_tts.csv (required for training)
# Download with sample limit (default: 1,000,000 per dataset, ~6M total combined)
python scripts/download_production_audio.py --dataset all --combine

# Or specify custom sample limit per dataset
python scripts/download_production_audio.py --dataset all --combine --max-samples 500000  # ~3M total

# Or download specific categories
python scripts/download_production_audio.py --dataset general --combine
python scripts/download_production_audio.py --dataset scientific --combine
```

**Features:**
- ✅ **`--combine` creates `production_asr.csv` and `production_tts.csv`** - files that `train_audio_enc.py` and `train_talker.py` use
- ✅ Fine-grained resumption (checkpoints by split/speaker)
- ✅ Diverse audio: General speech, Scientific talks, Environmental sounds
- ✅ Sample-based limits (default: 1,000,000 samples per dataset, combined totals: Text ~6M, Audio ~4M, Images ~4M)
- ✅ Ready to use - no formatting needed

---

### Stage C: Image Data (`train_vision.py`)

**Format:** JSON array of objects

**Location:** `data/images/production_annotations.json` (or individual dataset files)

**Example JSON:**
```json
[
  {
    "image": "coco/train2017/000000000139.jpg",
    "caption": "A person riding a motorcycle on a dirt road.",
    "category": "coco"
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
# ⚠️ Always use --combine to create production_annotations.json (required for training)
# Download with sample limit (default: 1,000,000 per dataset, ~7M total combined)
python scripts/download_production_image.py --dataset all --combine

# Or specify custom sample limit per dataset
python scripts/download_production_image.py --dataset all --combine --max-samples 500000  # ~3.5M total

# Or download specific categories
python scripts/download_production_image.py --dataset general --combine
python scripts/download_production_image.py --dataset nature --combine
```

**Features:**
- ✅ **`--combine` creates `production_annotations.json`** - the file that `train_vision.py` uses
- ✅ Fine-grained resumption (checkpoints by class)
- ✅ Diverse images: General, Scientific/Medical, Art, Nature, Domain-specific
- ✅ Sample-based limits (default: 1,000,000 samples per dataset, combined totals: Text ~6M, Audio ~4M, Images ~4M)
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
# Download all modalities with sample-based limits (default: 1M per dataset)
# Combined totals: Text ~10M, Audio ~4M, Images ~4M
python scripts/download_production_text.py --dataset all --combine
python scripts/download_production_image.py --dataset all --combine
python scripts/download_production_audio.py --dataset all --combine
```

**Features:**
- ✅ All formats match training script requirements
- ✅ Downloads from multiple categories when using `--dataset all`
- ✅ Ready to use - no formatting needed

---

### Optional: HiFi-GAN Vocoder Training (`train_vocoder.py`)

**Format:** Uses same audio data as TTS (CSV with `wav` or `text,wav` columns)

**Location:** `data/audio/production_tts.csv` or `data/audio/production_asr.csv`

**Config Key:** `train_csv`

**Requirements:**
- Same format as TTS/ASR data
- Audio files: 16kHz WAV/FLAC
- Clean speech, 1-10 seconds recommended
- Audio length automatically limited to 8192 samples (~0.5s) during training for memory efficiency

**Download:**
```bash
# Uses same audio data as Stage D (TTS)
python scripts/download_production_audio.py --dataset all --combine
# Uses: data/audio/production_tts.csv (or production_asr.csv as fallback)
```

**Features:**
- ✅ Uses existing TTS/ASR data (no additional download needed)
- ✅ Automatically handles both CSV formats (`wav,text` or `text,wav`)
- ✅ Audio length limiting for 12GB VRAM optimization
- ✅ Random cropping for data augmentation

**Note:** This is optional - Griffin-Lim vocoder works without training, but HiFi-GAN provides better quality.

---

### Optional: OCR Training (`train_ocr.py`)

**Format:** CSV file with header row

**Location:** `data/ocr/production_ocr.csv` (or individual dataset files)

**Example CSV:**
```csv
image,text
ocr/images/000001.png,"HELLO"
ocr/images/000002.png,"WORLD"
ocr/textocr/img1.jpg,"Sample text"
```

**Config Key:** `train_csv`

**Requirements:**
- Columns: `image`, `text` (in that order)
- Image files: JPG/PNG, will be resized to 224×224 during training
- Text: Ground truth text labels (characters extracted from images)
- Millions of samples for production training

**Download:**
```bash
# ⚠️ Always use --combine to create production_ocr.csv (required for training)
# Download with sample limit (default: 1,000,000 per dataset)
python scripts/download_production_ocr.py --dataset all --combine

# Or specify custom sample limit per dataset
python scripts/download_production_ocr.py --dataset all --combine --max-samples 500000
```

**Features:**
- ✅ **`--combine` creates `production_ocr.csv`** - the file that `train_ocr.py` uses
- ✅ Fine-grained resumption (checkpoints during processing)
- ✅ Diverse text: Synthetic text, Real-world scene text
- ✅ Sample-based limits (default: 1,000,000 samples per dataset)
- ✅ Ready to use - no formatting needed

**Note:** Some OCR datasets (MJSynth, TextOCR) may require manual download. See script output for instructions.

---

## 🛠️ Production Download Scripts

### Quick Start (Recommended)

**⚠️ Always use `--combine` flag** - it creates the final production files that training scripts require:

```bash
# Download all modalities with sample-based limits (default: 1M per dataset)
# Combined totals: Text ~6M samples (~600M tokens), Audio ~4M, Images ~4M, OCR ~1M
# Creates: production_corpus.txt, production_asr.csv, production_tts.csv, production_annotations.json, production_ocr.csv
python scripts/download_production_text.py --dataset all --combine
python scripts/download_production_image.py --dataset all --combine
python scripts/download_production_audio.py --dataset all --combine
python scripts/download_production_ocr.py --dataset all --combine  # Optional
```

**What `--combine` does:**
- **Text**: Combines all `.txt` files → `data/text/production_corpus.txt`
- **Audio**: Combines all `*_asr.csv` files → `data/audio/production_asr.csv` and `production_tts.csv`
- **Images**: Combines all `*_annotations.json` files → `data/images/production_annotations.json`
- **OCR**: Combines all `*_ocr.csv` files → `data/ocr/production_ocr.csv`

These are the files that training scripts (`train_text.py`, `train_audio_enc.py`, etc.) use.

### Advanced Options

```bash
# Specify custom sample limits per dataset (reduces combined totals)
python scripts/download_production_text.py --dataset all --combine --max-samples 500000  # ~3M total
python scripts/download_production_image.py --dataset all --combine --max-samples 500000  # ~3.5M total
python scripts/download_production_audio.py --dataset all --combine --max-samples 500000  # ~3M total

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
| `train_vocoder.py` | `train_csv` | CSV: `wav,text` or `text,wav` | `data/audio/production_tts.csv` or `production_asr.csv` |
| `train_ocr.py` | `train_csv`<br>`image_root` | CSV: `image,text` | `data/ocr/production_ocr.csv` |

**⚠️ Important: The `--combine` flag is REQUIRED** to create these production files:
- Without `--combine`: You get individual dataset files (e.g., `wikipedia.txt`, `books.txt`, `librispeech_asr.csv`)
- With `--combine`: You get the final production files (e.g., `production_corpus.txt`, `production_asr.csv`) that training scripts use

**✅ No additional formatting needed!** Data is ready to use directly after combining.

---

## 📝 Detailed Format Specifications

### Text Format (`train_text.py`)

**Format:** Plain text file, UTF-8 encoding
- One line per training sample
- Empty lines are skipped
- No special formatting needed

**Example:**
```
The cat sat on the mat.
Hello world, how are you?
Machine learning is fascinating.
```

**Config Key:** `train_text`

**Example Config:**
```json
{
  "train_text": "data/text/production_corpus.txt"
}
```

**Download Script Output:**
- ✅ `data/text/*.txt` files (one per dataset)
- ✅ `data/text/production_corpus.txt` (combined, if `--combine` used)
- ✅ Format: One line per sample, UTF-8

---

### Audio ASR Format (`train_audio_enc.py`)

**Format:** CSV file with header row
- Columns: `wav`, `text` (in that order)
- `wav`: Path to audio file (relative to script working directory or absolute)
- `text`: Transcription text
- Audio files: 16kHz WAV/FLAC (torchaudio handles both)

**Example CSV:**
```csv
wav,text
data/audio/librispeech/train-clean-100/19/198/19-198-0000.flac,"CHAPTER I"
data/audio/librispeech/train-clean-100/19/198/19-198-0001.flac,"THE BOY WHO LIVED"
```

**Config Key:** `train_csv`

**Example Config:**
```json
{
  "train_csv": "data/audio/production_asr.csv",
  "sample_rate": 16000,
  "mel_bins": 128
}
```

**Download Script Output:**
- ✅ `data/audio/*_asr.csv` files (one per dataset)
- ✅ `data/audio/production_asr.csv` (combined, if `--combine` used)
- ✅ Format: CSV with `wav`, `text` columns
- ⚠️ **Note:** Audio files may be FLAC (LibriSpeech) - training script uses `torchaudio.load()` which handles FLAC

---

### Vision Format (`train_vision.py`)

**Format:** JSON array of objects
- Each object must have:
  - `image`: Path to image file (relative to `image_root` or absolute)
  - `caption`: Text caption/description
- Optional: `category` field (not used by training script)

**Example JSON:**
```json
[
  {
    "image": "coco/train2017/000000000139.jpg",
    "caption": "A person riding a motorcycle on a dirt road.",
    "category": "coco"
  },
  {
    "image": "food101/food-101/images/apple_pie/12345.jpg",
    "caption": "A photo of apple pie",
    "category": "apple_pie"
  }
]
```

**Config Keys:** `train_manifest`, `image_root`

**Example Config:**
```json
{
  "train_manifest": "data/images/production_annotations.json",
  "image_root": "data/images",
  "img_size": 224
}
```

**Download Script Output:**
- ✅ `data/images/*_annotations.json` files (one per dataset)
- ✅ `data/images/production_annotations.json` (combined, if `--combine` used)
- ✅ Format: JSON array with `image`, `caption` fields
- ✅ Images resized to 224x224 during download

---

### TTS Format (`train_talker.py`)

**Format:** CSV file with header row
- Columns: `text`, `wav` (note: order is `text` first, then `wav` - **different from ASR!**)
- `text`: Text transcription
- `wav`: Path to audio file
- Audio files: 16kHz WAV/FLAC

**Example CSV:**
```csv
text,wav
"hello world",data/audio/librispeech/train-clean-100/19/198/19-198-0000.flac
"how are you",data/audio/librispeech/train-clean-100/19/198/19-198-0001.flac
```

**Config Key:** `tts_csv`

**Example Config:**
```json
{
  "tts_csv": "data/audio/production_tts.csv",
  "sample_rate": 16000,
  "n_mels": 128
}
```

**Download Script Output:**
- ✅ `data/audio/production_tts.csv` (automatically created when using `--combine` flag)
- ✅ Format: CSV with `text`, `wav` columns (different order from ASR)
- ✅ Same audio files as ASR, just different CSV column order

---

### Multimodal SFT Format (`sft_omni.py`)

**Format:** Mixed - requires all three modalities
- Text: Same as `train_text.py` (plain text, one line per sample)
- Images: Same as `train_vision.py` (JSON array with `image`, `caption`)
- Audio: Same as `train_audio_enc.py` (CSV with `wav`, `text`)

**Config Keys:** `text_path`, `image_manifest`, `image_root`, `asr_csv`

**Example Config:**
```json
{
  "text_path": "data/text/production_corpus.txt",
  "image_manifest": "data/images/production_annotations.json",
  "image_root": "data/images",
  "asr_csv": "data/audio/production_asr.csv"
}
```

**Download Script Output:**
- ✅ All formats match requirements
- ✅ Can use combined files from `--combine` flag

---

## ✅ Verification Checklist

### Text Data
- [x] Plain text format (`.txt`)
- [x] One line per sample
- [x] UTF-8 encoding
- [x] Outputs to `data/text/`
- [x] Combined file available with `--combine`

### Audio ASR Data
- [x] CSV format with header
- [x] Columns: `wav`, `text`
- [x] Audio files accessible (relative or absolute paths)
- [x] Outputs to `data/audio/`
- [x] Combined file available with `--combine`
- [x] Fine-grained resumption support

### Image Data
- [x] JSON array format
- [x] Fields: `image`, `caption`
- [x] Images accessible (relative to `image_root`)
- [x] Outputs to `data/images/`
- [x] Combined file available with `--combine`
- [x] Fine-grained resumption support

### Audio TTS Data
- [x] ✅ TTS format automatically created: `data/audio/production_tts.csv` (columns: `text`, `wav`)
- [x] Created automatically when using `--combine` flag
- [x] Same audio files as ASR, different column order

### OCR Data
- [x] CSV format with header
- [x] Columns: `image`, `text`
- [x] Image files accessible (relative to `image_root` or absolute paths)
- [x] Outputs to `data/ocr/`
- [x] Combined file available with `--combine`
- [x] Fine-grained resumption support

---

## 📊 File Structure After Download

```
data/
├── text/
│   ├── wikipedia.txt
│   ├── books.txt
│   ├── arxiv_physics.txt
│   ├── arxiv_chemistry.txt
│   ├── arxiv_math.txt
│   ├── arxiv_biology.txt
│   └── production_corpus.txt  (if --combine used)
│
├── audio/
│   ├── librispeech_asr.csv
│   ├── commonvoice_asr.csv
│   ├── production_asr.csv  (if --combine used)
│   ├── production_tts.csv  (if --combine used)
│   └── [audio files in subdirectories]
│
├── images/
│   ├── coco_annotations.json
│   ├── food101_annotations.json
│   ├── cifar10_annotations.json
│   ├── cifar100_annotations.json
│   ├── production_annotations.json  (if --combine used)
│   └── [image files in subdirectories]
│
└── ocr/
    ├── mjsynth_ocr.csv
    ├── textocr_ocr.csv
    ├── production_ocr.csv  (if --combine used)
    └── [OCR image files in subdirectories]
```

---

## 🎯 Quick Reference

**To prepare data for training:**

1. **Text:**
   ```bash
   python scripts/download_production_text.py --dataset all --combine
   # Use: data/text/production_corpus.txt
   ```

2. **Audio ASR:**
   ```bash
   python scripts/download_production_audio.py --dataset all --combine
   # Use: data/audio/production_asr.csv
   ```

3. **Images:**
   ```bash
   python scripts/download_production_image.py --dataset all --combine
   # Use: data/images/production_annotations.json with image_root=data/images
   ```

4. **Audio TTS:**
   ```bash
   python scripts/download_production_audio.py --dataset all --combine
   # Use: data/audio/production_tts.csv (automatically created)
   ```

5. **OCR:**
   ```bash
   python scripts/download_production_ocr.py --dataset all --combine
   # Use: data/ocr/production_ocr.csv
   ```

---

## 🎯 Download Features

### Multiple Categories
When using `--dataset all`, scripts download from multiple categories:
- **Text**: General (Wikipedia, Books), Scientific (ArXiv: physics/chemistry/math/biology)
- **Audio**: General speech, Scientific talks, Environmental sounds
- **Images**: General, Scientific/Medical, Art, Nature, Domain-specific

### Parallel Downloads
- **ArXiv**: Uses parallel API requests (5 concurrent requests) for faster downloads while respecting rate limits
- **Other datasets**: Sequential downloads with appropriate rate limiting

You can also download specific categories using `--dataset <category>`.

### Sample-Based Limits
- Default: 1,000,000 samples per dataset (configurable with `--max-samples`)
- Combined totals when using `--dataset all`:
  - **Text**: ~6M samples (~600M tokens) from 6 datasets
  - **Audio**: ~4M samples from 4 datasets
  - **Images**: ~4M samples from 4 datasets
- For 25.65M parameter model: Combined total provides sufficient data for single-epoch training
- Based on Chinchilla scaling laws: 20-200 tokens per parameter (minimum: 513M tokens)
- Automatically stops when reaching sample limit per dataset
- Example: `--max-samples 500000` for smaller combined totals (~3M text samples)

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

✅ **Start with sample-based download:** `--dataset all --combine` to download from multiple categories  
✅ **Production-grade:** Millions of samples (default: 1M per dataset, ~6M combined for text)  
✅ **Fine-grained resumption:** Safe to interrupt and resume  
✅ **No formatting needed:** Outputs are in final format ready for training  
✅ **Diverse knowledge:** Downloads from multiple categories  
✅ **Custom limits:** Use `--max-samples` to control dataset size  
✅ **Synthetic data:** Use `make_synthetic_datasets.py` for quick testing  
✅ **Update configs:** Run `scripts/update_configs_from_data.py` after downloading to optimize training parameters

---

## 📝 Important Notes

- ✅ All scripts support fine-grained resumption (checkpoints saved during processing)
- ✅ All outputs are in final format - no additional formatting needed
- ✅ **`--combine` flag is REQUIRED** to create production files (`production_corpus.txt`, `production_asr.csv`, etc.) that training scripts use
- ✅ Combined files are automatically created when using `--combine` flag
- ✅ Sample-based limits (`--max-samples`) control dataset size (default: 1,000,000 samples per dataset)
- ✅ Paths in CSV/JSON are relative to working directory or can be absolute
- ✅ TTS format (`text,wav`) is automatically created from ASR data when using `--combine`
- ✅ Audio files may be FLAC format (LibriSpeech) - training scripts handle both WAV and FLAC

---

## 📚 Additional Resources

- **Training script verification:** See `docs/TRAINING_SCRIPTS_VERIFICATION.md`
- **Configuration guide:** See [Chapter 34: Configuration Files](34-configuration-files.md)

---

[Continue to Chapter 36: Optimization Techniques →](36-optimization-techniques.md)

---
