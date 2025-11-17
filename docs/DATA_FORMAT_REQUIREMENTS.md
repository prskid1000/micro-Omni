# Data Format Requirements for Training Scripts

This document specifies the exact data formats required by each training script in μOmni.

## 📋 Summary

| Training Script | Format | Location | Required Fields |
|----------------|--------|----------|----------------|
| `train_text.py` | Plain text (one line per sample) | `data/text/*.txt` | Lines of text |
| `train_audio_enc.py` | CSV (ASR) | `data/audio/*_asr.csv` | `wav`, `text` columns |
| `train_vision.py` | JSON array | `data/images/*_annotations.json` | `image`, `caption` fields |
| `train_talker.py` | CSV (TTS) | `data/audio/*_tts.csv` | `text`, `wav` columns |
| `sft_omni.py` | Mixed (text + images + audio) | Multiple files | See below |

---

## 📝 Detailed Requirements

### 1. Text Training (`train_text.py`)

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

**Our Download Script Output:**
- ✅ `data/text/*.txt` files (one per dataset)
- ✅ `data/text/production_corpus.txt` (combined, if `--combine` used)
- ✅ Format: One line per sample, UTF-8

---

### 2. Audio ASR Training (`train_audio_enc.py`)

**Format:** CSV file with header row
- Columns: `wav`, `text`
- `wav`: Path to audio file (relative to script working directory or absolute)
- `text`: Transcription text
- Audio files: 16kHz WAV format

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

**Our Download Script Output:**
- ✅ `data/audio/*_asr.csv` files (one per dataset)
- ✅ `data/audio/production_asr.csv` (combined, if `--combine` used)
- ✅ Format: CSV with `wav`, `text` columns
- ⚠️ **Note:** Audio files may be FLAC (LibriSpeech) - training script uses `torchaudio.load()` which handles FLAC

---

### 3. Vision Training (`train_vision.py`)

**Format:** JSON array of objects
- Each object must have:
  - `image`: Path to image file (relative to `image_root` or absolute)
  - `caption`: Text caption/description
- Optional: `category` field (not used by training script)

**Example JSON:**
```json
[
  {
    "image": "imagenet_subset/train/n01440764/n01440764_18.JPEG",
    "caption": "An image of n01440764",
    "category": "n01440764"
  },
  {
    "image": "imagenet_subset/train/n01443537/n01443537_42.JPEG",
    "caption": "An image of n01443537",
    "category": "n01443537"
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

**Our Download Script Output:**
- ✅ `data/images/*_annotations.json` files (one per dataset)
- ✅ `data/images/production_annotations.json` (combined, if `--combine` used)
- ✅ Format: JSON array with `image`, `caption` fields
- ✅ Images resized to 224x224 during download

---

### 4. TTS Training (`train_talker.py`)

**Format:** CSV file with header row
- Columns: `text`, `wav` (note: order is `text` first, then `wav`)
- `text`: Text transcription
- `wav`: Path to audio file
- Audio files: 16kHz WAV format

**Example CSV:**
```csv
text,wav
"hello world",data/audio/tts/sample1.wav
"how are you",data/audio/tts/sample2.wav
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

**Our Download Script Output:**
- ⚠️ **Note:** Currently our audio script outputs ASR format (`wav`, `text`), not TTS format (`text`, `wav`)
- ✅ Can be converted by swapping column order
- ✅ Audio files compatible (16kHz WAV/FLAC)

---

### 5. Multimodal SFT (`sft_omni.py`)

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

**Our Download Script Output:**
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
- [x] ✅ **FIXED:** `combine_audio_csvs()` now creates both ASR and TTS formats
- [x] TTS format: `data/audio/production_tts.csv` (columns: `text`, `wav`)
- [x] Created automatically when using `--combine` flag

---

## 📊 File Structure After Download

```
data/
├── text/
│   ├── wikipedia.txt
│   ├── books.txt
│   ├── arxiv_physics.txt
│   └── production_corpus.txt  (if --combine used)
│
├── audio/
│   ├── librispeech_asr.csv
│   ├── commonvoice_asr.csv
│   ├── production_asr.csv  (if --combine used)
│   └── [audio files in subdirectories]
│
└── images/
    ├── imagenet_annotations.json
    ├── food101_annotations.json
    ├── production_annotations.json  (if --combine used)
    └── [image files in subdirectories]
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

---

## 📝 Notes

- All scripts support fine-grained resumption (checkpoints saved during processing)
- All outputs are in final format - no additional formatting needed
- Combined files are automatically created when using `--combine` flag
- Intelligent download (`--dataset all`) ensures 25-30GB with diversity balancing
- Paths in CSV/JSON are relative to working directory or can be absolute

