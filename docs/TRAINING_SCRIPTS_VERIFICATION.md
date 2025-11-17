# Training Scripts Format Verification

This document verifies that our download scripts output the exact formats required by each training script.

## ✅ Verification Results

### 1. `train_text.py` ✅ MATCHES

**Script Requirements:**
- Line 60: `ds = TextDataset(cfg["train_text"], tok, cfg["ctx_len"])`
- Format: Plain text file, one line per sample
- Reads: `cfg["train_text"]`

**Our Download Script Output:**
- ✅ `data/text/*.txt` files
- ✅ `data/text/production_corpus.txt` (when `--combine` used)
- ✅ Format: One line per sample, UTF-8
- ✅ Config key: `"train_text": "data/text/production_corpus.txt"`

**Status:** ✅ **PERFECT MATCH**

---

### 2. `train_audio_enc.py` ✅ MATCHES

**Script Requirements:**
- Line 76: `ds = ASRDataset(cfg["train_csv"], sr=sr, n_mels=n_mels, cfg=cfg)`
- Line 62: `path, text = row["wav"], row["text"]`
- Format: CSV with columns `wav`, `text` (in that order)
- Reads: `cfg["train_csv"]`

**Our Download Script Output:**
- ✅ `data/audio/*_asr.csv` files
- ✅ `data/audio/production_asr.csv` (when `--combine` used)
- ✅ Format: CSV with header `wav,text`
- ✅ Config key: `"train_csv": "data/audio/production_asr.csv"`

**Status:** ✅ **PERFECT MATCH**

---

### 3. `train_talker.py` ✅ MATCHES

**Script Requirements:**
- Line 90: `ds = TTSDataset(cfg["tts_csv"], sr=sr, n_mels=n_mels, frame_ms=frame_ms, cfg=cfg)`
- Line 75: `text, path = row["text"], row["wav"]`
- Format: CSV with columns `text`, `wav` (in that order - **note: different from ASR!**)
- Reads: `cfg["tts_csv"]`

**Our Download Script Output:**
- ✅ `data/audio/production_tts.csv` (automatically created when `--combine` used)
- ✅ Format: CSV with header `text,wav`
- ✅ Config key: `"tts_csv": "data/audio/production_tts.csv"`

**Status:** ✅ **PERFECT MATCH** (Fixed in previous update)

---

### 4. `train_vision.py` ✅ MATCHES

**Script Requirements:**
- Line 125: `ds = ImgCapDataset(cfg["train_manifest"], cfg["image_root"], cfg["img_size"])`
- Line 115: `img = Image.open(os.path.join(self.root, it["image"])).convert("RGB")`
- Line 116: `return self.tf(img), it["caption"]`
- Format: JSON array with objects containing `image` and `caption` fields
- Reads: `cfg["train_manifest"]` and `cfg["image_root"]`

**Our Download Script Output:**
- ✅ `data/images/*_annotations.json` files
- ✅ `data/images/production_annotations.json` (when `--combine` used)
- ✅ Format: JSON array `[{"image": "...", "caption": "..."}, ...]`
- ✅ Config keys: 
  - `"train_manifest": "data/images/production_annotations.json"`
  - `"image_root": "data/images"`

**Status:** ✅ **PERFECT MATCH**

**Note:** Fixed bug in `train_vision.py` line 561 (was trying to save non-existent `head` variable)

---

## 🔧 Bug Fix Applied

### `train_vision.py` Line 561
**Before (BUG):**
```python
checkpoint_data = {
    "vit": vit.state_dict(),
    "head": head.state_dict(),  # ❌ 'head' doesn't exist!
    ...
}
```

**After (FIXED):**
```python
checkpoint_data = {
    "vit": vit.state_dict(),
    "img_proj": img_proj.state_dict(),
    "text_proj": text_proj.state_dict(),
    "text_embed": text_embed.state_dict(),
    ...
}
```

---

## 📊 Complete Format Mapping

| Training Script | Config Key(s) | Expected Format | Our Output | Status |
|----------------|---------------|-----------------|------------|--------|
| `train_text.py` | `train_text` | Plain text (`.txt`) | `data/text/*.txt` | ✅ |
| `train_audio_enc.py` | `train_csv` | CSV: `wav,text` | `data/audio/*_asr.csv` | ✅ |
| `train_talker.py` | `tts_csv` | CSV: `text,wav` | `data/audio/production_tts.csv` | ✅ |
| `train_vision.py` | `train_manifest`<br>`image_root` | JSON array: `[{"image": "...", "caption": "..."}]` | `data/images/*_annotations.json` | ✅ |

---

## 🎯 Example Config Files

### `configs/thinker_tiny.json`
```json
{
  "train_text": "data/text/production_corpus.txt",
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

## ✅ Final Verification

All download scripts output data in the **exact format** required by training scripts:

1. ✅ **Text**: Plain text, one line per sample → `train_text.py`
2. ✅ **Audio ASR**: CSV with `wav,text` → `train_audio_enc.py`
3. ✅ **Audio TTS**: CSV with `text,wav` → `train_talker.py`
4. ✅ **Images**: JSON array with `image`, `caption` → `train_vision.py`

**No additional formatting needed!** Data is ready to use directly from download scripts.

