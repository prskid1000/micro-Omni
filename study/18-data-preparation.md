[← Previous: 17-ocr-model](17-ocr-model.md) | [Index](00-INDEX.md) | [Next: 19-training-pipeline →](19-training-pipeline.md)

# Chapter 18: Data Preparation

Training a multimodal model means feeding five different pipelines with five
different data formats. Get the format wrong and the training script silently
loads zero samples. This chapter covers exactly what each stage expects, how
to download and combine the production datasets, and how to verify everything
before you start burning GPU hours.

---

## 18.1 Data Formats by Stage

| Stage | File | Format | Key Columns / Fields |
|-------|------|--------|----------------------|
| A — Thinker | `production_corpus.txt` | Plain text, one sample per line | Raw text |
| B — Audio Encoder | `production_asr.csv` | CSV | `wav,text` (audio path first) |
| C — Vision | `production_annotations.json` | JSON manifest | `[{"image": "path.jpg", "caption": "..."}]` |
| D — Talker | `production_tts.csv` | CSV | `text,wav` (text first — reversed from ASR!) |
| E — OCR | `production_ocr.csv` | CSV | `image,text` |

> **Watch out:** Stages B and D both use CSV with audio and text columns, but
> the column order is swapped. The ASR CSV is `wav,text` because the model
> reads audio and produces text. The TTS CSV is `text,wav` because the model
> reads text and produces audio. Mix them up and you get a model that "listens"
> to text strings.

---

## 18.2 Download Scripts

Every production dataset has a corresponding download script under `scripts/`:

```
scripts/
  download_production_corpus.py      # Stage A — text
  download_production_asr.py         # Stage B — audio encoder
  download_production_vision.py      # Stage C — vision
  download_production_tts.py         # Stage D — talker
  download_production_ocr.py         # Stage E — OCR (if applicable)
```

### The --combine Flag (MANDATORY)

Each download script supports `--combine`, and you **must** use it:

```bash
python -m scripts.download_production_text --combine
python -m scripts.download_production_audio --combine
python -m scripts.download_production_image --combine
python -m scripts.download_production_ocr --combine
```

Without `--combine`, the script downloads raw shards but does not merge them
into the single file the training scripts expect. The training loop will start,
find zero samples, and silently produce garbage.

What `--combine` does:

```
download_production_asr.py
  1. Downloads individual shard CSVs   →  data/raw/asr_shard_*.csv
  2. --combine merges them             →  data/production_asr.csv
                                           (single file, ready for training)
```

---

## 18.3 Chinchilla Scaling

The Chinchilla scaling law gives a rule of thumb: optimal training uses roughly
20 tokens per parameter. For the 25M-parameter Thinker:

```
25,000,000 params  x  20 tokens/param  =  500,000,000 tokens minimum
```

Going below 500M tokens risks underfitting — the model has capacity it never
learns to use. Going above is fine but has diminishing returns.

For the other stages, the equivalent guidance is:

| Stage | Minimum Dataset Size (approximate) |
|-------|------------------------------------|
| Thinker | ~500M tokens |
| Audio Encoder | ~1000 hours of transcribed audio |
| Vision | ~500K image-caption pairs |
| Talker | ~500 hours of text-audio pairs |

---

## 18.4 Auto-Calculated Dataset Statistics

Training configs use **percentile-based maximum lengths** rather than
hard-coded values. This is critical for efficiency.

```
Example: max_text_length_percentile = 95
```

The data loader computes the 95th-percentile text length at startup and uses
that as the sequence length. This means:

- 95% of samples fit without truncation
- The remaining 5% of outliers are trimmed
- Padding waste is minimized (compared to using the absolute maximum)

The same logic applies to audio lengths and image dimensions:

```
+---------------------------------------------------------+
|                   Text Length Distribution               |
|                                                         |
|   ##                                                    |
|   ###                                                   |
|   ####                                                  |
|   ######                                                |
|   ########                                              |
|   ###########                                           |
|   ################                              |       |
|   #########################            95th pctl|       |
|   ##########################################    v       |
|   ############################################-|--###   |
+---------------------------------------------------------+
     Short                                          Long
                           ^
                    Most samples fit here.
                    Outliers past the line get truncated.
```

You can override percentiles in the training config:

```python
max_text_length_percentile = 95    # default — good for most datasets
max_audio_length_percentile = 95   # increase to 98 if audio is varied
```

---

## 18.5 Expected Directory Structure

Training scripts expect this layout:

```
data/
  production_corpus.txt          # Stage A
  production_asr.csv             # Stage B
  production_annotations.json    # Stage C
  images/                        # Stage C — referenced by JSON paths
    img_000001.jpg
    img_000002.jpg
    ...
  production_tts.csv             # Stage D
  audio/                         # Stages B & D — referenced by CSV paths
    clip_000001.wav
    clip_000002.wav
    ...
  production_ocr.csv             # Stage E (optional)
  ocr_images/                    # Stage E — referenced by CSV paths
    page_000001.png
    ...
```

All paths inside CSVs and JSON are **relative to the `data/` directory**.
A line in `production_asr.csv` looks like:

```
audio/clip_000001.wav,the quick brown fox
```

---

## 18.6 Data Quality Checklist

### Text
- UTF-8 encoding, no BOM
- Strip leading/trailing whitespace per line
- Remove blank lines (they become empty training samples)
- Normalize Unicode (NFC form) so tokenization is consistent
- Remove or replace control characters

### Audio
- Format: **16 kHz, mono, 16-bit WAV**
- All files must be decodable (`soundfile.read` should not throw)
- Trim leading/trailing silence (optional but helps)
- Reject clips shorter than 0.5s or longer than 30s
- Consistent volume normalization (peak normalize to -1 dBFS)

```bash
# Quick audio format check with ffprobe
ffprobe -v error -show_entries stream=sample_rate,channels,codec_name \
  -of csv=p=0 data/audio/clip_000001.wav
# Expected output: pcm_s16le,16000,1
```

### Images
- Common formats: JPEG or PNG
- Minimum resolution: 224x224 (below this, resize artifacts dominate)
- RGB color space (convert grayscale to RGB with 3 identical channels)
- No corrupt files (PIL should open without errors)

### Quick Validation Script

Before training, do a sanity pass:

```python
import csv, json, os

# Check ASR CSV
with open("data/production_asr.csv") as f:
    reader = csv.reader(f)
    for i, row in enumerate(reader):
        assert len(row) == 2, f"Row {i}: expected 2 columns, got {len(row)}"
        assert os.path.exists(f"data/{row[0]}"), f"Row {i}: missing {row[0]}"
    print(f"ASR CSV: {i+1} samples, all files exist")

# Check vision JSON
with open("data/production_annotations.json") as f:
    annots = json.load(f)
    for i, entry in enumerate(annots):
        assert "image" in entry and "caption" in entry
        assert os.path.exists(f"data/{entry['image']}")
    print(f"Vision JSON: {len(annots)} samples, all files exist")
```

---

## 18.7 Summary

```
+------------------+     --combine      +-------------------+
| download script  | ----------------> | merged data file  |
| (fetches shards) |    (MANDATORY)    | (training-ready)  |
+------------------+                   +-------------------+
                                              |
                                              v
                                     +------------------+
                                     | training script  |
                                     | reads from data/ |
                                     +------------------+
```

Get the data right and training is straightforward. Get it wrong and you spend
days debugging a model that was never fed proper input.

**Next:** Chapter 19 walks through the five training stages that consume this data.
