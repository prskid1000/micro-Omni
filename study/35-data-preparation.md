# Chapter 35: Data Preparation Guide

[← Previous: Configuration Files](34-configuration-files.md) | [Back to Index](00-INDEX.md) | [Next: Optimization Techniques →](36-optimization-techniques.md)

---

## 🎯 Preparing Training Data

Each training stage requires specific data formats. This chapter explains data preparation for all stages.

---

## 📊 Data Requirements by Stage

### Stage A: Text Data

```bash
# Format: Plain text file
# Location: data/text/corpus.txt

# Example content:
The cat sat on the mat.
Hello world, how are you?
Machine learning is fascinating.
# ... more text (aim for 10MB+ for tiny model)
```

### Stage B: Audio Data (ASR)

```csv
# Format: CSV with audio paths + transcriptions
# Location: data/audio/asr.csv

audio_path,transcription
data/audio/wav/sample1.wav,"hello world"
data/audio/wav/sample2.wav,"how are you"
data/audio/wav/sample3.wav,"the cat sat on the mat"

# Requirements:
# - .wav files, 16kHz sample rate
# - 1-10 second clips
# - ~1000 samples minimum
```

### Stage C: Image Data

```json
// Format: JSON with image paths + captions/labels
// Location: data/images/annotations.json

{
  "images": [
    {
      "id": 1,
      "file_name": "cat.jpg",
      "caption": "A cat sitting on a couch",
      "category": "animal"
    }
  ]
}

// Requirements:
// - .jpg or .png, 224×224 (will be resized)
// - ~1000 images minimum
```

### Stage D: Speech Data (TTS)

```bash
# Format: Audio files only (no transcriptions needed)
# Location: data/audio/tts/

data/audio/tts/
├── speech1.wav
├── speech2.wav
└── speech3.wav

# Requirements:
# - .wav files, 16kHz
# - Clean speech, 1-10 seconds
# - ~500-1000 samples
```

### Stage E: Multimodal Data

```bash
# Mix of text, image, and audio data
data/multimodal/
├── text/conversations.json
├── images/image_qa.json
└── audio/audio_qa.json

# Each with question-answer pairs including modality
```

---

## 🛠️ Data Generation Scripts

```bash
# Generate synthetic data (for testing)
python scripts/make_synthetic_datasets.py

# Download sample datasets
python scripts/download_datasets.py

# Verify data format
python scripts/check_data.py --stage A
```

---

## 💡 Tips

✅ **Start small:** 100-1000 samples per modality for proof-of-concept  
✅ **Quality > Quantity:** Clean data more important than large datasets  
✅ **Balance:** Equal amounts per modality for Stage E  
✅ **Synthetic data:** Good for initial testing/debugging

---

[Continue to Chapter 36: Optimization Techniques →](36-optimization-techniques.md)

---
