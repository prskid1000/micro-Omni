# Chapter 35: Data Preparation and Datasets

[← Previous: Configuration Files](34-configuration-files.md) | [Back to Index](00-INDEX.md) | [Next: Optimization Techniques →](36-optimization-techniques.md)

---

## 🎯 Data Requirements

μOmni trains on **small datasets** (<5GB per modality) to fit 12GB GPU.

---

## 📝 Text Data

### Format

```
# data/text/corpus.txt (plain text)
The quick brown fox jumps over the lazy dog.
Machine learning is a subset of artificial intelligence.
Neural networks can learn complex patterns from data.
...
```

### Preparation

```bash
# Option 1: Use your own text
cat your_documents.txt > data/text/corpus.txt

# Option 2: Download WikiText
python scripts/download_datasets.py --modality text

# Option 3: Generate synthetic (testing)
python scripts/make_synthetic_datasets.py
```

### Size Recommendations

| Dataset | Size | Tokens | Training Time |
|---------|------|--------|---------------|
| **Tiny (test)** | 2MB | ~400K | 30min |
| **Small** | 50MB | ~10M | 4hrs |
| **Medium** | 200MB | ~40M | 12hrs |

---

## 🎤 Audio Data (ASR)

### Format

```csv
# data/audio/asr.csv
audio_path,transcription
data/audio/wav/sample1.wav,"hello world"
data/audio/wav/sample2.wav,"how are you today"
data/audio/wav/sample3.wav,"the quick brown fox"
```

### Audio Requirements

- **Format**: WAV, 16kHz, mono
- **Duration**: 1-30 seconds per clip
- **Quality**: Clear speech, minimal background noise

### Preparation

```bash
# Download LibriSpeech (subset)
python scripts/download_datasets.py --modality audio_asr

# Or prepare your own
ffmpeg -i input.mp3 -ar 16000 -ac 1 output.wav
```

### Creating CSV

```python
import os
import pandas as pd

data = []
for wav_file in os.listdir("data/audio/wav/"):
    if wav_file.endswith(".wav"):
        # Extract transcription (from filename or separate file)
        transcription = extract_transcription(wav_file)
        data.append({
            "audio_path": f"data/audio/wav/{wav_file}",
            "transcription": transcription
        })

df = pd.DataFrame(data)
df.to_csv("data/audio/asr.csv", index=False)
```

---

## 🔊 Audio Data (TTS)

### Format

```
data/audio/tts/
├── audio1.wav
├── audio2.wav
├── audio3.wav
└── ...

No transcriptions needed!
```

### Requirements

- Same as ASR (16kHz WAV mono)
- Consistent speaker (optional but better)
- Clean, expressive speech

### Preparation

```bash
# Download LJSpeech (subset)
python scripts/download_datasets.py --modality audio_tts

# Or use your own recordings
# Ensure consistent quality and format
```

---

## 🖼️ Image Data

### Format

```json
// data/images/annotations.json
{
  "images": [
    {
      "id": 1,
      "file_name": "cat.jpg",
      "caption": "An orange tabby cat sitting on a blue couch"
    },
    {
      "id": 2,
      "file_name": "dog.jpg",
      "caption": "A golden retriever playing in the park"
    }
  ]
}
```

### Directory Structure

```
data/images/
├── annotations.json
├── cat.jpg
├── dog.jpg
└── ...
```

### Requirements

- **Format**: JPG or PNG
- **Size**: Any (will be resized to 224×224)
- **Quality**: Clear, well-lit images

### Preparation

```bash
# Download COCO (validation set)
python scripts/download_datasets.py --modality images

# Or create your own
python scripts/create_image_annotations.py
```

---

## 🌈 Multimodal Data (SFT)

### Text-Only Examples

```json
{
  "type": "text",
  "input": "What is the capital of France?",
  "output": "The capital of France is Paris."
}
```

### Image + Text Examples

```json
{
  "type": "image_text",
  "image_path": "data/images/cat.jpg",
  "input": "What animal is in this image?",
  "output": "This is a cat."
}
```

### Audio + Text Examples

```json
{
  "type": "audio_text",
  "audio_path": "data/audio/speech.wav",
  "input": "What did the speaker say?",
  "output": "The speaker said hello world."
}
```

### Combined Format

```json
// data/multimodal/sft_data.json
{
  "samples": [
    {
      "type": "text",
      "input": "Explain AI",
      "output": "AI is artificial intelligence..."
    },
    {
      "type": "image_text",
      "image_path": "images/cat.jpg",
      "input": "Describe this",
      "output": "A cat sitting on furniture"
    },
    {
      "type": "audio_text",
      "audio_path": "audio/hello.wav",
      "input": "Transcribe",
      "output": "Hello world"
    }
  ]
}
```

---

## 🔧 Data Preprocessing Scripts

### Create Synthetic Data (Testing)

```bash
python scripts/make_synthetic_datasets.py

Output:
- data/text/tiny_corpus.txt (~2MB)
- data/images/ with generated annotations
- data/audio/ with test samples
```

### Download Real Datasets

```bash
# Text
python scripts/download_datasets.py --modality text
# Downloads WikiText-103 (~200MB)

# Images
python scripts/download_datasets.py --modality images
# Downloads COCO validation (~1GB)

# Audio ASR
python scripts/download_datasets.py --modality audio_asr
# Downloads LibriSpeech subset (~3GB)

# Audio TTS
python scripts/download_datasets.py --modality audio_tts
# Downloads LJSpeech subset (~1.5GB)
```

---

## 📊 Dataset Size Guidelines

### Minimum (Testing)
```
Text: 1-2MB (test model works)
Images: 100 images
Audio: 100 clips
Time: ~1 hour total training
```

### Recommended (Good Quality)
```
Text: 50-200MB
Images: 5K-10K images
Audio: 2K-5K clips (10-20 hours)
Time: ~40-60 hours total training
```

### Optimal (Best Quality)
```
Text: 500MB-1GB
Images: 50K+ images
Audio: 10K+ clips (50+ hours)
Time: ~100+ hours training
```

---

## 💡 Data Quality Tips

### Text
✅ Diverse topics
✅ Proper grammar and spelling
✅ Varied sentence lengths
❌ Avoid repetitive content

### Audio
✅ Clear pronunciation
✅ Consistent recording quality
✅ Minimal background noise
❌ Avoid music/overlapping speech

### Images
✅ High resolution (>300px)
✅ Good lighting
✅ Clear subjects
❌ Avoid blurry/dark images

---

## 🔍 Verify Data

```python
# Check text data
with open("data/text/corpus.txt") as f:
    text = f.read()
    print(f"Text size: {len(text)} chars")
    print(f"Lines: {text.count(chr(10))}")

# Check audio data
import pandas as pd
df = pd.read_csv("data/audio/asr.csv")
print(f"Audio samples: {len(df)}")
print(df.head())

# Check image data
import json
with open("data/images/annotations.json") as f:
    data = json.load(f)
    print(f"Images: {len(data['images'])}")
```

---

## 💡 Key Takeaways

✅ **Small datasets** work (<5GB per modality)  
✅ **Consistent format** for each modality  
✅ **Quality > Quantity** (clean data essential)  
✅ **Synthetic data** great for testing  
✅ **Scripts provided** for common datasets

---

[Continue to Chapter 36: Optimization Techniques →](36-optimization-techniques.md)

