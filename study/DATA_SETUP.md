# Quick Start Option A Setup Guide

This guide helps you set up **Quick Start Option A** for training your μOmni model with industry-standard datasets.

## 🚀 Quick Start (Automated)

**Easiest way**: Use the automated download and format script:

```bash
# 1. Check your setup first
python scripts/check_setup.py

# 2. Download and format all datasets (supports resume)
python scripts/download_and_format_datasets.py

# 3. Resume if interrupted (automatically skips completed steps)
python scripts/download_and_format_datasets.py

# 4. Download specific dataset only
python scripts/download_and_format_datasets.py --dataset text
python scripts/download_and_format_datasets.py --dataset images
python scripts/download_and_format_datasets.py --dataset audio
```

**Features**:
- ✅ Automatic download with resume support
- ✅ Progress tracking (saves state to `data/.download_state.json`)
- ✅ Automatic format conversion
- ✅ Skips already downloaded/converted files
- ✅ Can resume from interruptions

**Manual Setup**: If you prefer manual setup or the script fails, see sections below.

---

## 📦 Dataset Requirements

**Total Storage Needed**: ~50-60 GB

### 1. Text/Conversational Data: DialogStudio (~10 GB)

**⚠️ IMPORTANT: DialogStudio requires special setup**

**Prerequisites**:
1. **Datasets library version**: DialogStudio uses loading scripts that require `datasets<3.0.0`
   ```bash
   pip install 'datasets<3.0.0'
   ```
   This is already specified in `requirements.txt`, but if you installed a newer version, downgrade it.

2. **Accept the license**: Visit https://huggingface.co/datasets/Salesforce/dialogstudio and accept the dataset license

3. **Authenticate with HuggingFace**:
   ```bash
   # Option 1: Login via CLI (recommended)
   huggingface-cli login
   # Enter your HuggingFace token when prompted
   
   # Option 2: Set environment variable
   export HF_TOKEN=your_token_here  # Linux/Mac
   set HF_TOKEN=your_token_here     # Windows
   ```

**Download Options**:
- **HuggingFace**: `datasets/Salesforce/dialogstudio` (recommended, requires authentication)
- **GitHub**: https://github.com/Salesforce/DialogStudio (alternative, no auth needed)

**Steps** (automated script handles this):
```bash
# The automated script will:
# 1. Check for authentication
# 2. Download DialogStudio from HuggingFace
# 3. Convert to text format automatically
python scripts/download_and_format_datasets.py --dataset text
```

**Manual Steps** (if needed):
```bash
# Using HuggingFace datasets (requires authentication and trust_remote_code)
pip install 'datasets<3.0.0' huggingface_hub
huggingface-cli login  # Authenticate first!
python -c "from datasets import load_dataset; ds = load_dataset('Salesforce/dialogstudio', trust_remote_code=True); ..."
```

**Format Conversion**:
- DialogStudio provides multiple dialogue datasets
- The automated script converts to text format (one conversation per line)
- Manual conversion: See `scripts/download_and_format_datasets.py` for reference

### 2. Image-Caption Data: COCO 2017 (~25 GB)

**Download**:
- **Official**: https://cocodataset.org/#download
- Download: "2017 Train images" (18GB) + "2017 Val images" (1GB) + "2017 Train/Val annotations" (241MB)

**Steps**:
```bash
# Create directories
mkdir -p data/images/coco_images
mkdir -p data/images/coco_annotations

# Download (use wget or browser)
# Then extract:
# - images to: data/images/coco_images/
# - annotations to: data/images/coco_annotations/
```

**Format Conversion**:
- COCO provides JSON annotations with image paths and captions
- Need to convert to your format: `{"image": "path", "caption": "text"}`

### 3. Audio-Speech Data: LibriSpeech train-clean-100 (~6.3 GB)

**Download**:
- **OpenSLR**: https://www.openslr.org/12/
- Download: "train-clean-100.tar.gz" (6.3 GB)

**Steps**:
```bash
# Create directory
mkdir -p data/audio/librispeech

# Download and extract
cd data/audio/librispeech
# Extract tar.gz file
# Structure: LibriSpeech/train-clean-100/...
```

**Format Conversion**:
- LibriSpeech provides WAV files + transcriptions
- Need to create CSV: `wav,text` format

---

## 📋 Post-Training Data Preparation

After initial training, you may want to fine-tune on custom datasets. Use the post-training data preparation script:

### Prepare Post-Training Data

**Text Data (for Thinker):**
```bash
python scripts/prep_post_training_data.py \
    --input data/raw/my_text.txt \
    --output data/post_training/text.txt \
    --format text
```

**Audio ASR Data (for Audio Encoder):**
```bash
python scripts/prep_post_training_data.py \
    --input data/raw/audio/ \
    --output data/post_training/asr.csv \
    --format audio_asr
```

**Audio TTS Data (for Talker):**
```bash
python scripts/prep_post_training_data.py \
    --input data/raw/audio/ \
    --output data/post_training/tts.csv \
    --format audio_tts
```

**Image Data (for Vision Encoder):**
```bash
python scripts/prep_post_training_data.py \
    --input data/raw/images/ \
    --output data/post_training/images.json \
    --format images \
    --caption_file data/raw/captions.txt
```

**Features:**
- ✅ Automatic format conversion
- ✅ Recursive file discovery
- ✅ Duplicate removal (for text)
- ✅ Transcript/caption file support
- ✅ Progress reporting

**For detailed post-training workflow, see [Post-Training Guide](14_Post_Training.md)**

---

## 🔧 Data Conversion Scripts

**Available Scripts in `scripts/` directory:**

1. **`download_datasets.py`** - Main script for downloading and converting standard datasets
   - Handles DialogStudio, COCO, LibriSpeech
   - Automatic format conversion
   - Resume support with state tracking

2. **`prep_post_training_data.py`** - Prepare datasets for post-training/fine-tuning
   - Converts raw data to training format
   - Supports text, audio (ASR/TTS), and images
   - See [Post-Training Guide](14_Post_Training.md) for details

3. **`check_setup.py`** - Verify your setup and data files
   - Checks for required files
   - Verifies data directories
   - Validates dataset formats

4. **`make_synthetic_datasets.py`** - Generate synthetic test data
   - Creates toy datasets for testing
   - Useful for quick validation

**Note**: The automated script (`scripts/download_datasets.py`) handles all conversions automatically. The scripts below are for reference or manual use.

### Convert DialogStudio to Text Format

The automated script handles this, but for manual conversion, see `scripts/download_and_format_datasets.py` or create `scripts/convert_dialogstudio.py`:
```python
import json
from datasets import load_dataset

# Load DialogStudio
ds = load_dataset('Salesforce/dialogstudio')

# Extract conversations and save as text
with open('data/text/dialogstudio.txt', 'w', encoding='utf-8') as f:
    for split in ['train', 'validation', 'test']:
        if split in ds:
            for item in ds[split]:
                # Extract conversation turns
                if 'conversations' in item:
                    conv_text = ' '.join([turn.get('content', '') for turn in item['conversations']])
                    f.write(conv_text + '\n')
```

### Convert COCO to Image Manifest

The automated script handles this, but for manual conversion, see `scripts/download_and_format_datasets.py` or create `scripts/convert_coco.py`:
```python
import json
import os

# Load COCO annotations
with open('data/images/coco_annotations/annotations/captions_train2017.json', 'r') as f:
    coco_data = json.load(f)

# Create image manifest
manifest = []
image_dir = 'data/images/coco_images/train2017'

for ann in coco_data['annotations']:
    # Find image info
    img_id = ann['image_id']
    img_info = next((img for img in coco_data['images'] if img['id'] == img_id), None)
    if img_info:
        manifest.append({
            "image": os.path.join(image_dir, img_info['file_name']),
            "caption": ann['caption']
        })

# Save manifest
with open('data/images/annotations.json', 'w', encoding='utf-8') as f:
    json.dump(manifest, f, indent=2, ensure_ascii=False)

print(f"Created manifest with {len(manifest)} image-caption pairs")
```

### Convert LibriSpeech to ASR CSV

The automated script handles this, but for manual conversion, see `scripts/download_and_format_datasets.py` or create `scripts/convert_librispeech.py`:
```python
import os
import csv
import glob

# LibriSpeech structure: LibriSpeech/train-clean-100/speaker/chapter/...
base_dir = 'data/audio/librispeech/LibriSpeech/train-clean-100'

rows = []
for speaker_dir in glob.glob(os.path.join(base_dir, '*')):
    for chapter_dir in glob.glob(os.path.join(speaker_dir, '*')):
        # Find .txt transcription file
        txt_files = glob.glob(os.path.join(chapter_dir, '*.txt'))
        if txt_files:
            with open(txt_files[0], 'r') as f:
                for line in f:
                    parts = line.strip().split(' ', 1)
                    if len(parts) == 2:
                        audio_id, text = parts
                        wav_path = os.path.join(chapter_dir, audio_id + '.flac')
                        if os.path.exists(wav_path):
                            rows.append({"wav": wav_path, "text": text})

# Save CSV
with open('data/audio/asr.csv', 'w', encoding='utf-8', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['wav', 'text'])
    writer.writeheader()
    writer.writerows(rows)

print(f"Created ASR CSV with {len(rows)} entries")
```

---

## ⚙️ Update Config Files

### Update `configs/omni_sft_tiny.json`:

```json
{
  "sft_mix": {
    "text_path": "data/text/dialogstudio.txt",
    "image_manifest": "data/images/annotations.json",
    "image_root": "data/images",
    "asr_csv": "data/audio/asr.csv"
  },
  "max_steps": 5000,
  "batch_size": 2,
  "use_amp": true
}
```

### Update `configs/thinker_tiny.json`:

```json
{
  "train_text": "data/text/dialogstudio.txt",
  "max_steps": 2000,
  "batch_size": 8,
  "use_amp": true
}
```

### Update `configs/vision_tiny.json`:

```json
{
  "train_manifest": "data/images/annotations.json",
  "image_root": "data/images",
  "max_steps": 10000,
  "batch_size": 8,
  "use_amp": true
}
```

### Update `configs/audio_enc_tiny.json`:

```json
{
  "train_csv": "data/audio/asr.csv",
  "max_steps": 1000,
  "batch_size": 4,
  "use_amp": true
}
```

---

## 🚀 Training Commands

### Stage 1: Thinker Pretraining
```bash
python train_text.py --config configs/thinker_tiny.json
```

### Stage 2: Audio Encoder
```bash
python train_audio_enc.py --config configs/audio_enc_tiny.json
```

### Stage 3: Vision Encoder
```bash
python train_vision.py --config configs/vision_tiny.json
```

### Stage 4: Talker Training
```bash
python train_talker.py --config configs/talker_tiny.json
```

### Stage 5: Multimodal SFT
```bash
python sft_omni.py --config configs/omni_sft_tiny.json
```

---

## ⏱️ Expected Training Times (RTX 5070 Ti)

- **Stage 1 (Thinker)**: ~2-3 hours
- **Stage 2 (Audio)**: ~2-3 hours  
- **Stage 3 (Vision)**: ~10-15 hours
- **Stage 4 (Talker)**: ~2-3 hours
- **Stage 5 (SFT)**: ~12-18 hours

**Total**: ~28-42 hours (1.5-2 days)

**With AMP enabled**: Expect 1.5-2x speedup → **~18-28 hours**

---

## ✅ Verification Checklist

**Quick Check**:
```bash
python scripts/check_setup.py
```

**Manual Checklist**:
- [ ] DialogStudio downloaded and converted (~10 GB)
- [ ] COCO 2017 downloaded and extracted (~25 GB)
- [ ] LibriSpeech train-clean-100 downloaded (~6.3 GB)
- [ ] All conversion scripts run successfully
- [ ] Config files updated with correct paths
- [ ] `use_amp: true` added to all configs
- [ ] At least 60 GB free space available
- [ ] GPU drivers and CUDA installed

**Verify Data Files**:
```bash
# Check if files exist
ls -lh data/text/dialogstudio.txt
ls -lh data/images/annotations.json
ls -lh data/audio/asr.csv
```

---

## 🐛 Troubleshooting

### Download Script Issues

**Issue**: "Download interrupted"
- Script automatically resumes - just run it again
- Check `data/.download_state.json` to see progress
- Use `--reset` flag to start over: `python scripts/download_and_format_datasets.py --reset`

**Issue**: "HuggingFace datasets download fails"
- Check internet connection
- Try: `pip install --upgrade datasets huggingface_hub`
- **DialogStudio requires authentication**: 
  - Visit https://huggingface.co/datasets/Salesforce/dialogstudio
  - Accept the license
  - Run: `huggingface-cli login`
  - Or set: `export HF_TOKEN=your_token_here`

**Issue**: "Dataset 'Salesforce/dialogstudio' is a gated dataset"
- This means you haven't authenticated or accepted the license
- Follow the authentication steps above
- The script will check authentication and provide clear error messages

**Issue**: "Dataset scripts are no longer supported"
- This means your `datasets` library version is too new (>=3.0.0)
- DialogStudio requires `datasets<3.0.0` to work with loading scripts
- Solution: `pip install 'datasets<3.0.0'`
- This is already in `requirements.txt`, so reinstalling from requirements.txt will fix it

**Issue**: "COCO download is slow"
- COCO files are large (18GB+ for train images)
- Download supports resume - safe to interrupt and restart
- Consider downloading during off-peak hours

**Issue**: "LibriSpeech .flac files not supported"
- torchaudio supports .flac files natively
- If issues occur, convert to .wav: `ffmpeg -i input.flac output.wav`

### Training Issues

**Issue**: "Dataset not found"
- Check file paths in config files
- Verify data directories exist: `python scripts/check_setup.py`
- Check file permissions

**Issue**: "Out of memory"
- Reduce batch_size in configs
- AMP should help (already enabled)
- Close other applications

**Issue**: "AMP not working"
- Verify CUDA is available: `python -c "import torch; print(torch.cuda.is_available())"`
- Check PyTorch version supports AMP (1.6+)
- Look for "Mixed precision training (AMP) enabled" message

**Issue**: "Too many gradient explosion warnings"
- This was fixed! Make sure you have the latest code
- Gradients are now checked after unscaling (accurate detection)
- If still seeing many warnings, try reducing learning rate

---

## 📝 Notes

- **AMP is now enabled by default** in all training scripts
- Training times are estimates; actual times may vary
- Start with smaller `max_steps` for testing, then increase for full training
- Monitor GPU temperature and usage during training
- Save checkpoints regularly (configured in configs)

---

## 📚 Related Documentation

- **[Post-Training Guide](14_Post_Training.md)** - Fine-tuning and domain adaptation
- **[Training Workflow](07_Training_Workflow.md)** - Complete training process
- **[Quick Reference](QUICK_REFERENCE.md)** - Command cheat sheet

---

Good luck with your training! 🚀

