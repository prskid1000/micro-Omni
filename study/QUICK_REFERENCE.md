# Quick Reference Guide

> **💡 Don't understand a term?** Check the [Glossary](GLOSSARY.md) for simple explanations of technical terms!

## File Structure

```
study/
├── README.md                    # Start here!
├── 00_Introduction.md           # What is μOmni?
├── 01_Neural_Networks_Basics.md # Fundamentals
├── 02_Architecture_Overview.md  # System design
├── 03_Thinker_Deep_Dive.md     # Core LLM
├── 04_Audio_Encoder.md         # Speech input
├── 05_Vision_Encoder.md         # Image input
├── 06_Talker_Codec.md          # Speech output
├── 07_Training_Workflow.md     # How to train
├── 08_Inference_Guide.md       # How to use
├── 09_Hands_On_Exercises.md    # Practice
├── 14_Post_Training.md         # Post-training & fine-tuning
├── QUICK_REFERENCE.md          # This  file
└── diagrams/                    # Visual aids
```

## Key Concepts

### Neural Networks
- **Neuron**: Basic processing unit
- **Layer**: Group of neurons
- **Forward Pass**: Data flows through network
- **Backward Pass**: Gradients flow back (training)

### Transformers
- **Attention**: Focus on relevant parts
- **Self-Attention**: Look at all positions
- **Embeddings**: Convert tokens to vectors
- **Position Encoding**: Handle sequence order

### μOmni Components
- **Thinker**: Core language model (see [Glossary](GLOSSARY.md) for "transformer", "autoregressive")
- **Audio Encoder**: Processes speech (ASR with 98-char vocabulary) - see [Glossary](GLOSSARY.md) for "ASR", "CTC"
- **Vision Encoder**: Processes images (contrastive learning, CLIP-style) - see [Glossary](GLOSSARY.md) for "CLIP", "contrastive learning"
- **Talker**: Generates speech (see [Glossary](GLOSSARY.md) for "TTS")
- **RVQ Codec**: Audio quantization (see [Glossary](GLOSSARY.md) for "RVQ", "quantization", "codebook")
- **Vocoder**: Improved Griffin-Lim (mel filterbank inversion) - see [Glossary](GLOSSARY.md) for "vocoder", "mel spectrogram"
- **Projectors**: Align modalities (see [Glossary](GLOSSARY.md) for "projector", "embedding")

## Training Commands

```bash
# Stage A: Thinker
python train_text.py --config configs/thinker_tiny.json

# Stage B: Audio Encoder
python train_audio_enc.py --config configs/audio_enc_tiny.json

# Stage C: Vision Encoder
python train_vision.py --config configs/vision_tiny.json

# Stage D: Talker
python train_talker.py --config configs/talker_tiny.json

# Stage E: Multimodal SFT
python sft_omni.py --config configs/omni_sft_tiny.json
```

## Post-Training Commands

**Prepare data for post-training:**
```bash
# Text data (Thinker)
python scripts/prep_post_training_data.py \
    --input data/your_data.txt \
    --output data/post_training/text.txt \
    --format text

# Audio ASR data (Audio Encoder)
python scripts/prep_post_training_data.py \
    --input data/your_audio/ \
    --output data/post_training/asr.csv \
    --format audio_asr

# Audio TTS data (Talker)
python scripts/prep_post_training_data.py \
    --input data/your_audio/ \
    --output data/post_training/tts.csv \
    --format audio_tts

# Image data (Vision Encoder)
python scripts/prep_post_training_data.py \
    --input data/your_images/ \
    --output data/post_training/images.json \
    --format images
```

**Post-train models:**
```bash
# Post-train Thinker (text)
# Saves to: checkpoints/post_training/thinker_post_*.pt
python post_train.py \
    --config configs/thinker_tiny.json \
    --checkpoint checkpoints/thinker_tiny/thinker.pt \
    --new_dataset data/post_training/text.txt

# Post-train Audio Encoder (ASR)
# Saves to: checkpoints/post_training/audio_enc_post_*.pt
python post_train.py \
    --config configs/audio_enc_tiny.json \
    --checkpoint checkpoints/audio_enc_tiny/audio_enc.pt \
    --new_dataset data/post_training/asr.csv

# Post-train Vision Encoder
# Saves to: checkpoints/post_training/vision_post_*.pt
python post_train.py \
    --config configs/vision_tiny.json \
    --checkpoint checkpoints/vision_tiny/vision.pt \
    --new_dataset data/post_training/images.json

# Post-train Talker (TTS)
# Saves to: checkpoints/post_training/talker_post_*.pt
python post_train.py \
    --config configs/talker_tiny.json \
    --checkpoint checkpoints/talker_tiny/talker.pt \
    --new_dataset data/post_training/tts.csv

# Fine-tune (reset optimizer, lower LR)
python post_train.py \
    --config configs/thinker_tiny.json \
    --checkpoint checkpoints/thinker_tiny/thinker.pt \
    --new_dataset data/post_training/text.txt \
    --reset_optimizer \
    --lr 0.0001
```

**Note:** All post-training checkpoints are saved to `checkpoints/post_training/` directory to keep them separate from original pretrained models.

**See [Post-Training Guide](14_Post_Training.md) for detailed documentation.**

## Inference Commands

```bash
# Text only
python infer_chat.py --ckpt_dir checkpoints/thinker_tiny --text "Hello"

# Image + Text
python infer_chat.py --ckpt_dir checkpoints/omni_sft_tiny \
  --image path/to/image.png --text "Describe this"

# Audio Input
python infer_chat.py --ckpt_dir checkpoints/omni_sft_tiny \
  --audio_in path/to/audio.wav --text "What did you hear?"

# Text-to-Speech
python infer_chat.py --ckpt_dir checkpoints/omni_sft_tiny \
  --text "Hello world" --audio_out output.wav
```

## Configuration Files

- `configs/thinker_tiny.json` - Thinker settings
- `configs/audio_enc_tiny.json` - Audio encoder
- `configs/vision_tiny.json` - Vision encoder
- `configs/talker_tiny.json` - Talker + RVQ
- `configs/omni_sft_tiny.json` - Multimodal SFT

## Key Parameters

### Model Size
- `vocab_size`: Number of tokens (5000)
- `d_model`: Embedding dimension (256)
- `n_layers`: Number of transformer blocks (4)
- `n_heads`: Attention heads (4)
- `d_ff`: Feedforward dimension (1024)

### Training
- `max_steps`: Training steps (1000-5000)
- `batch_size`: Examples per batch (8)
- `gradient_accumulation_steps`: Accumulate gradients (1)
- `lr`: Learning rate (0.0003)
- `warmup_steps`: Warmup period (10)
- `use_amp`: Mixed precision training (true)
- `checkpoint_freq`: Save checkpoint every N steps (500)

### Audio
- `sample_rate`: Audio sample rate (16000)
- `mel_bins`: Mel spectrogram bins (128)
- `downsample_time`: Temporal reduction (8x)
- `frame_rate`: Target frame rate (12.5 Hz)
- `ctc_vocab_size`: Character vocabulary size (98: printable ASCII + special tokens)
- `max_text_len`: Maximum text length (64 characters)

### Vision
- `img_size`: Image size (224)
- `patch`: Patch size (16)
- `n_patches`: Patches per image (196)
- `embed_dim`: Contrastive embedding dimension (128)
- `vocab_size`: Caption vocabulary size (10000)
- `temperature`: Contrastive loss temperature (0.07)

## Data Formats

### Text
- Format: Plain text files (one example per line)
- Location: `data/text/dialogstudio.txt` (training) or `data/post_training/text.txt` (post-training)
- Processing: BPE tokenization
- Preparation: Use `scripts/prep_post_training_data.py --format text` for custom data

### Images
- Format: PNG/JPG
- Size: 224×224 (resized automatically)
- Location: `data/images/images/` (training) or custom directory (post-training)
- Annotations: `data/images/annotations.json` (JSON manifest with `{"image": "path", "caption": "text"}`)
- Preparation: Use `scripts/prep_post_training_data.py --format images` for custom data

### Audio
- Format: WAV, FLAC, MP3, M4A
- Sample Rate: 16 kHz (default)
- Location: `data/audio/wav/` (training) or custom directory (post-training)
- Metadata: 
  - ASR: `data/audio/asr.csv` (format: `wav,text`)
  - TTS: `data/audio/tts.csv` (format: `text,wav`)
- Preparation: Use `scripts/prep_post_training_data.py --format audio_asr` or `--format audio_tts` for custom data

## Training Features

### Automatic Resume
- **Checkpoint Detection**: Automatically finds latest checkpoint on startup
- **Full State Recovery**: Loads model, optimizer, scheduler, scaler, step, best_val_loss
- **Checkpoint Frequency**: Saves every `checkpoint_freq` steps (default: 500)
- **Best Model**: Saves `{model}_best.pt` when validation improves
- **Resume Behavior**: Skips already-processed batches, continues from exact step

### Mixed Precision (AMP)
- **Enabled by Default**: All training scripts use FP16 forward passes
- **Speedup**: 1.5-2x faster training and inference
- **Memory**: ~50% less VRAM usage
- **Gradient Scaling**: Automatic gradient scaling prevents underflow

### Gradient Accumulation
- **Configurable**: Set `gradient_accumulation_steps` in config
- **Effective Batch Size**: `batch_size × gradient_accumulation_steps`
- **Memory Efficient**: Train with larger effective batches on limited VRAM

### Evaluation Metrics
- **Text**: Perplexity (exponential of cross-entropy loss)
- **Audio**: Word Error Rate (WER) for ASR evaluation
- **Vision**: Contrastive loss (InfoNCE) for image-caption alignment

## 🔒 Numerical Stability & Safety Features

### Automatic Checks
- **Model Forward Passes**: All models check for NaN/Inf automatically
  - ThinkerLM, TalkerTiny, AudioEncoderTiny, ViTTiny
  - Raises RuntimeError with detailed counts if detected
- **Loss Validation**: Training scripts validate all losses
  - Checks for NaN/Inf and out-of-bounds values
  - Automatically skips invalid batches
- **Gradient Explosion Detection**: Checks gradient norms before clipping
  - Default threshold: 100.0 (configurable)
  - Automatically skips exploded batches
  - Proper AMP handling: unscales gradients before checking

### Automatic NaN Recovery (Thinker Models)
- **Automatic Checkpoint Recovery**: When NaN is detected in attention probabilities during training
  - **Affected Scripts**: `train_text.py`, `sft_omni.py`, `post_train.py` (thinker mode)
  - **Behavior**: Automatically reloads from last saved checkpoint and continues training
  - **Recovery Process**:
    1. Catches `RuntimeError: NaN detected in attention probabilities after softmax`
    2. Finds latest checkpoint in save directory
    3. Reloads model, optimizer, scheduler, and scaler states
    4. Resumes training from recovered checkpoint step
  - **Checkpoint Prefixes**:
    - `train_text.py`: `thinker_step_`
    - `sft_omni.py`: `omni_step_`
    - `post_train.py`: `{model_type}_post_step_`
  - **No Manual Intervention**: Training continues automatically after recovery
  - **Logging**: All recovery actions are logged with step numbers and checkpoint paths

### Utilities
- `validate_loss(loss, min_loss=-1e6, max_loss=1e6)` - Validate loss values
- `check_gradient_explosion(model, max_grad_norm=100.0)` - Check gradients
- `check_numerical_stability(tensor, name="tensor")` - Check tensors
- `reload_from_last_checkpoint(save_dir, checkpoint_prefix, device, logger, model, opt, scheduler, scaler)` - Reload from checkpoint

## Troubleshooting

### Download Script Issues

**Issue**: "Download interrupted"
- Script automatically resumes - just run it again
- Check `data/.download_state.json` to see progress
- Use `--reset` flag to start over: `python scripts/download_datasets.py --reset`

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

**Issue**: "Dataset scripts are no longer supported"
- This means your `datasets` library version is too new (>=3.0.0)
- DialogStudio requires `datasets<3.0.0` to work with loading scripts
- Solution: `pip install 'datasets<3.0.0'`

**Issue**: "COCO download is slow"
- COCO files are large (18GB+ for train images)
- Download supports resume - safe to interrupt and restart
- Consider downloading during off-peak hours

**Issue**: "LibriSpeech .flac files not supported"
- torchaudio supports .flac files natively
- If issues occur, convert to .wav: `ffmpeg -i input.flac output.wav`

### Training Issues
- **Loss not decreasing**: Check learning rate, data loading
- **Out of memory**: Reduce batch size or increase `gradient_accumulation_steps`
- **Training interrupted**: Automatically resumes from latest checkpoint
- **NaN values**: 
  - **Automatic detection**: Models check for NaN/Inf automatically
  - **Loss validation**: Training scripts validate losses
  - **Automatic recovery (Thinker models)**: When NaN detected in attention, automatically reloads from last checkpoint
  - **Solutions**: Check learning rate, gradient clipping, data preprocessing
- **Gradient explosion**: 
  - **Automatic detection**: Training scripts check gradient norms (after AMP unscaling)
  - **Recovery**: Exploded batches are automatically skipped
  - **Solutions**: Reduce learning rate, increase gradient clipping threshold
- **NaN in attention (Thinker models)**:
  - **Automatic recovery**: Scripts automatically reload from last checkpoint and continue
  - **No data loss**: Training state (step, optimizer, scheduler) is fully restored
  - **Check logs**: Recovery actions are logged with checkpoint paths

### Inference
- **Model not found**: Check checkpoint path (look for `{model}_best.pt` or `{model}_step_{N}.pt`)
- **Poor quality**: Model needs more training, verify checkpoint loaded correctly
- **Slow generation**: Use GPU, enable KV cache, use AMP (automatic)

**Issue**: "AMP not working"
- Verify CUDA is available: `python -c "import torch; print(torch.cuda.is_available())"`
- Check PyTorch version supports AMP (1.6+)
- Look for "Mixed precision training (AMP) enabled" message

**Issue**: "Too many gradient explosion warnings"
- Gradients are now checked after unscaling (accurate detection)
- If still seeing many warnings, try reducing learning rate

## Code Locations

- `omni/thinker.py` - Core LLM
- `omni/audio_encoder.py` - Audio processing
- `omni/vision_encoder.py` - Image processing
- `omni/talker.py` - Speech generation
- `omni/codec.py` - RVQ codec
- `omni/tokenizer.py` - Text tokenization

## Learning Path

1. **Beginner**: Read 00-02, try exercises
2. **Intermediate**: Read 03-06, understand components
3. **Advanced**: Read 07-08, train and deploy
4. **Expert**: Modify code, experiment

## Available Scripts

**In `scripts/` directory:**

### 1. `check_setup.py` - Verify Setup

**Purpose**: Check if your environment is ready for training

**Usage**:
```bash
python scripts/check_setup.py
```

**What it checks**:
- ✓ Python version (3.8+)
- ✓ Required packages (torch, torchaudio, torchvision, datasets, etc.)
- ✓ CUDA/GPU availability
- ✓ Disk space (warns if <60GB free)
- ✓ Data files existence

**Example output**:
```
✓ Python 3.10.0
✓ torch installed
✓ CUDA available (GPU: NVIDIA RTX 5070 Ti)
✓ Disk space: 120.5 GB free
✗ Data files not found
```

---

### 2. `download_datasets.py` - Download Standard Datasets

**Purpose**: Download and convert industry-standard datasets (DialogStudio, COCO, LibriSpeech)

**Usage**:
```bash
# Download and convert all datasets
python scripts/download_datasets.py

# Download specific dataset only
python scripts/download_datasets.py --dataset text
python scripts/download_datasets.py --dataset images
python scripts/download_datasets.py --dataset audio

# Skip download, only convert existing data
python scripts/download_datasets.py --skip-download

# Skip conversion, only download
python scripts/download_datasets.py --skip-convert

# Reset and start over
python scripts/download_datasets.py --reset
```

**Features**:
- ✅ Automatic download with resume support
- ✅ Progress tracking (saves state to `data/.download_state.json`)
- ✅ Automatic format conversion
- ✅ Skips already downloaded/converted files
- ✅ Can resume from interruptions

**Output files**:
- `data/text/dialogstudio.txt` - Text data
- `data/images/annotations.json` - Image manifest
- `data/audio/asr.csv` - ASR data
- `data/audio/tts.csv` - TTS data

**Note**: DialogStudio requires HuggingFace authentication. See Dataset Setup section above for details.

---

### 3. `prep_post_training_data.py` - Prepare Post-Training Data

**Purpose**: Convert raw data to training format for fine-tuning/post-training

**Usage**:

**Text Data (for Thinker)**:
```bash
python scripts/prep_post_training_data.py \
    --input data/raw/my_text.txt \
    --output data/post_training/text.txt \
    --format text
```

**Audio ASR Data (for Audio Encoder)**:
```bash
python scripts/prep_post_training_data.py \
    --input data/raw/audio/ \
    --output data/post_training/asr.csv \
    --format audio_asr \
    --sample_rate 16000
```

**Audio TTS Data (for Talker)**:
```bash
python scripts/prep_post_training_data.py \
    --input data/raw/audio/ \
    --output data/post_training/tts.csv \
    --format audio_tts \
    --sample_rate 16000
```

**Image Data (for Vision Encoder)**:
```bash
python scripts/prep_post_training_data.py \
    --input data/raw/images/ \
    --output data/post_training/images.json \
    --format images \
    --caption_file data/raw/captions.txt  # Optional
```

**Arguments**:
- `--input`: Input path (file for text, directory for audio/images)
- `--output`: Output file path
- `--format`: `text`, `audio_asr`, `audio_tts`, or `images`
- `--sample_rate`: Audio sample rate (default: 16000)
- `--caption_file`: Caption file for images (one caption per line)
- `--encoding`: Text file encoding (default: utf-8)

**Features**:
- ✅ Recursive file discovery (finds all files in subdirectories)
- ✅ Duplicate removal (for text data)
- ✅ Transcript/caption file support
- ✅ Progress reporting

**See [Post-Training Guide](14_Post_Training.md) for complete workflow.**

---

### 4. `make_synthetic_datasets.py` - Generate Test Data

**Purpose**: Create synthetic datasets for quick testing (no download needed)

**Usage**:
```bash
python scripts/make_synthetic_datasets.py
```

**What it creates**:
- `data/text/tiny_corpus.txt` - 5000 synthetic text sentences
- `data/images/images/*.png` - 500 synthetic images (geometric shapes)
- `data/images/annotations.json` - Image-caption manifest
- `data/audio/wav/*.wav` - 300 synthetic audio files (beeps)
- `data/audio/asr.csv` - ASR metadata
- `data/audio/tts.csv` - TTS metadata

**Use case**: Quick testing without downloading large datasets

**Note**: Synthetic data is very simple and only for testing. Use real datasets for actual training.

---

### 5. `download_examples.py` - Download Example Files

**Purpose**: Download small example files for testing inference

**Usage**:
```bash
python scripts/download_examples.py
```

**What it creates**:
- `examples/sample_image.png` - Test image
- `examples/sample_audio.mp3` - Test audio
- `examples/sample_text.txt` - Test text

**Use case**: Quick testing of inference without preparing full datasets

---

### 6. `create_test_video.py` - Create Test Video

**Purpose**: Create a test video from image frames

**Usage**:
```bash
python scripts/create_test_video.py
```

**Requirements**:
- PyAV: `pip install av`
- ffmpeg installed on system

**What it creates**:
- `examples/sample_video.mp4` - Test video from image frames

**Alternative** (if PyAV not available):
```bash
ffmpeg -framerate 1 -i data/images/images/%06d.png \
    -c:v libx264 -pix_fmt yuv420p examples/sample_video.mp4
```

---

## Dataset Setup

### Quick Start (Automated)

**Easiest way**: Use the automated download and format script:

```bash
# 1. Check your setup first
python scripts/check_setup.py

# 2. Download and format all datasets (supports resume)
python scripts/download_datasets.py

# 3. Resume if interrupted (automatically skips completed steps)
python scripts/download_datasets.py

# 4. Download specific dataset only
python scripts/download_datasets.py --dataset text
python scripts/download_datasets.py --dataset images
python scripts/download_datasets.py --dataset audio
```

**Features**:
- ✅ Automatic download with resume support
- ✅ Progress tracking (saves state to `data/.download_state.json`)
- ✅ Automatic format conversion
- ✅ Skips already downloaded/converted files
- ✅ Can resume from interruptions

### Dataset Requirements

**Total Storage Needed**: ~50-60 GB

#### 1. Text/Conversational Data: DialogStudio (~10 GB)

**⚠️ IMPORTANT: DialogStudio requires special setup**

**Prerequisites**:
1. **Datasets library version**: DialogStudio uses loading scripts that require `datasets<3.0.0`
   ```bash
   pip install 'datasets<3.0.0'
   ```

2. **Accept the license**: Visit https://huggingface.co/datasets/Salesforce/dialogstudio and accept the dataset license

3. **Authenticate with HuggingFace**:
   ```bash
   # Option 1: Login via CLI (recommended)
   huggingface-cli login
   
   # Option 2: Set environment variable
   export HF_TOKEN=your_token_here  # Linux/Mac
   set HF_TOKEN=your_token_here     # Windows
   ```

**Download**: The automated script handles this:
```bash
python scripts/download_datasets.py --dataset text
```

#### 2. Image-Caption Data: COCO 2017 (~25 GB)

**Download**: https://cocodataset.org/#download
- "2017 Train images" (18GB) + "2017 Val images" (1GB) + "2017 Train/Val annotations" (241MB)

**Automated**: 
```bash
python scripts/download_datasets.py --dataset images
```

#### 3. Audio-Speech Data: LibriSpeech train-clean-100 (~6.3 GB)

**Download**: https://www.openslr.org/12/
- Download: "train-clean-100.tar.gz" (6.3 GB)

**Automated**:
```bash
python scripts/download_datasets.py --dataset audio
```

### Update Config Files

After downloading datasets, update config files with correct paths:

**`configs/thinker_tiny.json`**:
```json
{
  "train_text": "data/text/dialogstudio.txt",
  "max_steps": 1000000,
  "batch_size": 8,
  "use_amp": true
}
```

**`configs/vision_tiny.json`**:
```json
{
  "train_manifest": "data/images/annotations.json",
  "image_root": "data/images",
  "max_steps": 250000,
  "batch_size": 8,
  "use_amp": true
}
```

**`configs/audio_enc_tiny.json`**:
```json
{
  "train_csv": "data/audio/asr.csv",
  "max_steps": 20000,
  "batch_size": 4,
  "use_amp": true
}
```

**`configs/talker_tiny.json`**:
```json
{
  "tts_csv": "data/audio/tts.csv",
  "max_steps": 20000,
  "batch_size": 4,
  "use_amp": true
}
```

**`configs/omni_sft_tiny.json`**:
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

### Expected Training Times

**With AMP enabled** (RTX 5070 Ti or similar):
- **Stage 1 (Thinker)**: ~41 hours (1M steps, ~2 epochs)
- **Stage 2 (Audio)**: ~1.1 hours (20K steps, ~3 epochs)
- **Stage 3 (Vision)**: ~17.4 hours (250K steps, ~3.6 epochs)
- **Stage 4 (Talker)**: ~1.1 hours (20K steps, ~3 epochs)
- **Stage 5 (SFT)**: ~12-18 hours

**Total**: ~72-78 hours (~3 days) for full training

### Verification Checklist

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
ls -lh data/audio/tts.csv
```

## Script Usage Workflow

### Initial Setup
```bash
# 1. Check your setup
python scripts/check_setup.py

# 2. Download standard datasets (or use synthetic for testing)
python scripts/download_datasets.py
# OR
python scripts/make_synthetic_datasets.py  # For quick testing

# 3. Verify data files
python scripts/check_setup.py
```

### Post-Training Setup
```bash
# 1. Prepare your custom data
python scripts/prep_post_training_data.py \
    --input your_data.txt \
    --output data/post_training/text.txt \
    --format text

# 2. Post-train the model
python post_train.py \
    --config configs/thinker_tiny.json \
    --checkpoint checkpoints/thinker_tiny/thinker.pt \
    --new_dataset data/post_training/text.txt
```


## Important Notes

- **AMP is enabled by default** in all training scripts
- Training times are estimates; actual times may vary
- Start with smaller `max_steps` for testing, then increase for full training
- Monitor GPU temperature and usage during training
- Save checkpoints regularly (configured in configs)
- DialogStudio requires HuggingFace authentication - see setup section above

## Resources

- Main README: `../README.md`
- [Post-Training Guide](14_Post_Training.md) - Fine-tuning workflows
- [Training Workflow](07_Training_Workflow.md) - Complete training process
- PyTorch Docs: https://pytorch.org/docs/
- Transformer Paper: "Attention Is All You Need"
- ViT Paper: "An Image is Worth 16x16 Words"

## Tips

1. **Start Small**: Use tiny configs first
2. **Read Code**: Look at actual implementations
3. **Experiment**: Try different parameters
4. **Debug**: Add print statements, check shapes
5. **Visualize**: Plot losses, attention weights

---

**Need Help?**
- Check the main [README.md](../README.md)
- Review specific component guides
- Look at code examples in `examples/`
- Run `test_all_media.py` to verify setup

