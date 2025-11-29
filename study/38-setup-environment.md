# Chapter 38: Environment Setup

[← Previous: Debugging](37-debugging-troubleshooting.md) | [Back to Index](00-INDEX.md) | [Next: Running Training →](39-running-training.md)

---

## 🛠️ Setting Up μOmni

Complete setup guide for training and running μOmni.

---

## 📋 Prerequisites

**Hardware:**
- GPU: 12GB+ VRAM (RTX 3060, RTX 4060, or better)
- RAM: 16GB+ system memory
- Storage: 20GB+ free space

**Software:**
- Python 3.8+
- CUDA 11.8+ (for GPU)
- Git

---

## 🚀 Installation Steps

### 1. Clone Repository

```bash
git clone https://github.com/yourusername/μOmni.git
cd μOmni
```

### 2. Create Virtual Environment

```bash
# Using venv
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows

# Using conda (alternative)
conda create -n muomni python=3.10
conda activate muomni
```

### 3. Install Dependencies

```bash
# Core dependencies
pip install -r requirements.txt

# This installs:
# - torch (PyTorch)
# - torchaudio
# - torchvision
# - librosa (audio processing)
# - soundfile (audio I/O)
# - Pillow (image processing)
# - sentencepiece (tokenization)
# - numpy, scipy
# - tqdm (progress bars)
# - einops (tensor operations)
# - requests (HTTP downloads)

# Optional: Flash Attention (2-4x speedup)
pip install flash-attn --no-build-isolation
```

### 4. Verify Installation

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"

# Expected output:
PyTorch: 2.1.0+cu118
CUDA: True
```

---

## 📂 Create Required Directories

```bash
# Data directories
mkdir -p data/text data/images data/audio/asr data/audio/tts data/multimodal

# Checkpoint directories
mkdir -p checkpoints/thinker_tiny checkpoints/audio_enc_tiny 
mkdir -p checkpoints/vision_tiny checkpoints/talker_tiny 
mkdir -p checkpoints/omni_sft_tiny

# Examples directory (if not exists)
mkdir -p examples
```

---

## 🎯 Quick Test

```bash
# Test import
python -c "from omni import ThinkerLM, AudioEncoderTiny, ViTTiny; print('✓ All modules loaded')"

# Test GPU
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0)}')"
```

---

## 💡 Troubleshooting Setup

**Issue:** `CUDA not available`
```bash
# Check CUDA version
nvidia-smi

# Reinstall PyTorch with correct CUDA version
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**Issue:** `ImportError: No module named 'flash_attn'`
```bash
# Flash Attention is optional
# Skip if installation fails, training will work without it
```

---

[Continue to Chapter 39: Running Training →](39-running-training.md)

---
