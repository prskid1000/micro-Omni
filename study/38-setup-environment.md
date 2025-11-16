# Chapter 38: Setting Up Your Environment

[Back to Index](00-INDEX.md) | [Next: Running Training →](39-running-training.md)

---

## 🎯 What You'll Learn

- System requirements
- Installing dependencies
- Setting up Python environment
- Verifying installation
- Troubleshooting common issues

---

## 💻 System Requirements

### Minimum Requirements

```
Hardware:
- GPU: NVIDIA GPU with 12GB VRAM (e.g., RTX 3060, RTX 4060 Ti)
- RAM: 16GB system memory
- Storage: 50GB free space (for datasets + checkpoints)
- CPU: Any modern CPU (4+ cores recommended)

Software:
- OS: Windows 10/11, Linux (Ubuntu 20.04+), or macOS
- Python: 3.8 - 3.11
- CUDA: 11.7+ (for GPU support)
- Git: For cloning repository
```

### Recommended Requirements

```
Hardware:
- GPU: NVIDIA RTX 3080/4080 (12-16GB VRAM)
- RAM: 32GB system memory
- Storage: 100GB+ SSD
- CPU: 8+ cores

Software:
- Python 3.10 (optimal compatibility)
- CUDA 11.8 or 12.1
- PyTorch 2.0+ (for Flash Attention)
```

---

## 📥 Installation Steps

### Step 1: Clone Repository

```bash
# Clone the μOmni repository
git clone https://github.com/your-repo/muOmni.git
cd muOmni

# Check you're in the right directory
ls
# Should see: omni/, configs/, train_text.py, etc.
```

---

### Step 2: Create Virtual Environment

**Option A: venv (Python built-in)**

```bash
# Create virtual environment
python -m venv .venv

# Activate (Windows)
.venv\Scripts\activate

# Activate (Linux/Mac)
source .venv/bin/activate

# Verify activation (prompt should show (.venv))
which python
# Should point to .venv/bin/python
```

**Option B: conda**

```bash
# Create conda environment
conda create -n muomni python=3.10
conda activate muomni

# Verify
which python
# Should be in conda envs
```

---

### Step 3: Install Dependencies

```bash
# Upgrade pip
pip install --upgrade pip

# Install PyTorch (CUDA 11.8 example)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Verify PyTorch installation
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
# Should print version (e.g., 2.0.1) and True

# Install other dependencies
pip install -r requirements.txt
```

---

### Step 4: Verify GPU Setup

```python
# test_gpu.py
import torch

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"GPU count: {torch.cuda.device_count()}")

if torch.cuda.is_available():
    print(f"GPU name: {torch.cuda.get_device_name(0)}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# Test tensor creation on GPU
x = torch.randn(1000, 1000, device='cuda')
print("✓ GPU tensor creation successful!")
```

```bash
python test_gpu.py
# Expected output:
# PyTorch version: 2.0.1
# CUDA available: True
# CUDA version: 11.8
# GPU count: 1
# GPU name: NVIDIA GeForce RTX 3060
# GPU memory: 12.00 GB
# ✓ GPU tensor creation successful!
```

---

## 📦 Dependencies Explained

### Core Dependencies (requirements.txt)

```txt
torch>=2.0.0          # Deep learning framework
torchvision>=0.15.0   # Image processing
torchaudio>=2.0.0     # Audio processing
einops>=0.6.0         # Tensor operations
sentencepiece>=0.1.99 # Tokenizer
soundfile>=0.12.0     # Audio I/O
librosa>=0.10.0       # Audio processing
numpy>=1.24.0         # Numerical computing
Pillow>=9.0.0         # Image I/O
tqdm>=4.65.0          # Progress bars
```

### Optional Dependencies

```bash
# For visualization
pip install matplotlib seaborn

# For Jupyter notebooks
pip install jupyter ipywidgets

# For data processing
pip install pandas

# For model export
pip install onnx onnxruntime
```

---

## 🔧 Configuration Files

### Directory Structure

```
μOmni/
├── configs/
│   ├── thinker_tiny.json      # Language model config
│   ├── audio_enc_tiny.json    # Audio encoder config
│   ├── vision_tiny.json       # Vision encoder config
│   ├── talker_tiny.json       # Speech generator config
│   └── omni_sft_tiny.json     # Multimodal fine-tuning config
├── data/                      # Training data (create if needed)
│   ├── text/
│   ├── images/
│   └── audio/
├── checkpoints/               # Model weights (create if needed)
│   ├── thinker_tiny/
│   ├── audio_enc_tiny/
│   ├── vision_tiny/
│   ├── talker_tiny/
│   └── omni_sft_tiny/
└── examples/                  # Example inputs
    ├── sample_image.png
    ├── sample_audio.wav
    └── sample_text.txt
```

### Create Directories

```bash
# Create necessary directories
mkdir -p data/text data/images data/audio
mkdir -p checkpoints/thinker_tiny checkpoints/audio_enc_tiny
mkdir -p checkpoints/vision_tiny checkpoints/talker_tiny checkpoints/omni_sft_tiny
```

---

## ✅ Verify Installation

### Run Setup Check Script

```bash
python scripts/check_setup.py
```

Expected output:

```
μOmni Setup Verification
========================

✓ Python version: 3.10.6
✓ PyTorch version: 2.0.1
✓ CUDA available: True
✓ GPU: NVIDIA GeForce RTX 3060 (12.0 GB)

Checking dependencies:
✓ torch
✓ torchvision
✓ torchaudio
✓ einops
✓ sentencepiece
✓ soundfile
✓ librosa
✓ numpy
✓ PIL

Checking directories:
✓ data/
✓ checkpoints/
✓ configs/

All checks passed! You're ready to use μOmni.
```

---

## 🔍 Troubleshooting

### Issue 1: CUDA Not Available

```
Error: torch.cuda.is_available() returns False

Solutions:
1. Check NVIDIA driver version:
   nvidia-smi
   # Should show driver version (e.g., 525.xx)

2. Reinstall PyTorch with correct CUDA version:
   # Check CUDA version
   nvcc --version
   
   # Install matching PyTorch
   # For CUDA 11.8:
   pip install torch --index-url https://download.pytorch.org/whl/cu118
   
   # For CUDA 12.1:
   pip install torch --index-url https://download.pytorch.org/whl/cu121

3. Verify CUDA paths (Linux):
   export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

---

### Issue 2: Out of Memory

```
Error: CUDA out of memory

Solutions:
1. Reduce batch size in config files:
   "batch_size": 8  →  "batch_size": 4

2. Enable gradient checkpointing:
   "use_gradient_checkpointing": true

3. Use mixed precision training:
   "use_amp": true

4. Close other GPU applications:
   # Check GPU usage
   nvidia-smi
   
   # Kill processes if needed
   kill -9 <PID>
```

---

### Issue 3: Import Errors

```
Error: ModuleNotFoundError: No module named 'omni'

Solutions:
1. Ensure you're in the project root:
   cd /path/to/muOmni
   
2. Check Python path:
   python -c "import sys; print(sys.path)"
   
3. Add project to PYTHONPATH:
   export PYTHONPATH="${PYTHONPATH}:/path/to/muOmni"
```

---

### Issue 4: Sentencepiece Error

```
Error: sentencepiece not found

Solution:
pip uninstall sentencepiece
pip install sentencepiece --no-cache-dir
```

---

### Issue 5: Audio Library Errors

```
Error: soundfile failed to load

Solutions (Linux):
sudo apt-get install libsndfile1

Solutions (Mac):
brew install libsndfile

Solutions (Windows):
# Usually works out of the box, if not:
pip install --upgrade soundfile
```

---

## 🐳 Docker Setup (Alternative)

### Using Docker

```dockerfile
# Dockerfile
FROM pytorch/pytorch:2.0.1-cuda11.8-cudnn8-runtime

WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libsndfile1 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

# Set environment
ENV PYTHONPATH=/workspace

CMD ["/bin/bash"]
```

```bash
# Build Docker image
docker build -t muomni:latest .

# Run container
docker run --gpus all -it -v $(pwd)/data:/workspace/data muomni:latest
```

---

## 📝 Environment Variables

### Optional Configuration

```bash
# .env file
MUOMNI_DATA_DIR=./data
MUOMNI_CHECKPOINT_DIR=./checkpoints
MUOMNI_CACHE_DIR=./cache
CUDA_VISIBLE_DEVICES=0  # Use GPU 0
OMP_NUM_THREADS=8       # CPU threads
```

```python
# Load environment variables
from dotenv import load_dotenv
import os

load_dotenv()
data_dir = os.getenv('MUOMNI_DATA_DIR', './data')
```

---

## 💡 Key Takeaways

✅ **12GB+ GPU** required for training  
✅ **Python 3.8-3.11** supported  
✅ **PyTorch 2.0+** recommended (Flash Attention)  
✅ **Virtual environment** keeps dependencies isolated  
✅ **Verify GPU** setup before training  
✅ **Check setup script** confirms everything works

---

## 🎓 Self-Check Questions

1. What's the minimum GPU memory required?
2. Why use a virtual environment?
3. How do you verify CUDA is available?
4. What PyTorch version is recommended?
5. What should you do if you get "CUDA out of memory"?

<details>
<summary>📝 Answers</summary>

1. 12GB VRAM (e.g., RTX 3060)
2. To isolate dependencies and avoid conflicts with other projects
3. Run: `python -c "import torch; print(torch.cuda.is_available())"`
4. PyTorch 2.0+ (for Flash Attention support)
5. Reduce batch size, enable gradient checkpointing, use mixed precision, or close other GPU applications
</details>

---

[Continue to Chapter 39: Running Training Scripts →](39-running-training.md)

**Chapter Progress:** Practical Usage ●○○○○ (1/5 complete)

