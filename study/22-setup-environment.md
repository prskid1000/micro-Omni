[← Previous: 21-debugging](21-debugging.md) | [Index](00-INDEX.md) | [Next: 23-inference-chat →](23-inference-chat.md)

# Chapter 22: Setup & Environment

Getting your machine ready to train and run micro-Omni.

---

## Hardware Requirements

```
+---------------------------------------------------+
|  MINIMUM HARDWARE                                  |
+---------------------------------------------------+
|  GPU    : NVIDIA with 12 GB+ VRAM (RTX 3060+)     |
|  CPU    : 4+ cores, modern x86_64                  |
|  RAM    : 16 GB system memory                      |
|  Disk   : 50 GB free (datasets + checkpoints)      |
|  CUDA   : Compute Capability 7.0+                  |
+---------------------------------------------------+
|  RECOMMENDED                                       |
+---------------------------------------------------+
|  GPU    : RTX 3090 / 4090 (24 GB VRAM)             |
|  RAM    : 32 GB                                    |
|  Disk   : SSD, 100 GB+                             |
+---------------------------------------------------+
```

Micro-Omni's tiny architecture (25M-150M params) is designed to train on a
single consumer GPU. Multi-GPU setups are not required but will speed up
larger configurations.

---

## Software Prerequisites

| Software      | Minimum Version | Notes                              |
|---------------|----------------|------------------------------------|
| Python        | 3.8+           | 3.10 or 3.11 recommended          |
| CUDA Toolkit  | 11.8+          | Must match PyTorch CUDA version    |
| cuDNN         | 8.6+           | Bundled with most CUDA installs    |
| PyTorch       | 2.0+           | 2.1+ recommended for torch.compile |
| Git           | any            | For cloning the repository         |

---

## Installation

### Step 1: Create a Virtual Environment

```bash
python -m venv venv
source venv/Scripts/activate    # Windows Git Bash
# source venv/bin/activate      # Linux / macOS
```

### Step 2: Install PyTorch with CUDA

Visit pytorch.org for the exact command matching your CUDA version.
For CUDA 11.8:

```bash
pip install torch torchaudio torchvision --index-url https://download.pytorch.org/whl/cu118
```

For CUDA 12.1:

```bash
pip install torch torchaudio torchvision --index-url https://download.pytorch.org/whl/cu121
```

### Step 3: Install Dependencies

```bash
pip install sentencepiece einops librosa safetensors Pillow
```

Full one-liner:

```bash
pip install torch torchaudio torchvision sentencepiece einops librosa safetensors Pillow
```

### Step 4: Optional but Useful

```bash
pip install tensorboard matplotlib tqdm soundfile
```

---

## Directory Structure

Create the working directories the training scripts expect:

```bash
mkdir -p data/text data/audio data/images data/ocr
mkdir -p checkpoints
mkdir -p configs
mkdir -p exported
```

Result:

```
micro-Omni/
├── omni/                  # Model source code
│   ├── thinker.py
│   ├── audio_encoder.py
│   ├── vision_encoder.py
│   ├── codec.py
│   ├── talker.py
│   ├── ocr_model.py
│   ├── tokenizer.py
│   └── utils.py
├── configs/               # JSON config files
│   ├── synthetic_thinker.json
│   ├── synthetic_audio_enc.json
│   ├── synthetic_vision.json
│   └── ...
├── data/                  # Training data
│   ├── text/
│   ├── audio/
│   ├── images/
│   └── ocr/
├── checkpoints/           # Saved during training
├── exported/              # Merged models for deployment
├── train_text.py
├── train_audio_enc.py
├── train_vision.py
├── train_talker.py
├── train_vocoder.py
├── train_ocr.py
├── sft_omni.py
├── infer_chat.py
└── export.py
```

---

## Verification

Run this one-liner to confirm everything works:

```bash
python -c "
import torch
print('PyTorch version:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('CUDA version:', torch.version.cuda)
    print('GPU:', torch.cuda.get_device_name(0))
    print('VRAM:', round(torch.cuda.get_device_properties(0).total_mem / 1e9, 1), 'GB')
"
```

Expected output (example):

```
PyTorch version: 2.1.0+cu118
CUDA available: True
CUDA version: 11.8
GPU: NVIDIA GeForce RTX 3090
VRAM: 24.3 GB
```

Verify each dependency:

```bash
python -c "import sentencepiece; print('sentencepiece OK')"
python -c "import einops; print('einops OK')"
python -c "import librosa; print('librosa OK')"
python -c "import safetensors; print('safetensors OK')"
python -c "from PIL import Image; print('Pillow OK')"
python -c "import torchaudio; print('torchaudio OK')"
python -c "import torchvision; print('torchvision OK')"
```

---

## Windows Notes

Micro-Omni is developed on Windows. Keep these conventions in mind:

1. **Use forward slashes** in all paths within scripts and configs:
   ```python
   # Good
   path = "data/audio/train.wav"
   # Bad
   path = "data\\audio\\train.wav"
   ```

2. **Use Git Bash or WSL** for shell commands. The training scripts assume
   bash-style syntax (`source`, forward slashes, `/dev/null`).

3. **Long paths**: Enable long path support if you hit path-length errors:
   ```
   Windows Settings > Developer Settings > Enable long paths
   ```
   Or via registry: `HKLM\SYSTEM\CurrentControlSet\Control\FileSystem\LongPathsEnabled = 1`

4. **Memory-mapped files**: PyTorch DataLoaders with `num_workers > 0` can
   cause issues on Windows. If you see `BrokenPipeError`, set
   `num_workers: 0` in your config or use `persistent_workers: true`.

5. **Mixed precision**: AMP (`use_amp: true`) works well on Windows with
   NVIDIA GPUs. No special setup needed beyond CUDA.

---

## Troubleshooting

| Problem                          | Solution                                      |
|----------------------------------|-----------------------------------------------|
| `CUDA out of memory`            | Reduce `batch_size` or enable `use_amp`       |
| `torch.cuda.is_available()` = False | Reinstall PyTorch with correct CUDA version |
| `No module named 'sentencepiece'` | `pip install sentencepiece`                  |
| Slow data loading                | Set `num_workers: 2-4` (Linux) or `0` (Windows) |
| `RuntimeError: CUDA error`      | Update GPU drivers to latest version          |

---

## Next Steps

With your environment verified, proceed to:
- **Chapter 23**: Run inference and chat with the model
- **Chapter 2-21**: Train individual components from scratch
