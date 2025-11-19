# Chapter 46: Model Export and Deployment

[← Previous: Future Extensions](45-future-extensions.md) | [Back to Index](00-INDEX.md) | [Next: Quick Start Export →](47-quick-start-export.md)

---

## 🎯 Exporting μOmni for Deployment

This chapter explains how to merge all μOmni model components into a single safetensors file and what support files are needed for deployment.

---

## 📦 Overview

The μOmni model consists of multiple components that are trained separately:

- **Thinker**: The main LLM for text generation
- **Audio Encoder**: Processes audio input
- **Vision Encoder**: Processes image/video input
- **Talker**: Text-to-speech generation model
- **RVQ**: Residual Vector Quantization codec for audio
- **Projectors**: Linear layers that map audio/vision embeddings to Thinker's dimension
- **OCR** (optional): Optical Character Recognition model

For deployment, all these components are merged into a single safetensors file with prefixed keys to avoid naming conflicts.

---

## 🔧 Merging Model Components

### Prerequisites

Install the required dependency:

```bash
pip install safetensors
```

### Usage

Use the `export.py` script (in root directory) to combine all model components:

```bash
python export.py \
    --omni_ckpt checkpoints/omni_sft_tiny \
    --thinker_ckpt checkpoints/thinker_tiny \
    --audio_ckpt checkpoints/audio_enc_tiny \
    --vision_ckpt checkpoints/vision_tiny \
    --talker_ckpt checkpoints/talker_tiny \
    --ocr_ckpt checkpoints/ocr_tiny \
    --output_dir merged_model \
    --output_file model.safetensors
```

### Arguments

- `--omni_ckpt`: Directory containing `omni.pt` (contains projectors and optionally thinker)
- `--thinker_ckpt`: Directory containing `thinker.pt` (if not in omni checkpoint)
- `--audio_ckpt`: Directory containing `audio_enc.pt`
- `--vision_ckpt`: Directory containing `vision.pt`
- `--talker_ckpt`: Directory containing `talker.pt` (contains both talker and rvq)
- `--ocr_ckpt`: (Optional) Directory containing `ocr.pt`
- `--output_dir`: Output directory for merged model and support files
- `--output_file`: Name of the safetensors file (default: `model.safetensors`)
- `--configs_dir`: Directory containing config JSON files (default: `configs`)
- `--skip_component_configs`: Skip copying individual component configs (minimal export, only main config.json)

**Note:** The export script automatically uses step checkpoints if standard checkpoints are not found. For example, if `thinker.pt` doesn't exist, it will automatically find and use the latest `thinker_step_*.pt` checkpoint.

### What Gets Merged

The script collects weights from all components and prefixes them:

- `thinker.*` - All Thinker LLM weights
- `audio_encoder.*` - Audio encoder weights
- `vision_encoder.*` - Vision encoder weights
- `talker.*` - Talker TTS model weights
- `rvq.*` - RVQ codec weights
- `proj_a.*` - Audio projector weights
- `proj_v.*` - Vision projector weights
- `ocr.*` - OCR model weights (if provided)

---

## 📁 Support Files

The merge script automatically copies all necessary support files to the output directory:

### Required Files

1. **Main Configuration** (Hugging Face format):
   - `config.json` - Main model configuration (at root, not in subdirectory)
   - `tokenizer_config.json` - Tokenizer configuration
   - `generation_config.json` - Generation parameters
   - `chat_template.json` - Chat template for conversations
   - `preprocessor_config.json` - Multimodal preprocessor settings

2. **Component Config JSON files** (`configs/` subdirectory) - **OPTIONAL**:
   - `thinker_tiny.json` - Thinker model configuration (optional, for backward compatibility)
   - `audio_enc_tiny.json` - Audio encoder configuration (optional)
   - `vision_tiny.json` - Vision encoder configuration (optional)
   - `talker_tiny.json` - Talker model configuration (optional)
   - `omni_sft_tiny.json` - Overall training configuration (optional)
   - `ocr_tiny.json` - OCR model configuration (optional)
   - `vocoder_tiny.json` - Vocoder configuration (optional)
   
   **Note:** These are optional. The main `config.json` at root contains all necessary information for Hugging Face. Component configs are only included for backward compatibility with our inference script.

3. **Tokenizer**:
   - `tokenizer.model` - SentencePiece BPE tokenizer model file
   - **Note:** We use SentencePiece, so we don't need `vocab.json` or `merges.txt` (those are for GPT-2 style BPE tokenizers)

4. **Model Info**:
   - `model_info.json` - Custom metadata about included components

### Optional Files

- `hifigan.pt` - HiFi-GAN neural vocoder checkpoint (for better TTS quality)
- `ocr.pt` - OCR checkpoint with char mappings (for OCR functionality)

### Directory Structure

After merging, your output directory will look like:

```
merged_model/
├── model.safetensors          # All model weights
├── config.json                # Main HF config (at root) - REQUIRED
├── tokenizer_config.json      # Tokenizer config (SentencePiece) - REQUIRED
├── generation_config.json      # Generation settings - REQUIRED
├── chat_template.json          # Chat template - REQUIRED
├── preprocessor_config.json   # Multimodal preprocessor - REQUIRED
├── model_info.json            # Component metadata (optional)
├── tokenizer.model            # SentencePiece tokenizer (single file) - REQUIRED
├── hifigan.pt                 # Vocoder (optional)
├── ocr.pt                     # OCR checkpoint (optional)
└── configs/                   # Component-specific configs (OPTIONAL)
    ├── thinker_tiny.json      # Optional - for backward compatibility
    ├── audio_enc_tiny.json    # Optional
    ├── vision_tiny.json       # Optional
    ├── talker_tiny.json       # Optional
    ├── omni_sft_tiny.json    # Optional
    ├── ocr_tiny.json         # Optional
    └── vocoder_tiny.json     # Optional
```

**Important Notes:**

1. **Tokenizer files:** We use SentencePiece, which only requires `tokenizer.model`. We do NOT need:
   - ❌ `vocab.json` (used by GPT-2 style BPE tokenizers)
   - ❌ `merges.txt` (used by GPT-2 style BPE tokenizers)

2. **Component configs:** The `configs/` subdirectory is **OPTIONAL**. For Hugging Face upload, only the main `config.json` at root is required. Component configs are included for backward compatibility with our inference script, but the inference script will use `config.json` if available.

---

## 🚀 Inference with Safetensors

Use the standalone `infer_standalone.py` script (included in export folder) to run inference:

```bash
cd merged_model
pip install transformers safetensors sentencepiece torch
python infer_standalone.py --text "Hello, how are you?"
```

Or use Hugging Face transformers library directly:

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from safetensors.torch import load_file

tokenizer = AutoTokenizer.from_pretrained("merged_model")
# Load model weights from safetensors
state_dict = load_file("merged_model/model.safetensors")
```

### Arguments

- `--model_dir`: Directory containing `model.safetensors` and support files
- `--image`: Path to input image file
- `--video`: Path to input video file
- `--audio_in`: Path to input audio file
- `--audio_out`: Path to save generated audio (TTS)
- `--text`: Text prompt
- `--prompt`: Override default prompt
- `--ocr`: Extract text from image using OCR

### Example Usage

**Text-only chat:**
```bash
cd merged_model
python infer_standalone.py --text "Your question here"
```

**Using transformers directly:**
```bash
python -c "from transformers import AutoTokenizer; tokenizer = AutoTokenizer.from_pretrained('merged_model'); print(tokenizer.encode('Hello'))"
```

**Multimodal (image + text):**
```bash
cd merged_model
python infer_standalone.py \
    --image photo.jpg \
    --text "Describe this image"
```

**Text-to-speech:**
```bash
cd merged_model
python infer_standalone.py \
    --text "Hello world" \
    --audio_out hello.wav
```

**Audio input:**
```bash
cd merged_model
python infer_standalone.py \
    --audio_in speech.wav \
    --text "What did I say?"
```

---

## 🌐 Uploading to Hugging Face

To upload your merged model to Hugging Face:

1. **Create a model repository** on Hugging Face Hub

2. **Upload the merged model directory**:
   ```bash
   # Install huggingface_hub if needed
   pip install huggingface_hub
   
   # Login
   huggingface-cli login
   
   # Upload
   huggingface-cli upload your-username/muomni-tiny \
       merged_model/ \
       --repo-type model
   ```

3. **Required files for Hugging Face**:
   - `model.safetensors` - Model weights
   - `config.json` - Main model configuration (at root) - **REQUIRED**
   - `tokenizer_config.json` - Tokenizer configuration - **REQUIRED**
   - `generation_config.json` - Generation parameters - **REQUIRED**
   - `tokenizer.model` - SentencePiece tokenizer model - **REQUIRED**
   - `README.md` - Model card (create this)
   - `LICENSE` - License file (if applicable)
   
   **Optional files:**
   - `configs/*.json` - Component-specific configs (not needed for HF, included for backward compatibility)
   - `preprocessor_config.json` - Multimodal preprocessor (recommended)
   - `chat_template.json` - Chat template (recommended)

4. **Optional but recommended**:
   - `hifigan.pt` - Vocoder checkpoint
   - Example usage code in README

---

## 💾 File Size Considerations

The merged safetensors file will be approximately:

- **Tiny model**: ~50-100 MB (depending on components)
- **Small model**: ~200-500 MB
- **Base model**: ~1-2 GB
- **Large model**: ~5-10 GB+

The safetensors format is more efficient than PyTorch `.pt` files and provides:

- ✅ Faster loading
- ✅ Memory mapping support
- ✅ Better security (no arbitrary code execution)
- ✅ Cross-platform compatibility

---

## 🔍 Troubleshooting

### Missing Components

If a component is missing, the inference script will warn you but continue. Some components are optional:

- **OCR**: Only needed for text extraction from images
- **Vocoder**: Falls back to Griffin-Lim if HiFi-GAN not available
- **Talker**: Only needed for text-to-speech

### Config File Issues

If config files are missing, the script will use default values. However, this may cause issues if your model was trained with different hyperparameters. Always include all config files.

### Tokenizer Not Found

The tokenizer is required. Make sure `tokenizer.model` is copied to the output directory. The script looks for it in:

1. `model_dir/tokenizer.model`
2. `model_dir/../checkpoints/thinker_tiny/tokenizer.model`

---

## ✅ Verification

To verify your merged model is correct:

1. **Check file size**: Should match expected size for your model
2. **Load and test**: Run inference on a simple example
3. **Compare outputs**: Compare outputs from original checkpoints vs. safetensors

```python
# Quick verification script
from safetensors.torch import load_file
import os

model_path = "merged_model/model.safetensors"
if os.path.exists(model_path):
    state_dict = load_file(model_path)
    print(f"Loaded {len(state_dict)} parameter tensors")
    print(f"Components: {set(k.split('.')[0] for k in state_dict.keys())}")
else:
    print("Model file not found!")
```

---

## 💡 Key Takeaways

✅ **Single file deployment** - All weights in one safetensors file  
✅ **Automatic support files** - Configs and tokenizer copied automatically  
✅ **Hugging Face ready** - Direct upload support  
✅ **Standalone inference** - `infer_standalone.py` uses transformers library (no codebase needed)  
✅ **Production ready** - Secure, fast, cross-platform format

---

## 🔗 Related Chapters

- [Chapter 32: Inference Pipeline](32-inference-pipeline.md) - Understanding inference flow
- [Chapter 40: Inference Examples](40-inference-examples.md) - Using the original inference script
- [Chapter 39: Running Training](39-running-training.md) - Training the components before export

---

[Continue to Chapter 47: Quick Start Export →](47-quick-start-export.md)

---

