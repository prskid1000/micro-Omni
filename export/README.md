# μOmni Model Export Output

**This folder is for the OUTPUT of model export - standalone safetensors files ready for upload and inference.**

## Purpose

After running the export scripts (located in the root directory), place the exported model here. This folder contains everything needed to:
- Upload to Hugging Face Hub
- Run inference without the full codebase
- Share the model with others

## Export Scripts (in root directory)

- **`export.py`** - Merges all model components into a single safetensors file

## Standalone Inference Script

**`infer_standalone.py`** - Standalone model loading using Hugging Face transformers library (included in this folder)

> **Note**: This script demonstrates model loading but requires the full codebase for actual generation. For full multimodal inference, use `infer_chat.py` from the root directory.

This script can load the model using transformers library and safetensors. It demonstrates the structure for integration with transformers, but full text generation requires the model architecture classes from the `omni` package.

## Quick Start

### 1. Export Model (run from root)

```bash
python export.py \
    --omni_ckpt checkpoints/omni_sft_tiny \
    --thinker_ckpt checkpoints/thinker_tiny \
    --audio_ckpt checkpoints/audio_enc_tiny \
    --vision_ckpt checkpoints/vision_tiny \
    --talker_ckpt checkpoints/talker_tiny \
    --output_dir export
```

### 2. Run Inference

**Option A: Full Inference (recommended - uses full codebase)**

```bash
# From root directory
python infer_chat.py --ckpt_dir export --text "Hello, how are you?"
```

**Option B: Standalone Loading (demonstration only)**

```bash
cd export
pip install transformers safetensors sentencepiece torch
python infer_standalone.py --text "Hello, how are you?"
```

> **Note**: `infer_standalone.py` demonstrates model loading but requires the full codebase (`omni` package) for actual text generation. For production use, use `infer_chat.py` from the root directory.

## Documentation

For detailed documentation, see:
- [Chapter 46: Model Export and Deployment](../study/46-model-export-deployment.md)
- [Chapter 47: Quick Start Export](../study/47-quick-start-export.md)

## What Gets Created

The export script creates a complete model package with:

### Required Files
- `model.safetensors` - All model weights in one file (thinker, audio_encoder, vision_encoder, talker, rvq, projectors, optional OCR)
- `config.json` - Main Hugging Face config (vocab_size=32000, d_model=256, n_layers=4, etc.)
- `tokenizer_config.json` - Tokenizer configuration for SentencePiece
- `generation_config.json` - Generation parameters (BOS/EOS tokens, sampling settings)
- `chat_template.json` - Chat template for conversations
- `preprocessor_config.json` - Multimodal preprocessor settings (image size, audio params)
- `tokenizer.model` - SentencePiece tokenizer model file
- `model_info.json` - Custom metadata about included components

### Optional Files
- `configs/` - Component-specific config JSON files (for backward compatibility)
- `hifigan.pt` - HiFi-GAN neural vocoder checkpoint (if available)
- `ocr.pt` - OCR model checkpoint with character mappings (if provided)
- `infer_standalone.py` - Standalone model loading script (demonstration)

## Hugging Face Upload

The exported model is ready for Hugging Face upload:

```bash
pip install huggingface_hub
huggingface-cli login
huggingface-cli upload your-username/muomni-tiny export/ --repo-type model
```

## Using the Exported Model

### With Full Codebase (Recommended)

For full multimodal inference (text, image, audio, video):

```bash
# From root directory
python infer_chat.py --ckpt_dir export --text "Hello" --image path/to/image.jpg
python infer_chat.py --ckpt_dir export --audio_in path/to/audio.wav
python infer_chat.py --ckpt_dir export --text "Generate speech" --audio_out output.wav
```

### Model Configuration

The exported model uses the "tiny" configuration:
- **Thinker**: vocab_size=32000, d_model=256, n_layers=4, n_heads=4, use_gqa=True
- **Audio Encoder**: d_model=192, n_layers=4, n_heads=3
- **Vision Encoder**: d_model=128, n_layers=4, n_heads=2
- **Talker**: d_model=192, n_layers=4, n_heads=3

See `configs/*_tiny.json` for full component configurations.

