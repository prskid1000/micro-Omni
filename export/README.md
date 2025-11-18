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

**`infer_standalone.py`** - Standalone inference using Hugging Face transformers library (included in this folder)

This script can be used with just the exported model folder and transformers library. It uses transformers for tokenization and safetensors for model loading. No codebase required!

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

### 2. Run Inference (standalone - no codebase needed!)

```bash
cd export
pip install transformers safetensors sentencepiece torch
python infer_standalone.py --text "Hello, how are you?"
```

Or from anywhere:

```bash
python export/infer_standalone.py --model_dir export --text "Hello, how are you?"
```

## Documentation

For detailed documentation, see:
- [Chapter 46: Model Export and Deployment](../study/46-model-export-deployment.md)
- [Chapter 47: Quick Start Export](../study/47-quick-start-export.md)

## What Gets Created

The export script creates a complete model package with:
- `model.safetensors` - All model weights in one file
- `config.json` - Main Hugging Face config
- `tokenizer_config.json` - Tokenizer configuration
- `generation_config.json` - Generation parameters
- `chat_template.json` - Chat template
- `preprocessor_config.json` - Multimodal preprocessor settings
- `tokenizer.model` - SentencePiece tokenizer
- `configs/` - Component-specific configs
- Optional: `hifigan.pt`, `ocr.pt`
- `infer_standalone.py` - Standalone inference script (uses transformers library)

## Hugging Face Upload

The exported model is ready for Hugging Face upload:

```bash
pip install huggingface_hub
huggingface-cli login
huggingface-cli upload your-username/muomni-tiny merged_model/ --repo-type model
```

