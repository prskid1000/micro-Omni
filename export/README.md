# μOmni Model Export Tools

This folder contains scripts for exporting trained μOmni models to safetensors format for deployment.

## Files

- **`merge_to_safetensors.py`** - Merges all model components into a single safetensors file
- **`infer_safetensors.py`** - Inference script that loads from safetensors format

## Quick Start

### 1. Export Model

```bash
python export/merge_to_safetensors.py \
    --omni_ckpt checkpoints/omni_sft_tiny \
    --thinker_ckpt checkpoints/thinker_tiny \
    --audio_ckpt checkpoints/audio_enc_tiny \
    --vision_ckpt checkpoints/vision_tiny \
    --talker_ckpt checkpoints/talker_tiny \
    --output_dir merged_model
```

### 2. Run Inference

```bash
python export/infer_safetensors.py \
    --model_dir merged_model \
    --text "Hello, how are you?"
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

## Hugging Face Upload

The exported model is ready for Hugging Face upload:

```bash
pip install huggingface_hub
huggingface-cli login
huggingface-cli upload your-username/muomni-tiny merged_model/ --repo-type model
```

