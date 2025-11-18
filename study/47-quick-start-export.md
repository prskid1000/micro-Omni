# Chapter 47: Quick Start Export Guide

[← Previous: Model Export and Deployment](46-model-export-deployment.md) | [Back to Index](00-INDEX.md) | [End of Guide](#)

---

## 🚀 Quick Start: Exporting μOmni Model

A concise guide to exporting your trained μOmni model for deployment.

---

## Step 1: Merge All Components

```bash
python export.py \
    --omni_ckpt checkpoints/omni_sft_tiny \
    --thinker_ckpt checkpoints/thinker_tiny \
    --audio_ckpt checkpoints/audio_enc_tiny \
    --vision_ckpt checkpoints/vision_tiny \
    --talker_ckpt checkpoints/talker_tiny \
    --output_dir merged_model
```

This creates:

- `merged_model/model.safetensors` - All model weights
- `merged_model/configs/` - All config files
- `merged_model/tokenizer.model` - Tokenizer
- `merged_model/model_info.json` - Metadata

---

## Step 2: Test Inference

```bash
cd merged_model
pip install transformers safetensors sentencepiece torch
python infer_standalone.py --text "Hello, how are you?"
```

---

## Step 3: Upload to Hugging Face (Optional)

```bash
pip install huggingface_hub
huggingface-cli login
huggingface-cli upload your-username/muomni-tiny merged_model/ --repo-type model
```

---

## 📦 What Gets Merged?

All components are merged with prefixed keys:

- `thinker.*` - LLM weights
- `audio_encoder.*` - Audio encoder
- `vision_encoder.*` - Vision encoder  
- `talker.*` - TTS model
- `rvq.*` - Audio codec
- `proj_a.*` / `proj_v.*` - Multimodal projectors
- `ocr.*` - OCR model (if included)

---

## ✅ Support Files Required

**Required:**
- Config JSON files (in `configs/`)
- `tokenizer.model`

**Optional:**
- `hifigan.pt` (vocoder)
- `ocr.pt` (for OCR char mappings)

---

## 💡 Quick Tips

✅ **All components optional** - Script handles missing components gracefully  
✅ **Automatic copying** - Support files copied automatically  
✅ **Standalone inference** - `infer_standalone.py` uses transformers library (no codebase needed)  
✅ **Production ready** - Safetensors format is secure and efficient

---

## 🔗 For More Details

See [Chapter 46: Model Export and Deployment](46-model-export-deployment.md) for:
- Detailed explanation of the merge process
- Troubleshooting guide
- Hugging Face upload instructions
- File size considerations
- Verification methods

---

## 📚 Related Chapters

- [Chapter 32: Inference Pipeline](32-inference-pipeline.md) - Understanding inference
- [Chapter 40: Inference Examples](40-inference-examples.md) - Original inference examples
- [Chapter 39: Running Training](39-running-training.md) - Training before export

---

[Back to Index](00-INDEX.md)

---

