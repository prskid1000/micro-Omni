# Chapter 29: Stage C - Vision Encoder Training

[← Previous: Stage B Audio](28-stage-b-audio-encoder.md) | [Back to Index](00-INDEX.md) | [Next: Stage D Talker →](30-stage-d-talker.md)

---

## 🎯 Learning Objectives

- What Stage C trains and why
- Vision-language contrastive learning
- ViT architecture training specifics
- Configuration and metrics
- Expected progress

---

## 💡 Stage C: Teaching Vision Understanding

**Purpose:** Train Vision Encoder to understand images through contrastive learning (CLIP-style), enabling meaningful visual embeddings aligned with text for multimodal fusion in Stage E.

**Task:** Image-Caption contrastive learning (forces learning of visual features aligned with text descriptions)

---

## 📝 Training Details

### Configuration

```json
{
  "img_size": 224, "patch": 16,  // 14×14 = 196 patches
  "d_model": 128, "n_layers": 4,  // Compact ViT
  "n_heads": 2, "d_ff": 512,
  
  "train_manifest": "data/images/annotations.json",
  "image_root": "data/images",
  "use_thinker_for_text": true,  // Use Thinker model (true) or simple embedding (false)
  "thinker_ckpt": "checkpoints/thinker_tiny",  // Uses tokenizer from Stage A
  "ctx_len": 512,  // Context length for text encoding
  "vocab_size": 32000,  // Tokenizer vocabulary size
  "embed_dim": 128,  // Contrastive embedding dimension
  "temperature": 0.07,  // Contrastive loss temperature
  
  "batch_size": 8,
  "max_epochs": 3,
  "learning_rate": 3e-4
}
```

**Key Configuration Notes:**
- **`use_thinker_for_text`**: Whether to use Thinker model for text encoding
  - **`true` (recommended)**: Uses frozen Thinker model - better contextual embeddings, aligned with Stage E
  - **`false`**: Uses simple tokenizer + embedding - lighter, faster, but less contextual
- **`thinker_ckpt`**: Directory containing the trained tokenizer from Stage A (`tokenizer.model`) and optionally trained Thinker (`thinker.pt`)
- **`ctx_len`**: Context length for text encoding (matches Thinker's context length)
- **`vocab_size`**: Vocabulary size (automatically detected from tokenizer if available)
- If tokenizer not found, it will be trained from image captions

### Expected Progress

```
Random init → High contrastive loss (random alignment)
After 1 epoch → Loss decreasing (learning image-text alignment)
After 3 epochs → Good vision-language alignment (ready for Stage E)
```

### Metrics

- **Loss:** Contrastive loss (InfoNCE) - measures image-text alignment
- **Validation Loss:** Average contrastive loss on validation set
- **Target:** Low contrastive loss indicates good vision-language alignment

---

## 🎓 Output

```
checkpoints/vision_encoder_tiny/
├── vision_step_1000.pt   # Periodic checkpoints (every 1000 steps)
└── vision_step_2000.pt
```

Used in Stage E for multimodal image understanding!

**Note:** The same Vision Encoder architecture (ViT) is also used in the optional OCR model (`train_ocr.py`) for text extraction from images. OCR uses a similar ViT encoder but with a text decoder for sequence-to-sequence text generation.

---

[Continue to Chapter 30: Stage D - Talker →](30-stage-d-talker.md)

**Chapter Progress:** Training Pipeline ●●●●○○ (4/6 complete)

---
