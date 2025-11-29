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
  "img_size": 224,
  "patch": 16, // 14×14 = 196 patches
  "d_model": 768,  // ViT-Base
  "n_layers": 12,  // ViT-Base layers
  "n_heads": 12,   // ViT-Base heads
  "d_ff": 3072,    // ViT-Base FF dim

  "train_manifest": "data/images/production_annotations.json",
  "image_root": "data/images",
  "use_thinker_for_text": false, // Use Thinker (true) or TransformerTextEncoder (false)
  "thinker_ckpt": "checkpoints/thinker_tiny", // Uses tokenizer from Stage A
  "text_max_len": 77,  // CLIP standard context length
  "text_n_layers": 6,  // Text encoder layers
  "text_n_heads": 8,   // Text encoder heads
  "text_d_ff": 2048,   // Text encoder FF dim
  "vocab_size": 32000, // Tokenizer vocabulary size
  "embed_dim": 512,    // CLIP standard embedding dimension
  "temperature": 0.07, // Learnable temperature (CLIP standard)

  "batch_size": 32,
  "gradient_accumulation_steps": 8,  // Effective batch size: 256
  "lr": 0.0005,      // CLIP learning rate
  "wd": 0.2,         // CLIP weight decay
  "warmup_steps": 2000, // Longer warmup for stability
  "max_steps": 2500000,
  "max_epochs": 36,
  "use_augmentation": true  // Strong data augmentation
}
```

**Key Configuration Notes:**

- **`use_thinker_for_text`**: Whether to use Thinker model for text encoding
  - **`true`**: Uses frozen Thinker model - better contextual embeddings, aligned with Stage E
  - **`false` (recommended)**: Uses TransformerTextEncoder (CLIP-style) - proper Transformer with causal attention and final token pooling
- **`thinker_ckpt`**: Directory containing the trained tokenizer from Stage A (`tokenizer.model`) and optionally trained Thinker (`thinker.pt`)
- **`text_max_len`**: Context length for TransformerTextEncoder (CLIP standard: 77 tokens)
- **`embed_dim`**: Shared embedding dimension for contrastive learning (CLIP standard: 512)
- **`temperature`**: Learnable temperature parameter (adapts during training)
- **CLIP-style training**: MLP projection heads, symmetric loss, proper contrastive learning
- If tokenizer not found, it will be trained from image captions

### Expected Progress

```
Random init → High contrastive loss (random alignment)
After 10k steps → Loss decreasing (learning image-text alignment)
After 100k steps → Good vision-language alignment
After 2.5M steps → Excellent CLIP-style alignment (ready for Stage E)
```

### Metrics

- **Loss:** Symmetric contrastive loss (InfoNCE) - measures image-text alignment
- **Temperature:** Learnable parameter (starts at 0.07, adapts during training)
- **Validation Loss:** Average contrastive loss on validation set
- **Target:** Low contrastive loss indicates good vision-language alignment

**Expected Validation Loss:**

- Target Contrastive Loss: < 2.0 (CLIP-style training)
- Good: < 1.5
- Excellent: < 1.0

---

## 🎓 Output

```
checkpoints/vision_encoder_tiny/
├── model.pt                 # Latest model weights (overwritten)
└── model_metadata.json      # Training state (step, epoch, config)
```

Used in Stage E for multimodal image understanding!

**Note:** The same Vision Encoder architecture (ViT) is also used in the optional OCR model (`train_ocr.py`) for text extraction from images. OCR uses a similar ViT encoder but with a text decoder for sequence-to-sequence text generation.

---

[Continue to Chapter 30: Stage D - Talker →](30-stage-d-talker.md)

**Chapter Progress:** Training Pipeline ●●●●○○ (4/6 complete)

---
