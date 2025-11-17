# Chapter 29: Stage C - Vision Encoder Training

[← Previous: Stage B Audio](28-stage-b-audio-encoder.md) | [Back to Index](00-INDEX.md) | [Next: Stage D Talker →](30-stage-d-talker.md)

---

## 🎯 Learning Objectives

- What Stage C trains and why
- Image classification training
- ViT architecture training specifics
- Configuration and metrics
- Expected progress

---

## 💡 Stage C: Teaching Vision Understanding

**Purpose:** Train Vision Encoder to understand images through classification, enabling meaningful visual embeddings for multimodal fusion in Stage E.

**Task:** Image → Category label (forces learning of visual features, objects, spatial relationships)

---

## 📝 Training Details

### Configuration

```json
{
  "img_size": 224, "patch": 16,  // 14×14 = 196 patches
  "d_model": 128, "n_layers": 4,  // Compact ViT
  "n_heads": 2, "d_ff": 512,
  
  "data_path": "data/images/",
  "batch_size": 32,  // Images = less memory than audio
  "num_epochs": 15,
  "learning_rate": 3e-4
}
```

### Expected Progress

```
Random init → ~5% accuracy (guessing)
After 5 epochs → ~60% accuracy (learning features)
After 15 epochs → ~75-85% accuracy (good understanding!)
```

### Metrics

- **Loss:** Cross-entropy (standard classification)
- **Accuracy:** % correct predictions
- **Target:** >70% accuracy for small models

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
