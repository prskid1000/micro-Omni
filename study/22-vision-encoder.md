# Chapter 22: Vision Encoder (ViT-Tiny)

[Back to Index](00-INDEX.md)

---

## 🎯 Purpose

Convert images to semantic embeddings for the Thinker.

## 🏗️ Architecture

```
Image (3, 224, 224)
    ↓
Patch Embedding (16×16 patches)
  → 14×14 = 196 patches
    ↓
Add CLS Token + Positional Embeddings
  → (197, 128)
    ↓
Transformer Encoder (4 layers)
    ↓
Extract CLS Token
  → (1, 128)
    ↓
Vision Projector: Linear(128 → 256)
    ↓
Ready for Thinker: (1, 256)
```

## 📊 Specifications

| Parameter | Value |
|-----------|-------|
| **Input** | Image (224×224×3) |
| **Patch Size** | 16×16 |
| **Patches** | 196 + 1 CLS |
| **Dimension** | 128 |
| **Layers** | 4 |
| **Parameters** | ~15-20M |

## 🎓 Training

**Task**: Image classification/understanding  
**Loss**: Cross-entropy  
**Data**: Images + captions

## 💡 Key Takeaways

✅ **Vision Transformer** (patch-based)  
✅ **196 patch tokens + CLS token**  
✅ **CLS token aggregates** global information  
✅ **Output**: Single 256-dim vector per image

---

[Back to Index](00-INDEX.md)

