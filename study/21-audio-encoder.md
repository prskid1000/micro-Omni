# Chapter 21: Audio Encoder (AuT-Tiny)

[Back to Index](00-INDEX.md)

---

## 🎯 Purpose

Convert mel spectrograms to semantic embeddings for the Thinker.

## 🏗️ Architecture

```
Mel Spectrogram (B, T, 128)
    ↓
Conv2D Downsampling (8x)
  → Reduces T by 8x (100Hz → 12.5Hz)
    ↓
Flatten & Project
  → (B, T/8, 192)
    ↓
Transformer Encoder (4 layers)
  → Self-attention + FFN
    ↓
RMSNorm
    ↓
Output: (B, T/8, 192)
    ↓
Audio Projector: Linear(192 → 256)
    ↓
Ready for Thinker: (B, T/8, 256)
```

## 📊 Specifications

| Parameter | Value |
|-----------|-------|
| **Input** | Mel spectrogram (T, 128) |
| **Downsample** | 8x (100Hz → 12.5Hz) |
| **Dimension** | 192 |
| **Layers** | 4 |
| **Heads** | 3 |
| **Parameters** | ~10-15M |

## 🎓 Training

**Task**: ASR (Automatic Speech Recognition)  
**Loss**: CTC (Connectionist Temporal Classification)  
**Data**: Audio + transcriptions

## 💡 Key Takeaways

✅ **Processes mel spectrograms**  
✅ **8x temporal downsampling** (100Hz → 12.5Hz)  
✅ **Outputs 192-dim embeddings**  
✅ **Trained with CTC loss** on ASR task

---

[Back to Index](00-INDEX.md)

