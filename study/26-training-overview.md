# Chapter 26: Training Workflow Overview

[Back to Index](00-INDEX.md)

---

## 🎯 5-Stage Training Pipeline

```
┌────────────────────────────────────┐
│ Stage A: Thinker Pretraining      │
│ Task: Next-token prediction        │
│ Data: Text corpus                  │
│ Time: ~8-12 hours (12GB GPU)       │
└────────────┬───────────────────────┘
             ↓
┌────────────────────────────────────┐
│ Stage B: Audio Encoder (ASR)      │
│ Task: Speech-to-text (CTC loss)   │
│ Data: Audio + transcriptions       │
│ Time: ~6-10 hours                  │
└────────────┬───────────────────────┘
             ↓
┌────────────────────────────────────┐
│ Stage C: Vision Encoder            │
│ Task: Image classification         │
│ Data: Images + captions            │
│ Time: ~4-8 hours                   │
└────────────┬───────────────────────┘
             ↓
┌────────────────────────────────────┐
│ Stage D: Talker + RVQ Codec        │
│ Task: Speech code prediction       │
│ Data: Audio for TTS                │
│ Time: ~10-15 hours                 │
└────────────┬───────────────────────┘
             ↓
┌────────────────────────────────────┐
│ Stage E: Multimodal SFT            │
│ Task: Joint multimodal tuning      │
│ Data: Mixed (text+image+audio)     │
│ Time: ~6-12 hours                  │
└────────────────────────────────────┘
```

## 📊 Training Summary

| Stage | Model | Task | Loss Function | Key Metric |
|-------|-------|------|---------------|------------|
| **A** | Thinker | Language Modeling | Cross-Entropy | Perplexity |
| **B** | Audio Encoder | ASR | CTC | WER |
| **C** | Vision Encoder | Image Understanding | Cross-Entropy | Accuracy |
| **D** | Talker + RVQ | Speech Generation | Cross-Entropy + MSE | Reconstruction |
| **E** | All (Joint) | Multimodal | Cross-Entropy | Mixed Accuracy |

## 🎯 Training Strategy

### Modularity
- Each stage trains independently
- Debug issues in isolation
- Parallel development possible

### Efficiency
- Small datasets (<5GB per modality)
- Fits 12GB GPU with gradient accumulation
- Uses mixed precision (FP16)
- Gradient checkpointing for memory

### Progressive Learning
- Start with individual modalities
- End with joint understanding
- Specialized encoders preserved

## 💻 Quick Start

```bash
# Stage A
python train_text.py --config configs/thinker_tiny.json

# Stage B
python train_audio_enc.py --config configs/audio_enc_tiny.json

# Stage C
python train_vision.py --config configs/vision_tiny.json

# Stage D
python train_talker.py --config configs/talker_tiny.json

# Stage E
python sft_omni.py --config configs/omni_sft_tiny.json
```

## 💡 Key Takeaways

✅ **5 independent stages** (modular design)  
✅ **~40-60 hours total** training time (12GB GPU)  
✅ **Small datasets** (<5GB each)  
✅ **Progressive learning** (specialized → joint)

---

[Back to Index](00-INDEX.md)

