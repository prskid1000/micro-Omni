# Chapter 39: Running Training - Complete Walkthrough

[← Previous: Setup Environment](38-setup-environment.md) | [Back to Index](00-INDEX.md) | [Next: Inference Examples →](40-inference-examples.md)

---

## 🚀 Complete Training Pipeline

Step-by-step guide to training μOmni from scratch.

---

## 📝 Prerequisites

✅ Environment setup complete (Chapter 38)  
✅ Data prepared (Chapter 35)  
✅ GPU available (12GB+)

---

## 🎯 Training Sequence

### Stage A: Thinker (Text-Only)

```bash
# Train language model
python train_text.py --config configs/thinker_tiny.json

# Expected time: 8-12 hours
# Expected result: Perplexity < 10
# Output: checkpoints/thinker_tiny/thinker_best.pt
```

**Monitor:**
```bash
# Watch GPU
watch -n 1 nvidia-smi

# Check logs
tail -f checkpoints/thinker_tiny/training.log
```

### Stage B: Audio Encoder

```bash
python train_audio_enc.py --config configs/audio_enc_tiny.json

# Time: 6-10 hours
# Target: WER < 20%
# Output: checkpoints/audio_enc_tiny/audio_enc.pt
```

### Stage C: Vision Encoder

```bash
python train_vision.py --config configs/vision_tiny.json

# Time: 4-8 hours
# Target: Accuracy > 70%
# Output: checkpoints/vision_tiny/vision.pt
```

### Stage D: Talker + RVQ

```bash
python train_talker.py --config configs/talker_tiny.json

# Time: 10-15 hours
# Target: Intelligible speech
# Output: checkpoints/talker_tiny/talker.pt + rvq_codec.pt
```

### Stage E: Multimodal SFT

```bash
python sft_omni.py --config configs/omni_sft_tiny.json \
  --thinker checkpoints/thinker_tiny/thinker_best.pt \
  --audio_encoder checkpoints/audio_enc_tiny/audio_enc.pt \
  --vision_encoder checkpoints/vision_tiny/vision.pt \
  --talker checkpoints/talker_tiny/talker.pt

# Time: 6-12 hours
# Target: Good multimodal Q&A
# Output: checkpoints/omni_sft_tiny/omni_final.pt
```

---

## 📊 Monitoring Training

### Key Metrics to Watch

**Stage A (Thinker):**
- Loss decreasing steadily
- Perplexity < 10 (target)
- No NaN/Inf values

**Stage B (Audio):**
- CTC loss decreasing
- WER improving (lower is better)
- Target: WER < 20%

**Stage C (Vision):**
- Accuracy increasing
- Loss decreasing
- Target: Accuracy > 70%

**Stage D (Talker):**
- Reconstruction error < 0.05
- Speech codes perplexity < 15
- Generated speech intelligible

**Stage E (SFT):**
- All modalities improving
- Cross-modal accuracy increasing
- Target: >60% on mixed tasks

---

## 🛠️ Resuming Training

```bash
# Training interrupted? Resume from checkpoint
python train_text.py --config configs/thinker_tiny.json \
  --resume checkpoints/thinker_tiny/thinker_step_1000.pt
```

---

## 💡 Tips

✅ **Run stages in parallel** (if multiple GPUs)  
✅ **Start with small data** to verify pipeline  
✅ **Monitor GPU memory** with `nvidia-smi`  
✅ **Save checkpoints frequently** (every 500-1000 steps)  
✅ **Use tmux/screen** for long training sessions

---

[Continue to Chapter 40: Inference Examples →](40-inference-examples.md)

---
