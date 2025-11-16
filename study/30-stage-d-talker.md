# Chapter 30: Stage D - Talker & RVQ Codec Training

[← Previous: Stage C Vision](29-stage-c-vision-encoder.md) | [Back to Index](00-INDEX.md) | [Next: Stage E SFT →](31-stage-e-sft.md)

---

## 🎯 Learning Objectives

- What Stage D trains (two-part training)
- RVQ Codec training (part 1)
- Talker training (part 2)
- Configuration and metrics

---

## 💡 Stage D: Teaching Speech Generation

**Two-Part Training:**

1. **Part 1: RVQ Codec** - Learn to discretize mel spectrograms into codes
2. **Part 2: Talker** - Learn to predict those codes autoregressively

**Purpose:** Enable text-to-speech generation

---

## 📝 Training Details

### Part 1: RVQ Codec

**Task:** Mel frame → Discrete codes [base, residual]  
**Loss:** MSE (reconstruction error)  
**Target:** Low reconstruction error (<0.05)

```
Input: Mel frame (128,)
Output: Codes [42, 87]
Reconstructed: Mel frame (128,)
Loss: MSE(reconstructed, original)
```

### Part 2: Talker  

**Task:** Predict next speech codes given previous codes  
**Loss:** Cross-entropy (both base and residual heads)  
**Target:** Perplexity <15, intelligible speech

```
Input: [[0,0], [42,87], [56,91]]
Predict: [67, 103] (next frame)
```

### Configuration

```json
{
  "d_model": 192, "n_layers": 4, "n_heads": 3,
  "codebooks": 2, "codebook_size": 128,
  
  "data_path": "data/audio/tts/",
  "batch_size": 16, "num_epochs": 25,
  "learning_rate": 3e-4
}
```

---

## 📈 Expected Progress

```
RVQ Codec:
Epoch 1: recon_error=0.35 (poor)
Epoch 10: recon_error=0.08 (decent)
Epoch 15: recon_error=0.03 (good!)

Talker:
Epoch 1: loss=6.8, ppl=900 (random)
Epoch 10: loss=3.2, ppl=24 (learning patterns)
Epoch 25: loss=2.1, ppl=8 (good generation!)
```

---

## 🎓 Output

```
checkpoints/rvq_codec/
├── rvq_best.pt          # RVQ Codec

checkpoints/talker_tiny/
├── talker_best.pt       # Talker model
```

Enables text-to-speech in final system!

---

[Continue to Chapter 31: Stage E - Multimodal SFT →](31-stage-e-sft.md)

**Chapter Progress:** Training Pipeline ●●●●●○ (5/6 complete)

---
