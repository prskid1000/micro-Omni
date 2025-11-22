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
**Loss:** Cross-entropy (both base and residual heads) with padding mask  
**Target:** Perplexity <15, intelligible speech

**Padding Handling:**
- Loss calculation masks out padding frames using `mel_lengths`
- Only valid mel frames contribute to loss
- Prevents padding from diluting the loss signal

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
  "learning_rate": 3e-4,
  
  "use_compile": true,
  "max_mel_length_percentile": 95.0  // Optional: Percentile for auto-calculation (default: 95.0)
  // max_mel_length is auto-calculated from dataset - no need to set manually
  "frame_ms": 80
}
```

**Key Parameters for CUDA Graphs Compatibility:**

**Auto-Calculation:**
- `max_mel_length` is **automatically calculated** from your dataset during training
- Uses **95th percentile** by default to minimize padding while covering 95% of data
- Automatically rounds up to nearest 256 for better memory alignment
- ~5% of data will be truncated if longer (acceptable for outliers)

**Frame Rate Reference:**
- Frame rate = sample_rate / hop_length = 16000 / (16000 × 0.08) = 12.5 frames/second
- For 60 seconds: 60 × 12.5 = 750 frames
- For 20 seconds: 20 × 12.5 = 250 frames

**Note:** Talker uses different frame rate than audio encoder due to `frame_ms=80` parameter.

**Why Fixed Length?**
- CUDA graphs require uniform batch shapes
- Prevents "tensor size mismatch" errors
- Enables 10-20% speedup with compilation

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

**Expected Validation Loss:**
- Target Loss: < 2.5
- Target Perplexity: < 15
- Good: loss < 2.0, perplexity < 10
- Excellent: loss < 1.5, perplexity < 8
```

---

## 🎓 Output

```
checkpoints/talker_tiny/
├── talker_step_1000.pt   # Periodic checkpoints (every 1000 steps)
└── talker_step_2000.pt
```

Enables text-to-speech in final system!

---

## 🔊 Optional: HiFi-GAN Neural Vocoder Training

**After Stage D**, you can optionally train a neural vocoder for higher quality speech:

```bash
# Train HiFi-GAN vocoder (optional, improves speech quality)
python train_vocoder.py --config configs/vocoder_tiny.json

# Time: 2-4 hours (on 12GB GPU)
# Target: Natural-sounding speech
# Output: checkpoints/vocoder_tiny/hifigan.pt
```

**Benefits:**
- ✅ More natural speech than Griffin-Lim
- ✅ Better prosody and quality
- ✅ Automatic fallback to Griffin-Lim if unavailable

**Memory Optimized:**
- Batch size: 2 (with gradient accumulation: effective batch size 8)
- Audio length limit: 8192 samples (~0.5s)
- Mixed precision (FP16) enabled
- Optimized for 12GB VRAM

**Note:** Griffin-Lim vocoder works without training, but HiFi-GAN provides better quality.

---

[Continue to Chapter 31: Stage E - Multimodal SFT →](31-stage-e-sft.md)

**Chapter Progress:** Training Pipeline ●●●●●○ (5/6 complete)

---
