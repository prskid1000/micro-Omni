# Chapter 39: Running Training Scripts

[← Previous: Setup Environment](38-setup-environment.md) | [Back to Index](00-INDEX.md) | [Next: Inference Examples →](40-inference-examples.md)

---

## 🎯 What You'll Learn

- Running each training stage
- Understanding training outputs
- Monitoring training progress
- Hyperparameter tuning
- Common training issues

---

## 📋 Training Overview

### 5-Stage Pipeline

```
Stage A: Thinker Pretraining (Text)
  ↓
Stage B: Audio Encoder (ASR)
  ↓
Stage C: Vision Encoder (Image)
  ↓
Stage D: Talker + RVQ (Speech)
  ↓
Stage E: Multimodal SFT (Joint)
```

Each stage trains independently!

---

## 📝 Stage A: Thinker Pretraining

### Purpose

Train the core language model on text data.

### Command

```bash
python train_text.py --config configs/thinker_tiny.json
```

### Configuration (configs/thinker_tiny.json)

```json
{
  "vocab_size": 5000,
  "n_layers": 4,
  "d_model": 256,
  "n_heads": 4,
  "d_ff": 1024,
  "dropout": 0.1,
  "rope_theta": 10000,
  "ctx_len": 512,
  
  "data_path": "data/text/corpus.txt",
  "batch_size": 16,
  "num_epochs": 10,
  "learning_rate": 3e-4,
  "warmup_steps": 1000,
  "max_grad_norm": 1.0,
  
  "checkpoint_dir": "checkpoints/thinker_tiny",
  "save_every": 1000,
  "eval_every": 500
}
```

### Expected Output

```
[2024-11-16 10:30:15] Starting Thinker pretraining
[2024-11-16 10:30:20] Loaded 10000 text samples
[2024-11-16 10:30:25] Model parameters: 62,345,000

Epoch 1/10:
Step 100/2500: loss=4.234 ppl=68.9 lr=0.000120 [12.3 samples/sec]
Step 200/2500: loss=3.892 ppl=49.0 lr=0.000240 [12.5 samples/sec]
Step 500/2500: loss=3.156 ppl=23.4 lr=0.000300 [12.7 samples/sec]
  → Validation: loss=3.201 ppl=24.5
  → Saved checkpoint: checkpoints/thinker_tiny/step_500.pt
...
Epoch 10/10: loss=1.987 ppl=7.3
✓ Training completed!
✓ Best model saved: checkpoints/thinker_tiny/thinker_best.pt
```

### What to Monitor

```
1. Loss: Should decrease over time
   - Start: ~5-7 (random predictions)
   - End: ~1.5-2.5 (good)
   - <1.0 (possibly overfitting)

2. Perplexity (PPL): Lower is better
   - PPL = exp(loss)
   - Good: 5-10 range

3. Learning Rate: Should follow schedule
   - Warmup: Gradual increase
   - Plateau: Constant
   - Decay: Gradual decrease

4. Throughput: Samples/second
   - Should be stable
   - Drops indicate bottleneck
```

---

## 🎤 Stage B: Audio Encoder Training

### Purpose

Train audio encoder for speech understanding (ASR task).

### Command

```bash
python train_audio_enc.py --config configs/audio_enc_tiny.json
```

### Data Preparation

```bash
# Prepare audio data
# data/audio/asr.csv format:
# audio_path,transcription
data/audio/wav/sample1.wav,"hello world"
data/audio/wav/sample2.wav,"how are you"
...

# Or use LibriSpeech (recommended)
python scripts/download_datasets.py --modality audio_asr
```

### Expected Output

```
[2024-11-16 11:00:00] Training Audio Encoder
[2024-11-16 11:00:05] Loaded 5000 audio samples
[2024-11-16 11:00:10] Model parameters: 12,456,000

Epoch 1/20:
Step 100/1250: ctc_loss=45.23 [8.5 samples/sec]
Step 200/1250: ctc_loss=32.18 [8.7 samples/sec]
Step 500/1250: ctc_loss=18.67 [8.9 samples/sec]
  → Validation: ctc_loss=19.34 wer=45.2%
  → Saved checkpoint

Epoch 20/20: ctc_loss=8.45 wer=12.3%
✓ Training completed!
```

### Metrics

```
CTC Loss: Connectionist Temporal Classification
- Measures alignment quality
- Lower is better
- Good: <10

WER (Word Error Rate):
- Percentage of incorrect words
- Lower is better
- Excellent: <10%
- Good: 10-20%
- Needs work: >30%
```

---

## 🖼️ Stage C: Vision Encoder Training

### Purpose

Train vision encoder for image understanding.

### Command

```bash
python train_vision.py --config configs/vision_tiny.json
```

### Data Format

```json
// data/images/annotations.json
{
  "images": [
    {
      "id": 1,
      "file_name": "image1.jpg",
      "caption": "A cat sitting on a couch"
    },
    ...
  ]
}
```

### Expected Output

```
[2024-11-16 12:00:00] Training Vision Encoder
[2024-11-16 12:00:05] Loaded 10000 images
[2024-11-16 12:00:10] Model parameters: 18,234,000

Epoch 1/15:
Step 100/1875: loss=2.345 acc=35.2% [15.3 images/sec]
Step 500/1875: loss=1.678 acc=58.7% [15.5 images/sec]
  → Validation: loss=1.723 acc=57.1%

Epoch 15/15: loss=0.987 acc=78.9%
✓ Training completed!
```

---

## 🔊 Stage D: Talker + RVQ Training

### Purpose

Train speech code generator (Talker) and RVQ codec jointly.

### Command

```bash
python train_talker.py --config configs/talker_tiny.json
```

### Expected Output

```
[2024-11-16 13:00:00] Training Talker + RVQ
[2024-11-16 13:00:05] Loaded 8000 audio samples
[2024-11-16 13:00:10] Model parameters: 14,567,000

Epoch 1/25:
Step 100/2000: base_loss=3.456 res_loss=3.234 recon=0.087 [6.8 samples/sec]
Step 500/2000: base_loss=2.123 res_loss=2.087 recon=0.045 [7.1 samples/sec]
  → Validation: recon_loss=0.048

Epoch 25/25: recon_loss=0.012
✓ Training completed!
```

### Metrics

```
base_loss: Base codebook prediction loss
res_loss: Residual codebook prediction loss
recon: Reconstruction loss (mel spectrogram MSE)
- Lower is better
- Good: <0.02
```

---

## 🌈 Stage E: Multimodal SFT

### Purpose

Fine-tune all components together for multimodal understanding.

### Command

```bash
python sft_omni.py --config configs/omni_sft_tiny.json
```

### Configuration

```json
{
  "thinker_ckpt": "checkpoints/thinker_tiny/thinker_best.pt",
  "audio_ckpt": "checkpoints/audio_enc_tiny/audio_enc.pt",
  "vision_ckpt": "checkpoints/vision_tiny/vision.pt",
  
  "batch_size": 8,
  "num_epochs": 5,
  "learning_rate": 1e-4,
  "warmup_steps": 500,
  
  "freeze_encoders": true,  // Only train projectors + Thinker
  "data_mix": {
    "text_only": 0.4,
    "image_text": 0.3,
    "audio_text": 0.3
  }
}
```

### Expected Output

```
[2024-11-16 14:00:00] Multimodal SFT
[2024-11-16 14:00:05] Loaded pretrained models
[2024-11-16 14:00:08] Trainable parameters: 68,234,000
[2024-11-16 14:00:10] Mixed multimodal batches ready

Epoch 1/5:
Step 100/1500: loss=2.456 text_acc=45.2% [4.5 samples/sec]
Step 500/1500: loss=1.678 text_acc=62.8% [4.7 samples/sec]
  → Validation: loss=1.734 
  → Image QA acc: 58.3%
  → Audio transcription WER: 18.5%

Epoch 5/5: loss=1.123 accuracy=75.6%
✓ SFT completed!
✓ Final model: checkpoints/omni_sft_tiny/omni.pt
```

---

## 📊 Training Tips

### 1. Start Small

```bash
# Test with small data first
head -n 1000 data/text/corpus.txt > data/text/corpus_tiny.txt

# Use small config
{
  "batch_size": 4,
  "num_epochs": 1,
  "max_steps": 100
}

# Verify training loop works
python train_text.py --config configs/test.json
```

---

### 2. Monitor GPU Usage

```bash
# Terminal 1: Run training
python train_text.py --config configs/thinker_tiny.json

# Terminal 2: Monitor GPU
watch -n 1 nvidia-smi

# Look for:
# - GPU utilization: Should be 80-100%
# - Memory usage: Should be stable
# - Temperature: Should be <85°C
```

---

### 3. Use TensorBoard (Optional)

```bash
# Add to training script:
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('runs/experiment1')

# Log metrics
writer.add_scalar('Loss/train', loss, step)
writer.add_scalar('PPL/train', ppl, step)

# View in browser
tensorboard --logdir=runs
# Open http://localhost:6006
```

---

### 4. Hyperparameter Tuning

```python
# Learning rate: Most important!
# Too high: Loss diverges
# Too low: Slow convergence

Recommended starting points:
- Pretraining: 3e-4 to 1e-3
- Fine-tuning: 1e-4 to 3e-4
- SFT: 1e-5 to 1e-4

# Batch size:
# Larger: Faster, more stable, needs more VRAM
# Smaller: Slower, noisier gradients, less VRAM

Recommended:
- 12GB GPU: batch_size=4-8
- 24GB GPU: batch_size=16-32

# Warmup steps:
# Gradually increase LR to avoid instability
# Recommended: 1000-2000 steps
```

---

### 5. Resume Training

```bash
# Training interrupted? Resume from checkpoint
python train_text.py \
  --config configs/thinker_tiny.json \
  --resume checkpoints/thinker_tiny/step_2000.pt

# Continues from step 2000
```

---

## ⚠️ Common Issues

### Issue 1: NaN Loss

```
Symptoms: Loss becomes NaN after a few steps

Causes:
- Learning rate too high
- Gradient explosion
- Numerical instability

Solutions:
1. Reduce learning rate by 10x
2. Enable gradient clipping (max_norm=1.0)
3. Use mixed precision training
4. Check for inf/nan in data
```

---

### Issue 2: No Improvement

```
Symptoms: Loss plateaus or doesn't decrease

Causes:
- Learning rate too small
- Model too small
- Data issues

Solutions:
1. Increase learning rate
2. Check data quality
3. Increase model capacity
4. Train longer
```

---

### Issue 3: Out of Memory

```
Solutions:
1. Reduce batch_size
2. Enable gradient_checkpointing
3. Use gradient_accumulation_steps
4. Reduce sequence length
5. Use mixed precision (fp16)
```

---

## 💡 Key Takeaways

✅ **5 training stages** (can train independently)  
✅ **Monitor loss, perplexity, accuracy** during training  
✅ **Start small** to verify setup  
✅ **GPU should be 80-100% utilized**  
✅ **Resume from checkpoints** if interrupted  
✅ **Tune learning rate first** (most important hyperparameter)

---

## 🎓 Self-Check Questions

1. What are the 5 training stages?
2. What's a good final perplexity for text modeling?
3. What does CTC loss measure?
4. Why start with small data first?
5. What's the most important hyperparameter to tune?

<details>
<summary>📝 Answers</summary>

1. A: Thinker pretraining, B: Audio encoder ASR, C: Vision encoder, D: Talker+RVQ, E: Multimodal SFT
2. 5-10 range (lower is better, but <3 may indicate overfitting)
3. CTC loss measures alignment quality between audio frames and text transcription
4. To verify training loop works before committing to long training runs
5. Learning rate (impacts convergence speed and stability most)
</details>

---

[Continue to Chapter 40: Running Inference Examples →](40-inference-examples.md)

**Chapter Progress:** Practical Usage ●●○○○ (2/5 complete)

