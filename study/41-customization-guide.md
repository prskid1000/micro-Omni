# Chapter 41: Customization Guide

[← Previous: Inference Examples](40-inference-examples.md) | [Back to Index](00-INDEX.md) | [Next: Performance Tuning →](42-performance-tuning.md)

---

## 🎨 Customizing μOmni

How to adapt μOmni for your specific needs.

---

## 🔧 Common Customizations

### 1. Change Model Size

**Make it Larger (Better Quality):**
```json
// configs/thinker_large.json
{
  "d_model": 512,      // Was 256
  "n_layers": 8,       // Was 4
  "n_heads": 8,        // Was 4
  "d_ff": 2048         // Was 1024
}
// Parameters: ~80M (was 20.32M)
// GPU: Need 24GB+ VRAM
// Expected Performance: ~70-80% of max (vs 40-50% for tiny)
```

**Make it Smaller (Faster):**
```json
// configs/thinker_micro.json
{
  "d_model": 128,      // Was 256
  "n_layers": 2,       // Was 4
  "n_heads": 2,        // Was 4
  "d_ff": 512          // Was 1024
}
// Parameters: ~4M (was 20.32M)
// GPU: Works on 6GB
// Expected Performance: ~30-40% of max (faster inference)
```

**Performance vs Size Trade-offs:**

```
Performance vs Model Size:
- 25M (Tiny): ~40-50% performance, 50-100 TPS, 12GB VRAM
- 100M (Medium): ~70-80% performance, 20-40 TPS, 24GB VRAM
- 500M (Large): ~85-90% performance, 5-10 TPS, 40GB+ VRAM
- 1B+: ~90-95% performance, <5 TPS, Multi-GPU required

Key Insight: Performance scales sublinearly. Doubling parameters
doesn't double performance. Diminishing returns after ~500M.
```

### 2. Add New Modality

```python
# Example: Add video support
class VideoEncoder(nn.Module):
    def __init__(self):
        # Extract frames → Process with ViT
        # Output: (num_frames, 256) embeddings
        
# In inference:
video_emb = video_encoder(video_path)  # (T_v, 256)
text_emb = tokenizer(text)             # (T_t, 256)
combined = torch.cat([video_emb, text_emb], dim=0)
response = thinker(combined)
```

### 3. Fine-tune for Specific Task

```python
# Fine-tune on domain-specific data
# Example: Medical image understanding

# Prepare data
data = [
    {"image": "xray1.jpg", "question": "What do you see?", "answer": "..."},
    # ... more medical examples
]

# Fine-tune (Stage E style)
python sft_omni.py \
  --config configs/medical_sft.json \
  --data data/medical/ \
  --base_model checkpoints/omni_sft_tiny/omni_final.pt
```

### 4. Change Vocabulary Size

```json
{
  "vocab_size": 10000,  // Was 5000
  // Covers more words, but slower
  // Retrain tokenizer with more data
}
```

### 5. Modify Attention Mechanism

```python
# Use different attention (e.g., local attention)
class LocalAttention(nn.Module):
    def forward(self, q, k, v):
        # Attend only to nearby tokens
        # Reduces computation for long sequences
```

### 6. Customize Vocoder

**Use HiFi-GAN for Better Quality:**
```bash
# Train neural vocoder (optional, improves speech quality)
python train_vocoder.py --config configs/vocoder_tiny.json

# Customize for your GPU:
# - 12GB VRAM: batch_size=2, max_audio_length=8192
# - 24GB VRAM: batch_size=4, max_audio_length=16384
# - 6GB VRAM: batch_size=1, max_audio_length=4096
```

**Adjust Vocoder Config:**
```json
{
  "batch_size": 2,              // Reduce if OOM
  "max_audio_length": 8192,     // Shorter = less memory
  "gradient_accumulation_steps": 4,  // Simulate larger batch
  "lambda_mel": 45.0,           // Mel loss weight
  "lambda_fm": 2.0,             // Feature matching weight
  "lambda_adv": 1.0             // Adversarial loss weight
}
```

**Note:** Griffin-Lim works without training, but HiFi-GAN provides significantly better quality.

### 7. Customize OCR Model

**Train OCR for Text Extraction:**
```bash
# Train OCR model (optional, for text extraction from images)
python train_ocr.py --config configs/ocr_tiny.json

# Customize for your GPU:
# - 12GB VRAM: batch_size=4, gradient_accumulation_steps=2
# - 24GB VRAM: batch_size=8, gradient_accumulation_steps=1
# - 6GB VRAM: batch_size=2, gradient_accumulation_steps=4
```

**Adjust OCR Config:**
```json
{
  "vision_d_model": 128,        // Vision encoder dimension
  "vision_layers": 4,           // Vision encoder layers
  "decoder_d_model": 256,        // Text decoder dimension
  "decoder_layers": 4,          // Text decoder layers
  "batch_size": 4,              // Reduce if OOM
  "gradient_accumulation_steps": 2,  // Simulate larger batch
  "lr": 3e-4                    // Learning rate
}
```

**Usage in Inference:**
```bash
# Extract text from image
python infer_chat.py \
    --ckpt_dir checkpoints/ocr_tiny \
    --image path/to/image.png \
    --ocr

# Combine OCR with multimodal understanding
python infer_chat.py \
    --ckpt_dir checkpoints/omni_sft_tiny \
    --image path/to/image.png \
    --text "What text do you see?" \
    --ocr
```

---

## 🎯 Common Use Cases

### Academic Research
- Experiment with architectures
- Test new attention mechanisms
- Ablation studies

### Production Deployment
- Optimize for inference speed
- Quantize models (INT8)
- Reduce model size

### Domain-Specific Applications
- Medical: Radiology reports + X-rays
- Education: Tutoring with diagrams
- Customer Service: Voice + screen sharing

---

## 💡 Best Practices

✅ **Start from pretrained model** (transfer learning)  
✅ **Test on small data first** before full training  
✅ **Keep backups** of working checkpoints  
✅ **Document changes** in config files  
✅ **Validate improvements** with metrics

---

[Continue to Chapter 42: Performance Tuning →](42-performance-tuning.md)

---
