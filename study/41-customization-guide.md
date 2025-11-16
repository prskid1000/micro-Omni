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
// Parameters: ~60M (was 15M)
// GPU: Need 24GB+ VRAM
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
// Parameters: ~4M (was 15M)
// GPU: Works on 6GB
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
