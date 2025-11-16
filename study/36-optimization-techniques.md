# Chapter 36: Optimization Techniques

[← Previous: Data Preparation](35-data-preparation.md) | [Back to Index](00-INDEX.md) | [Next: Debugging →](37-debugging-troubleshooting.md)

---

## ⚡ Performance Optimizations

Key techniques to speed up training and inference in μOmni.

---

## 🚀 Training Optimizations

### 1. Mixed Precision (FP16)

**What:** Use 16-bit floats instead of 32-bit  
**Benefit:** 2x faster, 50% less memory  
**Enabled by default** in μOmni

```python
# Automatically uses torch.cuda.amp.autocast
with torch.cuda.amp.autocast():
    output = model(input)
```

### 2. Gradient Accumulation

**What:** Accumulate gradients over multiple batches  
**Benefit:** Simulate larger batch sizes without OOM

```json
{
  "batch_size": 4,
  "gradient_accumulation_steps": 4
  // Effective batch size = 16
}
```

### 3. Gradient Checkpointing

**What:** Trade compute for memory  
**Benefit:** Train larger models on same GPU

```python
model.gradient_checkpointing_enable()
```

### 4. Flash Attention

**What:** Memory-efficient attention implementation  
**Benefit:** 2-4x faster, less memory

```python
# Automatically used if available
pip install flash-attn
```

---

## 🎯 Inference Optimizations

### 1. KV Caching

**Essential for generation!** Reuses computed key-value pairs.

**Without KV cache:**
- Generate 100 tokens: ~30 seconds
- Recomputes attention for all previous tokens every step

**With KV cache:**
- Generate 100 tokens: ~3 seconds
- 10x speedup!

### 2. Batch Inference

Process multiple inputs simultaneously:
```python
responses = model.chat_batch([
    "Question 1?",
    "Question 2?",
    "Question 3?"
])
```

### 3. Quantization (INT8)

**What:** Convert weights to 8-bit integers  
**Benefit:** 4x smaller model, faster inference  
**Trade-off:** Slight quality degradation

```python
model = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)
```

---

## 💡 Best Practices

✅ **Always use FP16** for training  
✅ **Enable KV caching** for generation  
✅ **Use Flash Attention** if available  
✅ **Gradient accumulation** for large batches  
✅ **Monitor GPU memory** with `nvidia-smi`

---

[Continue to Chapter 37: Debugging →](37-debugging-troubleshooting.md)

---
