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

## 💾 Memory Optimizations

### 1. Lazy Dataset Loading

**What:** Load data on-demand instead of pre-loading into RAM  
**Benefit:** Reduces RAM usage by 90%+ for large datasets

**Implemented in all μOmni datasets:**
- **TextDataset**: File offset indexing (stores ~8 bytes/line vs full text)
- **ASRDataset/TTSDataset**: CSV row offset indexing (stores ~8 bytes/row vs full dicts)
- **ImgCapDataset**: JSON object offset indexing (stores ~16 bytes/object vs full JSON)

**Before:**
```python
# Loads entire file into RAM
self.lines = [l.strip() for l in open(path) if l.strip()]  # Could be GB!
```

**After:**
```python
# Only stores file positions (integers)
self.line_offsets = []  # Just 8 bytes per line
# Reads specific line on-demand in __getitem__
```

**Memory savings example:**
- 10M line text file: ~500MB → ~80MB (6x reduction)
- 1M row CSV: ~200MB → ~8MB (25x reduction)
- 100K image JSON: ~50MB → ~1.6MB (30x reduction)

### 2. DataLoader Workers

**What:** Parallel data loading processes  
**Trade-off:** More workers = faster loading but more RAM

```json
{
  "num_workers": 2,  // Default: 2 workers
  // Reduce to 0 or 1 if RAM is limited
}
```

**Recommendation:**
- **High RAM (32GB+)**: `num_workers: 2-4`
- **Medium RAM (16GB)**: `num_workers: 1-2`
- **Low RAM (8GB)**: `num_workers: 0-1`

### 3. Batch Size Tuning

**What:** Adjust batch size based on available memory  
**Benefit:** Maximize GPU utilization without OOM

```json
{
  "batch_size": 8,  // Start here
  "gradient_accumulation_steps": 4  // Simulate batch_size=32
}
```

**Strategy:**
1. Start with `batch_size: 4`
2. Increase until you hit OOM
3. Use gradient accumulation to simulate larger batches

---

## 💡 Best Practices

✅ **Always use FP16** for training  
✅ **Enable KV caching** for generation  
✅ **Use Flash Attention** if available  
✅ **Gradient accumulation** for large batches  
✅ **Lazy loading enabled** by default (no action needed)  
✅ **Monitor GPU memory** with `nvidia-smi`  
✅ **Monitor RAM usage** - should be much lower than VRAM now  
✅ **Reduce num_workers** if RAM is limited

---

[Continue to Chapter 37: Debugging →](37-debugging-troubleshooting.md)

---
