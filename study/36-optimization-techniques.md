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
**Status:** ✅ Implemented in all μOmni training scripts (2024 optimization)

**Technical Approach:**
- Uses binary file mode (`'rb'`) for accurate byte offset tracking
- Manually tracks file positions instead of relying on `f.tell()` (which can be unreliable)
- Stores only integer offsets, not actual data content
- Reads specific items on-demand in `__getitem__` using `f.seek()`

**Optimized Datasets (All Training Scripts):**

| Dataset | Files | Optimization | Memory Savings |
|---------|-------|--------------|----------------|
| **TextDataset** | `train_text.py` | File offset indexing | ~8 bytes/line vs full text |
| **ASRDataset** | `train_audio_enc.py` | CSV row offset indexing | ~8 bytes/row vs full dict |
| **TTSDataset** | `train_talker.py` | CSV row offset indexing | ~8 bytes/row vs full dict |
| **ImgCapDataset** | `train_vision.py` | JSON object offset indexing | ~16 bytes/object vs full JSON |
| **MixDataset** | `sft_omni.py` | All three types (text, CSV, JSON) | Combined savings |

**Implementation Details:**

**Text Files:**
```python
# Before: Loads entire file into RAM
self.lines = [l.strip() for l in open(path) if l.strip()]  # Could be GB!

# After: Only stores file positions (integers)
self.line_offsets = []  # Just 8 bytes per line
with open(path, 'rb') as f:
    offset = 0
    while True:
        line_start = offset
        line_bytes = f.readline()
        if not line_bytes:
            break
        if line_bytes.decode('utf-8').strip():
            self.line_offsets.append(line_start)
        offset += len(line_bytes)  # Manual tracking

# Reads on-demand in __getitem__
def __getitem__(self, i):
    with open(self.path, 'rb') as f:
        f.seek(self.line_offsets[i])
        text = f.readline().decode('utf-8').strip()
```

**CSV Files:**
```python
# Before: Loads all rows into RAM
self.rows = []
for r in csv.DictReader(open(csv_path)):
    self.rows.append(r)  # Could be hundreds of MB!

# After: Stores row offsets + header fieldnames
self.row_offsets = []  # Just 8 bytes per row
self.fieldnames = header_line.decode('utf-8').strip().split(',')
# Reads specific row on-demand with proper CSV parsing
```

**JSON Files:**
```python
# Before: Parses entire JSON array into RAM
self.items = json.load(open(manifest))  # Could be large!

# After: Custom JSON parser finds object boundaries
self.item_offsets = []  # 8 bytes per offset
self.item_lengths = []  # 8 bytes per length
# Parses JSON byte-by-byte to find { } boundaries
# Falls back to full load if parsing fails (robust)
```

**Memory Savings Examples:**
- **10M line text file**: ~500MB → ~80MB (6x reduction)
- **1M row CSV**: ~200MB → ~8MB (25x reduction)
- **100K image JSON**: ~50MB → ~1.6MB (30x reduction)
- **Combined (MixDataset)**: All three optimizations applied simultaneously

**Key Benefits:**
- ✅ RAM usage now typically **lower than VRAM** (was opposite before)
- ✅ Can train on systems with limited RAM (8GB+)
- ✅ No code changes needed - automatic in all datasets
- ✅ Robust fallback for malformed JSON files
- ✅ Maintains same training speed (minimal I/O overhead)

**Related Optimizations:**
- Removed `gc.collect()` calls (Python's GC handles this automatically)
- Removed `torch.cuda.empty_cache()` calls (PyTorch manages CUDA memory efficiently)
- These manual calls were unnecessary and could actually hurt performance

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
  "batch_size": 4,  // Start conservatively
  "gradient_accumulation_steps": 4  // Simulate batch_size=16
}
```

**Strategy:**
1. Start with `batch_size: 4` (conservative, works on most GPUs)
2. Increase gradually (8, 16, 32) until you hit OOM
3. Use gradient accumulation to simulate larger batches without OOM

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
