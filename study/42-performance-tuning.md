# Chapter 42: Performance Tuning and Scaling

[Back to Index](00-INDEX.md)

---

## 🎯 Optimization Strategies

### 1. Training Speed

**Enable Mixed Precision (FP16)**:
```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for batch in dataloader:
    with autocast():
        loss = model(batch)
    
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

**Gradient Accumulation** (for larger effective batch):
```python
accumulation_steps = 4

for i, batch in enumerate(dataloader):
    loss = model(batch) / accumulation_steps
    loss.backward()
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

**Gradient Checkpointing** (save memory):
```json
{
  "use_gradient_checkpointing": true
}
```

### 2. Inference Speed

**Use Flash Attention**:
```python
# Automatically enabled in PyTorch 2.0+
# 2-4x speedup for attention
```

**KV Caching**:
```python
model.enable_kv_cache(True)
model.reset_kv_cache()  # Before each sequence
```

**Batch Inference**:
```python
# Process multiple inputs together
images = [img1, img2, img3]
batch = torch.stack([transform(img) for img in images])
outputs = model(batch)  # Faster than one-by-one
```

### 3. Memory Optimization

**Reduce Batch Size**:
```json
{"batch_size": 4}  // Instead of 16
```

**Lower Context Length**:
```json
{"ctx_len": 512}  // Instead of 2048
```

**Use Gradient Checkpointing**:
Trades computation for memory (20-30% slower, 50% less memory).

### 4. Multi-GPU Training

```bash
# Data parallelism
python -m torch.distributed.launch \
  --nproc_per_node=2 \
  train_text.py --config configs/thinker_tiny.json
```

## 📊 Performance Benchmarks

| Configuration | Training Speed | Memory Usage | Quality |
|---------------|----------------|--------------|---------|
| **Default** | 100% | 10GB | Baseline |
| **+ FP16** | 150% | 6GB | Same |
| **+ Flash Attn** | 180% | 6GB | Same |
| **+ Grad Ckpt** | 140% | 4GB | Same |
| **All optimizations** | 200% | 4GB | Same |

## 💡 Key Takeaways

✅ **Mixed precision (FP16)** → 1.5x speedup  
✅ **Flash Attention** → 2-4x faster attention  
✅ **KV caching** → 10x faster generation  
✅ **Gradient checkpointing** → 50% less memory  
✅ **Combine optimizations** for best results

---

[Back to Index](00-INDEX.md)

