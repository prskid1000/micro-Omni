# Chapter 36: Optimization Techniques

[← Previous: Data Preparation](35-data-preparation.md) | [Back to Index](00-INDEX.md) | [Next: Debugging →](37-debugging-troubleshooting.md)

---

## 🚀 Training Optimizations

### 1. Mixed Precision (FP16)

**Speed-up**: 1.5-2x faster  
**Memory**: 30-50% reduction

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()
model = YourModel().cuda()

for batch in dataloader:
    optimizer.zero_grad()
    
    # Forward pass in FP16
    with autocast():
        output = model(batch)
        loss = criterion(output, targets)
    
    # Backward pass with scaled gradients
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

**Enable in config**:
```json
{"use_amp": true}
```

---

### 2. Gradient Checkpointing

**Memory**: 50% reduction  
**Speed**: 10-30% slower

```python
from torch.utils.checkpoint import checkpoint

class ModelWithCheckpointing(nn.Module):
    def forward(self, x):
        # Checkpointed block (saves memory)
        x = checkpoint(self.layer1, x)
        x = checkpoint(self.layer2, x)
        return x
```

**Enable in config**:
```json
{"use_gradient_checkpointing": true}
```

---

### 3. Gradient Accumulation

**Effective batch size = batch_size × accumulation_steps**

```python
accumulation_steps = 4
optimizer.zero_grad()

for i, batch in enumerate(dataloader):
    loss = model(batch) / accumulation_steps
    loss.backward()
    
    if (i + 1) % accumulation_steps == 0:
        # Update weights every 4 batches
        optimizer.step()
        optimizer.zero_grad()
```

**Benefit**: Train with larger effective batch size on limited memory

---

### 4. Flash Attention

**Speed-up**: 2-4x faster attention  
**Memory**: 50% less

```python
# Automatically enabled in PyTorch 2.0+
import torch.nn.functional as F

# Standard attention (slow)
scores = Q @ K.T / sqrt(d)
attn = softmax(scores)
output = attn @ V

# Flash Attention (fast)
output = F.scaled_dot_product_attention(Q, K, V)
```

**Requirement**: PyTorch 2.0+

**μOmni uses Flash Attention** automatically when available!

---

### 5. Gradient Clipping

**Prevents**: Exploding gradients

```python
# Clip gradients to max norm
torch.nn.utils.clip_grad_norm_(
    model.parameters(),
    max_norm=1.0  # Recommended value
)
```

**In config**:
```json
{"max_grad_norm": 1.0}
```

---

## ⚡ Inference Optimizations

### 1. KV Caching

**Speed-up**: 10x faster generation

```python
model.enable_kv_cache(True)
model.reset_kv_cache()

# First token (full prompt)
output = model(prompt_tokens)  # Slow, builds cache

# Subsequent tokens (one at a time)
for _ in range(max_tokens):
    output = model(next_token)  # Fast! Uses cache
```

**μOmni has KV caching** enabled by default in `infer_chat.py`

---

### 2. Batch Inference

**Speed-up**: 3-5x faster for multiple inputs

```python
# Slow: One at a time
for image in images:
    output = model(image)

# Fast: Batch processing
batch = torch.stack(images)
outputs = model(batch)  # Process all together
```

---

### 3. Model Compilation (PyTorch 2.0+)

**Speed-up**: 20-50% faster

```python
import torch

# Compile model
model = torch.compile(model, mode='reduce-overhead')

# Use normally
output = model(input)  # Faster!
```

**Note**: First run is slow (compilation), subsequent runs are faster

---

### 4. torch.inference_mode()

**Faster than** `torch.no_grad()`

```python
# Standard
with torch.no_grad():
    output = model(input)

# Better for inference
with torch.inference_mode():
    output = model(input)  # Slightly faster
```

---

## 💾 Memory Optimizations

### 1. Reduce Batch Size

```json
// Before (OOM error)
{"batch_size": 32}

// After (fits in memory)
{"batch_size": 8}
```

---

### 2. Reduce Sequence Length

```json
// Before
{"ctx_len": 2048}

// After
{"ctx_len": 512}
```

**Memory savings**: ~4x less (quadratic in sequence length)

---

### 3. Clear Cache

```python
import torch

# After training/inference
torch.cuda.empty_cache()
```

---

### 4. CPU Offloading (If Desperate)

```python
# Keep model on CPU, move batches to GPU
model = model.cpu()

for batch in dataloader:
    batch = batch.cuda()
    output = model.cuda()(batch)  # Temporary move
    model = model.cpu()  # Move back
```

**Warning**: Very slow, only for extreme memory constraints

---

## 🎯 Optimizer Choices

### AdamW (Recommended)

```python
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=3e-4,
    betas=(0.9, 0.999),
    eps=1e-8,
    weight_decay=0.01  # L2 regularization
)
```

**Benefits**:
- Adaptive learning rates
- Good default choice
- Works well for most tasks

---

### SGD with Momentum

```python
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.01,
    momentum=0.9,
    weight_decay=1e-4
)
```

**Benefits**:
- Better generalization sometimes
- Needs careful LR tuning

---

## 📊 Learning Rate Scheduling

### Warmup + Cosine Decay

```python
from torch.optim.lr_scheduler import CosineAnnealingLR

scheduler = CosineAnnealingLR(
    optimizer,
    T_max=num_epochs,
    eta_min=1e-6
)

# Training loop
for epoch in range(num_epochs):
    train(...)
    scheduler.step()
```

**Pattern**:
```
LR
 │    Warmup        Cosine Decay
 │      ╱╲___
 │    ╱     ╲___
 │  ╱           ╲___
 │╱                 ╲___
 └─────────────────────────→ Step
```

---

## 💡 Optimization Checklist

**Always Enable:**
- ✅ Mixed precision (FP16)
- ✅ Gradient clipping
- ✅ Flash Attention (PyTorch 2.0+)
- ✅ KV caching (inference)

**If Out of Memory:**
- ✅ Gradient checkpointing
- ✅ Reduce batch size
- ✅ Gradient accumulation
- ✅ Reduce sequence length

**For Speed:**
- ✅ Batch inference
- ✅ Model compilation (PyTorch 2.0+)
- ✅ Profile code to find bottlenecks

---

## 📊 Performance Comparison

| Configuration | Speed | Memory | Quality |
|---------------|-------|--------|---------|
| **Baseline** | 1.0x | 12GB | 100% |
| **+ FP16** | 1.5x | 7GB | 100% |
| **+ Flash Attn** | 2.2x | 6GB | 100% |
| **+ Grad Ckpt** | 1.8x | 4GB | 100% |
| **+ All** | 2.5x | 4GB | 100% |

---

## 💡 Key Takeaways

✅ **Mixed precision** → Free 1.5x speedup  
✅ **Flash Attention** → 2-4x faster attention  
✅ **KV caching** → Essential for generation  
✅ **Gradient checkpointing** → Halves memory  
✅ **Combine optimizations** → Multiplicative gains

---

[Continue to Chapter 37: Debugging and Troubleshooting →](37-debugging-troubleshooting.md)

