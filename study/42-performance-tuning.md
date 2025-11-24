# Chapter 42: Performance Tuning

[← Previous: Customization Guide](41-customization-guide.md) | [Back to Index](00-INDEX.md) | [Next: Mathematical Foundations →](43-mathematical-foundations.md)

---

## ⚡ Maximizing Performance

Advanced techniques to optimize μOmni for speed and efficiency.

---

## 🚀 Inference Speed Optimizations

### 1. Model Compilation

```python
# PyTorch 2.0+ compile
model = torch.compile(model, mode='reduce-overhead')
# 20-30% speedup on generation!
```

### 2. Quantization Strategies

**Dynamic INT8 Quantization:**
```python
import torch.quantization as quant
quantized_model = quant.quantize_dynamic(
    model, {nn.Linear}, dtype=torch.qint8
)
# 4x smaller, 2-3x faster, minimal quality loss
```

**Static INT8 (Best Quality):**
```python
# Requires calibration data
model.eval()
model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
torch.quantization.prepare(model, inplace=True)
# Run calibration data
torch.quantization.convert(model, inplace=True)
```

### 3. Batch Size Tuning

```python
# Find optimal batch size
for batch_size in [1, 2, 4, 8, 16]:
    time = benchmark(model, batch_size)
    throughput = batch_size / time
    print(f"Batch {batch_size}: {throughput:.1f} samples/sec")
# Use highest throughput
```

### 4. ONNX Export (Deployment)

```python
# Export to ONNX for production
torch.onnx.export(
    model,
    dummy_input,
    "model.onnx",
    opset_version=14,
    do_constant_folding=True
)
# Use with ONNX Runtime for faster inference
```

---

## 🎯 Training Speed Optimizations

### 1. Data Loading

```python
# Multi-worker data loading
dataloader = DataLoader(
    dataset,
    batch_size=16,
    num_workers=4,      # Parallel loading
    pin_memory=True,    # Faster GPU transfer
    prefetch_factor=2   # Prefetch batches
)
```

### 2. Learning Rate Scheduling

```python
# Cosine annealing for better convergence
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=num_epochs,
    eta_min=1e-6
)
```

### 3. Early Stopping

```python
# Stop when validation stops improving
if val_loss < best_loss:
    best_loss = val_loss
    patience_counter = 0
else:
    patience_counter += 1
    if patience_counter >= 5:  # 5 epochs no improvement
        print("Early stopping!")
        break
```

---

## 📊 Profiling & Benchmarking

### PyTorch Profiler

```python
with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ]
) as prof:
    model(input)

print(prof.key_averages().table(sort_by="cuda_time_total"))
```

### Memory Profiling

```python
import torch.cuda
torch.cuda.reset_peak_memory_stats()
output = model(input)
peak_memory = torch.cuda.max_memory_allocated() / 1024**3
print(f"Peak memory: {peak_memory:.2f} GB")
```

---

## 💡 Performance Targets

**Inference (12GB GPU):**
- Text generation: 30-50 tokens/sec
- Image processing: <100ms per image
- Audio transcription: 2-3x real-time
- Text-to-speech: 1-2x real-time
- OCR (text extraction): <200ms per image

**Training (12GB GPU):**
- Stage A: 8-12 hours
- Stage B: 6-10 hours
- Stage C: 4-8 hours
- Stage D: 10-15 hours
- Stage E: 6-12 hours
- **Total: 40-60 hours**
- Optional OCR: 4-8 hours
- Optional HiFi-GAN: 2-4 hours

---

[Continue to Chapter 43: Mathematical Foundations →](43-mathematical-foundations.md)

---
