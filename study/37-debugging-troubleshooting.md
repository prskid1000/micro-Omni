# Chapter 37: Debugging and Troubleshooting

[← Previous: Optimization Techniques](36-optimization-techniques.md) | [Back to Index](00-INDEX.md) | [Next: Setup Environment →](38-setup-environment.md)

---

## 🐛 Common Issues and Solutions

### 1. CUDA Out of Memory

**Error**:
```
RuntimeError: CUDA out of memory. Tried to allocate X MB
```

**Solutions**:

```python
# 1. Reduce batch size
{"batch_size": 4}  # Instead of 16

# 2. Enable gradient checkpointing
{"use_gradient_checkpointing": true}

# 3. Reduce sequence length
{"ctx_len": 512}  # Instead of 2048

# 4. Use gradient accumulation
accumulation_steps = 4

# 5. Clear cache
import torch
torch.cuda.empty_cache()

# 6. Check for memory leaks
# Don't hold onto intermediate tensors
# Use .detach() or .item() for scalars
loss_value = loss.item()  # Not loss
```

---

### 2. NaN Loss

**Error**: Loss becomes NaN after some steps

**Causes & Solutions**:

```python
# 1. Learning rate too high
{"learning_rate": 1e-5}  # Reduce by 10x

# 2. Gradient explosion
{"max_grad_norm": 1.0}  # Enable gradient clipping

# 3. Numerical instability
# Use mixed precision carefully
{"use_amp": false}  # Try without FP16

# 4. Check for inf/nan in data
assert not torch.isnan(inputs).any()
assert not torch.isinf(inputs).any()

# 5. Debug forward pass
def check_nan(name, tensor):
    if torch.isnan(tensor).any():
        print(f"NaN in {name}!")
        raise ValueError

check_nan("after_layer1", x)
```

---

### 3. Loss Not Decreasing

**Symptoms**: Loss stays constant or increases

**Solutions**:

```python
# 1. Learning rate too small
{"learning_rate": 1e-3}  # Increase

# 2. Check data loading
for batch in dataloader:
    print(batch.shape)  # Verify correct format
    print(batch.min(), batch.max())  # Check ranges
    break

# 3. Verify labels
assert targets.shape == outputs.shape
assert targets.min() >= 0
assert targets.max() < vocab_size

# 4. Check model output
logits = model(inputs)
print(logits.shape)  # Correct dimensions?
print(logits.mean(), logits.std())  # Reasonable values?

# 5. Simplify model (test if it can overfit small data)
tiny_data = data[:10]  # Just 10 samples
# Train until perfect fit
# If can't overfit → model/code issue
```

---

### 4. Import Errors

**Error**: `ModuleNotFoundError: No module named 'omni'`

**Solutions**:

```bash
# 1. Check you're in project root
pwd  # Should show .../μOmni

# 2. Add to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# 3. Check virtual environment
which python  # Should be in .venv or conda env

# 4. Reinstall dependencies
pip install -r requirements.txt
```

---

### 5. Slow Training

**Symptoms**: Much slower than expected

**Diagnose**:

```python
import time
import torch

# Profile data loading
start = time.time()
for batch in dataloader:
    pass
print(f"Data loading: {time.time() - start:.2f}s")

# Profile model forward
batch = next(iter(dataloader))
torch.cuda.synchronize()
start = time.time()
output = model(batch)
torch.cuda.synchronize()
print(f"Forward pass: {time.time() - start:.2f}s")

# Check GPU utilization
# Run in separate terminal:
watch -n 1 nvidia-smi
# Should see 80-100% GPU utilization
```

**Solutions**:

```python
# 1. Enable optimizations
{"use_amp": true, "use_flash_attention": true}

# 2. Increase num_workers
dataloader = DataLoader(..., num_workers=4)

# 3. Pin memory
dataloader = DataLoader(..., pin_memory=True)

# 4. Profile code
import torch.profiler
with torch.profiler.profile() as prof:
    model(batch)
print(prof.key_averages().table())
```

---

### 6. Poor Quality Outputs

**Symptoms**: Model generates gibberish

**Solutions**:

```python
# 1. Train longer
{"num_epochs": 20}  # Instead of 5

# 2. Check learning rate
# Too high → unstable
# Too low → slow convergence

# 3. Verify data quality
# Check a few samples manually

# 4. Monitor validation loss
# If train loss low but val loss high → overfitting

# 5. Check tokenizer
decoded = tokenizer.decode(tokenizer.encode("test"))
assert decoded == "test"

# 6. Temperature sampling
# Try different temperatures
output = sample(logits, temperature=0.7)
```

---

### 7. Checkpoint Loading Errors

**Error**: `RuntimeError: Error(s) in loading state_dict`

**Solutions**:

```python
# 1. Check architecture matches
model = ThinkerLM(vocab=5000, ...)  # Match training config

# 2. Strict loading (shows missing keys)
state_dict = torch.load("checkpoint.pt")
model.load_state_dict(state_dict, strict=False)

# 3. Inspect checkpoint
checkpoint = torch.load("checkpoint.pt")
print(checkpoint.keys())  # What's inside?

# 4. Load specific components
checkpoint = torch.load("omni.pt")
model.load_state_dict(checkpoint["thinker"])
proj_a.load_state_dict(checkpoint["proj_a"])
```

---

### 8. Audio Processing Issues

**Error**: Audio sounds wrong or crashes

**Solutions**:

```python
# 1. Verify audio format
import torchaudio
waveform, sr = torchaudio.load("audio.wav")
print(f"Sample rate: {sr}, Shape: {waveform.shape}")
assert sr == 16000, "Must be 16kHz!"

# 2. Resample if needed
if sr != 16000:
    waveform = torchaudio.functional.resample(waveform, sr, 16000)

# 3. Check mel spectrogram
mel_spec = torchaudio.transforms.MelSpectrogram(
    sample_rate=16000,
    n_fft=1024,
    hop_length=160,
    n_mels=128
)
mel = mel_spec(waveform)
print(f"Mel shape: {mel.shape}")  # Should be (1, 128, T)

# 4. Normalize audio
waveform = waveform / waveform.abs().max()
```

---

### 9. GPU Not Being Used

**Symptoms**: Training slow, nvidia-smi shows 0% GPU

**Solutions**:

```python
# 1. Check CUDA available
import torch
print(torch.cuda.is_available())  # Should be True

# 2. Move model to GPU
model = model.cuda()
# Or
model = model.to('cuda')

# 3. Move data to GPU
batch = batch.cuda()

# 4. Check device
print(next(model.parameters()).device)  # Should be cuda:0

# 5. Verify CUDA installation
import torch
print(torch.version.cuda)  # Should match nvidia-smi
```

---

### 10. Memory Leak

**Symptoms**: Memory usage keeps increasing

**Solutions**:

```python
# 1. Don't hold references to tensors
# BAD:
losses = []
for batch in dataloader:
    loss = model(batch)
    losses.append(loss)  # Keeps computation graph!

# GOOD:
losses = []
for batch in dataloader:
    loss = model(batch)
    losses.append(loss.item())  # Just the scalar

# 2. Clear gradients
optimizer.zero_grad()

# 3. Delete large tensors
del large_tensor
torch.cuda.empty_cache()

# 4. Use torch.no_grad() for validation
with torch.no_grad():
    val_loss = model(val_batch)
```

---

## 🔍 Debugging Tools

### 1. Print Shapes

```python
def debug_shapes(x, name="tensor"):
    print(f"{name}: {x.shape}, dtype={x.dtype}, device={x.device}")

debug_shapes(embeddings, "embeddings")
debug_shapes(attention_output, "attention")
```

### 2. Check Gradients

```python
# Check if gradients are flowing
for name, param in model.named_parameters():
    if param.grad is not None:
        print(f"{name}: grad_norm={param.grad.norm().item():.4f}")
    else:
        print(f"{name}: NO GRADIENT!")
```

### 3. Validate Inputs

```python
def validate_batch(batch, vocab_size):
    assert isinstance(batch, torch.Tensor)
    assert batch.dtype == torch.long
    assert batch.min() >= 0
    assert batch.max() < vocab_size
    assert not torch.isnan(batch).any()
```

### 4. Profile Memory

```python
import torch

print(f"Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
print(f"Cached: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
print(f"Max: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
```

---

## 💡 Prevention Tips

1. **Start small** - Test on tiny data first
2. **Monitor everything** - Loss, gradients, memory
3. **Save checkpoints** - Don't lose progress
4. **Version control** - Git commit working states
5. **Validate often** - Check val loss regularly
6. **Log everything** - Use print or tensorboard
7. **Read errors** - Full stack trace has clues

---

## 💡 Key Takeaways

✅ **Most common**: OOM, NaN loss, slow training  
✅ **Always check**: Data format, model device, shapes  
✅ **Debug tools**: Print shapes, check gradients, profile  
✅ **Prevention**: Start small, validate often, save checkpoints

---

[Continue to Chapter 38: Setting Up Your Environment →](38-setup-environment.md)

