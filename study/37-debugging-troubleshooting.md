# Chapter 37: Debugging & Troubleshooting

[← Previous: Optimization](36-optimization-techniques.md) | [Back to Index](00-INDEX.md) | [Next: Setup Environment →](38-setup-environment.md)

---

## 🔧 Common Issues & Solutions

---

## ⚠️ Training Issues

### 1. Loss Not Decreasing

**Symptoms:** Loss stays high or increases  
**Causes & Solutions:**
- **Learning rate too high:** Reduce to 1e-4 or 3e-5
- **Bad data:** Check data format and quality
- **Model too small:** Increase layers/dimensions
- **Gradient issues:** Check for NaN/Inf with `torch.isnan(loss)`

### 2. CUDA Out of Memory (OOM)

**Solutions:**
```python
# Option 1: Reduce batch size
"batch_size": 4  # Instead of 16

# Option 2: Enable gradient checkpointing
model.gradient_checkpointing_enable()

# Option 3: Use gradient accumulation
"gradient_accumulation_steps": 4

# Option 4: Reduce context length
"ctx_len": 256  # Instead of 512
```

### 3. NaN Loss / Gradient Explosion

**Causes:**
- Gradient explosion (gradients too large)
- Learning rate too high
- Numerical instability

**Solutions:**
```json
{
  "max_grad_norm": 1.0,  // Clip gradients (default: 1.0)
  "lr": 3e-4,  // Lower learning rate if explosions persist
  "use_amp": false  // Disable FP16 temporarily if needed
}
```

**How Gradient Handling Works:**
All training scripts use a robust gradient handling pattern:
1. **Clip gradients first** - Brings gradients down to `max_grad_norm` (typically 1.0)
2. **Check after clipping** - Only skips batch if gradients are still extremely high (>100.0) after clipping
3. **Automatic recovery** - Most gradient explosions are handled by clipping, allowing training to continue

This prevents batches from being skipped unnecessarily when gradients are moderately high (e.g., 50-100), as they get clipped to a safe range and training proceeds normally.

### 4. Memory Issues During Preprocessing

**Symptoms:** Out of memory when building vocabulary or counting tokens  
**Solutions:**
- ✅ **Automatic:** All preprocessing uses efficient streaming (no action needed)
- ✅ **Resumable:** If interrupted, just restart - will resume from checkpoint
- ✅ **Tokenizer training:** SentencePiece handles large files directly (no temp files for plain text)
- ✅ **Temp files:** Only used for CSV/JSON extraction, stored in `data/.temp/` and auto-cleaned
- ✅ **Vocabulary building:** Processes incrementally with checkpoints
- ✅ **Token counting:** Streams line-by-line with resume support
- ✅ **Datasets:** All use streaming `IterableDataset` (no cache files, minimal memory)

**Checkpoint Files:**
- Vocabulary: `{save_dir}/vocab_build_checkpoint.json`
- Token counting: `{file_path}.token_count_checkpoint.json`
- OCR vocabulary: `{csv_path}.vocab_checkpoint.json`
- Temp files: `data/.temp/*.txt` (auto-cleaned after use)
- All checkpoints auto-deleted after successful completion

---

## 🐛 Inference Issues

### 1. Slow Generation

**Check:**
- KV caching enabled? `model.generate(..., use_cache=True)`
- Using CPU instead of GPU? `model.to('cuda')`
- Flash attention installed? `pip install flash-attn`

### 2. Poor Quality Outputs

**Diagnosis:**
```python
# Check perplexity
ppl = model.compute_perplexity(test_data)
print(f"Perplexity: {ppl}")  # Should be <20

# Verify checkpoint loaded
print(model.config)  # Check parameters match training
```

### 3. Multimodal Not Working

**Common issues:**
- Wrong image size (must be 224×224)
- Audio not 16kHz sample rate
- Encoders not loaded correctly

---

## 🛠️ Resuming Training

**Automatic Resuming:** All training scripts automatically detect and resume from the latest checkpoint:

```bash
# Simply rerun the training command - it will auto-resume!
python train_text.py --config configs/thinker_tiny.json

# The script will:
# 1. Automatically find the latest checkpoint in save_dir
# 2. Load model, optimizer, scheduler, and scaler states
# 3. Set skip_samples on dataset to skip already-processed samples
# 4. Calculate correct epoch and batch position
# 5. Continue training from where it left off
```

**How it works:**
- ✅ Scans `save_dir` for checkpoints matching pattern (e.g., `thinker_step_*.pt`)
- ✅ Loads the highest step number checkpoint automatically
- ✅ Updates dataset `skip_samples` to skip processed samples
- ✅ Recreates DataLoader to ensure workers pick up new skip_samples
- ✅ Initializes progress bar at correct position
- ✅ Validation always uses full validation set (skip_samples temporarily reset)
- ✅ **Automatic skip_samples reset** - Dataset automatically resets `skip_samples` to 0 after each iteration completes
- ✅ **Dataset exhaustion handling** - Gracefully handles datasets smaller than one epoch or total epochs

**Dataset Exhaustion Scenarios:**
1. **Dataset exceeds max_steps:** Training stops when `max_steps` is reached
2. **Dataset smaller than one epoch:** Processes all available batches, resets, next epoch starts from beginning
3. **Dataset smaller than total epochs:** Each epoch processes all data from beginning, training continues until `max_steps`

**Checkpoint Format:**
```python
{
    "model": model.state_dict(),
    "optimizer": opt.state_dict(),
    "scheduler": scheduler.state_dict(),
    "scaler": scaler.state_dict(),  # If using AMP
    "step": step_number
}
```

**Note:** No `--resume` flag needed - resuming is automatic and seamless!

---

## 🛠️ Debug Commands

```bash
# Check GPU status
nvidia-smi

# Verify installation
python scripts/check_setup.py

# Test individual components
python -c "from omni import ThinkerLM; print('OK')"

# Check data format
python scripts/check_data.py --stage A

# Monitor training
tensorboard --logdir checkpoints/
```

---

## 💡 Best Practices

✅ **Start small:** Test on 10 samples first  
✅ **Check loss:** Should decrease steadily  
✅ **Monitor GPU:** Watch `nvidia-smi`  
✅ **Save often:** Checkpoints every 1000 steps  
✅ **Log everything:** Use tensorboard/wandb

---

[Continue to Chapter 38: Setup Environment →](38-setup-environment.md)

---
