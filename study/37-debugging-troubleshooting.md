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

### 3. NaN Loss

**Causes:**
- Gradient explosion
- Learning rate too high
- Numerical instability

**Solutions:**
```json
{
  "max_grad_norm": 0.5,  // Clip gradients
  "learning_rate": 1e-4,  // Lower LR
  "use_fp32": true  // Disable FP16 temporarily
}
```

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
