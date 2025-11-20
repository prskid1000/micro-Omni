# Chapter 39: Running Training - Complete Walkthrough

[← Previous: Setup Environment](38-setup-environment.md) | [Back to Index](00-INDEX.md) | [Next: Inference Examples →](40-inference-examples.md)

---

## 🚀 Complete Training Pipeline

Step-by-step guide to training μOmni from scratch.

---

## 📝 Prerequisites

✅ Environment setup complete (Chapter 38)  
✅ Data prepared (Chapter 35)  
✅ GPU available (12GB+)

---

## 🎯 Training Sequence

### Stage A: Thinker (Text-Only)

```bash
# Train language model
python train_text.py --config configs/thinker_tiny.json

# Expected time: 8-12 hours
# Expected result: Perplexity < 10
# Output: checkpoints/thinker_tiny/thinker_step_*.pt (every 1000 steps)
```

**Monitor:**
```bash
# Watch GPU
watch -n 1 nvidia-smi

# Check logs
tail -f checkpoints/thinker_tiny/training.log
```

### Stage B: Audio Encoder

```bash
python train_audio_enc.py --config configs/audio_enc_tiny.json

# Time: 6-10 hours
# Target: WER < 20%
# Output: checkpoints/audio_enc_tiny/audio_enc.pt
```

**Configuration Note:**
- If using `use_compile: true`, ensure `max_mel_length` is set appropriately
- Default: 2048 frames (~20 seconds)
- For longer audio: Increase `max_mel_length` (e.g., 6000 for 60 seconds)
- Check your dataset: `python scripts/check_mel_lengths.py --csv data/audio/production_asr.csv`

### Stage C: Vision Encoder

```bash
python train_vision.py --config configs/vision_tiny.json

# Time: 4-8 hours
# Target: Accuracy > 70%
# Output: checkpoints/vision_tiny/vision.pt
```

### Stage D: Talker + RVQ

```bash
python train_talker.py --config configs/talker_tiny.json

# Time: 10-15 hours
# Target: Intelligible speech
# Output: checkpoints/talker_tiny/talker.pt + rvq_codec.pt
```

**Configuration Note:**
- If using `use_compile: true`, ensure `max_mel_length` is set appropriately
- Talker uses different frame rate (12.5 fps with frame_ms=80) than audio encoder (100 fps)
- For 60 seconds: `max_mel_length: 750` frames
- Default: 2048 frames (covers ~164 seconds for talker)

### Stage E: Multimodal SFT

```bash
python sft_omni.py --config configs/omni_sft_tiny.json

# Time: 6-12 hours
# Target: Good multimodal Q&A
# Output: checkpoints/omni_sft_tiny/omni_final.pt
```

### Optional: HiFi-GAN Vocoder Training

```bash
# Train neural vocoder for better speech quality (optional)
python train_vocoder.py --config configs/vocoder_tiny.json

# Time: 2-4 hours (on 12GB GPU)
# Target: Natural-sounding speech
# Output: checkpoints/vocoder_tiny/hifigan.pt
# Note: Falls back to Griffin-Lim if checkpoint not available
```

**When to train:**
- After Stage D (Talker) is complete
- If you want higher quality speech output
- Griffin-Lim works fine for basic TTS, but HiFi-GAN is better

### Optional: OCR Training

```bash
# Train OCR model for text extraction from images (optional)
python train_ocr.py --config configs/ocr_tiny.json

# Time: 4-8 hours (on 12GB GPU)
# Target: Accurate text extraction from images
# Output: checkpoints/ocr_tiny/ocr.pt
# Note: Can be used with --ocr flag in inference
```

**Configuration Note:**
- If using `use_compile: true`, ensure `max_text_length` is set appropriately
- Default: 256 characters
- Adjust based on your text lengths (short: 128, long: 512)

**When to train:**
- If you need text extraction from images
- For document processing, scene text recognition
- Can be combined with multimodal understanding in Stage E

---

## 📊 Monitoring Training

### Key Metrics to Watch

**Stage A (Thinker):**
- Loss decreasing steadily
- Perplexity < 10 (target)
- No NaN/Inf values

**Stage B (Audio):**
- CTC loss decreasing
- WER improving (lower is better)
- Target: WER < 20%

**Stage C (Vision):**
- Accuracy increasing
- Loss decreasing
- Target: Accuracy > 70%

**Stage D (Talker):**
- Reconstruction error < 0.05
- Speech codes perplexity < 15
- Generated speech intelligible

**Stage E (SFT):**
- All modalities improving
- Cross-modal accuracy increasing
- Target: >60% on mixed tasks

---

## 🛠️ Resuming Training

**Automatic Resuming:** All training scripts automatically detect and resume from the latest checkpoint. Simply rerun the training command:

```bash
# Training interrupted? Just rerun - it will auto-resume!
python train_text.py --config configs/thinker_tiny.json

# The script automatically:
# ✅ Finds latest checkpoint in save_dir
# ✅ Loads all states (model, optimizer, scheduler, scaler)
# ✅ Skips already-processed samples via skip_samples
# ✅ Continues from correct epoch and batch position
# ✅ Shows accurate progress bar
```

**What happens during resume:**
1. Script scans `save_dir` for checkpoints (e.g., `thinker_step_*.pt`)
2. Selects checkpoint with highest step number
3. Loads model weights, optimizer state, scheduler state, and scaler (if using AMP)
4. Calculates `skip_samples = step * batch_size` and sets on dataset
5. Recreates DataLoader so workers pick up the new `skip_samples` value
6. Calculates starting epoch and batch index: `start_epoch = step // steps_per_epoch`, `start_batch_idx = step % steps_per_epoch`
7. Initializes progress bar at correct position
8. Training continues seamlessly from where it stopped

**Epoch completion:**
- Training continues through all epochs until `max_steps` is reached
- Model is saved at the end of each epoch for checkpointing
- Training only stops when `max_steps` is reached or manually interrupted

**Validation during resume:**
- Validation always processes the full validation set
- `skip_samples` is temporarily reset to 0 during validation
- Original `skip_samples` is restored after validation
- This ensures validation metrics are always computed on complete data

**No manual intervention needed** - resuming is fully automatic!

---

## 💡 Tips

✅ **Run stages in parallel** (if multiple GPUs)  
✅ **Start with small data** to verify pipeline  
✅ **Monitor GPU memory** with `nvidia-smi`  
✅ **Save checkpoints frequently** (every 1000 steps)  
✅ **Use tmux/screen** for long training sessions  
✅ **Automatic resuming** - just rerun training command if interrupted  
✅ **Consistent utilities** - all scripts share common checkpoint/resume logic

---

## 📦 After Training: Export for Deployment

Once all stages are complete, you can export your trained model for deployment:

```bash
# Export all components to safetensors
python export.py \
    --omni_ckpt checkpoints/omni_sft_tiny \
    --thinker_ckpt checkpoints/thinker_tiny \
    --audio_ckpt checkpoints/audio_enc_tiny \
    --vision_ckpt checkpoints/vision_tiny \
    --talker_ckpt checkpoints/talker_tiny \
    --output_dir merged_model
```

See [Chapter 46: Model Export and Deployment](46-model-export-deployment.md) for detailed instructions, or [Chapter 47: Quick Start Export](47-quick-start-export.md) for a quick reference.

---

[Continue to Chapter 40: Inference Examples →](40-inference-examples.md)

---
