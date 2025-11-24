# Chapter 49: Testing Scripts - Model Evaluation and Validation

[← Previous: OCR Model](48-ocr-model.md) | [Back to Index](00-INDEX.md) | [Next: Future Extensions →](45-future-extensions.md)

---

## 🎯 Learning Objectives

By the end of this chapter, you will understand:

- How to test and evaluate each model component
- What metrics are reported for each model type
- How to run comprehensive test suites
- How test scripts handle checkpoint loading and data sampling
- Best practices for model validation

---

## 💡 Overview

The μOmni project includes comprehensive test scripts for validating each model component. These scripts:

✅ **Load checkpoints robustly** using `find_checkpoint` utility  
✅ **Test on real data** from datasets (not dummy data)  
✅ **Report meaningful metrics** (accuracy, loss, perplexity, etc.)  
✅ **Evaluate on 100 samples** by default for statistical significance  
✅ **Handle checkpoint compatibility** (strips `_orig_mod` prefixes, handles GQA/MHA differences)

---

## 📋 Available Test Scripts

### Individual Component Tests

| Script              | Component          | Purpose                      | Metrics Reported                                          |
| ------------------- | ------------------ | ---------------------------- | --------------------------------------------------------- |
| `test_thinker.py`   | ThinkerLM          | Language model evaluation    | Loss, Perplexity, Accuracy                                |
| `test_ocr.py`       | OCRModel           | Text extraction from images  | Loss, Exact Match Rate, Character Accuracy, Edit Distance |
| `test_audio_enc.py` | AudioEncoderTiny   | Audio feature extraction     | Output Norm, Output Std                                   |
| `test_vision.py`    | ViTTiny            | Image feature extraction     | CLS Norm/Std, Grid Norm/Std                               |
| `test_talker.py`    | TalkerTiny + RVQ   | Speech generation            | Reconstruction Error (MSE)                                |
| `test_vocoder.py`   | HiFiGANVocoder     | Mel-to-audio conversion      | Audio Norm, Audio Std                                     |
| `test_asr_tts.py`   | ASR + TTS Pipeline | Round-trip speech processing | WER, CER, ASR Loss, TTS Reconstruction Error              |

### Integration Tests

| Script              | Purpose              | What It Tests                                            |
| ------------------- | -------------------- | -------------------------------------------------------- |
| `test_all_media.py` | Multimodal inference | All media types (text, image, audio) via `infer_chat.py` |

---

## 🔧 Common Features

All test scripts share these characteristics:

### 1. **Robust Checkpoint Loading**

```python
from omni.utils import find_checkpoint, strip_orig_mod

# Automatically finds checkpoints (model.pt or model_step_X.pt)
checkpoint_path, checkpoint = find_checkpoint(
    checkpoint_dir,
    "model.pt",      # Preferred final checkpoint
    "model_step_",   # Step-based checkpoint pattern
    device
)

# Strips _orig_mod prefixes from torch.compile() models
state_dict = strip_orig_mod(checkpoint["model"])
model.load_state_dict(state_dict, strict=False)
```

**Why this matters:**

- `torch.compile()` adds `_orig_mod.` prefixes to state_dict keys
- Training scripts save with these prefixes
- Inference must strip them for compatibility

### 2. **Dataset-Based Testing**

All scripts **always** pick random samples from actual datasets:

```python
from omni.utils import TextDataset, ASRDataset, OCRDataset, etc.

# Create dataset with shuffling
dataset = TextDataset(
    path="data/text",
    tokenizer=tokenizer,
    ctx=cfg.get("ctx_len", 512),
    shuffle_buffer_size=10000,
    seed=random.randint(0, 1000000)
)

# Get random samples
iterator = iter(dataset)
for i in range(num_samples):
    sample = next(iterator)  # Random sample from dataset
```

**Key points:**

- ❌ **Never** generates dummy/random data
- ✅ **Always** uses real data from training datasets
- ✅ Uses streaming datasets (memory-efficient)
- ✅ Shuffles for randomness

### 3. **Comprehensive Metrics**

Each script reports relevant metrics for its component:

**Language Models (Thinker):**

- Average Loss
- Perplexity (exp(loss))
- Token-level Accuracy

**OCR Models:**

- Average Loss
- Exact Match Rate (% of perfect transcriptions)
- Character Accuracy (% of correct characters)
- Average Edit Distance (Levenshtein distance)

**ASR Models:**

- Word Error Rate (WER)
- Character Error Rate (CER)
- Exact Match Rate
- Word/Character Accuracy

**Reconstruction Models (Talker, Vocoder):**

- Mean Squared Error (MSE)
- Mean Absolute Error (L1)
- Output statistics (norm, std)

### 4. **Standardized Arguments**

All scripts follow consistent argument patterns:

```bash
python test_<component>.py \
    --checkpoint checkpoints/<component>_tiny \
    --num_samples 100 \
    --device cuda
```

**Common arguments:**

- `--checkpoint`: Path to checkpoint directory
- `--num_samples`: Number of samples to evaluate (default: 100)
- `--device`: Device to use (cuda/cpu, default: auto-detect)

**Note:** Scripts **never** accept input file arguments. They always sample from datasets.

---

## 📊 Detailed Script Documentation

### 1. `test_thinker.py` - Language Model Testing

**Purpose:** Evaluate ThinkerLM performance on text generation tasks.

**Pipeline:**

```
Text Dataset → Tokenizer → ThinkerLM → Logits → Loss/Accuracy
```

**Metrics:**

- **Average Loss**: Cross-entropy loss on next-token prediction
- **Perplexity**: exp(loss) - measures model uncertainty
- **Accuracy**: Percentage of correctly predicted tokens

**Usage:**

```bash
python test_thinker.py \
    --checkpoint checkpoints/thinker_tiny \
    --num_samples 100
```

**Example Output:**

```
Evaluating on 100 samples...
  Processed 100/100 samples...

EVALUATION RESULTS:
Samples evaluated: 100
Average Loss: 2.3456
Perplexity: 10.4321
Accuracy: 45.67%
```

---

### 2. `test_ocr.py` - OCR Model Testing

**Purpose:** Evaluate OCR model's ability to extract text from images.

**Pipeline:**

```
Image Dataset → ViTTiny → OCRDecoder → Text Logits → CTC Decoding → Text
```

**Metrics:**

- **Average Loss**: CTC loss on character predictions
- **Exact Match Rate**: % of images with perfect transcription
- **Character Accuracy**: % of correctly predicted characters
- **Average Edit Distance**: Levenshtein distance (lower is better)

**Usage:**

```bash
python test_ocr.py \
    --checkpoint checkpoints/ocr_tiny \
    --num_samples 100
```

**Example Output:**

```
EVALUATION RESULTS:
Samples evaluated: 100
Average Loss: 0.1234
Exact Match Rate: 78.50%
Character Accuracy: 92.34%
Average Edit Distance: 1.23
```

---

### 3. `test_audio_enc.py` - Audio Encoder Testing

**Purpose:** Verify audio encoder produces reasonable embeddings.

**Pipeline:**

```
Audio Dataset → Mel Spectrogram → AudioEncoderTiny → Embeddings
```

**Metrics:**

- **Average Output Norm**: L2 norm of embeddings (should be stable)
- **Average Output Std**: Standard deviation (measures variability)

**Usage:**

```bash
python test_audio_enc.py \
    --checkpoint checkpoints/audio_enc_tiny \
    --num_samples 100
```

**Why these metrics?**

- Norm indicates embedding magnitude (shouldn't explode/vanish)
- Std indicates diversity of representations

---

### 4. `test_vision.py` - Vision Encoder Testing

**Purpose:** Verify vision encoder produces reasonable embeddings.

**Pipeline:**

```
Image Dataset → ViTTiny → CLS Token + Grid Features
```

**Metrics:**

- **Average CLS Norm/Std**: CLS token statistics
- **Average Grid Norm/Std**: Grid feature statistics

**Usage:**

```bash
python test_vision.py \
    --checkpoint checkpoints/vision_tiny \
    --num_samples 100
```

**Why separate CLS and Grid?**

- CLS token: Global image representation
- Grid features: Spatial feature map (for OCR, object detection, etc.)

---

### 5. `test_talker.py` - Talker + RVQ Testing

**Purpose:** Evaluate speech generation quality.

**Pipeline:**

```
Mel Dataset → RVQ Encode → Codes → RVQ Decode → Reconstructed Mel
```

**Metrics:**

- **Average Reconstruction Error (MSE)**: How well RVQ reconstructs mel spectrograms

**Usage:**

```bash
python test_talker.py \
    --checkpoint checkpoints/talker_tiny \
    --num_samples 100
```

**What this tests:**

- RVQ codec quality (encoding/decoding fidelity)
- Talker model loading (not generation, just reconstruction)

---

### 6. `test_vocoder.py` - Vocoder Testing

**Purpose:** Verify vocoder converts mel to audio correctly.

**Pipeline:**

```
Mel Dataset → HiFiGANVocoder → Audio Waveform
```

**Metrics:**

- **Average Audio Norm**: Audio signal magnitude
- **Average Audio Std**: Audio signal variability

**Usage:**

```bash
python test_vocoder.py \
    --checkpoint checkpoints/vocoder_tiny \
    --num_samples 100
```

**Why these metrics?**

- Ensures vocoder produces valid audio (not NaN, not all zeros)
- Checks for reasonable signal levels

---

### 7. `test_asr_tts.py` - ASR + TTS Round-Trip Testing

**Purpose:** Test full speech-to-text-to-speech pipeline.

**Pipeline:**

```
Audio → AudioEncoder + CTC → Text → Talker + RVQ + Vocoder → Audio
```

**Metrics:**

**ASR Metrics (Speech → Text):**

- Word Error Rate (WER): % of word errors
- Character Error Rate (CER): % of character errors
- Exact Match Rate: % of perfect transcriptions
- Word/Character Accuracy: Token-level correctness
- Average CTC Loss: ASR model loss

**TTS Metrics (Text → Speech):**

- TTS Success Rate: % of successful audio generations
- Audio Reconstruction MSE: How well reconstructed audio matches original
- Audio Reconstruction L1: Mean absolute error

**Usage:**

```bash
python test_asr_tts.py \
    --audio_ckpt checkpoints/audio_enc_tiny \
    --talker_ckpt checkpoints/talker_tiny \
    --num_samples 100
```

**Example Output:**

```
EVALUATION RESULTS:

📊 ASR Metrics (Speech → Text):
  Samples evaluated: 100
  Average CTC Loss: 1.2345
  Word Error Rate (WER): 15.67%
  Character Error Rate (CER): 8.90%
  Exact Match Rate: 45.00%
  Word Accuracy: 84.33%
  Character Accuracy: 91.10%

🎵 TTS Metrics (Text → Speech):
  TTS Success Rate: 95.00% (95/100)
  Audio Reconstruction MSE: 0.001234
  Audio Reconstruction L1: 0.012345
```

**Note:** TTS generation is currently **not text-conditioned** (Talker generates autoregressively without text input). This tests reconstruction quality, not text-to-speech accuracy.

---

### 8. `test_all_media.py` - Multimodal Integration Testing

**Purpose:** Test complete multimodal inference pipeline via `infer_chat.py`.

**What it tests:**

- Text-only chat
- Image + Text chat
- Audio + Text chat
- Image-only chat
- Audio-only chat
- OCR (text extraction from images)

**Pipeline:**

```
Random Samples → infer_chat.py → Success/Failure Tracking
```

**Metrics:**

- **Success Rate**: % of successful inference runs
- **Average Time**: Mean inference time per sample
- **Min/Max Time**: Time range

**Usage:**

```bash
python test_all_media.py --num_samples 100
```

**Example Output:**

```
EVALUATION SUMMARY
============================================================
Text-only          ✓ 98.50% (98/100) | Avg: 0.45s | Range: 0.32s-0.67s
Image+Text         ✓ 95.00% (95/100) | Avg: 1.23s | Range: 0.89s-2.34s
Audio+Text         ✓ 92.00% (92/100) | Avg: 1.56s | Range: 1.12s-3.45s
Image-only          ✓ 96.00% (96/100) | Avg: 1.12s | Range: 0.78s-2.11s
Audio-only          ✓ 94.00% (94/100) | Avg: 1.34s | Range: 0.98s-2.67s
OCR                 ✓ 88.00% (88/100) | Avg: 0.89s | Range: 0.56s-1.78s
------------------------------------------------------------
Overall: 94.00% success rate (563/600 samples)
```

**Key Features:**

- Tests actual `infer_chat.py` interface (not direct model calls)
- Uses file paths (not processed tensors) - appropriate for integration testing
- Memory-efficient file scanning (limits to 10k files per directory)
- Reports timing statistics

---

## 🔍 Checkpoint Loading Details

### The `find_checkpoint` Utility

All test scripts use `find_checkpoint` from `omni.utils`:

```python
def find_checkpoint(checkpoint_dir, final_name, step_pattern, device):
    """
    Find checkpoint file, prioritizing final model over step-based.

    Priority:
    1. {checkpoint_dir}/{final_name} (e.g., model.pt)
    2. Latest {checkpoint_dir}/{step_pattern}*.pt (e.g., model_step_1000.pt)
    """
```

**Why this matters:**

- Training saves final models as `model.pt` (with metadata in `model_metadata.json`)
- Tests prioritize `model.pt`, but fall back to latest step checkpoint
- Ensures tests work with both new and legacy checkpoints

### Handling `torch.compile()` Prefixes

Models trained with `torch.compile()` have `_orig_mod.` prefixes:

```python
# Training saves:
{
    "model": {
        "vision_encoder.proj._orig_mod.weight": ...,
        "vision_encoder.proj._orig_mod.bias": ...,
    }
}

# Inference expects:
{
    "vision_encoder.proj.weight": ...,
    "vision_encoder.proj.bias": ...,
}
```

**Solution:** `strip_orig_mod()` function:

```python
from omni.utils import strip_orig_mod

state_dict = strip_orig_mod(checkpoint["model"])
model.load_state_dict(state_dict, strict=False)
```

### Attention Weight Compatibility

**Important:** Test scripts **only** strip `_orig_mod` prefixes. They **do not** convert attention weights (q/k/v → qkv) because:

1. Training scripts save weights as-is (separate q/k/v for GQA, combined qkv for MHA)
2. Models are initialized with correct architecture (`use_gqa=True/False` from config)
3. State dict keys should match model architecture

**If you see size mismatch errors:**

- Check model config matches checkpoint architecture
- Ensure `use_gqa` parameter is set correctly
- Verify checkpoint was saved with same architecture

---

## 📈 Interpreting Results

### Language Model Metrics

**Loss:**

- Lower is better
- Typical range: 1.0-4.0 (depends on vocabulary size)
- < 2.0: Excellent
- 2.0-3.0: Good
- > 3.0: May need more training

**Perplexity:**

- Lower is better
- exp(loss) - measures average branching factor
- < 10: Excellent
- 10-50: Good
- > 50: May need more training

### OCR/ASR Metrics

**Exact Match Rate:**

- Higher is better (0-100%)
- > 80%: Excellent
- 50-80%: Good
- < 50%: Needs improvement

**WER/CER:**

- Lower is better (0-100%)
- < 10%: Excellent
- 10-25%: Good
- 25-50%: Moderate
- > 50%: Poor

**Edit Distance:**

- Lower is better
- Measures character-level differences
- 0 = perfect match

### Reconstruction Metrics

**MSE (Mean Squared Error):**

- Lower is better
- Measures pixel/audio value differences
- < 0.01: Excellent
- 0.01-0.1: Good
- > 0.1: May need improvement

**L1 (Mean Absolute Error):**

- Lower is better
- More robust to outliers than MSE
- Similar interpretation to MSE

---

## 🛠️ Troubleshooting

### Common Issues

**1. Checkpoint Not Found**

```
FileNotFoundError: Checkpoint not found in: checkpoints/model_tiny
```

**Solution:**

- Verify checkpoint directory exists
- Check for `model.pt` or `model_step_*.pt` files
- Ensure checkpoint path is correct

**2. Vocabulary Size Mismatch**

```
RuntimeError: size mismatch for weight: copying a param with shape torch.Size([98, 192])
from checkpoint, the shape in current model is torch.Size([99, 192]).
```

**Solution:**

- Scripts now auto-detect vocab size from checkpoint
- If still failing, check training config matches test config

**3. CUDA Out of Memory**

```
RuntimeError: CUDA out of memory
```

**Solution:**

- Reduce `--num_samples`
- Use `--device cpu` for CPU testing
- Reduce batch size in dataset (if applicable)

**4. Dataset Not Found**

```
FileNotFoundError: ASR CSV not found: data/audio/production_asr.csv
```

**Solution:**

- Verify dataset files exist
- Check config file paths
- Run data preparation scripts if needed

---

## 💡 Best Practices

### 1. **Always Test After Training**

Run component tests after each training stage:

- Stage A (Thinker): `test_thinker.py`
- Stage B (Audio): `test_audio_enc.py`
- Stage C (Vision): `test_vision.py`
- Stage D (Talker): `test_talker.py`, `test_vocoder.py`
- Stage E (SFT): `test_all_media.py`

### 2. **Use Consistent Sample Sizes**

For fair comparisons:

- Use same `--num_samples` across runs
- Default 100 is good for quick checks
- Use 1000+ for publication-quality metrics

### 3. **Monitor Metrics Over Time**

Track metrics across training:

- Save test results to logs
- Compare before/after training
- Identify regressions early

### 4. **Test on Validation Data**

For production:

- Use separate validation set
- Don't test on training data
- Report metrics on held-out data

### 5. **Integration Testing**

After individual tests pass:

- Run `test_all_media.py` for end-to-end validation
- Test actual user workflows
- Verify multimodal combinations work

---

## 🎓 Self-Check Questions

1. Why do test scripts strip `_orig_mod` prefixes but not convert attention weights?
2. What's the difference between WER and CER in ASR evaluation?
3. Why do test scripts always sample from datasets instead of generating dummy data?
4. What metrics would indicate a well-trained OCR model?
5. How does `find_checkpoint` prioritize checkpoint files?

<details>
<summary>📝 Click to see answers</summary>

1. **Why strip `_orig_mod` but not convert attention?**

   - `_orig_mod` is a PyTorch compilation artifact that must be removed
   - Attention weights (q/k/v vs qkv) depend on model architecture (GQA vs MHA)
   - Models are initialized with correct architecture, so weights should match as-is

2. **WER vs CER:**

   - WER (Word Error Rate): Word-level errors (insertions, deletions, substitutions)
   - CER (Character Error Rate): Character-level errors
   - CER is typically lower (more granular), WER is more interpretable

3. **Why sample from datasets?**

   - Tests should reflect real-world performance
   - Dummy data doesn't catch distribution shifts
   - Ensures models work on actual data distributions

4. **Well-trained OCR metrics:**

   - Exact Match Rate > 80%
   - Character Accuracy > 90%
   - Average Edit Distance < 2
   - Low CTC loss (< 0.5)

5. **Checkpoint prioritization:**
   - First: `model.pt` (final checkpoint)
   - Second: Latest `model_step_*.pt` (step-based)
   - Ensures tests work with either checkpoint type

</details>

---

## 📚 Related Chapters

- [Chapter 32: Inference Pipeline](32-inference-pipeline.md) - How inference works
- [Chapter 37: Debugging and Troubleshooting](37-debugging-troubleshooting.md) - Common issues
- [Chapter 40: Running Inference Examples](40-inference-examples.md) - Using `infer_chat.py`
- [Chapter 39: Running Training Scripts](39-running-training.md) - Training workflow

---

## 🚀 Quick Reference

### Run All Component Tests

```bash
# Individual components
python test_thinker.py --checkpoint checkpoints/thinker_tiny
python test_ocr.py --checkpoint checkpoints/ocr_tiny
python test_audio_enc.py --checkpoint checkpoints/audio_enc_tiny
python test_vision.py --checkpoint checkpoints/vision_tiny
python test_talker.py --checkpoint checkpoints/talker_tiny
python test_vocoder.py --checkpoint checkpoints/vocoder_tiny

# Round-trip ASR+TTS
python test_asr_tts.py \
    --audio_ckpt checkpoints/audio_enc_tiny \
    --talker_ckpt checkpoints/talker_tiny

# Integration test
python test_all_media.py --num_samples 100
```

### Expected Test Times

- Individual component tests: 1-5 minutes (100 samples)
- `test_asr_tts.py`: 5-15 minutes (100 samples, includes TTS generation)
- `test_all_media.py`: 10-30 minutes (100 samples per test type)

---

**Last Updated**: December 2024  
**Version**: 1.0
