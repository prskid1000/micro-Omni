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
| `test_vision.py`    | ViTTiny            | Image-text alignment & embeddings | Embedding Quality (Norm, Diversity, Collapse), Retrieval (R@1/R@5/R@10) |
| `test_talker.py`    | TalkerTiny + RVQ   | Speech generation            | Reconstruction Error (MSE)                                |
| `test_vocoder.py`   | HiFiGANVocoder     | Mel-to-audio conversion      | Audio Norm, Audio Std                                     |

### Integration Tests

| Script              | Purpose              | What It Tests                                            |
| ------------------- | -------------------- | -------------------------------------------------------- |


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

**Purpose:** Verify vision encoder produces reasonable embeddings and evaluate image-text alignment.

**Pipeline:**

```
Image Dataset → ViTTiny → Image Projection → Image Embeddings
Caption → Text Encoder (Thinker/TransformerTextEncoder) → Text Projection → Text Embeddings
→ Contrastive Similarity & Retrieval Metrics
```

**Metrics:**

**Embedding Quality:**
- **CLS Norm Mean/Std**: Embedding magnitude statistics
- **CLS Feature Std**: Feature diversity across dimensions
- **Diversity Score**: 1 - avg pairwise similarity (higher is better)
- **Avg Pairwise Similarity**: Cosine similarity between embeddings
- **Collapse Detection**: Warning if embeddings are too similar

**Retrieval Metrics (with `--retrieval` flag):**
- **Image-to-Text R@1/R@5/R@10**: Recall at rank 1/5/10
- **Text-to-Image R@1/R@5/R@10**: Recall at rank 1/5/10
- **Average Rank**: Mean retrieval rank for correct matches

**Usage:**

```bash
# Basic embedding quality test
python test_vision.py \
    --checkpoint checkpoints/vision_tiny \
    --num_samples 100

# With retrieval metrics (requires text encoder)
python test_vision.py \
    --checkpoint checkpoints/vision_tiny \
    --num_samples 100 \
    --retrieval

# Quick test
python test_vision.py \
    --checkpoint checkpoints/vision_tiny \
    --quick

# Single image test
python test_vision.py \
    --checkpoint checkpoints/vision_tiny \
    --image path/to/image.jpg
```

**Key Features:**

- **Proper Text Encoding**: Uses trained Thinker model or TransformerTextEncoder (matches training)
- **CLIP-style Evaluation**: Measures image-text alignment via contrastive learning
- **Embedding Quality Analysis**: Detects collapse, measures diversity
- **Retrieval Performance**: R@K metrics for image-text matching

**What to Expect:**

- **Good Model**: Diversity > 0.15, CLS Norm 5-15, No collapse
- **Excellent Model**: Diversity > 0.3, R@1 > 50%, R@5 > 80%
- **Needs Training**: Diversity < 0.05 (collapsed), high pairwise similarity > 0.95

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
  Audio Reconstruction L1: 0.012345
```

**Note:** TTS generation is currently **not text-conditioned** (Talker generates autoregressively without text input). This tests reconstruction quality, not text-to-speech accuracy.

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

### Vision Encoder Multi-Component Loading

The vision encoder test script loads multiple components:

1. **ViT Model** (`vit`): Vision transformer for image encoding
2. **Image Projection** (`img_proj`): Linear → Dropout → LayerNorm
3. **Text Projection** (`text_proj`): Linear → Dropout → LayerNorm
4. **Text Encoder** (optional): Thinker model or TransformerTextEncoder
5. **Tokenizer**: For text encoding (from Thinker checkpoint)

**Configuration-Based Text Encoder:**

- If `use_thinker_for_text=true`: Loads frozen Thinker model (better quality)
- If `use_thinker_for_text=false`: Loads TransformerTextEncoder (CLIP-style)
- Retrieval metrics only available if text encoder is present

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
- Stage E (SFT): run integration checks using `infer_chat.py` and component test scripts

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

- Run integration checks with `infer_chat.py` and the component test scripts for end-to-end validation
- Test actual user workflows
- Verify multimodal combinations work

---

## 🎓 Self-Check Questions

1. Why do test scripts strip `_orig_mod` prefixes but not convert attention weights?
2. What's the difference between WER and CER in ASR evaluation?
3. Why do test scripts always sample from datasets instead of generating dummy data?
4. What metrics would indicate a well-trained OCR model?
5. How does `find_checkpoint` prioritize checkpoint files?
6. What does embedding collapse mean in vision encoder testing?
7. How does the vision encoder test script use the text encoder?

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

6. **Embedding collapse:**
   - All embeddings become very similar (avg pairwise similarity > 0.95)
   - Model hasn't learned discriminative features
   - Indicates need for more training or different hyperparameters
   - Detected via diversity score < 0.05

7. **Text encoder usage:**
   - Loads trained Thinker model (frozen) or TransformerTextEncoder
   - Encodes captions using same method as training
   - Projects text embeddings to contrastive space
   - Enables proper image-text retrieval metrics
   - Uses tokenizer from Thinker checkpoint directory

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

# Integration test
# Use component test scripts and `infer_chat.py` for integration validation
```

### Expected Test Times

- Individual component tests: 1-5 minutes (100 samples)
- Integration testing time: varies (minutes to tens of minutes depending on samples and modalities)

---

## PowerShell (Windows) Examples

If you are running on Windows PowerShell / `pwsh`, here are handy commands to run tests from the repo root.

Run a single test (example):
```powershell
python .\test_thinker.py --checkpoint checkpoints\thinker_tiny
```

Run the export safetensor check:
```powershell
python .\export\test_safetensor.py
```

Run all `test_*.py` scripts recursively (prints name then runs each):
```powershell
Get-ChildItem -Path . -Filter 'test_*.py' -Recurse | ForEach-Object {
    Write-Host "Running $($_.FullName)" -ForegroundColor Cyan
    python $($_.FullName)
}
```

Notes:
- For CI or scripted runs, prefer explicit script ordering (component tests → integration tests).
- To force CPU-only runs, add `--device cpu` to each invocation.


**Last Updated**: December 2024  
**Version**: 1.0
