# Chapter 25: Testing & Validation

Verifying each component works correctly before deployment.

---

## Test Scripts Overview

Each component has a dedicated test script:

```
test_thinker.py      Validate language model generation
test_audio_enc.py    Validate audio encoding + CTC decoding
test_vision.py       Validate image embedding + retrieval
test_talker.py       Validate audio token generation
test_vocoder.py      Validate waveform synthesis
test_ocr.py          Validate text extraction from images
```

All test scripts follow the same pattern:
1. Load the component checkpoint
2. Run inference on sample inputs
3. Compute and report metrics
4. Exit with pass/fail status

---

## Running Tests

### General Syntax

```bash
python test_thinker.py --checkpoint checkpoints/thinker_tiny
python test_audio_enc.py --checkpoint checkpoints/audio_enc
python test_vision.py --checkpoint checkpoints/vision_tiny
python test_talker.py --checkpoint checkpoints/talker
python test_vocoder.py --checkpoint checkpoints/vocoder
python test_ocr.py --checkpoint checkpoints/ocr
```

### Common Flags

| Flag              | Default | Description                        |
|-------------------|---------|------------------------------------|
| `--checkpoint`    | required| Path to component checkpoint dir   |
| `--device`        | `cuda`  | Device to run on (`cuda` or `cpu`) |
| `--num_samples`   | `100`   | Number of test samples to evaluate |
| `--batch_size`    | `16`    | Batch size for evaluation          |
| `--verbose`       | `false` | Print individual sample results    |

---

## Component Metrics

### Thinker (Language Model)

```bash
python test_thinker.py --checkpoint checkpoints/thinker_tiny --num_samples 500
```

| Metric              | Target     | Description                          |
|---------------------|------------|--------------------------------------|
| Perplexity          | < 50       | Lower = better language modeling     |
| Generation quality  | subjective | Coherence of generated text          |
| Tokens/second       | > 20       | Generation speed                     |

The test generates completions for a set of prompts and computes perplexity
on held-out text. Sample outputs are printed for manual inspection.

### Audio Encoder

```bash
python test_audio_enc.py --checkpoint checkpoints/audio_enc --num_samples 200
```

| Metric | Target  | Description                                   |
|--------|---------|-----------------------------------------------|
| WER    | < 30%   | Word Error Rate (lower = better)              |
| CER    | < 15%   | Character Error Rate (lower = better)         |

WER and CER are computed by comparing CTC-decoded transcriptions against
ground-truth text. The test loads audio-text pairs from the test split.

```
Ground truth:  "the cat sat on the mat"
Predicted:     "the cat sat on a mat"
                                ^
WER = 1/6 = 16.7%    (1 word wrong out of 6)
CER = 1/22 = 4.5%    (1 char substitution)
```

### Vision Encoder

```bash
python test_vision.py --checkpoint checkpoints/vision_tiny --num_samples 1000
```

| Metric              | Target | Description                             |
|---------------------|--------|-----------------------------------------|
| R@1                 | > 0.1  | Recall at 1 (image-text retrieval)      |
| R@5                 | > 0.3  | Recall at 5                             |
| R@10                | > 0.5  | Recall at 10                            |
| Embedding diversity | > 0.5  | Cosine distance variance (avoid collapse)|

Retrieval metrics measure whether the correct image-text pair ranks highest
among distractors. Embedding diversity checks that the encoder produces
varied representations (not mode-collapsed).

### Talker

```bash
python test_talker.py --checkpoint checkpoints/talker --num_samples 100
```

| Metric                | Target | Description                          |
|-----------------------|--------|--------------------------------------|
| Reconstruction loss   | < 1.0  | L1 distance to target audio tokens   |
| Token accuracy        | > 60%  | Correct audio token prediction       |

The talker test feeds ground-truth thinker outputs and checks whether the
generated audio tokens match expected targets.

### Vocoder (HiFi-GAN)

```bash
python test_vocoder.py --checkpoint checkpoints/vocoder --num_samples 50
```

| Metric       | Target  | Description                              |
|--------------|---------|------------------------------------------|
| Mel loss     | < 0.5   | L1 distance in mel-spectrogram space     |
| Audio quality| listen  | Subjective quality of synthesized audio  |

The test synthesizes waveforms from mel spectrograms and compares against
ground-truth audio. Sample outputs are saved to disk for listening.

### OCR

```bash
python test_ocr.py --checkpoint checkpoints/ocr --num_samples 200
```

| Metric        | Target | Description                              |
|---------------|--------|------------------------------------------|
| CER           | < 10%  | Character Error Rate                     |
| Exact match   | > 50%  | Fraction of perfectly transcribed images |

---

## Performance: torch.inference_mode()

All test scripts use `torch.inference_mode()` for fastest execution:

```python
@torch.inference_mode()
def evaluate(model, test_data, device):
    model.eval()
    results = []
    for batch in test_data:
        batch = {k: v.to(device) for k, v in batch.items()}
        output = model(**batch)
        results.append(compute_metrics(output, batch))
    return aggregate(results)
```

`torch.inference_mode()` is faster than `torch.no_grad()` because it
disables not just gradient computation but also version counting and
autograd metadata tracking.

---

## Running All Tests

Create a simple test-all script:

```bash
#!/bin/bash
set -e

echo "=== Testing Thinker ==="
python test_thinker.py --checkpoint checkpoints/thinker_tiny --device cuda

echo "=== Testing Audio Encoder ==="
python test_audio_enc.py --checkpoint checkpoints/audio_enc --device cuda

echo "=== Testing Vision Encoder ==="
python test_vision.py --checkpoint checkpoints/vision_tiny --device cuda

echo "=== Testing Talker ==="
python test_talker.py --checkpoint checkpoints/talker --device cuda

echo "=== Testing Vocoder ==="
python test_vocoder.py --checkpoint checkpoints/vocoder --device cuda

echo "=== Testing OCR ==="
python test_ocr.py --checkpoint checkpoints/ocr --device cuda

echo "=== All tests passed ==="
```

Use `--device cpu` to run without a GPU (slower but works anywhere).

---

## Interpreting Results

```
+------------------+----------+----------+
|  Component       |  Metric  |  Status  |
+------------------+----------+----------+
|  Thinker         |  PPL 42  |  PASS    |
|  Audio Encoder   |  WER 25% |  PASS    |
|  Vision          |  R@1 0.15|  PASS    |
|  Talker          |  Acc 65% |  PASS    |
|  Vocoder         |  Mel 0.4 |  PASS    |
|  OCR             |  CER 8%  |  PASS    |
+------------------+----------+----------+
```

If a component fails:
1. Check that the checkpoint path is correct
2. Verify the test data format matches training data
3. Try more training steps if metrics are far from targets
4. Check for data leakage (test data seen during training)
