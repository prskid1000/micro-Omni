[← Previous: 22-setup-environment](22-setup-environment.md) | [Index](00-INDEX.md) | [Next: 24-export-deployment →](24-export-deployment.md)

# Chapter 23: Inference & Chat

Running the trained micro-Omni model for text chat, image QA, audio
transcription, TTS, and OCR.

---

## Overview

```
+-----------+     +-----------+     +------------+
|  User     |---->|  Thinker  |---->|  Response   |
|  Input    |     |  (LM)     |     |  (text)     |
+-----------+     +-----+-----+     +------------+
  text/img/audio        |
                        v
                  +-----+-----+
                  |  Talker   |---->  Speech output
                  +-----------+
```

The inference script `infer_chat.py` is the unified entry point. It loads all
components (thinker, audio encoder, vision encoder, talker, vocoder, OCR) and
dispatches based on input type.

---

## Quick Start

```bash
python -m test.infer_chat --ckpt checkpoints/
```

This starts an interactive chat session. The script auto-detects available
component checkpoints in the given directory.

---

## Modes of Operation

### Text Chat

Type any text at the prompt and receive a generated response:

```
You: What is the capital of France?
Bot: The capital of France is Paris...
```

The thinker model generates tokens autoregressively. Temperature, top-k, and
top-p sampling can be adjusted (defaults are usually fine).

### Image Question Answering

Provide an image path, then ask a question about it:

```
You: [image: data/images/cat.jpg] What animal is in this picture?
Bot: The image shows a cat sitting on a windowsill...
```

Pipeline:
1. Image is loaded and preprocessed (resize, normalize)
2. Vision encoder produces image embeddings
3. Embeddings are projected into thinker's token space
4. Thinker generates a text answer conditioned on image + question

### Audio Transcription

Provide an audio file path to get a transcription:

```
You: [audio: data/audio/speech.wav]
Bot: [Transcription] Hello, welcome to the demo...
```

Pipeline:
1. Audio is loaded and converted to mel spectrogram
2. Audio encoder produces embeddings with CTC alignment
3. Thinker decodes text from the audio representation

### Text-to-Speech (TTS)

When the model generates a response, it can also produce speech:

```
You: Say hello in a cheerful tone.
Bot: Hello there! [audio output saved to output.wav]
```

Pipeline:
1. Thinker generates text tokens + audio tokens
2. Talker converts audio tokens to codec codes
3. Vocoder (HiFi-GAN) synthesizes waveform from codec codes

### OCR (Optical Character Recognition)

Extract text from images containing text:

```
You: [ocr: data/images/document.png]
Bot: [OCR Result] Invoice #12345. Date: 2024-01-15...
```

Pipeline:
1. Image is preprocessed and split into patches
2. Vision encoder extracts patch features
3. OCR decoder generates text character by character

---

## Checkpoint Loading

The `find_checkpoint()` utility searches for checkpoints in order:

```
checkpoints/
├── model.pt                    # 1st: single merged checkpoint
├── thinker_tiny/
│   ├── step_10000.pt          # 3rd: specific step
│   └── step_5000.pt
├── audio_enc/
│   └── step_8000.pt
├── vision_tiny/
│   └── step_12000.pt
├── talker/
│   └── step_6000.pt
└── vocoder/
    └── step_4000.pt
```

Search order for each component:
1. `model.pt` in the checkpoint directory (merged checkpoint)
2. Latest `step_*.pt` by step number in the component subdirectory
3. Any `.pt` file found in the component subdirectory

If a component checkpoint is missing, that modality is disabled gracefully.
You can still chat with text even if audio/vision checkpoints are absent.

---

## Standalone Inference (Exported Models)

After exporting with `scripts/export.py` (see Chapter 24), use the standalone script:

```bash
python -m export.infer_standalone --model_dir export/
```

This loads a single `model.safetensors` file instead of separate component
checkpoints. The standalone script is self-contained and does not depend on
the training code.

Required files in `export/`:
- `model.safetensors` — all weights merged
- `tokenizer.model` — sentencepiece tokenizer
- `config.json` — model architecture parameters

---

## Speed Benchmarks

Measured on RTX 3090, Tiny model (25M params), PyTorch 2.1, CUDA 11.8:

| Task                  | Speed             | Notes                    |
|-----------------------|-------------------|--------------------------|
| Text generation       | ~30 tokens/sec    | Autoregressive decoding  |
| Image encoding        | ~50 ms/image      | Single forward pass      |
| Audio transcription   | 2x real-time      | 10s audio in ~5s         |
| TTS synthesis         | ~3x real-time     | Including vocoder        |
| OCR                   | ~100 ms/image     | Depends on text length   |

### Optimization Tips

- **torch.compile**: Add `--compile` flag for 20-40% speedup (first run is
  slow due to compilation)
- **Flash Attention**: Enabled by default when available, reduces memory and
  improves speed for long sequences
- **Half precision**: Use `--dtype float16` or `--dtype bfloat16` to halve
  memory usage with minimal quality loss
- **Batch inference**: Process multiple inputs at once for higher throughput

---

## Common Flags

```bash
python -m test.infer_chat \
    --ckpt checkpoints/          \  # Checkpoint directory
    --device cuda                \  # Device: cuda, cpu
    --dtype float16              \  # Precision: float32, float16, bfloat16
    --max_tokens 512             \  # Max tokens to generate
    --temperature 0.7            \  # Sampling temperature
    --top_k 50                   \  # Top-k sampling
    --top_p 0.9                     # Nucleus sampling
```

---

## Troubleshooting

| Problem                        | Solution                                    |
|--------------------------------|---------------------------------------------|
| `No checkpoint found`         | Check path; ensure `.pt` files exist        |
| `CUDA out of memory`          | Use `--dtype float16` or `--device cpu`     |
| Garbled audio output           | Check vocoder checkpoint is loaded          |
| Slow first inference           | Normal with `torch.compile`; subsequent runs fast |
| Image mode not working         | Verify vision checkpoint exists             |
