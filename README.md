# μOmni — Tiny Multimodal AI (fits 16GB VRAM)

A from-scratch **multimodal AI** stack (text + image + speech in/out) trainable on a single GPU. Based on Qwen3 Omni's Thinker-Talker architecture.

```
Image ──→ ViT Encoder ──→ Projector ──┐
Audio ──→ Audio Encoder ─→ Projector ──┤
Text  ──→ Token Embeddings ───────────┤
                                       ├──→ Thinker (LLM) ──→ Text Output
                                       └──→ Talker ──→ RVQ ──→ Vocoder ──→ Speech
```

> **~13.9M parameters** (synthetic config) | **16GB VRAM** (RTX 5070 Ti) | Reference learning repo — compact and readable | Production configs available in `.bak` files

## Benchmark Results (Synthetic Data, 2000 samples)

| Component | Metric | Score | Rating |
|-----------|--------|-------|--------|
| **Thinker** (GQA+MTP) | Top-1 Accuracy | 65.09% | EXCELLENT |
| | Top-5 Accuracy | 92.92% | EXCELLENT |
| | Top-10 Accuracy | 97.80% | EXCELLENT |
| | Perplexity | 2.71 | EXCELLENT |
| **Audio Encoder** (8x, 12.5Hz) | Val Loss | **0.0000688** | NEAR-ZERO |
| | Beam CER | 7.05% | GOOD |
| **Vision Encoder** (CLIP) | Embedding Diversity | **0.93** | EXCELLENT |
| **Talker** (FFN 8/3) | Top-5 Base | **92.33%** | EXCELLENT |
| | Top-5 Residual | **93.00%** | EXCELLENT |
| **SFT** (Multimodal) | Val Loss | 1.078 | GOOD |

**Architecture** (Qwen3.5-aligned, synthetic config):
| Feature | Setting |
|---------|---------|
| GQA | Enabled (kv_groups=2, 2:1 Q:KV ratio) |
| FFN Ratio | 8/3 × d_model (344 for d=128) — Qwen standard |
| Audio Downsample | 8x (12.5Hz) — matches Qwen3-Omni AuT |
| Multi-Token Prediction | 2 heads (predict t+2, t+3) |
| Sliding Window Attention | Infrastructure ready (window_size=0 default) |
| YaRN RoPE | Infrastructure ready (scaling_factor=1.0 default) |
| Label Smoothing | 0.1 across all training |

**Training time** (RTX 5070 Ti Laptop GPU, synthetic 2000 samples):
| Stage | Clean Run | Epochs |
|-------|-----------|--------|
| A: Thinker | ~15 min | 500 |
| B: Audio Encoder | ~5 min | 50 |
| C: Vision Encoder | ~12 min | 50 |
| D: Talker | ~8 min | 50 |
| E: SFT | ~25 min | 50 |
| F: Vocoder (optional) | ~15 min | 50 |
| G: OCR (optional) | ~10 min | 50 |
| **Total (A-E required)** | **~65 min** | |
| **Total (all stages)** | **~90 min** | |

*Note: B+C can run in parallel. First-time setup with config tuning may take 2+ hours.*

---

## Quick Start

```bash
# 1. Install
pip install -r requirements.txt

# 2. Generate synthetic data (for testing)
python scripts/make_synthetic_datasets.py

# 3. Train (any order for A/B/C, then D needs A, E needs all)
python train_text.py --config configs/thinker_tiny.json       # Stage A: Thinker LLM
python train_audio_enc.py --config configs/audio_enc_tiny.json # Stage B: Audio Encoder
python train_vision.py --config configs/vision_tiny.json       # Stage C: Vision Encoder
python train_talker.py --config configs/talker_tiny.json       # Stage D: Talker + RVQ
python sft_omni.py --config configs/omni_sft_tiny.json         # Stage E: Multimodal SFT

# 4. Inference
python infer_chat.py --ckpt_dir checkpoints/thinker_tiny                                    # Text chat
python infer_chat.py --ckpt_dir checkpoints/omni_sft_tiny --image photo.jpg "describe this"  # Image QA
python infer_chat.py --ckpt_dir checkpoints/omni_sft_tiny --audio_in speech.wav              # Audio transcription
python infer_chat.py --ckpt_dir checkpoints/omni_sft_tiny --image doc.jpg --ocr              # OCR
```

---

## File Map

### Core Models (`omni/`)

| File | Component | Params | Purpose |
|------|-----------|--------|---------|
| `thinker.py` | ThinkerLM | ~13.9M* | Decoder-only LLM — processes all modalities, generates text. Includes Block, Attention (RoPE, GQA, Sliding Window), MLP (SwiGLU), MoE, MTP, YaRN RoPE, Arthemis extensions. *13.9M for synthetic config; production configs in `.bak` files |
| `audio_encoder.py` | AudioEncoderTiny | ~2.0M | Mel spectrogram → transformer encoder. CTC mode (ASR) or contrastive mode (CLAP) |
| `vision_encoder.py` | ViTTiny | ~914K | Image patches → transformer encoder → CLS token. Also contains TransformerTextEncoder for CLIP training |
| `talker.py` | TalkerTiny | ~2.2M | Autoregressive speech code predictor — predicts RVQ codebook indices frame by frame |
| `codec.py` | RVQ + Vocoders | ~49K | RVQ (2 codebooks, 128 codes), HiFi-GAN neural vocoder (generator + MPD + MSD discriminators), Griffin-Lim fallback |
| `ocr_model.py` | OCRModel | ~2.1M | ViT encoder + cross-attention decoder for extracting text from images |
| `tokenizer.py` | BPETokenizer | — | SentencePiece BPE wrapper (encode/decode text to token IDs) |
| `utils.py` | Utilities | — | RoPE (cached), RMSNorm, EMA, LR scheduler, datasets (streaming IterableDataset), collate functions, checkpoint management, LR finder, gradient utilities |

### Training Scripts

| File | Stage | What It Trains | Loss Function |
|------|-------|----------------|---------------|
| `train_text.py` | A | Thinker LLM on text corpus | Cross-entropy (next-token prediction) |
| `train_audio_enc.py` | B | Audio encoder for ASR | CTC loss (sequence alignment) |
| `train_vision.py` | C | Vision encoder + text encoder | InfoNCE contrastive loss (CLIP-style) |
| `train_talker.py` | D | Talker + RVQ codec for TTS | Cross-entropy on RVQ codes |
| `train_vocoder.py` | F | HiFi-GAN vocoder (optional) | Adversarial + feature matching + mel L1 |
| `train_ocr.py` | G | OCR model (optional) | Cross-entropy on characters |
| `sft_omni.py` | E | All components jointly on mixed data | Cross-entropy on text tokens |

### Inference & Export

| File | Purpose |
|------|---------|
| `infer_chat.py` | Interactive multimodal inference — text chat, image QA, audio transcription, TTS, OCR, video |
| `export.py` | Merge all component checkpoints into a single `model.safetensors` file |
| `export/infer_standalone.py` | Inference from merged safetensors (no separate checkpoints needed) |
| `export/test_safetensor.py` | Validate exported safetensors file |

### Test Scripts

| File | Tests | Key Metrics |
|------|-------|-------------|
| `test_thinker.py` | Thinker LLM | Perplexity, generation quality |
| `test_audio_enc.py` | Audio encoder | WER/CER (word/character error rate) |
| `test_vision.py` | Vision encoder | R@1/R@5/R@10 retrieval, embedding diversity |
| `test_talker.py` | Talker + RVQ | Reconstruction quality |
| `test_vocoder.py` | HiFi-GAN vocoder | Mel loss, audio quality |
| `test_ocr.py` | OCR model | Character accuracy, edit distance |

### Utility Scripts (`scripts/`)

| File | Purpose |
|------|---------|
| `make_synthetic_datasets.py` | Generate synthetic data for all modalities (quick testing) |
| `run_synthetic_full.py` | End-to-end: generate data → train all stages → test |
| `download_production_text.py` | Download real text corpus |
| `download_production_audio.py` | Download real audio dataset (ASR + TTS) |
| `download_production_image.py` | Download real image dataset + captions |
| `download_production_ocr.py` | Download real OCR dataset |
| `calculate_model_size.py` | Print parameter counts for all components |

### Other

| File | Purpose |
|------|---------|
| `find_lr.py` | Learning rate finder (Smith 2017 range test) |
| `CLAUDE.md` | Project instructions for Claude Code AI assistant |

---

## Configuration

Configs in `configs/` — one per training stage. Two variants:

- **Production** (`*_tiny.json`): Full training with real datasets
- **Synthetic** (`synthetic_*.json`): Quick runs with generated data (smaller vocab, fewer steps)

Key settings across all configs:

| Setting | Default | Effect |
|---------|---------|--------|
| `use_amp` | `true` | Mixed precision — halves VRAM, 2x throughput |
| `use_compile` | `false` | torch.compile — 20-50% speedup (not on RTX 50-series) |
| `use_gqa` | `true` | Grouped Query Attention — faster KV cache |
| `use_swiglu` | `true` | SwiGLU activation — better quality than GELU |
| `use_moe` | `false` | Mixture of Experts — more capacity, same compute |
| `use_mtp` | `false` | Multi-Token Prediction — predict t+2, t+3 during training |
| `window_size` | `128` | Sliding Window Attention — O(n*w) for alternating layers |
| `rope_scaling_factor` | `1.0` | YaRN RoPE — context extension beyond training length |
| `label_smoothing` | `0.1` | Label smoothing across all training for better calibration |
| `use_spiking` | `false` | Arthemis spiking attention (experimental) |
| `use_ltc` | `false` | Arthemis liquid time constants (experimental) |

---

## Datasets

### Option A: Synthetic (recommended for quick start)
```bash
python scripts/make_synthetic_datasets.py
# Full pipeline: create → train → test
python scripts/run_synthetic_full.py --num-samples 1000
```

### Option B: Real datasets (each < 5GB)
```bash
python scripts/download_production_text.py --combine
python scripts/download_production_audio.py --combine
python scripts/download_production_image.py --combine
python scripts/download_production_ocr.py --combine
```

Data formats:
- **Text**: Plain `.txt`, one sample per line
- **ASR Audio**: CSV with `wav,text` columns
- **TTS Audio**: CSV with `text,wav` columns (reversed!)
- **Images**: JSON manifest with `image` + `caption` fields
- **OCR**: CSV with `image,text` columns

---

## Performance Optimizations

### Model-Level
- **Multi-Token Prediction (MTP)** — predict t+2, t+3 during training for richer gradients
- **Sliding Window Attention** — O(n*w) complexity for alternating layers, enabling longer sequences
- **YaRN RoPE** — context extension beyond training length via rope_scaling_factor
- Cached RoPE cos/sin tables (not recomputed every forward pass)
- Pre-allocated causal masks (sliced, not recreated)
- Zero-copy GQA expansion (expand+reshape instead of repeat_interleave)
- Sorted MoE dispatch (batched, not nested Python loops)
- Efficient RVQ encoding (torch.cdist)
- No forward-pass NaN checks (removed GPU-syncing isnan/isinf)

### Training-Level
- Mixed precision (AMP float16/bfloat16)
- Label smoothing (0.1) for better calibration and reduced overconfidence
- `zero_grad(set_to_none=True)` (frees gradient memory)
- `pin_memory=True` on DataLoaders
- `cudnn.benchmark=True` (auto-tuned convolutions)
- Flash Attention (PyTorch 2.0+ scaled_dot_product_attention)
- TF32 matmul enabled globally
- Single gradient norm check (clip + threshold in one pass)
- Streaming IterableDataset (90%+ RAM reduction)

### Generation-Level
- **ThinkerLM.generate()** with temperature, top-k, top-p (nucleus), and repetition penalty
- Repetition penalty for preventing degenerate loops and improving output quality
- Top-k / top-p (nucleus) sampling for controlled diversity during text generation

---

## Testing

```bash
# Single component
python test_thinker.py --checkpoint checkpoints/thinker_tiny

# All tests (PowerShell)
Get-ChildItem -Filter 'test_*.py' -Recurse | ForEach-Object { python $_.FullName }

# Export and test
python export.py --ckpt_dir checkpoints/ --output_dir exported/
python export/test_safetensor.py
```

Flags: `--device cpu` (no GPU), `--num_samples N` (limit test size)

---

## Learning Guide

See [`study/`](study/) for a complete zero-to-master tutorial (25 chapters + 5 appendices). Covers everything from "What is AI?" to deployment, with real-life analogies and ASCII diagrams.

## License

MIT. Replace datasets with those compatible with your needs.
