# configs/ — Training Configuration Files

JSON configs that control all training hyperparameters. **These are the source of truth** — model class defaults in .py files may differ.

## Two Variants Per Stage

| Production (`*_tiny.json`) | Synthetic (`synthetic_*.json`) |
|---|---|
| Real datasets, full training | Generated toy data, quick validation |
| `vocab_size: 32000` | `vocab_size: 1024` |
| `max_steps: 100K-3.8M` | `max_steps: 2K-5K` |
| Hours to train | Minutes to train |

## File → Training Script Mapping

| Config | Script | Stage |
|--------|--------|-------|
| `thinker_tiny.json` | `train_text.py` | A — Thinker LLM |
| `audio_enc_tiny.json` | `train_audio_enc.py` | B — Audio Encoder |
| `vision_tiny.json` | `train_vision.py` | C — Vision Encoder |
| `talker_tiny.json` | `train_talker.py` | D — Talker + RVQ |
| `vocoder_tiny.json` | `train_vocoder.py` | F — HiFi-GAN (optional) |
| `ocr_tiny.json` | `train_ocr.py` | G — OCR (optional) |
| `omni_sft_tiny.json` | `sft_omni.py` | E — Multimodal SFT |

## Key Parameters (RTX 5070 Ti optimized)

| Parameter | What It Does | Typical Values |
|-----------|-------------|----------------|
| `use_amp` | Mixed precision (float16/bfloat16) | `true` always |
| `use_compile` | torch.compile Inductor | `false` (broken on Blackwell GPUs) |
| `use_gqa` / `kv_groups` | Grouped Query Attention | `true` / `2` (Thinker, Talker) |
| `use_swiglu` | SwiGLU activation in FFN | `true` |
| `use_moe` | Mixture of Experts | `false` (optional) |
| `use_spiking` / `use_ltc` | Arthemis neuromorphic extensions | `false` (experimental) |
| `use_mtp` / `num_mtp_heads` | Multi-token prediction | `true` / `2` |
| `window_size` | Sliding window attention size | `null` (full) or integer |
| `rope_scaling_factor` | RoPE frequency scaling for context extension | `1.0` (default) |
| `label_smoothing` | Label smoothing in cross-entropy loss | `0.1` |
| `gradient_accumulation_steps` | Effective batch multiplier | 2-8 depending on stage |
| `num_workers` | DataLoader parallelism | `2` |
| `temperature` | CLIP contrastive temperature | `0.07` (vision only) |

## FFN Dimension
The FFN hidden dimension (`d_ff`) uses a ratio of 8/3 x `d_model`. For `d_model=128`, this gives `d_ff=344` (rounded to nearest even). This follows the SwiGLU convention from LLaMA.

## Common Gotchas
- Audio encoder `dropout` should be `0.1` not `0.3` (kills learning)
- Audio encoder `wd` should be `0.01` not `0.1` (causes underfitting)
- ASR CSV format is `wav,text` but TTS is `text,wav` (reversed!)
- `max_audio_length_percentile: 30` discards 70% of data — use `50+`
- Vision `temperature: 0.3` makes contrastive loss trivial — use `0.07` (CLIP standard)
- SFT `checkpoint_freq: 30` causes massive disk I/O — use `500+`
