# configs/ — Training Configuration Files

JSON configs that control all training hyperparameters. **These are the source of truth** — model class defaults in .py files may differ.

## File → Training Script Mapping

| Config | Script | Stage |
|--------|--------|-------|
| `synthetic_thinker.json` | `train_thinker.py` | A — Thinker LLM |
| `synthetic_audio_enc.json` | `train_audio_enc.py` | B — Audio Encoder |
| `synthetic_vision.json` | `train_vision.py` | C — Vision Encoder |
| `synthetic_talker.json` | `train_talker.py` | D — Talker + RVQ |
| `synthetic_vocoder.json` | `train_vocoder.py` | F — HiFi-GAN (optional) |
| `synthetic_ocr.json` | `train_ocr.py` | G — OCR (optional) |
| `synthetic_omni_sft.json` | `sft_omni.py` | E — Multimodal SFT |

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
| `label_smoothing` | Label smoothing in cross-entropy loss | `0.1` (including SFT) |
| `proj_lr_mult` | Projector LR multiplier vs thinker (SFT only) | `5.0` |
| `gradient_accumulation_steps` | Effective batch multiplier | 2-8 depending on stage |
| `num_workers` | DataLoader parallelism | `2` |
| `temperature` | CLIP contrastive temperature | `0.07` (vision only) |
| `use_lr_spike` | Enable LR spike on val plateau | `true` (via TrainingMonitor) |
| `use_early_stopping` | Stop when val loss plateaus | `true` for SFT, `false` for pretraining |
| `early_stopping_patience` | Evals without improvement before stop | `5` |
| `early_stopping_min_delta` | Min improvement to count as progress | `0.001` |
| `val_loss_threshold` | Max spike above checkpoint before reload | Per-stage: 0.3-10.0 |

## FFN Dimension
The FFN hidden dimension (`d_ff`) uses a ratio of 8/3 x `d_model`. For `d_model=128`, this gives `d_ff=344` (rounded to nearest even). This follows the standard SwiGLU convention used by modern LLMs.

## Common Gotchas
- Audio encoder `dropout` should be `0.1` not `0.3` (kills learning)
- Audio encoder `wd` should be `0.01` not `0.1` (causes underfitting)
- ASR CSV format is `wav,text` but TTS is `text,wav` (reversed!)
- `max_audio_length_percentile: 30` discards 70% of data — use `50+`
- Vision `temperature: 0.3` makes contrastive loss trivial — use `0.07` (CLIP standard)
- SFT `checkpoint_freq: 30` causes massive disk I/O — use `500+`
- SFT audio/vision encoders are **frozen** — they're pretrained, not in optimizer
- SFT projectors need higher LR (`proj_lr_mult: 5.0`) — they're randomly initialized
- SFT `label_smoothing` is set to `0.1` in the actual config
- All optimizers must use `fused=True` on CUDA — free 10-20% speedup
- Use `setup_cuda()` not manual `torch.backends` lines — centralized in `omni/training_utils.py`
- Use `TrainingMonitor(cfg)` not separate `LRSpike()` — handles spike + early stop + best weights
- Synthetic configs: `val_loss_threshold: 999.0` disables reload (small data noise); production uses real thresholds
- Training scripts copy config.json to checkpoint dir — test scripts read ONLY from there, never `configs/`
