# train/ — Training Entrypoints

## Scripts
| Script | Stage | Purpose |
|--------|-------|---------|
| `train_thinker.py` | A | Thinker LLM (RoPE, GQA, SwiGLU, optional MoE/MTP) |
| `train_audio_enc.py` | B | Audio Encoder (CTC, mel → embeddings) |
| `train_vision.py` | C | Vision Encoder (CLIP contrastive) |
| `train_talker.py` | D | Talker TTS (RVQ speech codes) |
| `sft_omni.py` | E | Multimodal SFT (frozen encoders, train projectors) |
| `train_vocoder.py` | F | HiFi-GAN vocoder (optional) |
| `train_ocr.py` | G | OCR encoder-decoder (optional) |
| `tune.py` | All | HP tuning via Optuna (TPE + ASHA pruning) |

## Usage
```bash
# Training
python -m train.train_thinker --config configs/synthetic_thinker.json

# HP Tuning (any stage)
python -m train.tune --stage A --n_trials 30 --max_steps 2000
python -m train.tune --stage E --n_trials 20 --params '["lr","wd","proj_lr_mult"]'
```

## HP Tuning
`tune.py` is a generic Optuna wrapper that works for all 7 stages:
- Reads search spaces from `server/api/tuning.py` (51 unique params across all stages)
- Each trial modifies the base config, runs training for `--max_steps`, reads val_loss
- Results stored in `logs/hp_tuning_<stage>.db` (SQLite, resumable)
- Best config auto-saved to `configs/tuned_<original_config>.json`
- Can also be launched from the dashboard: HP Tuning panel → Start Tuning

Always use:
- `.venv/Scripts/python.exe`
- `PYTHONIOENCODING=utf-8`
