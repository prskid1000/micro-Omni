# train/ — Training Entrypoints

## Scripts
- `train_thinker.py` — Stage A (Thinker)
- `train_audio_enc.py` — Stage B (Audio encoder)
- `train_vision.py` — Stage C (Vision encoder)
- `train_talker.py` — Stage D (Talker + RVQ)
- `sft_omni.py` — Stage E (multimodal SFT)
- `train_vocoder.py` — Stage F (HiFi-GAN vocoder, optional)
- `train_ocr.py` — Stage G (OCR, optional)

## Usage
Run from repo root, for example:

```bash
python -m train.train_thinker --config configs/synthetic_thinker.json
```

Always use:
- `.venv/Scripts/python.exe`
- `PYTHONIOENCODING=utf-8`
