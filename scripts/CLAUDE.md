# scripts/ — Data & Utility Scripts

## Files

| Script | Purpose | Key Flags |
|--------|---------|-----------|
| `generate_synthetic_data.py` | Generate synthetic data for all modalities (text, audio, images, OCR) for quick testing | `--num-samples 1000` |
| `download_production_text.py` | Download real text corpus (wikitext) | `--combine` (MANDATORY) |
| `download_production_audio.py` | Download real audio datasets (Common Voice, LJSpeech) | `--combine` (MANDATORY) |
| `download_production_image.py` | Download real image dataset (COCO) with captions | `--combine` (MANDATORY) |
| `download_production_ocr.py` | Download real OCR dataset | `--combine` (MANDATORY) |
| `calculate_model_size.py` | Print parameter counts for all model components | — |
| `export.py` | Merge checkpoints into HF-compatible export artifacts | `--output_dir export/` |
| `run_metrics_viewer.py` | Launch local server and open metrics viewer | — |

## Critical: `--combine` Flag
Download scripts without `--combine` only create per-dataset files. Training scripts expect `production_*.txt/csv/json` files which are ONLY created with `--combine`. Always use it.

## Output Locations
```
data/text/production_corpus.txt          ← Thinker training
data/audio/production_asr.csv            ← Audio encoder training (wav,text)
data/audio/production_tts.csv            ← Talker training (text,wav — reversed!)
data/images/production_annotations.json  ← Vision encoder training
data/ocr/production_ocr.csv              ← OCR training
```

## Export Script Location
`scripts/export.py` is the canonical export entrypoint. Run from repo root:

```bash
python -m scripts.export --output_dir export/
```

## Metrics Viewer
Run from repo root:

```bash
python -m scripts.run_metrics_viewer
```
