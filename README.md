
# μOmni (Tiny Qwen3-style Omni) — fits 12GB VRAM

A tiny, from-scratch **omni** stack (text + image + speech-in/out) that you can train on a single 12 GB GPU
with **small datasets** (each modality well under **5GB**). Includes:
- Minimal Qwen3-style **Thinker** (decoder-only LLM) with TM-RoPE-lite
- **Audio encoder** (AuT-Tiny) for ASR / audio understanding (12.5 Hz rate)
- **Vision encoder** (ViT-Tiny) for image features
- **RVQ codec** (2 codebooks) + **Talker** (speech code predictor)
- **Griffin-Lim** vocoder by default (no heavy TTS training required)
- Training scripts for each stage + an end-to-end **SFT/omni** trainer
- Simple **inference** CLI for text/chat, image QA, speech chat

> This is a **reference learning repo**—compact and readable. It trades SOTA quality for simplicity and VRAM thrift.

## Environment
```
pip install -r requirements.txt
```

## Datasets (< 5GB per modality)
You have two options:

### Option A: Tiny synthetic samples (included)
Run once to generate toy data that exercises the pipeline:
```
python scripts/make_synthetic_datasets.py
```
This creates:
- `data/text/tiny_corpus.txt` (~2MB)
- `data/images/{images,annotations}.json` (~20MB)
- `data/audio/{wav/*.wav, asr.csv, tts.csv}` (~15MB)

### Option B: Real small datasets (each <5GB locally)
Scripts are provided but commented with URLs (so you can choose mirrors). Examples:
- **Text**: `wikitext-2` (~35MB) or `wikitext-103` (~200MB)
- **Images**: **COCO 2017 val** images (5k imgs, ~1GB) + captions (~250MB)
- **Audio (ASR)**: a **subset** of Common Voice EN (e.g., first ~10–20 hours, ~1–3GB)
- **Audio (TTS)**: a **subset** of LJSpeech (e.g., 3k clips, ~1.7GB)

Use:
```
# edit scripts/download_small_datasets.py for paths you prefer, then
python scripts/download_small_datasets.py --modality text images audio_asr audio_tts
```
All datasets are kept under **5GB per modality** on disk via subsetting.

## Training (stages)
Train in stages to fit 12GB easily.

### A) Text LLM pretrain (Thinker)
```
python train_text.py --config configs/thinker_tiny.json
```

### B) Audio encoder (ASR) pretrain
```
python train_audio_enc.py --config configs/audio_enc_tiny.json
```

### C) Vision encoder train (captioning contrastive pretraining)
```
python train_vision.py --config configs/vision_tiny.json
```

### D) Talker (speech code predictor) + codec pretrain
```
python train_talker.py --config configs/talker_tiny.json
```

### E) Omni SFT/alignment (mix multimodal mini-batches)
```
python sft_omni.py --config configs/omni_sft_tiny.json
```

## Inference
### Text chat:
```
python infer_chat.py --ckpt_dir checkpoints/thinker_tiny
```

### Image QA / caption:
```
python infer_chat.py --ckpt_dir checkpoints/omni_sft_tiny --image path/to.jpg "describe this"
```

### Speech chat (ASR in → Thinker → Talker → Griffin-Lim)
```
python infer_chat.py --ckpt_dir checkpoints/omni_sft_tiny --audio_in path/to.wav
```

## Notes
- Default voice uses **Griffin-Lim** (no vocoder training needed).
- All configs target ~**120–140M** total params across modules and fit a **12GB** GPU with gradient accumulation + checkpointing.
- Use smaller context (`ctx=1024`) during pretrain; go `2048` for SFT if VRAM allows.

## License
MIT for this scaffold. Replace datasets with those compatible with your needs.
