# μOmni — Project Instructions for Claude Code

## Hardware
- **RTX 5070 Ti** (16GB VRAM, Blackwell, compute capability 12.0)
- **Windows 11** — use Unix shell syntax (forward slashes, /dev/null not NUL)
- `torch.compile()` does NOT work (Triton/Inductor lacks Blackwell support) — keep `use_compile: false`
- BFloat16 is supported and preferred over float16

## Architecture (Thinker-Talker, ~13.9M params with synthetic config)

```
omni/thinker.py       ThinkerLM          Decoder-only LLM (RoPE, GQA, SwiGLU, optional MoE/Arthemis), generate() with temperature/top-k/top-p/repetition_penalty
omni/audio_encoder.py AudioEncoderTiny   Mel → Conv2D 8x downsample → Transformer → CTC/CLAP
omni/vision_encoder.py ViTTiny           Image patches → Transformer → CLS token (CLIP training)
omni/talker.py        TalkerTiny         AR speech code predictor (2 codebooks × 128 codes)
omni/codec.py         RVQ + Vocoders     RVQ codec (~49K), HiFi-GAN (generator+discriminators), Griffin-Lim
omni/ocr_model.py     OCRModel           ViT encoder + cross-attention decoder → character output
omni/tokenizer.py     BPETokenizer       SentencePiece BPE (256 vocab synthetic, 32K production)
omni/utils.py         Utilities          RoPE (cached), RMSNorm, EMA, streaming datasets, collate fns,
                                         checkpoint mgmt, LR scheduler, LR finder, gradient utilities
```

## Training Pipeline

```
Stage A: python train_text.py      --config configs/synthetic_thinker.json     # Thinker (cross-entropy)
Stage B: python train_audio_enc.py --config configs/synthetic_audio_enc.json   # Audio Encoder (CTC loss)
Stage C: python train_vision.py    --config configs/synthetic_vision.json      # Vision Encoder (InfoNCE)
Stage D: python train_talker.py    --config configs/synthetic_talker.json      # Talker + RVQ (cross-entropy on codes)
Stage E: python sft_omni.py        --config configs/synthetic_omni_sft.json    # Multimodal SFT (all modalities)
Stage F: python train_vocoder.py   --config configs/synthetic_vocoder.json     # HiFi-GAN vocoder (optional)
Stage G: python train_ocr.py       --config configs/synthetic_ocr.json         # OCR model (optional)
```

Dependencies: A/B/C can run in parallel. D needs A. E needs A+B+C+D. F and G are independent.

## Running Training (CLI / Claude Code)
- Always use `.venv/Scripts/python.exe` (not system `python`) — torch is in the venv
- Always set `PYTHONIOENCODING=utf-8` — Windows cp1252 breaks emoji in print statements
- **Never pipe output through `tail`** — it buffers everything and you can't monitor progress
- Write output to a log file and `tail -f` it, or just let it stream:
```bash
# CORRECT — output streams live, can check progress anytime with: tail -5 logs/stage_a.log
export PYTHONIOENCODING=utf-8
.venv/Scripts/python.exe train_text.py --config configs/synthetic_thinker.json 2>&1 | tee logs/stage_a.log

# CORRECT for background — output goes to file, check with: tail -5 logs/stage_a.log
export PYTHONIOENCODING=utf-8
.venv/Scripts/python.exe train_text.py --config configs/synthetic_thinker.json > logs/stage_a.log 2>&1

# WRONG — buffers all output, can't see progress until finished
.venv/Scripts/python.exe train_text.py --config configs/synthetic_thinker.json 2>&1 | tail -20
```
- Kill all training: `taskkill //F //IM python.exe //T`
- Check running: `tasklist //FI "IMAGENAME eq python.exe" //FO TABLE`

## Inference

```
python infer_chat.py --ckpt_dir checkpoints/thinker_tiny                                   # Text chat
python infer_chat.py --ckpt_dir checkpoints/omni_sft_tiny --image photo.jpg "describe"     # Image QA
python infer_chat.py --ckpt_dir checkpoints/omni_sft_tiny --audio_in speech.wav            # ASR
python infer_chat.py --ckpt_dir checkpoints/omni_sft_tiny --image doc.jpg --ocr            # OCR
python export.py --ckpt_dir checkpoints/ --output_dir exported/                            # Export
python export/infer_standalone.py --model_dir exported/                                    # Standalone
python export/test_hf_text.py                                                              # Test HF text model
python export/test_hf_multimodal.py                                                        # Test HF multimodal model
```

## HuggingFace Integration
- Export produces HF-compatible format — `from_pretrained("exported/")` works out of the box
- `MuOmniForCausalLM` for text-only, `MuOmniMultimodalModel` for full multimodal
- `model.safetensors` uses HF flat keys; `model_full.safetensors` has all components prefixed

## Performance Rules (MUST follow)
- `device = setup_cuda()` at top of every training script — sets cudnn.benchmark + TF32 in one call
- `use_amp: true` always — halves VRAM, 2x throughput
- `AdamW(fused=True)` on all CUDA training — fuses optimizer into single kernel (~15% faster)
- `TrainingMonitor(cfg)` replaces separate LRSpike/early-stopping — one object handles LR spike + early stopping + best weight tracking
- `opt.zero_grad(set_to_none=True)` not `opt.zero_grad()` — frees gradient memory
- `pin_memory=True` on all DataLoaders (except vocoder)
- `torch.backends.cudnn.benchmark = True` in all training scripts
- `num_workers: 2` minimum — keeps GPU fed
- **Never** add `torch.isnan()`/`torch.isinf()` checks in model forward passes — they sync the GPU
- Causal masks: use `self._causal_mask[:, :, :T, :T]` (pre-allocated buffer), never `torch.tril(torch.ones(...))`
- RoPE: frequencies are cached in `self.inv_freq` buffer with lazy cos/sin rebuild
- GQA: expand KV heads via `unsqueeze().expand().reshape()` (zero-copy), never `repeat_interleave`
- MoE: sorted batched dispatch (sort tokens by expert ID, process in one loop), never nested Python loops
- RVQ: use `torch.cdist` for codebook distance, never broadcasting `(residual[:,None,:] - code[None,:,:])`
- Gradient clipping: `clip_gradients()` returns the norm — check that directly, don't call `check_gradient_explosion()` separately
- `use_compile: false` on this machine (RTX 5070 Ti / Blackwell)

## Code Conventions
- Pre-norm architecture: `x = x + sublayer(norm(x))`
- RMSNorm everywhere (not LayerNorm), except ViT which uses nn.TransformerEncoderLayer
- `register_buffer(..., persistent=False)` for non-saveable tensors (masks, RoPE caches)
- Type hints on all model `__init__` and `forward` methods
- Checkpoints: `{model_name}.pt` (dict with model + optimizer + scheduler + scaler + monitor states)
- Metadata: `{model_name}_metadata.json` (step, epoch, dataset stats like char_to_idx, max_mel_length)
- Config: `config.json` saved to checkpoint dir during training — test scripts read ONLY from checkpoint dir
- Checkpoint dirs are self-contained: `config.json` + `{model_name}.pt` + `{model_name}_metadata.json` + tokenizer files
- Streaming datasets: all training uses `IterableDataset` with hash-based train/val split and shuffle buffer
- Config files: `synthetic_*.json` in `configs/` (small vocab/steps for quick iteration)

## Data Formats
- **Text** (Thinker): plain `.txt`, one sample per line → `data/text/production_corpus.txt`
- **ASR Audio**: CSV `wav,text` → `data/audio/production_asr.csv`
- **TTS Audio**: CSV `text,wav` (REVERSED from ASR!) → `data/audio/production_tts.csv`
- **Images**: JSON `[{image, caption}]` → `data/images/production_annotations.json`
- **OCR**: CSV `image,text` → `data/ocr/production_ocr.csv`
- Download scripts require `--combine` flag to produce production_* files

## Key Config Values (synthetic configs, RTX 5070 Ti)
- Thinker: d=128, layers=4, heads=4, d_ff=344 (8/3 x d_model), ctx=64, vocab=256, use_gqa=true kv_groups=2, batch=32, accum=1
- Audio Enc: d=128, layers=4, heads=4, d_ff=344, downsample=8x (12.5Hz), dropout=0.1, wd=0.01, batch=16, accum=1
- Vision: d=128, layers=4, heads=4, d_ff=344, embed_dim=128, temperature=0.07, batch=64, accum=1
- Talker: d=128, layers=4, heads=4, d_ff=344, codebooks=2x128, batch=16, accum=1
- Vocoder: batch=2, accum=2, max_audio_percentile=50%, shuffle_buffer=1000
- SFT: batch=4, accum=2, checkpoint_freq=500, lr=3e-5, proj_lr_mult=5.0, label_smoothing=0.1, encoders frozen

## Testing
```
python test_thinker.py --checkpoint checkpoints/thinker_tiny
python test_audio_enc.py --checkpoint checkpoints/audio_enc_tiny
python test_vision.py --checkpoint checkpoints/vision_tiny
python test_talker.py --checkpoint checkpoints/talker_tiny
python test_vocoder.py --checkpoint checkpoints/vocoder_tiny
python test_ocr.py --checkpoint checkpoints/ocr_tiny
python export/test_hf_text.py                                    # HF text model test
python export/test_hf_multimodal.py                              # HF multimodal model test
```
All tests use `torch.inference_mode()` and `torch.set_float32_matmul_precision('high')`.
