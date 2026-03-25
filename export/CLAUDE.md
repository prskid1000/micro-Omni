# export/ — Model Export & Standalone Inference

## Files

| File | Purpose |
|------|---------|
| `infer_standalone.py` | Run inference from a merged `model.safetensors` file (no separate checkpoints needed) |
| `test_safetensor.py` | Validate exported safetensors — checks all component prefixes, parameter counts, dtype |

## Export Workflow
```bash
# 1. Merge all component checkpoints into one file
python export.py --ckpt_dir checkpoints/ --output_dir exported/

# 2. Test the export
python export/test_safetensor.py

# 3. Run standalone inference
python export/infer_standalone.py --model_dir exported/
```

## Merged Key Prefixes
The `model.safetensors` file uses prefixed keys to identify components:
```
thinker.*      — ThinkerLM weights
audio_enc.*    — AudioEncoderTiny weights
vision.*       — ViTTiny weights
talker.*       — TalkerTiny weights
rvq.*          — RVQ codec weights
proj_a.*       — Audio projector (Linear audio_dim → thinker_dim)
proj_v.*       — Vision projector (Linear vision_dim → thinker_dim)
vocoder.*      — HiFi-GAN generator (optional)
ocr.*          — OCR model (optional)
```

## Required Output Files
```
exported/
├── model.safetensors   ← All weights merged
├── tokenizer.model     ← SentencePiece BPE model
└── config.json         ← Architecture config for reconstruction
```

## Notes
- `export.py` lives in the project root, not in this folder
- `_orig_mod.` prefixes from `torch.compile()` are automatically stripped during export
- The `find_checkpoint()` utility tries: `model.pt` → `{name}.pt` → latest `{name}_step_*.pt`
