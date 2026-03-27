# export/ — Model Export & Standalone Inference

## Files

| File | Purpose |
|------|---------|
| `infer_standalone.py` | Run inference from a merged `model.safetensors` file (no separate checkpoints needed) |
| `test_safetensor.py` | Validate exported safetensors — checks all component prefixes, parameter counts, dtype |
| `modeling_muomni.py` | HuggingFace-compatible model definition (954 lines) — `MuOmniForCausalLM` + `MuOmniMultimodalModel` for `from_pretrained` loading |
| `test_hf_text.py` | Test HF text-only inference via `MuOmniForCausalLM` |
| `test_hf_multimodal.py` | Test HF multimodal inference (text + audio/vision) via `MuOmniMultimodalModel` |

## Export Workflow
```bash
# 1. Merge all component checkpoints into one file
python -m scripts.export --output_dir export/

# 2. Test the export
python -m export.test_safetensor

# 3. Run standalone inference
python -m export.infer_standalone --model_dir export/
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
export/
├── model.safetensors       ← HF-compatible flat keys (for from_pretrained)
├── model_full.safetensors  ← All components with prefixed keys (thinker.*, audio_enc.*, etc.)
├── tokenizer.model         ← SentencePiece BPE model
└── config.json             ← Architecture config for reconstruction
```

## HuggingFace Integration
- `model.safetensors` uses HF flat keys (no component prefixes) — loadable via `MuOmniForCausalLM.from_pretrained("export/")`
- `model_full.safetensors` keeps all component prefixes — used by `infer_standalone.py` and `MuOmniMultimodalModel`
- `modeling_muomni.py` (954 lines) defines both `MuOmniForCausalLM` (text-only) and `MuOmniMultimodalModel` (full multimodal)

## Notes
- `scripts/export.py` lives under the `scripts/` folder, not in this folder
- `_orig_mod.` prefixes from `torch.compile()` are automatically stripped during export
- The `find_checkpoint()` utility tries: `model.pt` → `{name}.pt` → latest `{name}_step_*.pt`
