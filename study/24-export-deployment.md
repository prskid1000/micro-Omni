# Chapter 24: Export & Deployment

Merging trained component checkpoints into a single deployable file.

---

## Why Export?

During training, each component saves its own checkpoint independently:

```
checkpoints/
├── thinker_tiny/step_10000.pt      ~100 MB
├── audio_enc/step_8000.pt          ~5 MB
├── vision_tiny/step_12000.pt       ~15 MB
├── talker/step_6000.pt             ~20 MB
├── vocoder/step_4000.pt            ~10 MB
└── ocr/step_3000.pt                ~8 MB
```

For deployment you want **one file** with everything merged. The export
script combines all component weights into a single `model.safetensors`
using prefixed keys so each component's parameters stay namespaced.

```
exported/
├── model.safetensors     # All weights, ~150 MB
├── tokenizer.model       # Sentencepiece vocab
└── config.json           # Architecture params
```

---

## Running the Export

```bash
python export.py --ckpt_dir checkpoints/ --output_dir exported/
```

The script:
1. Loads each component's latest checkpoint
2. Prefixes every parameter key with the component name
3. Saves everything into a single safetensors file
4. Copies `tokenizer.model` and generates `config.json`

---

## Key Prefixes

Each component's weights are namespaced to avoid collisions:

| Prefix       | Component             | Example Key                          |
|--------------|-----------------------|--------------------------------------|
| `thinker.*`  | ThinkerLM             | `thinker.blocks.0.attn.wq.weight`   |
| `audio_enc.*`| AudioEncoderTiny      | `audio_enc.conv_down.0.weight`       |
| `vision.*`   | ViTTiny               | `vision.patch_embed.proj.weight`     |
| `talker.*`   | TalkerTiny            | `talker.blocks.0.attn.wq.weight`    |
| `rvq.*`      | RVQ codec             | `rvq.codebooks.0.weight`            |
| `proj_a.*`   | Audio projection      | `proj_a.linear.weight`              |
| `proj_v.*`   | Vision projection     | `proj_v.linear.weight`              |

When loading, the standalone inference script splits on the first dot to
route each key to the correct module.

---

## Output Files

### model.safetensors

The safetensors format stores tensors in a memory-mappable, zero-copy layout.
Benefits over `.pt` files:
- Faster loading (no pickle deserialization)
- Safe (no arbitrary code execution)
- Memory-efficient (mmap support)

### tokenizer.model

The sentencepiece BPE tokenizer file. Copied directly from the training
directory. Must be present for the standalone script to encode/decode text.

### config.json

Generated from the training configs. Contains all architecture parameters
needed to reconstruct the models:

```json
{
  "thinker": {
    "vocab_size": 32000,
    "d_model": 512,
    "n_layers": 8,
    "n_heads": 8,
    "ctx_len": 2048
  },
  "audio_enc": {
    "n_mels": 80,
    "d_model": 256,
    "n_layers": 4
  },
  "vision": {
    "img_size": 224,
    "patch": 16,
    "embed_dim": 384
  },
  "talker": {
    "d_model": 256,
    "n_layers": 4
  }
}
```

---

## Standalone Inference

Once exported, run inference without the training codebase:

```bash
python export/infer_standalone.py --model_dir exported/
```

The standalone script:
1. Reads `config.json` to build model architectures
2. Loads `model.safetensors` and dispatches weights by prefix
3. Loads `tokenizer.model` for text encoding/decoding
4. Runs the same interactive chat loop as `infer_chat.py`

```
+-------------------+
|  config.json      |----> Build model skeletons
+-------------------+
|  model.safetensors|----> Load weights by prefix
+-------------------+
|  tokenizer.model  |----> Encode/decode text
+-------------------+
         |
         v
  [ Interactive Chat ]
```

---

## HuggingFace Upload (Optional)

To share your trained model on HuggingFace Hub:

```bash
pip install huggingface_hub

python -c "
from huggingface_hub import HfApi
api = HfApi()
api.create_repo('your-username/micro-omni-tiny', exist_ok=True)
api.upload_folder(
    folder_path='exported/',
    repo_id='your-username/micro-omni-tiny',
    commit_message='Upload micro-Omni tiny model'
)
"
```

This uploads all files in `exported/` to a new HuggingFace repository.

---

## Deployment Checklist

```
[ ] All component checkpoints trained and saved
[ ] export.py runs without errors
[ ] exported/model.safetensors exists and has expected size
[ ] exported/tokenizer.model is present
[ ] exported/config.json has correct architecture params
[ ] infer_standalone.py loads and runs correctly
[ ] Text chat works
[ ] Image QA works (if vision was trained)
[ ] Audio transcription works (if audio encoder was trained)
[ ] TTS works (if talker + vocoder were trained)
[ ] OCR works (if OCR was trained)
```

---

## File Size Reference

| Model Size | Approximate Export Size |
|------------|------------------------|
| Tiny (25M) | ~100 MB                |
| Small (50M)| ~200 MB                |
| Base (150M)| ~600 MB                |
| Large (500M+)| ~2 GB+              |

Safetensors uses float32 by default. You can quantize to float16 before
export to halve the file size with minimal quality loss for inference.
