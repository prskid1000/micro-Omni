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
├── model.safetensors          # Flat keys (thinker weights only, HF-compatible)
├── model_full.safetensors     # Prefixed keys (all components, multimodal)
├── modeling_muomni.py         # Custom HF model classes
├── tokenizer.model            # Sentencepiece vocab
└── config.json                # Architecture params + auto_map for HF
```

---

## Running the Export

```bash
python export.py --ckpt_dir checkpoints/ --output_dir exported/
```

The script:
1. Loads each component's latest checkpoint
2. Prefixes every parameter key with the component name
3. Saves **two** safetensors files (see below)
4. Copies `tokenizer.model` and generates `config.json`
5. Copies `modeling_muomni.py` for HuggingFace compatibility

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

## Two Safetensors Files

The export produces **two** weight files for different use cases:

### model.safetensors (flat keys, text-only)

Contains the Thinker weights with **flat keys** — no prefix. This is what
HuggingFace `from_pretrained` loads by default. Keys look like:

```
blocks.0.attn.wq.weight
blocks.0.attn.wk.weight
blocks.0.ffn.w1.weight
tok_emb.weight
lm_head.weight
```

Use this for text-only inference through the HuggingFace API.

### model_full.safetensors (prefixed keys, all components)

Contains **all** component weights with prefixed keys (see Key Prefixes table
below). This is the file used by the standalone multimodal inference script
and by `MuOmniMultimodalModel` in modeling_muomni.py.

Both files use the safetensors format — memory-mappable, zero-copy, no pickle.

### tokenizer.model

The sentencepiece BPE tokenizer file. Copied directly from the training
directory. Must be present for the standalone script to encode/decode text.

### config.json

Generated from the training configs. Contains all architecture parameters
needed to reconstruct the models **plus** the `auto_map` field that tells
HuggingFace how to discover the custom model class:

```json
{
  "model_type": "muomni",
  "auto_map": {
    "AutoModelForCausalLM": "modeling_muomni.MuOmniForCausalLM"
  },
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

The `auto_map` entry is the key that makes `from_pretrained` work — see the
HuggingFace Integration section below.

### modeling_muomni.py

The custom model definition file. Copied into the exported directory so that
HuggingFace can load the model without having the training codebase installed.
See the next section for details.

---

## HuggingFace Integration

### modeling_muomni.py

This file lives in `export/modeling_muomni.py` and is copied into the exported
directory during export. It defines three classes:

| Class                     | Purpose                                          |
|---------------------------|--------------------------------------------------|
| `MuOmniConfig`           | Extends `PretrainedConfig`, reads config.json     |
| `MuOmniForCausalLM`      | Text-only model, loads from `model.safetensors`   |
| `MuOmniMultimodalModel`  | Full multimodal, loads from `model_full.safetensors` |

**MuOmniConfig** subclasses HuggingFace's `PretrainedConfig` and sets
`model_type = "muomni"`. It reads the thinker, audio_enc, vision, and talker
sections from config.json and exposes them as attributes.

**MuOmniForCausalLM** subclasses `PreTrainedModel`. It rebuilds the Thinker
architecture from the config and implements `generate()` and `forward()` so
standard HuggingFace text generation works out of the box.

**MuOmniMultimodalModel** extends `MuOmniForCausalLM` with audio encoder,
vision encoder, talker, vocoder, and projection layers. It loads from
`model_full.safetensors` and exposes methods for multimodal inference.

### How from_pretrained Works

When you call `AutoModelForCausalLM.from_pretrained("path/to/exported")`,
HuggingFace:

1. Reads `config.json` and finds the `auto_map` field
2. `auto_map` says `"AutoModelForCausalLM": "modeling_muomni.MuOmniForCausalLM"`
3. HuggingFace downloads/loads `modeling_muomni.py` from the same directory
4. Imports `MuOmniForCausalLM` from that file
5. Calls `MuOmniForCausalLM.from_pretrained()` which loads `model.safetensors`

```
config.json              modeling_muomni.py         model.safetensors
    |                         |                          |
    |  auto_map field         |  MuOmniForCausalLM       |  flat keys
    |  points to class        |  defines architecture    |  (no prefix)
    +----------+--------------+                          |
               |                                         |
               v                                         v
   AutoModelForCausalLM.from_pretrained("exported/")
               |
               v
      Working HF model ready for .generate()
```

You must pass `trust_remote_code=True` because the model class lives in a
local Python file, not in the HuggingFace transformers library.

### Text Generation with HuggingFace

```python
from transformers import AutoModelForCausalLM
from omni.tokenizer import BPETokenizer
import torch

# Load model (discovers MuOmniForCausalLM via auto_map in config.json)
model = AutoModelForCausalLM.from_pretrained(
    "exported/", trust_remote_code=True
)
model.eval()

# Tokenize
tok = BPETokenizer("exported/tokenizer.model")
prompt = "The meaning of life is"
ids = torch.tensor([tok.encode(prompt)])

# Generate
with torch.inference_mode():
    out = model.generate(ids, max_new_tokens=50)

print(tok.decode(out[0].tolist()))
```

### Multimodal Inference with HuggingFace

```python
from export.modeling_muomni import MuOmniMultimodalModel, MuOmniConfig
from omni.tokenizer import BPETokenizer
import torch, torchaudio
from PIL import Image
from torchvision import transforms

# Load full multimodal model
config = MuOmniConfig.from_pretrained("exported/")
model = MuOmniMultimodalModel.from_pretrained(
    "exported/", config=config, trust_remote_code=True
)
model.eval()

tok = BPETokenizer("exported/tokenizer.model")

# Prepare image
img = Image.open("test.jpg")
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3),
])
img_tensor = transform(img).unsqueeze(0)

# Prepare audio
wav, sr = torchaudio.load("test.wav")
if sr != 16000:
    wav = torchaudio.transforms.Resample(sr, 16000)(wav)

# Run multimodal forward pass
prompt = "Describe this image and audio."
ids = torch.tensor([tok.encode(prompt)])

with torch.inference_mode():
    out = model.generate(ids, images=img_tensor, audio=wav, max_new_tokens=100)

print(tok.decode(out[0].tolist()))
```

### Test Scripts

Two test scripts validate the HuggingFace export:

**export/test_hf_text.py** — Loads the exported model via
`AutoModelForCausalLM.from_pretrained`, runs a set of text prompts, and
checks that perplexity and generation quality match the standalone script.
Returns a scored pass/fail.

```bash
python export/test_hf_text.py --model_dir exported/ --device cuda
```

**export/test_hf_multimodal.py** — Loads `MuOmniMultimodalModel`, feeds it
image+audio+text inputs, and validates that all modalities produce sensible
outputs. Checks vision embedding quality, audio transcription accuracy, and
text generation coherence. Returns a scored pass/fail.

```bash
python export/test_hf_multimodal.py --model_dir exported/ --device cuda
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

## HuggingFace Upload

To share your trained model on HuggingFace Hub:

```bash
pip install huggingface_hub
huggingface-cli login  # paste your HF token when prompted

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

This uploads all files in `exported/` — including `modeling_muomni.py` and
both safetensors files — to a new HuggingFace repository. After upload,
anyone can load your model with:

```python
model = AutoModelForCausalLM.from_pretrained(
    "your-username/micro-omni-tiny", trust_remote_code=True
)
```

HuggingFace will download `config.json`, `modeling_muomni.py`, and
`model.safetensors` automatically, then use the `auto_map` field to
discover and instantiate `MuOmniForCausalLM`.

---

## Deployment Checklist

```
[ ] All component checkpoints trained and saved
[ ] export.py runs without errors
[ ] exported/model.safetensors exists (flat keys, text-only)
[ ] exported/model_full.safetensors exists (prefixed keys, all components)
[ ] exported/modeling_muomni.py is present
[ ] exported/tokenizer.model is present
[ ] exported/config.json has correct architecture params + auto_map
[ ] infer_standalone.py loads and runs correctly
[ ] Text chat works
[ ] Image QA works (if vision was trained)
[ ] Audio transcription works (if audio encoder was trained)
[ ] TTS works (if talker + vocoder were trained)
[ ] OCR works (if OCR was trained)
[ ] test_hf_text.py passes (HuggingFace text generation)
[ ] test_hf_multimodal.py passes (HuggingFace multimodal)
[ ] Upload to HuggingFace Hub succeeds (optional)
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
