# Appendix D: Code Structure

Complete map of the micro-Omni codebase.

---

## Directory Tree

```
micro-Omni/
├── omni/                          # Core model library
│   ├── thinker.py                 # Language model (Thinker)
│   ├── audio_encoder.py           # Audio encoder
│   ├── vision_encoder.py          # Vision encoder (ViT + CLIP)
│   ├── codec.py                   # RVQ codec + HiFi-GAN vocoder
│   ├── talker.py                  # Talker (audio token generator)
│   ├── ocr_model.py               # OCR model
│   ├── tokenizer.py               # BPE tokenizer wrapper
│   └── utils.py                   # Shared utilities
│
├── configs/                       # JSON configuration files
│   ├── thinker_tiny.json
│   ├── audio_enc.json
│   ├── vision_tiny.json
│   ├── talker.json
│   ├── vocoder.json
│   └── ocr.json
│
├── train_text.py                  # Train the Thinker LM
├── train_audio_enc.py             # Train audio encoder
├── train_vision.py                # Train vision encoder
├── train_talker.py                # Train Talker
├── train_vocoder.py               # Train HiFi-GAN vocoder
├── train_ocr.py                   # Train OCR model
├── sft_omni.py                    # Supervised fine-tuning (all modalities)
│
├── test_thinker.py                # Test scripts
├── test_audio_enc.py
├── test_vision.py
├── test_talker.py
├── test_vocoder.py
├── test_ocr.py
├── test_sft.py                    # Multimodal SFT test
│
├── infer_chat.py                  # Interactive multimodal chat
├── export.py                      # Merge checkpoints for deployment
│
├── export/
│   ├── infer_standalone.py        # Inference from exported model
│   ├── modeling_muomni.py         # HuggingFace-compatible model classes
│   ├── test_hf_text.py            # Test HF text-only export
│   └── test_hf_multimodal.py      # Test HF multimodal export
│
├── data/                          # Training data (user-provided)
│   ├── text/
│   ├── audio/
│   ├── images/
│   └── ocr/
│
├── checkpoints/                   # Saved during training
└── exported/                      # Merged model for deployment
```

---

## omni/thinker.py

The core language model. Contains all building blocks for the Thinker.

| Class / Function       | Description                                        |
|------------------------|----------------------------------------------------|
| `ThinkerLM`           | Top-level model: embedding + blocks + LM head      |
| `Block`               | Single transformer block: norm + attn + norm + ffn  |
| `Attention`           | Multi-head attention with RoPE, optional GQA        |
| `MLP`                 | Standard feed-forward: up -> activation -> down     |
| `SwiGLU`              | Gated FFN: swish(gate(x)) * up(x) -> down          |
| `MoE`                 | Mixture of Experts: router + N expert FFNs          |
| `SpikingNeuron`       | Leaky Integrate-and-Fire neuron (Arthemis)          |
| `LiquidTimeConstant`  | Adaptive time-constant neuron (Arthemis)            |

```
ThinkerLM
├── tok_emb          Embedding(vocab_size, d_model)
├── blocks[]         N x Block
│   ├── norm1        RMSNorm
│   ├── attn         Attention
│   │   ├── wq       Linear (d_model -> d_model)
│   │   ├── wk       Linear (d_model -> d_kv)
│   │   ├── wv       Linear (d_model -> d_kv)
│   │   ├── wo       Linear (d_model -> d_model)
│   │   └── rope     RoPE (from utils)
│   ├── norm2        RMSNorm
│   └── ffn          SwiGLU | MLP | MoE
├── norm_f           RMSNorm
└── lm_head          Linear (d_model -> vocab_size)
```

---

## omni/audio_encoder.py

Encodes raw audio into embeddings for the Thinker.

| Class / Function       | Description                                        |
|------------------------|----------------------------------------------------|
| `AudioEncoderTiny`    | Full encoder: mel -> conv -> transformer -> pool    |
| `ConvDown`            | Convolutional downsampling stack                    |
| `EncoderBlock`        | Transformer block specialized for audio             |
| `AttentionPooling`    | Weighted pooling over time with learned query       |

```
AudioEncoderTiny
├── mel_spec         MelSpectrogram transform
├── conv_down        ConvDown (temporal downsampling)
├── blocks[]         N x EncoderBlock
│   ├── norm1        RMSNorm
│   ├── attn         Attention
│   ├── norm2        RMSNorm
│   └── ffn          SwiGLU
├── attn_pool        AttentionPooling
└── ctc_head         Linear (d_model -> vocab_size)  [for CTC training]
```

---

## omni/vision_encoder.py

ViT-based image encoder with CLIP-style contrastive training.

| Class / Function           | Description                                    |
|----------------------------|------------------------------------------------|
| `ViTTiny`                 | Vision Transformer: patches -> blocks -> pool   |
| `TransformerTextEncoder`  | Text encoder for contrastive pairing            |
| `AttentionPooling`        | Learned query attention over patch tokens       |

```
ViTTiny
├── patch_embed       Conv2d(3, embed_dim, patch, patch)
├── pos_embed         Learnable positional embedding
├── blocks[]          N x Block
│   ├── norm1         LayerNorm
│   ├── attn          Attention
│   ├── norm2         LayerNorm
│   └── ffn           MLP
├── norm              LayerNorm
├── attn_pool         AttentionPooling
└── proj              Linear (embed_dim -> shared_dim)

TransformerTextEncoder
├── tok_emb           Embedding
├── pos_emb           Embedding
├── blocks[]          N x Block
├── norm              LayerNorm
└── proj              Linear -> shared_dim
```

---

## omni/codec.py

Audio codec (RVQ) and vocoder (HiFi-GAN) for speech synthesis.

| Class / Function           | Description                                    |
|----------------------------|------------------------------------------------|
| `RVQ`                     | Residual Vector Quantization (N codebooks)      |
| `HiFiGANVocoder`         | Neural vocoder: mel/codes -> waveform           |
| `GriffinLimVocoder`      | Baseline vocoder (no learning)                  |
| `MultiPeriodDiscriminator`| Discriminator for adversarial training          |
| `MultiScaleDiscriminator` | Multi-resolution discriminator                  |

```
RVQ
├── codebooks[]       N x Embedding(codebook_size, codebook_dim)
└── encode/decode     Quantize residuals through codebook chain

HiFiGANVocoder
├── conv_pre          Conv1d (input channels -> base channels)
├── ups[]             ConvTranspose1d (upsample stages)
├── resblocks[]       Residual blocks with dilated convolutions
└── conv_post         Conv1d (base channels -> 1)
```

---

## omni/talker.py

Generates audio tokens from the Thinker's hidden states.

| Class / Function   | Description                                          |
|--------------------|------------------------------------------------------|
| `TalkerTiny`      | Transformer that maps thinker states to audio tokens  |

```
TalkerTiny
├── proj_in          Linear (thinker_dim -> d_model)
├── blocks[]         N x Block
│   ├── norm1        RMSNorm
│   ├── attn         Attention (with RoPE)
│   ├── norm2        RMSNorm
│   └── ffn          SwiGLU
├── norm_f           RMSNorm
└── audio_head       Linear (d_model -> n_codebooks * codebook_size)
```

---

## omni/ocr_model.py

Extracts text from images using vision encoder + autoregressive decoder.

| Class / Function       | Description                                    |
|------------------------|------------------------------------------------|
| `OCRModel`            | Full OCR: vision features -> text              |
| `OCRDecoder`          | Autoregressive decoder for character generation|
| `OCRDecoderBlock`     | Decoder block with self-attn + cross-attn      |

```
OCRModel
├── vision_enc        ViTTiny (shared or separate)
├── decoder           OCRDecoder
│   ├── tok_emb       Embedding(char_vocab, decoder_dim)
│   ├── blocks[]      N x OCRDecoderBlock
│   │   ├── self_attn    Masked self-attention
│   │   ├── cross_attn   Cross-attention to image features
│   │   └── ffn          Feed-forward
│   └── head          Linear (decoder_dim -> char_vocab)
└── proj              Linear (embed_dim -> decoder_dim)
```

---

## omni/tokenizer.py

Wraps sentencepiece for BPE tokenization.

| Class / Function   | Description                                          |
|--------------------|------------------------------------------------------|
| `BPETokenizer`    | Load .model file, encode text to IDs, decode back    |

Key methods: `encode(text) -> list[int]`, `decode(ids) -> str`,
`vocab_size -> int`

---

## omni/utils.py

Shared utilities used across all components.

| Class / Function       | Description                                        |
|------------------------|----------------------------------------------------|
| `RoPE`                | Rotary Position Embedding computation               |
| `RMSNorm`             | Root Mean Square Layer Normalization                 |
| `EMA`                 | Exponential Moving Average of model parameters       |
| `CosineScheduler`     | Learning rate schedule with warmup + cosine decay    |
| `TextDataset`         | Dataset for text training (tokenized chunks)         |
| `AudioTextDataset`    | Dataset for audio-text pairs                         |
| `ImageTextDataset`    | Dataset for image-text pairs                         |
| `find_checkpoint()`   | Locate latest checkpoint in a directory              |
| `save_checkpoint()`   | Save model + optimizer + step to .pt file            |
| `load_checkpoint()`   | Load model + optimizer + step from .pt file          |
| `count_parameters()`  | Count trainable parameters                           |
| `set_seed()`          | Set random seed for reproducibility                  |

---

## Training Scripts

| Script              | Trains            | Key Config               |
|---------------------|-------------------|--------------------------|
| `train_text.py`     | ThinkerLM         | `thinker_tiny.json`      |
| `train_audio_enc.py`| AudioEncoderTiny  | `audio_enc.json`         |
| `train_vision.py`   | ViTTiny           | `vision_tiny.json`       |
| `train_talker.py`   | TalkerTiny        | `talker.json`            |
| `train_vocoder.py`  | HiFiGANVocoder    | `vocoder.json`           |
| `train_ocr.py`      | OCRModel          | `ocr.json`               |
| `sft_omni.py`       | All components    | Combined config          |

All training scripts follow the same pattern:
1. Load config JSON
2. Build model and datasets
3. Training loop with AMP, gradient clipping, cosine LR
4. Periodic validation and checkpoint saving

---

## export/modeling_muomni.py

HuggingFace-compatible model definitions for loading exported models with
`from_pretrained`.

| Class / Function              | Description                                    |
|-------------------------------|------------------------------------------------|
| `MuOmniConfig`               | Extends `PretrainedConfig`, reads config.json   |
| `MuOmniForCausalLM`          | Text-only model, loads flat-key safetensors     |
| `MuOmniMultimodalModel`      | Full multimodal model, loads prefixed-key safetensors |

`MuOmniForCausalLM` rebuilds the Thinker from config and implements
`forward()` and `generate()` for standard HuggingFace text generation.
`MuOmniMultimodalModel` extends it with audio encoder, vision encoder,
talker, vocoder, and projection layers for full multimodal inference.

---

## export/test_hf_text.py

Validates the HuggingFace text-only export. Loads the model via
`AutoModelForCausalLM.from_pretrained`, runs text prompts, and checks
perplexity, generation quality, and weight consistency. Returns scored
pass/fail.

---

## export/test_hf_multimodal.py

Validates the HuggingFace multimodal export. Loads `MuOmniMultimodalModel`
from `model_full.safetensors`, tests image, audio, and text inputs, and
checks cross-modal output consistency. Returns scored pass/fail.

---

## test_sft.py

Validates the multimodal supervised fine-tuning pipeline. Loads all component
checkpoints together and runs end-to-end inference on multimodal samples
(text + image + audio). Tests the connected system trained by `sft_omni.py`.

---

## Inference Scripts

| Script                        | Purpose                              |
|-------------------------------|--------------------------------------|
| `infer_chat.py`              | Interactive chat (all modalities)     |
| `export/infer_standalone.py` | Chat from exported safetensors model  |
| `export.py`                  | Merge checkpoints into one file       |
