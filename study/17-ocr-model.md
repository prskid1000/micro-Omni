# Chapter 17: OCR Model -- Reading Text from Images

The OCR (Optical Character Recognition) model extracts text from images. While the main vision pipeline (Chapter 15) compresses an entire image into a single CLS token for general understanding, the OCR model preserves **spatial detail** -- it needs to know where individual characters are, not just what the image "means."

Think of it like the difference between glancing at a sign (vision encoder: "it is a street sign") and actually reading the sign (OCR model: "STOP" or "Main Street").

---

## Role

The OCR model is a standalone **encoder-decoder** architecture:

- **Encoder**: A ViT that converts the image into a sequence of patch embeddings (spatial features)
- **Decoder**: An autoregressive transformer that generates text characters one at a time, using cross-attention to look at the image patches

This is fundamentally different from the Thinker pipeline. The Thinker receives a single CLS token and reasons about the image at a high level. The OCR decoder receives all 196 patch tokens and can attend to specific spatial locations -- essential for reading text that may be small, rotated, or scattered across the image.

---

## Architecture Overview

```
Input image: (B, 3, 224, 224)
    |
    v
+-------------------------------+
|  ViT Encoder                  |
|  (ViTTiny, d=192, 2 layers)  |
+-------------------------------+
    |
    v
cls: (B, 1, 192)     grid: (B, 196, 192)
                             |
                             | (use grid, not CLS)
                             v
+----------------------------------------------+
|  OCR Decoder (3 layers)                      |
|                                              |
|  Input: text_ids (B, T) -- teacher forced    |
|                                              |
|  For each decoder block:                     |
|    1. Self-attention (causal, with RoPE)     |
|    2. Cross-attention (to image grid)        |
|    3. SwiGLU FFN                             |
|                                              |
+----------------------------------------------+
    |
    v
logits: (B, T, vocab_size)    character predictions
```

---

## Configuration (from `configs/ocr_tiny.json`)

### Vision Encoder (inside OCR)

| Parameter | Value | Meaning |
|-----------|-------|---------|
| `img_size` | 224 | Input image size |
| `patch` | 16 | Patch size (14x14 = 196 patches) |
| `vision_d_model` | 192 | Vision encoder dimension |
| `vision_layers` | 2 | Encoder depth (lighter than standalone ViT's 8) |
| `vision_heads` | 3 | Attention heads |
| `vision_d_ff` | 768 | Feedforward dimension |

### Text Decoder

| Parameter | Value | Meaning |
|-----------|-------|---------|
| `decoder_d_model` | 384 | Decoder dimension |
| `decoder_layers` | 3 | Decoder depth |
| `decoder_heads` | 6 | Attention heads |
| `decoder_d_ff` | 1,536 | Feedforward dimension |
| `use_gqa` | true | Grouped Query Attention |
| `use_swiglu` | true | SwiGLU activation |
| `dropout` | 0.1 | Dropout rate |

---

## The Decoder Block in Detail

Each `OCRDecoderBlock` has three sub-layers, each with its own RMSNorm and residual connection:

```
Input x: (B, T, 384)     img_features: (B, 196, 192)
    |                          |
    v                          |
+---RMSNorm---+                |
|             |                |
| Self-Attn   |  (causal, RoPE -- can only see previous characters)
| (Attention) |
|             |                |
+------+------+                |
       |                       |
       + Residual              |
       |                       |
       v                       |
+---RMSNorm---+                |
|             |                |
| Cross-Attn  |<---------------+
| (Q from     |  img_proj: Linear(192 -> 384)
|  text, K/V  |  projects image features to decoder dim
|  from image)|
|             |
+------+------+
       |
       + Residual
       |
       v
+---RMSNorm---+
|             |
|  SwiGLU FFN |
|             |
+------+------+
       |
       + Residual
       |
       v
Output: (B, T, 384)
```

### Self-Attention (Causal)

Uses the same `Attention` class from the Thinker (Chapter 13) with RoPE and GQA. Causal masking ensures character `t` can only attend to characters `0..t`, enforcing left-to-right generation order.

### Cross-Attention (to Image)

Uses PyTorch's `nn.MultiheadAttention` with:
- **Query**: from the text decoder (what character am I generating?)
- **Key/Value**: from the image encoder (where in the image should I look?)

This is the key mechanism: when generating the character "S" in "STOP", the cross-attention learns to focus on the patch containing the letter S in the image.

The image features are projected from 192-dim (vision encoder output) to 384-dim (decoder dimension) by `img_proj: Linear(192, 384)`.

### SwiGLU FFN

Same as the Thinker: `gate_proj`, `up_proj`, `down_proj` with swish gating.

---

## Character-Level Vocabulary

Unlike the Thinker which uses a 32,000-entry BPE vocabulary, the OCR model uses a **character-level vocabulary** built dynamically from the training dataset. This typically includes:

- Lowercase and uppercase letters (a-z, A-Z)
- Digits (0-9)
- Common punctuation (. , ! ? - ' " : ; / @ # etc.)
- Special tokens: `<pad>`, `<sos>` (start of sequence), `<eos>` (end of sequence)

The vocabulary size is typically 80-128 characters, much smaller than the Thinker's BPE vocabulary. Character-level tokenization is preferred for OCR because:

1. Every individual character matters (BPE might merge "STOP" into one token, hiding character boundaries)
2. The model needs to output exact character sequences
3. Smaller vocabulary = smaller output head = fewer parameters

---

## Training: Teacher-Forced Autoregressive

Training follows the same pattern as the Talker (Chapter 16): given ground-truth characters shifted by one position, predict the next character.

```
Image: photo of a sign saying "HELLO"
Target text: <sos> H E L L O <eos>

Teacher forcing:
  Input to decoder:  [<sos>, H,   E,   L,   L,   O  ]
  Expected output:   [H,     E,   L,   L,   O,   <eos>]

Loss: cross-entropy between predicted and expected characters
```

At each position, the decoder:
1. Attends to previous characters (self-attention, causal)
2. Attends to image patches (cross-attention, bidirectional)
3. Predicts the next character

The cross-entropy loss is computed only on non-padding positions.

---

## Inference: Autoregressive Decoding

At inference time, there is no teacher forcing. The model generates one character at a time:

```
Step 0: input [<sos>]           -> predict 'H'
Step 1: input [<sos>, H]       -> predict 'E'
Step 2: input [<sos>, H, E]    -> predict 'L'
Step 3: input [<sos>, H, E, L] -> predict 'L'
Step 4: input [<sos>, H, E, L, L] -> predict 'O'
Step 5: input [<sos>, H, E, L, L, O] -> predict '<eos>'
Stop: <eos> detected
```

KV caching (same mechanism as Thinker and Talker) speeds up inference by reusing previously computed key/value pairs in self-attention.

---

## Why a Separate OCR Model?

You might wonder: why not just route OCR through the main Thinker pipeline? Three reasons:

1. **Spatial detail**: The Thinker receives only 1 CLS token per image. OCR needs all 196 patch tokens to locate individual characters.

2. **Cross-attention**: The encoder-decoder pattern with cross-attention is more effective for OCR than a decoder-only model. The decoder can selectively attend to different image regions at each character position.

3. **Character vocabulary**: OCR benefits from character-level output. The Thinker's BPE tokenizer would merge characters into subwords, making precise character extraction harder.

The tradeoff: the OCR model is a separate component that must be trained and loaded independently. It cannot benefit from the Thinker's language understanding.

---

## Full Data Flow Example

```
Input: photo of a license plate "ABC 1234"

1. Preprocess: resize to (3, 224, 224), normalize

2. ViT Encoder (2 layers):
   (3, 224, 224) -> patch_embed -> (196, 192) -> +CLS+pos -> transformer
   -> grid: (196, 192)

3. Image projection:
   img_proj: (196, 192) -> (196, 384)

4. Decoder generates:
   <sos> -> 'A'    (cross-attn focuses on left side of plate)
   A     -> 'B'    (cross-attn shifts slightly right)
   B     -> 'C'    (continues right)
   C     -> ' '    (gap between letters and numbers)
   ' '   -> '1'    (cross-attn jumps to number section)
   1     -> '2'
   2     -> '3'
   3     -> '4'
   4     -> <eos>  (end of text on plate)

Output: "ABC 1234"
```

---

## Comparison with Other Components

| Feature | OCR Model | Thinker + ViT |
|---------|-----------|--------------|
| Image tokens | 196 (full grid) | 1 (CLS only) |
| Attention to image | Cross-attention | Self-attention (mixed sequence) |
| Output vocabulary | ~128 characters | 32,000 BPE tokens |
| Best for | Exact text extraction | Image understanding, Q&A |
| Architecture | Encoder-decoder | Decoder-only |

---

## File Reference

- **Source**: `omni/ocr_model.py`
- **Config**: `configs/ocr_tiny.json`
- **Classes**: `OCRModel`, `OCRDecoder`, `OCRDecoderBlock`
- **Dependencies**: Reuses `ViTTiny` from `omni/vision_encoder.py`, `Attention` and `MLP` from `omni/thinker.py`

---

*Next: Chapter 18 covers data preparation -- how to format and organize the training data for each stage of the pipeline.*
