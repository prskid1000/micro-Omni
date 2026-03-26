[← Previous: 14-audio-encoder](14-audio-encoder.md) | [Index](00-INDEX.md) | [Next: 16-talker-speech →](16-talker-speech.md)

# Chapter 15: Vision Encoder (ViT-Tiny)

The Vision Encoder is the system's eye. It takes an image and produces a compact embedding that the Thinker can reason about. Unlike the audio encoder which outputs a sequence of frames, the vision encoder compresses an entire image into a **single CLS token** -- one 192-dimensional vector that captures the image's semantic meaning.

Think of it like an art critic at a museum: they look at a painting and produce a brief, dense description that captures its essence -- not a pixel-by-pixel inventory, but a high-level understanding.

---

## Role

The Vision Encoder (ViT-Tiny = Vision Transformer Tiny) converts images into embeddings through **CLIP-style contrastive training**. Rather than training with image classification labels ("this is a cat"), it learns by aligning images with their text descriptions. This produces embeddings that are naturally compatible with language -- exactly what we need for a multimodal system.

---

## Configuration (from `configs/synthetic_vision.json`)

| Parameter | Value | Meaning |
|-----------|-------|---------|
| `img_size` | 224 | Input image size (224x224 pixels) |
| `patch` | 16 | Patch size (16x16 pixels) |
| `d_model` | 128 | Vision encoder hidden dimension |
| `n_layers` | 4 | Transformer encoder layers |
| `n_heads` | 4 | Attention heads |
| `d_ff` | 344 | Feedforward dimension (8/3 x d_model) |
| `embed_dim` | 128 | CLIP shared embedding dimension |
| `dropout` | 0.1 | Dropout rate |
| `temperature` | 0.07 | Initial CLIP temperature (learnable) |

---

## Pipeline with Tensor Shapes

```
Input image: (B, 3, 224, 224)    RGB image
    |
    v
+-------------------------------------------+
|  Patch Embedding (Conv2d)                 |
|  Conv2d(3, 192, kernel=16, stride=16)     |
+-------------------------------------------+
    |
    v
(B, 192, 14, 14)              14x14 = 196 patches
    |
    v  rearrange to sequence format
(B, 196, 192)                 196 patch tokens, each 192-dim
    |
    v
+-------------------------------------------+
|  Prepend CLS token                        |
|  CLS: learnable (1, 1, 192)              |
+-------------------------------------------+
    |
    v
(B, 197, 192)                 196 patches + 1 CLS = 197 tokens
    |
    v
+-------------------------------------------+
|  Add Position Embeddings                  |
|  pos: learnable (1, 197, 192)             |
+-------------------------------------------+
    |
    v
(B, 197, 192)                 position-aware patch embeddings
    |
    v
+===================================+
|  TransformerEncoderLayer 0        |
|  (norm-first, GELU, bidirectional)|
+===================================+
    |  ... 7 more layers ...
    v
+===================================+
|  TransformerEncoderLayer 7        |
+===================================+
    |
    v
+-------------------------------------------+
|  RMSNorm (final)                          |
+-------------------------------------------+
    |
    v
(B, 197, 192)
    |
    +---> CLS token: x[:, :1, :] => (B, 1, 192)   [global image representation]
    |
    +---> Grid tokens: x[:, 1:, :] => (B, 196, 192) [spatial patch features]
```

---

## How Patches Work

As covered in Chapter 6, ViT treats an image like a sentence: it divides the image into non-overlapping patches, each of which becomes a "token."

```
224x224 image, patch size 16x16:

+----+----+----+----+----+----+----+----+----+----+----+----+----+----+
| P0 | P1 | P2 | P3 | P4 | P5 | P6 | P7 | P8 | P9 |P10 |P11 |P12 |P13 |
+----+----+----+----+----+----+----+----+----+----+----+----+----+----+
| P14| P15| ...                                                 |P27 |
+----+----+                                                     +----+
| ...                    14 x 14 = 196 patches                       |
|                        each 16x16x3 = 768 values                    |
|                        projected to 192-dim by Conv2d               |
+----+----+----+----+----+----+----+----+----+----+----+----+----+----+
|P182|P183|P184|P185|P186|P187|P188|P189|P190|P191|P192|P193|P194|P195|
+----+----+----+----+----+----+----+----+----+----+----+----+----+----+
```

The Conv2d with kernel=16 and stride=16 acts as the patch embedding: it processes each 16x16x3 patch independently and produces a 192-dim vector.

---

## The CLS Token

A special learnable token (initialized as random noise scaled by 0.02) is prepended to the patch sequence. After passing through the transformer, this CLS token has attended to all 196 patches and aggregates global image information.

Why use CLS instead of averaging all patches?
- CLS is a dedicated "summary" token that learns to aggregate during training
- It produces exactly 1 token per image, saving context length in the Thinker
- Mean-pooling would work too, but CLS is the standard ViT convention

For the multimodal pipeline, only the CLS token (projected from 192 to 384) is sent to the Thinker. The grid tokens are used by the OCR model (Chapter 17) where spatial detail matters.

---

## CLIP-Style Contrastive Training

The vision encoder is not trained with classification labels. Instead, it learns through **contrastive learning**: given a batch of (image, text) pairs, the model learns to maximize similarity between matching pairs and minimize similarity between non-matching pairs.

```
CLIP Training Setup:

     Image Branch                    Text Branch
  +---------------+              +------------------+
  | image_0.jpg   |              | "a dog on grass" |
  | image_1.jpg   |              | "sunset beach"   |
  | image_2.jpg   |              | "red sports car" |
  | image_3.jpg   |              | "city skyline"   |
  +-------+-------+              +--------+---------+
          |                               |
          v                               v
   +-------------+                +----------------+
   |  ViT-Tiny   |                | Text Encoder   |
   |  (d=192)    |                | (d=192 or      |
   |             |                |  frozen Thinker)|
   +------+------+                +--------+-------+
          |                               |
          | CLS (B, 192)                  | pooled (B, 192)
          v                               v
   +-------------+                +----------------+
   | image_proj  |                |  text_proj     |
   | 192 -> 256  |                |  192 -> 256    |
   +------+------+                +--------+-------+
          |                               |
          v                               v
     img_emb (B, 256)             txt_emb (B, 256)
          |                               |
          +---> L2 normalize <---+        |
          |                      |        |
          v                      v        v
   +------------------------------------------+
   |     Similarity Matrix (B x B)            |
   |                                          |
   |     sim[i][j] = img_emb[i] . txt_emb[j] |
   |                 / temperature             |
   |                                          |
   |     [0.9  0.1  0.0  0.1]  <- img_0      |
   |     [0.0  0.8  0.1  0.0]  <- img_1      |
   |     [0.1  0.0  0.9  0.0]  <- img_2      |
   |     [0.0  0.0  0.1  0.8]  <- img_3      |
   |                                          |
   |     diagonal = matching pairs (should    |
   |     be high), off-diagonal = non-matching|
   |     (should be low)                      |
   +------------------------------------------+
          |
          v
     InfoNCE Loss (cross-entropy on rows + columns)
```

### InfoNCE Loss

The loss treats each row and each column of the similarity matrix as a classification problem:

- **Image-to-text**: For each image (row), the correct text is on the diagonal. Apply softmax across the row and compute cross-entropy loss.
- **Text-to-image**: For each text (column), the correct image is on the diagonal. Apply softmax down the column and compute cross-entropy loss.

Total loss = average of both directions.

### Learnable Temperature

The temperature parameter controls the sharpness of the similarity distribution. Initialized at 0.07 (from the config), it is learned during training:

- Low temperature (0.01): very sharp distribution, model is very confident
- High temperature (1.0): flat distribution, model is uncertain
- The model learns the optimal temperature that balances precision and recall

```python
self.temperature = nn.Parameter(torch.tensor(0.07))  # learnable
sim = img_emb @ txt_emb.T / self.temperature.exp()
```

---

## Text Encoder Options

CLIP needs both an image encoder and a text encoder. The system offers two choices:

### Option 1: Standalone TransformerTextEncoder (default)

A dedicated transformer with:
- `d_model=192`, `n_layers=6`, `n_heads=3`, `d_ff=768`
- Causal masking (like GPT-style)
- Uses the final token embedding (EOS position) as the text representation
- `max_len=77` (following original CLIP convention)

### Option 2: Frozen Thinker

Set `use_thinker_for_text=true` in the config to use a frozen (non-trainable) copy of the Thinker as the text encoder. This is recommended when you have a well-trained Thinker, as it provides richer text representations. The Thinker's output is pooled and projected to the 256-dim CLIP embedding space.

---

## Integration with the Thinker

After CLIP training, the vision encoder is used at inference time:

```
image -> ViT-Tiny -> CLS token (1, 192) -> Linear(192, 384) -> (1, 384)
```

This single 384-dim vector is concatenated with text embeddings and fed to the Thinker. The projection from 192 to 384 is a simple linear layer added during SFT (Chapter 19).

Why does ViT use d=192 internally while the Thinker uses d=384? To save parameters. The vision encoder only needs to capture visual features; the heavier reasoning happens in the Thinker. A linear projection bridges the dimension gap.

---

## File Reference

- **Source**: `omni/vision_encoder.py`
- **Config**: `configs/synthetic_vision.json`
- **Classes**: `ViTTiny`, `TransformerTextEncoder`, `AttentionPooling`

---

*Next: Chapter 16 covers the Talker -- how the system generates speech by predicting RVQ codes and converting them back to audio.*
