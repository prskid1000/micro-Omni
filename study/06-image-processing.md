[← Previous: 05-audio-processing](05-audio-processing.md) | [Index](00-INDEX.md) | [Next: 07-normalization-activations →](07-normalization-activations.md)

# Chapter 06: Images — From Pixels to Patches

## Digital Images: Height x Width x 3

A digital image is a grid of pixels. Each pixel has three numbers (0-255) for Red, Green, and Blue.

```
One pixel:  [R=142, G=87, B=201]  → a shade of purple

Full image (224 × 224):

     ┌─────────── 224 pixels ───────────┐
     │ [R,G,B] [R,G,B] [R,G,B] ... ... │
 224 │ [R,G,B] [R,G,B] [R,G,B] ... ... │
rows │ [R,G,B] [R,G,B] [R,G,B] ... ... │
     │  ...     ...     ...     ... ... │
     └──────────────────────────────────┘

Total numbers: 224 × 224 × 3 = 150,528
```

**Analogy:** A mosaic made of 50,176 tiny colored tiles, each described by three paint-mixing instructions (how much red, green, blue).

---

## Normalization: Making Pixels Model-Friendly

Raw pixel values (0-255) are too large and inconsistent for neural networks. We normalize in two steps:

### Step 1: Scale to [0, 1]

Divide every pixel value by 255:

```
[142, 87, 201]  →  [0.557, 0.341, 0.788]
```

### Step 2: Standardize with ImageNet Statistics

Subtract the mean and divide by standard deviation of each channel, using statistics computed from millions of ImageNet images:

```
Mean: [0.485, 0.456, 0.406]   (R, G, B)
Std:  [0.229, 0.224, 0.225]

normalized_R = (0.557 - 0.485) / 0.229 = 0.314
normalized_G = (0.341 - 0.456) / 0.224 = -0.513
normalized_B = (0.788 - 0.406) / 0.225 = 1.698
```

**Why?** Neural networks learn fastest when inputs are centered around zero with unit variance. Using a shared standard (ImageNet stats) means pretrained weights transfer well.

**Analogy:** Converting temperatures from Fahrenheit to a standardized scale. 72F means little to a formula — but "0.3 standard deviations above average" is immediately useful.

---

## Convolutions: Sliding Filter Detection

Before Vision Transformers, Convolutional Neural Networks (CNNs) dominated computer vision. Understanding convolutions helps appreciate what ViTs replaced.

A **convolution** slides a small filter (e.g., 3x3) across the image, computing a dot product at each position:

```
Image region:        Filter (edge detector):      Output:
┌───┬───┬───┐       ┌────┬────┬────┐
│ 10│ 10│ 10│       │ -1 │  0 │  1 │
├───┼───┼───┤   ×   ├────┼────┼────┤   = sum = 60
│ 10│ 10│ 10│       │ -1 │  0 │  1 │     (strong
├───┼───┼───┤       ├────┼────┼────┤      vertical
│ 10│ 50│ 50│       │ -1 │  0 │  1 │      edge!)
└───┴───┴───┘       └────┴────┴────┘
```

**The flashlight analogy:** Imagine scanning a dark room with a flashlight that only reveals a 3x3 patch. At each position, the filter asks a specific question ("Is there an edge here? A corner? A texture?"). Stack many filters and the network builds up from edges → textures → parts → objects.

**Limitation of CNNs:** Each filter only sees a small local region. Global understanding requires stacking many layers (local → slightly bigger → bigger → global). This is slow and indirect.

---

## Vision Transformer (ViT): Divide and Conquer

The **Vision Transformer** takes a radically different approach: chop the image into patches and treat each patch as a "word" in a sequence.

### Step 1: Divide into 16x16 Patches

```
224 × 224 image divided into 16×16 patches:

┌──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┐
│ 1│ 2│ 3│ 4│ 5│ 6│ 7│ 8│ 9│10│11│12│13│14│
├──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┤
│15│16│17│18│19│20│21│22│23│24│25│26│27│28│  14 rows
├──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┤
│  │  │  │  │  │  │  │  │  │  │  │  │  │  │  ×
├──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┤
│  │  │  │  │  │  │  │  │  │  │  │  │  │  │  14 columns
...
└──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┘

224 / 16 = 14 patches per side
14 × 14 = 196 patches total
```

Each patch is 16×16×3 = **768 numbers** (a small image tile).

**Analogy:** Cutting a photograph into 196 equal squares, like puzzle pieces. Each piece becomes one "word" that the transformer reads.

---

## Patch Embedding: Each Patch Becomes a Vector

We use a single convolution to project each patch into a `d`-dimensional vector:

```python
patch_embed = nn.Conv2d(
    in_channels=3,        # RGB
    out_channels=d,       # model dimension (192 for micro-Omni)
    kernel_size=16,       # patch size
    stride=16             # non-overlapping patches
)
```

This is equivalent to: flatten each 16x16x3 patch into a 768-dim vector, then multiply by a 768×d weight matrix. The conv just does it efficiently.

```
Patch (16×16×3 = 768 numbers)
       │
       ▼
  [Linear projection]  (768 → d)
       │
       ▼
  Patch token (d numbers)
```

After this step: **196 patches → 196 vectors of dimension d**.

---

## The CLS Token: Summarizer-in-Chief

We prepend a special learnable token called **CLS** (classification) to the sequence. It doesn't correspond to any image patch — it starts as random numbers and learns during training.

```
Before CLS:  [patch_1, patch_2, ..., patch_196]     (196 tokens)
After CLS:   [CLS, patch_1, patch_2, ..., patch_196] (197 tokens)
```

As the CLS token passes through transformer layers, it attends to all patch tokens and **aggregates information from the entire image** into a single vector.

**The class president analogy:** In a classroom of 196 students (patches), each knows about their local area of the image. The CLS token is the class president who talks to everyone and then delivers a single summary speech to represent the whole class.

After the final transformer layer, we extract **only the CLS token** as the image representation. The 196 patch tokens are discarded.

---

## Positional Embeddings: Where Each Patch Lives

Just like text transformers need position information, ViT needs to know where each patch came from in the image. Without position, a shuffled image would look identical to the original.

ViT uses **learned positional embeddings**: a matrix of shape `(197, d)` — one vector per token (including CLS) — that is **added** to the patch embeddings.

```
final_tokens = patch_embeddings + position_embeddings

Position embedding for patch 1 (top-left) learns to encode "top-left."
Position embedding for patch 196 (bottom-right) learns to encode "bottom-right."
```

Unlike text (which uses RoPE), image positions are typically learned directly because images have a fixed size and 2D structure that the model can memorize.

---

## Forward Pass: Putting It All Together

```python
# Pseudocode
patches = patch_embed(image)          # (1, 196, d)
tokens  = cat([cls_token, patches])   # (1, 197, d)
tokens  = tokens + pos_embed          # (1, 197, d)

for block in transformer_blocks:
    tokens = block(tokens)            # (1, 197, d)

output = tokens[:, 0, :]             # (1, d) ← extract CLS only
```

---

## micro-Omni ViT Configuration

| Parameter | Value |
|-----------|-------|
| Image size | 224 x 224 |
| Patch size | 16 x 16 |
| Number of patches | 196 |
| Model dimension (d) | 192 |
| Transformer layers | 8 |
| Attention heads | 3 |
| Head dimension | 64 (192 / 3) |
| Output | CLS token: shape (1, 192) |

Total sequence: 197 tokens (1 CLS + 196 patches), each of dimension 192.

---

## Full Pipeline Diagram

```
 ═══════════════════════════════════════════════════════
              IMAGE PROCESSING PIPELINE
 ═══════════════════════════════════════════════════════

 Input image: 224 × 224 × 3  (150,528 numbers)
       │
       ▼
 ┌──────────────────┐
 │  Normalize       │   Scale to [0,1], standardize
 │  (ImageNet stats)│   with mean=[.485,.456,.406]
 └────────┬─────────┘
          │
          ▼
 ┌──────────────────┐
 │  Patch Embed     │   Conv2d(3, 192, 16, 16)
 │  (16×16 patches) │   224/16 = 14 → 14×14 = 196 patches
 └────────┬─────────┘
          │
          ▼
 196 patch vectors, each dim 192
          │
          ▼
 ┌──────────────────┐
 │  Prepend CLS     │   [CLS, p1, p2, ..., p196]
 │  token           │   → 197 tokens
 └────────┬─────────┘
          │
          ▼
 ┌──────────────────┐
 │  Add Position    │   Learned embeddings (197, 192)
 │  Embeddings      │
 └────────┬─────────┘
          │
          ▼
 ┌──────────────────┐
 │  8 Transformer   │   Self-attention + FFN
 │  Blocks          │   CLS attends to all patches
 └────────┬─────────┘
          │
          ▼
 ┌──────────────────┐
 │  Extract CLS     │   tokens[:, 0, :]
 │  token           │
 └────────┬─────────┘
          │
          ▼
 Image representation: (1, 192)

 One vector that summarizes the entire image!
 ═══════════════════════════════════════════════════════
```

---

## ViT vs CNN: A Comparison

```
         CNN                              ViT
 ┌─────────────────────┐    ┌──────────────────────────┐
 │ Layer 1: sees 3×3   │    │ Layer 1: every patch sees │
 │ Layer 2: sees 5×5   │    │          every other patch│
 │ Layer 3: sees 9×9   │    │          (global from     │
 │ ...                 │    │           the start!)     │
 │ Layer N: sees whole │    │                           │
 │          image      │    │                           │
 └─────────────────────┘    └──────────────────────────┘
  Local → gradually global    Global from layer 1
  Good at textures/edges      Good at relationships
  Translation invariant       Needs more data to train
  Fixed receptive field       Flexible attention patterns
```

| Aspect | CNN | ViT |
|--------|-----|-----|
| Receptive field | Starts small, grows with depth | Global from layer 1 |
| Inductive bias | Locality, translation invariance | Minimal (learns everything from data) |
| Data efficiency | Good with small datasets | Needs more data (or pretraining) |
| Scalability | Diminishing returns at scale | Scales well with data and compute |
| Used in micro-Omni? | No (except patch embedding conv) | Yes — 8-layer ViT |

**Key insight:** ViT trades CNN's strong built-in assumptions (locality) for flexibility. With enough data, flexibility wins because the model can learn whatever patterns exist — including ones that CNNs' assumptions would miss.

---

## Summary

| Concept | What It Does | micro-Omni Setting |
|---------|-------------|-------------------|
| Patch embedding | Splits image into 16x16 patches, projects to vectors | Conv2d(3, 192, 16, 16) |
| CLS token | Learnable summary token prepended to sequence | 1 token of dim 192 |
| Position embeddings | Learned vectors added to each token | (197, 192) |
| ViT backbone | 8 transformer layers process all tokens | 8 layers, 3 heads |
| Output | CLS token extracted as image representation | (1, 192) |

**Key takeaway:** A Vision Transformer converts a 224x224 image (150K numbers) into a single 192-dimensional vector by treating image patches as tokens in a sequence — reusing the same transformer architecture that works for text.
