# Chapter 22: Vision Encoder (ViT-Tiny)

[← Previous: Audio Encoder](21-audio-encoder.md) | [Back to Index](00-INDEX.md) | [Next: RVQ Codec →](23-codec-rvq.md)

---

## 🎯 Learning Objectives

By the end of this chapter, you will understand:

- What the Vision Encoder does and why we need it
- How Vision Transformers (ViT) work
- Patch-based image processing
- The role of the CLS token
- Complete architecture breakdown
- How it connects to the Thinker
- Training process

---

## 💡 What is the Vision Encoder?

### The Image Understanding Module

**Analogy: Looking at a Photo Album**

```
Think of processing an image like understanding a photo:

RAW IMAGE PIXELS:
224×224×3 = 150,528 numbers!
↓
Like seeing: Millions of colored dots
- Too detailed (every pixel!)
- No structure
- Hard to understand meaning

PATCHES (16×16 chunks):
196 patches, each 16×16 pixels
↓
Like seeing: Small tiles of the image
- Top-left: "orange fur"
- Top-middle: "pointy ears"
- Center: "cat face"
- Bottom: "whiskers"

VISION ENCODER OUTPUT:
Single 256-dim embedding
↓
Like understanding: "This is a cat"
- Captures MEANING, not just pixels
- Efficient (one vector for whole image!)
- Ready for reasoning (Thinker can use it)

The Vision Encoder is the INTERPRETER:
Pixels → Meaningful understanding!
```

**Why Do We Need This?**

```
Problem: Thinker can't work with raw pixels!
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Raw image issues:
❌ Too many pixels (224×224×3 = 150,528 numbers!)
❌ No structure (just RGB values)
❌ Wrong dimension (need 256, not 150,528!)
❌ Too low-level (pixels, not concepts)

Solution: Vision Encoder!
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Vision Encoder transforms:
✅ 150,528 pixels → 1 embedding (massive compression!)
✅ Low-level pixels → High-level concept
✅ 3-channel RGB → 256-dim semantic embedding
✅ Aligns with text/audio embeddings (all 256-dim)

Now Thinker can:
- Process images efficiently
- Understand meaning (not just pixels)
- Combine with text and audio seamlessly!
```

---

## 🏗️ Detailed Architecture Breakdown

### The Complete Pipeline

```
INPUT: Cat photo (224×224 pixels, RGB)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Step 1: Start with RGB image
Shape: (3, 224, 224)
- 3 channels (Red, Green, Blue)
- 224×224 pixels
- Total: 150,528 numbers!

Step 2: Divide into patches (16×16)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

WHY patches instead of pixels?
- Processing 150K pixels individually = too slow!
- Patches = natural visual units (like "words" in images)
- 16×16 patch = meaningful visual element

HOW many patches?
- Horizontal: 224 ÷ 16 = 14 patches
- Vertical: 224 ÷ 16 = 14 patches
- Total: 14 × 14 = 196 patches

Each patch:
- Size: (3, 16, 16) = 768 numbers
- Contains: Small piece of image (part of cat ear, nose, etc.)

Visual:
┌─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┐
├─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┤  ← Each square
├─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┤    is a 16×16
├─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┤    patch
├─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┤
├─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┤  14×14 = 196
├─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┤  patches total
├─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┤
├─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┤
├─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┤
├─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┤
├─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┤
├─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┤
├─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┤
└─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┘

Step 3: Patch Embedding
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Linear projection: (3×16×16) → 128 dimensions

Each patch (768 numbers) → 128-dim vector

Why? Reduce dimensionality for efficient processing!
- 768 numbers per patch → 128 (6x compression)
- Still captures all important visual info

Result: (196, 128)
- 196 patch embeddings
- 128 dimensions each

Step 4: Add CLS Token + Positional Embeddings
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

CLS Token (Classification Token):
- Special learnable token added at the beginning
- Acts as "summary" token
- Will collect information from all patches
- Think: "representative of the entire image"

Positional Embeddings:
- Add position information to each patch
- Patch 0 knows it's top-left
- Patch 195 knows it's bottom-right
- Same concept as in text transformers!

Result: (197, 128)
- 1 CLS token + 196 patch tokens = 197 total
- 128 dimensions each

Layout:
[CLS, patch₀, patch₁, patch₂, ..., patch₁₉₅]
  ↑       ↑                          ↑
special  top-left                 bottom-right

Step 5: Transformer Encoder (4 layers)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Process with attention:
- All 197 tokens attend to each other
- CLS token gathers info from all patches
- Patches share information with neighbors

4 layers of:
  - Self-attention (tokens talk to each other)
  - Feedforward network (process each token)
  - RMSNorm (stabilize)

After layer 1:
  CLS: "I see some orange and pointy shapes"
  Patch 0: "I'm orange fur"
  Patch 50: "I'm part of an ear"
  ...

After layer 4:
  CLS: "This is a cat!" ← Aggregated understanding
  Patches: Enhanced with global context

Output: (197, 128)

Step 6: Extract CLS Token
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Take only the CLS token (first position):
- CLS has gathered information from ALL patches
- Contains holistic understanding of the image
- Represents entire image in 128 dimensions!

Result: (1, 128)

Discard 196 patch tokens:
- Already served their purpose
- Information aggregated into CLS
- Only need the summary!

Step 7: Vision Projector
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Linear projection: 128 → 256 dimensions

WHY? Align with Thinker's dimension!
- Thinker expects 256-dim embeddings
- Text embeddings: 256-dim
- Audio embeddings: 256-dim
- Image embeddings: 128-dim → 256-dim ✓

Final output: (1, 256)

READY FOR THINKER! 🎉

One embedding captures the entire image!
```

### Visual Architecture

```
┌─────────────────────────────────────────┐
│  INPUT: Cat Photo                       │
│  Shape: (3, 224, 224)                   │
│  RGB image of a cat                     │
└────────────────┬────────────────────────┘
                 ↓
┌─────────────────────────────────────────┐
│  DIVIDE INTO PATCHES                    │
│  16×16 patches                          │
│  ┌───────────────────────────────────┐ │
│  │ [Patch 0] [Patch 1] ... [Patch N]│ │
│  │   16×16      16×16        16×16   │ │
│  └───────────────────────────────────┘ │
│  Total: 14×14 = 196 patches             │
└────────────────┬────────────────────────┘
                 ↓
┌─────────────────────────────────────────┐
│  PATCH EMBEDDING                        │
│  Linear projection per patch            │
│  Each (3×16×16) → 128 dims              │
│  Output: (196, 128)                     │
└────────────────┬────────────────────────┘
                 ↓
┌─────────────────────────────────────────┐
│  ADD CLS TOKEN & POSITIONAL ENCODING    │
│  ┌────────────────────────────────────┐ │
│  │ [CLS] + [Patch₀] + ... + [Patch₁₉₅]│ │
│  │   ↑         ↑                ↑    │ │
│  │ special  top-left      bottom-right│ │
│  └────────────────────────────────────┘ │
│  + Positional embeddings                │
│  Output: (197, 128)                     │
└────────────────┬────────────────────────┘
                 ↓
┌─────────────────────────────────────────┐
│  TRANSFORMER ENCODER                    │
│  (Supports Flash Attention for speed)   │
│  ┌────────────────────────────────────┐ │
│  │ Block 1: Attention + FFN + Norm   │ │
│  │  CLS gathers info from patches    │ │
│  └────────────────────────────────────┘ │
│  ┌────────────────────────────────────┐ │
│  │ Block 2: Attention + FFN + Norm   │ │
│  │  Patches share with neighbors     │ │
│  └────────────────────────────────────┘ │
│  ┌────────────────────────────────────┐ │
│  │ Block 3: Attention + FFN + Norm   │ │
│  │  Global understanding emerges     │ │
│  └────────────────────────────────────┘ │
│  ┌────────────────────────────────────┐ │
│  │ Block 4: Attention + FFN + Norm   │ │
│  │  CLS has full image understanding │ │
│  └────────────────────────────────────┘ │
│  Output: (197, 128)                     │
└────────────────┬────────────────────────┘
                 ↓
┌─────────────────────────────────────────┐
│  EXTRACT CLS TOKEN                      │
│  Take first token: CLS[0]               │
│  Discard patches (already aggregated)   │
│  Output: (1, 128)                       │
└────────────────┬────────────────────────┘
                 ↓
┌─────────────────────────────────────────┐
│  VISION PROJECTOR                       │
│  Linear: 128 dim → 256 dim             │
│  Align with Thinker's dimension!       │
└────────────────┬────────────────────────┘
                 ↓
┌─────────────────────────────────────────┐
│  OUTPUT: Image Embedding                │
│  Shape: (1, 256)                        │
│  Single token representing "cat"        │
│  Ready for Thinker to process! ✓        │
└─────────────────────────────────────────┘
```

---

## 🔍 Why Patches? Why Not Pixels?

### The Patch-Based Approach

**Analogy: Reading a Book**

```
PIXEL-BY-PIXEL (reading letter by letter):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
T-h-e- -c-a-t- -s-a-t- -o-n- -t-h-e- -m-a-t

Problems:
❌ Too slow (150,528 letters to read!)
❌ No context (each letter alone is meaningless)
❌ Expensive (process every single letter)

PATCH-BASED (reading word by word):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
The cat sat on the mat

Benefits:
✅ Much faster (6 words vs 18 letters)
✅ Natural units (words have meaning)
✅ Efficient (process meaningful chunks)

Same idea for images!
```

**Technical Benefits:**

```
Pixel-level processing (if we tried):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

224×224 = 50,176 pixels (grayscale)
With RGB: 150,528 numbers

Self-attention on 50K pixels:
- Attention matrix: 50K × 50K = 2.5 billion entries!
- Memory: 10 GB just for one layer!
- Computation: Hours per image!
- Completely impractical! ❌

Patch-level processing (what we do):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

196 patches (16×16 each)

Self-attention on 196 patches:
- Attention matrix: 196 × 196 = 38,416 entries
- Memory: ~150 KB per layer
- Computation: Milliseconds per image!
- Completely practical! ✓

Speed-up: ~256x faster!

Why 16×16 patches work:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Visual reasoning:
- 16×16 = 256 pixels per patch
- Large enough to see meaningful features:
  * Edge of an ear
  * Part of an eye
  * Bit of fur texture
- Small enough to capture details
- Natural "visual word" size

Proven effective:
- ViT (Vision Transformer) uses 16×16
- Beats CNNs on many benchmarks
- Standard in modern vision models!
```

---

## 🎯 The CLS Token: The Aggregator

### Understanding the Special CLS Token

**Analogy: Team Meeting**

```
TEAM MEETING (like transformer attention):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Attendees:
- Manager (CLS token)
- Engineer 1 (Patch 0 - top-left corner)
- Engineer 2 (Patch 1 - top edge)
- ...
- Engineer 196 (Patch 195 - bottom-right)

Layer 1 (First meeting):
Manager: "Everyone, tell me what you see"
Engineer 1: "I see orange fur"
Engineer 50: "I see a pointy shape (ear?)"
Engineer 100: "I see white whiskers"
Manager: "Hmm, gathering information..."

Layer 2 (Second meeting):
Engineers share with each other too!
Engineer 1 to Engineer 2: "I'm orange, are you?"
Engineer 50 to Engineer 51: "Pointy shape continues here"
Manager: "Okay, getting clearer picture..."

Layer 3 (Third meeting):
More information sharing
Manager: "This is starting to look like an animal"

Layer 4 (Final meeting):
Manager: "Got it! This is definitely a CAT!"

Result: Manager (CLS) has complete understanding!
We only need the manager's summary (CLS token)!
```

**Technical Explanation:**

```
Why CLS token works:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Mechanism:
1. CLS token has NO positional bias
   → Can aggregate from anywhere in image

2. Through attention:
   CLS attends to ALL patches
   → Gathers information from entire image

3. Through layers:
   Layer 1: CLS sees individual patches
   Layer 2: CLS sees patch relationships
   Layer 3: CLS understands regions
   Layer 4: CLS grasps whole image concept

4. Final CLS embedding:
   → Contains holistic understanding
   → "This is a cat with orange fur, pointy ears..."

Alternative approach (without CLS):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Could average all 196 patch embeddings:
avg = (patch₀ + patch₁ + ... + patch₁₉₅) / 196

Problems:
❌ Simple averaging loses spatial relationships
❌ No learned aggregation strategy
❌ Treats all patches equally (but some more important!)

CLS token approach:
✅ Learns optimal aggregation through attention
✅ Can weight important patches more
✅ Captures spatial relationships
✅ Proven more effective!
```

---

## 📊 Detailed Specifications

> **Note**: These are the "tiny" configuration values from `configs/vision_tiny.json`. The code defaults may differ, but config files override them.

### Architecture Parameters

```
PATCH EMBEDDING:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Input: (3, 224, 224)
Patch size: 16×16
Number of patches: 14×14 = 196
Patch flatten: 3×16×16 = 768 dims
Linear projection: 768 → 128 dims
Output: (196, 128)

CLS TOKEN & POSITIONAL EMBEDDING:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CLS token: Learnable (128 dims)
Positional embeddings: Learnable (197, 128)
Added: CLS + patches + positions
Output: (197, 128)

TRANSFORMER ENCODER:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Dimension: 128
Layers: 4
Attention heads: 2
FFN dimension: 512 (4 × 128)
Dropout: 0.1
Normalization: LayerNorm (standard ViT uses LayerNorm)

VISION PROJECTOR:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Linear: 128 → 256 (no bias)

TOTAL PARAMETERS:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Patch embedding: ~100K
Positional embeddings: ~25K
Transformer blocks: ~14M
Projector: ~33K
Total: ~914K parameters
```

### Comparison Table

| Component         | Input         | Output     | Purpose                |
| ----------------- | ------------- | ---------- | ---------------------- |
| **Patch Embed**   | (3, 224, 224) | (196, 128) | Visual tokenization    |
| **Add CLS + Pos** | (196, 128)    | (197, 128) | Aggregation + position |
| **Transformer**   | (197, 128)    | (197, 128) | Visual understanding   |
| **Extract CLS**   | (197, 128)    | (1, 128)   | Global representation  |
| **Projector**     | (1, 128)      | (1, 256)   | Dimension alignment    |

---

## 🎓 Training Process

### Pretraining Strategy

**Contrastive Learning (CLIP-style):**

```
Goal: Teach vision encoder to align images with text descriptions

Task: Image-Caption contrastive learning
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Input: Image + Caption pair
Example: [Cat photo] + "A cat sitting on a mat"

This forces the encoder to:
✅ Learn visual features (edges, textures, shapes)
✅ Understand objects and parts
✅ Align visual concepts with text descriptions
✅ Capture semantic meaning shared between vision and language

Perfect pretraining for multimodal understanding!
Uses trained tokenizer from Stage A for consistent text encoding.
```

**Training Loop:**

```python
for batch in dataloader:
    images, captions = batch  # (B, 3, 224, 224), (B,) list of strings

    # 1. Encode images
    cls_output = vit(images)  # (B, 128) - CLS token
    img_emb = img_proj(cls_output)  # (B, embed_dim)
    img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)  # L2 normalize

    # 2. Encode captions (configurable: Thinker or SimpleTextEncoder with attention pooling)
    text_embs = []
    for caption in captions:
        token_ids = tokenizer.encode(caption)  # Use trained tokenizer
        token_ids = [1] + token_ids[:ctx_len-1]  # Add BOS/CLS, truncate

        if use_thinker_for_text:
            # Option 1: Use Thinker model (frozen) for contextual embeddings
            token_tensor = torch.tensor(token_ids).unsqueeze(0)  # (1, T)
            with torch.no_grad():
                text_emb = think(idx=token_tensor)  # (1, T, thinker_d_model)
            text_emb = text_emb.squeeze(0).mean(dim=0)  # (thinker_d_model,) - mean pooling
        else:
            # Option 2: Use SimpleTextEncoder with learned attention pooling
            # Attention pooling learns which tokens are important for each caption
            token_tensor = torch.tensor(token_ids)  # (T,)
            text_emb = text_encoder(token_tensor, return_cls=True)  # (d_model,)
            # Uses AttentionPooling internally - learns to weight tokens by importance

        text_embs.append(text_emb)
    text_embs = torch.stack(text_embs)  # (B, d_model or thinker_d_model)
    text_emb = text_proj(text_embs)  # (B, embed_dim)
    text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)  # L2 normalize

    # 3. Contrastive loss (InfoNCE)
    logits = torch.matmul(img_emb, text_emb.t()) / temperature  # (B, B)
    labels = torch.arange(B, device=device)  # Positive pairs on diagonal
    loss = cross_entropy(logits, labels)

    # 4. Backprop and update
    loss.backward()
    optimizer.step()
```

**Key Features:**

- Uses **trained tokenizer** from Stage A (`thinker_ckpt/tokenizer.model`)
- If tokenizer not found, trains new one from image captions
- **Configurable text encoding** via `use_thinker_for_text`:
  - **`true` (recommended)**: Uses frozen Thinker model for contextual embeddings - better quality, aligned with Stage E
  - **`false`**: Uses **SimpleTextEncoder with learned attention pooling** - lighter, faster, learns to weight important tokens
    - **AttentionPooling**: Learns which tokens are semantically important for each caption
    - **Masked attention**: Handles variable-length captions properly (ignores padding)
    - **Better than mean pooling**: Reduces train/val gap, improves text-image alignment
- **Contrastive learning** aligns image and text embeddings in shared space
- **InfoNCE loss** encourages matching image-caption pairs to be similar

---

## 🔗 Connection to Thinker

### How Images Flow into Multimodal Processing

```
COMPLETE PIPELINE:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. User uploads cat photo
   Image: (3, 224, 224)

2. Vision Encoder processes:
   → Divide into 196 patches
   → Process with transformer
   → Extract CLS token
   → Project to 256-dim: (1, 256)

3. User types: "What animal is this?"
   Text tokens: [15, 234, 89, 42, 156]
   → Embed: (5, 256)

4. Concatenate:
   Combined input: (6, 256)
   = [1 image token, 5 text tokens]

5. Thinker processes:
   → Cross-modal attention
   → Image token interacts with text tokens
   → Understands: User asking about the image

6. Generate response:
   Token by token: "This", "is", "a", "cat", "."

Vision encoder enabled visual understanding! ✓

Efficiency comparison:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Without vision encoder:
- Process 150,528 pixels directly? Impossible!

With vision encoder:
- Process 1 image token! ✓
- 150,528 → 1 (massive compression)
- All visual information preserved
- Ready for multimodal reasoning
```

---

## 💡 Key Takeaways

✅ **Vision Encoder** translates images into semantic embeddings  
✅ **Patch-based processing** (16×16 patches) for efficiency  
✅ **196 patches** from 224×224 image  
✅ **CLS token** aggregates global image understanding  
✅ **Transformer encoder** captures visual relationships  
✅ **Projects to 256-dim** to align with Thinker  
✅ **Single embedding** represents entire image  
✅ **~914K parameters** - compact and efficient  
✅ **Enables multimodal** text+image+audio understanding  
✅ **Also used in OCR** model for text extraction from images

**Note:** The Vision Encoder (ViT) architecture is also used in the optional OCR model (`train_ocr.py`), where it processes image patches to extract visual features that are then decoded into text sequences.

---

## 🎓 Self-Check Questions

1. Why do we use patches instead of processing pixels directly?
2. What is the CLS token and what role does it play?
3. How many patches does a 224×224 image become?
4. Why do we only keep the CLS token and discard the patch tokens?
5. Why project from 128 to 256 dimensions at the end?

<details>
<summary>📝 Click to see answers</summary>

1. Processing 150K pixels directly would require massive computation (50K×50K attention matrix). Patches (196 total) are much more efficient (196×196) while capturing meaningful visual units
2. CLS token is a special learnable token that aggregates information from all patches through attention. It serves as a global representation of the entire image
3. 224÷16 = 14 patches per side, so 14×14 = 196 patches total
4. Through transformer layers, CLS token gathers all relevant information from patches via attention. The final CLS embedding contains the holistic image understanding, so patch tokens are no longer needed
5. To align with Thinker's input dimension (256) - all modalities (text, image, audio) must be 256-dim for unified multimodal processing
</details>

---

[Continue to Chapter 23: RVQ Codec →](23-codec-rvq.md)

**Chapter Progress:** μOmni Components ●●○○○ (2/5 complete)

---

## 📊 Specifications

| Parameter      | Value             |
| -------------- | ----------------- |
| **Input**      | Image (224×224×3) |
| **Patch Size** | 16×16             |
| **Patches**    | 196 + 1 CLS       |
| **Dimension**  | 128               |
| **Layers**     | 4                 |
| **Parameters** | ~914K             |

## 🎓 Training

**Task**: Image-Caption contrastive learning (CLIP-style)  
**Loss**: Contrastive loss (InfoNCE)  
**Data**: Images + text captions  
**Text Encoding**: Uses trained tokenizer from Stage A (`thinker_ckpt/tokenizer.model`)

## 💡 Key Takeaways

✅ **Vision Transformer** (patch-based)  
✅ **196 patch tokens + CLS token**  
✅ **CLS token aggregates** global information  
✅ **Output**: Single 256-dim vector per image

---

[Back to Index](00-INDEX.md)
