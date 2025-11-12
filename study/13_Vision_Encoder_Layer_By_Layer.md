# Vision Encoder: Complete Layer-by-Layer Breakdown

## Overview

This document explains **every single layer** in the Vision Encoder (ViT), combining **deep theoretical understanding** with **practical implementation**. We'll explore the **why** behind each design choice and **what value** it provides.

## Theoretical Foundation: Why Vision Transformer?

### The Image Understanding Problem

Images are fundamentally different from text:
- **2D spatial structure**: Pixels arranged in 2D grid
- **High dimensionality**: 224×224×3 = 150,528 values
- **Local and global patterns**: Both matter
- **Translation invariance**: Object can be anywhere

### Why Vision Transformer (ViT) Over CNN?

**CNN approach** (traditional):
- Uses convolutional filters
- Processes locally, builds up receptive field
- Requires many layers for global understanding
- Translation invariance built-in

**ViT approach** (modern):
- Splits image into patches
- Processes patches as sequence
- Global attention from first layer
- Learns translation invariance

**Why ViT for μOmni?**
- **Unified architecture**: Same transformer as Thinker
- **Better for multimodal**: Easier to fuse with text/audio
- **Global understanding**: Sees full image immediately
- **Proven**: State-of-the-art results

### Why Patches?

**The patch idea**:
- Treat image as sequence of patches
- Each patch = "word" in image language
- Enables transformer processing

**Why 16×16 patches?**
- **Balance**: Not too small (too many patches), not too large (loses detail)
- **Standard**: Widely used size
- **Efficiency**: 196 patches is manageable

**What gets captured**:
- **Small patches**: Fine details, textures
- **Large patches**: Coarse structure, objects
- **16×16**: Good balance for both

### Why CLS Token?

**The aggregation problem**:
- 196 patch tokens (too many for downstream tasks)
- Need single representation
- Options: Average, max, or CLS token

**Why CLS token?**
- **Learnable**: Model learns how to aggregate
- **Fixed size**: Always 1 token (efficient)
- **Proven**: Works in BERT, ViT
- **Flexible**: Can learn task-specific aggregation

## Layer-by-Layer Deep Dive

## Complete Architecture

```
Input: Image (224×224×3)
    ↓
[Patch Embedding] → (B, 196, 128)
    ↓
[Add CLS Token] → (B, 197, 128)
    ↓
[Add Position Embeddings] → (B, 197, 128)
    ↓
┌─────────────────────────────────┐
│  Encoder Block 1                │
│  ┌───────────────────────────┐ │
│  │ Pre-Norm                  │ │
│  │ Self-Attention            │ │
│  │ Residual                  │ │
│  │ Pre-Norm                  │ │
│  │ MLP (GELU)                │ │
│  │ Residual                  │ │
│  └───────────────────────────┘ │
└───────────────┬─────────────────┘
                ↓
    [Repeat for N blocks (default: 4)]
                ↓
[Final RMSNorm] → (B, 197, 128)
    ↓
[Extract CLS Token] → (B, 1, 128)
    ↓
Output: CLS Token Embedding
```

## Layer 1: Patch Embedding

### Purpose
Split image into non-overlapping patches and convert to embeddings.

### Implementation

```python
# From omni/vision_encoder.py
class ViTTiny(nn.Module):
    def __init__(self, img_size=224, patch=16, d=128, ...):
        # Patch embedding: Conv2d with kernel=patch, stride=patch
        self.proj = nn.Conv2d(
            3,           # Input channels (RGB)
            d,           # Output channels (d_model)
            kernel_size=patch,  # 16×16 patches
            stride=patch        # Non-overlapping
        )
```

### What Happens

```python
# Input: Image tensor
x = torch.randn(1, 3, 224, 224)  # (B, C, H, W)

# Patch embedding
x = self.proj(x)  # (1, 3, 224, 224) → (1, 128, 14, 14)
```

### Step-by-Step

1. **Input**: `(1, 3, 224, 224)` - RGB image
2. **Conv2D**: 
   - Kernel: 16×16
   - Stride: 16
   - No padding
3. **Output**: `(1, 128, 14, 14)`
   - 14×14 = 196 patches
   - Each patch → 128-dim embedding

### Visual Representation

```
Image (224×224):
┌─────────────────────────┐
│ ██ ██ ██ ██ ██ ██ ... │  ← 14 patches wide
│ ██ ██ ██ ██ ██ ██ ... │
│ ██ ██ ██ ██ ██ ██ ... │
│ ...                    │  ← 14 patches tall
└─────────────────────────┘

Each ██ = 16×16 pixels → 128-dim embedding
Total: 14 × 14 = 196 patches
```

### Deep Theoretical Analysis: Patch Embedding

#### Why Convolutional Patch Embedding?

**Option 1: Flatten pixels**:
- 16×16×3 = 768 values per patch
- Too high-dimensional
- Loses spatial structure within patch

**Option 2: Convolutional embedding** (used):
- Conv2d with 16×16 kernel, stride 16
- Learns to extract features from patches
- Maintains some spatial understanding
- Projects to lower dimension (128)

**Why this works**:
- **Feature extraction**: Learns useful patch representations
- **Dimensionality reduction**: 768 → 128
- **Spatial awareness**: Maintains patch structure
- **Learnable**: Adapts to task

#### What Gets Learned in Patch Embedding?

During training, patch embedding learns:
1. **Edge detection**: Identifies edges and boundaries
2. **Texture patterns**: Recognizes textures within patches
3. **Color patterns**: Captures color relationships
4. **Spatial relationships**: Understands patch structure

This is similar to **first layer of CNN** - learns low-level features.

#### Information Content Per Patch

Each 16×16 patch contains:
- **256 pixels** (16×16)
- **3 color channels** (RGB)
- **768 raw values**

After embedding:
- **128 dimensions**
- **Compressed representation**
- **Task-relevant features**

**Compression ratio**: 768 → 128 (6× compression)
- Forces model to extract essential information
- Discards redundant details
- Learns efficient representations

### What Value Do We Get from Patch Embedding?

1. **Dimensionality Reduction**: 768 → 128 (efficient)
2. **Feature Learning**: Learns useful patch representations
3. **Spatial Awareness**: Maintains patch structure
4. **Flexibility**: Adapts to different image types
5. **Efficiency**: Single convolution operation

### Mathematical View

```python
# Conv2D operation
# For each 16×16 patch:
patch = image[:, :, i*16:(i+1)*16, j*16:(j+1)*16]  # (B, 3, 16, 16)
embedding = conv2d(patch)  # (B, 128, 1, 1)
# Flatten: (B, 128)
```

## Layer 2: Reshape to Sequence

```python
# From ViTTiny.forward
# Rearrange: (B, d, H', W') → (B, H'*W', d)
x = rearrange(x, "b d h w -> b (h w) d")
# (1, 128, 14, 14) → (1, 196, 128)
```

### What Happens

- **Before**: `(B, 128, 14, 14)` - 2D grid of embeddings
- **After**: `(B, 196, 128)` - Sequence of patch embeddings
- Each of 196 patches is now a token in the sequence

## Layer 3: Add CLS Token

### Purpose
Special token that aggregates global image information.

### Implementation

```python
# From ViTTiny.__init__
self.cls = nn.Parameter(torch.zeros(1, 1, d))  # Learnable CLS token

# From ViTTiny.forward
B, N, D = x.shape  # (B, 196, 128)
cls = self.cls.expand(B, -1, -1)  # (B, 1, 128)
x = torch.cat([cls, x], dim=1)  # (B, 197, 128)
```

### Step-by-Step

```python
# CLS token
cls_token = nn.Parameter(torch.randn(1, 1, 128))  # Learnable

# Expand to batch size
cls = cls_token.expand(B, -1, -1)  # (1, 1, 128) → (B, 1, 128)

# Prepend to sequence
x = torch.cat([cls, x], dim=1)
# Before: (B, 196, 128)
# After:  (B, 197, 128)
#         [CLS, patch1, patch2, ..., patch196]
```

### Why CLS Token?

- **Global Representation**: Aggregates information from all patches
- **Fixed Size**: Always 1 token, regardless of image size
- **Efficient**: Easier to use than averaging all patches

## Layer 4: Add Position Embeddings

### Purpose
Tell the model where each patch is located in the image.

### Implementation

```python
# From ViTTiny.__init__
num_patches = (img_size // patch) ** 2  # 14×14 = 196
self.pos = nn.Parameter(torch.zeros(1, 1 + num_patches, d))
# (1, 197, 128) - 1 CLS + 196 patches

# From ViTTiny.forward
x = x + self.pos[:, :x.size(1), :]
```

### Step-by-Step

```python
# Position embeddings
pos_emb = nn.Parameter(torch.randn(1, 197, 128))
# Learnable embeddings for each position:
# - Position 0: CLS token
# - Position 1-196: Patch positions

# Add to tokens
x = x + pos_emb[:, :x.size(1), :]
# Element-wise addition: (B, 197, 128) + (1, 197, 128)
```

### Position Encoding Structure

```
Position 0:  CLS token position
Position 1:  Top-left patch (row 0, col 0)
Position 2:  Top patch (row 0, col 1)
...
Position 14: End of row 0
Position 15: Start of row 1
...
Position 196: Bottom-right patch (row 13, col 13)
```

## Layer 5: Encoder Blocks

### Block Structure

```python
# From omni/vision_encoder.py
# Uses PyTorch's TransformerEncoderLayer
self.blocks = nn.ModuleList([
    nn.TransformerEncoderLayer(
        d_model=d,           # 128
        nhead=heads,         # 2
        dim_feedforward=ff,  # 512
        dropout=dropout,     # 0.1
        batch_first=True,    # (B, T, D) format
        norm_first=True,     # Pre-norm architecture
        activation='gelu'    # GELU activation
    )
    for _ in range(layers)  # 4 blocks
])
```

### TransformerEncoderLayer Internals

PyTorch's layer contains:

```python
class TransformerEncoderLayer:
    def __init__(self, ...):
        self.norm1 = nn.LayerNorm(d_model)  # Pre-norm
        self.self_attn = nn.MultiheadAttention(...)
        self.dropout1 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)  # Pre-norm
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.activation = nn.GELU()
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout3 = nn.Dropout(dropout)
    
    def forward(self, x):
        # Pre-norm + Attention + Residual
        x = x + self.dropout1(
            self.self_attn(
                self.norm1(x), self.norm1(x), self.norm1(x)
            )[0]
        )
        
        # Pre-norm + MLP + Residual
        x = x + self.dropout3(
            self.linear2(
                self.dropout2(
                    self.activation(
                        self.linear1(self.norm2(x))
                    )
                )
            )
        )
        
        return x
```

### Step-by-Step: Block Forward Pass

#### 5.1: Pre-Norm

```python
x_norm = self.norm1(x)  # (B, 197, 128)
```

#### 5.2: Self-Attention

```python
# Multi-head self-attention
# Q = K = V = x_norm (self-attention)
attn_output, _ = self.self_attn(
    x_norm,  # Query
    x_norm,  # Key
    x_norm,  # Value
    need_weights=False
)
# Output: (B, 197, 128)
```

**What Self-Attention Does**:
- Each patch can attend to **all other patches** (bidirectional)
- Learns spatial relationships
- Example: "cat head" patch attends to "cat body" patches

#### 5.3: Residual Connection

```python
x = x + self.dropout1(attn_output)  # (B, 197, 128)
```

#### 5.4: MLP

```python
# MLP: Linear → GELU → Linear
x_norm = self.norm2(x)  # Pre-norm
x_expanded = self.linear1(x_norm)  # (B, 197, 128) → (B, 197, 512)
x_activated = self.activation(x_expanded)  # GELU
x_activated = self.dropout2(x_activated)
x_contracted = self.linear2(x_activated)  # (B, 197, 512) → (B, 197, 128)
x = x + self.dropout3(x_contracted)  # Residual
```

### MLP Details

```python
# MLP structure
linear1 = nn.Linear(128, 512)  # Expand
activation = nn.GELU()
linear2 = nn.Linear(512, 128)  # Contract

# Forward
x = linear1(x)    # (B, 197, 128) → (B, 197, 512)
x = gelu(x)       # (B, 197, 512)
x = linear2(x)    # (B, 197, 512) → (B, 197, 128)
```

## Layer 6: Repeat Blocks

```python
# From ViTTiny.forward
for blk in self.blocks:
    x = blk(x)  # Each block: (B, 197, 128) → (B, 197, 128)
```

## Layer 7: Final Normalization

```python
# From ViTTiny.forward
x = self.norm(x)  # Final RMSNorm
# Output: (B, 197, 128)
```

## Layer 8: Extract CLS Token

```python
# From ViTTiny.forward
cls = x[:, :1, :]   # (B, 1, 128) - CLS token
grid = x[:, 1:, :]  # (B, 196, 128) - Patch tokens

return cls, grid
```

### Why Only CLS?

- **Efficiency**: Single vector instead of 196
- **Global Info**: CLS aggregates all patch information
- **Standard Practice**: Used in BERT, ViT, etc.

### Deep Dive: How CLS Token Aggregates Information

#### Attention-Based Aggregation

The CLS token doesn't just average patches - it uses **attention**:
- CLS attends to all patches
- Attention weights determine which patches are important
- Weighted combination creates rich global representation

**Example**:
- Image: Cat in corner, background elsewhere
- CLS attends strongly to "cat" patches
- CLS attends weakly to "background" patches
- Result: CLS represents "cat" more than background

#### Why This is Better Than Averaging

**Simple average**:
- All patches weighted equally
- Loses important information
- No selectivity

**Attention-based** (CLS):
- Important patches weighted more
- Selective information aggregation
- Learns what's important

#### Information Flow

```
Patches → Attention → CLS Token
196 patches → 196 attention weights → 1 aggregated representation
```

The CLS token becomes a **learned summary** of the entire image.

### What Value Do We Get from CLS Token?

1. **Efficient Representation**: Single vector for entire image
2. **Selective Aggregation**: Focuses on important parts
3. **Learnable**: Adapts to task requirements
4. **Fixed Size**: Always same size (easy to use)
5. **Interpretable**: Can visualize attention to see what CLS focuses on

## Complete Forward Pass with Shapes

```python
# Input: Image
image = torch.randn(1, 3, 224, 224)  # (B, C, H, W)

# Step 1: Patch embedding
x = proj(image)  # (1, 3, 224, 224) → (1, 128, 14, 14)

# Step 2: Reshape to sequence
x = rearrange(x, "b d h w -> b (h w) d")  # (1, 196, 128)

# Step 3: Add CLS token
cls = cls_token.expand(1, -1, -1)  # (1, 1, 128)
x = torch.cat([cls, x], dim=1)  # (1, 197, 128)

# Step 4: Add position embeddings
x = x + pos_emb  # (1, 197, 128)

# Step 5: Encoder blocks
x = block1(x)  # (1, 197, 128)
x = block2(x)  # (1, 197, 128)
x = block3(x)  # (1, 197, 128)
x = block4(x)  # (1, 197, 128)

# Step 6: Final norm
x = norm(x)  # (1, 197, 128)

# Step 7: Extract CLS
cls = x[:, :1, :]  # (1, 1, 128)
grid = x[:, 1:, :]  # (1, 196, 128)

# Output: CLS token embedding
```

## Attention Visualization

The model learns spatial relationships:

```
Input Image:          Attention from CLS:
┌─────────┐          ┌─────────┐
│  Cat    │          │  ████   │  ← High attention to
│  Face   │    →     │  ████   │     cat patches
│         │          │  ░░░    │  ← Low attention to
└─────────┘          └─────────┘     background
```

## Key Parameters

From `configs/vision_tiny.json`:

```json
{
  "img_size": 224,      // Input image size
  "patch": 16,          // Patch size
  "d_model": 128,       // Embedding dimension
  "n_layers": 4,        // Number of encoder blocks
  "n_heads": 2,         // Attention heads
  "d_ff": 512           // MLP feedforward dimension
}
```

## Calculations

### Number of Patches

```
Image size: 224×224
Patch size: 16×16
Patches per side: 224 / 16 = 14
Total patches: 14 × 14 = 196
Total tokens: 196 + 1 (CLS) = 197
```

### Memory and Computation

### Per Block:
- **Parameters**: ~0.3M (d_model=128, n_heads=2, d_ff=512)
- **Memory**: O(B * 197 * d_model) for activations
- **Computation**: O(B * 197² * d_model) for attention

### Total Model:
- **Parameters**: ~10M (4 blocks + patch_emb + pos_emb)
- **Input**: 224×224×3 = 150,528 pixels
- **Output**: 1 CLS token (128 dims)

## Comparison: ViT vs CNN

| Feature | ViT | CNN |
|---------|-----|-----|
| Processing | Sequential patches | Local convolutions |
| Attention | Global (all patches) | Local (receptive field) |
| Position | Learned embeddings | Implicit (convolution) |
| Architecture | Transformer | Convolutional |

**Why ViT for μOmni?**
- Unified architecture (same as Thinker)
- Better for multimodal fusion
- Scales well with data

---

**Next:**
- [11_Thinker_Layer_By_Layer.md](11_Thinker_Layer_By_Layer.md) - Thinker layers
- [12_Audio_Encoder_Layer_By_Layer.md](12_Audio_Encoder_Layer_By_Layer.md) - Audio layers
- [05_Vision_Encoder.md](05_Vision_Encoder.md) - Vision encoder overview

