# Chapter 11: Image Processing for AI

[← Previous: Audio Processing](10-audio-processing.md) | [Back to Index](00-INDEX.md) | [Next: Vector Quantization →](12-quantization.md)

---

## 🎯 What You'll Learn

- Image representation basics
- Convolutional operations
- Vision Transformers (ViT)
- Patch-based image encoding
- How μOmni processes images

---

## 🖼️ Image Representation

### RGB Images

```
Digital image = 3D tensor (Height × Width × Channels)

Example: 224×224 RGB image

Channel layout:
R: [[255, 200, ...],     G: [[100, 150, ...],     B: [[50, 80, ...],
    [180, 220, ...],         [120, 140, ...],         [60, 70, ...],
    ...]                     ...]                     ...]
    
Shape: (224, 224, 3) = 150,528 pixels × 3 values = 451,584 numbers!
```

---

### Normalization

```python
# Raw image: values 0-255
image = PIL.Image.open("photo.jpg")
img_array = np.array(image)  # (H, W, 3) with values [0, 255]

# Normalize to [0, 1]
img_normalized = img_array / 255.0

# Or standardize (mean=0, std=1)
mean = [0.485, 0.456, 0.406]  # ImageNet statistics
std = [0.229, 0.224, 0.225]
img_standardized = (img_normalized - mean) / std

# PyTorch format: (C, H, W)
img_tensor = torch.from_numpy(img_standardized).permute(2, 0, 1)
```

---

## 🔲 Convolutional Operations

### Convolution Basics

```
Input image (5×5):      Filter/Kernel (3×3):
┌─────────────┐        ┌─────┐
│ 1 2 3 4 5  │        │ 1 0 1│
│ 6 7 8 9 10 │    *   │ 0 1 0│
│11 12 13 14 15│        │ 1 0 1│
│16 17 18 19 20│        └─────┘
│21 22 23 24 25│
└─────────────┘

Convolution operation:
Slide filter over image, compute dot product

Output (3×3 with stride=1):
┌────────┐
│ ?  ?  ?│
│ ?  ?  ?│
│ ?  ?  ?│
└────────┘

Example: Top-left calculation
(1×1)+(2×0)+(3×1)+(6×0)+(7×1)+(8×0)+(11×1)+(12×0)+(13×1) = 32
```

---

### Convolutional Layers

```python
import torch.nn as nn

# 2D Convolution
conv = nn.Conv2d(
    in_channels=3,      # RGB input
    out_channels=64,    # 64 feature maps
    kernel_size=3,      # 3×3 filter
    stride=1,           # Step size
    padding=1          # Preserve spatial size
)

# Input: (B, 3, 224, 224)
# Output: (B, 64, 224, 224)
```

---

## 🎯 Vision Transformer (ViT)

### Patch-Based Approach

```
Traditional CNN:
Image → Conv layers → Features

ViT:
Image → Split into patches → Transformer

Why patches?
✅ Reduces sequence length
✅ Captures local patterns
✅ Compatible with transformers
```

---

### ViT Architecture

```
Image (224×224×3)
    ↓
┌──────────────────────────────────────┐
│ Patch Embedding (16×16 patches)     │
│                                      │
│ Image grid: 224/16 = 14 patches     │
│ Total patches: 14×14 = 196          │
│                                      │
│ Each patch: 16×16×3 = 768 values    │
│ Project to d_model dimensions        │
└──────────────┬───────────────────────┘
               ↓
┌──────────────────────────────────────┐
│ Add CLS Token + Position Embeddings │
│                                      │
│ [CLS] [P1] [P2] ... [P196]          │
│   ↓     ↓    ↓         ↓             │
│  +pos  +pos +pos     +pos            │
└──────────────┬───────────────────────┘
               ↓
┌──────────────────────────────────────┐
│ Transformer Encoder Blocks          │
│ (Self-attention + FFN) × L layers   │
└──────────────┬───────────────────────┘
               ↓
┌──────────────────────────────────────┐
│ Extract CLS Token                   │
│ Use as image representation         │
└──────────────────────────────────────┘

μOmni: 14×14 patches, d_model=128, 4 layers
```

---

## 💻 μOmni's Vision Encoder

### ViT-Tiny Implementation

```python
# From omni/vision_encoder.py (simplified)
class ViTTiny(nn.Module):
    def __init__(self, img_size=224, patch=16, d=128, layers=4):
        super().__init__()
        
        # Patch embedding (convolution with stride=patch)
        self.proj = nn.Conv2d(3, d, kernel_size=patch, stride=patch)
        
        # CLS token (learnable)
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        
        # Position embeddings (learnable)
        num_patches = (img_size // patch) ** 2
        self.pos = nn.Parameter(torch.zeros(1, 1 + num_patches, d))
        
        # Transformer encoder
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(d, heads=2, dim_feedforward=512)
            for _ in range(layers)
        ])
        
        self.norm = RMSNorm(d)
    
    def forward(self, x):
        # x: (B, 3, 224, 224)
        
        # Patch embedding
        x = self.proj(x)  # (B, d, 14, 14)
        x = x.flatten(2).transpose(1, 2)  # (B, 196, d)
        
        # Add CLS token
        B = x.shape[0]
        cls = self.cls.expand(B, -1, -1)  # (B, 1, d)
        x = torch.cat([cls, x], dim=1)    # (B, 197, d)
        
        # Add position embeddings
        x = x + self.pos
        
        # Transformer blocks
        for block in self.blocks:
            x = block(x)
        
        x = self.norm(x)
        
        # Extract CLS and patch tokens
        cls_token = x[:, 0:1, :]   # (B, 1, d)
        patch_tokens = x[:, 1:, :]  # (B, 196, d)
        
        return cls_token, patch_tokens
```

---

## 📊 Image Processing Pipeline

### μOmni's Complete Flow

```
1. Load & Resize
   Raw image (any size) → (224, 224, 3)

2. Normalize
   [0, 255] → [0, 1] → standardize

3. To Tensor
   (224, 224, 3) → (3, 224, 224)

4. Patch Embedding
   (3, 224, 224) → (196, 128)

5. Add CLS + Positional
   (196, 128) → (197, 128)

6. Transformer Encoder
   (197, 128) → (197, 128)

7. Extract CLS Token
   (197, 128) → (1, 128)

8. Vision Projector
   (1, 128) → (1, 256)

9. Ready for Thinker!
   Single token representing entire image
```

---

## 🎨 Data Augmentation for Images

### Training-Time Augmentations

```python
from torchvision import transforms

train_transform = transforms.Compose([
    # Geometric
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    
    # Color
    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2,
        hue=0.1
    ),
    
    # Normalize
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])
```

---

## 🔍 Why CLS Token?

### Aggregating Image Information

```
Patch tokens: Local information
[P1]: Top-left corner features
[P2]: Top-middle features
...
[P196]: Bottom-right corner features

CLS token: Global information
Through self-attention, CLS attends to all patches:
CLS → [P1, P2, ..., P196]

Result: CLS aggregates entire image into single vector!

Alternative approaches:
- Average all patch tokens (simple but loses structure)
- Use all patch tokens (too many for μOmni)
- CLS token (efficient aggregation) ← μOmni uses this
```

---

## 📊 ViT vs CNN Comparison

| Feature | CNN | ViT |
|---------|-----|-----|
| **Processing** | Local receptive field | Global attention |
| **Inductive bias** | Strong (locality) | Weak (learned) |
| **Data efficiency** | Good (small data) | Requires more data |
| **Scalability** | Limited | Excellent |
| **Context** | Hierarchical | Global from start |

```
CNN: Local → Regional → Global (hierarchical)
ViT: Global attention from layer 1 (patch interactions)

μOmni uses ViT (better for multimodal integration)
```

---

## 💡 Key Takeaways

✅ **Images** are 3D tensors (H×W×C)  
✅ **Convolutions** detect local patterns  
✅ **ViT** splits images into patches (16×16)  
✅ **Patch embeddings** reduce sequence length (196 patches)  
✅ **CLS token** aggregates global image information  
✅ **μOmni uses ViT-Tiny** (128-dim, 4 layers)  
✅ **Output**: Single 256-dim vector per image

---

## 🎓 Self-Check Questions

1. What are the dimensions of a 224×224 RGB image?
2. How many patches does ViT create from a 224×224 image with patch size 16?
3. What is the CLS token used for?
4. What's the advantage of ViT over traditional CNNs?
5. How many dimensions is μOmni's final image representation?

<details>
<summary>📝 Answers</summary>

1. (224, 224, 3) or (3, 224, 224) in PyTorch format
2. (224/16) × (224/16) = 14 × 14 = 196 patches
3. CLS token aggregates global image information through self-attention with all patches
4. ViT uses global attention from the start (vs CNN's local receptive fields), better scalability
5. 256 dimensions (after vision projector maps 128→256)
</details>

---

[Continue to Chapter 12: Vector Quantization →](12-quantization.md)

**Chapter Progress:** Core Concepts ●●●●●●○ (6/7 complete)

