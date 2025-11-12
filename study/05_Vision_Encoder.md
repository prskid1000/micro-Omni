# Vision Encoder: Understanding Images

## What is the Vision Encoder?

The **Vision Encoder** converts images into embeddings that Thinker can understand.

Think of it as "eyes" for the AI - it processes visual information and extracts meaningful features.

## Image Basics

### Pixels

Images are grids of pixels, each with color values:

```
RGB Image (224×224×3)
- 224 pixels tall
- 224 pixels wide  
- 3 color channels (Red, Green, Blue)
```

Each pixel value: 0-255 (or 0.0-1.0 normalized)

### Image Preprocessing

```python
# Standard preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to fixed size
    transforms.ToTensor()            # Convert to tensor, normalize to [0,1]
])

image = Image.open("cat.jpg")
tensor = transform(image)  # Shape: (3, 224, 224)
```

## Architecture: Vision Transformer (ViT)

μOmni uses a **Vision Transformer (ViT)**, which treats images like sequences of patches.

### High-Level Flow

```
Image (224×224×3)
    ↓
Patch Embedding (14×14 patches)
    ↓
Add CLS Token + Position Embeddings
    ↓
Transformer Encoder
    ↓
CLS Token (1×128) → Thinker
```

## Step-by-Step Processing

### 1. Patch Embedding

Split image into non-overlapping patches:

```python
# Image: (3, 224, 224)
# Patch size: 16×16
# Result: 14×14 = 196 patches

def patch_embedding(image):
    # Conv2d with kernel=16, stride=16
    patches = conv2d(image, kernel=16, stride=16)
    # Shape: (196, d_model)
    return patches
```

**Visual**:
```
Image:        Patches:
████████      ██ ██ ██ ██
████████  →   ██ ██ ██ ██
████████      ██ ██ ██ ██
████████      ██ ██ ██ ██
```

### 2. CLS Token

A special token that aggregates global image information:

```python
# Learnable CLS token
cls_token = nn.Parameter(torch.randn(1, 1, d_model))

# Prepend to patches
tokens = torch.cat([cls_token, patches], dim=1)
# Shape: (197, d_model) - 1 CLS + 196 patches
```

**Why CLS Token?**
- Provides a single vector representing the whole image
- Similar to BERT's [CLS] token
- Easier to use than averaging all patches

### 3. Position Embeddings

Tell the model where each patch is located:

```python
# Learnable position embeddings
pos_emb = nn.Parameter(torch.randn(197, d_model))  # 1 CLS + 196 patches

# Add to tokens
tokens = tokens + pos_emb
```

**Why needed?**
- Patches lose spatial information when flattened
- Position embeddings restore location awareness

### 4. Transformer Encoder

Process the sequence of patches:

```python
# Similar to audio encoder
for block in encoder_blocks:
    tokens = block(tokens)  # Self-attention + MLP
```

**Attention in Vision**:
- Patches can "attend" to other patches
- Model learns spatial relationships
- Example: "cat" patch attends to "whiskers" patches

### 5. Extract CLS Token

Use only the CLS token for downstream tasks:

```python
cls_token = tokens[0]  # First token
# Shape: (1, d_model)
```

**Why only CLS?**
- Contains global image information
- Fixed size regardless of image size
- Efficient for Thinker integration

## Code Structure

```python
# From omni/vision_encoder.py

class ViTTiny(nn.Module):
    def __init__(self, img_size, patch, d_model, ...):
        # Patch embedding
        self.patch_emb = nn.Conv2d(
            3, d_model, 
            kernel_size=patch, 
            stride=patch
        )
        
        # CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        
        # Position embeddings
        num_patches = (img_size // patch) ** 2
        self.pos_emb = nn.Parameter(
            torch.randn(1, num_patches + 1, d_model)
        )
        
        # Transformer encoder
        self.blocks = nn.ModuleList([
            EncoderBlock(d_model, n_heads, ...)
            for _ in range(n_layers)
        ])
        
        # Final norm
        self.norm = RMSNorm(d_model)
    
    def forward(self, img):
        # Patch embedding
        patches = self.patch_emb(img)  # (B, d_model, H', W')
        patches = patches.flatten(2).transpose(1, 2)  # (B, N, d_model)
        
        # Add CLS token
        cls = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls, patches], dim=1)
        
        # Add position embeddings
        tokens = tokens + self.pos_emb
        
        # Encode
        for block in self.blocks:
            tokens = block(tokens)
        
        # Extract CLS
        cls = tokens[:, 0]  # (B, d_model)
        patches = tokens[:, 1:]  # (B, N, d_model)
        
        return cls, patches
```

## Training: Image Classification

The vision encoder is trained on a simple classification task:

```python
# Simplified training
image = load_image("red_square.png")
caption = "This is a red square"

# Encode
cls_token = vision_encoder(image)

# Classify
logits = classifier(cls_token)
# Predict: "red" in caption? → label 0, else → label 1
loss = cross_entropy(logits, label)
```

**Note**: This is a simplified pretraining task. Real models use contrastive learning (CLIP-style).

## Configuration

From `configs/vision_tiny.json`:

```json
{
  "img_size": 224,
  "patch": 16,
  "d_model": 128,
  "n_layers": 4,
  "n_heads": 2,
  "d_ff": 512
}
```

**Calculations**:
- Patches per side: 224 / 16 = 14
- Total patches: 14 × 14 = 196
- Total tokens: 196 + 1 (CLS) = 197

## Multimodal Integration

After encoding, image embeddings are projected to Thinker's dimension:

```python
# Vision encoder output
cls_token, patches = vision_encoder(image)  # (1, 128)

# Project to Thinker dimension
image_emb = vision_projector(cls_token)  # (1, 256)

# Now Thinker can process it!
thinker_input = torch.cat([image_emb, text_emb], dim=1)
output = thinker(thinker_input)
```

**Note**: Only CLS token is used (not individual patches) for efficiency.

## Attention Visualization

The model learns to focus on relevant parts:

```
Input Image:          Attention Weights:
┌─────────┐          ┌─────────┐
│  Cat    │          │  ████   │  ← High attention
│  Face   │    →     │  ████   │
│         │          │  ░░░    │  ← Low attention
└─────────┘          └─────────┘
```

## Common Issues

### 1. Image Size Mismatch

**Problem**: Image not 224×224

**Solution**: Always resize
```python
transform = transforms.Resize((224, 224))
```

### 2. Wrong Color Format

**Problem**: Grayscale or RGBA

**Solution**: Convert to RGB
```python
image = image.convert("RGB")
```

### 3. Image Orientation

**Problem**: EXIF rotation not applied

**Solution**: Use PIL's transpose
```python
from PIL import Image
image = Image.open(path)
image = ImageOps.exif_transpose(image)
```

## Visual Guide

```
Image File (JPG/PNG)
    ↓
[Load] → PIL Image (H×W×3)
    ↓
[Resize] → (224×224×3)
    ↓
[ToTensor] → (3, 224, 224)
    ↓
[Patch Embedding] → (196, 128)
    ↓
[Add CLS Token] → (197, 128)
    ↓
[Add Position Embeddings] → (197, 128)
    ↓
[Transformer Encoder] → (197, 128)
    ↓
[Extract CLS] → (1, 128)
    ↓
[Projector] → (1, 256) → Thinker
```

## Training Example

```python
# Load image and caption
image = Image.open("example.png")
caption = "This is a red square"

# Preprocess
img_tensor = transform(image)  # (3, 224, 224)

# Encode
cls_token, _ = vision_encoder(img_tensor)  # (1, 128)

# Classify
logits = classifier(cls_token)
label = 0 if "red" in caption else 1
loss = cross_entropy(logits, label)
```

## Performance Tips

1. **Batch Processing**: Process multiple images together
2. **Data Augmentation**: Random crops, flips, color jitter
3. **Normalization**: Use ImageNet stats for pretrained models
4. **Mixed Precision**: Use float16 for faster training

## Comparison: ViT vs CNN

| Feature | ViT | CNN |
|---------|-----|-----|
| Architecture | Transformer | Convolutional |
| Processing | Sequential patches | Local filters |
| Attention | Global | Local (receptive field) |
| Position | Learned embeddings | Implicit (convolution) |
| Use Case | Large-scale, pretrained | Small-scale, custom |

**Why ViT for μOmni?**
- Unified architecture (same as Thinker)
- Better for multimodal fusion
- Scales well with data

---

**Next:**
- [06_Talker_Codec.md](06_Talker_Codec.md) - How speech is generated
- [07_Training_Workflow.md](07_Training_Workflow.md) - Training the encoder
- [08_Inference_Guide.md](08_Inference_Guide.md) - Using vision in inference

**See Also:**
- [Architecture Overview](02_Architecture_Overview.md)
- [Thinker Deep Dive](03_Thinker_Deep_Dive.md)

