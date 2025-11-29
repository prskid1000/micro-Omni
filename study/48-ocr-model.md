# Chapter 48: OCR Model - Text Extraction from Images

[← Previous: Model Export & Deployment](46-model-export-deployment.md) | [Back to Index](00-INDEX.md) | [Next: Future Extensions →](45-future-extensions.md)

---

## 🎯 Learning Objectives

By the end of this chapter, you will understand:

- What OCR is and why we need it
- Architecture: ViT encoder + Transformer decoder with cross-attention
- How the model extracts text from images
- RoPE positional encoding for text sequences
- Cross-attention mechanism connecting vision and text
- Training process and data requirements
- Integration with multimodal understanding

---

## 💡 What is OCR?

### Optical Character Recognition

**Analogy: Reading a Book**

```
Think of OCR like reading text from an image:

IMAGE WITH TEXT:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Photo of a sign: "STOP"
↓
Human can read: "S-T-O-P"
↓
Computer needs to: Extract "STOP" as text

OCR MODEL:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Input: Image (224×224×3)
↓
Vision Encoder: Understands image content
↓
Text Decoder: Generates characters autoregressively
↓
Output: "STOP" (text string)

The OCR model is the TEXT READER:
Images → Text extraction!
```

**Why Do We Need This?**

```
Problem: Images contain text, but models see pixels!
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Multimodal understanding needs:
❌ Vision encoder sees: "orange pixels, white pixels"
❌ Doesn't know: "This says 'STOP'"
❌ Can't extract: Text content from images

Solution: OCR Model!
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

OCR enables:
✅ Extract text from images
✅ Understand document content
✅ Process screenshots, signs, labels
✅ Enhance multimodal reasoning
✅ Combine visual + textual understanding

Use cases:
- Document processing
- Screenshot analysis
- Sign reading
- Handwritten text recognition
- Multimodal question answering
```

---

## 🏗️ Architecture Overview

### Two-Component System

```
OCR MODEL ARCHITECTURE:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Image (B, 3, 224, 224)
    ↓
[Vision Encoder (ViT)]
    ↓
Image Features (B, N, 192)
    ↓
[Image Projection]
    ↓
Image Features (B, N, 384)
    ↓
[Text Decoder]
    ├─ Self-Attention (causal, with RoPE)
    ├─ Cross-Attention (to image features)
    └─ Feedforward (SwiGLU)
    ↓
Character Logits (B, T, vocab_size)
    ↓
Text: "STOP"
```

### Component Breakdown

**1. Vision Encoder (ViT-Tiny)**

- Input: Image `(B, 3, 224, 224)`
- Process: Patch embedding + Transformer layers
- Output: Grid features `(B, N, 192)` where N = (224/16)² = 196 patches
- Purpose: Extract visual features from image patches

**2. Image Projection**

- Projects vision features from 192-dim to decoder dimension (384-dim)
- Aligns vision and text embeddings

**3. Text Decoder**

- Input: Character token IDs `(B, T)`
- Process: Autoregressive generation with cross-attention
- Output: Character logits `(B, T, vocab_size)`
- Purpose: Generate text from visual features

---

## 🔧 Decoder Architecture

### OCRDecoderBlock Structure

```
DECODER BLOCK (per layer):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Input: Text embeddings (B, T, 1024)
    ↓
[1. Self-Attention (Causal)]
    ├─ Norm1 (RMSNorm)
    ├─ Attention (with RoPE)
    └─ Residual + Dropout
    ↓
[2. Cross-Attention (to Image)]
    ├─ Norm2 (RMSNorm)
    ├─ MultiheadAttention (query=text, key/value=image)
    └─ Residual + Dropout
    ↓
[3. Feedforward]
    ├─ Norm3 (RMSNorm)
    ├─ MLP (SwiGLU)
    └─ Residual + Dropout
    ↓
Output: (B, T, 1024)
```

### Key Features

**1. Separate Norm Instances**

- Each sub-layer has its own `RMSNorm` instance
- Prevents parameter sharing across layers
- Matches Thinker's Block pattern

**2. RoPE for Self-Attention**

- Rotary Position Embedding applied to queries and keys
- Enables relative position understanding
- Supports longer sequences than training

**3. Cross-Attention Mechanism**

- Text tokens (query) attend to image features (key/value)
- Allows decoder to "look" at relevant image regions
- No RoPE needed (cross-attention doesn't use positional encoding)

**4. Causal Masking**

- Self-attention is causal (can only see previous tokens)
- Enables autoregressive text generation
- Prevents information leakage

---

## 📐 Mathematical Details

### Forward Pass Flow

```
1. IMAGE ENCODING:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Image: (B, 3, 224, 224)
    ↓ ViT Encoder
Grid: (B, 196, 192)  [196 = (224/16)² patches]
    ↓ Image Projection
Img Features: (B, 196, 384)

2. TEXT EMBEDDING:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Text IDs: (B, T)  [T = sequence length]
    ↓ Character Embedding
Text Embed: (B, T, 384)

3. DECODER PROCESSING:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
For each decoder layer:
    x = Text Embed (B, T, 384)

    # Self-attention (causal)
    x = x + Dropout(SelfAttn(Norm1(x), RoPE))

    # Cross-attention
    x = x + Dropout(CrossAttn(Norm2(x), Img Features))

    # Feedforward
    x = x + Dropout(MLP(Norm3(x)))

4. OUTPUT:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
x = Norm(x)  # Final normalization
Logits = Linear(x)  # (B, T, vocab_size)
```

### Attention Mechanisms

**Self-Attention (Causal)**

```
Query: Q = Text Embeddings (B, T, 384)
Key:   K = Text Embeddings (B, T, 384)
Value: V = Text Embeddings (B, T, 384)

Apply RoPE to Q and K
Apply causal mask (lower triangular)
Attention = Softmax(QK^T / √d) V
```

**Cross-Attention**

```
Query: Q = Text Embeddings (B, T, 384)
Key:   K = Image Features (B, N, 384)
Value: V = Image Features (B, N, 384)

No RoPE (cross-attention doesn't need position)
No causal mask (can attend to all image patches)
Attention = Softmax(QK^T / √d) V
```

---

## 🎓 Training Process

### Data Format

```
CSV Format:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
image,text
data/ocr/img1.jpg,"STOP"
data/ocr/img2.jpg,"Hello World"
data/ocr/img3.jpg,"123 Main St"

Requirements:
- Images: Any format (JPG, PNG, etc.)
- Text: Plain text strings
- Character vocabulary: Built from dataset
```

### Training Objective

```
TEACHER FORCING:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Input:  [BOS] + "S" + "T" + "O"
Target: "S" + "T" + "O" + [EOS]

Model predicts next character given:
- Previous characters (self-attention)
- Image features (cross-attention)

Loss: Cross-entropy on character predictions

**Expected Validation Loss:**
- Target Loss: < 1.0
- Target Character Accuracy: > 90%
- Good: loss < 0.8, accuracy > 95%
- Excellent: loss < 0.5, accuracy > 98%
Ignore: PAD tokens (index 0)
```

### Training Configuration

```json
{
  "img_size": 224,
  "patch": 16,
  "vision_d_model": 192,
  "vision_layers": 2,
  "vision_heads": 3,
  "vision_d_ff": 768,
  "decoder_d_model": 384,
  "decoder_layers": 3,
  "decoder_heads": 6,
  "decoder_d_ff": 1536,
  "dropout": 0.1,
  "use_gqa": true,
  "use_swiglu": true,
  "use_flash": true,
  "rope_theta": 10000.0,
  "vocab_size": "<dynamic from dataset>"
}
```

---

## 🚀 Key Features

### 1. Modern Architecture

```
✅ ViT Encoder: State-of-the-art vision processing
✅ Transformer Decoder: Autoregressive text generation
✅ Cross-Attention: Connects vision and text
✅ RoPE: Relative position encoding
✅ SwiGLU: Modern activation function
✅ Flash Attention: 2-4x speedup (optional)
```

### 2. Optimizations

```
✅ Separate Norm Instances: Per-layer normalization
✅ KV Caching: Fast autoregressive generation
✅ Gradient Accumulation: Train with larger effective batch size
✅ Mixed Precision: FP16 training for efficiency
✅ Gradient Clipping: Prevents exploding gradients
```

### 3. Integration

```
✅ Character Vocabulary: Built dynamically from dataset
✅ Variable Length: Handles different text lengths
✅ Multimodal Ready: Can be integrated with Thinker
✅ Inference Support: KV caching for fast generation
```

---

## 💻 Usage Example

### Training

```python
# Train OCR model
python train_ocr.py --config configs/ocr_tiny.json
```

### Inference

```python
from omni.ocr_model import OCRModel
import torch
from PIL import Image
import torchvision.transforms as T

# Load model
model = OCRModel(...)
model.load_state_dict(torch.load("checkpoints/ocr_tiny/model.pt")["model"])
model.eval()

# Process image
image = Image.open("sign.jpg")
transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
])
img_tensor = transform(image).unsqueeze(0)  # (1, 3, 224, 224)

# Generate text
text_ids = torch.tensor([[1]])  # BOS token
model.decoder.enable_kv_cache(True)

for _ in range(max_length):
    logits = model(img_tensor, text_ids)
    next_id = logits[0, -1].argmax().item()
    if next_id == 2:  # EOS
        break
    text_ids = torch.cat([text_ids, torch.tensor([[next_id]])], dim=1)

# Decode text
text = decode_text(text_ids)
print(f"Extracted text: {text}")
```

---

## 🔬 Architecture Comparison

### Similar to Modern OCR Models

**VISTA-OCR (2024)**

- ✅ ViT encoder + Transformer decoder
- ✅ Cross-attention from text to image
- ✅ Autoregressive text generation

**UPOCR (2023)**

- ✅ Vision Transformer encoder
- ✅ Transformer decoder with cross-attention
- ✅ Unified image-to-text approach

**Our Implementation**

- ✅ Matches modern OCR architectures
- ✅ Uses proven components (ViT, Transformer)
- ✅ Optimized with Flash Attention, RoPE, SwiGLU

---

## 📊 Model Parameters

### Tiny Configuration

```
Vision Encoder (ViT-Tiny):
- Layers: 2
- Heads: 3
- Dimension: 192
- FFN: 768
- Patches: 196 (14×14)

Text Decoder:
- Layers: 3
- Heads: 6
- Dimension: 384
- FFN: 1536
- Vocabulary: Dynamic (from dataset)

Total Parameters: ~21.5M (from calculate_model_size.py)
```

---

## 🎯 Key Takeaways

✅ **OCR extracts text from images** using vision encoder + text decoder  
✅ **Cross-attention** connects visual features to text generation  
✅ **RoPE** enables relative position understanding in text sequences  
✅ **Separate norms** per layer (matches Thinker pattern)  
✅ **Autoregressive generation** with causal masking  
✅ **KV caching** for fast inference  
✅ **Character-level** vocabulary built from dataset  
✅ **Modern architecture** aligned with state-of-the-art OCR models

---

## 🔗 Related Chapters

- [Chapter 22: Vision Encoder](22-vision-encoder.md) - ViT architecture
- [Chapter 13: Decoder-Only LLM](13-decoder-only-llm.md) - Transformer decoder
- [Chapter 08: Positional Encoding](08-positional-encoding.md) - RoPE details
- [Chapter 07: Attention Mechanism](07-attention-mechanism.md) - Attention basics
- [Chapter 16: SwiGLU Activation](16-swiglu-activation.md) - SwiGLU details
- [Chapter 15: GQA Attention](15-gqa-attention.md) - Grouped Query Attention

---

[← Previous: Model Export & Deployment](46-model-export-deployment.md) | [Back to Index](00-INDEX.md) | [Next: Future Extensions →](45-future-extensions.md)
