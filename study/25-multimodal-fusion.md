# Chapter 25: Multimodal Fusion Strategy

[Back to Index](00-INDEX.md)

---

## 🎯 Fusion Approach

μOmni uses **hybrid fusion**: specialized encoders + unified processing.

## 🏗️ Complete Flow

```
🖼️ IMAGE                 🎤 AUDIO              📝 TEXT
   ↓                        ↓                     ↓
Vision Encoder         Audio Encoder         Tokenizer
   ↓                        ↓                     ↓
(1, 128)              (T_a, 192)            Token IDs
   ↓                        ↓                     ↓
Project (128→256)     Project (192→256)     Embed (5000→256)
   ↓                        ↓                     ↓
(1, 1, 256)           (1, T_a, 256)        (1, T_t, 256)
   ↓                        ↓                     ↓
   └────────────────────────┴─────────────────────┘
                           ↓
              Concatenate along sequence dim
                           ↓
              (1, 1+T_a+T_t, 256)
                           ↓
              ┌──────────────────────┐
              │  Thinker (Unified)   │
              │  - Cross-modal attn  │
              │  - All tokens interact│
              └──────────┬───────────┘
                         ↓
                   Text Output
```

## 🎯 Key Principles

### 1. Specialized Encoding
- Each modality uses optimized encoder
- Vision: ViT for spatial patterns
- Audio: Conv+Transformer for temporal
- Text: Tokenization + embeddings

### 2. Common Embedding Space
- All project to d_model=256
- Enables cross-modal attention
- Single unified processing

### 3. Flexible Input
```python
# Text only
input = [text_tokens]

# Image + Text
input = [img_token, text_tokens]

# Audio + Text
input = [audio_tokens, text_tokens]

# All modalities
input = [img_token, audio_tokens, text_tokens]
```

## 📊 Token Budget Example

```
Context: 512 tokens

Image: 1 token (CLS)
Audio (3s): ~38 tokens (at 12.5Hz)
Text prompt: ~20 tokens
---------------------------------
Used: 59 tokens
Available for generation: 453 tokens
```

## 💡 Key Takeaways

✅ **Hybrid fusion** = specialized + unified  
✅ **All modalities** project to 256-dim  
✅ **Concatenate** embeddings before Thinker  
✅ **Cross-modal attention** emerges naturally  
✅ **Flexible input** (any modality combination)

---

[Back to Index](00-INDEX.md)

