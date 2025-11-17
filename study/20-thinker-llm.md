# Chapter 20: The Thinker - Core Language Model

[← Previous: μOmni Overview](19-muomni-overview.md) | [Back to Index](00-INDEX.md)

---

## 🎯 Learning Objectives

By the end of this chapter, you will understand:
- What the Thinker is and why it's the "brain" of μOmni
- Detailed architecture breakdown
- How multimodal embeddings flow through the model
- All the optimizations working together
- Step-by-step inference process
- Why this design is effective

---

## 💡 What is the Thinker?

### The Central Reasoning Engine

**Analogy: The Brain of the Operation**

```
Think of μOmni as a complete sensory system:

EYES (Vision Encoder):
"I see a cat in the image"
→ Converts image to embeddings

EARS (Audio Encoder):
"I hear someone saying 'meow'"
→ Converts audio to embeddings

BRAIN (Thinker): ⭐ THIS IS WHAT WE'RE LEARNING NOW!
Receives all sensory input and:
- Processes all information together
- Understands relationships
- Reasons about the world
- Generates intelligent responses

MOUTH (Talker):
Speaks the response
→ Converts text to speech codes

The THINKER is the brain - it's where all the magic happens!
```

**Why "Thinker"?**

```
The name represents its role:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

THINK about the inputs:
- Text: "What animal is this?"
- Image: [cat photo embeddings]
- Audio: [meow sound embeddings]

REASON about relationships:
- The image shows a furry animal
- The audio sounds like a cat
- The question asks about animals
- Connect: This must be a cat!

GENERATE intelligent response:
"This is a cat. The image shows a feline, 
 and the meow sound confirms it's a cat."

All of this complex reasoning happens in the Thinker!
```

---

## 🏗️ Detailed Architecture Breakdown

```
Token/Embeddings Input (B, T, 256)
    ↓
┌─────────────────────────────┐
│  Transformer Block 1        │
│  - Multi-head Attention     │
│  - Feed-forward Network     │
│  - RMSNorm + Residuals      │
└─────────────┬───────────────┘
              ↓
┌─────────────────────────────┐
│  Transformer Block 2        │
└─────────────┬───────────────┘
              ↓
┌─────────────────────────────┐
│  Transformer Block 3        │
└─────────────┬───────────────┘
              ↓
┌─────────────────────────────┐
│  Transformer Block 4        │
└─────────────┬───────────────┘
              ↓
     RMSNorm
              ↓
   LM Head (Linear)
              ↓
   Logits (B, T, vocab_size)
```

## 📊 Specifications

| Parameter | Value |
|-----------|-------|
| **Model Dimension** | 256 |
| **Layers** | 4 |
| **Attention Heads** | 4 |
| **Feedforward Dim** | 1024 |
| **Vocabulary** | 5000 tokens |
| **Context Length** | 512-2048 |
| **Parameters** | ~20.32M |

## 🔑 Key Features

### 1. **Causal Attention**
- Autoregressive generation
- Each token attends only to previous tokens
- Enables text generation one token at a time

### 2. **RoPE Positional Encoding**
- Rotary position embeddings
- Better extrapolation to longer sequences
- No additional parameters

### 3. **KV Caching**
- Caches key/value tensors during generation
- Speeds up autoregressive decoding from O(T²) to O(T)
- Essential for interactive applications

### 4. **Optional Optimizations**
- **GQA** (Grouped Query Attention): Reduces KV parameters
- **SwiGLU**: Modern activation function
- **MoE** (Mixture of Experts): Sparse computation
- **Flash Attention**: 2-4x speedup

## 💻 Implementation

```python
# From omni/thinker.py
class ThinkerLM(nn.Module):
    def __init__(self, vocab, n_layers=4, d=256, heads=4, 
                 ff=1024, dropout=0.1, rope_theta=10000, ctx=512):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab, d)
        self.blocks = nn.ModuleList([
            Block(d, heads, ff, rope_theta, dropout) 
            for _ in range(n_layers)
        ])
        self.norm = RMSNorm(d)
        self.lm_head = nn.Linear(d, vocab, bias=False)
        self.ctx = ctx
        self.kv_cache = None
    
    def forward(self, idx=None, embeddings=None, attn_mask=None):
        # Accept either token IDs or embeddings (multimodal)
        if embeddings is not None:
            x = embeddings
        elif idx is not None:
            x = self.tok_emb(idx)
        else:
            raise ValueError("Provide idx or embeddings")
        
        # Process through transformer blocks
        for block in self.blocks:
            x, _ = block(x, mask=attn_mask, cache=self.kv_cache)
        
        x = self.norm(x)
        logits = self.lm_head(x)
        return logits
```

## 💡 Key Takeaways

✅ **Decoder-only** architecture (GPT-style)  
✅ **256-dim embeddings**, 4 layers, 4 heads  
✅ **Causal attention** for autoregressive generation  
✅ **Accepts multimodal embeddings** (text + image + audio)  
✅ **KV caching** for fast inference  
✅ **~20.32M parameters**

---

[Back to Index](00-INDEX.md)

