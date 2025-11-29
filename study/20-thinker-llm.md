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

> **Note**: These are the "tiny" configuration values from `configs/thinker_tiny.json`. The code defaults may differ, but config files override them.

| Parameter           | Value        |
| ------------------- | ------------ |
| **Model Dimension** | 256          |
| **Layers**          | 4            |
| **Attention Heads** | 4            |
| **Feedforward Dim** | 1024         |
| **Vocabulary**      | 32000 tokens |
| **Context Length**  | 512-2048     |
| **Parameters**      | ~20.32M      |

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

### 4. **Advanced Optimizations**

- **GQA** (Grouped Query Attention): Reduces KV cache size for faster inference
- **SwiGLU**: Modern activation function for better performance
- **MoE** (Mixture of Experts): Sparse computation (activates subset of parameters)
- **Flash Attention**: 2-4x speedup using optimized kernels
- **Numerical Stability**: Built-in NaN/Inf detection for robust training

## 💻 Implementation

```python
# From omni/thinker.py
class ThinkerLM(nn.Module):
    def __init__(self, vocab: int, n_layers: int = 16, d: int = 512, heads: int = 8, ff: int = 2048, 
                 dropout: float = 0.1, rope_theta: float = 10000, ctx: int = 1024, 
                 use_gqa: bool = False, use_swiglu: bool = True, use_moe: bool = False, 
                 num_experts: int = 8, num_experts_per_tok: int = 2, use_flash: bool = True,
                 compile_model: bool = False) -> None:
        """
        ThinkerLM with optional Qwen3 Omni features and performance optimizations.
        
        Args:
            vocab: vocabulary size
            n_layers: number of transformer layers
            d: model dimension
            heads: number of attention heads
            ff: feedforward dimension
            dropout: dropout rate
            rope_theta: RoPE theta parameter
            ctx: context length
            use_gqa: use Grouped Query Attention (default: False)
            use_swiglu: use SwiGLU activation (default: True)
            use_moe: use Mixture of Experts (default: False)
            num_experts: number of experts for MoE (default: 8)
            num_experts_per_tok: number of experts to activate per token (default: 2)
            use_flash: use Flash Attention for 2-4x speedup (default: True, requires PyTorch 2.0+)
            compile_model: use torch.compile() for 30-50% speedup (default: False, requires PyTorch 2.0+)
        """
        super().__init__()
        
        # Structural check
        if d % heads != 0:
            raise ValueError(f"Model dimension d ({d}) must be divisible by number of heads ({heads}).")
            
        self.tok_emb = nn.Embedding(vocab, d)
        self.pos_cache = None
        self.blocks = nn.ModuleList([
            Block(d, heads, ff, rope_theta, dropout, use_gqa=use_gqa, use_swiglu=use_swiglu,
                  use_moe=use_moe, num_experts=num_experts, num_experts_per_tok=num_experts_per_tok,
                  use_flash=use_flash) 
            for _ in range(n_layers)
        ])
        self.norm = RMSNorm(d)
        self.lm_head = nn.Linear(d, vocab, bias=False)
        self.ctx = ctx
        
        # KV cache for autoregressive generation
        self.kv_cache = None
        self.use_kv_cache = False
```

    def forward(self, idx=None, embeddings=None, attn_mask=None):
        # ... (input handling) ...

        # Process through transformer blocks with stability checks
        for block in self.blocks:
            x, _ = block(x, mask=attn_mask, cache=self.kv_cache)
            if torch.isnan(x).any(): raise RuntimeError("NaN detected")

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
