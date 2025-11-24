# Chapter 15: Grouped Query Attention (GQA)

[← Previous: KV Caching](14-kv-caching.md) | [Back to Index](00-INDEX.md) | [Next: SwiGLU →](16-swiglu-activation.md)

---

## 🎯 Learning Objectives

By the end of this chapter, you will understand:
- The memory bottleneck in multi-head attention
- How GQA reduces parameters and memory
- The trade-off between MHA, GQA, and MQA
- Why GQA is a sweet spot for efficiency
- How μOmni uses GQA

---

## ❓ The Problem: KV Cache is Expensive!

### Understanding the Bottleneck

We just learned KV caching is amazing for speed! But there's a catch...

**Analogy: Storing Textbooks**

```
Imagine you're a student with 8 classes:

OPTION 1: Full Library (Multi-Head Attention)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Class 1 → Full textbook set (1000 pages)
Class 2 → Full textbook set (1000 pages)
Class 3 → Full textbook set (1000 pages)
...
Class 8 → Full textbook set (1000 pages)

Total: 8,000 pages stored!
Problem: Carrying 8,000 pages is HEAVY! 📚📚📚

OPTION 2: Shared Library (Grouped Query Attention)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Classes 1-4 → Share ONE textbook set (1000 pages)
Classes 5-8 → Share ONE textbook set (1000 pages)

Total: 2,000 pages stored!
Problem solved: Only 25% of the weight! 🎒

Can classes 1-4 still learn? YES!
They take turns using the same textbook.
```

**The Technical Problem:**

```
Multi-Head Attention Memory Usage:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

8 attention heads, each with separate K, V:

Q: 8 heads × 64 dim = 512 dimensions
K: 8 heads × 64 dim = 512 dimensions  ← EXPENSIVE!
V: 8 heads × 64 dim = 512 dimensions  ← EXPENSIVE!

KV Cache size (per layer, 1000 tokens):
K: 8 × 64 × 1000 = 512,000 values
V: 8 × 64 × 1000 = 512,000 values
Total: 1,024,000 values = 4 MB per layer

For 32 layers (GPT-3 scale):
4 MB × 32 = 128 MB just for KV cache!

For long context (10,000 tokens):
128 MB × 10 = 1.28 GB just for KV cache! 😱

This is a HUGE bottleneck for:
- Long context generation
- Batch processing
- Limited GPU memory
```

---

## 💡 The Solution: Grouped Query Attention (GQA)

### The Key Insight: Share K, V Across Multiple Q Heads!

**The Brilliant Idea:**

```
Do we REALLY need 8 separate K, V pairs?
What if multiple Query heads shared the same K, V?

Think about it:
- Q heads ask different questions
- But they can look at the SAME information (K, V)!

Like: Multiple students reading the SAME textbook
      But asking different questions about it!
```

**How GQA Works:**

```
Standard Multi-Head Attention (MHA):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Q head 0 ──→ K head 0, V head 0
Q head 1 ──→ K head 1, V head 1
Q head 2 ──→ K head 2, V head 2
Q head 3 ──→ K head 3, V head 3
Q head 4 ──→ K head 4, V head 4
Q head 5 ──→ K head 5, V head 5
Q head 6 ──→ K head 6, V head 6
Q head 7 ──→ K head 7, V head 7

8 Q heads, 8 K heads, 8 V heads
Total KV: 16 head-sets

Grouped Query Attention (GQA):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Q head 0 ─┐
Q head 1 ─┼──→ K head 0, V head 0  ← SHARED!
Q head 2 ─┤
Q head 3 ─┘

Q head 4 ─┐
Q head 5 ─┼──→ K head 1, V head 1  ← SHARED!
Q head 6 ─┤
Q head 7 ─┘

8 Q heads, 2 K heads, 2 V heads
Total KV: 4 head-sets (4x reduction!)

Group size: 4 Q heads per KV head
```

---

## 🏗️ Architecture Comparison

### Standard Multi-Head Attention (MHA)

```
Configuration: 8 heads, d_model=512

Per head dimension: 512 / 8 = 64

Q projection: 512 → 8 × 64 = 512 dims
K projection: 512 → 8 × 64 = 512 dims
V projection: 512 → 8 × 64 = 512 dims

Parameters:
Q: 512 × 512 = 262,144
K: 512 × 512 = 262,144  ← Full K
V: 512 × 512 = 262,144  ← Full V
O: 512 × 512 = 262,144
─────────────────────
Total: 1,048,576 parameters

KV Cache (1000 tokens):
K: 8 heads × 64 × 1000 = 512,000 values
V: 8 heads × 64 × 1000 = 512,000 values
Total: 1,024,000 values = 4 MB
```

### Grouped Query Attention (GQA)

```
Configuration: 8 Q heads, 2 KV heads, d_model=512

Per head dimension: 512 / 8 = 64

Q projection: 512 → 8 × 64 = 512 dims
K projection: 512 → 2 × 64 = 128 dims  ← 4x smaller!
V projection: 512 → 2 × 64 = 128 dims  ← 4x smaller!

Parameters:
Q: 512 × 512 = 262,144
K: 512 × 128 = 65,536   ← 4x smaller!
V: 512 × 128 = 65,536   ← 4x smaller!
O: 512 × 512 = 262,144
─────────────────────
Total: 655,360 parameters (37% reduction)

KV Cache (1000 tokens):
K: 2 heads × 64 × 1000 = 128,000 values  ← 4x smaller!
V: 2 heads × 64 × 1000 = 128,000 values  ← 4x smaller!
Total: 256,000 values = 1 MB (4x smaller!)
```

---

## 📊 The Spectrum: MQA vs GQA vs MHA

### Three Options for K, V Sharing

```
OPTION 1: Multi-Query Attention (MQA)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ALL Q heads share ONE K, V pair!

Q head 0 ─┐
Q head 1 ─┤
Q head 2 ─┼──→ K head 0, V head 0  ← ONE for ALL!
Q head 3 ─┤
...       │
Q head 7 ─┘

KV heads: 1 (most aggressive sharing)
Parameters: Smallest
KV Cache: Smallest (8x reduction)
Quality: Slightly reduced

OPTION 2: Grouped Query Attention (GQA)  ⭐ Sweet spot!
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Groups of Q heads share K, V!

Q heads 0-3 ──→ K head 0, V head 0
Q heads 4-7 ──→ K head 1, V head 1

KV heads: 2-4 (moderate sharing)
Parameters: Medium
KV Cache: Medium (2-4x reduction)
Quality: Near-identical to MHA

OPTION 3: Multi-Head Attention (MHA)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Each Q head has its own K, V!

Q head 0 ──→ K head 0, V head 0
Q head 1 ──→ K head 1, V head 1
...
Q head 7 ──→ K head 7, V head 7

KV heads: 8 (no sharing)
Parameters: Largest
KV Cache: Largest
Quality: Best (baseline)
```

### Comparison Table

| Feature | MQA | GQA | MHA |
|---------|-----|-----|-----|
| **Q Heads** | 8 | 8 | 8 |
| **KV Heads** | 1 | 2-4 | 8 |
| **KV Parameters** | 1x | 2-4x | 8x |
| **KV Cache Size** | 1x | 2-4x | 8x |
| **Inference Speed** | Fastest | Fast | Baseline |
| **Quality** | Good | Excellent | Best |
| **Use Case** | Extreme efficiency | Balanced | Maximum quality |

---

## ⚖️ Why GQA is the Sweet Spot

```
GQA balances efficiency and quality:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Advantages over MHA:
✅ 2-4x smaller KV cache (memory efficient)
✅ 2-4x fewer KV parameters (faster training)
✅ Faster inference (less memory bandwidth)
✅ Better batch processing (more sequences fit in memory)

Advantages over MQA:
✅ Better quality (multiple KV perspectives)
✅ More expressive (not bottlenecked on single KV)
✅ Empirically proven to match MHA quality

Real-world impact:
- LLaMA 2: Uses GQA with 8 Q heads, 2 KV heads
- GPT-4: Rumored to use GQA
- μOmni: Optional GQA support

Why it works:
- Multiple Q heads can ask different questions
- But 2-4 KV "databases" are enough to answer them!
- Like: 8 students sharing 2 textbooks vs 8 textbooks
```

---

## 💻 Implementation Details

### How GQA Repeats K, V

```python
class GroupedQueryAttention:
    def __init__(self, d_model=512, q_heads=8, kv_heads=2):
        self.q_heads = q_heads
        self.kv_heads = kv_heads
        self.d_k = d_model // q_heads
        self.group_size = q_heads // kv_heads  # 8 / 2 = 4
        
        # Projections
        self.W_q = nn.Linear(d_model, q_heads * self.d_k)     # 512 → 512
        self.W_k = nn.Linear(d_model, kv_heads * self.d_k)    # 512 → 128
        self.W_v = nn.Linear(d_model, kv_heads * self.d_k)    # 512 → 128
        self.W_o = nn.Linear(d_model, d_model)
    
    def forward(self, x):
        B, T, D = x.shape
        
        # Project
        Q = self.W_q(x).view(B, T, self.q_heads, self.d_k)   # (B, T, 8, 64)
        K = self.W_k(x).view(B, T, self.kv_heads, self.d_k)  # (B, T, 2, 64)
        V = self.W_v(x).view(B, T, self.kv_heads, self.d_k)  # (B, T, 2, 64)
        
        # Repeat K, V to match Q heads
        # Each KV head serves group_size Q heads
        K = K.repeat_interleave(self.group_size, dim=2)  # (B, T, 8, 64)
        V = V.repeat_interleave(self.group_size, dim=2)  # (B, T, 8, 64)
        
        # Now K, V match Q's shape!
        # K[..., 0:4, :] are all copies of original K[..., 0, :]
        # K[..., 4:8, :] are all copies of original K[..., 1, :]
        
        # Standard attention computation
        Q = Q.transpose(1, 2)  # (B, 8, T, 64)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)
        
        scores = Q @ K.transpose(-2, -1) / math.sqrt(self.d_k)
        attn = torch.softmax(scores, dim=-1)
        output = attn @ V
        
        output = output.transpose(1, 2).reshape(B, T, D)
        return self.W_o(output)
```

**Key Insight: repeat_interleave**

```
Original K (2 heads):
K_head_0: [k0_data]
K_head_1: [k1_data]

After repeat_interleave(4):
K_head_0: [k0_data]  ← Copy of K_head_0
K_head_1: [k0_data]  ← Copy of K_head_0
K_head_2: [k0_data]  ← Copy of K_head_0
K_head_3: [k0_data]  ← Copy of K_head_0
K_head_4: [k1_data]  ← Copy of K_head_1
K_head_5: [k1_data]  ← Copy of K_head_1
K_head_6: [k1_data]  ← Copy of K_head_1
K_head_7: [k1_data]  ← Copy of K_head_1

Now compatible with 8 Q heads!
Q heads 0-3 all attend to k0_data
Q heads 4-7 all attend to k1_data
```

---

## 🎯 μOmni's GQA Configuration

```python
# μOmni supports both MHA and GQA

# Multi-Head Attention (default):
thinker = ThinkerLM(
    d_model=256,
    heads=4,
    use_gqa=False  # MHA: 4 Q heads, 4 KV heads
)

# Grouped Query Attention (optional):
thinker = ThinkerLM(
    d_model=256,
    heads=4,
    use_gqa=True   # GQA: 4 Q heads, 2 KV heads (2x reduction)
)

When to use GQA in μOmni:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✅ Long context generation (>1024 tokens)
✅ Limited GPU memory (6GB or less)
✅ Batch inference (process multiple sequences)
✅ Faster inference needed

When to use MHA:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✅ Maximum quality needed
✅ Plenty of GPU memory (12GB+)
✅ Short context (<512 tokens)
✅ Training speed not critical
```

---

## 💡 Key Takeaways

✅ **GQA** shares K, V heads across groups of Q heads  
✅ **Memory efficient**: 2-4x smaller KV cache than MHA  
✅ **Parameter efficient**: Fewer KV projection parameters  
✅ **Quality preserved**: Near-identical performance to MHA  
✅ **Sweet spot**: Better than MQA (quality) and MHA (efficiency)  
✅ **Modern standard**: Used in LLaMA 2, recommended for production  
✅ **μOmni supports GQA**: Optional for memory-constrained setups

---

## 🎓 Self-Check Questions

1. What problem does GQA solve?
2. How does GQA differ from standard MHA?
3. What's the trade-off between MQA, GQA, and MHA?
4. Why is GQA a "sweet spot"?
5. When should you use GQA vs MHA?

<details>
<summary>📝 Click to see answers</summary>

1. GQA solves the memory bottleneck from storing separate K, V for each attention head (large KV cache)
2. GQA uses fewer KV heads than Q heads (e.g., 8 Q heads share 2 KV heads), while MHA has equal numbers (8 Q, 8 KV)
3. MQA: Most efficient (1 KV head) but slightly lower quality. MHA: Best quality but most memory. GQA: Balanced (2-4 KV heads) with near-MHA quality and much better efficiency
4. GQA provides most of MHA's quality benefits while achieving significant memory/speed improvements (2-4x reduction in KV cache)
5. Use GQA when: limited memory, long context, batch processing. Use MHA when: maximum quality needed, plenty of memory, short context
</details>

---

[Continue to Chapter 16: SwiGLU Activation →](16-swiglu-activation.md)

**Chapter Progress:** Advanced Architecture ●●●○ (3/4 complete)

---

