# Chapter 07: Attention Mechanism Deep Dive

[← Previous: Embeddings](06-embeddings-explained.md) | [Back to Index](00-INDEX.md) | [Next: Positional Encoding →](08-positional-encoding.md)

---

## 🎯 Learning Objectives

- Deep understanding of attention mechanism
- Query, Key, Value mathematics
- Multi-head attention in detail
- Attention patterns and interpretability
- How μOmni implements attention

---

## 🔍 The Attention Mechanism Explained

### Core Formula

```
Attention(Q, K, V) = softmax(QKᵀ / √dₖ) V

Where:
- Q (Query): "What am I looking for?"
- K (Key): "What do I offer?"
- V (Value): "My actual content"
- dₖ: Key dimension (for scaling)
```

### Step-by-Step Breakdown

```
1. SCORE COMPUTATION
   scores = Q @ Kᵀ  (dot product)
   
2. SCALING
   scores = scores / √dₖ  (prevent large values)
   
3. SOFTMAX
   weights = softmax(scores)  (normalize to probabilities)
   
4. WEIGHTED SUM
   output = weights @ V  (combine values)
```

---

## 🎯 Detailed Example: Attention in Action

```
Sentence: "The cat sat on mat"
Process: What does "sat" attend to?

Step 1: Create Q, K, V for each word
        Q_sat = [0.5, 0.8]   (what "sat" is looking for)
        K_the = [0.1, 0.2]   K_cat = [0.7, 0.9]
        K_sat = [0.5, 0.8]   K_on  = [0.3, 0.1]
        K_mat = [0.2, 0.4]

Step 2: Compute attention scores
        score_the = Q_sat · K_the = (0.5)(0.1) + (0.8)(0.2) = 0.21
        score_cat = Q_sat · K_cat = (0.5)(0.7) + (0.8)(0.9) = 1.07
        score_sat = Q_sat · K_sat = (0.5)(0.5) + (0.8)(0.8) = 0.89
        score_on  = Q_sat · K_on  = (0.5)(0.3) + (0.8)(0.1) = 0.23
        score_mat = Q_sat · K_mat = (0.5)(0.2) + (0.8)(0.4) = 0.42

Step 3: Scale (√dₖ = √2 = 1.41)
        scores = [0.15, 0.76, 0.63, 0.16, 0.30]

Step 4: Softmax (convert to probabilities)
        weights = [0.08, 0.34, 0.29, 0.08, 0.21]
                   ↑     ↑     ↑     ↑     ↑
                  the  cat   sat   on   mat

        "sat" attends mostly to "cat" (34%) and itself (29%)!

Step 5: Weighted sum of values
        output = 0.08*V_the + 0.34*V_cat + 0.29*V_sat + ...
```

---

## 👥 Multi-Head Attention

### Why Multiple Heads?

```
Single head = One perspective
Multi-head = Multiple perspectives simultaneously

Example: "The quick brown fox jumped"

Head 1 (Syntax):          Head 2 (Semantics):
"quick" → "brown"         "fox" → "jumped"
(adjectives cluster)      (subject-verb)

Head 3 (Dependencies):    Head 4 (Position):
"fox" → "The"             Adjacent word relations
(long-range reference)    (local context)
```

### Mathematical Formulation

```python
class MultiHeadAttention:
    def __init__(self, d_model, num_heads):
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # Separate Q, K, V projections for each head
        self.W_q = Linear(d_model, d_model)
        self.W_k = Linear(d_model, d_model)
        self.W_v = Linear(d_model, d_model)
        self.W_o = Linear(d_model, d_model)  # Output projection
    
    def forward(self, x):
        B, T, D = x.shape
        
        # 1. Project to Q, K, V
        Q = self.W_q(x)  # (B, T, D)
        K = self.W_k(x)
        V = self.W_v(x)
        
        # 2. Split into multiple heads
        Q = Q.view(B, T, self.num_heads, self.d_k).transpose(1, 2)  # (B, H, T, d_k)
        K = K.view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        
        # 3. Scaled dot-product attention per head
        scores = (Q @ K.transpose(-2, -1)) / sqrt(self.d_k)
        attn = softmax(scores, dim=-1)
        out = attn @ V  # (B, H, T, d_k)
        
        # 4. Concatenate heads
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        
        # 5. Final projection
        return self.W_o(out)
```

---

## 🎭 Attention Patterns

### Pattern 1: Local Attention

```
"The quick brown fox"

Attention weights:
     The quick brown fox
The   ██  ▓▓   ░░    ░░
quick ▓▓  ██   ▓▓    ░░
brown ░░  ▓▓   ██    ▓▓
fox   ░░  ░░   ▓▓    ██

Pattern: Strong diagonal (attend to nearby words)
Use case: Local syntax, immediate context
```

---

### Pattern 2: Long-Range Attention

```
"The fox that the hunter saw ran away"

Key relationship: "fox" (position 1) ↔ "ran" (position 7)

Attention from "ran":
     The fox that the hunter saw ran away
ran  ░░  ██  ░░   ░░  ░░     ░░  ██  ▓▓

Pattern: Long-distance connection (subject-verb)
Use case: Grammatical dependencies
```

---

### Pattern 3: Broadcast Attention

```
"<CLS> This is a sentence"

<CLS> token attends to all:
      <CLS> This is a sentence
<CLS>  ██   ▓▓  ▓▓ ▓▓  ▓▓

Pattern: One token aggregates from all
Use case: Sentence representation (BERT-style)
```

---

### Pattern 4: Causal Attention

```
"The cat sat on"

In autoregressive generation:
      The cat sat on
The   ██  ░░  ░░  ░░   (can only see "The")
cat   ▓▓  ██  ░░  ░░   (can see "The cat")
sat   ▓▓  ▓▓  ██  ░░   (can see up to "sat")
on    ▓▓  ▓▓  ▓▓  ██   (can see up to "on")

Pattern: Lower triangular matrix
Use case: GPT-style generation
```

📌 **μOmni's Thinker uses causal attention** (autoregressive generation)

---

## ⚙️ Attention Variants in μOmni

### 1. Standard Multi-Head Attention (MHA)

```
Q, K, V have same number of heads:

Q: 4 heads × 64 dim = 256 total
K: 4 heads × 64 dim = 256 total
V: 4 heads × 64 dim = 256 total

Memory: O(T²) for attention matrix
Speed: Fast with parallelization
```

---

### 2. Grouped Query Attention (GQA)

```
Q has more heads than K, V:

Q: 4 query heads × 64 dim = 256 total
K: 2 KV heads × 64 dim = 128 total (shared)
V: 2 KV heads × 64 dim = 128 total (shared)

Grouping:
Q_head_0, Q_head_1 → share K_head_0, V_head_0
Q_head_2, Q_head_3 → share K_head_1, V_head_1

Benefits:
✅ Fewer KV parameters (memory efficient)
✅ Faster KV caching during generation
✅ Similar performance to MHA
```

📌 **μOmni supports GQA** (optional, for efficiency)

---

## 🚀 Attention Optimizations

### 1. Flash Attention

```python
# Standard attention (slow)
scores = Q @ K.transpose(-1, -2) / sqrt(d_k)  # Materialize full matrix
attn = softmax(scores)
output = attn @ V

# Flash Attention (fast)
output = torch.nn.functional.scaled_dot_product_attention(
    Q, K, V, is_causal=True
)
# Fused operation, no intermediate matrix, 2-4x faster!
```

Benefits:
- ✅ 2-4x faster
- ✅ Lower memory usage
- ✅ Same mathematical result

📌 **μOmni uses Flash Attention** (PyTorch 2.0+)

---

### 2. KV Caching

For autoregressive generation:

```
Without KV caching:
Step 1: Process "The"           → compute K, V for "The"
Step 2: Process "The cat"       → recompute K, V for "The", compute for "cat"
Step 3: Process "The cat sat"   → recompute all K, V again

With KV caching:
Step 1: Process "The"           → compute & cache K, V for "The"
Step 2: Process "cat"           → reuse cached "The", compute & add "cat"
Step 3: Process "sat"           → reuse cached "The cat", add "sat"

Speed-up: O(T²) → O(T) per token!
```

```python
class AttentionWithCache:
    def forward(self, x, cache=None):
        Q, K, V = self.project(x)
        
        if cache is not None:
            # Append to cached K, V
            K = torch.cat([cache['K'], K], dim=2)
            V = torch.cat([cache['V'], V], dim=2)
        
        # Attention
        output = attention(Q, K, V)
        
        # Update cache
        new_cache = {'K': K, 'V': V}
        return output, new_cache
```

📌 **μOmni implements KV caching** for fast generation

---

## 📊 Attention Complexity

### Time and Space Complexity

```
Input sequence length: T
Model dimension: D
Number of heads: H

Operations:
1. QKV projection:   O(T × D²)
2. Attention scores: O(T² × D)
3. Softmax:          O(T²)
4. Weighted sum:     O(T² × D)
5. Output project:   O(T × D²)

Total: O(T² × D + T × D²)

Bottleneck: T² term (quadratic in sequence length!)
```

### Memory Usage

```
Attention matrix: (B, H, T, T)

For μOmni:
B = 1 (inference)
H = 4 (heads)
T = 512 (context)

Memory = 1 × 4 × 512 × 512 × 4 bytes (float32)
       = 4,194,304 bytes ≈ 4 MB

For T=2048: ~67 MB (16x more!)

This is why long context is expensive!
```

---

## 🎨 Visualizing Attention

### Attention Heatmaps

```python
# Extract attention weights
attn_weights = model.get_attention_weights(input_text)
# Shape: (num_layers, num_heads, seq_len, seq_len)

# Visualize for layer 2, head 0
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10, 10))
sns.heatmap(attn_weights[2, 0].cpu().numpy(), 
            xticklabels=tokens, 
            yticklabels=tokens,
            cmap='viridis')
plt.title('Attention Pattern: Layer 2, Head 0')
plt.show()
```

Example output:
```
         The  cat  sat  on  the  mat
The      ███  ▓▓   ░░   ░░  ░░   ░░
cat      ▓▓   ███  ▓▓   ░░  ░░   ░░
sat      ▓▓   ▓▓   ███  ▓▓  ░░   ░░
on       ▓▓   ░░   ▓▓   ███ ▓▓   ░░
the      ▓▓   ░░   ░░   ▓▓  ███  ▓▓
mat      ▓▓   ▓▓   ░░   ▓▓  ▓▓   ███
```

---

## 🧮 Attention vs Other Mechanisms

### Comparison Table

| Mechanism | Complexity | Parallelizable | Long-range | Memory |
|-----------|------------|----------------|------------|--------|
| **Attention** | O(T²) | ✅ Yes | ✅ Yes | O(T²) |
| **RNN** | O(T) | ❌ No | ❌ Fades | O(T) |
| **CNN** | O(T) | ✅ Yes | ❌ Limited | O(T) |
| **Sparse Attention** | O(T√T) | ✅ Yes | ✅ Yes | O(T√T) |

---

## 💻 μOmni's Attention Implementation

### Code Structure

```python
# From omni/thinker.py (simplified)
class Attention(nn.Module):
    def __init__(self, d, heads, rope_theta, use_gqa=False):
        super().__init__()
        self.heads = heads
        self.d_k = d // heads
        self.use_gqa = use_gqa
        
        if use_gqa:
            # GQA: fewer KV heads
            self.kv_groups = heads // 2
            self.q = nn.Linear(d, heads * self.d_k)
            self.k = nn.Linear(d, self.kv_groups * self.d_k)
            self.v = nn.Linear(d, self.kv_groups * self.d_k)
        else:
            # Standard MHA
            self.qkv = nn.Linear(d, 3 * d)
        
        self.o = nn.Linear(d, d)
        self.rope = RoPE(self.d_k, theta=rope_theta)
    
    def forward(self, x, mask=None, cache=None):
        B, T, D = x.shape
        
        # Project to Q, K, V
        if self.use_gqa:
            Q = self.q(x).view(B, T, self.heads, self.d_k).transpose(1, 2)
            K = self.k(x).view(B, T, self.kv_groups, self.d_k).transpose(1, 2)
            V = self.v(x).view(B, T, self.kv_groups, self.d_k).transpose(1, 2)
            # Repeat K, V to match Q heads
            K = K.repeat_interleave(self.heads // self.kv_groups, dim=1)
            V = V.repeat_interleave(self.heads // self.kv_groups, dim=1)
        else:
            qkv = self.qkv(x).chunk(3, dim=-1)
            Q, K, V = [t.view(B, T, self.heads, self.d_k).transpose(1, 2) 
                       for t in qkv]
        
        # Apply RoPE
        Q, K = self.rope(Q, K, positions)
        
        # KV caching
        if cache is not None:
            K = torch.cat([cache['K'], K], dim=2)
            V = torch.cat([cache['V'], V], dim=2)
        
        # Attention
        scores = (Q @ K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attn = torch.softmax(scores, dim=-1)
        out = attn @ V
        
        # Concatenate heads and project
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        return self.o(out), {'K': K, 'V': V}
```

---

## 💡 Key Takeaways

✅ **Attention** = Query-Key matching + Value retrieval  
✅ **Scaling** (√dₖ) prevents softmax saturation  
✅ **Multi-head** learns different relationship types  
✅ **Causal masking** enables autoregressive generation  
✅ **GQA** reduces KV parameters for efficiency  
✅ **Flash Attention** speeds up computation 2-4x  
✅ **KV caching** makes generation O(T) instead of O(T²)  
✅ **μOmni uses** 4 heads, optional GQA, Flash Attention, KV caching

---

## 🧠 Advanced: SpikingAttention (Arthemis Extension)

μOmni includes an optional neuromorphic attention mechanism inspired by spiking neural networks.

### Spiking Neural Networks (SNNs)

**Biological Inspiration:**
```
Real neurons communicate via discrete spikes:
- Neurons either fire (spike) or remain silent
- Information encoded in spike timing and frequency
- Energy-efficient (sparse) computation
```

**Mathematical Model:**
```
Leaky Integrate-and-Fire (LIF) Neuron:

τ * dV/dt = -V + I(t)           # Membrane potential dynamics
if V(t) ≥ V_threshold: spike    # Threshold crossing
V → V_reset                     # Reset after spiking
```

### SpikingAttention Implementation

```
Standard Attention:
Q, K, V = Linear_projections(input)
Attention_weights = softmax(Q @ K^T / √d)

Arthemis SpikingAttention:
Q, K, V = Linear_projections(input)
↓
Spiking_Q = LIF_Neuron(Q)    ← Convert to spike trains
Spiking_K = LIF_Neuron(K)    ← Event-driven processing
Spiking_V = LIF_Neuron(V)    ← Sparse representation
↓
RoPE_Q, RoPE_K = apply_rotary_embedding(Spiking_Q, Spiking_K)
↓
Attention_weights = softmax(RoPE_Q @ RoPE_K^T / √d)
Output = Attention_weights @ Spiking_V
```

### Benefits

- **Energy Efficiency**: Sparse spike-based computation
- **Temporal Dynamics**: Natural handling of sequential patterns
- **Neuromorphic Hardware**: Compatible with specialized chips
- **Biological Plausibility**: Closer to brain-like processing

### Configuration

Enable in config files:
```json
{
  "use_spiking": true
}
```

---

## 🎓 Self-Check Questions

1. What are Q, K, V in attention mechanism?
2. Why do we scale by √dₖ?
3. What's the complexity of attention?
4. How does GQA differ from standard MHA?
5. What is KV caching and why is it important?

<details>
<summary>📝 Click to see answers</summary>

1. Q (Query): what token is looking for, K (Key): what each token offers, V (Value): actual content to retrieve
2. To prevent dot products from getting too large (which causes vanishing gradients in softmax)
3. O(T² × D) for attention computation (quadratic in sequence length)
4. GQA uses fewer KV heads than query heads (Q heads share KV heads), reducing parameters and memory
5. KV caching stores previously computed keys/values during generation, avoiding recomputation (O(T²)→O(T))
</details>

---

[Continue to Chapter 08: Positional Encoding →](08-positional-encoding.md)

---

**Chapter Progress:** Core Concepts ●●○○○○○ (2/7 complete)

