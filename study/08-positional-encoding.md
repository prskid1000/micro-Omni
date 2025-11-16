# Chapter 08: Positional Encodings (RoPE)

[← Previous: Attention Mechanism](07-attention-mechanism.md) | [Back to Index](00-INDEX.md) | [Next: Tokenization →](09-tokenization.md)

---

## 🎯 What You'll Learn

- Why transformers need positional information
- Different positional encoding methods
- RoPE (Rotary Position Embedding) explained
- How μOmni uses RoPE

---

## ❓ The Position Problem

Attention is **permutation invariant**:

```
"cat chased dog" → Attention → Same result as "dog chased cat"

Problem: Word order is lost!
Solution: Add position information
```

---

## 📍 Positional Encoding Methods

### 1. **Absolute Positional Encoding** (Original Transformer)

Add position-specific vectors:

```python
# Sinusoidal position encoding
PE(pos, 2i) = sin(pos / 10000^(2i/d))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d))

Position 0: [sin(0/10000^0), cos(0/10000^0), ...]
Position 1: [sin(1/10000^0), cos(1/10000^0), ...]
```

❌ Limitations:
- Fixed context length
- Doesn't generalize to longer sequences
- Relative positions not explicit

---

### 2. **Learned Positional Embeddings**

Learn position embeddings like word embeddings:

```python
pos_embedding = nn.Embedding(max_length, d_model)
x = token_emb + pos_embedding(positions)
```

❌ Limitations:
- Fixed maximum length
- No generalization beyond training length

---

### 3. **RoPE (Rotary Position Embedding)** ⭐

Encode position through rotation in complex space!

```
Instead of adding position:
  embedding + position_encoding

RoPE rotates embeddings:
  Rotate(embedding, angle=position)

Benefits:
✅ Relative position naturally encoded
✅ Generalizes to longer sequences
✅ Efficient computation
✅ Better long-range modeling
```

📌 **μOmni uses RoPE** (modern standard)

---

## 🌀 RoPE Explained

### Core Idea: Rotation

```
2D example:

Position 0:  →     (0° rotation)
Position 1:  ↗     (15° rotation)
Position 2:  ↑     (30° rotation)
Position 3:  ↖     (45° rotation)

Relative position emerges from rotation difference!
```

### Mathematical Formula

```
For each (q, k) pair at dimension (2i, 2i+1):

RoPE(q, pos) = [q_{2i}   ] [ cos(θ pos)  -sin(θ pos)] 
                [q_{2i+1}] [ sin(θ pos)   cos(θ pos)]

Where θ = 1 / (10000^(2i/d))

Result: Rotates q by angle θ×pos
```

---

## 💻 RoPE Implementation

```python
# From omni/utils.py (simplified)
class RoPE(nn.Module):
    def __init__(self, dim, theta=10000.0):
        super().__init__()
        self.dim = dim
        self.theta = theta
        
        # Precompute frequencies
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
    
    def forward(self, q, k, positions):
        # q, k: (B, H, T, D)
        # positions: (T,)
        
        # Compute angles: pos × frequency
        angles = positions[:, None] * self.inv_freq[None, :]  # (T, D/2)
        
        # Create rotation matrix
        cos = torch.cos(angles)
        sin = torch.sin(angles)
        
        # Apply rotation to q and k
        q_rot = self.rotate(q, cos, sin)
        k_rot = self.rotate(k, cos, sin)
        
        return q_rot, k_rot
    
    def rotate(self, x, cos, sin):
        # Split into even/odd dimensions
        x1 = x[..., ::2]   # Even dims
        x2 = x[..., 1::2]  # Odd dims
        
        # Rotation
        x_rot1 = x1 * cos - x2 * sin
        x_rot2 = x1 * sin + x2 * cos
        
        # Interleave back
        x_rot = torch.stack([x_rot1, x_rot2], dim=-1).flatten(-2)
        return x_rot
```

---

## 📊 RoPE Benefits

### 1. **Relative Position Encoding**

```
Query at position m, Key at position n

After RoPE:
q_m · k_n depends on (m-n), not absolute positions!

Example:
"cat" at pos 5 attending to "dog" at pos 3
Same pattern as
"cat" at pos 105 attending to "dog" at pos 103

Both have relative distance = 2
```

---

### 2. **Extrapolation to Longer Sequences**

```
Trained on sequences up to 512:
  Token positions: 0, 1, 2, ..., 511

Can handle longer at inference:
  Token positions: 0, 1, 2, ..., 2048

RoPE naturally handles this!
(Traditional absolute encodings fail)
```

---

### 3. **No Additional Parameters**

```
Learned Position Embeddings:
  max_len × d_model parameters (e.g., 2048 × 256 = 524K params)

RoPE:
  0 parameters! (computed on-the-fly)
```

---

## 🎯 μOmni's RoPE Configuration

```python
# From configs/thinker_tiny.json
{
    "rope_theta": 10000,  # Standard value
    "ctx_len": 512        # Context length
}

# RoPE applied in attention:
class Attention:
    def __init__(self, d, heads, rope_theta=10000):
        self.rope = RoPE(d // heads, theta=rope_theta)
    
    def forward(self, x, positions):
        Q, K, V = self.project_qkv(x)
        Q, K = self.rope(Q, K, positions)  # Apply RoPE
        # ... attention computation
```

---

## 📈 Visualizing RoPE

### Position Effect on Attention

```
Without RoPE:
Token "cat" at position 0 vs position 10:
  Identical attention pattern (no position info)

With RoPE:
Token "cat" at position 0:
  ████ Strong attention to nearby tokens
  ▓▓▓▓ Moderate attention to medium distance
  ░░░░ Weak attention to far tokens

Token "cat" at position 10:
  Different rotation angle → Different attention pattern
  Position information implicitly encoded!
```

---

## 🔬 Advanced: RoPE Math Deep Dive

For those interested in the mathematics:

```
Goal: Encode position in dot product

Standard dot product:
  q · k = Σ q_i × k_i

RoPE dot product:
  RoPE(q, m) · RoPE(k, n) = f(q, k, m-n)

Achieved through rotation matrices:
  R(θ, m) = [[cos(mθ), -sin(mθ)],
              [sin(mθ),  cos(mθ)]]

Properties:
  R(θ, m) · R(θ, n) = R(θ, m-n)  (rotation composition)
  
This makes relative position natural!
```

---

## 💡 Key Takeaways

✅ **Transformers need position info** (attention is permutation invariant)  
✅ **RoPE** encodes position through rotation  
✅ **Relative positions** emerge naturally from rotation difference  
✅ **No extra parameters** (computed on-the-fly)  
✅ **Extrapolates** to longer sequences than training  
✅ **μOmni uses RoPE** with theta=10000 (standard)

---

## 🎓 Self-Check Questions

1. Why do transformers need positional encoding?
2. What's the main advantage of RoPE over learned position embeddings?
3. How does RoPE encode relative positions?
4. Does RoPE add parameters to the model?
5. Can RoPE handle sequences longer than training length?

<details>
<summary>📝 Answers</summary>

1. Attention is permutation invariant - without position info, word order is lost
2. RoPE naturally encodes relative positions, extrapolates to longer sequences, and uses no extra parameters
3. Through rotation - relative position emerges from rotation angle difference
4. No, RoPE is computed on-the-fly using rotation matrices
5. Yes, RoPE can extrapolate to longer sequences (unlike learned position embeddings)
</details>

---

[Continue to Chapter 09: Tokenization →](09-tokenization.md)

**Chapter Progress:** Core Concepts ●●●○○○○ (3/7 complete)

