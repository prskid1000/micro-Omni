# Chapter 43: Mathematical Foundations

[Back to Index](00-INDEX.md)

---

## 🎯 Key Mathematical Concepts

### 1. Attention Mechanism

```
Scaled Dot-Product Attention:

Attention(Q, K, V) = softmax(QK^T / √d_k) V

Where:
- Q: Query matrix (n × d_k)
- K: Key matrix (m × d_k)
- V: Value matrix (m × d_v)
- d_k: Key dimension (for scaling)
- softmax: Converts scores to probabilities
```

### 2. RoPE (Rotary Position Embedding)

```
Rotation matrix for position m:

R_m = [cos(mθ)  -sin(mθ)]
      [sin(mθ)   cos(mθ)]

Applied to query/key:
q_m = R_m @ q
k_n = R_n @ k

Dot product encodes relative position:
q_m^T k_n = q^T R_m^T R_n k = q^T R_{m-n} k

Depends only on (m-n)!
```

### 3. Softmax Temperature

```
Standard softmax:
p_i = exp(z_i) / Σ exp(z_j)

With temperature τ:
p_i = exp(z_i/τ) / Σ exp(z_j/τ)

τ > 1: More uniform (creative)
τ < 1: More peaked (conservative)
τ = 1: Standard
```

### 4. Cross-Entropy Loss

```
For classification:
L = -Σ y_i log(ŷ_i)

Where:
- y: True distribution (one-hot)
- ŷ: Predicted distribution (softmax output)

Minimizing L maximizes likelihood of correct class
```

### 5. Gradient Flow

```
Chain rule for backpropagation:

∂L/∂W₁ = ∂L/∂y × ∂y/∂h₃ × ∂h₃/∂h₂ × ∂h₂/∂h₁ × ∂h₁/∂W₁

Residual connections help:
h_{l+1} = h_l + F(h_l)

∂h_{l+1}/∂h_l = 1 + ∂F/∂h_l

The "+1" ensures gradient flow!
```

## 💡 Key Takeaways

✅ **Attention** = Weighted combination via softmax  
✅ **RoPE** encodes relative positions via rotation  
✅ **Temperature** controls sampling randomness  
✅ **Cross-entropy** measures prediction quality  
✅ **Residuals** enable training deep networks

---

[Back to Index](00-INDEX.md)

