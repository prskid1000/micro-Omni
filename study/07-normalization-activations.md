[← Previous: 06-image-processing](06-image-processing.md) | [Index](00-INDEX.md) | [Next: 08-decoder-llm-kv-cache →](08-decoder-llm-kv-cache.md)

# Chapter 07: Normalization & Activations

## Why Normalization? Taming Exploding Values

Without normalization, values passing through a neural network can grow or shrink uncontrollably with each layer.

```
Layer 1 output:  [2.1, 3.4, 1.8]
Layer 2 output:  [15.7, 28.3, 12.1]
Layer 3 output:  [245.0, 512.8, 198.4]
Layer 10 output: [∞, ∞, ∞]           ← values explode!
```

Or the opposite — values shrink toward zero and the model stops learning (vanishing signals).

**The microphone analogy:** Imagine a chain of microphones, each feeding into the next. Without volume control, a whisper vanishes by the third mic and a shout causes deafening feedback. Normalization is the automatic volume knob at each stage — it keeps the signal in a healthy range regardless of what came before.

---

## LayerNorm: Center and Scale

**Layer Normalization** normalizes across all features (dimensions) within a single sample.

### Formula

For a vector `x` of dimension `d`:

```
          x_i - mean(x)
y_i = ─────────────────── × gamma_i + beta_i
           std(x) + eps
```

Where:
- `mean(x)` = average of all d values
- `std(x)` = standard deviation of all d values
- `gamma` (scale) and `beta` (shift) = learnable parameters
- `eps` = tiny constant (e.g., 1e-6) to prevent division by zero

### Worked Example

```
Input x = [4.0, 2.0, 0.0, 6.0]

mean = (4 + 2 + 0 + 6) / 4 = 3.0
std  = sqrt(((4-3)^2 + (2-3)^2 + (0-3)^2 + (6-3)^2) / 4)
     = sqrt((1 + 1 + 9 + 9) / 4)
     = sqrt(5)
     ≈ 2.236

Normalized (before gamma/beta):
  (4-3)/2.236 =  0.447
  (2-3)/2.236 = -0.447
  (0-3)/2.236 = -1.342
  (6-3)/2.236 =  1.342

Result: [0.447, -0.447, -1.342, 1.342]   ← mean≈0, std≈1
```

The output is centered around zero with unit variance. The learnable `gamma` and `beta` let the model undo the normalization if needed — but in practice, they keep values well-behaved.

---

## RMSNorm: Simpler and Faster

**RMS Normalization** (Root Mean Square Norm) skips the mean-subtraction step and just divides by the root-mean-square.

### Formula

```
              x_i
y_i = ──────────────────── × gamma_i
       RMS(x) + eps

where RMS(x) = sqrt( (x_1^2 + x_2^2 + ... + x_d^2) / d )
```

### Comparison with LayerNorm

```
LayerNorm:  1) compute mean    2) compute std    3) subtract mean & divide by std
RMSNorm:    1) compute RMS     2) divide by RMS

RMSNorm skips the mean computation entirely.
```

### Why It's Faster

- No mean subtraction = fewer operations
- **15-20% faster** than LayerNorm in practice
- Empirically works just as well for transformers
- Used by modern LLMs and **micro-Omni**

**Analogy:** LayerNorm is like centering a seesaw *and* adjusting its scale. RMSNorm just adjusts the scale — and it turns out the seesaw is usually close enough to centered already.

### Worked Example

```
Input x = [4.0, 2.0, 0.0, 6.0]

RMS = sqrt((16 + 4 + 0 + 36) / 4)
    = sqrt(56 / 4)
    = sqrt(14)
    ≈ 3.742

Normalized (assuming gamma = [1,1,1,1]):
  4.0 / 3.742 = 1.069
  2.0 / 3.742 = 0.534
  0.0 / 3.742 = 0.000
  6.0 / 3.742 = 1.603

Result: [1.069, 0.534, 0.000, 1.603]
```

Notice: the result is NOT zero-centered (unlike LayerNorm). RMSNorm only controls the *scale*, not the *center*.

---

## Pre-Norm vs Post-Norm

Where you place normalization relative to the sublayer (attention or FFN) matters for training stability.

### Post-Norm (Original Transformer, 2017)

```
x ──► [Attention] ──► [Add x] ──► [Norm] ──► output
```

The residual is added first, then normalized. Can be unstable early in training because gradients flow through unnormalized attention outputs.

### Pre-Norm (Modern Standard, used by micro-Omni)

```
x ──► [Norm] ──► [Attention] ──► [Add x] ──► output
```

Normalize *before* the sublayer. The residual skip connection carries the clean, unnormalized signal directly.

**micro-Omni's pattern:**
```python
# Pre-norm residual connection
x = x + attention(norm(x))
x = x + ffn(norm(x))
```

### Why Pre-Norm Wins

```
Pre-norm gradient flow:

x ─────────────────────────────► + ──► output
     │                           ▲
     ▼                           │
   [Norm] ──► [Sublayer] ────────┘

The gradient has a "highway" (the skip connection) that
bypasses the sublayer entirely. This prevents vanishing
gradients even in very deep networks.
```

**Analogy:** Post-norm is like proofreading *after* combining two drafts — errors from either draft can compound. Pre-norm is like proofreading each draft *before* combining — the merge is always clean.

---

## Why Activations? Breaking Linearity

Without activation functions, stacking layers is pointless:

```
Layer 1:  y = W1 × x + b1
Layer 2:  z = W2 × y + b2
         = W2 × (W1 × x + b1) + b2
         = (W2 × W1) × x + (W2 × b1 + b2)
         = W_combined × x + b_combined

Two layers collapsed into one!
```

No matter how many linear layers you stack, the result is still a single linear transformation. **Activation functions** introduce non-linearity, giving the network the ability to learn curves, boundaries, and complex patterns.

**Analogy:** Linear layers are like rulers — they can only draw straight lines. Activation functions bend the ruler, letting the network draw any shape.

---

## ReLU: The Simple Gate

**Rectified Linear Unit** — the activation that launched deep learning:

```
ReLU(x) = max(0, x)

If x > 0:  pass it through unchanged
If x ≤ 0:  output 0
```

```
Output
  │        ╱
  │       ╱
  │      ╱
  │     ╱
  │    ╱
  │   ╱
──┼──╱──────── Input
  │ 0
  │
```

**Pros:** Dead simple. Fast to compute. Gradient is either 0 or 1 (no vanishing gradient for positive values).

**Cons:** "Dead neurons" — if a neuron's input is always negative, it always outputs 0 and never learns again. Also, the sharp corner at 0 can cause optimization issues.

**Analogy:** A one-way valve. Positive signals flow freely; negative signals are blocked completely.

---

## GELU: The Smooth Gate

**Gaussian Error Linear Unit** — used in BERT, GPT-2, and many transformers:

```
GELU(x) = x × Phi(x)

where Phi(x) is the cumulative distribution function of
the standard normal distribution (probability that a
random normal value is ≤ x).
```

In practice:
```
GELU(x) ≈ 0.5 × x × (1 + tanh(sqrt(2/pi) × (x + 0.044715 × x^3)))
```

```
Output
  │         ╱
  │        ╱
  │      ╱╱
  │    ╱╱
  │  ╱╱
──┼─╱────────── Input
  │╱
  ├─
  │
```

**Key difference from ReLU:** Instead of a hard cutoff at 0, GELU smoothly transitions. Small negative values get a small (not zero!) output. This keeps gradients flowing and avoids dead neurons.

**Analogy:** Instead of a strict bouncer who blocks everyone under a threshold, GELU is a bouncer who lets almost everyone positive in, blocks strongly negative values, but gives borderline cases a small chance.

---

## SwiGLU: The Gated Powerhouse

**SwiGLU** (Swish-Gated Linear Unit) is the activation used in modern LLMs and **micro-Omni**.

### The Idea: Two Parallel Pathways Merged

Instead of one linear projection through an activation, SwiGLU uses two projections and combines them via gating:

```
           ┌──► W_gate(x) ──► Swish ──┐
           │                           ├──► element-wise multiply ──► W_down ──► output
input x ───┤                           │
           └──► W_up(x) ──────────────┘
```

### Formula

```
SwiGLU(x) = Swish(x × W_gate) ⊙ (x × W_up)

where:
  Swish(z) = z × sigmoid(z)
  ⊙ = element-wise multiplication
```

Then the result is projected back down:

```
FFN(x) = W_down × SwiGLU(x)
```

### Step by Step

```
Input: x (dimension d)

1. Gate path:   g = x × W_gate     (d → d_ff)
2. Swish:       g = g × sigmoid(g)  (smooth gating)
3. Up path:     u = x × W_up       (d → d_ff)
4. Merge:       h = g ⊙ u          (element-wise multiply)
5. Down:        out = h × W_down   (d_ff → d)
```

### Why SwiGLU Works

The gate `g` controls how much of the up-projection `u` passes through. This is like having a smart filter:
- The **up path** proposes features
- The **gate path** decides which features matter
- Multiplying them keeps only the useful ones

```
       Proposed features (up path):  [3.2, -1.1, 0.8, 2.5]
       Gate values (after Swish):    [0.9,  0.1, 0.7, 0.0]
                                      ×     ×     ×     ×
       Output:                       [2.88,-0.11,0.56, 0.0]

  Feature 1: strong gate → passes through
  Feature 4: gate ≈ 0   → blocked
```

### The Swish Function

```
Swish(x) = x × sigmoid(x)

           sigmoid(x) = 1 / (1 + exp(-x))

For large positive x:  sigmoid ≈ 1, so Swish ≈ x  (like ReLU)
For large negative x:  sigmoid ≈ 0, so Swish ≈ 0  (like ReLU)
For x near 0:          smooth curve (unlike ReLU's sharp corner)
```

---

## Activation Function Comparison (ASCII Shapes)

```
 ReLU                    GELU                    Swish
 output                  output                  output
   │        ╱              │         ╱              │         ╱
   │       ╱               │        ╱               │        ╱
   │      ╱                │      ╱╱                │      ╱╱
   │     ╱                 │    ╱╱                  │    ╱╱
   │    ╱                  │  ╱╱                    │  ╱╱
   │   ╱                   │╱╱                      │╱╱
 ──┼──╱──── input        ──┼╱───── input          ──┼╱───── input
   │ 0                     │╲                       │╲╱
   │                       │ (small dip)            │ (small dip)
   │                       │                        │

 Hard cutoff at 0.       Smooth transition.       Smooth, with slight
 Dead neurons possible.  No dead neurons.         negative region.
                         Most transformers.       Used inside SwiGLU.
```

### Comparison Table

| Activation | Formula | Pros | Cons | Used in |
|-----------|---------|------|------|---------|
| ReLU | max(0, x) | Simple, fast | Dead neurons, sharp corner | Early CNNs |
| GELU | x * Phi(x) | Smooth, no dead neurons | Slightly slower than ReLU | BERT, GPT-2 |
| Swish | x * sigmoid(x) | Smooth, slight negative values | Slightly slower than ReLU | Inside SwiGLU |
| SwiGLU | Swish(xW_g) * (xW_u) | Best empirical performance | 3 weight matrices instead of 2 | Modern LLMs, micro-Omni |

---

## SwiGLU in micro-Omni: The Full FFN

```python
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        self.w_gate = nn.Linear(d_model, d_ff, bias=False)
        self.w_up   = nn.Linear(d_model, d_ff, bias=False)
        self.w_down = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x):
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))
        #                  ^^^^ silu = swish = x*sigmoid(x)
```

Note: `F.silu` is PyTorch's name for the Swish function (SiLU = Sigmoid Linear Unit = Swish).

### Parameter Count

SwiGLU has 3 weight matrices instead of the usual 2 in a standard FFN. To keep the total parameter count similar, the hidden dimension `d_ff` is typically set to `(8/3) × d_model` instead of the traditional `4 × d_model`:

```
Standard FFN:  2 × d × 4d    = 8d^2 parameters
SwiGLU FFN:    3 × d × (8/3)d = 8d^2 parameters  (same total!)
```

---

## How It All Fits Together in micro-Omni

Each transformer block follows this pattern:

```
 ┌──────────────────────────────────────────────┐
 │              TRANSFORMER BLOCK                │
 │                                               │
 │  input x                                      │
 │     │                                         │
 │     ├───────────────────┐                     │
 │     │                   │                     │
 │     ▼                   │                     │
 │  [RMSNorm]              │  (pre-norm)         │
 │     │                   │                     │
 │     ▼                   │                     │
 │  [Multi-Head Attention] │                     │
 │     │                   │                     │
 │     ▼                   │                     │
 │  [  +  ] ◄─────────────┘  (residual)         │
 │     │                                         │
 │     ├───────────────────┐                     │
 │     │                   │                     │
 │     ▼                   │                     │
 │  [RMSNorm]              │  (pre-norm)         │
 │     │                   │                     │
 │     ▼                   │                     │
 │  [SwiGLU FFN]           │                     │
 │     │                   │                     │
 │     ▼                   │                     │
 │  [  +  ] ◄─────────────┘  (residual)         │
 │     │                                         │
 │     ▼                                         │
 │  output                                       │
 └──────────────────────────────────────────────┘
```

```
x = x + attention(rmsnorm(x))    ← pre-norm + residual
x = x + swiglu_ffn(rmsnorm(x))   ← pre-norm + residual
```

---

## Summary

| Component | What It Does | micro-Omni Choice |
|-----------|-------------|-------------------|
| LayerNorm | Centers (mean=0) and scales (std=1) | Not used |
| RMSNorm | Scales only (by root-mean-square), 15-20% faster | Yes |
| Pre-norm | Normalize before sublayer, more stable training | Yes: `x + sublayer(norm(x))` |
| ReLU | Hard cutoff at zero | Not used |
| GELU | Smooth cutoff | Not used |
| SwiGLU | Gated activation: Swish(gate) * up, best performance | Yes, in every FFN block |

**Key takeaway:** Normalization keeps values in a healthy range so deep networks can train stably. Activations introduce non-linearity so stacking layers actually increases the network's power. micro-Omni uses the modern best practices for both: RMSNorm (fast, effective) with pre-norm placement, and SwiGLU (best empirical performance) in every feed-forward block.
