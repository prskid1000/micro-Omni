# Chapter 43: Mathematical Foundations

[← Previous: Performance Tuning](42-performance-tuning.md) | [Back to Index](00-INDEX.md) | [Next: Research Papers →](44-research-papers.md)

---

## 📐 Core Mathematical Concepts

Mathematical foundations underlying μOmni's architecture.

---

## 🎯 Attention Mechanism

### Formula

```
Attention(Q, K, V) = softmax(QK^T / √d_k) V

Where:
- Q (queries): (seq_len, d_k)
- K (keys): (seq_len, d_k)
- V (values): (seq_len, d_v)
- d_k: key dimension (for scaling)
```

### Why It Works

**Dot product similarity:** `QK^T` measures how related each query is to each key

**Scaling:** `/ √d_k` prevents large values in softmax (gradient stability)

**Soft selection:** `softmax()` converts to probabilities (0-1, sum to 1)

**Weighted sum:** Multiply by V to get attended values

---

## 🔄 RoPE (Rotary Position Embedding)

### Formula

```
RoPE(x, m) = [
  [cos(mθ₁)  -sin(mθ₁)]   [x₁]
  [sin(mθ₁)   cos(mθ₁)] × [x₂]
]

Where:
- m: position index
- θᵢ = 10000^(-2i/d): frequency for dimension i
- Rotates embedding by position-dependent angle
```

### Properties

- **Relative positioning:** Naturally encodes relative distances
- **Extrapolation:** Works for sequences longer than training
- **Efficient:** No learned parameters

---

## 📊 Cross-Entropy Loss

### Formula

```
Loss = -Σ yᵢ log(ŷᵢ)

For classification:
Loss = -log(ŷ_true_class)

Where:
- y: true distribution (one-hot)
- ŷ: predicted probabilities (after softmax)
```

### Intuition

- Penalizes low probability on correct class
- Perfect prediction: loss = 0
- Completely wrong: loss = ∞

---

## 🎵 CTC Loss (for ASR)

### Formula

```
L_CTC = -log P(y|x)

Where P(y|x) sums over all valid alignments:
P(y|x) = Σ_{π: B(π)=y} Π_t P(πₜ|x)

B(π): CTC collapse function (removes blanks, repeated chars)
```

### Why CTC

- **Variable length:** Audio frames ≠ character count
- **No alignment needed:** Automatically finds best alignment
- **Efficient:** Dynamic programming for computation

---

## 🔢 Layer Normalization

### Formula

```
LayerNorm(x) = γ (x - μ) / √(σ² + ε) + β

Where:
- μ = mean(x): mean over features
- σ² = var(x): variance over features
- γ, β: learnable scale and shift
- ε: small constant (1e-6) for stability
```

### RMSNorm (μOmni uses this)

```
RMSNorm(x) = x / RMS(x) × γ

RMS(x) = √(mean(x²))

Simpler, faster, same effect!
```

---

## 🎲 Softmax Function

### Formula

```
softmax(xᵢ) = exp(xᵢ) / Σⱼ exp(xⱼ)

Properties:
- Output: probabilities (0-1, sum to 1)
- Differentiable: enables gradient descent
- Amplifies differences: large xᵢ → high probability
```

### Temperature Scaling

```
softmax(x/T) where T > 0

T = 1: standard
T → 0: argmax (deterministic)
T → ∞: uniform (random)
```

---

## 📈 Gradient Descent

### Formula

```
θₜ₊₁ = θₜ - η ∇L(θₜ)

Where:
- θ: model parameters
- η: learning rate
- ∇L: gradient of loss
```

### Adam Optimizer (μOmni uses)

```
m̂ₜ = β₁mₜ + (1-β₁)gₜ    // momentum
v̂ₜ = β₂vₜ + (1-β₂)gₜ²   // variance
θₜ₊₁ = θₜ - η m̂ₜ/√(v̂ₜ + ε)

Benefits: Adaptive learning rates, momentum
```

---

## 🎵 HiFi-GAN Vocoder Losses

### LSGAN (Least Squares GAN) Loss

**Discriminator Loss:**
```
L_D = ½ E[(D(x_real) - 1)²] + ½ E[D(x_fake)²]

Where:
- D(x_real): discriminator output for real audio (should be 1)
- D(x_fake): discriminator output for fake audio (should be 0)
- Uses MSE instead of cross-entropy (more stable gradients)
```

**Generator Adversarial Loss:**
```
L_G_adv = ½ E[(D(x_fake) - 1)²]

Generator wants discriminator to output 1 for fake audio
```

### Feature Matching Loss

```
L_FM = E[Σᵢ |fᵢ(x_real) - fᵢ(x_fake)|]

Where:
- fᵢ: intermediate feature maps from discriminator layers
- L1 distance between real and fake features
- Stabilizes training by matching intermediate representations
```

### Mel Spectrogram Loss

```
L_mel = E[|mel_real - mel_fake|]

Where:
- mel_real: mel spectrogram of real audio
- mel_fake: mel spectrogram of generated audio
- L1 loss ensures frequency content matches
```

### Total Generator Loss

```
L_G = λ_adv × L_G_adv + λ_fm × L_FM + λ_mel × L_mel

Default weights (μOmni):
- λ_adv = 1.0   (adversarial)
- λ_fm = 2.0    (feature matching)
- λ_mel = 45.0  (mel spectrogram - highest weight)
```

### Why These Losses?

- **LSGAN:** More stable than standard GAN (smoother gradients)
- **Feature Matching:** Prevents mode collapse, improves quality
- **Mel Loss:** Ensures frequency content matches (critical for audio)

---

## 🔊 HiFi-GAN Architecture

### Generator: Mel → Audio

**Upsampling Formula:**
```
T_audio = T_mel × hop_length

Example: 33 mel frames × 256 hop = 8448 audio samples
```

**Multi-Receptive Field (MRF) Blocks:**
```
x_out = (x₁ + x₂ + ... + xₙ) / n

Where each xᵢ is from a residual block with different:
- Kernel sizes: [3, 7, 11]
- Dilation rates: [1, 3, 5]

Captures different temporal patterns simultaneously
```

**Output Activation:**
```
audio = Tanh(conv_post(x))

Tanh ensures output in [-1, 1] range (standard audio format)
```

### Multi-Period Discriminator (MPD)

**Period Reshaping:**
```
For period p:
x_reshaped = reshape(x, [B, 1, T//p, p])

Then applies 2D convolutions to capture periodic patterns
```

**Periods Used:** [2, 3, 5, 7, 11] - captures different temporal periodicities

### Multi-Scale Discriminator (MSD)

**Scale Downsampling:**
```
Scale 1: Original audio
Scale 2: AvgPool1d(4, 2) - 2× downsampled
Scale 3: AvgPool1d(4, 2) again - 4× downsampled

Captures patterns at different time scales
```

---

## 💡 Key Insights

✅ **Attention:** Weighted averaging based on similarity  
✅ **RoPE:** Encodes position via rotation  
✅ **Cross-entropy:** Measures prediction quality  
✅ **CTC:** Handles variable-length alignment  
✅ **Normalization:** Stabilizes training  
✅ **Softmax:** Converts scores to probabilities  
✅ **LSGAN:** Stable adversarial training with MSE loss  
✅ **Feature Matching:** Prevents mode collapse in GANs  
✅ **Mel Loss:** Ensures frequency-domain accuracy  
✅ **MRF Blocks:** Captures multi-scale temporal patterns

---

[Continue to Chapter 44: Research Papers →](44-research-papers.md)

---
