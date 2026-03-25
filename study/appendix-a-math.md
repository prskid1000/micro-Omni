# Appendix A: Mathematical Foundations

A concise reference of every key formula used throughout micro-Omni.
Explanations and intuitions are in the main chapters; this appendix
collects the math in one place.

---

## Attention

### Scaled Dot-Product Attention

```
Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V
```

Where:
- Q: queries, shape (seq_len, d_k)
- K: keys, shape (seq_len, d_k)
- V: values, shape (seq_len, d_v)
- d_k: key dimension (scaling prevents softmax saturation)

### Multi-Head Attention

```
MultiHead(Q, K, V) = Concat(head_1, ..., head_h) W_O

where head_i = Attention(Q W_Qi, K W_Ki, V W_Vi)
```

### Grouped Query Attention (GQA)

Same as multi-head, but K and V projections are shared across groups:

```
h_q heads, h_kv groups  (h_q must be divisible by h_kv)
Each group of (h_q / h_kv) query heads shares one K, V head
```

---

## Positional Encoding

### RoPE (Rotary Position Embedding)

For position m and dimension pair (2i, 2i+1):

```
theta_i = 10000^(-2i / d)

R(m, theta_i) = | cos(m * theta_i)  -sin(m * theta_i) |
                | sin(m * theta_i)   cos(m * theta_i) |
```

Applied to query and key vectors:

```
q_rot = R(m) * q
k_rot = R(n) * k

dot(q_rot, k_rot) depends only on relative position (m - n)
```

---

## Normalization

### RMSNorm

```
RMSNorm(x) = x / sqrt(mean(x^2) + epsilon) * gamma

where mean(x^2) = (1/d) * sum(x_i^2) for i = 1..d
```

- gamma: learnable scale parameter, shape (d,)
- epsilon: small constant for numerical stability (typically 1e-6)
- No mean subtraction (unlike LayerNorm), so no centering

---

## Activations

### SwiGLU

```
SwiGLU(x) = W_down( swish(W_gate(x)) * W_up(x) )

where swish(z) = z * sigmoid(z)
      sigmoid(z) = 1 / (1 + exp(-z))
```

- W_gate: d_model -> d_ff
- W_up: d_model -> d_ff
- W_down: d_ff -> d_model
- Element-wise multiply of gated and ungated branches

---

## Loss Functions

### Cross-Entropy Loss

For next-token prediction:

```
L_CE = -sum( y_i * log(p_i) )  for i = 1..V

where y_i = 1 for the correct token, 0 otherwise
      p_i = softmax(logits)_i
```

In practice with hard labels:

```
L_CE = -log(p_correct)
```

### CTC Loss (Connectionist Temporal Classification)

For sequence-to-sequence alignment without forced alignment:

```
L_CTC = -log P(y | x)

P(y | x) = sum over all valid alignments pi:
            product( p(pi_t | x) )  for t = 1..T
```

- Introduces a blank token to handle variable-length alignments
- Dynamic programming (forward-backward algorithm) computes the sum
  efficiently
- Used in audio encoder training

### InfoNCE (Contrastive Loss)

For vision-language contrastive learning:

```
L_InfoNCE = -log( exp(sim(z_i, z_i+) / tau) /
                  sum( exp(sim(z_i, z_j) / tau) ) )  for j = 1..N

where sim(a, b) = dot(a, b) / (||a|| * ||b||)   (cosine similarity)
      tau = learned temperature parameter
```

- Positive pair: matching image-text (z_i, z_i+)
- Negative pairs: all other combinations in the batch
- Temperature tau controls sharpness of the distribution

---

## HiFi-GAN Losses

### LSGAN Adversarial Loss (Least Squares)

Generator:

```
L_G_adv = sum( (D_k(G(s)) - 1)^2 )  for k = 1..K discriminators
```

Discriminator:

```
L_D = sum( (D_k(x) - 1)^2 + D_k(G(s))^2 )  for k = 1..K
```

### Feature Matching Loss

```
L_fm = sum over layers l, discriminators k:
       ||D_k^l(x) - D_k^l(G(s))||_1
```

Compares intermediate activations of the discriminator on real vs.
generated audio.

### Mel-Spectrogram Reconstruction Loss

```
L_mel = ||MelSpec(x) - MelSpec(G(s))||_1
```

L1 distance between mel spectrograms of real and generated audio.

### Total Vocoder Loss

```
L_total = lambda_adv * L_G_adv + lambda_fm * L_fm + lambda_mel * L_mel
```

Default weights: lambda_adv = 1.0, lambda_fm = 2.0, lambda_mel = 45.0

---

## Optimizer

### AdamW Update Rule

```
m_t = beta_1 * m_{t-1} + (1 - beta_1) * g_t          (first moment)
v_t = beta_2 * v_{t-1} + (1 - beta_2) * g_t^2        (second moment)

m_hat = m_t / (1 - beta_1^t)                          (bias correction)
v_hat = v_t / (1 - beta_2^t)

theta_t = theta_{t-1} - lr * (m_hat / (sqrt(v_hat) + eps) + wd * theta_{t-1})
```

Key difference from Adam: weight decay (`wd * theta`) is applied directly
to parameters, not through the gradient. This decouples regularization
from the adaptive learning rate.

Typical values: beta_1 = 0.9, beta_2 = 0.999, eps = 1e-8

---

## Learning Rate Schedule

### Cosine Decay with Warmup

```
if step < warmup_steps:
    lr = lr_max * step / warmup_steps           (linear warmup)
else:
    progress = (step - warmup_steps) / (max_steps - warmup_steps)
    lr = lr_min + 0.5 * (lr_max - lr_min) * (1 + cos(pi * progress))
```

---

## Quick Reference Table

| Formula           | Used In              | Chapter |
|--------------------|---------------------|---------|
| Scaled dot-product | All attention layers | 3, 5    |
| RoPE               | Thinker, Talker      | 4       |
| RMSNorm            | All transformer blocks| 3      |
| SwiGLU             | FFN layers           | 3       |
| Cross-entropy      | Text training        | 7       |
| CTC                | Audio encoder        | 9       |
| InfoNCE            | Vision training      | 11      |
| HiFi-GAN losses   | Vocoder training     | 13      |
| AdamW              | All training         | 7       |
| Cosine schedule    | All training         | 7       |
