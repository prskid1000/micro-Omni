[← Previous: 12-system-overview](12-system-overview.md) | [Index](00-INDEX.md) | [Next: 14-audio-encoder →](14-audio-encoder.md)

# Chapter 13: The Thinker -- Core Language Model

The Thinker is the brain of the system. Every modality passes through it. Every text response originates from it. If you understand the Thinker, you understand the heart of the architecture.

---

## Role

The Thinker is a **decoder-only transformer language model**. As we saw in Chapter 8, "decoder-only" means it uses causal (left-to-right) masking so each token can only attend to itself and previous tokens. This makes it autoregressive: it generates one token at a time, each conditioned on everything before it.

But unlike a text-only LLM, the Thinker accepts **two kinds of input**:

1. **Token IDs** (integers) -- for pure text, looked up in an embedding table
2. **Raw embeddings** (384-dim vectors) -- for multimodal input where image/audio encoders have already produced embeddings

This dual-input design is what makes the system multimodal. The Thinker does not know or care whether a 384-dim vector came from text, an image, or audio. It just processes sequences of vectors.

---

## Configuration (from `configs/synthetic_thinker.json`)

| Parameter | Value | Meaning |
|-----------|-------|---------|
| `vocab_size` | 256 | Character-level vocabulary (synthetic) |
| `d_model` | 128 | Hidden dimension throughout |
| `n_layers` | 4 | Number of transformer blocks |
| `n_heads` | 4 | Query attention heads |
| `d_ff` | 344 | Feedforward intermediate dimension (8/3 x d_model) |
| `ctx_len` | 64 | Maximum context length |
| `dropout` | 0.0 | Dropout rate |
| `rope_theta` | 10,000 | RoPE base frequency |
| `use_gqa` | true | Grouped Query Attention enabled |
| `kv_groups` | 2 | 2 KV groups shared across 4 query heads |
| `use_swiglu` | true | SwiGLU activation in FFN |
| `use_moe` | false | Mixture of Experts (optional, off by default) |
| `use_mtp` | false | Multi-Token Prediction (2 auxiliary heads) |
| `window_size` | 0 | Sliding window attention size (0 = disabled) |
| `scaling_factor` | 1.0 | YaRN RoPE scaling factor (1.0 = no extension) |

---

## Block Structure

Each of the 8 transformer blocks follows this pattern:

```
Input x (B, T, 384)
    |
    v
+--RMSNorm--+
|            |
|  Attention |  (RoPE, GQA with 6 query / 3 KV heads)
|            |
+-----+------+
      |
      + Residual connection (x + attn_out)
      |
      v
+--RMSNorm--+
|            |
|    FFN     |  (SwiGLU: gate_proj, up_proj, down_proj)
|            |  (or MoE: 8 experts, top-2 routing)
+-----+------+
      |
      + Residual connection
      |
      v
Output (B, T, 384)
```

ASCII diagram of the complete Thinker forward pass:

```
  idx (B,T)                    embeddings (B,T,384)
      |                              |
      v                              |
  +----------+                       |
  | tok_emb  |  (32000, 384)         |
  +----------+                       |
      |                              |
      v                              v
  x = tok_emb(idx)    OR     x = embeddings
      |
      v
  pos = [0, 1, 2, ..., T-1]
      |
      v
  mask = causal_mask[:T, :T]  (pre-allocated, sliced)
      |
      v
  +================================+
  |  Block 0                       |
  |  RMSNorm -> Attention -> +res  |
  |  RMSNorm -> SwiGLU FFN -> +res |
  +================================+
      |
      v
  +================================+
  |  Block 1 ... Block 7           |
  |  (same structure)              |
  +================================+
      |
      v
  +----------+
  | RMSNorm  |  (final normalization)
  +----------+
      |
      +---------+---------+
      |                   |
      v                   v
  +----------+     return_embeddings=True
  | lm_head  |     => return x (B,T,384)
  | (384 ->  |        (for Talker input)
  |  32000)  |
  +----------+
      |
      v
  logits (B, T, 32000)
```

---

## Input Modes in Detail

### Text-Only Mode

```python
logits = thinker(idx=token_ids)  # idx: (B, T) integers
```

The model looks up each token ID in `tok_emb` (a 32000x384 embedding table), producing a (B, T, 384) tensor. Standard language modeling.

### Multimodal Mode

```python
logits = thinker(embeddings=combined_embeddings)  # (B, T_total, 384)
```

The caller has already:
1. Run the image through ViT-Tiny and projected the CLS token to 384-dim
2. Run audio through AuT-Tiny (already outputs 384-dim)
3. Looked up text token embeddings from `tok_emb`
4. Concatenated all three along the sequence dimension

The Thinker receives this pre-assembled sequence and processes it identically to text -- it is all just 384-dim vectors.

---

## Attention: RoPE + GQA

As covered in Chapters 4 and 9, the Thinker uses:

**RoPE (Rotary Position Embeddings)**: Position information is injected by rotating Q and K vectors based on their position index. This allows the model to generalize to unseen positions and naturally encodes relative distances.

**GQA (Grouped Query Attention)**: With 6 query heads and 3 KV groups, every 2 query heads share the same key/value head. This cuts KV cache memory by 50% while maintaining nearly the same quality as full multi-head attention.

```
Query heads:    Q0  Q1  Q2  Q3  Q4  Q5     (6 heads)
                 \  /    \  /    \  /
KV groups:       KV0     KV1     KV2        (3 groups)
```

Head dimension: d_model / n_heads = 384 / 6 = **64** per head.

---

## The FFN: SwiGLU

As we saw in Chapter 7, SwiGLU replaces the standard two-layer MLP with a gated mechanism:

```
output = down_proj( swish(gate_proj(x)) * up_proj(x) )
```

Where `swish(x) = x * sigmoid(x)`.

Dimensions:
- `gate_proj`: 384 -> 1536
- `up_proj`: 384 -> 1536
- `down_proj`: 1536 -> 384

This gives the model more expressive power per parameter than a standard GELU MLP.

---

## KV Caching for Generation

During autoregressive generation, the model generates one token at a time. Without caching, generating token N requires recomputing attention for all N previous tokens -- quadratic cost.

With KV caching (as explained in Chapter 8):

1. **Prefill**: Process the full prompt, store K and V tensors for each layer
2. **Decode**: For each new token, only compute Q for the new token, reuse cached K/V

```
Step 1 (prefill):  process [tok_0, tok_1, ..., tok_N]
                   cache K,V for all N tokens in all 8 layers

Step 2 (decode):   process [tok_N+1] only
                   Q for 1 token, K/V from cache (N+1 tokens)
                   append new K,V to cache

Step 3 (decode):   process [tok_N+2] only
                   ...
```

The Thinker stores caches per layer: `self.kv_cache[layer_idx] = {'k': ..., 'v': ..., 'pos': ...}`

---

## Optional Features

### Mixture of Experts (MoE)

When `use_moe=true` (off by default), the SwiGLU FFN in each block is replaced by a MoE layer with 8 experts and top-2 routing (as described in Chapter 10). Each token is routed to its 2 best-matching experts, and their outputs are weighted-summed. This increases model capacity without increasing per-token compute.

### Arthemis Extensions

Two experimental features inspired by neuromorphic computing:

- **Spiking Attention** (`use_spiking=true`): Replaces continuous Q/K/V with binary spike signals through Leaky Integrate-and-Fire neurons. Uses a straight-through estimator for gradient flow through the discrete spike operation.

- **Liquid Time Constants** (`use_ltc=true`): Adds a dynamical system to the FFN. Each token's hidden state evolves according to a learned ODE: `new_state = state + (f(x, state) - state) / tau`. The time constant `tau` is input-dependent, allowing the network to process different inputs at different speeds -- like a biological neuron that fires faster for stronger stimuli.

Both are disabled by default (`use_spiking=false`, `use_ltc=false`).

### Multi-Token Prediction (MTP)

When `use_mtp=true`, the Thinker adds 2 auxiliary linear heads alongside the main `lm_head`. These predict tokens at positions t+2 and t+3, in addition to the standard next-token (t+1) prediction from `lm_head`.

```
Hidden state x (B, T, 384)
    |
    +---> lm_head     -> logits for t+1  (standard)
    +---> mtp_head_0  -> logits for t+2  (auxiliary)
    +---> mtp_head_1  -> logits for t+3  (auxiliary)
```

During training (`train_text.py`), the MTP loss is averaged with the main loss, providing richer gradient signal per training example. At inference, only the main `lm_head` is used -- MTP heads are training-only.

Think of it like a weather forecaster who is asked to predict not just tomorrow's weather, but also the day after and the day after that. Predicting further ahead forces the model to build deeper internal representations of the underlying patterns, even though at deployment you only need the one-day forecast.

### Sliding Window Attention (SWA)

When `window_size > 0`, even-numbered layers (0, 2, 4, 6) use sliding window attention where each token only attends to the most recent `window_size` tokens. Odd-numbered layers (1, 3, 5, 7) continue to use full causal attention.

```
Layer 0 (SWA):  token at position 100 attends to positions [100-W+1, 100]
Layer 1 (full): token at position 100 attends to positions [0, 100]
Layer 2 (SWA):  sliding window again
Layer 3 (full): full attention again
...
```

This alternating pattern gives the best of both worlds: SWA layers handle local patterns efficiently (O(T*W) instead of O(T^2)), while full attention layers maintain long-range dependencies. This is the same strategy used by Mistral and other production models.

### YaRN RoPE Extension

The RoPE module (in `omni/utils.py`) supports a `scaling_factor` parameter for extending the effective context length beyond what was used during training. When `scaling_factor > 1.0`, it uses NTK-by-parts interpolation:

- High-frequency components (local patterns) are left unmodified
- Low-frequency components (long-range dependencies) are interpolated
- An `mscale` correction factor (`0.1 * log(scaling_factor) + 1.0`) is applied to Q and K to maintain attention magnitude

For example, with `scaling_factor=4.0`, a model trained on 256-token context can generalize to ~1024 tokens at inference time without retraining.

---

## Performance Optimizations

### Pre-allocated Causal Mask

Instead of creating a new causal mask tensor every forward pass:

```python
# Created once at init (ctx x ctx upper-triangular matrix)
self.register_buffer("_causal_mask",
    torch.tril(torch.ones(ctx, ctx)).unsqueeze(0).unsqueeze(0))

# Sliced at runtime -- no allocation
mask = self._causal_mask[:, :, :T, :T]
```

This avoids GPU memory allocation during the hot path.

### Cached RoPE Tables

RoPE requires computing `cos(position * frequency)` and `sin(position * frequency)` for each position. These are computed once and cached in the RoPE module (see `omni/utils.py`), avoiding redundant trigonometric calculations.

### Flash Attention

When PyTorch 2.0+ is available, attention uses `scaled_dot_product_attention` which fuses the Q*K, softmax, and attention*V operations into a single GPU kernel. This is 2-4x faster and uses O(T) memory instead of O(T^2).

### torch.compile()

Optional compilation with `torch.compile(backend='inductor')` fuses additional operations for 10-20% speedup. Disabled by default due to compilation overhead on first run.

---

## Numerical Stability

The Thinker includes a stability check at the output:

```python
if torch.isnan(logits).any() or torch.isinf(logits).any():
    raise RuntimeError(f"Numerical instability: NaN={nan_count}, Inf={inf_count}")
```

And a diagnostic method `check_weights_stability()` that scans all parameters for NaN/Inf values -- useful when debugging training issues (more in Chapter 21).

---

## The `generate()` Method

Beyond computing logits in the forward pass, the Thinker provides a `generate()` method for autoregressive text generation with production-grade sampling controls.

### Sampling Parameters

| Parameter | Default | What It Does |
|-----------|---------|--------------|
| `temperature` | 1.0 | Scales logits before softmax |
| `top_k` | 50 | Keep only the top-k highest-probability tokens |
| `top_p` | 0.9 | Keep the smallest set of tokens whose cumulative probability exceeds p |
| `repetition_penalty` | 1.0 | Penalize tokens that already appeared in the context |

### Real-Life Analogies

**Temperature** is like the "adventurousness dial" on a restaurant recommendation app. At temperature=0.1 (low), you always get the same safe suggestion -- the most popular restaurant. At temperature=2.0 (high), the app throws in random hole-in-the-wall places. At 1.0, the original probability distribution is unchanged.

**Top-k** is like asking a friend for movie recommendations but saying "give me your top 5 picks only." Even if they know 1000 movies, you cut off the long tail of obscure choices. With top_k=50, only the 50 most likely next tokens survive; everything else gets zero probability.

**Top-p (nucleus sampling)** is like filling a shopping bag until it is 90% full by weight. You add items in order of weight (probability), and once the bag hits 90% capacity, you stop adding. This adapts to the shape of the distribution: if one token has 95% probability, only that token survives. If probability is spread out, many tokens survive.

**Repetition penalty** is like a conversation rule that says "do not repeat yourself." Every token already in the context has its logit divided by the penalty factor (e.g., 1.2), making it less likely to be chosen again. This prevents the degenerate loops where a model writes "the the the the..." endlessly.

### Generation Flow

```
prompt tokens ──→ forward pass ──→ logits (32000)
                                      |
                                      v
                              apply repetition_penalty
                                      |
                                      v
                              logits / temperature
                                      |
                                      v
                              top-k filtering
                                      |
                                      v
                              top-p filtering
                                      |
                                      v
                              softmax ──→ sample token
                                      |
                                      v
                              append to sequence, repeat
```

---

## File Reference

- **Source**: `omni/thinker.py`
- **Config**: `configs/synthetic_thinker.json`
- **Classes**: `ThinkerLM`, `Block`, `Attention`, `MLP`, `MoE`, `SwiGLU`, `SpikingNeuron`, `LiquidTimeConstant`, `RoPE` (in utils.py)

---

*Next: Chapter 14 covers the Audio Encoder -- how raw speech becomes the embeddings that the Thinker understands.*
