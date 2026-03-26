[← Previous: 08-decoder-llm-kv-cache](08-decoder-llm-kv-cache.md) | [Index](00-INDEX.md) | [Next: 10-mixture-of-experts →](10-mixture-of-experts.md)

# Chapter 09: Efficient Attention -- GQA & Flash

---

## Learning Objectives

By the end of this chapter, you will understand:
- Why standard Multi-Head Attention (MHA) is expensive on memory
- How Grouped Query Attention (GQA) compresses the KV cache without losing much quality
- How Flash Attention eliminates the quadratic memory bottleneck
- How micro-Omni uses both techniques together

---

## The Memory Problem with Standard MHA

In Chapter 03 you learned Multi-Head Attention: split Q, K, V into H heads, compute attention independently per head, concatenate results. Each head has its own Q, K, and V projections.

In Chapter 08 you learned KV caching: store K and V from all previous tokens so you do not recompute them. Here is where those two ideas collide.

### KV Cache Size for Standard MHA

Every attention head stores its own K and V cache. For a model with:
- H = 8 attention heads
- d_k = 64 (head dimension)
- L = 16 layers
- T = 1024 tokens (sequence length)
- fp16 (2 bytes per value)

```
KV cache per layer  = 2 (K,V) x H x T x d_k x 2 bytes
                    = 2 x 8 x 1024 x 64 x 2
                    = 2,097,152 bytes = 2 MB

Total across layers = 16 x 2 MB = 32 MB
```

That is fine for micro-Omni. But scale up to a production model (128 heads, 80 layers, 8192 tokens) and you get:

```
2 x 128 x 8192 x 128 x 2 = 4,294,967,296 bytes = 4 GB per sequence!
```

When you are serving thousands of users simultaneously, KV cache memory becomes the dominant cost. Every head storing its own K and V is wasteful if many heads end up learning similar patterns.

---

## Multi-Query Attention (MQA): The Extreme Approach

The simplest fix: all H query heads share a single K head and a single V head.

```
STANDARD MHA (H=6):
  Q heads: Q1  Q2  Q3  Q4  Q5  Q6     (6 separate Q projections)
  K heads: K1  K2  K3  K4  K5  K6     (6 separate K projections)
  V heads: V1  V2  V3  V4  V5  V6     (6 separate V projections)

  KV cache: 6 K heads + 6 V heads = 12 head-caches

MQA (H=6 query heads, 1 shared KV):
  Q heads: Q1  Q2  Q3  Q4  Q5  Q6     (6 separate Q projections)
  K heads: K1  K1  K1  K1  K1  K1     (1 K projection, shared by all)
  V heads: V1  V1  V1  V1  V1  V1     (1 V projection, shared by all)

  KV cache: 1 K head + 1 V head = 2 head-caches
```

Think of it like a library with 6 researchers (query heads) who each need to search the card catalog. MHA gives each researcher their own private catalog. MQA gives them one shared catalog. It is 6x cheaper to maintain, but the researchers sometimes get in each other's way -- there is some quality loss.

**KV cache savings: 6x** (or generally H times smaller)

The downside: compressing 6 different K/V patterns into 1 loses expressiveness. Quality drops are measurable, especially on complex reasoning tasks.

---

## Grouped Query Attention (GQA): The Middle Ground

GQA splits the difference: instead of H separate KV heads (MHA) or 1 shared KV head (MQA), you use G groups where G is between 1 and H.

```
GQA (H=6 query heads, G=3 KV groups):
  Q heads: Q1  Q2 | Q3  Q4 | Q5  Q6
  K heads: K1  K1 | K2  K2 | K3  K3
  V heads: V1  V1 | V2  V2 | V3  V3

  Group 1: Q1,Q2 share K1,V1
  Group 2: Q3,Q4 share K2,V2
  Group 3: Q5,Q6 share K3,V3

  KV cache: 3 K heads + 3 V heads = 6 head-caches (vs 12 for MHA)
```

Think of it like a hospital with 6 doctors (query heads) organized into 3 departments (groups). Each department shares a common patient record system (K/V). Doctors within the same department see the same records, but different departments maintain separate records. More sharing than individual offices, but more specialization than one central system.

### Visual Comparison

```
MHA (H=6, G=6)         GQA (H=6, G=3)         GQA (H=6, G=2)         MQA (H=6, G=1)
Q: |1|2|3|4|5|6|       Q: |1|2|3|4|5|6|       Q: |1|2|3|4|5|6|       Q: |1|2|3|4|5|6|
K: |1|2|3|4|5|6|       K: |1|1|2|2|3|3|       K: |1|1|1|2|2|2|       K: |1|1|1|1|1|1|
V: |1|2|3|4|5|6|       V: |1|1|2|2|3|3|       V: |1|1|1|2|2|2|       V: |1|1|1|1|1|1|

KV heads: 6             KV heads: 3             KV heads: 2             KV heads: 1
Cache: 1x               Cache: 0.5x             Cache: 0.33x            Cache: 0.17x
Quality: best            Quality: near-MHA       Quality: slight loss    Quality: noticeable loss
```

### Memory Savings Table

| Method | Query Heads | KV Heads | KV Cache Size | Quality |
|--------|------------|----------|---------------|---------|
| MHA | H | H | Baseline | Best |
| GQA (H/2 groups) | H | H/2 | 50% of MHA | Near-MHA |
| GQA (H/4 groups) | H | H/4 | 25% of MHA | Slight loss |
| MQA | H | 1 | 1/H of MHA | Noticeable loss |

### micro-Omni Configuration

micro-Omni uses GQA in both Thinker and Talker when enabled:

- **Thinker**: 8 query heads, `kv_groups` configurable (default: `heads // 2 = 4`)
- **Talker**: 6 query heads, `kv_groups` configurable (default: `heads // 2 = 3`)

The `kv_groups` parameter is set in the model config. When `use_gqa=True`, separate Q, K, V projections are created:

```
Q projection: d -> heads * d_k        (full-size, one per query head)
K projection: d -> kv_groups * d_k    (smaller, one per group)
V projection: d -> kv_groups * d_k    (smaller, one per group)
```

### The Expand Trick: Zero-Copy KV Replication

After computing K and V for each group, we need to replicate them so each query head has a K and V to attend to. The naive approach would copy the data, but micro-Omni uses a zero-copy expand:

```python
# k_combined shape: (B, G, T, d_k)    G = number of KV groups
# We need:          (B, H, T, d_k)    H = number of query heads

repeat_factor = H // G   # e.g., 8 // 4 = 2

k = k_combined.unsqueeze(2)                    # (B, G, 1, T, d_k)
k = k.expand(B, G, repeat_factor, T, d_k)     # (B, G, 2, T, d_k)  <- NO memory allocation!
k = k.reshape(B, H, T, d_k)                   # (B, 8, T, d_k)
```

The `.expand()` call uses PyTorch's stride tricks to create a view of the same memory with repeated elements. No new memory is allocated. The reshape just reinterprets the dimensions. This is why GQA has essentially zero computational overhead compared to MHA -- the only cost is smaller K/V projections, which is actually faster.

---

## Flash Attention: Taming Quadratic Memory

GQA reduces the size of each K/V head-cache. Flash Attention attacks a different bottleneck: the T x T attention matrix created during the attention computation itself.

### The Problem: Materializing the Full Attention Matrix

Standard attention computes:

```
scores = Q @ K^T     # shape: (B, H, T, T)  <- THIS is the problem
weights = softmax(scores)
output = weights @ V
```

For T = 4096 tokens with fp16:
```
Attention matrix size = B x H x T x T x 2 bytes
                      = 1 x 8 x 4096 x 4096 x 2
                      = 256 MB for ONE layer, ONE sample!
```

This is quadratic in sequence length. Double the sequence and you need 4x the memory. It is also slow because this huge matrix must be written to GPU global memory (slow), then read back (slow again).

### Flash Attention: Never Materialize the Full Matrix

Flash Attention is a fused CUDA kernel that computes the exact same mathematical result as standard attention, but never creates the T x T matrix in GPU memory.

The analogy: imagine you need to compute the average height of every student in a school of 4000 students compared to every other student. The naive approach creates a 4000 x 4000 spreadsheet (16 million cells). Flash Attention processes students in small groups of 64, computing partial results and accumulating them, never needing more than a 64 x 64 workspace.

### How It Works (Conceptually)

```
STANDARD ATTENTION:                     FLASH ATTENTION:

1. Compute full Q @ K^T                1. Split Q into tiles of ~64 tokens
   (T x T matrix in memory)            2. Split K, V into tiles of ~64 tokens
2. Apply mask                           3. For each Q-tile:
3. Softmax over full row                   For each K-tile, V-tile:
4. Multiply by V                              Compute small score block
                                              Update running softmax
Memory: O(T^2)                                Accumulate partial output
                                        4. Write final output

                                        Memory: O(T) -- only tile-sized buffers!

Standard:                               Flash:
+------+------+------+------+           Process one tile at a time:
|      |      |      |      |           +------+
|  Q1K1|  Q1K2|  Q1K3|  Q1K4|          | Q1K1 | -> update output1
|      |      |      |      |           +------+
+------+------+------+------+           +------+
|      |      |      |      |           | Q1K2 | -> update output1
|  Q2K1|  Q2K2|  Q2K3|  Q2K4|          +------+
|      |      |      |      |           +------+
+------+------+------+------+           | Q1K3 | -> update output1
|      |      |      |      |           +------+
|  Q3K1|  Q3K2|  Q3K3|  Q3K4|          ... and so on
|      |      |      |      |
+------+------+------+------+           Never stores the full grid!
| Full T x T matrix in memory |
```

### The Results

| Metric | Standard Attention | Flash Attention |
|--------|--------------------|-----------------|
| Memory | O(T^2) | O(T) |
| Speed | Baseline | 2-4x faster |
| Numerical result | Exact | Exact (same math) |
| Requires | Nothing | Fused CUDA kernel |

Flash Attention is faster AND uses less memory. There is no quality tradeoff -- it computes the identical result, just more efficiently.

### Flash Attention in micro-Omni

micro-Omni uses PyTorch's built-in `scaled_dot_product_attention` (available in PyTorch 2.0+), which automatically selects the best available kernel (Flash Attention v2, memory-efficient attention, or standard math).

```python
# At module init:
HAS_FLASH_ATTENTION = hasattr(torch.nn.functional, 'scaled_dot_product_attention')

# In forward:
if self.use_flash:
    y = scaled_dot_product_attention(q, k, v, attn_mask=mask,
                                     dropout_p=self.dropout_p if self.training else 0.0)
else:
    # Manual fallback: Q @ K^T, mask, softmax, @ V
    att = torch.einsum("bhtd,bhTd->bhtT", q, k) / math.sqrt(self.dk)
    att = att.masked_fill(mask == 0, float("-inf"))
    att = att.softmax(dim=-1)
    y = torch.einsum("bhtT,bhTd->bhtd", att, v)
```

Flash Attention is enabled by default (`use_flash=True`). If PyTorch 2.0+ is not available, the model transparently falls back to the manual implementation with a warning.

---

## GQA + Flash Attention Together

These two optimizations are complementary and stack:

```
Standard MHA + Standard Attention:
  KV cache: 100%    Attention memory: O(T^2)

GQA + Standard Attention:
  KV cache: ~50%    Attention memory: O(T^2)

Standard MHA + Flash Attention:
  KV cache: 100%    Attention memory: O(T)

GQA + Flash Attention (what micro-Omni uses):
  KV cache: ~50%    Attention memory: O(T)

  Both memory savings stack!
```

### Complete Attention Pipeline in micro-Omni

```
Input x: (B, T, D)
         |
    +----v----+
    | Q proj  |---> (B, T, H * d_k)  ---> reshape (B, H, T, d_k)
    | K proj  |---> (B, T, G * d_k)  ---> reshape (B, G, T, d_k)  [G < H for GQA]
    | V proj  |---> (B, T, G * d_k)  ---> reshape (B, G, T, d_k)
    +---------+
         |
    +----v-----------+
    | Apply RoPE     |  (position encoding to Q and K)
    +----------------+
         |
    +----v-----------+
    | KV Cache       |  concat with cached K, V from previous steps
    | append new K,V |
    +----------------+
         |
    +----v-----------+
    | Expand K,V     |  (B, G, T, d_k) -> (B, H, T, d_k)  [zero-copy]
    +----------------+
         |
    +----v-----------+
    | Flash Attention|  Q @ K^T / sqrt(d_k), mask, softmax, @ V
    | (fused kernel) |  Never materializes T x T matrix
    +----------------+
         |
    +----v-----------+
    | Output proj    |  (B, T, D)
    +----------------+
```

---

## Summary

| Technique | What It Does | Savings |
|-----------|-------------|---------|
| MHA | Each head has own K, V | Baseline |
| MQA | All heads share 1 K, 1 V | H x less KV cache, quality loss |
| GQA | Groups of heads share K, V | 2-4x less KV cache, minimal quality loss |
| Flash Attention | Tile-based fused kernel | 2-4x faster, O(T) vs O(T^2) memory |
| GQA + Flash | Both optimizations | Best of both worlds |

micro-Omni defaults: GQA available via config (`use_gqa=True`), Flash Attention enabled by default when PyTorch 2.0+ is detected.

---

[← Back to Index](00-INDEX.md) | [Previous: Decoder-Only LLMs](08-decoder-llm-kv-cache.md) | [Next: Mixture of Experts →](10-mixture-of-experts.md)
