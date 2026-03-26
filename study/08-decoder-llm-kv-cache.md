[← Previous: 07-normalization-activations](07-normalization-activations.md) | [Index](00-INDEX.md) | [Next: 09-efficient-attention →](09-efficient-attention.md)

# Chapter 08: Decoder-Only LLMs & KV Caching

---

## Learning Objectives

By the end of this chapter, you will understand:
- The three transformer architectures and why decoder-only won
- How causal masking enforces left-to-right generation
- The autoregressive generation loop and sampling strategies
- Why naive generation is painfully slow, and how KV caching fixes it

---

## The Three Transformer Architectures

Not all transformers are created equal. Since the original 2017 "Attention Is All You Need" paper, three distinct architectures have emerged, each optimized for different tasks.

### Encoder-Only (BERT)

An encoder-only model sees the entire input at once. Every token attends to every other token -- past, present, and future. This is called **bidirectional attention**.

Think of it like a teacher grading an essay. The teacher reads the whole essay before deciding what any particular sentence means. Context flows in both directions.

Best for: classification, named entity recognition, sentence similarity.

### Encoder-Decoder (T5)

The encoder reads the full input bidirectionally, then passes a compressed representation to the decoder. The decoder generates output one token at a time, attending to both the encoder's output and its own previously generated tokens.

Think of it like a simultaneous interpreter at the UN. First, they listen to the entire sentence in French (encoder). Then they produce the English translation word by word (decoder), referring back to the French meaning as needed.

Best for: translation, summarization, question answering with long inputs.

### Decoder-Only (used by most modern LLMs, and our own micro-Omni)

A decoder-only model generates text one token at a time, and each token can only see tokens that came before it. There is no separate encoder. The input (prompt) and output (completion) are just one continuous sequence.

Think of it like writing a story word by word, where you can only look back at what you have already written -- never peek ahead.

Best for: text generation, chat, code completion, and increasingly everything else.

```
ENCODER-ONLY (BERT)           ENCODER-DECODER (T5)         DECODER-ONLY (GPT/uOmni)
+---+---+---+---+             +---+---+---+   +---+---+    +---+---+---+---+
| A | B | C | D |  <-- input  | A | B | C |   | X | Y |   | A | B | C | ? |
+---+---+---+---+             +---+---+---+   +---+---+   +---+---+---+---+
  |   |   |   |                 |   |   |       |   |       |   |   |   |
  v   v   v   v                 v   v   v       v   v       v   v   v   v
[Full attention]              [Bidirectional] [Cross+Causal] [Causal only]
  |   |   |   |                    |               |         |   |   |   |
  v   v   v   v                    v               v         v   v   v   v
[CLS] [tok] [tok] [tok]       [encoded]       [X'] [Y']    [B'] [C'] [?'] [D']
                                                             next token: D
Each token sees ALL tokens    Encoder sees all  Decoder      Each token sees
                              Decoder sees      generates    only PAST tokens
                              encoder + past    left-to-right
```

**Why decoder-only won:** It turns out that a single architecture that treats everything as "predict the next token" scales remarkably well. You do not need a separate encoder because the prompt itself is processed by the same causal decoder. Simpler architecture, fewer design choices, easier to scale.

---

## Causal Masking: No Peeking Allowed

The key mechanism that makes decoder-only models work is the **causal mask** (also called an attention mask). It is a lower-triangular matrix that prevents each token from attending to future tokens.

### The Exam Analogy

Imagine an exam where questions are revealed one at a time. When you are working on question 3, you can see questions 1, 2, and 3 -- but question 4 is still covered. This prevents you from using future information to answer current questions.

### The Mask Matrix

For a sequence of 5 tokens, the causal mask looks like this:

```
Token:    A  B  C  D  E
       +--+--+--+--+--+
  A    | 1  0  0  0  0 |   A can only see A
  B    | 1  1  0  0  0 |   B can see A, B
  C    | 1  1  1  0  0 |   C can see A, B, C
  D    | 1  1  1  1  0 |   D can see A, B, C, D
  E    | 1  1  1  1  1 |   E can see everything
       +--+--+--+--+--+

  1 = allowed to attend
  0 = masked out (set to -inf before softmax)
```

In code, this is just `torch.tril(torch.ones(T, T))`. micro-Omni pre-allocates this mask once at initialization:

```
self.register_buffer("_causal_mask",
    torch.tril(torch.ones(ctx, ctx)).unsqueeze(0).unsqueeze(0))
```

Then at runtime, it slices `self._causal_mask[:, :, :T, :T]` -- no allocation, no garbage collection, zero overhead.

### Masking in Attention

Where the mask has a 0, the attention score is replaced with negative infinity before the softmax. This drives the softmax output to zero for those positions:

```
scores = Q @ K^T / sqrt(d_k)           # (T, T) raw scores
scores = scores.masked_fill(mask==0, -inf)  # future tokens -> -inf
weights = softmax(scores)               # -inf -> 0.0 after softmax
output = weights @ V                    # future tokens contribute nothing
```

---

## Autoregressive Generation

With causal masking in place, generation works as a loop: predict one token, append it, predict the next, repeat.

### The Generation Loop

```
Input:  "The cat sat on the"

Step 1: Model sees ["The", "cat", "sat", "on", "the"]
        Predicts next token distribution -> samples "mat"
        Sequence: ["The", "cat", "sat", "on", "the", "mat"]

Step 2: Model sees ["The", "cat", "sat", "on", "the", "mat"]
        Predicts next token distribution -> samples "."
        Sequence: ["The", "cat", "sat", "on", "the", "mat", "."]

Step 3: Model sees full sequence
        Predicts -> [EOS] (end of sequence)
        Done!
```

It is like writing a story word by word. You write one word, read everything you have so far, write the next word, and continue until you decide the story is finished.

### Sampling Strategies: Temperature and Top-k/Top-p

The model outputs a probability distribution over all possible next tokens (the full vocabulary). How you choose from that distribution matters enormously.

**Temperature** controls randomness. The logits (raw scores) are divided by the temperature before softmax:

```
probabilities = softmax(logits / temperature)

temperature = 0.1  ->  Almost deterministic (picks the highest)
temperature = 1.0  ->  Normal distribution
temperature = 2.0  ->  Very random (flattens the distribution)
```

Think of temperature like a confidence dial. Low temperature means the model is very sure of its choice. High temperature means it is willing to explore unusual options.

**Top-k sampling** restricts choices to the k most likely tokens, then renormalizes:

```
logits = [0.5, 0.3, 0.1, 0.05, 0.03, 0.02]
top-k=3: keep [0.5, 0.3, 0.1], discard rest
renormalize: [0.556, 0.333, 0.111]
```

**Top-p (nucleus) sampling** keeps the smallest set of tokens whose cumulative probability exceeds p:

```
sorted probs: [0.5, 0.3, 0.1, 0.05, 0.03, 0.02]
cumulative:   [0.5, 0.8, 0.9, 0.95, 0.98, 1.0]
top-p=0.9:    keep [0.5, 0.3, 0.1]  (cumsum reaches 0.9)
```

Top-p is adaptive -- for confident predictions it keeps fewer tokens, for uncertain predictions it keeps more.

---

## The Speed Problem

Here is the painful truth about naive autoregressive generation. At each step, the model recomputes attention over the ENTIRE sequence from scratch.

### Why This Is Wasteful

```
Step 1: Process tokens [A, B, C, D, E]
        Compute Q, K, V for ALL 5 tokens
        Compute 5x5 attention matrix

Step 2: Process tokens [A, B, C, D, E, F]
        Compute Q, K, V for ALL 6 tokens   <-- A-E recomputed!
        Compute 6x6 attention matrix

Step 3: Process tokens [A, B, C, D, E, F, G]
        Compute Q, K, V for ALL 7 tokens   <-- A-F recomputed!
        Compute 7x7 attention matrix
```

To generate T tokens, you do roughly T + (T+1) + (T+2) + ... = O(T^2) attention computations. For a 1000-token generation, that is approximately 500,000 attention operations, most of them redundant.

---

## KV Caching: The Fix

The key insight: when generating token at position t, the Q/K/V values for positions 0 through t-1 are exactly the same as in the previous step. Only the new token's Q, K, and V are novel.

KV caching stores the K and V tensors from all previous steps. At each new step, we only compute Q for the new token, then attend over the cached K and V plus the new K and V.

### Before vs After KV Cache

```
WITHOUT KV CACHE (naive):

Step 1: input=[A,B,C,D,E]  -> compute Q,K,V for 5 tokens -> 5x5 attn -> next token F
Step 2: input=[A,B,C,D,E,F] -> compute Q,K,V for 6 tokens -> 6x6 attn -> next token G
Step 3: input=[A,B,C,D,E,F,G] -> compute Q,K,V for 7 tokens -> 7x7 attn -> next token H

Total Q,K,V computations: 5+6+7 = 18 sets


WITH KV CACHE:

Prefill: input=[A,B,C,D,E]  -> compute Q,K,V for 5 tokens -> 5x5 attn
         Store K_cache=[K_A,K_B,K_C,K_D,K_E], V_cache=[V_A,...,V_E]

Step 1:  input=[F] only     -> compute Q_F, K_F, V_F (1 token!)
         K_all = concat(K_cache, K_F)  -> 1x6 attn -> next token G
         Update cache: append K_F, V_F

Step 2:  input=[G] only     -> compute Q_G, K_G, V_G (1 token!)
         K_all = concat(K_cache, K_G)  -> 1x7 attn -> next token H
         Update cache: append K_G, V_G

Total Q,K,V computations: 5+1+1 = 7 sets   (vs 18 without cache)
```

### Complexity Comparison

| Metric | Without Cache | With Cache |
|--------|--------------|------------|
| Per-token Q/K/V computation | O(T) tokens | O(1) token |
| Per-token attention | O(T) dot products | O(T) dot products |
| Total for T-token generation | O(T^2) | O(T) per step |
| Speedup | Baseline | 10-50x for long sequences |

The cache trades memory for speed. You store more data (all previous K and V tensors), but you avoid recomputing them.

### Cache Memory

Each layer stores K and V tensors. For a model with L layers, H heads, sequence length T, and head dimension d_k:

```
Cache size = 2 (K and V) x L (layers) x H (heads) x T (tokens) x d_k (head dim) x bytes_per_element

Example for micro-Omni Thinker (d=128, heads=4, 4 layers, fp16):
  d_k = 128 // 4 = 32
  2 x 4 x 4 x T x 32 x 2 bytes = 2,048 x T bytes
  At T=64: ~128 KB  (with GQA kv_groups=2, only 2 KV heads cached → ~1,024 x T → ~64 KB)
```

This grows linearly with sequence length -- manageable but worth monitoring for very long sequences.

---

## KV Cache in micro-Omni

Both the Thinker (main LLM) and the Talker (speech decoder) implement KV caching. The pattern is the same in both:

### Two-Phase Generation

**Phase 1: Prefill.** The full prompt is processed in one forward pass. K and V for every layer are computed and stored in the cache.

**Phase 2: Decode.** One token at a time. Only the new token's embedding is fed through the model. Each attention layer retrieves cached K/V, concatenates the new K/V, and computes attention.

```
PREFILL PHASE                           DECODE PHASE (repeated)
+------------------+                    +------------------+
| Full prompt      |                    | Single new token |
| [A, B, C, D, E]  |                    | [F]              |
+--------+---------+                    +--------+---------+
         |                                       |
    Embed all 5                              Embed 1
         |                                       |
    +----v----+                             +----v----+
    | Layer 1 |---> store K1,V1             | Layer 1 |---> read K1,V1
    +---------+     in cache                +---------+     append new
    | Layer 2 |---> store K2,V2             | Layer 2 |---> read K2,V2
    +---------+                             +---------+     append new
    |   ...   |                             |   ...   |
    +---------+                             +---------+
    | Layer L |---> store KL,VL             | Layer L |---> read KL,VL
    +---------+                             +---------+
         |                                       |
    Logits for                              Logits for
    position 5                              position 6
    (predict F)                             (predict G)
```

### Using the Cache in Code

The Thinker exposes a simple interface:

```python
model.enable_kv_cache(True)   # Turn on caching
model.reset_kv_cache()        # Clear cache before new sequence

# Prefill: process entire prompt
logits = model(idx=prompt_tokens)      # Stores K/V in cache

# Decode loop: one token at a time
for _ in range(max_new_tokens):
    logits = model(idx=next_token.unsqueeze(0))  # Uses cached K/V
    next_token = sample(logits[:, -1, :])         # Temperature, top-k, etc.
```

Internally, each Attention layer checks whether a cache exists. If it does, it concatenates old and new K/V before computing attention. The position encoding (RoPE) uses the stored position counter to assign the correct position to the new token.

---

## Summary

| Concept | Key Point |
|---------|-----------|
| Encoder-only | Bidirectional attention, good for understanding (BERT) |
| Encoder-decoder | Encode input, decode output, good for translation (T5) |
| Decoder-only | Causal attention, good for generation (GPT, micro-Omni) |
| Causal mask | Lower triangular matrix, prevents seeing future tokens |
| Autoregressive | Predict token, append, repeat -- one token at a time |
| Temperature | Controls randomness: low=deterministic, high=creative |
| Top-k / Top-p | Restricts sampling to likely tokens |
| Naive generation | O(T^2) total work, recomputes everything each step |
| KV cache | Store K/V from previous steps, only compute new Q |
| Cache speedup | 10-50x for long sequences, O(T) per token instead of O(T^2) total |

---

[← Back to Index](00-INDEX.md) | [Next: Efficient Attention →](09-efficient-attention.md)
