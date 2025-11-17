# Chapter 04: Introduction to Transformers

[← Previous: Training Basics](03-training-basics.md) | [Back to Index](00-INDEX.md) | [Next: Multimodal AI →](05-multimodal-ai.md)

---

## 🎯 Learning Objectives

By the end of this chapter, you will understand:
- What transformers are and why they're revolutionary
- The attention mechanism at the heart of transformers
- Key components: self-attention, feedforward layers, positional encoding
- Why transformers replaced RNNs for sequence processing
- The transformer architecture in μOmni

---

## 🚀 The Transformer Revolution

### Before Transformers (Pre-2017)

Imagine you're reading a book one word at a time, and you can only remember what you just read. By the time you get to the end of a long sentence, you've forgotten the beginning! This was the problem with **RNNs (Recurrent Neural Networks)**.

**Analogy: The Telephone Game**
```
Person 1 whispers to Person 2: "The cat"
Person 2 whispers to Person 3: "The cat sat"
Person 3 whispers to Person 4: "The cat sat on"
...

By the end, the message might be garbled!
This is like RNN's vanishing gradient problem.
```

**RNNs (Recurrent Neural Networks)** were the standard for sequences:

```
Processing "The cat sat on the mat"

RNN processes sequentially (like reading word by word):
Step 1: "The"       → hidden state h₁
Step 2: "The cat"   → hidden state h₂  (uses h₁)
Step 3: "The cat sat" → hidden state h₃  (uses h₂)
...

Think of it like a relay race:
Runner 1 → Runner 2 → Runner 3 → ...
Each runner can only pass info to the next one.
If Runner 1 drops the baton, info is lost!

Problems:
❌ Sequential (slow, can't parallelize) - like waiting in a single-file line
❌ Long-range dependencies fade - info from word 1 is weak by word 100
❌ Information bottleneck - everything must squeeze through hidden state
❌ Training is slow - can't process all words simultaneously
```

---

### After Transformers (2017+)

The paper **"Attention Is All You Need"** (Vaswani et al., 2017) changed everything.

```
Processing "The cat sat on the mat"

Transformer processes in parallel:
All words → Self-Attention → All words interact simultaneously!

Benefits:
✅ Fully parallelizable (fast training)
✅ Long-range dependencies captured
✅ No information bottleneck
✅ Scalable to huge models
```

---

## 👁️ The Core Idea: Attention

### What is Attention?

Think about how YOU read this sentence: "The animal didn't cross the street because **it** was too tired."

When you read "it", your brain automatically looks back to figure out what "it" means. You consider both "animal" and "street", but you quickly realize "it" = "animal" (because streets don't get tired!).

**This is exactly what attention does!**

**Attention** lets the model focus on relevant parts of the input, just like your brain does.

```
Example: Translating "The animal didn't cross the street because it was too tired"

When translating "it", what does "it" refer to?
- The animal? ✓ (makes sense: animals get tired)
- The street? ✗ (streets don't get tired)

How your brain decides (and how attention works):
1. You see "it"
2. You look back at all previous words
3. You score each word: "Could 'it' refer to this?"
   - "animal" → High score (makes sense!)
   - "street" → Low score (doesn't make sense)
4. You conclude "it" = "animal"

Attention mechanism does the SAME thing:
"it" pays attention to → "animal" (high weight)
"it" ignores → "street" (low weight)
```

**Analogy: Searching for Information**
```
You: "What's the capital of France?"

Your brain's attention:
- "capital" → Important! Focus on this.
- "France" → Important! Focus on this.
- "What's", "the", "of" → Less important

The model does the same:
- Focuses on key words ("capital", "France")
- Ignores filler words
- Retrieves answer: "Paris"
```

---

### Attention Visualization

```
Input: "The cat sat on the mat"

When processing "sat", attention weights:

Word:    The   cat   sat   on   the   mat
Weight: 0.05  0.60  1.00 0.15  0.05  0.20
        ░░░░  ████  ████ ░░░░  ░░░░  ░░░░
                ↑     ↑
        "sat" attends strongly to "cat" (subject)
```

---

## 🔍 Self-Attention Mechanism

### Understanding Query, Key, Value (Q, K, V)

Before we dive into the math, let's understand these concepts with a **real-world analogy**.

**Analogy: Searching a Library**

Imagine you're in a library looking for information:

```
YOU walk in with a QUERY:
"I need information about cats"

Each BOOK has a KEY (title/description):
Book 1: "Guide to Cats" (KEY: cats, felines, pets)
Book 2: "Car Repair Manual" (KEY: cars, vehicles)
Book 3: "Cat Breeds Encyclopedia" (KEY: cats, breeds)

You MATCH your Query with each Key:
- Your query "cats" matches Book 1's key "cats" → HIGH match!
- Your query "cats" vs Book 2's key "cars" → LOW match
- Your query "cats" matches Book 3's key "cats" → HIGH match!

You retrieve the VALUES (actual content):
- High matches: Read Books 1 & 3 (cat information)
- Low match: Skip Book 2 (car information)

Final result: Combined information from Books 1 & 3!
```

**This is EXACTLY how attention works:**

```
1. QUERY (Q): "What am I looking for?"
   → Your search question / What this word wants to know
   Example: "cat" wants to know "what did I do?"

2. KEY (K): "What information does each word offer?"
   → Book titles/descriptions / What each word can provide
   Example: "sat" offers "action/verb information"

3. VALUE (V): "The actual content"
   → Book contents / The actual meaning/embedding
   Example: "sat" → [0.3, 0.9, ...] (the actual vector)

Process (Step by Step):
Step 1: Compare Query with all Keys
        "cat" compares with "The", "cat", "sat", "on", "mat"
        → Get similarity scores

Step 2: Find relevant words (high similarity)
        "cat" is most similar to "sat" (subject-verb relation)

Step 3: Retrieve Values of relevant words
        Get the actual embeddings of "sat", "cat", etc.

Step 4: Combine them (weighted by similarity)
        Final output = mix of all relevant information!
```

---

### Mathematical Formulation

Now let's see the math. Don't worry if it looks complex - we'll break it down!

**The Formula:**
```
Attention(Q, K, V) = softmax(Q·Kᵀ / √d_k) · V
```

**Breaking it down step by step:**

```
For each word, create three vectors:
Query (Q):  What information am I looking for?
Key (K):    What information do I contain?
Value (V):  The actual information I hold

Think of it like this:
- Q = Your question
- K = What each person knows about
- V = What each person actually says

Step 1: Compute attention scores (Dot Product)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
scores = Q · Kᵀ
       (How much should I attend to each word?)

Example: Q = [0.5, 0.8]  (what "cat" is looking for)
         K_sat = [0.5, 0.8]  (what "sat" offers)
         
         score = 0.5×0.5 + 0.8×0.8 = 0.25 + 0.64 = 0.89
         (High score = similar = relevant!)

Why dot product?
- Large dot product = vectors point in same direction = similar
- Small dot product = vectors point in different directions = not similar

Step 2: Scale by √d_k (prevent large values)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
scaled_scores = scores / √d_k

Why scale? As dimension gets larger, dot products get larger.
Large values → softmax becomes too sharp → gradients vanish!

Example: d_k = 64, so √d_k = 8
         scores = [10, 5, 15] → scaled = [1.25, 0.625, 1.875]

Step 3: Softmax to get probabilities
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
attention_weights = softmax(scaled_scores)

Softmax converts scores to probabilities (sum = 1.0)

Example: scaled_scores = [1.0, 2.0, 0.5]
         softmax → [0.21, 0.58, 0.13]
         (These are now attention weights! They sum to 1.0)

Think: "How much should I pay attention to each word?"
- 21% attention to word 1
- 58% attention to word 2 (most relevant!)
- 13% attention to word 3

Step 4: Weighted sum of values
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
output = attention_weights · V
       = 0.21×V₁ + 0.58×V₂ + 0.13×V₃

This is the final output!
- Combines information from all words
- More weight on relevant words (V₂ gets 58%)
- Less weight on irrelevant words
```

**Visual Summary:**
```
Input: "The cat sat"

           Q        K        V
The    [0.1,0.2] [0.1,0.2] [actual embedding]
cat    [0.5,0.8] [0.7,0.9] [actual embedding]  ← We're processing "cat"
sat    [0.2,0.3] [0.5,0.8] [actual embedding]

Step 1: Q_cat × all Keys → scores: [0.21, 1.07, 0.89]
Step 2: Scale → [0.15, 0.76, 0.63]
Step 3: Softmax → [0.08, 0.34, 0.29] (attention weights)
Step 4: Weighted sum → 0.08×V_The + 0.34×V_cat + 0.29×V_sat
                     → Final embedding for "cat" (context-aware!)
```

---

### Concrete Example

```
Input: "cat sat"

For word "sat":

1. Create Q, K, V for each word:
   cat: Q=[1,0], K=[1,0], V=[0.8, 0.2]
   sat: Q=[0,1], K=[0,1], V=[0.3, 0.9]

2. Compute scores (sat's Q with all Ks):
   cat score = [0,1]·[1,0] = 0
   sat score = [0,1]·[0,1] = 1

3. Softmax:
   cat weight = e⁰/(e⁰+e¹) = 0.27
   sat weight = e¹/(e⁰+e¹) = 0.73

4. Weighted sum:
   output = 0.27·[0.8,0.2] + 0.73·[0.3,0.9]
          = [0.216, 0.054] + [0.219, 0.657]
          = [0.435, 0.711]

Result: "sat" attends mostly to itself (73%) and a bit to "cat" (27%)
```

---

## 👥 Multi-Head Attention

Instead of one attention mechanism, use multiple in parallel!

```
Single-Head Attention:
Input → Attention → Output
        (one perspective)

Multi-Head Attention (8 heads):
Input → ┌─ Head 1 (syntax patterns)
        ├─ Head 2 (semantic relations)
        ├─ Head 3 (long-range deps)
        ├─ Head 4 (local context)
        ├─ Head 5 (entity references)
        ├─ Head 6 (verb-subject)
        ├─ Head 7 (adjective-noun)
        └─ Head 8 (positional info)
             ↓
        Concatenate → Linear → Output

Each head learns different relationships!
```

### Why Multiple Heads?

```
Example: "The big red ball rolled down the hill"

Head 1 might focus on: Grammar structure
  "The" → "ball" (determiner-noun)
  "rolled" → "ball" (verb-subject)

Head 2 might focus on: Attributes
  "big" → "ball" (adjective-noun)
  "red" → "ball" (adjective-noun)

Head 3 might focus on: Actions
  "rolled" → "down" (verb-preposition)
  "down" → "hill" (preposition-noun)

Combined = Rich, multi-faceted understanding!
```

📌 **μOmni's Thinker uses** 4 attention heads (multi-head attention)

---

## 🏗️ Complete Transformer Architecture

### Transformer Block Structure

```
┌────────────────────────────────────────┐
│              INPUT                     │
│         [word embeddings]              │
└──────────────┬─────────────────────────┘
               │
┌──────────────▼─────────────────────────┐
│     Multi-Head Self-Attention          │
│  (words attend to each other)          │
└──────────────┬─────────────────────────┘
               │
        Add & Normalize (RMSNorm)
               │
┌──────────────▼─────────────────────────┐
│     Feedforward Network (MLP)          │
│  (process each position independently) │
└──────────────┬─────────────────────────┘
               │
        Add & Normalize (RMSNorm)
               │
┌──────────────▼─────────────────────────┐
│              OUTPUT                    │
└────────────────────────────────────────┘

This is ONE transformer block.
Stack many blocks (e.g., 4-96) for deep learning!
```

---

### Key Components

#### 1. **Multi-Head Self-Attention**
- Words/tokens attend to each other
- Captures relationships and dependencies
- Parallel processing

#### 2. **Feedforward Network (MLP)**
```python
# Simple 2-layer MLP
FFN(x) = Linear2(GELU(Linear1(x)))

# In μOmni (with SwiGLU):
FFN(x) = down_proj(swish(gate_proj(x)) * up_proj(x))
```

#### 3. **Residual Connections**
```
        Input (x)
          │
          ├────────────┐ (skip connection)
          │            │
     Attention         │
          │            │
          └────→ + ←───┘ (add residual)
               │
             Output

Benefit: Helps gradients flow, easier to train deep networks
```

#### 4. **Layer Normalization**
```
Before each sub-layer:
x = Normalize(x)  # Stabilize training

μOmni uses RMSNorm (efficient variant)
```

---

## 📍 Positional Encoding

### The Problem

Attention has no built-in sense of order!

```
"cat chased dog" = same as "dog chased cat"
(if we only look at attention without position info)

But meaning is different!
```

### The Solution: Positional Encodings

Add position information to embeddings:

```
Word embedding:      [0.2, 0.5, 0.1, 0.8, ...]  (semantic meaning)
Positional encoding: [0.1, -0.2, 0.3, -0.1, ...] (position info)
                     +
Combined embedding:  [0.3, 0.3, 0.4, 0.7, ...]  (meaning + position)
```

---

### RoPE (Rotary Position Embedding)

μOmni uses **RoPE** - a modern approach that encodes position through rotation.

```
Traditional: Add position vector
RoPE: Rotate embedding vectors based on position

Position 1:  →  (rotate 0°)
Position 2:  ↗  (rotate 15°)
Position 3:  ↑  (rotate 30°)
Position 4:  ↖  (rotate 45°)
...

Benefit: Naturally captures relative positions
```

```python
# RoPE in μOmni (simplified)
def apply_rope(q, k, positions):
    # Rotate q and k based on positions
    q_rot = rotate(q, angle=positions)
    k_rot = rotate(k, angle=positions)
    return q_rot, k_rot
```

📌 **μOmni uses RoPE** with theta=10000 (standard value)

---

## 🎯 Decoder-Only Transformers (GPT-style)

### Two Transformer Variants

#### 1. **Encoder-Only** (BERT-style)
```
Input: Full sentence
↓
Encoder blocks (bidirectional attention)
↓
Output: Understanding of sentence

Use: Classification, understanding tasks
```

#### 2. **Decoder-Only** (GPT-style) ⭐
```
Input: Partial sentence
↓
Decoder blocks (causal/autoregressive attention)
↓
Output: Next word prediction

Use: Text generation, chat, completion
```

📌 **μOmni's Thinker is decoder-only** (like GPT)

---

### Causal Attention (Autoregressive)

```
Generate text autoregressively (one token at a time):

Step 1: Input: "The"
        Output: "cat" → "The cat"

Step 2: Input: "The cat"
        Output: "sat" → "The cat sat"

Step 3: Input: "The cat sat"
        Output: "on" → "The cat sat on"
...

Attention Mask (Causal):
         The  cat  sat  on  the  mat
The      ✓    ✗    ✗    ✗   ✗    ✗   (can only see "The")
cat      ✓    ✓    ✗    ✗   ✗    ✗   (can see "The cat")
sat      ✓    ✓    ✓    ✗   ✗    ✗   (can see up to "sat")
on       ✓    ✓    ✓    ✓   ✗    ✗
the      ✓    ✓    ✓    ✓   ✓    ✗
mat      ✓    ✓    ✓    ✓   ✓    ✓   (can see everything)

Lower triangular = causal mask (can't see future!)
```

---

## 🚄 Why Transformers Won

### Comparison with RNNs

| Feature | RNN | Transformer |
|---------|-----|-------------|
| **Parallelization** | ❌ Sequential | ✅ Fully parallel |
| **Long-range deps** | ❌ Fades over time | ✅ Direct attention |
| **Training speed** | Slow | Fast (with GPUs) |
| **Scalability** | Limited | Excellent (scales to billions of params) |
| **Memory** | O(T) | O(T²) attention |

```
Training Speed Comparison:

RNN:  ████████████ (12 hours)
      ↓ (sequential bottleneck)

Transformer: ███ (3 hours)
            ↓ (parallel processing)
```

---

## 💻 Transformer in Code (Simplified)

```python
import torch
import torch.nn as nn

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff):
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, n_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)
    
    def forward(self, x, mask=None):
        # Self-attention with residual
        attn_out, _ = self.attention(x, x, x, attn_mask=mask)
        x = self.norm1(x + attn_out)
        
        # Feedforward with residual
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        
        return x

# Stack multiple blocks
class Transformer(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, d_ff, n_layers):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff)
            for _ in range(n_layers)
        ])
        self.output = nn.Linear(d_model, vocab_size)
    
    def forward(self, x):
        x = self.embed(x)
        for block in self.blocks:
            x = block(x)
        return self.output(x)
```

---

## 🎛️ μOmni's Transformer Configuration

```
Thinker (Decoder-Only Transformer):

Vocabulary:    5000 tokens
Layers:        4 transformer blocks
d_model:       256 (embedding dimension)
Heads:         4 (multi-head attention)
d_ff:          1024 (feedforward dimension)
Context:       512-2048 tokens
RoPE theta:    10000

Total params:  ~20.32M

Components:
✅ Multi-head self-attention (with GQA option)
✅ SwiGLU feedforward (modern activation)
✅ RMSNorm (efficient normalization)
✅ RoPE (rotary position embeddings)
✅ KV caching (fast inference)
```

---

## 📊 Attention Pattern Examples

### What Attention Learns

```
Sentence: "The quick brown fox jumps over the lazy dog"

Layer 1 (shallow): Syntax patterns
  "The" → "fox" (determiner-noun)
  "quick" → "fox" (adjective-noun)
  "brown" → "fox" (adjective-noun)

Layer 2 (middle): Semantic groups
  "fox" → "jumps" (subject-verb)
  "jumps" → "over" (verb-preposition)
  "over" → "dog" (preposition-object)

Layer 3 (deep): Long-range dependencies
  "fox" → "dog" (subject-object relation across distance)
  "quick" → "lazy" (contrasting attributes)

Layer 4 (deepest): Abstract relationships
  Entire sentence structure
  Topic understanding
  Context building
```

---

## 💡 Key Takeaways

✅ **Transformers** use attention instead of recurrence  
✅ **Self-attention** lets tokens attend to each other in parallel  
✅ **Multi-head attention** learns different types of relationships  
✅ **Positional encoding** (RoPE) adds position information  
✅ **Decoder-only** transformers generate text autoregressively  
✅ **Causal masking** prevents looking at future tokens  
✅ **Transformers scale** efficiently to billions of parameters

---

## 🎓 Self-Check Questions

1. What problem do transformers solve compared to RNNs?
2. What are the three components of attention (Q, K, V)?
3. Why do we need multiple attention heads?
4. What is causal/autoregressive attention?
5. Why do we need positional encodings?

<details>
<summary>📝 Click to see answers</summary>

1. Transformers process in parallel (vs sequential), handle long-range dependencies better, and scale to larger models
2. Query (what I'm looking for), Key (what each position offers), Value (actual content)
3. Multiple heads learn different types of relationships (syntax, semantics, long-range, etc.) simultaneously
4. Causal attention prevents tokens from seeing future tokens (for autoregressive generation)
5. Attention has no built-in notion of sequence order, so we add position information
</details>

---

## ➡️ Next Steps

Now you understand transformers! Let's explore how they work with multiple modalities.

[Continue to Chapter 05: What is Multimodal AI? →](05-multimodal-ai.md)

Or return to the [Index](00-INDEX.md) to choose a different chapter.

---

**Chapter Progress:** Foundation ●●●●○ (4/5 complete)

