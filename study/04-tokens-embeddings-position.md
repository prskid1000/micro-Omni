[← Previous: 03-transformers](03-transformers.md) | [Index](00-INDEX.md) | [Next: 05-audio-processing →](05-audio-processing.md)

# Chapter 04: Tokens, Embeddings & Position

## Why Computers Need Numbers, Not Words

Computers are calculators. They multiply matrices, add vectors, and compute gradients — all operations on numbers. The word "hello" means nothing to a GPU. We need a pipeline that converts raw text into dense numerical vectors that capture meaning.

```
"The cat sat" ──► [tokenize] ──► [embed] ──► [add position] ──► vectors ready for transformer
```

---

## Tokenization: Breaking Text into Pieces

Tokenization splits text into discrete units the model can look up. There are three levels:

### Character-level

Split every character: `"cat"` → `["c", "a", "t"]`

- **Pro:** Tiny vocabulary (~256 for ASCII).
- **Con:** Sequences become extremely long. The model must learn that `c-a-t` means a furry animal — a huge burden.

**Analogy:** Reading a book one letter at a time. You can read anything, but it's painfully slow and you lose the big picture.

### Word-level

Split on spaces: `"the cat sat"` → `["the", "cat", "sat"]`

- **Pro:** Each token carries clear meaning.
- **Con:** Vocabulary explodes (English has 170,000+ words). Rare words like "defenestration" get an UNK token and lose all information.

**Analogy:** A dictionary that only has common words. If you encounter a rare word, you shrug and say "unknown."

### Subword-level (the winner)

Split into meaningful pieces: `"unhappiness"` → `["un", "happi", "ness"]`

- **Pro:** Small vocabulary, handles rare words, captures morphology.
- **Con:** Slightly more complex to build.

**Analogy:** LEGO bricks. You have a fixed set of bricks (subwords) that snap together to build any word — even ones you've never seen before.

---

## BPE: Byte Pair Encoding (Step by Step)

BPE is the most common subword algorithm. It starts with characters and repeatedly merges the most frequent pair.

### Worked Example: `"low lower lowest"`

**Step 0 — Start with characters (plus end-of-word marker `_`):**

```
Vocabulary: { l, o, w, e, r, s, t, _ }

Corpus (with frequencies):
  l o w _       ×1  (from "low")
  l o w e r _   ×1  (from "lower")
  l o w e s t _ ×1  (from "lowest")
```

**Step 1 — Count all adjacent pairs:**

```
(l, o) = 3    ← most frequent
(o, w) = 3
(w, _) = 1
(w, e) = 2
(e, r) = 1
(e, s) = 1
(r, _) = 1
(s, t) = 1
(t, _) = 1
```

Tie between `(l,o)` and `(o,w)` — pick `(l,o)`. Merge into `lo`.

**Step 2 — Updated corpus:**

```
lo w _         ×1
lo w e r _     ×1
lo w e s t _   ×1

Top pair: (lo, w) = 3 → merge into "low"
```

**Step 3 — Updated corpus:**

```
low _          ×1
low e r _      ×1
low e s t _    ×1

Top pair: (low, e) = 2 → merge into "lowe"
```

**Step 4 — Updated corpus:**

```
low _          ×1
lowe r _       ×1
lowe s t _     ×1

Top pair: (lowe, r) = 1, (lowe, s) = 1, (low, _) = 1, ...
```

We stop when the vocabulary reaches our target size. The merge rules are saved and applied at inference time.

### Key Insight

Common words like "low" stay as one token. Rare words get split into known pieces. The model never needs UNK for normally-spelled words.

---

## SentencePiece and the 32K Vocabulary

SentencePiece is a library that treats the input as a raw byte stream (no pre-tokenization by spaces). It applies BPE directly, making it language-agnostic — equally good for English, Chinese, or code.

**micro-Omni (Thinker) uses:**
- Algorithm: BPE via SentencePiece
- Vocabulary size: **32,000 tokens**
- Treats whitespace as a special character (`▁` = beginning of word)

```
Input:  "Hello world"
Tokens: ["▁Hello", "▁world"]
IDs:    [12045, 3186]
```

---

## Special Tokens

Every vocabulary reserves slots for housekeeping:

| Token | ID | Purpose |
|-------|-----|---------|
| `PAD` | 0 | Padding — fills short sequences to a uniform length |
| `BOS` | 1 | Beginning of sequence — signals "start here" |
| `EOS` | 2 | End of sequence — signals "stop generating" |
| `UNK` | 3 | Unknown — fallback for truly unrecognizable input |

**Analogy:** PAD is silence on a music track. BOS is the conductor raising the baton. EOS is the final note. UNK is a smudged word on a page.

---

## One-Hot Encoding (The Bad Way)

Represent each token as a vector with a single `1` and all other entries `0`:

```
Vocabulary: [cat, dog, fish]   (size = 3)

cat  = [1, 0, 0]
dog  = [0, 1, 0]
fish = [0, 0, 1]
```

**Problems:**
1. **Sparse:** With 32,000 tokens, each vector has 31,999 zeros. Wasteful.
2. **No meaning:** The distance between "cat" and "dog" equals the distance between "cat" and "refrigerator." Every pair is equally unrelated.

**Analogy:** Giving every student a unique locker number. Locker 5 and locker 6 aren't "more similar" than locker 5 and locker 9000. The numbers carry no meaning.

---

## Dense Embeddings (The Good Way)

Instead of sparse one-hot vectors, map each token to a **short, dense vector** (e.g., 384 dimensions) where similar words end up close together in space.

```
cat  = [0.21, -0.45, 0.73, ..., 0.11]   (384 numbers)
dog  = [0.19, -0.42, 0.71, ..., 0.13]   (close to cat!)
fish = [0.55,  0.12, 0.33, ..., -0.28]  (farther away)
```

**Analogy:** Instead of locker numbers, give each student GPS coordinates of their home. Now students in the same neighborhood (= similar meaning) are physically close.

### The Famous Example: king - man + woman ≈ queen

Trained embeddings capture relationships as directions in space:

```
         man ─────────────────► woman
          │                       │
          │   same direction!     │
          ▼                       ▼
        king ─────────────────► queen

vector("king") - vector("man") + vector("woman") ≈ vector("queen")
```

The direction from "man" to "woman" encodes the concept of gender. Adding that direction to "king" lands near "queen." This means the embedding space has learned real semantic structure — not from a dictionary, but purely from seeing words in context.

---

## nn.Embedding: A Learnable Lookup Table

In PyTorch, `nn.Embedding(num_tokens, d_model)` creates a matrix of shape `(num_tokens, d_model)`. Looking up a token ID simply indexes a row.

```python
embed = nn.Embedding(32000, 384)   # 32,000 tokens × 384 dimensions

# Input: token IDs [1, 547, 23]
# Output: 3 vectors of size 384
#   embed.weight[1]    → vector for BOS
#   embed.weight[547]  → vector for some word
#   embed.weight[23]   → vector for another word
```

For **micro-Omni Thinker**: the table is `32000 x 384` = **12.3 million learnable parameters** just for the embedding layer.

**Analogy:** A filing cabinet with 32,000 folders. Each folder contains a card with 384 numbers. To embed a token, you open the right folder and pull the card. During training, the numbers on every card get updated to better capture meaning.

---

## The Position Problem

Recall from Chapter 03 that self-attention computes a weighted sum over all tokens. But attention treats its inputs as a **set** — it has no notion of order.

```
"dog bites man"  ──attention──►  same attention scores
"man bites dog"  ──attention──►  as above (just permuted)
```

Without position information, the model thinks these two sentences are equivalent. But they have very different meanings! We need to inject position into each token's representation.

---

## RoPE: Rotary Position Embeddings

RoPE (Rotary Position Embedding) encodes position by **rotating** each token's query and key vectors by an angle proportional to the token's position.

### The Clock Hand Analogy

Imagine each embedding dimension pair as a clock hand:

```
Position 0:  clock hand at 12 o'clock  (0°)
Position 1:  clock hand at 1 o'clock   (30°)
Position 2:  clock hand at 2 o'clock   (60°)
...

Different dimension pairs rotate at different speeds:
  - Pair 1 (fast):   ●─── rotates 30° per position
  - Pair 2 (medium): ●─── rotates 15° per position
  - Pair 3 (slow):   ●─── rotates 5° per position
```

When computing attention between token at position `m` and token at position `n`, the dot product depends on `m - n` (the relative distance), not on `m` or `n` individually.

### How It Works (Simplified)

Take a vector's dimensions in pairs `(x1, x2)` and rotate by angle `theta * position`:

```
[ x1' ]   [ cos(m*theta)  -sin(m*theta) ] [ x1 ]
[ x2' ] = [ sin(m*theta)   cos(m*theta) ] [ x2 ]
```

Each pair of dimensions uses a different base frequency `theta`, creating a spectrum from fast-rotating (captures nearby positions) to slow-rotating (captures distant positions).

### RoPE Benefits

1. **Relative position:** Attention naturally depends on distance between tokens, not absolute position. "The cat" means the same whether it starts at position 0 or position 500.

2. **Extrapolation:** Because it uses continuous rotation, RoPE can generalize to sequence lengths longer than those seen during training (with some techniques like NTK-aware scaling).

3. **Zero extra parameters:** RoPE is a fixed mathematical function — no learnable position embeddings needed. This saves memory and avoids overfitting.

---

## Full Pipeline: Text to Embedding Vectors

```
                        TOKENIZATION PIPELINE
 ═══════════════════════════════════════════════════════════════

 Input text:  "The cat sat on the mat"

       │
       ▼
 ┌─────────────┐
 │  Tokenizer  │   SentencePiece BPE (32K vocab)
 │  (BPE)      │
 └──────┬──────┘
        │
        ▼
 Token IDs:  [1, 450, 6234, 1772, 289, 450, 8451, 2]
              BOS  The   cat   sat   on   the   mat  EOS

        │
        ▼
 ┌──────────────────┐
 │  nn.Embedding    │   Lookup table: 32000 × 384
 │  (32000 × 384)  │
 └───────┬──────────┘
         │
         ▼
 Raw embeddings:  shape (8, 384)
   [ [0.12, -0.34, ...],    ← BOS
     [0.45,  0.21, ...],    ← The
     [0.78, -0.11, ...],    ← cat
     ...                     ...
     [0.33,  0.56, ...] ]   ← EOS

         │
         ▼
 ┌──────────────────┐
 │  RoPE            │   Rotate Q,K vectors by position angle
 │  (applied in     │   (inside each attention layer)
 │   attention)     │
 └───────┬──────────┘
         │
         ▼
 Position-aware vectors ready for transformer blocks!
```

---

## Summary

| Concept | What It Does | micro-Omni Setting |
|---------|-------------|-------------------|
| BPE Tokenizer | Splits text into subword tokens | SentencePiece, 32K vocab |
| Special tokens | PAD, BOS, EOS, UNK | IDs 0, 1, 2, 3 |
| nn.Embedding | Maps token ID to dense vector | 32000 x 384 |
| RoPE | Encodes position via rotation | Applied in attention layers |

**Key takeaway:** The tokenization pipeline transforms arbitrary text into a sequence of dense, position-aware vectors — the universal language that transformers understand.
