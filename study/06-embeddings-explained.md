# Chapter 06: Understanding Embeddings

[← Previous: Multimodal AI](05-multimodal-ai.md) | [Back to Index](00-INDEX.md) | [Next: Attention Mechanism →](07-attention-mechanism.md)

---

## 🎯 Learning Objectives

By the end of this chapter, you will understand:
- What embeddings are and why they're crucial
- How embeddings represent semantic meaning
- Different types of embeddings (word, token, positional)
- Embedding dimensions and their trade-offs
- How μOmni uses embeddings

---

## 🔢 The Problem: Computers Don't Understand Words

### Computers Only Understand Numbers

**Fundamental Problem:** You and I understand "cat" means a furry animal that meows. But to a computer:

```
Computer sees "cat": 
c = 99 (ASCII code)
a = 97
t = 116

Just three numbers! No meaning!

The computer doesn't know:
- "cat" is an animal
- "cat" is similar to "dog"
- "cat" is different from "car"

Need to convert: "cat" → Numbers that CAPTURE MEANING!
```

**Why This Matters:**

Think about these questions:
```
"What's similar to a cat?"
→ Human: "A dog! Both are pets."
→ Computer: "???" (cat = 'c','a','t', dog = 'd','o','g' - no connection!)

"King is to Queen as Man is to ____?"
→ Human: "Woman!"
→ Computer: "???" (How to compute this from letters?)

Need: A representation that captures SEMANTIC MEANING (meaning)
```

### Naive Approach: One-Hot Encoding

**First attempt:** Just assign each word a unique number slot.

```
Vocabulary: ["cat", "dog", "bird", "fish"]

"cat"  → [1, 0, 0, 0]
"dog"  → [0, 1, 0, 0]
"bird" → [0, 0, 1, 0]
"fish" → [0, 0, 0, 1]

Analogy: It's like hotel rooms!
Room 1 = cat
Room 2 = dog
Room 3 = bird
Room 4 = fish

Put a "1" in your room, "0" everywhere else.
```

**Testing One-Hot Encoding:**
```
Question: How similar are "cat" and "dog"?

Method: Compare vectors
cat  = [1, 0, 0, 0]
dog  = [0, 1, 0, 0]

Similarity = 1×0 + 0×1 + 0×0 + 0×0 = 0
(No similarity!)

But wait! Cat and dog ARE similar (both pets)!

Question: How similar are "cat" and "car"?
cat  = [1, 0, 0, 0]
car  = [0, 0, 0, 0, 1] (if car is in position 5)

Similarity = 0
(Same as cat-dog similarity!)

This is WRONG! Cat-dog should be MORE similar than cat-car!
```

**Problems with One-Hot Encoding:**
```
❌ High dimensionality
   10,000 words = 10,000-dimensional vectors!
   (99.99% zeros for each word)

❌ No semantic meaning
   "cat" and "dog" are as different as "cat" and "car"
   Distance("cat", "dog") = Distance("cat", "computer") = Distance("cat", "love")
   All pairs have similarity = 0!

❌ Sparse representation (mostly zeros)
   [0,0,0,0,...,1,...,0,0,0] ← One "1" in 10,000 positions
   Wastes memory and computation

❌ No relationships
   Can't do: "king - man + woman = queen"
   Can't do: "cat + orange = ??? (orange cat?)"

We need something BETTER!
```

---

## 💎 The Solution: Dense Embeddings

### What are Embeddings?

**Embeddings** are dense, low-dimensional vector representations that capture semantic meaning.

**The Big Idea:**

Instead of: "Is the word in slot #427?" (one-hot encoding)

Do this: "Plot the word in meaning-space!" (embeddings)

**Analogy: GPS Coordinates vs Street Address**

```
Old way (One-Hot): Street Address
"123 Main Street" - unique, but doesn't tell you anything about location
"125 Main Street" might be 1000 miles away for all you know!

New way (Embeddings): GPS Coordinates
"cat" = (latitude: 40.7, longitude: -74.0) [pet animals region]
"dog" = (latitude: 40.8, longitude: -73.9) [nearby in pet animals region!]
"car" = (latitude: 35.2, longitude: -80.1) [far away, in vehicles region]

Distance in coordinate space = Similarity!
```

**Technical View:**

```
Instead of: "cat" → [1, 0, 0, 0, ..., 0] (10,000 dimensions, sparse)
                     ↑
                     Only ONE meaningful value!

Use: "cat" → [0.2, -0.5, 0.3, 0.8, -0.1, 0.7, ...] (256 dimensions, dense)
              ↑    ↑     ↑    ↑    ↑     ↑
              ALL values are meaningful!

Each dimension captures some aspect of meaning:
- Dimension 1 (0.2):  Animal-ness
- Dimension 2 (-0.5): Size (negative = small)
- Dimension 3 (0.3):  Domesticated-ness
- Dimension 4 (0.8):  Cuteness
- Dimension 5 (-0.1): Aggression (negative = friendly)
... (251 more dimensions capturing other aspects)

Benefits:
✅ Low-dimensional (256-1024 dims instead of 10,000+)
   Efficient! Less memory, faster computation
   
✅ Dense (all values meaningful)
   Every number tells you something!
   
✅ Semantic meaning (similar words → similar vectors)
   Now "cat" and "dog" will have similar coordinates!
```

**Concrete Example:**

```
Let's see how embeddings capture meaning (simplified to 3 dimensions):

"cat"   = [0.8, 0.3, 0.7]   (animal: high, size: medium, domestic: high)
"dog"   = [0.9, 0.5, 0.8]   (animal: high, size: medium-large, domestic: high)
"tiger" = [0.9, 0.9, 0.1]   (animal: high, size: large, domestic: low)
"car"   = [0.1, 0.7, 0.2]   (animal: low, size: medium-large, domestic: medium)

Distance (cat, dog) = √[(0.8-0.9)² + (0.3-0.5)² + (0.7-0.8)²]
                    = √[0.01 + 0.04 + 0.01] = √0.06 = 0.24
                    (CLOSE! Similar concepts!)

Distance (cat, car) = √[(0.8-0.1)² + (0.3-0.7)² + (0.7-0.2)²]
                    = √[0.49 + 0.16 + 0.25] = √0.90 = 0.95
                    (FAR! Different concepts!)

Embeddings capture meaning through distance!
```

**Why This Is Revolutionary:**

```
Old (One-Hot):
All words equally different
→ No semantic understanding
→ Can't generalize

New (Embeddings):
Similar words close together
→ Semantic understanding
→ Can generalize!

Example generalization:
You learn: "The cat is sleeping"
Embeddings automatically know: "The dog is sleeping" makes sense too
  (because cat ≈ dog in embedding space!)
```

---

## 🎨 Semantic Similarity in Embedding Space

### Similar Concepts Are Close

```
2D visualization (actual embeddings are 256-1024 dimensional):

Animals:                    Vehicles:
  cat ●                       car ●
      ↘                           ↘
  dog ●─ kitten ●           bike ●─ truck ●
      ↘                           ↘
  puppy ●                     train ●

  
Foods:                      Colors:
  pizza ●                     red ●
      ↘                          ↘
  pasta ●─ sandwich ●      blue ●─ green ●
      ↘                          ↘
  burger ●                   yellow ●

Distance between vectors = Semantic similarity!
```

### Mathematical Distance

```python
import numpy as np

cat_emb =  [0.2, -0.5, 0.3,  0.8]
dog_emb =  [0.3, -0.4, 0.2,  0.7]  # Similar to cat
car_emb =  [-0.8, 0.6, -0.2, 0.1]  # Different from cat

# Cosine similarity
def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

print(cosine_sim(cat_emb, dog_emb))  # 0.99 (very similar!)
print(cosine_sim(cat_emb, car_emb))  # -0.15 (not similar)
```

---

## 📚 Types of Embeddings

### 1. **Word Embeddings**

Traditional approach: One embedding per word.

```
Vocabulary: 50,000 words
Embedding dimension: 300

Embedding matrix: 50,000 × 300 = 15 million parameters

Examples:
"king"    → [0.23, -0.45, 0.67, ..., 0.12]
"queen"   → [0.21, -0.43, 0.65, ..., 0.15]  (similar!)
"man"     → [0.18, -0.52, 0.41, ..., 0.09]
"woman"   → [0.16, -0.50, 0.39, ..., 0.11]  (similar!)

Famous equation:
king - man + woman ≈ queen
```

**Classic methods:** Word2Vec, GloVe

---

### 2. **Token Embeddings** (Modern Approach)

Subword-based: Handle any word by breaking into pieces.

```
Word: "unhappiness"

Traditional: [UNK] (unknown word) ❌

Token-based:
"unhappiness" → ["un", "happiness"] → [token_15, token_234]
                                    ↓
                        [emb_15] + [emb_234]

Benefits:
✅ Handle rare/new words
✅ Smaller vocabulary (5K-50K tokens vs 100K+ words)
✅ Multilingual capability
```

📌 **μOmni uses BPE (Byte-Pair Encoding)** with 32000 tokens

---

### 3. **Positional Embeddings**

Encode position information.

```
Without position:
"cat chased dog" = "dog chased cat" (identical embeddings!)

With position:
Position 0: [0.1, 0.0, 0.1, ...]
Position 1: [0.0, 0.1, 0.2, ...]
Position 2: [-0.1, 0.1, 0.0, ...]

"cat" at position 0:
word_emb + pos_emb_0 = [0.2, -0.5, 0.3, ...] + [0.1, 0.0, 0.1, ...]
                     = [0.3, -0.5, 0.4, ...]

Now "cat" at different positions has different representations!
```

📌 **μOmni uses RoPE** (Rotary Position Embeddings) - doesn't add, rotates!

---

### 4. **Multimodal Embeddings**

Embeddings for non-text modalities.

```
IMAGE EMBEDDINGS:
Image → Vision Encoder → Patch embeddings
  🖼️  → ViT           → (196 patches, 128 dim each)
                       ↓ Vision Projector
                       → (196 patches, 256 dim) [aligned with text]

AUDIO EMBEDDINGS:
Audio → Audio Encoder → Frame embeddings
  🎤  → AuT           → (T frames, 192 dim each)
                       ↓ Audio Projector
                       → (T frames, 256 dim) [aligned with text]

All embeddings in same 256-dim space!
```

---

## 🎯 Embedding Dimensions

### Trade-offs

| Dimension | Pros | Cons | Use Case |
|-----------|------|------|----------|
| **64-128** | Fast, memory-efficient | Limited expressiveness | Small models, mobile |
| **256-512** | Good balance | Moderate memory | μOmni, efficient models |
| **768-1024** | Very expressive | Higher memory | BERT, GPT-2 |
| **2048-4096** | Maximum capacity | Very expensive | GPT-3, Large models |

```
Visualization (capacity):

64-dim:  ▓       (basic relationships)
256-dim: ▓▓▓▓    (μOmni - good expressiveness)
768-dim: ▓▓▓▓▓▓▓ (BERT - high expressiveness)
```

### Parameter Count

```python
# μOmni's Thinker embeddings
vocab_size = 5000
d_model = 256

params = vocab_size × d_model = 1,280,000

# GPT-3 embeddings
vocab_size = 50,257
d_model = 12,288

params = vocab_size × d_model = 617,558,016 (617 million just for embeddings!)
```

---

## 🏗️ How Embeddings Are Learned

### 1. **Random Initialization**

```python
import torch
import torch.nn as nn

# Start with random embeddings
embedding_layer = nn.Embedding(vocab_size=5000, embedding_dim=256)

# Initial embedding for token 42:
print(embedding_layer.weight[42])
# → tensor([0.0231, -0.0145, 0.0367, ...]) (random noise)
```

---

### 2. **Training Updates Embeddings**

```
Training process:

Epoch 1:
"The cat sat" → Predict "on"
Embedding for "cat" gets updated based on context

Epoch 2:
"A cat sleeps" → Predict "quietly"
Embedding for "cat" gets updated again

...

After thousands of updates:
"cat" embedding now captures:
- It's an animal
- Often appears with "dog", "pet"
- Subject of verbs like "sleep", "eat", "play"
- Semantic meaning learned from context!
```

---

### 3. **Contextualized Embeddings** (Transformers)

Modern approach: Embeddings change based on context!

```
Static embedding (Word2Vec):
"bank" → always [0.2, -0.5, 0.3, ...]  (same regardless of context)

Contextualized embedding (Transformer):
"river bank" → [0.2, -0.5, 0.3, ...]  (geological meaning)
"bank account" → [0.8, 0.3, -0.2, ...] (financial meaning)
                  ↑ Different embedding for same word!

How:
1. Start with static embedding
2. Pass through transformer layers
3. Attention mechanism adjusts embedding based on surrounding words
```

---

## 💻 Embedding Lookup in Code

```python
import torch
import torch.nn as nn

# Create embedding layer
vocab_size = 5000
d_model = 256
embedding = nn.Embedding(vocab_size, d_model)

# Input: token IDs
input_ids = torch.tensor([15, 234, 42, 1893])  # [the, cat, sat, on]

# Lookup embeddings
embeddings = embedding(input_ids)
# Output shape: (4, 256)
# Each of the 4 tokens now has a 256-dimensional vector

print(embeddings.shape)  # torch.Size([4, 256])
print(embeddings[1])     # Embedding for "cat" (token 234)
```

---

## 🎨 Visualizing Embeddings

### Dimensionality Reduction

Embeddings are high-dimensional (256+), but we can visualize in 2D:

```
Technique: t-SNE or UMAP

256-dim embeddings → 2-dim visualization

Result:
      ┌────────────────────────┐
      │    ●cat ●dog           │
      │  ●kitten  ●puppy       │
      │                        │
      │             ▲car ▲bike│
      │           ▲truck       │
      │                        │
      │   ■pizza               │
      │■pasta    ■burger       │
      └────────────────────────┘

Similar concepts cluster together!
```

---

## 🔗 Embedding Similarity Operations

### 1. **Finding Similar Words**

```python
def find_similar(word_embedding, all_embeddings, top_k=5):
    # Compute cosine similarity with all words
    similarities = cosine_similarity(word_embedding, all_embeddings)
    # Get top-k most similar
    top_indices = similarities.argsort()[-top_k:]
    return top_indices

# Example
similar_to_cat = find_similar(embedding_layer.weight[cat_id], 
                               embedding_layer.weight)
# Returns: [dog_id, kitten_id, feline_id, pet_id, animal_id]
```

---

### 2. **Embedding Arithmetic**

```python
# Famous example: king - man + woman ≈ queen
king_emb = embedding_layer.weight[king_id]
man_emb = embedding_layer.weight[man_id]
woman_emb = embedding_layer.weight[woman_id]

result = king_emb - man_emb + woman_emb

# Find closest embedding
closest_id = find_most_similar(result, embedding_layer.weight)
print(id_to_word[closest_id])  # → "queen"
```

---

## 🎯 μOmni's Embedding Strategy

### Token Embeddings (Thinker)

```python
# In omni/thinker.py
class ThinkerLM(nn.Module):
    def __init__(self, vocab, d_model, ...):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab, d_model)
        # vocab=5000, d_model=256
        # Parameters: 5000 × 256 = 1,280,000
```

---

### Vision Embeddings

```
Image (224×224×3) 
    ↓
Patch Embedding (16×16 patches)
    ↓
196 patch embeddings (14×14 grid)
    ↓
Add CLS token
    ↓
197 tokens × 128 dim
    ↓
Vision Encoder (ViT-Tiny)
    ↓
CLS token extracted (1, 128)
    ↓
Vision Projector (Linear 128→256)
    ↓
Final embedding (1, 256)
```

---

### Audio Embeddings

```
Audio waveform (16kHz)
    ↓
Mel Spectrogram (T, 128)
    ↓
Convolutional Downsampling (4x or 8x)
    ↓
(T/8, 128) time-frequency features
    ↓
Flatten & Project (128×C → 192)
    ↓
Audio Encoder Transformer
    ↓
Frame embeddings (T/8, 192)
    ↓
Audio Projector (Linear 192→256)
    ↓
Final embeddings (T/8, 256)
```

---

## 📊 Embedding Quality Metrics

### How to Evaluate Embeddings?

1. **Intrinsic Evaluation**
```
Word similarity tasks:
Human rating: "cat" and "dog" similarity = 8/10
Model cosine sim: 0.82
→ High correlation = good embeddings
```

2. **Extrinsic Evaluation**
```
Use embeddings in downstream task:
- Text classification accuracy
- Translation quality
- Question answering performance

Better embeddings → Better task performance
```

3. **Visualization**
```
t-SNE plot:
- Clear clusters = good structure
- Semantic relationships visible = meaningful embeddings
```

---

## ⚡ Practical Tips

### 1. **Embedding Initialization**

```python
# Good practice: Use reasonable initialization
embedding = nn.Embedding(vocab_size, d_model)
nn.init.normal_(embedding.weight, mean=0, std=0.02)
# Prevents large initial values that could cause training instability
```

---

### 2. **Embedding Dropout**

```python
# Prevent overfitting on specific token embeddings
embedding_dropout = nn.Dropout(0.1)
embeddings = embedding_dropout(embedding(input_ids))
```

---

### 3. **Freezing Embeddings**

```python
# When fine-tuning, sometimes freeze embeddings to focus on other layers
embedding.weight.requires_grad = False  # Freeze
embedding.weight.requires_grad = True   # Unfreeze
```

---

### 4. **Embedding Normalization**

```python
# Normalize embeddings to unit length (sometimes helpful)
import torch.nn.functional as F
normalized_emb = F.normalize(embedding.weight, p=2, dim=1)
```

---

## 💡 Key Takeaways

✅ **Embeddings** = Dense vectors that represent semantic meaning  
✅ **Learned from data** during training via backpropagation  
✅ **Similar concepts** have similar embeddings (close in vector space)  
✅ **Token embeddings** handle subwords (better than word embeddings)  
✅ **Positional embeddings** add position information  
✅ **Multimodal embeddings** project different modalities to shared space  
✅ **Dimension choice** balances expressiveness vs computational cost  
✅ **μOmni uses 256-dim** embeddings across all modalities

---

## 🎓 Self-Check Questions

1. What problem do embeddings solve compared to one-hot encoding?
2. Why are similar words close in embedding space?
3. What's the difference between word and token embeddings?
4. How many parameters are in an embedding layer with vocab=10000 and d=512?
5. Why does μOmni project all modalities to the same dimension (256)?

<details>
<summary>📝 Click to see answers</summary>

1. Embeddings are dense, low-dimensional, and capture semantic meaning (vs sparse, high-dimensional, no meaning)
2. Because embeddings are learned from context - words appearing in similar contexts get similar vectors
3. Word embeddings: one vector per word. Token embeddings: use subword units (can handle any word by combining tokens)
4. 10,000 × 512 = 5,120,000 parameters
5. To enable cross-modal attention - all modalities must be in the same embedding space to interact in the Thinker
</details>

---

## ➡️ Next Steps

Now you understand embeddings! Let's explore the attention mechanism that processes them.

[Continue to Chapter 07: Attention Mechanism Deep Dive →](07-attention-mechanism.md)

Or return to the [Index](00-INDEX.md) to choose a different chapter.

---

**Chapter Progress:** Core Concepts ●○○○○○○ (1/7 complete)

