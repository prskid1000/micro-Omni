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

```
Computer sees "cat": ???
What does this mean? How to process it mathematically?

Need to convert: "cat" → Numbers
```

### Naive Approach: One-Hot Encoding

```
Vocabulary: ["cat", "dog", "bird", "fish"]

"cat"  → [1, 0, 0, 0]
"dog"  → [0, 1, 0, 0]
"bird" → [0, 0, 1, 0]
"fish" → [0, 0, 0, 1]

Problems:
❌ High dimensionality (vocab size = vector length)
❌ No semantic meaning ("cat" and "dog" are equally different from each other)
❌ Sparse representation (mostly zeros)
```

---

## 💎 The Solution: Dense Embeddings

### What are Embeddings?

**Embeddings** are dense, low-dimensional vector representations that capture semantic meaning.

```
Instead of: "cat" → [1, 0, 0, 0, ..., 0] (10,000 dimensions, sparse)

Use: "cat" → [0.2, -0.5, 0.3, 0.8, -0.1, ...] (256 dimensions, dense)

Benefits:
✅ Low-dimensional (256-1024 dims instead of vocab size)
✅ Dense (all values meaningful)
✅ Semantic meaning (similar words → similar vectors)
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

📌 **μOmni uses BPE (Byte-Pair Encoding)** with 5000 tokens

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

