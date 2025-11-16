# Chapter 09: Tokenization and Vocabularies

[← Previous: Positional Encoding](08-positional-encoding.md) | [Back to Index](00-INDEX.md) | [Next: Audio Processing →](10-audio-processing.md)

---

## 🎯 What You'll Learn

- What tokenization is and why it's needed
- Different tokenization methods (BPE, WordPiece, SentencePiece)
- Building and using vocabularies
- How μOmni tokenizes text

---

## 📝 What is Tokenization?

**Tokenization** = Converting text into smaller units (tokens) that the model can process.

```
Input text: "Hello world!"

Character-level:
["H", "e", "l", "l", "o", " ", "w", "o", "r", "l", "d", "!"]

Word-level:
["Hello", "world", "!"]

Subword-level (BPE):
["Hello", "world", "!"]
OR
["Hel", "lo", "world", "!"]
```

---

## 🔤 Tokenization Methods

### 1. **Character-Level**

```
Text: "cat"
Tokens: ["c", "a", "t"]

Pros:
✅ Small vocabulary (~100 characters)
✅ No unknown words

Cons:
❌ Long sequences
❌ Less semantic meaning per token
❌ Harder to learn word-level patterns
```

---

### 2. **Word-Level**

```
Text: "The cat sat"
Tokens: ["The", "cat", "sat"]

Pros:
✅ Semantic meaning per token
✅ Shorter sequences

Cons:
❌ Huge vocabulary (100K+ words)
❌ Unknown words become [UNK]
❌ Doesn't handle morphology ("running" ≠ "run")
```

---

### 3. **Subword-Level** (Modern Standard) ⭐

```
Text: "unhappiness"
Tokens: ["un", "happiness"]

Pros:
✅ Moderate vocabulary (5K-50K)
✅ No unknown words (can build any word)
✅ Handles morphology and rare words
✅ Efficient sequence length

Common algorithms:
- BPE (Byte-Pair Encoding)
- WordPiece (BERT)
- SentencePiece (multilingual)
```

📌 **μOmni uses BPE** with 5000 tokens

---

## 🔨 BPE (Byte-Pair Encoding)

### Algorithm

```
Step 1: Start with character vocabulary
Vocab: {a, b, c, d, ...}

Step 2: Count adjacent pair frequencies
Text: "aaabdaaabac"
Pairs: aa=4, ab=2, bd=1, da=2, ac=1

Step 3: Merge most frequent pair (aa → Z)
Text: "ZabdZabac"
New token: Z = "aa"

Step 4: Repeat until desired vocabulary size
```

---

### Example: Building BPE Vocabulary

```
Corpus: "low low low lowest"

Iteration 1:
Frequencies: l=4, o=4, w=4, e=1, s=1, t=1
Pair frequencies: "lo"=4, "ow"=4
Merge: "lo" → token_1
Result: "token_1w token_1w token_1w token_1west"

Iteration 2:
Pair frequencies: "token_1w"=3
Merge: "token_1w" → token_2
Result: "token_2 token_2 token_2 token_1west"

Final vocabulary:
{l, o, w, e, s, t, token_1="lo", token_2="low"}
```

---

## 📚 Vocabulary Creation

### μOmni's Tokenizer

```python
# From omni/tokenizer.py (simplified)
class BPETokenizer:
    def __init__(self, model_path):
        # Load pre-trained BPE model
        self.sp = sentencepiece.SentencePieceProcessor()
        self.sp.Load(model_path)
        
        # Special tokens
        self.bos_id = 1  # Beginning of sequence
        self.eos_id = 2  # End of sequence
        self.pad_id = 0  # Padding
        self.unk_id = 3  # Unknown
    
    def encode(self, text):
        """Convert text to token IDs"""
        return self.sp.EncodeAsIds(text)
    
    def decode(self, ids):
        """Convert token IDs back to text"""
        return self.sp.DecodeIds(ids)
```

---

### Example Usage

```python
tokenizer = BPETokenizer("tokenizer.model")

# Encoding
text = "The cat sat on the mat"
tokens = tokenizer.encode(text)
print(tokens)  # [15, 234, 42, 89, 15, 156]

# Decoding
decoded = tokenizer.decode(tokens)
print(decoded)  # "The cat sat on the mat"

# Handle unknown/rare words
text = "supercalifragilisticexpialidocious"
tokens = tokenizer.encode(text)
# Breaks into subwords: ["super", "cal", "ifrag", "ilis", "tic", ...]
```

---

## 🎯 Special Tokens

```
Common special tokens:

[PAD]  (ID: 0)  - Padding (for batching)
[BOS]  (ID: 1)  - Beginning of sequence
[EOS]  (ID: 2)  - End of sequence
[UNK]  (ID: 3)  - Unknown token
[MASK] (ID: 4)  - Masked token (BERT-style)
[SEP]  (ID: 5)  - Separator
[CLS]  (ID: 6)  - Classification token

μOmni uses:
- BOS: Start of generation
- EOS: End of generation
- PAD: Batch padding
```

---

## 📊 Vocabulary Size Trade-offs

| Vocab Size | Sequence Length | Training Speed | Memory | Coverage |
|------------|-----------------|----------------|--------|----------|
| **1K** | Very long | Fast | Low | Poor (many UNK) |
| **5K** | Long | Fast | Low | Good | ← **μOmni**
| **30K** | Medium | Medium | Medium | Excellent |
| **50K** | Short | Slow | High | Excellent |

```
Example: "Hello world"

Vocab 1K:   ["Hel", "lo", "wo", "rld"]  (4 tokens)
Vocab 5K:   ["Hello", "world"]           (2 tokens)
Vocab 50K:  ["Hello", "world"]           (2 tokens, but 10x more params)
```

---

## 🌍 Multilingual Tokenization

### Challenge

```
Different languages, different scripts:

English: "Hello world"
Spanish: "Hola mundo"
Chinese: "你好世界"
Arabic:  "مرحبا بالعالم"

How to handle all with one tokenizer?
```

### Solution: SentencePiece + BPE

```python
# SentencePiece handles raw text (language-agnostic)
# No preprocessing needed

tokenizer = sentencepiece.SentencePieceProcessor()
tokenizer.Load("multilingual.model")

# Works for any language
en_tokens = tokenizer.encode("Hello")
zh_tokens = tokenizer.encode("你好")
ar_tokens = tokenizer.encode("مرحبا")
```

📌 **μOmni's tokenizer** is trained on English but can extend to multilingual

---

## 💻 Training Your Own Tokenizer

```python
import sentencepiece as spm

# Train BPE tokenizer
spm.SentencePieceTrainer.train(
    input='corpus.txt',           # Training text
    model_prefix='tokenizer',     # Output prefix
    vocab_size=5000,              # Desired vocabulary size
    model_type='bpe',             # BPE algorithm
    character_coverage=0.9995,    # Character coverage
    pad_id=0,                     # Padding token ID
    bos_id=1,                     # BOS token ID
    eos_id=2,                     # EOS token ID
    unk_id=3                      # Unknown token ID
)

# Output: tokenizer.model, tokenizer.vocab
```

---

## 🔍 Tokenization in Action

### Example: Complete Pipeline

```python
# 1. Initialize tokenizer
tokenizer = BPETokenizer("checkpoints/thinker_tiny/tokenizer.model")

# 2. Encode text
text = "The quick brown fox"
token_ids = [1] + tokenizer.encode(text)  # Add BOS token
print(f"Text: {text}")
print(f"Token IDs: {token_ids}")
# [1, 15, 234, 876, 342]

# 3. Convert to tensor
import torch
input_ids = torch.tensor(token_ids).unsqueeze(0)  # (1, seq_len)

# 4. Get embeddings
model = ThinkerLM(vocab=5000, d_model=256, ...)
embeddings = model.tok_emb(input_ids)
print(f"Embeddings shape: {embeddings.shape}")
# torch.Size([1, 5, 256])

# 5. Process through model
output_logits = model(input_ids)

# 6. Decode output
predicted_id = torch.argmax(output_logits[0, -1]).item()
next_token = tokenizer.decode([predicted_id])
print(f"Next token: {next_token}")
```

---

## 🎯 Tokenization Best Practices

### 1. **Consistent Preprocessing**

```python
# Always preprocess the same way
def preprocess(text):
    text = text.lower()           # Lowercase (optional)
    text = text.strip()           # Remove whitespace
    return text

# Training
train_tokens = tokenizer.encode(preprocess(train_text))

# Inference
test_tokens = tokenizer.encode(preprocess(test_text))
```

---

### 2. **Handle Special Cases**

```python
# Numbers
tokenizer.encode("1234567")
# Better: ["123", "456", "7"] than ["1", "2", "3", "4", "5", "6", "7"]

# URLs/emails
tokenizer.encode("user@example.com")
# Should handle as single token or meaningful subwords

# Code
tokenizer.encode("def function():")
# Should preserve syntax structure
```

---

### 3. **Vocabulary Management**

```python
# Check if token exists
if tokenizer.encode("rare_word") == [tokenizer.unk_id]:
    print("Unknown word!")

# Get vocabulary size
vocab_size = tokenizer.vocab_size()
print(f"Vocabulary size: {vocab_size}")

# Inspect vocabulary
for i in range(10):
    token = tokenizer.id_to_piece(i)
    print(f"ID {i}: {token}")
```

---

## 💡 Key Takeaways

✅ **Tokenization** breaks text into processable units  
✅ **Subword tokenization** (BPE) is the modern standard  
✅ **BPE** merges frequent character pairs iteratively  
✅ **Small vocabulary** (5K-50K) balances efficiency and coverage  
✅ **Special tokens** (BOS, EOS, PAD) manage sequences  
✅ **SentencePiece** enables language-agnostic tokenization  
✅ **μOmni uses BPE** with 5000 tokens

---

## 🎓 Self-Check Questions

1. What's the difference between word-level and subword-level tokenization?
2. How does BPE build its vocabulary?
3. Why is subword tokenization better than word-level for rare words?
4. What are special tokens used for?
5. What vocabulary size does μOmni use?

<details>
<summary>📝 Answers</summary>

1. Word-level: each word is one token. Subword: words broken into smaller meaningful units
2. BPE starts with characters and iteratively merges the most frequent adjacent pairs
3. Subword can build any rare word from subword pieces (no [UNK] tokens needed)
4. Special tokens mark boundaries (BOS/EOS), padding (PAD), and handle unknowns (UNK)
5. 5000 tokens (balances efficiency and expressiveness)
</details>

---

[Continue to Chapter 10: Audio Processing →](10-audio-processing.md)

**Chapter Progress:** Core Concepts ●●●●○○○ (4/7 complete)

