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

### The Fundamental Problem

Before we define tokenization, let's understand WHY we need it:

**The Problem: Computers Can't Read**

```
You type: "Hello world!"

What YOU see: Words with meaning

What the COMPUTER sees: A string of characters
'H' 'e' 'l' 'l' 'o' ' ' 'w' 'o' 'r' 'l' 'd' '!'

The computer needs to:
1. Break this into pieces it can understand
2. Convert each piece to a number (remember: neural networks need numbers!)
3. Look up embeddings for each piece

But what SIZE pieces? That's where tokenization comes in!
```

**Tokenization** = The process of converting text into smaller units (tokens) that the model can process.

**Analogy: Cutting a Sandwich**

```
You have a long sandwich (text):
"Hello world!"

How do you cut it?

Option 1: Slice into tiny pieces (characters)
[H][e][l][l][o][ ][w][o][r][l][d][!]
Pros: Every letter separate
Cons: Too many pieces! Hard to see the "words"

Option 2: Cut into large chunks (words)
[Hello] [world] [!]
Pros: Clear units of meaning
Cons: What about "antidisestablishmentarianism"? One huge piece!

Option 3: Smart cutting (subwords)
[Hel][lo] [world] [!]
OR for long words:
"antidisestablishmentarianism" → [anti][dis][establish][ment][ari][an][ism]
Pros: Balance between size and meaning!
```

**Different Tokenization Strategies:**

```
Input text: "Hello world!"

Character-level (cut every letter):
["H", "e", "l", "l", "o", " ", "w", "o", "r", "l", "d", "!"]
→ 12 tokens

Word-level (cut by spaces):
["Hello", "world", "!"]
→ 3 tokens

Subword-level (BPE - smart cutting):
["Hello", "world", "!"]
OR
["Hel", "lo", "world", "!"]
→ 3-4 tokens (flexible based on vocabulary!)
```

**Why This Matters:**

```
Fewer tokens = Faster processing, less memory
But: Need large vocabulary (more parameters)

More tokens = More processing, more memory
But: Smaller vocabulary (fewer parameters)

Subword tokenization = The Goldilocks solution! ⭐
(Not too many tokens, not too large vocabulary - just right!)
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

### Understanding BPE (The Smart Way to Build a Vocabulary!)

**The Big Idea:**

Start with individual characters, then gradually merge the most common pairs!

**Analogy: Building LEGO Blocks**

```
You have individual LEGO pieces (characters):
a, b, c, d, e, ...

You notice you're ALWAYS building the same combinations:
- "th" appears together ALL THE TIME
- "ing" appears together FREQUENTLY
- "cat" appears together OFTEN

Instead of grabbing "c", "a", "t" separately every time,
create a PRE-BUILT block "cat"!

That's BPE! Build frequently-used combinations into single tokens!
```

### Algorithm (Step by Step)

Let me walk you through BPE with a simple example:

**Example Text:** "aaabdaaabac"

```
INITIAL STATE:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Vocabulary: {a, b, c, d}  (just individual characters)
Text: "aaabdaaabac"

Think: Every letter is separate!
Visual: [a][a][a][b][d][a][a][a][b][a][c]

ITERATION 1: Find most common pair
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Count all adjacent pairs:
- "aa" appears: 1, 2, 3 (at start), 5, 6, 7 (middle) = 4 times!
- "ab" appears: position 3-4, position 7-8 = 2 times
- "bd" appears: position 4-5 = 1 time
- "da" appears: position 5-6 = 1 time
- "ba" appears: position 8-9 = 1 time
- "ac" appears: position 9-10 = 1 time

Winner: "aa" (appears 4 times - most frequent!)

MERGE: Create new token for "aa"
Let's call it "Z"

Update vocabulary: {a, b, c, d, Z}  where Z = "aa"
Update text: "aaabdaaabac"
             ↓↓         ↓↓
           "ZabdZabac"

Visual: [Z][a][b][d][Z][a][b][a][c]
(Reduced from 11 tokens to 9 tokens!)

ITERATION 2: Find most common pair (again)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Current text: "ZabdZabac"
Count pairs:
- "Za" appears: position 0-1, position 4-5 = 2 times
- "ab" appears: position 1-2, position 5-6 = 2 times
- "bd" appears: position 2-3 = 1 time
- "dZ" appears: position 3-4 = 1 time
- "ba" appears: position 6-7 = 1 time
- "ac" appears: position 7-8 = 1 time

Tie! Let's pick "Za" (arbitrary choice in tie)

MERGE: Create new token for "Za"
Let's call it "Y"

Update vocabulary: {a, b, c, d, Z, Y}  where Z="aa", Y="Za"="Zaa"
Update text: "ZabdZabac"
             ↓           ↓
           "YbdYbac"

Visual: [Y][b][d][Y][b][a][c]
(Reduced from 9 tokens to 7 tokens!)

KEEP REPEATING...
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Until vocabulary reaches desired size (e.g., 5000 tokens)
```

**Key Insight:**

```
BPE builds vocabulary from the DATA itself!

Frequent combinations get their own tokens:
- If "cat" appears often → it becomes ONE token
- If "dog" appears often → it becomes ONE token
- If "antidis" appears rarely → stays as ["anti", "dis"]

Result: Efficient representation!
- Common words: Few tokens
- Rare words: Multiple subword tokens (but still representable!)
```

**Real Example: "unhappiness"**

```
Step-by-step tokenization:

Initial: [u][n][h][a][p][p][i][n][e][s][s] (11 characters)

After BPE training on large corpus:
- "un" is common → merge to token
- "happ" is common → merge to token
- "iness" is common → merge to token

Result: [un][happ][iness] (3 tokens!)

If "unhappiness" was rare, might be:
[un][happiness] (2 tokens)

If it never appeared in training:
[un][h][app][i][n][ess] (6 tokens - still representable!)
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
from omni.tokenizer import BPETokenizer

# Train BPE tokenizer (μOmni's implementation)
tokenizer = BPETokenizer.train_new(
    text_path='corpus.txt',           # Training text file
    out_model='tokenizer.model',      # Output model path
    vocab_size=32000,                 # Vocabulary size
    max_sentence_length=100000,       # Max sentence length in bytes
    input_sentence_size=10000000      # Default: 10M sentences (faster training)
)

# Use all sentences (slower but uses more data):
tokenizer = BPETokenizer.train_new(
    text_path='corpus.txt',
    out_model='tokenizer.model',
    vocab_size=32000,
    input_sentence_size=0             # 0 = use all sentences
)

# The train_new method automatically:
# - Enables train_extremely_large_corpus=True (64-bit indexing for files > 2GB)
# - Uses BPE algorithm (faster than Unigram)
# - Handles files of any size (no file size checking or streaming)
```

**Key Implementation Details:**
- ✅ **Always enables `train_extremely_large_corpus`:** Uses 64-bit indexing instead of 32-bit, allowing training on files > 2GB
- ✅ **BPE model type:** Faster than Unigram, good balance of speed and quality
- ✅ **Default speed optimization:** `input_sentence_size=10000000` (10M sentences) for faster training by default
- ✅ **Use all data:** Set `input_sentence_size=0` to use entire corpus (slower but uses more data)
- ✅ **No file size limits:** Works with files of any size
- ✅ **No streaming:** Files are passed directly to SentencePiece (which loads entire file into memory)
- ⚠️ **Memory note:** SentencePiece loads the entire file into RAM during training, regardless of size

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

