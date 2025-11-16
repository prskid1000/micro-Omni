# Chapter 13: Decoder-Only Language Models

[← Previous: Vector Quantization](12-quantization.md) | [Back to Index](00-INDEX.md) | [Next: KV Caching →](14-kv-caching.md)

---

## 🎯 Learning Objectives

By the end of this chapter, you will understand:
- What decoder-only models are and why they're powerful
- How causal attention works
- Autoregressive generation step-by-step
- Why μOmni uses decoder-only architecture
- Difference between encoder and decoder models

---

## 📖 Understanding Decoder-Only Models

### The Big Picture: Two Ways to Build Language Models

**Analogy: Reading a Book**

```
ENCODER (BERT-style): Reading for Comprehension
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
You read an entire sentence, then answer questions about it.

"The cat sat on the mat"
↓ Read the WHOLE thing
↓ Can look back and forward
↓ Understand the complete meaning

Use case: "What is on the mat?" → "The cat"
Best for: Understanding, classification, question answering

DECODER (GPT-style): Writing a Story
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
You write one word at a time, building as you go.

"The cat sat on ___"
↓ You can only look at what you've written so far
↓ Can't peek at future words (they don't exist yet!)
↓ Predict the next word: "the"

Use case: Text generation, chat, completion
Best for: Generation, conversation, creativity

μOmni uses DECODER (GPT-style)! ⭐
```

### Why "Decoder-Only"?

```
The name comes from the original Transformer paper which had:
- Encoder: Processes input
- Decoder: Generates output

"Decoder-only" means:
- We only use the decoder part!
- No separate encoder needed
- Just generate, generate, generate!

Famous decoder-only models:
- GPT (all versions)
- LLaMA
- PaLM
- μOmni's Thinker ✓
```

---

## 🏗️ Architecture Deep Dive

### How Decoder-Only Models Work

**Step-by-Step Example:**

```
Task: Complete "The cat sat on ___"

INITIAL STATE:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Input: "The cat sat on"
Tokens: [15, 234, 42, 89]

PROCESSING:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. Embed tokens
   ↓
2. Add positional information (RoPE)
   ↓
3. Causal Self-Attention (can only see previous tokens)
   - "The" can only see "The"
   - "cat" can see "The cat"
   - "sat" can see "The cat sat"
   - "on" can see "The cat sat on"
   ↓
4. Feedforward Network (process each position)
   ↓
5. Layer Normalization
   ↓
6. Repeat for multiple layers (e.g., 4 layers)
   ↓
7. Output: Probability distribution over vocabulary

PREDICTION:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Logits for next token:
- "the": 0.45 (45% probability) ← Most likely!
- "a": 0.20 (20%)
- "mat": 0.15 (15%)
- "floor": 0.10 (10%)
- ...

Pick "the" → Output: "The cat sat on the"
```

**The Architecture in Detail:**

```
┌───────────────────────────────────────┐
│     INPUT: Token IDs [15, 234, 42]   │
└──────────────┬────────────────────────┘
               ↓
┌───────────────────────────────────────┐
│     Token Embedding Layer             │
│     [15, 234, 42] → [[0.2,...], ...]  │
└──────────────┬────────────────────────┘
               ↓
┌───────────────────────────────────────┐
│     Positional Encoding (RoPE)        │
│     Add position information          │
└──────────────┬────────────────────────┘
               ↓
    ┌──────────────────────┐
    │   DECODER BLOCK 1    │
    │  ┌────────────────┐  │
    │  │ Causal Attn    │  │  ← Can't see future!
    │  └───────┬────────┘  │
    │          ↓            │
    │  ┌────────────────┐  │
    │  │ Feedforward    │  │
    │  └────────────────┘  │
    └──────────┬───────────┘
               ↓
    ┌──────────────────────┐
    │   DECODER BLOCK 2    │
    └──────────┬───────────┘
               ↓
    ┌──────────────────────┐
    │   DECODER BLOCK 3    │
    └──────────┬───────────┘
               ↓
    ┌──────────────────────┐
    │   DECODER BLOCK 4    │
    └──────────┬───────────┘
               ↓
┌───────────────────────────────────────┐
│     Output Linear Layer               │
│     256 dim → 5000 vocab size         │
└──────────────┬────────────────────────┘
               ↓
┌───────────────────────────────────────┐
│     Softmax                           │
│     Convert to probabilities          │
└──────────────┬────────────────────────┘
               ↓
┌───────────────────────────────────────┐
│     PREDICTION                        │
│     Next token ID: 156                │
└───────────────────────────────────────┘
```

---

## 🔑 Key Feature: Causal Masking

### Understanding "Causal" Attention

**Analogy: Writing an Essay vs Reading an Essay**

```
WRITING (Causal - Decoder):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
You're writing: "The cat sat on the ___"

When deciding what to write next:
✓ You CAN look at: "The cat sat on the"
✗ You CAN'T look at: Future words (they don't exist yet!)

This is CAUSAL attention - you can only see the PAST!

READING (Bidirectional - Encoder):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
You're reading: "The cat sat on the mat"

When understanding "sat":
✓ You CAN look at: "The cat" (before)
✓ You CAN look at: "on the mat" (after)

This is BIDIRECTIONAL attention - you can see EVERYTHING!
```

### The Attention Mask Visualized

```
Attention mask (lower triangular):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
         The  cat  sat  on  the  mat
The      ██   ░░   ░░   ░░  ░░   ░░   ← "The" can only see itself
cat      ██   ██   ░░   ░░  ░░   ░░   ← "cat" sees "The" and "cat"
sat      ██   ██   ██   ░░  ░░   ░░   ← "sat" sees up to "sat"
on       ██   ██   ██   ██  ░░   ░░   ← "on" sees up to "on"
the      ██   ██   ██   ██  ██   ░░   ← "the" sees up to "the"
mat      ██   ██   ██   ██  ██   ██   ← "mat" sees everything

██ = Can attend (look at)
░░ = Masked out (can't see)

Shape: Lower triangular matrix
Why: Prevents "cheating" by looking at future tokens!
```

**Why is This Important?**

```
WITHOUT Causal Masking (cheating!):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Training: "The cat sat on the mat"
When predicting "on":
- Model can see "the mat" (future words!)
- Learns to cheat: "Oh, 'mat' comes later, so 'on' makes sense"

Testing: "The cat sat on ___"
- No future words available!
- Model is confused: "Where's 'mat'? I need it!"
- Performance collapses! ❌

WITH Causal Masking (no cheating!):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Training: "The cat sat on the mat"
When predicting "on":
- Model can only see "The cat sat on"
- Learns genuine patterns: "After 'on', what usually comes?"

Testing: "The cat sat on ___"
- Same setup as training!
- Model works perfectly! ✓

Causal masking ensures training = testing conditions!
```

---

## 🔄 Autoregressive Generation

### What Does "Autoregressive" Mean?

```
AUTO = Self
REGRESSIVE = Using previous outputs as inputs

In simple terms: Use your own output as the next input!

Like a conversation with yourself:
You: "The cat"
You: "sat" (based on "The cat")
You: "on" (based on "The cat sat")
You: "the" (based on "The cat sat on")
...
```

### Generation Process Step-by-Step

```
Goal: Generate "The cat sat on the mat"

STEP 1: Start with prompt
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Input: "The cat"
       [15, 234]
       ↓ Model
Predict: "sat" (token 42)
Output: "The cat sat"

STEP 2: Use previous output as new input
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Input: "The cat sat"
       [15, 234, 42]
       ↓ Model
Predict: "on" (token 89)
Output: "The cat sat on"

STEP 3: Keep going...
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Input: "The cat sat on"
       [15, 234, 42, 89]
       ↓ Model
Predict: "the" (token 15)
Output: "The cat sat on the"

STEP 4: Continue until done
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Input: "The cat sat on the"
       [15, 234, 42, 89, 15]
       ↓ Model
Predict: "mat" (token 156)
Output: "The cat sat on the mat"

STEP 5: Stop condition
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Model predicts [EOS] token (end of sequence)
OR max length reached
→ Stop generating!

Final output: "The cat sat on the mat" ✓
```

---

## 🆚 Encoder vs Decoder Comparison

### Detailed Comparison Table

| Feature | Encoder (BERT) | Decoder (GPT/μOmni) |
|---------|----------------|---------------------|
| **Attention** | Bidirectional (see all) | Causal (see past only) |
| **Task** | Understanding | Generation |
| **Training** | Masked LM (fill blanks) | Next-token prediction |
| **Input** | Complete sentence | Partial sentence |
| **Output** | Embeddings/classification | Next token |
| **Use Cases** | Classification, QA, NER | Chat, completion, generation |
| **Examples** | BERT, RoBERTa | GPT, LLaMA, μOmni |

### Visual Comparison

```
ENCODER (BERT):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Input: "The [MASK] sat on the mat"
       ↓ Bidirectional attention
Output: "cat" (fill in the blank)

Use: Understanding what fits in context

DECODER (GPT/μOmni):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Input: "The cat sat on the"
       ↓ Causal attention
Output: "mat" (predict next word)

Use: Generate continuation
```

---

## 💡 Why μOmni Uses Decoder-Only

```
Reasons for choosing decoder-only:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

✅ GENERATION TASKS:
   μOmni needs to:
   - Generate text responses
   - Generate speech (RVQ codes)
   → Decoder is perfect for generation!

✅ SIMPLICITY:
   Encoder-decoder: Two models to train
   Decoder-only: One model to train
   → Simpler architecture, easier to train

✅ PROVEN EFFECTIVENESS:
   GPT-3, GPT-4, LLaMA all use decoder-only
   → We know it works well!

✅ UNIFIED PROCESSING:
   Same architecture handles:
   - Text generation
   - Speech code generation
   → Consistent approach across modalities

✅ INTERACTIVE USE:
   Great for:
   - Chat applications
   - Completion tasks
   - Creative writing
   → Perfect for μOmni's use cases!
```

---

## 💡 Key Takeaways

✅ **Decoder-only** models generate text autoregressively  
✅ **Causal attention** prevents seeing future tokens (lower triangular mask)  
✅ **Autoregressive** means using previous outputs as new inputs  
✅ **One token at a time** generation (sequential process)  
✅ **Perfect for generation tasks** (text, speech, etc.)  
✅ **μOmni's Thinker** is decoder-only (GPT-style)  
✅ **Simpler than encoder-decoder** (one model, not two)

---

## 🎓 Self-Check Questions

1. What does "decoder-only" mean?
2. Why do we need causal masking?
3. What is autoregressive generation?
4. What's the difference between encoder and decoder models?
5. Why does μOmni use decoder-only architecture?

<details>
<summary>📝 Click to see answers</summary>

1. Using only the decoder part of transformers (no separate encoder), which generates outputs one token at a time
2. To prevent the model from "cheating" by seeing future tokens during training, ensuring training matches inference conditions
3. Using the model's own previous outputs as inputs for generating the next output (self-feeding generation loop)
4. Encoder: bidirectional attention for understanding. Decoder: causal attention for generation
5. Because μOmni needs to generate text and speech, and decoder-only architecture is proven effective for generation tasks
</details>

---

[Continue to Chapter 14: KV Caching →](14-kv-caching.md)

**Chapter Progress:** Advanced Architecture ●○○○ (1/4 complete)

---

