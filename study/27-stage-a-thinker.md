# Chapter 27: Stage A - Thinker Pretraining

[← Previous: Training Overview](26-training-overview.md) | [Back to Index](00-INDEX.md) | [Next: Stage B Audio →](28-stage-b-audio-encoder.md)

---

## 🎯 Learning Objectives

By the end of this chapter, you will understand:
- What Stage A trains and why it's the foundation
- How language modeling (next-token prediction) works
- The training process, loss functions, and metrics
- How to interpret perplexity
- Configuration parameters and their impact
- Expected training progress and outputs

---

## 💡 What is Stage A?

### The Foundation of μOmni

**Analogy: Learning to Read Before Multimodal Understanding**

```
Think of Stage A like learning your native language:

BEFORE MULTIMODAL (Stage A):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Baby learning:
- Hears words → "cat", "dog", "mat"
- Learns grammar → "The cat sat"
- Understands language → "on the mat"

This is FOUNDATIONAL!
Must understand language FIRST!

AFTER LANGUAGE (Later stages):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Now ready for:
- Looking at pictures (Stage C - Vision)
- Listening to sounds (Stage B - Audio)
- Speaking words (Stage D - Talker)

Language is the GLUE that connects everything!

STAGE A TRAINS THE THINKER:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Purpose: Teach the core LLM to understand text
- Grammar and syntax
- Common sense reasoning
- World knowledge
- Next-word prediction

Without this foundation, multimodal learning fails!
```

**Why Stage A Must Come First:**

```
Problem: What if we skip text pretraining?
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Try to teach multimodal directly:
- Show image + text: "What is this? A cat"
- Model has NO language foundation
- Doesn't understand "what", "is", "this", "a"
- Can't form coherent responses
- Training fails! ❌

With Stage A first:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. Learn language (Stage A)
   → Understands grammar, syntax, meaning

2. Add vision (later)
   → Already knows how to say "This is a cat"
   → Just needs to learn WHEN to say it!

3. Add audio (later)
   → Already understands transcriptions
   → Just needs to learn audio → text mapping!

Foundation enables efficient multimodal learning! ✓
```

---

## 📝 The Task: Next-Token Prediction

### How Language Models Learn

**The Fundamental Task:**

```
LANGUAGE MODELING = PREDICT NEXT WORD
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Given context: "The cat sat on the ___"
Task: Predict next word

Training example:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Input:  "The cat sat on the"
Target: "mat"

How it works:
1. Tokenize: ["The", "cat", "sat", "on", "the"] → [15, 234, 89, 42, 156]
2. Embed: Token IDs → (5, 256) embeddings
3. Thinker processes: (5, 256) → (5, 256)
4. Last position: (256,) → Project to vocab
5. Logits: (5000,) scores for each word
6. Softmax: Convert to probabilities
7. Loss: Compare to target "mat" (ID 923)

The model learns:
✅ "mat" follows "on the" (common phrase)
✅ Grammar: need noun after "the"
✅ Context: cats sit on mats/couches/laps
```

**Step-by-Step Example:**

```
Training on: "The cat sat on the mat"
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Position 0: Predict "cat"
Context: ["The"]
Target: "cat"
Model learns: After "The", often comes noun

Position 1: Predict "sat"
Context: ["The", "cat"]
Target: "sat"
Model learns: Subjects perform actions (verbs)

Position 2: Predict "on"
Context: ["The", "cat", "sat"]
Target: "on"
Model learns: "sat" often followed by preposition

Position 3: Predict "the"
Context: ["The", "cat", "sat", "on"]
Target: "the"
Model learns: Prepositions followed by articles

Position 4: Predict "mat"
Context: ["The", "cat", "sat", "on", "the"]
Target: "mat"
Model learns: Context suggests surface (mat/couch/lap)

Each prediction teaches language patterns! ✓
```

---

## 🏗️ Training Process Detailed

### From Random Weights to Language Understanding

**The Training Loop:**

```python
# Simplified training loop
for epoch in range(num_epochs):
    for batch in dataloader:
        # Batch: ["The cat sat...", "Hello world...", ...]
        
        # 1. Tokenize texts
        input_ids, target_ids = tokenize_and_shift(batch)
        # input_ids: (B, T-1) - all but last token
        # target_ids: (B, T-1) - all but first token
        
        # 2. Forward pass
        logits = thinker(input_ids)  # (B, T-1, vocab_size)
        
        # 3. Compute loss
        loss = cross_entropy(logits.view(-1, vocab_size),
                            target_ids.view(-1))
        
        # 4. Backward pass
        loss.backward()  # Compute gradients
        
        # 5. Gradient clipping (prevent explosions)
        torch.nn.utils.clip_grad_norm_(thinker.parameters(),
                                      max_grad_norm=1.0)
        
        # 6. Update weights
        optimizer.step()
        optimizer.zero_grad()
        
        # 7. Log progress
        perplexity = torch.exp(loss)
        print(f"Loss: {loss:.3f}, PPL: {perplexity:.1f}")
```

**What Happens Over Time:**

```
EPOCH 1 (Random initialization):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Step 1:
Context: "The cat sat on the"
Predictions: Random! [0.0002, 0.0003, ..., 0.0001]
Generated: "purple banana democracy" ← Nonsense!
Loss: 8.5 (very high)
Perplexity: 4914 (terrible!)

Step 100:
Context: "The cat sat on the"
Predictions: Slightly better [0.001, 0.05, ..., 0.002]
Generated: "a big thing" ← Vague but grammatical!
Loss: 4.2
Perplexity: 67

EPOCH 5 (Learning grammar):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Step 2500:
Context: "The cat sat on the"
Predictions: Much better! [0.001, 0.3, 0.4, ..., 0.002]
Generated: "mat" (40% prob), "couch" (30% prob)
Loss: 2.5
Perplexity: 12

EPOCH 10 (Good understanding):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Final:
Context: "The cat sat on the"
Predictions: Excellent! [0.001, 0.6, 0.25, ..., 0.001]
Generated: "mat" (60% prob), "couch" (25% prob), "floor" (10%)
Loss: 2.0
Perplexity: 7.4

Model has learned language! ✓
```

---

## 📊 Configuration Parameters Explained

### Understanding Each Setting

```json
{
  // MODEL ARCHITECTURE
  "vocab_size": 32000,      // How many unique words/tokens
                             // Smaller = faster, but less coverage
                             // 32000 = good balance (matches thinker_tiny.json)
  
  "n_layers": 4,             // Transformer depth
                             // More layers = more complex patterns
                             // 4 = tiny model (GPT-3 has 96!)
  
  "d_model": 256,            // Embedding dimension
                             // Bigger = more expressive
                             // 256 = efficient for small model
  
  "n_heads": 4,              // Attention heads
                             // More heads = different perspectives
                             // 4 = good for 256-dim model
  
  "d_ff": 1024,              // FFN hidden size (4x d_model)
                             // Standard ratio
  
  "dropout": 0.1,            // Prevent overfitting
                             // 10% neurons randomly dropped
  
  "rope_theta": 10000,       // RoPE frequency base
                             // Higher = slower position decay
  
  "ctx_len": 512,            // Max context length (tokens)
                             // How much text model can see
  
  // TRAINING DATA
  "data_path": "data/text/corpus.txt",  // Text file with training data
  
  "batch_size": 16,          // Examples per batch
                             // Larger = more stable, but more memory
  
  "num_epochs": 10,          // Complete passes through data
                             // More epochs = more learning
  
  "learning_rate": 3e-4,     // How fast to update weights
                             // 0.0003 = standard for Adam
  
  "warmup_steps": 1000,      // Gradually increase LR
                             // Prevents early instability
  
  "max_grad_norm": 1.0,      // Gradient clipping
                             // Prevents exploding gradients
  
  // CHECKPOINTING
  "save_every": 1000,        // Save checkpoint every N steps
  "eval_every": 500          // Evaluate every N steps
}
```

**Why These Values?**

```
TINY MODEL DESIGN:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Vocab 32000 vs 50000:
- 32000: Good balance, covers common words (used in tiny config)
- 50000: Better coverage, slower training

Layers 4 vs 12:
- 4: ~15M parameters, 12GB GPU OK
- 12: ~45M parameters, needs 24GB+ GPU

Context 512 vs 2048:
- 512: Most conversations fit
- 2048: Long documents, but 4x memory

We optimize for: FAST TRAINING ON SINGLE GPU
Good enough for proof-of-concept! ✓
```

---

## 📈 Metrics Explained

### Understanding Loss and Perplexity

**Cross-Entropy Loss:**

```
WHAT IS LOSS?
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Measures: How "surprised" model is by correct answer

Example:
Context: "The cat sat on the ___"
Target: "mat"

Model predictions (probabilities):
- "mat": 0.6    ← High probability for correct word!
- "couch": 0.25
- "floor": 0.10
- "banana": 0.001  ← Low for wrong word

Loss = -log(0.6) = 0.51  ← LOW LOSS (good!)

If model predicted:
- "mat": 0.01   ← Low probability for correct word!
- "banana": 0.5  ← High for wrong word

Loss = -log(0.01) = 4.6  ← HIGH LOSS (bad!)

Lower loss = better predictions! ✓
```

**Perplexity:**

```
WHAT IS PERPLEXITY?
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Perplexity = exp(loss)

Intuition: "How many words is model confused between?"

Examples:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Perplexity = 1:
Model is CERTAIN (100% confidence)
Like choosing from 1 word

Perplexity = 10:
Model considers about 10 plausible words
Pretty good!

Perplexity = 100:
Model is very confused
Considering 100 possible words!

Perplexity = 5000:
Random guessing (vocab size)
Model learned nothing!

TARGET FOR μOmni:
Perplexity 5-10 = Good understanding! ✓
```

---

## 📊 Expected Training Progress

### Typical Learning Curve

```
TRAINING TIMELINE (10 epochs, ~10 hours):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Hour 0 (Random Init):
Step 1: loss=8.517 ppl=4981
→ Model is guessing randomly
→ Output: "purple banana democracy xyz"

Hour 1 (Early Learning):
Step 100: loss=4.234 ppl=68.9
→ Learning basic patterns
→ Output: "a thing something word"

Hour 2-3 (Grammar Emerges):
Step 500: loss=3.156 ppl=23.4
Validation: loss=3.201 ppl=24.5
→ Valid sentences forming!
→ Output: "the mat on floor"

Hour 5 (Refinement):
Epoch 5/10:
Step 2500: loss=2.456 ppl=11.7
→ Good grammar and coherence
→ Output: "the soft mat"

Hour 10 (Final):
Epoch 10/10:
Final: loss=1.987 ppl=7.3
→ Excellent language understanding!
→ Output: "the comfortable mat"

READY FOR STAGE E! ✓
```

---

## 🎓 Output Files

### What Gets Saved

```
checkpoints/thinker_tiny/
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

├── thinker_step_1000.pt     # Periodic checkpoints (every 1000 steps)
├── thinker_step_2000.pt     # (Resume if crash!)
├── thinker_step_3000.pt
│
├── tokenizer.model          # Tokenizer (BPE model)
│   Maps: "hello" → 234
│         234 → "hello"
│
└── training_log.json        # Metrics history
    Step-by-step loss/perplexity

Load for Stage E:
```python
# Load the latest checkpoint
checkpoint = torch.load('checkpoints/thinker_tiny/thinker_step_3000.pt')
thinker.load_state_dict(checkpoint['model'])
```
```

---

## 💻 Running Stage A

### Complete Command

```bash
# Stage A: Thinker Pretraining
python train_text.py --config configs/thinker_tiny.json

# Expected output:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Loading data from data/text/corpus.txt...
Vocab size: 32000
Training samples: 50000

Initializing Thinker (15.2M parameters)...
Using device: cuda:0

Starting training...

Epoch 1/10:
[Step 100/5000] loss=4.234 ppl=68.9 lr=0.00030 | 2.3s/step
[Step 1000/5000] loss=3.156 ppl=23.4 lr=0.00030 | 2.1s/step
→ Validation: loss=3.201 ppl=24.5
✓ Saved checkpoint: thinker_step_1000.pt

...

Epoch 10/10:
[Step 5000/5000] loss=1.987 ppl=7.3 lr=0.00030 | 2.0s/step
→ Final validation: loss=2.012 ppl=7.5
✓ Saved checkpoint: thinker_step_5000.pt

Training complete! Time: 10h 24m
Final validation PPL: 7.5

Ready for Stage E! 🎉
```

---

## 💡 Key Takeaways

✅ **Stage A** trains the Thinker on text-only data  
✅ **Next-token prediction** teaches language understanding  
✅ **Perplexity 5-10** indicates good learning  
✅ **~10 hours** on 12GB GPU for tiny model  
✅ **Foundation** for all multimodal capabilities  
✅ **Cross-entropy loss** measures prediction accuracy  
✅ **Checkpoints** enable resuming training (saved every 1000 steps)

---

## 🎓 Self-Check Questions

1. Why must Stage A (text training) come before multimodal stages?
2. What does perplexity of 10 mean intuitively?
3. What is next-token prediction and why is it effective?
4. How does cross-entropy loss measure model performance?
5. What happens if we skip Stage A and train multimodal directly?

<details>
<summary>📝 Click to see answers</summary>

1. Stage A provides the language foundation. The Thinker must understand text (grammar, syntax, semantics) before it can process multimodal inputs or generate meaningful responses
2. Perplexity of 10 means the model is "confused" between about 10 plausible next words. It's considering a reasonable set of candidates, indicating good understanding
3. Next-token prediction: given context "The cat sat on", predict "the". It's effective because it forces the model to learn grammar, context, and meaning to predict correctly
4. Cross-entropy measures how "surprised" the model is by the correct answer. Lower loss = higher probability assigned to correct word = better prediction
5. Without Stage A, the model has no language foundation. It can't understand text inputs, form coherent sentences, or reason about concepts. Multimodal training would fail completely
</details>

---

[Continue to Chapter 28: Stage B - Audio Encoder →](28-stage-b-audio-encoder.md)

**Chapter Progress:** Training Pipeline ●●○○○○ (2/6 complete)

---
