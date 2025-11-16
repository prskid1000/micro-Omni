# Chapter 13: Decoder-Only Language Models

[← Previous: Vector Quantization](12-quantization.md) | [Back to Index](00-INDEX.md) | [Next: KV Caching →](14-kv-caching.md)

---

## 🎯 Overview

Decoder-only models (GPT-style) generate text autoregressively using causal attention.

## 🏗️ Architecture

```
Input: "The cat sat"
↓
Causal Self-Attention (can only see previous tokens)
↓  
Feedforward Network
↓
Output: Predict next token → "on"
```

## 🔑 Key Features

### Causal Masking
```
Attention mask (lower triangular):
     The  cat  sat  on
The   ✓    ✗    ✗   ✗
cat   ✓    ✓    ✗   ✗
sat   ✓    ✓    ✓   ✗
on    ✓    ✓    ✓   ✓

Each position can only attend to previous positions
```

### Autoregressive Generation
```
Step 1: Input "The cat" → Predict "sat"
Step 2: Input "The cat sat" → Predict "on"
Step 3: Input "The cat sat on" → Predict "the"
...
```

## 🆚 Encoder vs Decoder

| Feature | Encoder (BERT) | Decoder (GPT) |
|---------|----------------|---------------|
| **Attention** | Bidirectional | Causal |
| **Task** | Understanding | Generation |
| **Training** | Masked LM | Next-token prediction |

## 💡 Key Takeaways

✅ **Causal attention** prevents seeing future tokens  
✅ **Autoregressive** generation one token at a time  
✅ **μOmni's Thinker** is decoder-only (GPT-style)

---

[Back to Index](00-INDEX.md)

