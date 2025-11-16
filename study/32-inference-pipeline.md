# Chapter 32: Inference Pipeline

[Back to Index](00-INDEX.md)

---

## 🎯 Complete Inference Flow

### Text-Only Chat
```
User Input: "What is AI?"
    ↓
Tokenize → [15, 234, 89, 42]
    ↓
Token Embeddings → (1, 4, 256)
    ↓
Thinker (with KV caching)
    ↓
Next token logits → (1, 4, 5000)
    ↓
Argmax → token_id = 156
    ↓
Decode → "AI is..."
    ↓
Repeat until <EOS>
```

### Multimodal (Image + Text)
```
Image + "Describe this"
    ↓
Vision Encoder → (1, 1, 256)
    ↓
Tokenize text → (1, 3, 256)
    ↓
Concatenate → (1, 4, 256)
    ↓
Thinker → Generate response
```

### Text-to-Speech
```
"Hello world"
    ↓
Tokenize (optional conditioning)
    ↓
Talker → RVQ codes (T, 2)
    ↓
RVQ Decode → Mel (T, 128)
    ↓
Griffin-Lim → Audio waveform
```

## ⚡ Optimizations

1. **KV Caching**: Reuse computed K, V
2. **Mixed Precision**: FP16 for speed
3. **Flash Attention**: 2-4x faster
4. **Batch Processing**: Multiple inputs together

## 💡 Key Takeaways

✅ **KV caching** essential for speed  
✅ **Autoregressive** generation (one token at a time)  
✅ **Multimodal** handled via concatenation  
✅ **Multiple output modes** (text, speech)

---

[Back to Index](00-INDEX.md)

