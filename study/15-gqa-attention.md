# Chapter 15: Grouped Query Attention (GQA)

[← Previous: KV Caching](14-kv-caching.md) | [Back to Index](00-INDEX.md) | [Next: SwiGLU →](16-swiglu-activation.md)

---

## 🎯 Motivation

**Problem**: Multi-head attention requires many parameters for K, V projections.

**Solution**: Share K, V heads across multiple query heads!

## 🏗️ Architecture Comparison

### Standard Multi-Head Attention (MHA)
```
Q heads: 8 × 64 = 512 dims
K heads: 8 × 64 = 512 dims  ← Full K, V for each head
V heads: 8 × 64 = 512 dims

Total: 1536 dims
```

### Grouped Query Attention (GQA)
```
Q heads: 8 × 64 = 512 dims
K heads: 2 × 64 = 128 dims  ← Shared across Q heads!
V heads: 2 × 64 = 128 dims

Grouping:
Q_head_0, Q_head_1, Q_head_2, Q_head_3 → share K_head_0, V_head_0
Q_head_4, Q_head_5, Q_head_6, Q_head_7 → share K_head_1, V_head_1

Total: 768 dims (50% reduction!)
```

## ✅ Benefits

| Feature | MHA | GQA |
|---------|-----|-----|
| **Parameters** | High | Lower (2-4x reduction) |
| **KV Cache Size** | Large | Smaller |
| **Inference Speed** | Baseline | Faster (less memory traffic) |
| **Quality** | Best | Near-identical |

## 💡 Key Takeaways

✅ **GQA** shares K, V heads across Q heads  
✅ **Reduces parameters** by 2-4x  
✅ **Faster inference** (smaller KV cache)  
✅ **μOmni supports GQA** (optional)

---

[Back to Index](00-INDEX.md)

