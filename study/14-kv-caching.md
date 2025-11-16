# Chapter 14: KV Caching Optimization

[← Previous: Decoder-Only LLM](13-decoder-only-llm.md) | [Back to Index](00-INDEX.md) | [Next: GQA →](15-gqa-attention.md)

---

## 🎯 The Problem

```
Without caching:
Step 1: Process "The" → Compute K, V for "The"
Step 2: Process "The cat" → Recompute K, V for "The" + compute for "cat"
Step 3: Process "The cat sat" → Recompute all K, V again!

Complexity: O(T²) - very slow!
```

## ✅ The Solution: KV Caching

```
With caching:
Step 1: Process "The" → Compute & cache K, V for "The"
Step 2: Process "cat" → Reuse cached "The", compute & cache "cat"
Step 3: Process "sat" → Reuse cached "The cat", compute & cache "sat"

Complexity: O(T) - much faster!
```

## 💻 Implementation

```python
class AttentionWithCache:
    def forward(self, x, cache=None):
        Q, K, V = self.project(x)
        
        if cache is not None:
            # Append new K, V to cached
            K = torch.cat([cache['K'], K], dim=2)
            V = torch.cat([cache['V'], V], dim=2)
        
        # Compute attention
        output = attention(Q, K, V)
        
        # Return output and updated cache
        return output, {'K': K, 'V': V}
```

## 📊 Speed-up

```
Generation without caching:
100 tokens: ~5.0 seconds

Generation with caching:
100 tokens: ~0.5 seconds

10x faster! ✓
```

## 💡 Key Takeaways

✅ **KV caching** stores computed keys/values  
✅ **Speeds up** generation from O(T²) to O(T)  
✅ **Essential** for interactive applications  
✅ **μOmni uses KV caching** in Thinker and Talker

---

[Back to Index](00-INDEX.md)

