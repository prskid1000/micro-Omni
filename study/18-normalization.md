# Chapter 18: Normalization Techniques

[← Previous: MoE](17-mixture-of-experts.md) | [Back to Index](00-INDEX.md) | [Next: μOmni Overview →](19-muomni-overview.md)

---

## 🎯 Why Normalize?

**Problem**: Activations can have very different scales  
**Solution**: Normalize to stabilize training

## 📊 Common Normalization Methods

### 1. LayerNorm (Original Transformer)
```python
mean = x.mean(dim=-1, keepdim=True)
std = x.std(dim=-1, keepdim=True)
normalized = (x - mean) / (std + eps)
output = normalized * gamma + beta
```

### 2. RMSNorm (Modern, Faster) ⭐
```python
rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True))
normalized = x / (rms + eps)
output = normalized * gamma
```

**Differences**:
- RMSNorm: No mean subtraction, no bias
- ~15% faster than LayerNorm
- Similar performance

## 💻 Implementation

```python
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x):
        rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x / rms) * self.weight
```

## 🎯 Pre-norm vs Post-norm

### Post-norm (Original)
```
x → Attention → Add & Norm → FFN → Add & Norm
```

### Pre-norm (Modern)
```
x → Norm → Attention → Add → Norm → FFN → Add
```

**Pre-norm** is more stable for deep networks!

## 💡 Key Takeaways

✅ **Normalization** stabilizes training  
✅ **RMSNorm** is faster than LayerNorm  
✅ **Pre-norm** is more stable than post-norm  
✅ **μOmni uses RMSNorm** throughout

---

[Back to Index](00-INDEX.md)

