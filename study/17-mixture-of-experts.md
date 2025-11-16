# Chapter 17: Mixture of Experts (MoE)

[← Previous: SwiGLU](16-swiglu-activation.md) | [Back to Index](00-INDEX.md) | [Next: Normalization →](18-normalization.md)

---

## 🎯 Core Idea

Instead of one large feedforward network, use **multiple expert networks** and route each token to a subset.

## 🏗️ Architecture

```
Input token
    ↓
┌─────────────────────┐
│  Router Network     │ → Selects top-k experts
└─────────────────────┘
    ↓
Probabilities: [0.1, 0.05, 0.45, 0.02, 0.38, ...]
Top-2 experts: Expert 2 (0.45), Expert 4 (0.38)
    ↓
┌───────────┐   ┌───────────┐
│ Expert 2  │   │ Expert 4  │
└───────────┘   └───────────┘
    ↓               ↓
output_2 × 0.45 + output_4 × 0.38
    ↓
Combined output
```

## ✅ Benefits

```
Traditional FFN:
All tokens → Same large network
Cost: O(tokens × FFN_size)

MoE:
Each token → Top-k of N experts
Cost: O(tokens × (FFN_size / N) × k)

If N=8, k=2: Cost = 1/4 of traditional!
But total capacity = Same or more
```

## 📊 Trade-offs

| Feature | Dense FFN | MoE |
|---------|-----------|-----|
| **Computation** | High | Low (sparse) |
| **Parameters** | Lower | Higher |
| **Capacity** | Limited | High |
| **Complexity** | Simple | Complex (routing) |

## 💡 Key Takeaways

✅ **MoE** = Multiple expert networks + router  
✅ **Sparse activation** (only k of N experts used)  
✅ **Higher capacity** with same computation  
✅ **μOmni supports MoE** (optional, experimental)

---

[Back to Index](00-INDEX.md)

