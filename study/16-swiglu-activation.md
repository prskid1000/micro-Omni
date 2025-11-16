# Chapter 16: SwiGLU Activation Function

[← Previous: GQA](15-gqa-attention.md) | [Back to Index](00-INDEX.md) | [Next: MoE →](17-mixture-of-experts.md)

---

## 🎯 What is SwiGLU?

**SwiGLU** = Swish-Gated Linear Unit  
A modern activation function used in feedforward layers.

## 📊 Comparison with Other Activations

### Traditional: ReLU
```python
FFN(x) = W2 · ReLU(W1 · x)
       = W2 · max(0, W1 · x)
```

### Modern: GELU
```python
FFN(x) = W2 · GELU(W1 · x)
```

### SwiGLU (Best!)
```python
FFN(x) = W_down · (Swish(W_gate · x) ⊙ W_up · x)

Where:
- Swish(x) = x · sigmoid(x)
- ⊙ = element-wise multiplication
- Uses 3 projections (gate, up, down)
```

## 🎨 Visualization

```
Input x
   ↓
┌─────────────────┐
│   W_gate · x    │ → Apply Swish → gate_activated
└─────────────────┘
         ↓
┌─────────────────┐
│   W_up · x      │ → up
└─────────────────┘
         ↓
  gate_activated ⊙ up (element-wise multiply)
         ↓
┌─────────────────┐
│  W_down · ...   │
└─────────────────┘
         ↓
     Output
```

## ✅ Benefits

- ✅ Better gradient flow than ReLU
- ✅ Smoother than GELU
- ✅ Empirically better performance
- ✅ Used in modern LLMs (LLaMA, Qwen, etc.)

## 💡 Key Takeaways

✅ **SwiGLU** = Gated activation with Swish  
✅ **Better performance** than ReLU/GELU  
✅ **3 projections** (gate, up, down)  
✅ **μOmni uses SwiGLU** (optional)

---

[Back to Index](00-INDEX.md)

