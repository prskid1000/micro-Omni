# Chapter 23: RVQ Codec for Speech

[Back to Index](00-INDEX.md)

---

## 🎯 Purpose

Quantize mel spectrograms into discrete codes for autoregressive speech generation.

## 🏗️ Architecture

```
Mel Frame (128,)
    ↓
Project to d=64
    ↓
┌─────────────────────┐
│ Codebook 0 (Base)   │
│ 128 codes           │
└─────────┬───────────┘
          ↓
    Quantize → Code 0
    Residual = input - quantized_0
          ↓
┌─────────────────────┐
│ Codebook 1 (Res)    │
│ 128 codes           │
└─────────┬───────────┘
          ↓
    Quantize → Code 1
          ↓
Output: [Code 0, Code 1]
```

## 📊 Specifications

| Parameter | Value |
|-----------|-------|
| **Codebooks** | 2 |
| **Codes per book** | 128 |
| **Codebook dim** | 64 |
| **Total combinations** | 16,384 |
| **Parameters** | ~100K |

## 🔄 Encoding & Decoding

```python
# Encode mel to codes
codes = rvq.encode(mel_frame)  # → [42, 87]

# Decode codes to mel
reconstructed = rvq.decode(codes)  # → (128,)
```

## 💡 Key Takeaways

✅ **2 codebooks** of 128 codes each  
✅ **Residual quantization** for better quality  
✅ **16,384 total combinations**  
✅ **Enables autoregressive** speech generation

---

[Back to Index](00-INDEX.md)

