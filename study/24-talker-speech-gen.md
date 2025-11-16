# Chapter 24: The Talker - Speech Generator

[Back to Index](00-INDEX.md)

---

## 🎯 Purpose

Generate speech by autoregressively predicting RVQ codes.

## 🏗️ Architecture

```
Previous Codes (B, T, 2)
    ↓
Embed Base + Residual Codes
    ↓
Transformer Decoder (4 layers)
  - Causal attention
  - RoPE
  - KV caching
    ↓
Separate Heads:
  - Base Head → (B, T, 128) logits
  - Residual Head → (B, T, 128) logits
    ↓
Predict: [base_code, res_code]
```

## 📊 Specifications

| Parameter | Value |
|-----------|-------|
| **Dimension** | 192 |
| **Layers** | 4 |
| **Heads** | 3 |
| **Codebooks** | 2 |
| **Output** | 2 × 128 logits |
| **Parameters** | ~10-15M |

## 🔄 Generation Process

```
1. Start: codes = [[0, 0]]  (start token)

2. Predict next frame:
   base_logits, res_logits = talker(codes)
   base = argmax(base_logits)  # → 42
   res = argmax(res_logits)    # → 87
   codes = [[0,0], [42,87]]

3. Repeat for T frames...

4. Decode with RVQ:
   mel = rvq.decode(codes)

5. Vocode with Griffin-Lim:
   audio = vocoder.mel_to_audio(mel)
```

## 💡 Key Takeaways

✅ **Autoregressive** code prediction  
✅ **2 separate heads** (base + residual)  
✅ **Uses KV caching** for speed  
✅ **Works with RVQ + Griffin-Lim** vocoder

---

[Back to Index](00-INDEX.md)

