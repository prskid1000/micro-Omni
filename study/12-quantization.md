# Chapter 12: Vector Quantization

[← Previous: Image Processing](11-image-processing.md) | [Back to Index](00-INDEX.md) | [Next: Decoder-Only LLMs →](13-decoder-only-llm.md)

---

## 🎯 What You'll Learn

- What vector quantization is
- Codebooks and discrete representations
- Residual Vector Quantization (RVQ)
- How μOmni uses RVQ for speech

---

## 🔢 Continuous vs Discrete Representations

### The Challenge

```
Continuous (Float):
Audio feature: [0.234, -0.567, 0.891, ...]
Problem: Infinite possibilities, hard to model with autoregressive

Discrete (Integer):
Audio code: [42, 156, 7, ...]
Benefit: Finite vocabulary, easy to predict like text tokens!
```

---

## 📚 Vector Quantization Basics

### Core Idea

```
Continuous vector → Find nearest discrete code

Codebook (learned):
Code 0: [0.1, 0.2, 0.3]
Code 1: [0.5, -0.3, 0.8]
Code 2: [-0.2, 0.7, 0.1]
...
Code 127: [0.3, -0.5, 0.6]

Input vector: [0.12, 0.18, 0.32]

Find nearest:
dist_0 = ||input - code_0|| = 0.03  ← Closest!
dist_1 = ||input - code_1|| = 1.42
dist_2 = ||input - code_2|| = 0.76

Output: Code ID = 0
```

---

## 🎯 Residual Vector Quantization (RVQ)

### Multi-Stage Quantization

```
Problem with single codebook:
Limited expressiveness (only 128-512 codes)

Solution: Multiple codebooks (residual quantization)

Stage 1: Quantize with codebook_0
         residual_1 = input - quantized_0

Stage 2: Quantize residual with codebook_1
         residual_2 = residual_1 - quantized_1

Final: quantized = quantized_0 + quantized_1

Benefits:
✅ More expressive (128×128 = 16,384 combinations)
✅ Progressive refinement
✅ Better reconstruction quality
```

---

### RVQ Example

```
Input vector: [0.5, 0.8]

Codebook 0 (base):
Code 0: [0.0, 0.0]
Code 1: [0.5, 0.5]  ← Closest
Code 2: [1.0, 1.0]

Stage 1:
Quantized: [0.5, 0.5] (code_1)
Residual: [0.5, 0.8] - [0.5, 0.5] = [0.0, 0.3]

Codebook 1 (residual):
Code 0: [0.0, 0.0]
Code 1: [0.0, 0.3]  ← Closest!
Code 2: [0.5, 0.5]

Stage 2:
Quantized: [0.0, 0.3] (code_1)
Residual: [0.0, 0.3] - [0.0, 0.3] = [0.0, 0.0]

Final reconstruction:
[0.5, 0.5] + [0.0, 0.3] = [0.5, 0.8]  ✓ Perfect!

Output codes: [1, 1] (base_code=1, residual_code=1)
```

---

## 💻 RVQ Implementation

### μOmni's RVQ Codec

```python
# From omni/codec.py (simplified)
class RVQ(nn.Module):
    def __init__(self, num_codebooks=2, codebook_size=128, d=64):
        super().__init__()
        self.num_codebooks = num_codebooks
        self.codebook_size = codebook_size
        
        # Input/output projections
        self.proj_in = nn.Linear(128, d)   # Mel bins → codebook dim
        self.proj_out = nn.Linear(d, 128)  # Codebook dim → Mel bins
        
        # Codebooks (learnable embeddings)
        self.codebooks = nn.ModuleList([
            nn.Embedding(codebook_size, d)
            for _ in range(num_codebooks)
        ])
    
    def encode(self, x):
        """
        x: (B, 128) mel frame
        Returns: (B, num_codebooks) discrete codes
        """
        x = self.proj_in(x)  # (B, d)
        codes = []
        residual = x
        
        for codebook in self.codebooks:
            # Find nearest code
            distances = torch.cdist(residual, codebook.weight)  # (B, codebook_size)
            indices = torch.argmin(distances, dim=-1)  # (B,)
            codes.append(indices)
            
            # Compute residual
            quantized = codebook(indices)
            residual = residual - quantized
        
        return torch.stack(codes, dim=-1)  # (B, num_codebooks)
    
    def decode(self, codes):
        """
        codes: (B, num_codebooks) discrete codes
        Returns: (B, 128) reconstructed mel frame
        """
        B = codes.shape[0]
        quantized = torch.zeros(B, self.codebooks[0].weight.shape[1], 
                               device=codes.device)
        
        # Sum quantized vectors from all codebooks
        for i, codebook in enumerate(self.codebooks):
            quantized += codebook(codes[:, i])
        
        # Project back to mel
        return self.proj_out(quantized)  # (B, 128)
```

---

## 🎤 RVQ for Speech

### μOmni's Speech Representation

```
Mel Spectrogram frame: (128,) continuous floats
         ↓ RVQ encode
Discrete codes: [base_code, residual_code] = [42, 87]
         ↓ RVQ decode
Reconstructed mel: (128,) continuous floats

Parameters:
- 2 codebooks
- 128 codes per codebook
- 64-dimensional codebook space
- Total combinations: 128 × 128 = 16,384
```

---

### Complete Speech Pipeline

```
TEXT-TO-SPEECH:

Text: "Hello"
  ↓ Tokenizer
Token IDs: [15, 234, 89]
  ↓ Thinker (optional conditioning)
Text embeddings
  ↓ Talker (autoregressive)
RVQ codes: [[42,87], [103,12], [67,91], ...]
         (T_frames, 2)
  ↓ RVQ Decode
Mel spectrogram: (T_frames, 128)
  ↓ Griffin-Lim Vocoder
Audio waveform: (T_samples,)
```

---

## 📊 Codebook Statistics

### Trade-offs

| Codebooks | Codes/book | Total Combinations | Quality | Memory |
|-----------|------------|-------------------|---------|--------|
| **1** | 128 | 128 | Low | 8KB |
| **2** | 128 | 16,384 | Good | 16KB | ← **μOmni**
| **3** | 128 | 2,097,152 | High | 24KB |
| **4** | 256 | 4,294,967,296 | Very High | 64KB |

```
μOmni uses 2 codebooks of 128 codes:
- Good quality/efficiency trade-off
- Fast encoding/decoding
- Fits in 12GB GPU easily
```

---

## 🎯 Training RVQ

### End-to-End Training

```python
# Training loop (simplified)
for mel_frames in dataloader:
    # Encode to discrete codes
    codes = rvq.encode(mel_frames)      # (B, T, 2)
    
    # Decode back to mel
    reconstructed = rvq.decode(codes)   # (B, T, 128)
    
    # Reconstruction loss
    loss = F.mse_loss(reconstructed, mel_frames)
    
    # Backpropagation updates:
    # - proj_in weights
    # - codebook embeddings
    # - proj_out weights
    loss.backward()
    optimizer.step()
```

---

### Straight-Through Estimator

```
Problem: Quantization is not differentiable!
  argmin() has no gradient

Solution: Straight-through estimator
  Forward: Use discrete codes (argmin)
  Backward: Pass gradients as if continuous

gradient ───→ quantized ←─── gradient
               ↑    (pretend it's continuous)
             argmin
               ↑
             continuous
```

---

## 🔊 Vocoder: Mel to Waveform

### Griffin-Lim Algorithm

```
Mel Spectrogram → Waveform
(no neural network, classical algorithm)

Steps:
1. Mel → Linear spectrogram (inverse mel filterbank)
2. Initialize random phase
3. Iteratively refine:
   - ISTFT (get time-domain signal)
   - STFT (get frequency-domain)
   - Keep magnitude, update phase
   - Repeat 32 times
4. Final ISTFT → audio waveform

Benefits:
✅ No training required
✅ Deterministic
✅ Fast inference

Drawbacks:
❌ Lower quality than neural vocoders (HiFi-GAN, etc.)
❌ Can sound robotic

μOmni uses Griffin-Lim for simplicity!
```

---

## 💻 Complete Code Example

```python
# μOmni TTS generation example

# 1. Generate RVQ codes with Talker
talker.eval()
rvq.eval()

codes = torch.zeros(1, 1, 2, dtype=torch.long)  # Start token

for _ in range(200):  # Generate 200 frames (~3.2 seconds at 12.5Hz)
    base_logit, res_logit = talker(codes)
    base_code = torch.argmax(base_logit[0, -1])
    res_code = torch.argmax(res_logit[0, -1])
    next_codes = torch.tensor([[[base_code, res_code]]])
    codes = torch.cat([codes, next_codes], dim=1)

# 2. Decode RVQ codes to mel
mel_frames = []
for t in range(codes.shape[1]):
    frame_codes = codes[0, t, :]  # (2,)
    mel_frame = rvq.decode(frame_codes.unsqueeze(0))  # (1, 128)
    mel_frames.append(mel_frame)

mel = torch.stack(mel_frames, dim=0)  # (T, 128)

# 3. Vocoder: Mel to audio
vocoder = GriffinLimVocoder()
audio = vocoder.mel_to_audio(mel.numpy())

# 4. Save
import soundfile as sf
sf.write("output.wav", audio, 16000)
```

---

## 💡 Key Takeaways

✅ **Vector Quantization** converts continuous vectors to discrete codes  
✅ **Codebooks** are learned discrete vocabularies  
✅ **RVQ** uses multiple codebooks for progressive refinement  
✅ **μOmni uses 2 codebooks** of 128 codes each (16,384 combinations)  
✅ **Straight-through estimator** enables gradient flow  
✅ **Griffin-Lim vocoder** converts mel to audio (no training)  
✅ **Enables autoregressive speech generation** like text!

---

## 🎓 Self-Check Questions

1. What is vector quantization?
2. Why use multiple codebooks (RVQ) instead of one?
3. How many total code combinations does μOmni's RVQ have?
4. What is the straight-through estimator?
5. What does Griffin-Lim do?

<details>
<summary>📝 Answers</summary>

1. Converting continuous vectors to discrete codes by finding nearest codebook entry
2. Multiple codebooks provide more expressiveness through residual quantization (better reconstruction)
3. 128 × 128 = 16,384 combinations (2 codebooks, 128 codes each)
4. Gradient estimation trick: discrete forward pass, continuous backward pass
5. Griffin-Lim converts mel spectrogram to audio waveform (iterative phase reconstruction)
</details>

---

[Continue to Chapter 13: Decoder-Only Language Models →](13-decoder-only-llm.md)

**Chapter Progress:** Core Concepts ●●●●●●● (7/7 complete)  
**Next Section:** Advanced Architecture →

