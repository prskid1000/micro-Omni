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

### Understanding the Fundamental Problem

Let me start with a simple question: **How do we generate speech?**

**Analogy: Drawing with Different Tools**

```
CONTINUOUS (Pencil - infinite shades):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
You have a pencil that can draw ANY shade of gray:
- 0.0 = Pure white
- 0.234 = Very light gray
- 0.567 = Medium gray
- 1.0 = Pure black

Problem: "Draw the next shade after 0.234"
- Could be 0.235
- Could be 0.236
- Could be 0.2341
- INFINITE possibilities! Very hard to predict!

DISCRETE (Crayon box - limited colors):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
You have 8 crayons with specific colors:
0 = White
1 = Very light gray
2 = Light gray
3 = Medium gray
...
7 = Black

Problem: "Pick the next color after color 2"
- Could be color 0
- Could be color 1
- Could be color 3
- Only 8 possibilities! Easy to predict!

This is EXACTLY the difference between continuous and discrete!
```

### The Challenge in Speech Generation

**Continuous Approach (Hard!):**

```
Continuous (Float):
Audio feature: [0.234, -0.567, 0.891, ...]
Each value can be ANYTHING between -1.0 and 1.0

Generating next frame:
"What's the next number after 0.234?"
Options: 0.235, 0.236, 0.233, 0.2341, 0.23401, ...
→ INFINITE possibilities!

How to predict? Very hard for neural networks!

Think: Imagine predicting the EXACT temperature tomorrow.
      Will it be 72.5°F? 72.51°F? 72.512°F? 72.5123°F?
      Infinitely precise predictions are nearly impossible!
```

**Discrete Approach (Easy!):**

```
Discrete (Integer):
Audio code: [42, 156, 7, ...]
Each value is one of 128 codes (0-127)

Generating next code:
"What's the next code after 42?"
Options: 0, 1, 2, ..., 127
→ Only 128 possibilities!

How to predict? Just like text! Softmax over 128 options!

Think: "What's the weather tomorrow?"
      Options: Sunny, Rainy, Cloudy, Snowy (4 options)
      Easy to predict - just pick one!
```

**Why This Matters:**

```
Text generation (we already know how to do this!):
"The cat sat on the ___"
Options: "mat", "floor", "chair", ... (from vocabulary)
→ Softmax over vocabulary → Pick most likely word
→ THIS WORKS GREAT!

Speech with continuous values:
Previous frame: [0.234, -0.567, 0.891, ...]
Next frame: [???, ???, ???, ...]
→ Can't use softmax (infinite options!)
→ Regression? Very hard to train!

Speech with discrete codes (what μOmni does!):
Previous code: 42
Next code: ???
Options: 0, 1, 2, ..., 127 (from codebook)
→ Softmax over 128 codes → Pick most likely code
→ SAME AS TEXT! We can use the same techniques!

Benefit:
✅ Speech generation becomes like text generation!
✅ Can use transformers, autoregressive modeling
✅ Can use teacher forcing, cross-entropy loss
✅ All the tools that work for text now work for speech!
```

---

## 📚 Vector Quantization Basics

### The Core Idea: Rounding to Nearest Option!

**Analogy: Paint Color Matching**

```
You walk into a paint store with a sample color:
Your sample: RGB(120, 185, 225) - a specific shade of blue

But the store only has these pre-mixed paints:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Paint 0: RGB(100, 180, 220) - Sky Blue
Paint 1: RGB(50, 100, 200) - Ocean Blue
Paint 2: RGB(150, 200, 240) - Light Blue
Paint 3: RGB(80, 140, 180) - Steel Blue
...

Which paint is CLOSEST to your sample?

Paint 0 distance: √[(120-100)² + (185-180)² + (225-220)²] = 21
Paint 1 distance: √[(120-50)² + (185-100)² + (225-200)²] = 110
Paint 2 distance: √[(120-150)² + (185-200)² + (225-240)²] = 38
Paint 3 distance: √[(120-80)² + (185-140)² + (225-180)²] = 74

Paint 0 is closest! → Buy Paint 0

You wanted RGB(120, 185, 225)
You get RGB(100, 180, 220) ← Close enough!

This is EXACTLY what vector quantization does!
```

**Technical Explanation:**

```
Continuous vector → Find nearest discrete code

Think of it as: "Round to the nearest option"

Codebook (learned - like the store's paint selection):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Code 0: [0.1, 0.2, 0.3]     ← Pre-defined option 0
Code 1: [0.5, -0.3, 0.8]    ← Pre-defined option 1
Code 2: [-0.2, 0.7, 0.1]    ← Pre-defined option 2
...
Code 127: [0.3, -0.5, 0.6]  ← Pre-defined option 127

128 total codes in our "paint store"!

Input vector (what we actually want):
[0.12, 0.18, 0.32]  ← Continuous, precise value

Find nearest (which pre-defined code is closest?):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
dist_0 = ||input - code_0||
       = √[(0.12-0.1)² + (0.18-0.2)² + (0.32-0.3)²]
       = √[0.0004 + 0.0004 + 0.0004]
       = 0.03  ← Very close!

dist_1 = ||input - code_1||
       = √[(0.12-0.5)² + (0.18-(-0.3))² + (0.32-0.8)²]
       = √[0.1444 + 0.2304 + 0.2304]
       = 1.42  ← Far!

dist_2 = ||input - code_2||
       = √[(0.12-(-0.2))² + (0.18-0.7)² + (0.32-0.1)²]
       = √[0.1024 + 0.2704 + 0.0484]
       = 0.76  ← Medium distance

Code 0 is closest!

Output: Code ID = 0
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

What we wanted:  [0.12, 0.18, 0.32]  (continuous)
What we output:  0                   (discrete code)
What it represents: [0.1, 0.2, 0.3] (code 0's vector)

Approximation error: 0.03 (pretty good!)
```

**The Magic: Compression!**

```
BEFORE Quantization:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
To store [0.12, 0.18, 0.32]:
- 3 floats × 4 bytes each = 12 bytes

For 100 vectors: 1,200 bytes

AFTER Quantization:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
To store code ID 0:
- 1 integer (0-127 needs 1 byte) = 1 byte

For 100 codes: 100 bytes

Compression: 1,200 → 100 bytes (12x smaller!)

And we can still reconstruct:
Code 0 → Look up in codebook → [0.1, 0.2, 0.3]
(Close to original [0.12, 0.18, 0.32]!)
```

**Why This Works:**

```
Key insight: Most audio features are SIMILAR!

Example: Saying "aaaaa" (long vowel sound)
Frame 1: [0.12, 0.18, 0.32, ...]
Frame 2: [0.13, 0.19, 0.31, ...] ← Very similar!
Frame 3: [0.11, 0.17, 0.33, ...]
...

All these frames map to Code 0!

With 128 codes, we can represent most common audio patterns!
Rare patterns might be slightly less accurate, but that's okay!
```

---

## 🎯 Residual Vector Quantization (RVQ)

### The Problem with Single Codebook

**Analogy: Approximating Your Height**

```
Someone asks: "How tall are you?"

With ONLY 4 options (single codebook):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Option 0: 5 feet
Option 1: 5 feet 6 inches
Option 2: 6 feet
Option 3: 6 feet 6 inches

Your actual height: 5 feet 9 inches

Best fit: Option 2 (6 feet)
Error: +3 inches (too tall!)

With only 4 options, accuracy is limited!
```

### Multi-Stage Quantization (RVQ): The Smart Solution!

**The Key Idea: Fix the Error Progressively**

```
Your height: 5 feet 9 inches

STAGE 1: Rough approximation
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Codebook 1 options (feet):
- 5 feet
- 6 feet  ← Pick this (closest!)

Approximation: 6 feet
Residual (error): 5'9" - 6'0" = -3 inches

STAGE 2: Fix the error!
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Codebook 2 options (inches):
- -6 inches
- -3 inches  ← Pick this (closest to our error!)
- 0 inches
- +3 inches

Correction: -3 inches

FINAL RESULT:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
6 feet + (-3 inches) = 5 feet 9 inches ✓ Perfect!

With 2 stages, we can be much more precise!
```

**Technical Explanation:**

```
Problem with single codebook:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Limited expressiveness (only 128 codes)

Example: Trying to approximate [0.5, 0.8]
Codebook has: [0.5, 0.5], [0.0, 1.0], [1.0, 0.0], ...
Best match: [0.5, 0.5]
Error: [0.0, 0.3] ← Can't capture this detail!

Solution: Residual Vector Quantization (RVQ)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Use multiple codebooks to progressively refine!

Stage 1: Quantize with codebook_0 (coarse approximation)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
input = [0.5, 0.8]
best_code_0 = [0.5, 0.5]  (code ID: 1)
residual_1 = input - best_code_0
          = [0.5, 0.8] - [0.5, 0.5]
          = [0.0, 0.3]  ← What's left to approximate

Stage 2: Quantize residual with codebook_1 (fine details)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
residual_1 = [0.0, 0.3]
best_code_1 = [0.0, 0.3]  (code ID: 7)
residual_2 = [0.0, 0.3] - [0.0, 0.3]
          = [0.0, 0.0]  ← Perfect! No error left!

Final reconstruction:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
quantized = quantized_0 + quantized_1
         = [0.5, 0.5] + [0.0, 0.3]
         = [0.5, 0.8]  ✓ Exactly right!

Output codes: [1, 7]
Two integers represent the original vector!
```

**Benefits:**

```
EXPRESSIVENESS:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Single codebook: 128 options
Two codebooks: 128 × 128 = 16,384 combinations!
Three codebooks: 128³ = 2,097,152 combinations!

Much more expressive without many more parameters!

PROGRESSIVE REFINEMENT:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Stage 1: Coarse approximation (big picture)
Stage 2: Fine details (fix the errors)
Stage 3: Even finer details (if needed)

Like painting:
1. Rough sketch
2. Add details
3. Add fine details

BETTER QUALITY:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Single codebook: Average error = 0.15
RVQ (2 codebooks): Average error = 0.03
→ 5x better reconstruction!

EFFICIENCY:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
To get 16,384 combinations:
- Single codebook: Need 16,384 codes (huge!)
- RVQ: Need 128 + 128 = 256 codes (tiny!)

256 codes vs 16,384 codes → 64x fewer parameters!
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

| Codebooks | Codes/book | Total Combinations | Quality   | Memory |
| --------- | ---------- | ------------------ | --------- | ------ | ----------- |
| **1**     | 128        | 128                | Low       | 8KB    |
| **2**     | 128        | 16,384             | Good      | 16KB   | ← **μOmni** |
| **3**     | 128        | 2,097,152          | High      | 24KB   |
| **4**     | 256        | 4,294,967,296      | Very High | 64KB   |

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

### Vocoder Options

**μOmni supports two vocoders:**

1. **HiFi-GAN (Neural Vocoder)** - Recommended for production

   - ✅ High quality, natural speech
   - ✅ Better prosody
   - ⚠️ Requires training (~2-4 hours)
   - ✅ Automatic fallback to Griffin-Lim if unavailable

2. **Griffin-Lim (Classical Algorithm)** - Default fallback
   - ✅ No training required
   - ✅ Deterministic
   - ✅ Fast inference
   - ❌ Lower quality than neural vocoders
   - ❌ Can sound robotic

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
```

### Training HiFi-GAN

```bash
# Train neural vocoder (optional, improves quality)
python train_vocoder.py --config configs/vocoder_tiny.json
```

**Training Details:**

- Uses adversarial training (Generator vs Discriminators)
- Multi-Period Discriminator (MPD) + Multi-Scale Discriminator (MSD)
- Optimized for 12GB VRAM (batch_size=2, gradient accumulation=4)
- Audio length limit: 8192 samples (~0.5s)
- Mixed precision (FP16) enabled

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
# Automatically uses HiFi-GAN if available, falls back to Griffin-Lim
from omni.codec import NeuralVocoder
vocoder = NeuralVocoder(checkpoint_path="checkpoints/vocoder_tiny/model.pt")
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
✅ **Vocoder** converts mel to audio (HiFi-GAN if trained, else Griffin-Lim)  
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
