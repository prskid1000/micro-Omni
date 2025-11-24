# Chapter 23: RVQ Codec for Speech

[← Previous: Vision Encoder](22-vision-encoder.md) | [Back to Index](00-INDEX.md) | [Next: Talker →](24-talker-speech-gen.md)

---

## 🎯 Learning Objectives

By the end of this chapter, you will understand:

- What RVQ codec does and why it's needed for speech
- How residual vector quantization works in detail
- The role of codebooks in speech generation
- Complete encoding and decoding process
- How it enables autoregressive speech generation
- Connection to the Talker and Griffin-Lim vocoder

---

## 💡 What is the RVQ Codec?

### The Speech Discretizer

**Analogy: Musical Notes vs Sound Waves**

```
Think of speech generation like music:

CONTINUOUS SOUND (Mel Spectrogram):
[0.234, -0.567, 0.891, -0.123, ...]
↓
Like: Actual sound waves (continuous, analog)
- Infinitely precise
- Can be ANY value
- Hard to generate autoregressively
- "What's the next number after 0.234?" → Infinite options!

DISCRETE CODES (RVQ Output):
[42, 87]
↓
Like: Musical notes (discrete, digital)
- Finite set of options
- Can only be specific values
- Easy to generate autoregressively
- "What's the next code after 42?" → Pick from 128 options!

The RVQ Codec is the CONVERTER:
Continuous mel → Discrete codes → Speech generation possible!
```

**Why Do We Need This?**

```
Problem: Can't generate continuous values autoregressively!
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Remember from Chapter 12 (Vector Quantization):

Text generation works because:
- Words are DISCRETE: "cat", "dog", "mat" (finite vocabulary)
- Model predicts: Softmax over 5000 words → Pick one
- This works great! ✓

Speech with continuous mel frames:
- Values are CONTINUOUS: [0.234, -0.567, 0.891, ...]
- Model predicts: ??? Can't use softmax on infinite values!
- This doesn't work! ❌

Solution: RVQ Codec!
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

RVQ Codec transforms:
✅ Continuous mel frame → Discrete codes [42, 87]
✅ Infinite possibilities → 16,384 combinations (128×128)
✅ Regression problem → Classification problem
✅ Now we can use softmax! Same as text generation!

Speech generation becomes:
"What's the next code?" → Softmax over 128 → Pick one
Just like text generation! 🎉
```

---

## 🏗️ Detailed Architecture Breakdown

### The Complete RVQ Pipeline

**Reminder: We already learned Vector Quantization in Chapter 12!**

```
Quick Recap from Chapter 12:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Vector Quantization = "Round to nearest option"
- Codebook = Set of pre-defined vectors
- Quantize = Find nearest codebook entry
- Residual = Error left after quantization

RVQ (Residual VQ) = Multi-stage quantization
- Stage 1: Quantize with codebook 0 (coarse)
- Stage 2: Quantize residual with codebook 1 (refine)
- More codebooks = Better quality!

Now let's see it in action for speech!
```

### Step-by-Step Process

```
INPUT: One frame of mel spectrogram
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Mel frame: (128,)
- 128 mel frequency bins
- Continuous values: [0.5, -0.3, 0.8, 0.2, ...]
- Example: One 10ms slice of "hello" sound

Step 1: Project to Codebook Dimension
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Linear projection: 128 → 64 dimensions

Why 64? Codebook space dimension
- Smaller than 128 (more efficient)
- Large enough to capture features
- Standard for audio codecs

Input: (128,)
Output: (64,) = [0.4, -0.2, 0.7, 0.1, ...]

Step 2: Codebook 0 (Base Quantization)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Codebook 0: 128 learned vectors, each 64-dim
Think: 128 "standard mel patterns"

Example codebook entries:
Code 0: [0.1, 0.2, 0.3, ...] "silence"
Code 42: [0.5, -0.2, 0.8, ...] "vowel 'a'"
Code 87: [-0.3, 0.6, 0.1, ...] "consonant 't'"
...
Code 127: [0.7, 0.1, -0.4, ...] "breath"

Find nearest:
Input: [0.4, -0.2, 0.7, 0.1, ...]

Distances to all 128 codes:
dist_0 = 0.85   (far)
...
dist_42 = 0.12  ← Closest!
...
dist_127 = 0.93 (far)

Best match: Code 42
Quantized_0 = Codebook_0[42] = [0.45, -0.18, 0.72, 0.08, ...]

Compute residual (error):
Residual_1 = Input - Quantized_0
          = [0.4, -0.2, 0.7, 0.1, ...] - [0.45, -0.18, 0.72, 0.08, ...]
          = [-0.05, -0.02, -0.02, 0.02, ...]

This residual captures what Code 42 missed!

Step 3: Codebook 1 (Residual Quantization)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Now quantize the RESIDUAL to capture fine details!

Codebook 1: Another 128 learned vectors, 64-dim
Think: 128 "correction patterns"

Find nearest to residual:
Residual_1: [-0.05, -0.02, -0.02, 0.02, ...]

Distances to all 128 codes:
dist_0 = 0.15   (okay)
...
dist_87 = 0.03  ← Closest!
...
dist_127 = 0.42 (far)

Best match: Code 87
Quantized_1 = Codebook_1[87] = [-0.04, -0.02, -0.03, 0.02, ...]

Final residual:
Residual_2 = Residual_1 - Quantized_1
          = [-0.05, -0.02, -0.02, 0.02, ...] - [-0.04, -0.02, -0.03, 0.02, ...]
          = [-0.01, 0.00, 0.01, 0.00, ...]

Very small! Most error captured!

Step 4: Combine for Reconstruction
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Reconstructed = Quantized_0 + Quantized_1
             = Codebook_0[42] + Codebook_1[87]
             = [0.45, -0.18, 0.72, 0.08, ...] + [-0.04, -0.02, -0.03, 0.02, ...]
             = [0.41, -0.20, 0.69, 0.10, ...]

Original input: [0.4, -0.2, 0.7, 0.1, ...]
Reconstructed:  [0.41, -0.20, 0.69, 0.10, ...]
Error: Very small! ✓

Step 5: Project Back to Mel
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Linear projection: 64 → 128 dimensions

Reconstructed (64,) → Mel frame (128,)

OUTPUT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Discrete codes: [42, 87]
- Code 42 from Codebook 0 (base pattern)
- Code 87 from Codebook 1 (refinement)

These 2 integers represent the mel frame!
Can now be generated like text tokens!
```

### Visual Architecture

```
┌─────────────────────────────────────────┐
│  INPUT: Mel Frame                       │
│  Shape: (128,)                          │
│  Continuous values                      │
│  [0.5, -0.3, 0.8, 0.2, ...]            │
└────────────────┬────────────────────────┘
                 ↓
┌─────────────────────────────────────────┐
│  PROJECT TO CODEBOOK SPACE              │
│  Linear: 128 dim → 64 dim              │
│  Output: (64,)                          │
└────────────────┬────────────────────────┘
                 ↓
┌─────────────────────────────────────────┐
│  CODEBOOK 0 (Base Quantization)         │
│  ┌────────────────────────────────────┐ │
│  │ 128 learned codes, each 64-dim    │ │
│  │ Code 0: [...]                     │ │
│  │ Code 42: [...] ← Nearest!         │ │
│  │ Code 127: [...]                   │ │
│  └────────────────────────────────────┘ │
│  Selected: Code 42                      │
│  Quantized_0 = Codebook_0[42]          │
└────────────────┬────────────────────────┘
                 ↓
┌─────────────────────────────────────────┐
│  COMPUTE RESIDUAL                       │
│  Residual_1 = Input - Quantized_0      │
│  This is the "error" to refine!        │
└────────────────┬────────────────────────┘
                 ↓
┌─────────────────────────────────────────┐
│  CODEBOOK 1 (Residual Quantization)     │
│  ┌────────────────────────────────────┐ │
│  │ 128 learned codes, each 64-dim    │ │
│  │ Code 0: [...]                     │ │
│  │ Code 87: [...] ← Nearest to residual!│ │
│  │ Code 127: [...]                   │ │
│  └────────────────────────────────────┘ │
│  Selected: Code 87                      │
│  Quantized_1 = Codebook_1[87]          │
└────────────────┬────────────────────────┘
                 ↓
┌─────────────────────────────────────────┐
│  RECONSTRUCTION                         │
│  Reconstructed = Quantized_0 + Quantized_1│
│  Shape: (64,)                           │
└────────────────┬────────────────────────┘
                 ↓
┌─────────────────────────────────────────┐
│  PROJECT BACK TO MEL                    │
│  Linear: 64 dim → 128 dim              │
│  Output: (128,)                         │
│  Reconstructed mel frame!               │
└────────────────┬────────────────────────┘
                 ↓
┌─────────────────────────────────────────┐
│  OUTPUT: Discrete Codes                 │
│  [42, 87]                               │
│  2 integers represent the mel frame!    │
│  Ready for autoregressive generation! ✓ │
└─────────────────────────────────────────┘
```

---

## 🔄 Encoding and Decoding

### The Two-Way Process

**ENCODING: Mel → Codes**

```python
def encode(mel_frame):
    """
    mel_frame: (128,) continuous mel spectrogram frame
    Returns: [base_code, residual_code] - 2 integers
    """
    # Step 1: Project to codebook space
    x = proj_in(mel_frame)  # (128,) → (64,)

    # Step 2: Quantize with codebook 0
    distances_0 = compute_distances(x, codebook_0)  # (128,)
    base_code = argmin(distances_0)  # Find nearest
    quantized_0 = codebook_0[base_code]  # (64,)

    # Step 3: Compute residual
    residual = x - quantized_0  # (64,)

    # Step 4: Quantize residual with codebook 1
    distances_1 = compute_distances(residual, codebook_1)
    residual_code = argmin(distances_1)

    # Output: 2 discrete codes
    return [base_code, residual_code]

# Example:
mel = [0.5, -0.3, 0.8, ..., 0.2]  # (128,)
codes = encode(mel)
print(codes)  # [42, 87]
```

**DECODING: Codes → Mel**

```python
def decode(codes):
    """
    codes: [base_code, residual_code] - 2 integers
    Returns: (128,) reconstructed mel frame
    """
    base_code, residual_code = codes

    # Step 1: Look up in codebooks
    quantized_0 = codebook_0[base_code]      # (64,)
    quantized_1 = codebook_1[residual_code]  # (64,)

    # Step 2: Sum quantized vectors
    reconstructed = quantized_0 + quantized_1  # (64,)

    # Step 3: Project back to mel space
    mel_frame = proj_out(reconstructed)  # (64,) → (128,)

    return mel_frame

# Example:
codes = [42, 87]
mel_reconstructed = decode(codes)
print(mel_reconstructed.shape)  # (128,)
```

**Complete Round Trip:**

```
Original mel:     [0.5, -0.3, 0.8, ..., 0.2]
      ↓ encode
Codes:            [42, 87]
      ↓ decode
Reconstructed mel: [0.51, -0.29, 0.79, ..., 0.21]

Reconstruction error: < 0.05 (very good!)
```

---

## 📊 Detailed Specifications

### Codebook Configuration

```
ARCHITECTURE PARAMETERS:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Input dimension: 128 (mel bins)
Codebook dimension: 64
Number of codebooks: 2
Codes per codebook: 128

Projections:
- proj_in: Linear(128 → 64, bias=False)
- proj_out: Linear(64 → 128, bias=False)

Codebooks (learned):
- codebook_0: Embedding(128, 64) ← 128 base patterns
- codebook_1: Embedding(128, 64) ← 128 refinement patterns

TOTAL COMBINATIONS:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Base codes: 128 options
Residual codes: 128 options
Total: 128 × 128 = 16,384 unique combinations!

This is like having a vocabulary of 16,384 "audio words"!

PARAMETERS:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

proj_in: 128 × 64 = 8,192
proj_out: 64 × 128 = 8,192
codebook_0: 128 × 64 = 8,192
codebook_1: 128 × 64 = 8,192
────────────────────────────
Total: ~33K parameters

Tiny! Yet represents 16K unique patterns!
```

### Comparison: Single vs RVQ

```
SINGLE CODEBOOK (what if we only used 1?):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

To get 16,384 combinations with single codebook:
- Need 16,384 codes
- Parameters: 16,384 × 64 = 1,048,576
- Huge! Won't fit in memory efficiently

Quality:
- One-shot quantization
- Average error: ~0.15

RVQ (2 CODEBOOKS - what we actually use):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

16,384 combinations with 2×128 codes:
- Need 256 codes total
- Parameters: 256 × 64 = 16,384
- 64x smaller! ✓

Quality:
- Progressive refinement
- Average error: ~0.03
- 5x better quality with 64x fewer parameters!

This is the power of RVQ!
```

---

## 🎯 Enabling Autoregressive Generation

### Why This Matters for the Talker

**The Key Insight:**

```
Text Generation (we know this works):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Vocabulary: 5000 words
Input: "The cat sat on the ___"
Model: Softmax over 5000 words
Output: Probabilities for each word
Pick: "mat" (highest probability)

Easy! Works great! ✓

Speech Generation with RVQ (same idea!):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

"Vocabulary": 128 base codes, 128 residual codes
Input: Previous speech codes [[15,23], [42,87], ...]
Model (Talker): Softmax over 128 base codes
                Softmax over 128 residual codes
Output: Probabilities for next codes
Pick: [56, 91] (highest probabilities)

Same mechanism as text! ✓

Without RVQ (continuous mel):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Input: Previous mel frames [[0.5,-0.3,...], [0.8,0.2,...], ...]
Model: Predict next mel frame [???, ???, ...]
Output: 128 continuous values
Pick: ??? Can't use softmax on continuous values!

Doesn't work! ❌

RVQ makes speech generation possible!
```

**The Talker Pipeline:**

```
COMPLETE SPEECH GENERATION FLOW:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. Start with BOS (beginning of speech)
   codes = [[0, 0]]  # Start token

2. Talker predicts next codes:
   Input: [[0, 0]]
   Output: Logits for base (128,) and residual (128,)
   Softmax → Probabilities
   Sample/Argmax → [42, 87]

3. Append and repeat:
   codes = [[0, 0], [42, 87]]
   Predict next: [56, 91]
   codes = [[0, 0], [42, 87], [56, 91]]
   ...

4. Generate T frames (e.g., 200 for ~16 seconds at 12.5 Hz):
   codes = [[0,0], [42,87], [56,91], ..., [12,34]]
   Shape: (200, 2)

5. Decode all frames with RVQ:
   mel_frames = []
   for code_pair in codes:
       mel = rvq.decode(code_pair)  # [42,87] → (128,)
       mel_frames.append(mel)
   mel_spectrogram = stack(mel_frames)  # (200, 128)

6. Vocoder converts mel to audio:
   # Uses HiFi-GAN if available, falls back to Griffin-Lim
   audio_waveform = vocoder.mel_to_audio(mel_spectrogram)

7. Save audio file:
   save_wav("output.wav", audio_waveform)

Speech generated! 🎉
```

---

## 🎓 Training the RVQ Codec

### Learning Good Codebooks

**Training Objective:**

```
Goal: Learn codebooks that reconstruct mel frames accurately

Loss Function: Mean Squared Error (MSE)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Input: Original mel frame
Output: Reconstructed mel frame
Loss: MSE(reconstructed, original)

Training forces:
✅ Codebooks to capture common mel patterns
✅ Projections to preserve information
✅ Residual quantization to refine details
```

**Training Loop:**

```python
for batch in dataloader:
    mel_frames = batch  # (B, 128)

    # Forward: Encode then decode
    codes = rvq.encode(mel_frames)           # (B, 2)
    reconstructed = rvq.decode(codes)        # (B, 128)

    # Compute reconstruction loss
    loss = mse_loss(reconstructed, mel_frames)

    # Backprop updates:
    # - proj_in weights
    # - codebook_0 embeddings
    # - codebook_1 embeddings
    # - proj_out weights
    loss.backward()
    optimizer.step()
```

**Straight-Through Estimator (for gradients):**

```
Problem: argmin() has no gradient!
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

code = argmin(distances)  # Discrete operation
↑ Can't backprop through this!

Solution: Straight-through estimator
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Forward pass: Use discrete codes (argmin)
Backward pass: Pretend gradient flows through

gradient ───→ quantized ←─── gradient
               ↑    (as if continuous)
             argmin
               ↑
            continuous

This trick allows training!
```

---

## 🔊 The Vocoder: HiFi-GAN

While RVQ decodes codes back to mel spectrograms, we still need to convert those spectrograms into audio waveforms. This is the job of the **Vocoder**.

μOmni uses **HiFi-GAN**, a state-of-the-art neural vocoder that is both fast and high-quality.

### Architecture

**1. Generator (The Synthesizer)**

- **Input**: Mel spectrogram (128 channels)
- **Upsampling**: 4 stages of upsampling to match the hop length (256x total upsampling)
  - Rates: `[8, 8, 2, 2]`
  - Kernel Sizes: `[16, 16, 4, 4]`
  - Progressively increases time resolution from mel frames to audio samples
- **MRF (Multi-Receptive Field Fusion)**:
  - After each upsampling, the signal passes through multiple parallel residual blocks
  - Each block has different kernel sizes `[3, 5, 7]` and dilation rates `[[1, 2], [1, 2], [1, 2]]`
  - This allows the model to capture patterns at different temporal resolutions simultaneously
- **Output**: Raw audio waveform

**2. Discriminators (The Critics - Training Only)**
To train the generator to produce realistic audio, HiFi-GAN uses two types of discriminators:

- **MPD (Multi-Period Discriminator)**:
  - Reshapes audio into 2D chunks with different periods `[2, 3, 5]` (Codebase default)
  - Crucial for capturing periodic structures in speech (pitch, harmonics)
- **MSD (Multi-Scale Discriminator)**:
  - Analyzes audio at different scales (raw audio, 2x downsampled)
  - Default uses 2 scales to ensure realistic structure at both fine and coarse levels

**Why HiFi-GAN?**

- **Non-Autoregressive**: Generates audio in parallel (very fast inference)
- **High Fidelity**: Produces natural-sounding speech without metallic artifacts common in older vocoders
- **Efficient**: Optimized architecture suitable for real-time generation

## 🔗 Connection to Complete Pipeline

### RVQ in the μOmni Ecosystem

```
WHERE RVQ FITS:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

TEXT-TO-SPEECH PIPELINE:

User types: "Hello world"
      ↓
   Thinker (generates text response)
      ↓
Text: "Hello world, how are you?"
      ↓
   Talker (generates speech codes)
      ↓
Codes: [[42,87], [56,91], [12,34], ...]  ← Discrete!
      ↓
   RVQ Decoder ⭐
      ↓
Mel: (T, 128) ← Continuous!
      ↓
   Vocoder (HiFi-GAN or Griffin-Lim)
      ↓
Audio waveform
      ↓
Save: output.wav

RVQ is the bridge between discrete and continuous!

TRAINING PREPARATION:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. Train RVQ Codec first (Stage D-part1):
   - Learn good codebooks from speech data
   - Minimize reconstruction error

2. Train Talker with frozen RVQ (Stage D-part2):
   - Talker learns to predict codes
   - RVQ provides targets via encoding

3. Inference:
   - Talker generates codes
   - RVQ decodes to mel
   - Vocoder creates audio (HiFi-GAN if trained, else Griffin-Lim)

4. Optional: Train HiFi-GAN vocoder:
   - Run `train_vocoder.py` after Stage D
   - Improves speech quality significantly
   - Automatic fallback to Griffin-Lim if unavailable
```

---

## 💡 Key Takeaways

✅ **RVQ Codec** converts continuous mel to discrete codes  
✅ **2 codebooks** of 128 codes each (progressive refinement)  
✅ **16,384 combinations** (128×128) with only 256 codes  
✅ **Residual quantization** captures fine details  
✅ **Enables autoregressive** speech generation like text  
✅ **Tiny** (~33K parameters) yet effective  
✅ **Two-way**: Encoding (mel→codes) and Decoding (codes→mel)  
✅ **Bridge** between discrete generation and continuous audio

---

## 🎓 Self-Check Questions

1. Why do we need to convert mel frames to discrete codes?
2. How many total combinations can RVQ represent with 2 codebooks of 128 codes each?
3. What is residual quantization and why is it better than single-stage?
4. How does RVQ enable autoregressive speech generation?
5. What are the two directions of RVQ processing?

<details>
<summary>📝 Click to see answers</summary>

1. Because we can't use softmax on continuous values. Discrete codes allow us to use the same autoregressive generation technique as text (softmax + sampling)
2. 128 × 128 = 16,384 unique combinations
3. Residual quantization quantizes the error after first quantization. It's better because it progressively refines the reconstruction - first codebook captures coarse pattern, second captures fine details
4. RVQ converts continuous mel to discrete codes that can be predicted like text tokens. Talker outputs softmax over 128 codes for each codebook, enabling standard autoregressive generation
5. Encoding (mel frame → discrete codes) and Decoding (discrete codes → reconstructed mel frame)
</details>

---

[Continue to Chapter 24: The Talker →](24-talker-speech-gen.md)

**Chapter Progress:** μOmni Components ●●●○○ (3/5 complete)

---

## 📊 Specifications

| Parameter              | Value  |
| ---------------------- | ------ |
| **Codebooks**          | 2      |
| **Codes per book**     | 128    |
| **Codebook dim**       | 64     |
| **Total combinations** | 16,384 |
| **Parameters**         | ~100K  |

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
