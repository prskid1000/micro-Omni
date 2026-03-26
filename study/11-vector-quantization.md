[← Previous: 10-mixture-of-experts](10-mixture-of-experts.md) | [Index](00-INDEX.md) | [Next: 12-system-overview →](12-system-overview.md)

# Chapter 11: Vector Quantization & Speech Codes

---

## Learning Objectives

By the end of this chapter, you will understand:
- Why converting continuous audio to discrete codes is the key to speech generation
- How vector quantization works and why a single codebook is not enough
- How Residual VQ (RVQ) stacks codebooks for high-fidelity compression
- How vocoders convert mel spectrograms back to audible waveforms
- micro-Omni's complete audio codec pipeline

---

## The Key Insight: Make Speech Look Like Text

Text generation is a solved problem (Chapters 03, 08): predict the next token from a vocabulary, append it, repeat. The transformer is excellent at this.

But speech is continuous -- a stream of floating-point amplitude values at 16,000 samples per second. You cannot predict "the next float" from a vocabulary. There is no vocabulary for continuous signals.

The breakthrough: convert continuous audio into discrete codes (integers from a finite set), then generate those codes exactly like text tokens. A speech generation model becomes just another language model, predicting the next code instead of the next word.

```
TEXT GENERATION:            SPEECH GENERATION (with VQ):

Vocabulary: [hello, world,  Vocabulary: [code_0, code_1, ..., code_127]
             the, cat, ...]
                            Audio frame -> Quantize -> code_42
Predict next word token     Predict next speech code
Append to sequence          Append to sequence
Repeat                      Repeat
                            Decode codes back to audio waveform
```

---

## Vector Quantization: Finding the Nearest Code

Vector quantization (VQ) maps a continuous vector to the nearest entry in a learned codebook. Think of it like choosing paint colors.

### The Paint Store Analogy

You walk into a paint store wanting to paint your wall the exact shade of blue you see in a sunset. But the store only carries 128 pre-mixed colors. You hold your ideal color swatch next to each paint chip and pick the closest match. You lose some precision (it is not exactly your shade), but now your color choice can be described as a single number: "paint #47."

This is vector quantization:
- Your ideal color = the continuous input vector
- The 128 paint chips = the codebook entries
- "Paint #47" = the quantized index
- The difference between your ideal and #47 = the quantization error

### How It Works

```
Input: continuous vector z (e.g., a mel spectrogram frame projected to 64 dimensions)

Codebook: 128 learned vectors, each 64-dimensional
  c_0 = [0.12, -0.34, 0.56, ...]
  c_1 = [0.78, 0.23, -0.11, ...]
  ...
  c_127 = [-0.45, 0.67, 0.33, ...]

Step 1: Compute distance from z to every codebook entry
  dist(z, c_i) = ||z - c_i||^2    for i = 0, 1, ..., 127

Step 2: Pick the nearest one
  index = argmin(dist)             e.g., index = 47

Step 3: The quantized vector is c_47
  quantized = codebook[47]

  z  -------> [distance computation] -------> index 47 -------> c_47
  (64-dim)    (compare to 128 entries)        (integer)         (64-dim)
```

micro-Omni computes distances efficiently using `torch.cdist`:

```python
dist = torch.cdist(residual.unsqueeze(1), cb.weight.unsqueeze(0)).squeeze(1)  # (B, 128)
ind = dist.argmin(dim=-1)  # (B,)
```

---

## The Problem with Single VQ: Only 128 Sounds

With one codebook of 128 entries, you can represent exactly 128 distinct sounds. That is nowhere near enough for intelligible speech. Human speech has thousands of distinct phonetic nuances -- pitch, vowel quality, consonant articulation, speaker identity, emotion.

Increasing the codebook to 16,384 entries seems like it would solve this, but it creates new problems:
- Distance computation becomes expensive (compare against 16,384 vectors per frame)
- Many codebook entries go unused (codebook collapse)
- Training becomes unstable

---

## Residual Vector Quantization (RVQ): Stacking Codebooks

RVQ is an elegant solution: use multiple small codebooks in sequence, where each one quantizes the error (residual) left by the previous one.

### The Address Analogy

Think of describing a location with increasing precision:
1. **Country**: "United States" (narrows to ~9.8 million km^2)
2. **State**: "California" (narrows to ~424,000 km^2)
3. **City**: "San Francisco" (narrows to ~121 km^2)
4. **Street**: "Market Street" (narrows to ~0.1 km^2)

Each level does not describe the full location -- it describes the *remaining error* after the previous level. "California" does not mean much on its own, but "United States, California" is quite specific. Adding more levels adds more precision.

RVQ works the same way:
1. **Codebook 1**: Find the nearest code to the input. This captures the coarse structure.
2. **Residual**: Compute the error -- what Codebook 1 missed.
3. **Codebook 2**: Find the nearest code to the residual. This captures fine detail.
4. **Combined**: The full representation is (code1, code2).

### RVQ Step by Step

```
Input mel frame z (projected to 64 dims):

CODEBOOK 1 (coarse):
  z = [0.82, -0.15, 0.43, ...]

  Nearest entry: c1_47 = [0.80, -0.20, 0.40, ...]
  Index: 47

  Residual r1 = z - c1_47 = [0.02, 0.05, 0.03, ...]
  (what codebook 1 could not capture)

CODEBOOK 2 (fine detail):
  Nearest entry to r1: c2_93 = [0.02, 0.04, 0.03, ...]
  Index: 93

  Residual r2 = r1 - c2_93 = [0.00, 0.01, 0.00, ...]
  (very small -- most information captured)

OUTPUT: (47, 93)   <-- two integers fully describe this audio frame
```

### Combinatorial Expressiveness

With 2 codebooks of 128 entries each:
- Codebook 1 alone: 128 possible sounds
- Codebook 1 + Codebook 2: 128 x 128 = 16,384 unique combinations

You get 16,384 distinct sounds using only 256 codebook entries (128 + 128). This is far more memory-efficient than a single codebook of 16,384 entries.

---

## micro-Omni's RVQ Codec

micro-Omni uses 2 codebooks with 128 entries each, operating in a 64-dimensional embedding space.

### Encoding: Mel Frame to Codes

```
Mel frame (B, 128)              128-dim mel spectrogram values
       |
  proj_in: Linear(128, 64)      Project to 64-dim codebook space
       |
       v
  z: (B, 64)                    Continuous representation
       |
  +----v-----------+
  | Codebook 1     |  torch.cdist(z, codebook1)  -> find nearest
  | index: idx1    |  quantized1 = codebook1[idx1]
  | residual = z - quantized1
  +----+-----------+
       |
  +----v-----------+
  | Codebook 2     |  torch.cdist(residual, codebook2)  -> find nearest
  | index: idx2    |  quantized2 = codebook2[idx2]
  +----+-----------+
       |
       v
  Output: (idx1, idx2)          Two integers per frame
          shape: (B, 2)
```

### Decoding: Codes to Mel Frame

```
Input: (idx1, idx2)             Two codebook indices
       |
  +----v-----------+
  | Look up both   |  emb1 = codebook1[idx1]   (B, 64)
  | codebook entries|  emb2 = codebook2[idx2]   (B, 64)
  +----+-----------+
       |
       v
  z_hat = emb1 + emb2           Sum embeddings: (B, 64)
       |
  proj_out: Linear(64, 128)     Project back to mel dimension
       |
       v
  mel_hat: (B, 128)             Reconstructed mel frame
```

### Batched Processing

micro-Omni handles both single frames `(B, 128)` and sequences `(B, T, 128)`. For sequences, it reshapes to `(B*T, 128)`, processes all frames at once, then reshapes back to `(B, T, 128)`. This is much faster than looping over frames.

---

## Training VQ: The Straight-Through Estimator

There is a fundamental problem with training VQ: the `argmin` operation (finding the nearest codebook entry) is not differentiable. You cannot compute gradients through "pick the closest one."

### The Problem

```
z  --->  argmin(dist(z, codebook))  --->  idx  --->  codebook[idx]  --->  loss
                    ^
                    |
         This has zero gradient everywhere
         (output is a discrete integer)
```

### The Solution: Straight-Through Estimator (STE)

The trick: during the forward pass, use the quantized value. During the backward pass, pretend the quantization did not happen and pass gradients straight through.

```
Forward:  quantized = codebook[argmin(dist)]     (discrete, correct)
Backward: grad flows as if quantized = z         (continuous, approximate)
```

In code, this is implemented with the detach trick:

```python
quantized = codebook[idx]
# Straight-through: gradient flows through as if quantized = z
quantized_st = z + (quantized - z).detach()
# Forward: quantized_st == quantized (because detach stops grad)
# Backward: grad(quantized_st) == grad(z) (because (quantized-z).detach() has no grad)
```

The codebook entries themselves learn through a separate commitment loss that pulls them toward the encoder's output, and a codebook loss that pulls the encoder's output toward the codebook entries.

---

## Vocoders: From Mel Spectrograms to Sound Waves

The RVQ codec converts between mel spectrograms and discrete codes. But humans hear waveforms, not mel spectrograms. A **vocoder** bridges this final gap: mel spectrogram to audible audio.

### Why Do We Need a Vocoder?

A mel spectrogram contains magnitude information but throws away phase information (Chapter 05). Reconstructing audio requires recovering or synthesizing plausible phase.

Think of it like a blueprint of a building. The blueprint shows where walls go (magnitude) but does not specify the exact brand of every brick (phase). You need a builder (vocoder) to turn the blueprint into an actual structure, making reasonable choices about the details.

### Griffin-Lim: The Classical Approach

Griffin-Lim is an iterative algorithm that reconstructs phase by repeatedly applying the STFT and its inverse:

```
1. Start with mel magnitude spectrogram (no phase)
2. Invert mel filterbank to get linear magnitude spectrogram
3. Assign random phase
4. Repeat N times:
   a. Inverse STFT -> waveform
   b. Forward STFT -> new magnitude + new phase
   c. Replace magnitude with original (keep new phase)
5. Final inverse STFT -> output waveform
```

**Pros**: No training needed, no neural network, deterministic.
**Cons**: Buzzy quality, slow (32+ iterations), phase is only approximate.

micro-Omni implements Griffin-Lim as a fallback using librosa's efficient implementation.

### HiFi-GAN: The Neural Approach

HiFi-GAN is a generative adversarial network (GAN) trained specifically for vocoding. It learns to generate realistic waveforms from mel spectrograms.

```
GENERATOR (what we use at inference):

  Mel spectrogram (B, 128, T)
         |
    Conv1d (initial)
         |
    +----v----+----+----+
    | Upsample 8x  |    |    Transposed convolutions increase
    | Upsample 8x  |    |    temporal resolution to match
    | Upsample 2x  |    |    audio sample rate
    | Upsample 2x  |    |    (total: 8*8*2*2 = 256x)
    +----+----+----+----+
         |
    Multi-Receptive Field Fusion (MRF)
    [ResBlocks with dilations 1,2 at kernels 3,5,7]
    (captures patterns at multiple time scales)
         |
    Conv1d (final) -> Tanh
         |
    Waveform (B, T * 256)


DISCRIMINATORS (training only):

  Multi-Period Discriminator (MPD):
    Reshapes audio into 2D with periods [2, 3, 5]
    Captures periodic patterns (pitch, harmonics)

  Multi-Scale Discriminator (MSD):
    Processes audio at original and downsampled scales
    Captures both fine and coarse temporal patterns
```

The generator learns to produce audio so realistic that the discriminators cannot tell it apart from real recordings. This adversarial training produces much higher quality than Griffin-Lim.

**Pros**: High quality audio, fast inference (single forward pass).
**Cons**: Requires training data, needs GPU for training.

### Vocoder Comparison

| Feature | Griffin-Lim | HiFi-GAN |
|---------|------------|----------|
| Training required | No | Yes (adversarial) |
| Audio quality | Acceptable, buzzy | High quality, natural |
| Inference speed | Slow (iterative) | Fast (single pass) |
| GPU required | No | Yes (for training) |
| Phase recovery | Approximate | Learned |

micro-Omni's `NeuralVocoder` wrapper automatically selects HiFi-GAN when a trained checkpoint is available, falling back to Griffin-Lim otherwise.

---

## The Complete Audio Pipeline

Putting it all together -- from text tokens to audible speech:

```
TEXT TOKENS (from Thinker)
       |
       v
+------------------+
| Talker (AR model)|  Predicts (code1, code2) per audio frame
| Ch. 08: decoder  |  using autoregressive generation with
| with KV cache    |  KV caching for speed
+------------------+
       |
       v
SPEECH CODES: (idx1, idx2) per frame
  e.g., [(47, 93), (12, 55), (81, 7), ...]
       |
       v
+------------------+
| RVQ Decoder      |  codebook1[47] + codebook2[93] -> 64-dim
|                  |  project 64-dim -> 128-dim mel frame
+------------------+
       |
       v
MEL SPECTROGRAM: (T_frames, 128)
       |
       v
+------------------+
| Vocoder          |  HiFi-GAN or Griffin-Lim
| mel -> waveform  |  128-dim mel frames -> 16kHz audio
+------------------+
       |
       v
AUDIO WAVEFORM: (T_samples,) at 16,000 Hz
  Play through speaker!
```

### Shape Summary

| Stage | Shape | Description |
|-------|-------|-------------|
| Talker output | (B, T, 2) | Two codebook indices per frame |
| RVQ decode: lookup | (B, T, 64) + (B, T, 64) | Two 64-dim embeddings |
| RVQ decode: sum | (B, T, 64) | Combined embedding |
| RVQ decode: project | (B, T, 128) | Reconstructed mel frame |
| Vocoder input | (B, 128, T) | Mel spectrogram (transposed) |
| Vocoder output | (B, T*256) | Audio waveform samples |

---

## Summary

| Concept | Key Point |
|---------|-----------|
| Core insight | Discrete codes let you generate speech like text |
| Vector quantization | Map continuous vector to nearest codebook entry |
| Codebook | 128 learned representative vectors in 64-dim space |
| Single VQ limitation | Only 128 distinct sounds -- not enough |
| Residual VQ | Quantize residual of previous codebook: 128^2 = 16,384 sounds |
| Straight-through estimator | Pass gradients through non-differentiable argmin |
| Griffin-Lim vocoder | Classical iterative phase recovery, no training needed |
| HiFi-GAN vocoder | Neural adversarial vocoder, high quality, needs training |
| micro-Omni codec | 2 codebooks, 128 entries each, 64-dim embeddings |

---

[← Back to Index](00-INDEX.md) | [Previous: Mixture of Experts](10-mixture-of-experts.md) | [Next: System Overview →](12-system-overview.md)
