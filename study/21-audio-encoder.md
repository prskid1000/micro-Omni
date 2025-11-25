# Chapter 21: Audio Encoder (AuT-Tiny)

[Back to Index](00-INDEX.md) | [Next: Vision Encoder →](22-vision-encoder.md)

---

## 🎯 Learning Objectives

By the end of this chapter, you will understand:

- What the Audio Encoder does and why we need it
- How mel spectrograms are processed into embeddings
- The 8x downsampling strategy and why it matters
- Complete architecture breakdown
- How it connects to the Thinker
- Training process with CTC loss

---

## 💡 What is the Audio Encoder?

### The Speech Understanding Module

**Analogy: A Translator for Sound**

```
Think of audio processing like understanding a foreign language:

RAW AUDIO (waveform):
[0.5, -0.3, 0.8, -0.2, 0.1, ...]
↓
Like hearing: "Blah blah blah blah"
- You hear sounds, but don't understand meaning
- Too detailed (16,000 numbers per second!)
- Hard to process

MEL SPECTROGRAM:
100 frames per second, 128 frequency bins
↓
Like seeing phonetic notation: "kæt sæt ɒn mæt"
- Shows sound patterns visually
- Still very detailed (100 frames/second)
- Better, but still a lot to process

AUDIO ENCODER OUTPUT:
12.5 embeddings per second, 256 dimensions
↓
Like understanding concepts: "cat" "sat" "on" "mat"
- Captures MEANING, not just sound
- Efficient (12.5 per second, not 100!)
- Ready for reasoning (Thinker can use it)

The Audio Encoder is the TRANSLATOR:
Sound patterns → Meaningful representations!
```

**Why Do We Need This?**

```
Problem: Thinker can't work with raw mel spectrograms!
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Mel spectrogram issues:
❌ Too many frames (100 per second = 300 frames for 3 seconds)
❌ Wrong dimension (128, but Thinker needs 256)
❌ Too low-level (acoustic features, not semantic)
❌ Doesn't align with text/image embeddings

Solution: Audio Encoder!
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Audio Encoder transforms:
✅ 100 frames/sec → 12.5 frames/sec (8x reduction)
✅ 128 acoustic features → 256 semantic embeddings
✅ Low-level sound → High-level meaning
✅ Aligns with text/image embeddings (all 256-dim)

Now Thinker can:
- Process audio efficiently
- Understand meaning (not just sound)
- Combine with text and images seamlessly!
```

---

## 🏗️ Detailed Architecture Breakdown

### Two Operating Modes

The Audio Encoder supports two modes for different use cases:

```
MODE 1: CTC Mode (use_attention_pooling=False) - DEFAULT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Use case: Speech recognition (ASR)
Output: Frame sequence (B, T/8, d_model)
Loss: CTC (Connectionist Temporal Classification)
Benefits: Preserves temporal alignment for ASR

MODE 2: Contrastive Mode (use_attention_pooling=True)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Use case: Audio-text contrastive learning (CLAP-style)
Output: Pooled embedding (B, d_model)
Loss: Contrastive (audio-text similarity)
Benefits: Fixed-size representation for retrieval/classification
Based on: LAION-CLAP (2023 ICASSP paper)
```

### The Complete Pipeline (CTC Mode)

```
INPUT: 3 seconds of speech
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Step 1: Start with mel spectrogram
Shape: (300, 128)
- 300 frames (3 seconds × 100 Hz)
- 128 mel frequency bins
Size: 300 × 128 = 38,400 numbers!

Step 2: Reshape for convolution
Shape: (1, 128, 300)  [batch, channels, time]
- Treat like a 1D "image"
- Height = 128 (frequency)
- Width = 300 (time)

Step 3: Convolutional Downsampling (4x or 8x)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

WHY downsample?
- 100 frames/sec is TOO MUCH for a language model
- Speech doesn't change that fast
- Most phonemes last ~50-100ms (5-10 frames)

HOW? Stack of convolutional layers:
ConvDown Block (4x reduction):
  Conv1: Stride 2 → 300 → 150 frames (50 Hz)
  Conv2: Stride 2 → 150 → 75 frames (25 Hz)

Optional Extra Conv (for 8x total):
  Conv3: Stride 2 → 75 → 37.5 frames (12.5 Hz)

Total: 4x or 8x reduction (300 → 75 or 37.5 frames)

Result: (1, 64, 75) or (1, 64, 37)
- 75 or 37 frames (25 Hz or 12.5 Hz)
- 64 channels (learned features)

Step 4: Flatten & Project
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Reshape: (1, 37, 192)
- 37 time steps
- 192 dimensions per step

Now it's a sequence! Like tokens in text.

Step 5: Transformer Encoder (4 layers)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Process with attention:
- Each frame attends to all other frames
- Captures temporal dependencies
- "The sound at time 5 relates to sounds at time 3 and 7"

4 layers of:
  - Self-attention (frames talk to each other)
  - Feedforward network (process each frame)
  - RMSNorm (stabilize)

Output: (1, 37, 192)
Now each frame has SEMANTIC meaning!

Step 6: Final Normalization
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RMSNorm for stability
Output: (1, 37, 192)

Step 7a: CTC Mode (default)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Output frame sequence: (1, 37, 192)
Ready for CTC head → ASR training

Step 7b: Contrastive Mode (optional)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Attention Pooling:
- Learns importance weights for each frame
- Weighted sum across time dimension
- Handles variable-length audio with masking

Process:
  weights = Linear(192 → 1)(x)  # (1, 37, 1)
  weights = softmax(weights, dim=1)  # normalize
  pooled = sum(x * weights)  # weighted average
  
Output: (1, 192) pooled embedding
Ready for contrastive learning (CLAP-style)

Step 8: Audio Projector (External)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Linear projection: 192 → 256 dimensions

WHY? Align with Thinker's dimension!
- Thinker expects 256-dim embeddings
- Text embeddings: 256-dim
- Image embeddings: 256-dim
- Audio embeddings: 192-dim → 256-dim ✓

Final output: 
- CTC mode: (1, 37, 256) frame sequence
- Contrastive mode: (1, 256) pooled embedding

READY FOR THINKER! 🎉
```

### Visual Architecture

```
┌─────────────────────────────────────────┐
│  INPUT: Mel Spectrogram                 │
│  Shape: (batch=1, time=300, freq=128)   │
│  "meow" spoken for 3 seconds            │
└────────────────┬────────────────────────┘
                 ↓
┌─────────────────────────────────────────┐
│  CONVOLUTIONAL DOWNSAMPLING             │
│  (Configurable: 4x or 8x reduction)     │
│  ┌────────────────────────────────────┐ │
│  │ ConvDown Block (4x reduction)      │ │
│  │ - Conv2d (stride 2) + GELU         │ │
│  │   Input: (1, 128, 300)             │ │
│  │   Output: (1, 64, 150)             │ │
│  │ - Conv2d (stride 2) + GELU         │ │
│  │   Input: (1, 64, 150)              │ │
│  │   Output: (1, 64, 75)              │ │
│  └────────────────────────────────────┘ │
│  ┌────────────────────────────────────┐ │
│  │ Extra Conv (optional, for 8x)      │ │
│  │ - Conv2d (stride 2) + GELU         │ │
│  │   Input: (1, 64, 75)               │ │
│  │   Output: (1, 64, 37)              │ │
│  │ (Used when downsample_factor=8)    │ │
│  └────────────────────────────────────┘ │
│  Output: (1, 64, 75) or (1, 64, 37)     │
│  Temporal reduction complete! ✓         │
└────────────────┬────────────────────────┘
                 ↓
┌─────────────────────────────────────────┐
│  FLATTEN & PROJECT                      │
│  (1, 64, T) → (1, T, 64*freq_bins)     │
│  Then Linear → (1, T, 192)             │
│  Now it's a sequence of T vectors!     │
└────────────────┬────────────────────────┘
                 ↓
┌─────────────────────────────────────────┐
│  TRANSFORMER ENCODER                    │
│  (Supports Flash Attention for speed)   │
│  ┌────────────────────────────────────┐ │
│  │ Block 1: Attention + FFN + Norm   │ │
│  └────────────────────────────────────┘ │
│  ┌────────────────────────────────────┐ │
│  │ Block 2: Attention + FFN + Norm   │ │
│  └────────────────────────────────────┘ │
│  ┌────────────────────────────────────┐ │
│  │ Block 3: Attention + FFN + Norm   │ │
│  └────────────────────────────────────┘ │
│  ┌────────────────────────────────────┐ │
│  │ Block 4: Attention + FFN + Norm   │ │
│  └────────────────────────────────────┘ │
│  Each frame now understands context!   │
└────────────────┬────────────────────────┘
                 ↓
┌─────────────────────────────────────────┐
│  FINAL RMSNORM                          │
│  Stabilize outputs                      │
│  Output: (1, T, 192)                    │
└────────────────┬────────────────────────┘
                 ↓
          ┌──────┴──────┐
          │   MODE?     │
          └──────┬──────┘
         ┌───────┴────────┐
         ↓                ↓
┌────────────────┐  ┌────────────────┐
│  CTC MODE      │  │ CONTRASTIVE    │
│  (default)     │  │ MODE (CLAP)    │
└────────┬───────┘  └────────┬───────┘
         ↓                   ↓
┌────────────────┐  ┌────────────────┐
│ Frame Sequence │  │ Attention Pool │
│ (1, T, 192)    │  │ Linear(192→1)  │
│                │  │ + Softmax      │
│                │  │ + Weighted Sum │
│                │  │ → (1, 192)     │
└────────┬───────┘  └────────┬───────┘
         └──────┬─────────────┘
                ↓
┌─────────────────────────────────────────┐
│  AUDIO PROJECTOR (External)             │
│  Linear: 192 dim → 256 dim             │
│  Align with Thinker's dimension!       │
└────────────────┬────────────────────────┘
                 ↓
┌─────────────────────────────────────────┐
│  OUTPUT: Audio Embeddings               │
│  CTC mode: (1, T, 256) frame sequence   │
│  Contrastive: (1, 256) pooled embedding │
│  Ready for respective tasks! ✓          │
└─────────────────────────────────────────┘
```

---

## 🔍 Why 8x Downsampling?

### The Temporal Resolution Trade-off

**Analogy: Video Frame Rate**

```
VERY HIGH FRAME RATE (240 fps):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- Captures every tiny movement
- Extremely smooth
- BUT: 240 frames per second!
- TOO MUCH data to process
- Expensive storage and computation

NORMAL FRAME RATE (30 fps):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- Still captures motion well
- Smooth enough for viewing
- 8x less data than 240 fps
- Easier to process
- Good balance! ✓

Same idea for audio!
```

**Technical Reasoning:**

```
Speech characteristics:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Phoneme duration: ~50-150 milliseconds
- That's 5-15 frames at 100 Hz
- Or 0.6-2 frames at 12.5 Hz

Word duration: ~200-500 milliseconds
- That's 20-50 frames at 100 Hz
- Or 2.5-6 frames at 12.5 Hz

Key insight:
- You don't need 100 frames/sec to understand speech!
- Phonemes don't change that fast
- 12.5 frames/sec captures all meaningful changes
- 8x less computation, same understanding!

Benefits of 12.5 Hz:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

✅ Efficiency: 8x fewer tokens to process
✅ Context: Fit more seconds of audio in same context
✅ Alignment: Closer to text token rate (~3-5 tokens/word)
✅ Quality: Still captures all phonetic information

Example:
3 seconds of speech:
- At 100 Hz: 300 tokens (too many!)
- At 12.5 Hz: 37 tokens (perfect!)

Compare to text:
"The cat sat on the mat" = 6 words ≈ 6-12 tokens
Spoken in ~2 seconds = 25 audio tokens at 12.5 Hz
Similar scale! ✓
```

---

## 📊 Detailed Specifications

> **Note**: These are the "tiny" configuration values from `configs/audio_enc_tiny.json`. The code defaults may differ, but config files override them.

### Architecture Parameters

```
CONVOLUTIONAL DOWNSAMPLER:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Conv1:
  Input channels: 128 (mel bins)
  Output channels: 64
  Kernel size: 5
  Stride: 2 (downsample by 2x)
  → 300 frames → 150 frames

Conv2:
  Input channels: 64
  Output channels: 128
  Kernel size: 5
  Stride: 2 (downsample by 2x)
  → 150 frames → 75 frames

Conv3:
  Input channels: 128
  Output channels: 192
  Kernel size: 5
  Stride: 2 (downsample by 2x)
  → 75 frames → 37 frames

Total downsampling: 2 × 2 × 2 = 8x

TRANSFORMER ENCODER:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Dimension: 192
Layers: 4
Attention heads: 3
FFN dimension: 768 (4 × 192)
Dropout: 0.1
Normalization: RMSNorm

PROJECTOR:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Linear: 192 → 256 (no bias)

TOTAL PARAMETERS:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Convolutional layers: ~500K
Transformer blocks: ~9M
Projector: ~50K
Total: ~2.05M parameters
```

### Comparison Table

| Component           | Input      | Output     | Purpose                |
| ------------------- | ---------- | ---------- | ---------------------- |
| **Conv Downsample** | (T, 128)   | (T/8, 192) | Temporal compression   |
| **Transformer**     | (T/8, 192) | (T/8, 192) | Semantic understanding |
| **Projector**       | (T/8, 192) | (T/8, 256) | Dimension alignment    |

---

## 🎓 Training Process

### Pretraining with ASR (Automatic Speech Recognition)

**Why ASR for Pretraining?**

```
Goal: Teach audio encoder to understand speech content

ASR Task: Audio → Text transcription
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Input: Audio of someone saying "hello world"
Output: Text "hello world"

This forces the encoder to:
✅ Learn phonetic patterns
✅ Understand word boundaries
✅ Capture semantic meaning
✅ Ignore irrelevant details (noise, speaker identity)

Perfect pretraining for multimodal understanding!
```

**CTC Loss: Connectionist Temporal Classification**

```
The Challenge:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Audio frames: 37 tokens for "hello"
Text: 5 characters "h e l l o"

Problem: How do we align 37 frames to 5 characters?
- Frame 1-8: "h"?
- Frame 9-15: "e"?
- Frame 16-20: "l"?
- ...

We don't know the alignment!

CTC Solution:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Allows flexible alignment:
- Frames can map to any character
- Special "blank" token for silence/transitions
- Automatically learns best alignment!

Example alignment:
Frame 1-3:   blank (silence)
Frame 4-10:  "h" (stretched)
Frame 11-12: blank (transition)
Frame 13-18: "e"
Frame 19-20: "l"
Frame 21-23: "l"
Frame 24-28: "o"
Frame 29-37: blank (end)

Collapse repeats: hhhhh → h, ll → l
Result: "hello" ✓

CTC handles variable-length alignment automatically!
```

**Training Loop:**

```python
for batch in dataloader:
    audio, text = batch

    # 1. Extract mel spectrogram
    mel = audio_to_mel(audio)  # (B, T, 128)
    # Note: All mel spectrograms are padded to max_mel_length
    # for CUDA graphs compatibility (when use_compile: true)

    # 2. Encode with audio encoder
    embeddings = audio_encoder(mel)  # (B, T/8, 192)

    # 3. Project to CTC prediction head
    logits = ctc_head(embeddings)  # (B, T/8, vocab_size)

    # 4. Compute CTC loss
    loss = ctc_loss(logits, text)

    # 5. Backprop and update
    loss.backward()
    optimizer.step()
```

**CUDA Graphs Compatibility:**

- When using `use_compile: true`, all batches must have uniform shapes
- `max_mel_length` is auto-calculated from dataset (95th percentile, typically ~2048 frames = ~20 seconds)
- All mel spectrograms are padded to this fixed length (samples exceeding threshold are skipped during dataset iteration)
- Prevents "tensor size mismatch" errors with CUDA graphs compilation
- See Chapter 34 (Configuration Files) for details

---

## 🔗 Connection to Thinker

### How Audio Flows into Multimodal Processing

```
COMPLETE PIPELINE:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. User says: "Show me a cat"
   Raw audio: 48,000 samples (3 seconds at 16kHz)

2. Convert to mel:
   Mel spectrogram: (300, 128) at 100 Hz

3. Audio Encoder processes:
   → Downsample 8x: 300 → 37 frames
   → Understand semantics via transformer
   → Project to 256-dim: (37, 256)

4. Tokenize text prompt:
   "show me a cat" → [15, 234, 42, 89, 234]
   → Embed: (5, 256)

5. Concatenate:
   Combined input: (42, 256)
   = [37 audio tokens, 5 text tokens]

6. Thinker processes:
   → Cross-modal attention
   → Audio tokens interact with text tokens
   → Understands: User wants to see a cat image

7. Generate response:
   "Here is an image of a cat..."

Audio encoder enabled multimodal understanding! ✓
```

---

## 💡 Key Takeaways

✅ **Audio Encoder** translates sound into semantic embeddings  
✅ **Two operating modes**: CTC (ASR) and Contrastive (CLAP)  
✅ **4x or 8x downsampling** (100Hz → 25Hz or 12.5Hz) for efficiency  
✅ **Convolutional layers** compress temporal dimension  
✅ **Transformer encoder** captures semantic meaning  
✅ **Attention pooling** (contrastive mode) for fixed-size embeddings  
✅ **Projects to 256-dim** to align with Thinker  
✅ **Pretrained with CTC loss** on ASR task (current)  
✅ **~2.05M parameters** - compact and efficient  
✅ **Enables multimodal** audio+text+image understanding  
✅ **CLAP-compatible** for audio-text contrastive learning

---

## 🎓 Self-Check Questions

1. Why do we need an audio encoder instead of feeding mel spectrograms directly to the Thinker?
2. What does 8x downsampling mean and why is it beneficial?
3. What is CTC loss and why is it used for training?
4. How many tokens does 3 seconds of speech become after the audio encoder?
5. Why do we project from 192 to 256 dimensions at the end?
6. What are the two operating modes and when would you use each?
7. How does attention pooling work in contrastive mode?

<details>
<summary>📝 Click to see answers</summary>

1. Mel spectrograms are too low-level (acoustic features), too many frames (100/sec), and wrong dimension (128). Audio encoder converts them to semantic embeddings (meaningful), efficient rate (12.5/sec or 25/sec), and correct dimension (256)
2. 8x downsampling means reducing frame rate from 100 Hz to 12.5 Hz (100/8). Beneficial because: 8x less computation, captures all phonetic info, aligns better with text token rate
3. CTC (Connectionist Temporal Classification) allows flexible alignment between audio frames and text characters without requiring explicit time stamps - perfect for ASR training
4. With 8x downsampling: 3 seconds × 12.5 Hz = 37-38 tokens. With 4x: 3 seconds × 25 Hz = 75 tokens
5. To align with Thinker's input dimension (256) - all modalities (text, image, audio) must be 256-dim for unified processing
6. CTC mode (default): For ASR, outputs frame sequence. Contrastive mode: For CLAP-style audio-text retrieval/classification, outputs pooled embedding with learned attention weights
7. Attention pooling learns importance weights for each frame via Linear(d_model→1), applies softmax to normalize, then computes weighted sum across time. Handles variable-length audio with masking
</details>

---

[Continue to Chapter 22: Vision Encoder →](22-vision-encoder.md)

**Chapter Progress:** μOmni Components ●○○○○ (1/5 complete)

---

## 📊 Specifications

| Parameter               | Value                                                                                        |
| ----------------------- | -------------------------------------------------------------------------------------------- |
| **Input**               | Mel spectrogram (T, 128)                                                                     |
| **Downsample**          | 4x (25Hz) or 8x (12.5Hz)                                                                     |
| **Dimension**           | 192                                                                                          |
| **Layers**              | 4                                                                                            |
| **Heads**               | 3                                                                                            |
| **Parameters**          | ~2.05M                                                                                       |
| **Modes**               | CTC (frame sequence) or Contrastive (pooled)                                                 |
| **Attention Pooling**   | Learned weights (contrastive mode only)                                                      |
| **max_mel_length**      | Auto-calculated from dataset (95th percentile, ~20s typical) - for CUDA graphs compatibility |
| **Flash Attention**     | Supported (PyTorch 2.0+)                                                                     |
| **torch.compile()**     | Optional (30-50% speedup)                                                                    |

## 🎓 Training

**Current Task**: ASR (Automatic Speech Recognition)  
**Current Loss**: CTC (Connectionist Temporal Classification)  
**Current Data**: Audio + transcriptions

**Future Task**: Audio-text contrastive learning (CLAP)  
**Future Loss**: Contrastive (InfoNCE)  
**Future Data**: Audio + captions

## 💡 Quick Summary

✅ **Processes mel spectrograms** into semantic embeddings  
✅ **4x or 8x temporal downsampling** for efficiency  
✅ **Two modes**: CTC (ASR) and Contrastive (CLAP)  
✅ **Outputs 192-dim embeddings** (projected to 256-dim)  
✅ **Trained with CTC loss** on ASR task (current)  
✅ **CLAP-compatible** with attention pooling

---

[Back to Index](00-INDEX.md)
