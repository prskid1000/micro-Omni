# Chapter 10: Audio Processing for AI

[← Previous: Tokenization](09-tokenization.md) | [Back to Index](00-INDEX.md) | [Next: Image Processing →](11-image-processing.md)

---

## 🎯 What You'll Learn

- Audio representation fundamentals
- Mel spectrograms and why they're used
- Audio preprocessing pipeline
- How μOmni processes audio

---

## 🎵 Audio Basics

### Understanding Sound as Data

Before we talk about processing, let's understand what sound IS to a computer:

**What YOU Hear vs What the COMPUTER Sees:**

```
When you hear someone say "Hello":
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
YOU: 
- Hear a voice
- Recognize the word "Hello"
- Understand the meaning

THE COMPUTER:
- Sees a bunch of numbers!
- Each number = air pressure at a moment in time
```

**Analogy: Ocean Waves**

```
Think of sound like ocean waves:

Sound in air:
- Speaker pushes air → High pressure (peak)
- Speaker pulls air → Low pressure (valley)
- This creates a wave pattern

Your ear:
- Detects these pressure changes
- Sends signals to brain
- Brain interprets as sound!

The computer:
- Microphone measures pressure many times per second
- Saves each measurement as a number
- This sequence of numbers = audio waveform!
```

### Waveform Representation

```
Amplitude (air pressure)
    ↑
  1 │    ╱╲        ╱╲         ← High pressure (speaker pushed)
    │   ╱  ╲      ╱  ╲
  0 │──╱────╲────╱────╲──→ Time
    │        ╲  ╱        ╲╱   ← Low pressure (speaker pulled)
 -1 │         ╲╱

Digital audio: sequence of amplitude values
Think of it as: [0.5, 0.8, 1.0, 0.8, 0.3, -0.2, -0.5, -0.8, -0.5, 0.0, ...]

Sample rate: 16,000 samples/second (16kHz)
Meaning: We measure the air pressure 16,000 times per second!

Why 16,000?
- Human speech frequencies: ~85 Hz to 8,000 Hz
- Nyquist theorem says: Need 2× highest frequency
- 2 × 8,000 = 16,000 Hz (perfect for speech!)

1 second audio = 16,000 numbers
3 seconds audio = 48,000 numbers (too many for a neural network!)
```

**Concrete Example:**

```
You say "Hi" (0.5 seconds):

Microphone samples at 16kHz:
0.5 seconds × 16,000 samples/sec = 8,000 measurements!

The waveform might look like:
Time (ms):   0    10    20    30    40    50   ...  500
Amplitude: 0.0  0.3  0.8  0.5 -0.2 -0.6  ... 0.0
            ↑    ↑    ↑    ↑    ↑    ↑         ↑
         Start  H    H    i    i   end    silence

Just 8,000 numbers represent "Hi"!
But this is still TOO MUCH data for efficient processing!
```

---

## 🔊 From Waveform to Spectrogram

### Problem with Raw Waveforms

```
3 seconds of audio at 16kHz = 48,000 samples
Too long for neural networks!

Solution: Convert to time-frequency representation
```

---

### Spectrogram

```
Time-Frequency representation:

Frequency (Hz)
    ↑
8000│ ░░░░░░░░████░░░░  (high frequencies)
4000│ ▓▓▓▓████▓▓▓▓▓▓▓▓
2000│ ████▓▓▓▓████████
1000│ ████████████████
 100│ ████████████████  (low frequencies)
    └───────────────────→ Time (sec)
    0    0.5   1.0  1.5

Brightness = Energy at that frequency at that time
```

---

## 🎨 Mel Spectrogram

### Why Mel Scale? (Matching Human Hearing!)

**Understanding Human Hearing:**

Your ears don't hear frequencies linearly - they hear logarithmically!

**Experiment: Listen to These Frequency Differences**

```
LOW FREQUENCIES (easy to distinguish):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
100 Hz vs 200 Hz:
You: "These sound VERY different!" ✓
Difference: 100 Hz

200 Hz vs 300 Hz:
You: "These sound pretty different!" ✓
Difference: 100 Hz (same as above!)

HIGH FREQUENCIES (hard to distinguish):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
10,000 Hz vs 10,100 Hz:
You: "These sound almost identical!" 
Difference: 100 Hz (same as before!)

10,000 Hz vs 11,000 Hz:
You: "Now I can hear a difference!"
Difference: 1,000 Hz!

INSIGHT:
At low frequencies, 100 Hz difference is HUGE!
At high frequencies, 100 Hz difference is TINY!

Your brain cares about RATIOS, not absolute differences!
```

**Analogy: Money and Perception**

```
Low amounts (like low frequencies):
$100 vs $200 = HUGE difference! You definitely notice!
Difference: $100

High amounts (like high frequencies):
$10,000 vs $10,100 = Barely notice the difference
Difference: $100 (same absolute difference!)

$10,000 vs $11,000 = Now you notice!
Difference: $1,000

Same principle with sound frequencies!
```

**Linear Scale vs Mel Scale:**

```
Linear scale (equal spacing):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
0Hz ───── 1000Hz ───── 2000Hz ───── 3000Hz
     ↕️           ↕️            ↕️
  +1000 Hz    +1000 Hz     +1000 Hz

Problem: Wastes resolution!
- Too much detail at high frequencies (where we can't hear well)
- Too little detail at low frequencies (where we hear best)

Mel scale (perceptual spacing):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
0Hz ──────────── 1000Hz ─── 2000Hz ── 3000Hz
      ↕️              ↕️         ↕️
   +1000 Hz       +1000 Hz   +1000 Hz

But in mel units:
0 mel ─── 500 mel ─── 800 mel ─── 1000 mel
      ↕️          ↕️          ↕️
  +500 mel   +300 mel    +200 mel

Benefit: More detail where we hear well (low freq)
         Less detail where we don't (high freq)

Mel scale matches human perception!
```

**Why This Matters for AI:**

```
If we use LINEAR frequency bins:
- Wasting computation on frequencies humans barely hear
- Missing important details in frequencies humans hear well

If we use MEL frequency bins:
- Focus computation where it matters (human-perceivable differences)
- More efficient representation
- Better for speech recognition!

μOmni uses mel bins because:
✓ Efficient (don't waste resources on imperceptible differences)
✓ Perceptual (matches how humans actually hear)
✓ Standard (proven effective for speech AI)
```

---

### Mel Spectrogram Pipeline

```
1. Audio Waveform
   ↓
2. Short-Time Fourier Transform (STFT)
   [Sliding window, compute frequencies]
   ↓
3. Power Spectrum
   [Convert to energy/magnitude]
   ↓
4. Mel Filter Bank
   [Group frequencies into mel bins]
   ↓
5. Log Scale
   [log(power) for better range]
   ↓
6. Mel Spectrogram (T, 128)

μOmni uses:
- Sample rate: 16kHz
- FFT size: 1024
- Hop length: 160 (frame rate 100Hz)
- Mel bins: 128
```

---

## 💻 Computing Mel Spectrogram

```python
import torchaudio

# Create mel spectrogram transform
mel_spec = torchaudio.transforms.MelSpectrogram(
    sample_rate=16000,
    n_fft=1024,          # FFT window size
    hop_length=160,      # Step size (16000/160 = 100 frames/sec)
    win_length=400,      # Window length (25ms)
    n_mels=128          # Number of mel bins
)

# Apply to audio
waveform, sr = torchaudio.load("audio.wav")  # (1, samples)
mel = mel_spec(waveform)                     # (1, 128, T_frames)

# Transpose for μOmni: (1, T_frames, 128)
mel = mel.transpose(1, 2)
```

---

## 📊 Frame Rate Calculation

```
Input: 3-second audio at 16kHz
Samples: 3 × 16,000 = 48,000

After Mel Spectrogram (hop=160):
Frames: 48,000 / 160 = 300 frames
Frame rate: 100 Hz (100 frames per second)

After Audio Encoder downsampling (8x):
Frames: 300 / 8 = 37.5 ≈ 38 frames
Final rate: 12.5 Hz

Result: 3 seconds → 38 tokens!
Much more manageable than 48,000 samples
```

---

## 🎯 μOmni's Audio Pipeline

### Complete Flow

```
┌─────────────────────────────────────┐
│  1. Load Audio (16kHz WAV)          │
│     waveform: (1, 48000)            │
└─────────────┬───────────────────────┘
              ↓
┌─────────────────────────────────────┐
│  2. Mel Spectrogram Transform       │
│     mel: (1, 128, 300)              │
└─────────────┬───────────────────────┘
              ↓
┌─────────────────────────────────────┐
│  3. Transpose                       │
│     mel: (1, 300, 128)              │
└─────────────┬───────────────────────┘
              ↓
┌─────────────────────────────────────┐
│  4. Audio Encoder (AuT-Tiny)        │
│     - ConvDown (8x downsample)      │
│     - Transformer encoder           │
│     Output: (1, 38, 192)            │
└─────────────┬───────────────────────┘
              ↓
┌─────────────────────────────────────┐
│  5. Audio Projector                 │
│     Linear(192 → 256)               │
│     Output: (1, 38, 256)            │
└─────────────┬───────────────────────┘
              ↓
┌─────────────────────────────────────┐
│  6. Concatenate with Text/Image     │
│     Ready for Thinker!              │
└─────────────────────────────────────┘
```

---

## 🎤 Speech vs Music vs Noise

### Different Audio Types

```
SPEECH:
Frequency: 85-255 Hz (fundamental)
           Up to 8kHz (harmonics)
Pattern: Periodic with gaps
    ████░░████░░░████░░████

MUSIC:
Frequency: 20Hz - 20kHz (full range)
Pattern: Rich harmonics, continuous
    ████████████████████████

NOISE:
Frequency: Spread across spectrum
Pattern: Random, no structure
    ░▓░▓▓░░▓░▓░░▓▓░▓░░▓░░▓

μOmni is trained primarily on speech!
```

---

## ⚡ Audio Augmentation

### Training Techniques

```python
# 1. Time Stretching (speed change)
stretched = torchaudio.transforms.TimeStretch()(mel, rate=0.9)

# 2. Pitch Shifting
shifted = torchaudio.transforms.PitchShift(
    sample_rate=16000, n_steps=2
)(waveform)

# 3. Add Noise
noisy = waveform + 0.005 * torch.randn_like(waveform)

# 4. Volume Perturbation
louder = waveform * random.uniform(0.8, 1.2)

# 5. SpecAugment (mask time/frequency)
masked = mask_along_axis(mel, mask_width=20, axis=1)  # Time
masked = mask_along_axis(masked, mask_width=10, axis=0)  # Freq
```

---

## 📊 Audio Representations Compared

| Representation | Dimensions | Size (3s) | Info Preserved |
|----------------|------------|-----------|----------------|
| **Raw Waveform** | (48000,) | 48K values | Complete |
| **STFT Spectrogram** | (513, 300) | 154K values | High |
| **Mel Spectrogram** | (128, 300) | 38K values | Perceptual |
| **Audio Encoder Output** | (38, 192) | 7.3K values | Semantic |

```
Compression: 48,000 → 7,300 (6.5x reduction)
While preserving semantic meaning!
```

---

## 💡 Key Takeaways

✅ **Waveforms** are raw audio (amplitude over time)  
✅ **Spectrograms** show frequency content over time  
✅ **Mel scale** matches human perception  
✅ **μOmni uses 128 mel bins** at 100Hz frame rate  
✅ **Downsampling** (8x) reduces to 12.5Hz  
✅ **Final output** is manageable sequence length

---

## 🎓 Self-Check Questions

1. What is a mel spectrogram?
2. Why use mel scale instead of linear frequency scale?
3. What is the frame rate after mel spectrogram in μOmni?
4. How many mel bins does μOmni use?
5. What's the final frame rate after audio encoder downsampling?

<details>
<summary>📝 Answers</summary>

1. Time-frequency representation using perceptually-motivated mel scale
2. Mel scale matches human hearing (more sensitive to low frequencies)
3. 100 Hz (100 frames per second, from hop_length=160 at 16kHz sample rate)
4. 128 mel bins
5. 12.5 Hz (100Hz / 8x downsampling)
</details>

---

[Continue to Chapter 11: Image Processing →](11-image-processing.md)

**Chapter Progress:** Core Concepts ●●●●●○○ (5/7 complete)

