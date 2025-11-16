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

### Waveform Representation

```
Amplitude
    ↑
  1 │    ╱╲        ╱╲
    │   ╱  ╲      ╱  ╲
  0 │──╱────╲────╱────╲──→ Time
    │        ╲  ╱        ╲╱
 -1 │         ╲╱

Digital audio: sequence of amplitude values
Sample rate: 16,000 samples/second (16kHz)
1 second audio = 16,000 numbers
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

### Why Mel Scale?

```
Human hearing is logarithmic:
- More sensitive to low frequencies
- Less sensitive to high frequencies

Linear scale:        Mel scale (perceptual):
0Hz ──── 1000Hz     0Hz ──────────── 1000Hz
1000 ──── 2000      1000 ───── 1500 ── 2000
2000 ──── 3000      2000 ── 2500 ──── 3000
...                 ...

Mel scale matches human perception!
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

