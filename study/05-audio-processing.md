# Chapter 05: Audio — From Sound Waves to Tokens

## Sound = Air Pressure Over Time

When you speak, your vocal cords vibrate, pushing air molecules back and forth. A microphone measures this pressure change thousands of times per second and converts it into a sequence of numbers.

```
Sound wave (continuous):

  pressure
     │   ╱╲      ╱╲      ╱╲
     │  ╱  ╲    ╱  ╲    ╱  ╲
  0  │─╱────╲──╱────╲──╱────╲──► time
     │╱      ╲╱      ╲╱      ╲
     │

Digitized (sampled):

  pressure
     │   •       •       •
     │  • •     • •     • •
  0  │─•───•───•───•───•───•──► time
     │•     • •     • •     •
     │       •       •
```

Each dot is one **sample** — a single number representing air pressure at that instant.

---

## Sample Rate: 16,000 Measurements Per Second

The **sample rate** determines how many times per second we measure the waveform. For speech AI, the standard is **16 kHz** (16,000 samples/second).

### Why 16 kHz? The Nyquist Theorem

The Nyquist theorem states: to capture a frequency, you must sample at **twice** that frequency.

- Human speech contains useful information from roughly **85 Hz to 8,000 Hz**
- To capture up to 8,000 Hz, we need at least 16,000 samples/second
- Music uses 44.1 kHz (captures up to ~22 kHz, the edge of human hearing)

**Analogy:** Imagine filming a spinning wheel. If you take photos too slowly, the wheel appears to spin backwards (aliasing). You need at least two photos per revolution to see the true motion. Same idea — sample fast enough to capture the highest frequency you care about.

---

## The Problem: Too Many Numbers

Even at 16 kHz, raw audio produces a flood of data:

```
3 seconds of speech = 3 × 16,000 = 48,000 numbers

That's 48,000 input tokens if we treat each sample as one token.
A transformer with 48,000 tokens? Attention is O(n^2):
  48,000^2 = 2.3 BILLION operations per attention layer!
```

We need a much more compact representation. Enter the **mel spectrogram**.

---

## The Mel Spectrogram: A Time-Frequency Heatmap

Instead of feeding raw samples, we convert audio into a 2D image-like representation: a **mel spectrogram**. It shows **which frequencies are present at each moment in time**.

**Analogy:** Raw audio is like listening to an orchestra with your eyes closed — you hear everything mixed together. A mel spectrogram is like reading the sheet music — you can see each instrument's part (frequency) laid out over time.

```
            Mel Spectrogram
  frequency
  (mel bins)
     128 │░░░░░▓▓▓▓░░░░░░▓▓▓░░░░░░│  high freq
         │░░░▓▓████▓▓░░▓▓███▓░░░░░│
         │░▓▓██████████▓█████▓▓░░░│
         │▓████████████████████▓▓░│
       1 │████████████████████████│  low freq
         └────────────────────────┘
          0s        1s        2s      time →

  Bright = loud at that frequency and time
  Dark   = quiet
```

---

## STFT: Sliding Window FFT

The mel spectrogram is built using the **Short-Time Fourier Transform (STFT)**.

### How It Works

1. Take a small **window** of audio (e.g., 400 samples = 25ms)
2. Apply the **FFT** (Fast Fourier Transform) to decompose that window into its frequency components
3. **Slide** the window forward by a hop (e.g., 160 samples = 10ms)
4. Repeat until the end of the audio

```
Raw audio samples:
|████████████████████████████████████████████████████|

Window 1:  [========]
                  ↓ FFT → frequency snapshot 1

Window 2:     [========]
                     ↓ FFT → frequency snapshot 2

Window 3:        [========]
                        ↓ FFT → frequency snapshot 3
    ...

           hop ──►|  |◄── overlap

Result: a grid of frequency snapshots over time
```

**Analogy:** Reading sheet music bar by bar. Each bar (window) tells you which notes are played. Slide to the next bar, read again. Stack all the bars and you have the full score.

### Key Parameters

- **Window size (n_fft):** How many samples per window. Larger = better frequency resolution, worse time resolution. Typical: 400-1024.
- **Hop length:** How far to slide between windows. Smaller hop = more overlap = finer time resolution.

---

## The Mel Scale: Hearing Is Logarithmic

The FFT gives us frequencies on a linear scale (0 Hz, 100 Hz, 200 Hz, ...). But human ears don't hear linearly.

**The piano keys analogy:** On a piano, each octave doubles in frequency:
- A3 = 220 Hz
- A4 = 440 Hz (+220 Hz)
- A5 = 880 Hz (+440 Hz)
- A6 = 1760 Hz (+880 Hz)

Each octave sounds like the "same step up" to our ears, but the frequency gap doubles every time. We perceive pitch **logarithmically**.

The **mel scale** warps the frequency axis to match human perception:

```
Linear Hz:    |100|200|300|400|500|  ...  |7000|7500|8000|
               ▼   ▼   ▼   ▼   ▼          ▼    ▼    ▼
Mel scale:    |──|──|──|──|──|            |─|─|─|
              wide spacing               narrow spacing
              (we hear big               (we barely notice
               differences                differences
               at low freq)               at high freq)
```

The mel spectrogram applies triangular filter banks on the mel scale, giving us **128 mel bins** that emphasize the frequencies humans actually distinguish.

---

## micro-Omni Mel Settings

| Parameter | Value | Meaning |
|-----------|-------|---------|
| Sample rate | 16,000 Hz | Standard for speech |
| n_mels | 128 | Number of mel frequency bins |
| hop_length | 160 | Slide window every 160 samples (10ms) |
| n_fft | 400 | Window size for FFT (25ms) |
| Frames/second | 100 | 16000 / 160 = 100 frames per second |

So **3 seconds of audio** becomes a mel spectrogram of shape **(128, 300)** — 128 frequency bins by 300 time frames.

---

## Convolutional Downsampling: Compressing Time

300 frames for 3 seconds is still a lot. We use **2D convolutions with stride** to downsample the time axis.

**Analogy:** Imagine summarizing a book. Instead of reading every word (raw samples) or every sentence (mel frames), you write a summary for each chapter. The summary captures the essential information in far fewer words.

### How Conv Downsampling Works

A convolution with **stride 2** looks at overlapping patches but only outputs at every other position, halving the resolution:

```
Input:   [f1][f2][f3][f4][f5][f6][f7][f8]   (8 frames)
          ╲──╱   ╲──╱   ╲──╱   ╲──╱
Stride 2:  [o1]    [o2]    [o3]    [o4]      (4 frames)
```

Stack multiple conv layers to get larger downsample factors:

```
Layer 1 (stride 2):  100 frames/sec → 50 frames/sec    (2x)
Layer 2 (stride 2):  50 → 25                            (4x)
Layer 3 (stride 2):  25 → 12.5                          (8x)
```

### The 8x Downsample

With an **8x total downsample**:

```
3 seconds of audio:
  Raw samples:     48,000 numbers
  Mel spectrogram: 300 frames (100 Hz)
  After 8x conv:   38 tokens  (12.5 Hz)

Each "audio token" now represents ~80ms of audio.
```

**38 tokens** is on par with a short text sentence — perfectly manageable for a transformer!

---

## Full Audio Pipeline (with Shapes)

```
 ════════════════════════════════════════════════════════════
                    AUDIO PROCESSING PIPELINE
 ════════════════════════════════════════════════════════════

 Raw waveform (3 seconds at 16kHz)
 Shape: (48000,)
       │
       ▼
 ┌─────────────────────────┐
 │  STFT + Mel Filter Bank │   window=400, hop=160, 128 mels
 └───────────┬─────────────┘
             │
             ▼
 Mel Spectrogram
 Shape: (128, 300)              ← 128 freq bins × 300 time frames
       │
       ▼
 ┌─────────────────────────┐
 │  Log Scaling            │   log(mel + 1e-6)  — compress range
 └───────────┬─────────────┘
             │
             ▼
 Log-Mel Spectrogram
 Shape: (1, 128, 300)           ← add channel dim for conv
       │
       ▼
 ┌─────────────────────────┐
 │  Conv2d (stride 2)      │   Layer 1: (1,128,300) → (C,64,150)
 │  + ReLU                 │
 ├─────────────────────────┤
 │  Conv2d (stride 2)      │   Layer 2: (C,64,150) → (C,32,75)
 │  + ReLU                 │
 ├─────────────────────────┤
 │  Conv2d (stride 2)      │   Layer 3: (C,32,75) → (C,16,38)
 │  + ReLU                 │
 └───────────┬─────────────┘
             │
             ▼
 Downsampled features
 Shape: (C, 16, 38)             ← C channels, 16 freq, 38 time
       │
       ▼
 ┌─────────────────────────┐
 │  Flatten freq + channel │   Reshape: (C×16, 38)
 │  + Linear projection    │   Project to model dim: (d_model, 38)
 └───────────┬─────────────┘
             │
             ▼
 Audio tokens
 Shape: (38, d_model)           ← 38 tokens, each is a d_model vector

 Ready to feed into the transformer!
 ════════════════════════════════════════════════════════════
```

---

## Audio Augmentation

Training on clean audio alone makes a fragile model. **Data augmentation** creates artificial variety so the model generalizes to real-world conditions.

### Time Stretching

Speed up or slow down the audio without changing pitch. The model learns that the same word can be spoken at different speeds.

```
Original:   "hello"     (0.5 sec)
Stretched:  "h e l l o" (0.7 sec)   ← slower, same pitch
Compressed: "hello"     (0.3 sec)   ← faster, same pitch
```

**Analogy:** Playing a vinyl record at different speeds — but magically keeping the singer's voice at the same pitch.

### Pitch Shifting

Raise or lower the pitch without changing speed. Helps the model handle speakers with different voice depths.

### Noise Addition

Mix in background noise (cafe chatter, traffic, white noise) at random volumes. Forces the model to focus on speech, not silence.

```
Clean:     "How are you?"
+ Noise:   "How are you?" + [cafe sounds at 10dB SNR]
```

### SpecAugment

Directly mask portions of the mel spectrogram — the audio equivalent of dropout.

Two masking strategies:

```
 Frequency masking:          Time masking:
 ┌────────────────┐          ┌────────────────┐
 │████████████████│          │█████░░░████████│
 │████████████████│          │█████░░░████████│
 │░░░░░░░░░░░░░░░░│ ← masked│█████░░░████████│
 │░░░░░░░░░░░░░░░░│          │█████░░░████████│
 │████████████████│          │█████░░░████████│
 │████████████████│          │█████░░░████████│
 └────────────────┘          └────────────────┘
  Block out some              Block out a time
  frequency bands             segment
```

**Why it works:** By hiding parts of the spectrogram, the model can't rely on any single frequency band or time window. It must learn robust, redundant representations — just like a student who studies the whole textbook because they don't know which questions will be on the test.

---

## Summary

| Stage | Input | Output | Reduction |
|-------|-------|--------|-----------|
| Microphone | Sound wave | 48,000 samples (3s) | — |
| Mel spectrogram | 48,000 samples | (128, 300) | Structured |
| Conv downsample (8x) | (128, 300) | 38 tokens | 1263x from raw |
| Projection | 38 feature vectors | 38 vectors of d_model | Ready for transformer |

**Key takeaway:** The audio pipeline converts a flood of 48,000 raw numbers into just 38 rich, information-dense tokens — a 1000x compression that makes transformer processing feasible while preserving the content of speech.
