# Chapter 16: Talker & Speech Generation

The Talker is the system's voice. While the Thinker generates text responses, the Talker converts those responses into speech by predicting sequences of RVQ (Residual Vector Quantization) codes -- discrete tokens that represent audio frames. These codes are then decoded back to a mel spectrogram and finally to an audible waveform by a vocoder.

Think of it like a skilled voice actor reading a script: the Thinker writes the script (text), and the Talker performs it (speech), deciding the rhythm, pacing, and acoustic details.

---

## The Full TTS Pipeline

```
"Hello world"
    |
    v
+---------------------+
|    Thinker           |   Generates text tokens + hidden states
|    (Chapter 13)      |
+---------------------+
    |
    v  hidden states (B, T, 384)
+---------------------+
|    Talker            |   Predicts RVQ codes frame by frame
|    (TalkerTiny)      |
+---------------------+
    |
    v  RVQ codes (B, T_frames, 2)     e.g., [[42, 17], [89, 3], ...]
+---------------------+
|    RVQ Decode        |   Looks up codebook vectors, sums residuals
|    (codec.py)        |
+---------------------+
    |
    v  mel spectrogram (B, T_frames, 128)
+---------------------+
|    Vocoder           |   Converts mel to audio waveform
|    (HiFi-GAN or     |
|     Griffin-Lim)     |
+---------------------+
    |
    v
Audio waveform (16kHz PCM)
```

---

## Part 1: The Talker (TalkerTiny)

### Architecture

The Talker is a **decoder-only transformer** that reuses the exact same `Block` class as the Thinker (Chapter 13). It has the same RMSNorm -> Attention -> RMSNorm -> SwiGLU structure, the same RoPE positional encoding, and the same optional GQA.

The difference is in what it predicts: instead of next-word tokens from a 32,000-word vocabulary, it predicts pairs of RVQ codebook indices from two codebooks of 128 entries each.

### Configuration (from `configs/talker_tiny.json`)

| Parameter | Value | Meaning |
|-----------|-------|---------|
| `d_model` | 384 | Hidden dimension (matches Thinker) |
| `n_layers` | 8 | Transformer blocks |
| `n_heads` | 6 | Attention heads |
| `d_ff` | 1,536 | Feedforward dimension |
| `codebooks` | 2 | Number of RVQ levels |
| `codebook_size` | 128 | Entries per codebook |
| `use_gqa` | true | Grouped Query Attention (3 KV groups) |
| `use_swiglu` | true | SwiGLU activation |
| `frame_rate` | 12.5 | Output frames per second |
| `rope_theta` | 10,000 | RoPE frequency base |

**Parameters**: ~2.24M

### How It Works

Each audio frame is represented by two integers: a **base code** and a **residual code** (as explained in Chapter 11). The Talker predicts these autoregressively -- given all previous frames, predict the next frame's codes.

```
Training (teacher forcing):

Input:   [START, frame_0, frame_1, ..., frame_{T-1}]
Target:  [frame_0, frame_1, frame_2, ..., frame_T  ]

Each frame = (base_code, residual_code), both integers in [0, 127]
```

Step by step:

```
prev_codes: (B, T, 2)         previous (base, residual) pairs
    |
    v
+-------------------------------------------+
|  Embedding lookup + sum                   |
|  emb(base_code) + emb(residual_code)      |
|  Both use same Embedding(128, 384)        |
+-------------------------------------------+
    |
    v
token_emb: (B, T, 384)
    |
    v
+-------------------------------------------+
|  Prepend START token                      |
|  start: learnable (1, 1, 384)             |
+-------------------------------------------+
    |
    v
x: (B, T+1, 384)              [START, emb_0, emb_1, ..., emb_{T-1}]
    |
    v
+===================================+
|  Block 0                          |
|  RMSNorm -> Attn(RoPE,GQA) -> +  |
|  RMSNorm -> SwiGLU FFN -> +      |
+===================================+
    |  ... 7 more blocks ...
    v
+===================================+
|  Block 7                          |
+===================================+
    |
    v
+-------------------------------------------+
|  RMSNorm (final)                          |
+-------------------------------------------+
    |
    v
x: (B, T+1, 384)
    |
    v  Remove START token position, keep T positions
x: (B, T, 384)
    |
    +---> base_head:  Linear(384, 128) => base_logits  (B, T, 128)
    |
    +---> res_head:   Linear(384, 128) => res_logits   (B, T, 128)
```

### Teacher Forcing

During training, the model sees the ground-truth codes shifted by one position (standard autoregressive training). At position `t`, it uses the ground-truth code at position `t-1` as input and tries to predict the code at position `t`.

### Masked Loss

Not all frames in a batch are real audio -- shorter utterances are padded. The training loss uses `mel_lengths` to create a mask that excludes padding frames from the loss computation, preventing the model from wasting capacity on predicting silence.

### KV Caching

Like the Thinker, the Talker supports KV caching for fast autoregressive inference. Generate one frame at a time, appending K/V to the cache rather than recomputing from scratch.

---

## Part 2: The RVQ Codec

The RVQ (Residual Vector Quantization) codec bridges the gap between the Talker's discrete codes and continuous mel spectrograms. As covered in Chapter 11, RVQ works by:

1. Projecting a mel frame (128-dim) to a codebook dimension (64-dim)
2. Finding the nearest vector in codebook 1 (base)
3. Computing the residual (original - quantized)
4. Finding the nearest vector in codebook 2 (residual)

### Encoding (mel to codes)

```
mel frame: (B, 128)
    |
    v
proj_in: Linear(128, 64) => z: (B, 64)
    |
    v
Codebook 0 (128 vectors, each 64-dim):
    nearest neighbor => base_idx
    residual = z - codebook_0[base_idx]
    |
    v
Codebook 1 (128 vectors, each 64-dim):
    nearest neighbor on residual => res_idx
    |
    v
codes: (B, 2) = [base_idx, res_idx]
```

### Decoding (codes to mel)

```
codes: [base_idx, res_idx]
    |
    v
z = codebook_0[base_idx] + codebook_1[res_idx]    sum the vectors
    |
    v
proj_out: Linear(64, 128) => mel_reconstructed: (B, 128)
```

**Parameters**: ~49K (two codebook embeddings + two linear projections)

The RVQ handles both single frames `(B, 128)` and batched frame sequences `(B, T, 128)` by reshaping internally.

---

## Part 3: Vocoders

The final step converts a mel spectrogram to an audible waveform. Two options are available:

### HiFi-GAN (Neural Vocoder)

A GAN-based neural network that produces high-quality audio. It learns the mapping from mel spectrograms to raw waveforms through adversarial training.

**Generator architecture**:

```
mel: (B, 128, T_mel)
    |
    v
conv_pre: Conv1d(128, 256, k=7) => (B, 256, T)
    |
    v
+------------------------------------------+
| Upsample Block 0: ConvTranspose1d        |
|   (256 -> 128, rate=8x)                  |
|   + MRF (Multi-Receptive Field Fusion)   |
|     ResBlock(k=3, dil=[1,2])             |
|     ResBlock(k=5, dil=[1,2])             |
|     ResBlock(k=7, dil=[1,2])             |
|     average all three                    |
+------------------------------------------+
    |
    v  (B, 128, T*8)
+------------------------------------------+
| Upsample Block 1: rate=8x               |
+------------------------------------------+
    |
    v  (B, 64, T*64)
+------------------------------------------+
| Upsample Block 2: rate=2x               |
+------------------------------------------+
    |
    v  (B, 32, T*128)
+------------------------------------------+
| Upsample Block 3: rate=2x               |
+------------------------------------------+
    |
    v  (B, 16, T*256)
+------------------------------------------+
| conv_post: Conv1d(16, 1, k=7) + Tanh    |
+------------------------------------------+
    |
    v
audio: (B, T*256)    total upsample = 8*8*2*2 = 256x
                      matches hop_length=256
```

The total upsampling factor (256) matches the STFT hop length, so a mel spectrogram with T frames produces T*256 audio samples.

**Multi-Receptive Field (MRF) blocks**: Each upsample stage uses three parallel residual blocks with different kernel sizes (3, 5, 7) and dilations. These capture patterns at different temporal scales and are averaged together. Like three listeners focusing on syllable-level, word-level, and phrase-level patterns simultaneously.

**Discriminators** (training only):

Two discriminators provide adversarial training signal:

- **Multi-Period Discriminator (MPD)**: Reshapes audio into 2D with periods [2, 3, 5] to detect artifacts at different periodic scales. Like checking a record for scratches at different rotation speeds.

- **Multi-Scale Discriminator (MSD)**: Processes audio at multiple resolutions using average pooling. The first sub-discriminator uses spectral normalization for training stability.

### Griffin-Lim (Fallback Vocoder)

A classical signal processing algorithm that requires no training:

1. Convert mel spectrogram to linear spectrogram using pseudo-inverse of the mel filterbank
2. Initialize random phase
3. Iteratively refine phase using STFT -> magnitude replacement -> iSTFT
4. Converge in ~32 iterations

Quality is lower than HiFi-GAN (robotic / metallic artifacts) but requires zero training and works immediately.

```python
# Simplified Griffin-Lim loop:
for i in range(32):
    stft = STFT(waveform)
    stft_mag_replaced = target_magnitude * exp(j * angle(stft))
    waveform = iSTFT(stft_mag_replaced)
```

---

## End-to-End TTS Example

```
Input text: "The weather is nice today"

1. Tokenize: [450, 14826, 338, 7575, 9826]  (5 tokens)

2. Thinker generates response tokens + hidden states
   hidden: (1, 5, 384)

3. Talker (autoregressive):
   Step 0: [START] -> predict frame_0 codes: [42, 17]
   Step 1: [START, frame_0] -> predict frame_1 codes: [89, 3]
   Step 2: [START, frame_0, frame_1] -> predict frame_2: [7, 91]
   ...
   Step 24: predict frame_24 (about 2 seconds at 12.5 Hz)

   Output codes: (1, 25, 2)

4. RVQ decode:
   For each frame, look up codebook vectors and sum:
   mel_frame_0 = proj_out(codebook_0[42] + codebook_1[17])
   mel_frame_1 = proj_out(codebook_0[89] + codebook_1[3])
   ...
   Output: (1, 25, 128) mel spectrogram

5. Vocoder (HiFi-GAN):
   mel (1, 128, 25) -> upsample 256x -> audio (1, 6400)
   6400 samples at 16kHz = 0.4 seconds

   Or Griffin-Lim:
   mel -> pseudo-inverse -> Griffin-Lim iterations -> audio
```

---

## Training Details

### Talker Training

- Loss: cross-entropy on both base and residual code predictions
- The two losses are summed: `loss = CE(base_logits, base_targets) + CE(res_logits, res_targets)`
- Padding frames are masked out using `mel_lengths`
- Uses the same optimizer setup as the Thinker (AdamW, cosine schedule with warmup)

### HiFi-GAN Training

A multi-loss adversarial training procedure:

| Loss | Weight | Purpose |
|------|--------|---------|
| Mel reconstruction | 45.0 | L1 distance between input and reconstructed mel |
| Feature matching | 2.0 | L1 distance between discriminator intermediate features |
| Adversarial (generator) | 1.0 | Fool the discriminators |
| Adversarial (discriminator) | 1.0 | Distinguish real from generated |

The mel reconstruction weight starts at 45.0 and decays to 22.5 after 10,000 steps, gradually shifting emphasis from reconstruction to adversarial quality.

---

## File Reference

- **Talker**: `omni/talker.py` -- `TalkerTiny` class (~2.24M params)
- **Codec**: `omni/codec.py` -- `RVQ` class (~49K params)
- **Vocoder**: `omni/codec.py` -- `HiFiGANVocoder`, `GriffinLimVocoder`, `NeuralVocoder`
- **Discriminators**: `omni/codec.py` -- `MultiPeriodDiscriminator`, `MultiScaleDiscriminator`
- **Configs**: `configs/talker_tiny.json`, `configs/vocoder_tiny.json`

---

*Next: Chapter 17 covers the OCR Model -- a dedicated encoder-decoder for reading text from images.*
