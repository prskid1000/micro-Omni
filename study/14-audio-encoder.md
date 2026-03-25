# Chapter 14: Audio Encoder (AuT-Tiny)

The Audio Encoder is the system's ear. It takes raw speech -- represented as a mel spectrogram -- and converts it into a sequence of 384-dimensional embeddings that the Thinker can process alongside text and image tokens.

Think of it like a simultaneous interpreter at the United Nations: they listen to speech in one language and produce a stream of meaning in a universal language that everyone understands.

---

## Role

The Audio Encoder (AuT-Tiny = Audio Transformer Tiny) has two jobs:

1. **CTC mode** (default): Produce a sequence of frame embeddings for automatic speech recognition. Each frame represents ~80ms of audio. The CTC loss (Connectionist Temporal Classification) aligns these frames to text characters without requiring frame-level labels.

2. **Contrastive mode** (CLAP-style): Produce a single pooled embedding representing the entire audio clip. Used for audio-text matching tasks (similar to how CLIP works for images, covered in Chapter 15).

---

## Configuration (from `configs/audio_enc_tiny.json`)

| Parameter | Value | Meaning |
|-----------|-------|---------|
| `d_model` | 384 | Hidden dimension |
| `n_layers` | 8 | Transformer encoder layers |
| `n_heads` | 6 | Attention heads |
| `d_ff` | 1,536 | Feedforward dimension |
| `mel_bins` | 128 | Mel spectrogram frequency bins |
| `sample_rate` | 16,000 | Audio sample rate (16kHz) |
| `frame_hop` | 160 | Hop length (10ms at 16kHz) |
| `downsample_time` | 8 | Temporal downsampling factor |
| `target_hz` | 12.5 | Output frame rate (100Hz / 8) |
| `dropout` | 0.1 | Dropout rate |

**Parameters**: ~2.05M

---

## Full Pipeline with Tensor Shapes

```
Audio waveform (16kHz)
    |
    v
Mel Spectrogram extraction (not part of the model -- done in preprocessing)
    |
    v
mel: (B, T_mel, 128)          e.g., 10 seconds => T_mel = 1000 frames at 100Hz
    |
    v  reshape to image-like format for Conv2d
(B, 1, T_mel, 128)            1 channel, T_mel time frames, 128 frequency bins
    |
    v
+-------------------------------------------+
|  ConvDown (8x temporal downsample)        |
|                                           |
|  Conv2d(1, 64, k=3, s=2, pad=1) + GELU   |  => (B, 64, T/2, 64)
|  Conv2d(64, 64, k=3, s=2, pad=1) + GELU  |  => (B, 64, T/4, 32)
|  Conv2d(64, 64, k=3, s=2, pad=1) + GELU  |  => (B, 64, T/8, 16)
|                                           |
+-------------------------------------------+
    |
    v
(B, 64, T/8, 16)              64 channels, T/8 time steps, 16 freq bins
    |
    v  Reshape: permute(0,2,1,3) then flatten last two dims
(B, T/8, 64*16) = (B, T/8, 1024)
    |
    v
+-------------------------------------------+
|  Linear Projection (1024 -> 384)          |
+-------------------------------------------+
    |
    v
(B, T/8, 384)                 Now in the shared 384-dim embedding space
    |
    v
+===================================+
|  Transformer Encoder Block 0      |
|  RMSNorm -> Self-Attention -> +res|  (bidirectional -- no causal mask)
|  RMSNorm -> GELU MLP -> +res     |
+===================================+
    |
    v
+===================================+
|  Blocks 1 ... 7 (same structure)  |
+===================================+
    |
    v
+-------------------------------------------+
|  RMSNorm (final)                          |
+-------------------------------------------+
    |
    v
(B, T/8, 384)                 Frame sequence output (CTC mode)
    |
    +---> CTC Mode: return full sequence for ASR
    |
    +---> Contrastive Mode: AttentionPooling -> (B, 384) single vector
```

---

## The ConvDown Module

The ConvDown is the most critical design choice in the audio encoder. Raw mel spectrograms at 100Hz produce too many tokens -- 10 seconds would be 1000 frames, consuming the entire context window. Downsampling reduces this to a manageable rate.

```
Input mel: 100 Hz (one frame per 10ms)

Conv2d stride=2:  100 / 2 = 50 Hz
Conv2d stride=2:   50 / 2 = 25 Hz
Conv2d stride=2:   25 / 2 = 12.5 Hz    (with downsample_factor=8)
```

At 12.5 Hz, 10 seconds of audio produces 125 tokens -- fitting comfortably within the 256-token context.

Each Conv2d layer downsamples in **both** time and frequency dimensions simultaneously:
- Time: 1000 -> 500 -> 250 -> 125
- Frequency: 128 -> 64 -> 32 -> 16

The frequency dimension is then flattened with the channel dimension and projected to 384.

Why not use a 4x downsample? You could (set `downsample_factor=4` for 25Hz), but 8x is preferred because:
- Fewer tokens = faster attention (quadratic cost)
- 12.5Hz still captures phoneme-level detail (~80ms per frame, roughly one phoneme)
- Matches the Talker's frame rate, simplifying the TTS pipeline

---

## Encoder Blocks vs Thinker Blocks

The audio encoder uses `EncoderBlock`, not the Thinker's `Block`. Key differences:

| Feature | Thinker Block | Audio EncoderBlock |
|---------|--------------|-------------------|
| Attention | Causal (masked) | Bidirectional (no mask) |
| RoPE | Yes | No (uses standard attention) |
| QKV projection | Fused or separate (GQA) | Single fused `qkv_proj` |
| FFN | SwiGLU | Standard GELU MLP |
| Bias | No bias | Has bias in QKV and output |

The encoder uses bidirectional attention because it processes complete audio inputs -- unlike the Thinker which generates tokens left-to-right, the encoder can look at the entire audio clip at once, letting early frames attend to later frames and vice versa.

Both support Flash Attention for the same 2-4x speedup.

---

## CTC Mode: Frame Sequence for ASR

In the default mode (`use_attention_pooling=False`), the encoder outputs a sequence of frame embeddings: `(B, T/8, 384)`.

For ASR training, a linear CTC head (not part of the encoder itself -- added by the training script) projects these to character probabilities:

```
encoder output:   (B, T/8, 384)
CTC head:         Linear(384, num_characters)  => (B, T/8, C)
CTC loss:         aligns frame predictions to target text
```

The beauty of CTC: you do NOT need frame-level alignment labels. You provide:
- Input: mel spectrogram of "hello"
- Target: the string "hello"

CTC figures out which frames correspond to which characters by marginalizing over all valid alignments. As covered in Chapter 5, this uses a special blank token to handle repeated characters and silence.

---

## Contrastive Mode: CLAP-Style Pooling

When `use_attention_pooling=True`, the frame sequence is compressed to a single vector using learned attention pooling:

```
frames: (B, T/8, 384)
    |
    v
AttentionPooling:
    weights = softmax(Linear(384 -> 1))     per-frame importance
    pooled = sum(frames * weights, dim=time)
    |
    v
(B, 384)                                    single audio embedding
```

This mode is used for CLAP (Contrastive Language-Audio Pretraining), where audio and text embeddings are trained to be similar for matching pairs -- the same principle as CLIP for images (Chapter 15).

---

## Frame Rate Math

Understanding the frame rate is important for connecting audio duration to token count:

```
Sample rate:     16,000 Hz
Hop length:      160 samples
Mel frame rate:  16000 / 160 = 100 Hz (one mel frame per 10ms)
Conv downsample: 8x
Output rate:     100 / 8 = 12.5 Hz (one token per 80ms)

Duration -> Tokens:
  1 second  =>  12.5 tokens  =>  ~13 tokens
  5 seconds =>  62.5 tokens  =>  ~63 tokens
 10 seconds => 125.0 tokens  => 125 tokens
 20 seconds => 250.0 tokens  => 250 tokens (nearly fills ctx=256)
```

---

## Integration with the Thinker

After the audio encoder produces `(B, T/8, 384)`, these embeddings are concatenated with text embeddings along the sequence dimension and fed to the Thinker:

```
audio_emb:  (B, 125, 384)    # 10 seconds of audio
text_emb:   (B,  20, 384)    # "Transcribe this audio:"
combined:   (B, 145, 384)    # concatenated
                 |
                 v
            Thinker processes combined sequence
                 |
                 v
            generates transcription tokens
```

The Thinker's causal attention means each generated text token can attend to all audio frames (they come first in the sequence) -- it "listens" to the entire audio before responding.

---

## File Reference

- **Source**: `omni/audio_encoder.py`
- **Config**: `configs/audio_enc_tiny.json`
- **Classes**: `AudioEncoderTiny`, `ConvDown`, `EncoderBlock`, `AttentionPooling`
- **Parameters**: ~2.05M

---

*Next: Chapter 15 covers the Vision Encoder -- how images become the embeddings that the Thinker understands, trained with CLIP-style contrastive learning.*
