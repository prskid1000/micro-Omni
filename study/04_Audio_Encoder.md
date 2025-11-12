# Audio Encoder: Understanding Speech

## What is the Audio Encoder?

The **Audio Encoder** converts raw audio waveforms into embeddings that Thinker can understand.

Think of it as "ears" for the AI - it processes sound and extracts meaningful features.

## Audio Basics

### Waveform

Audio is a sequence of amplitude values over time:

```
Time: 0ms    10ms   20ms   30ms
      |       |      |      |
      ▼       ▼      ▼      ▼
Amplitude: [0.1, 0.5, -0.3, 0.8, ...]
```

### Sample Rate

Number of samples per second:
- **16 kHz**: 16,000 samples/second (μOmni uses this)
- **44.1 kHz**: CD quality
- **48 kHz**: Professional audio

### Mel Spectrogram

A visual representation of audio showing frequency content over time:

```
Frequency
    ↑
    |  ████
    |  ████  ████
    |  ████  ████  ████
    |  ████  ████  ████
    └──────────────────→ Time
```

**Why Mel?**: Human ears perceive frequencies logarithmically - mel scale matches this.

## Architecture

```
Audio Waveform (16kHz)
    ↓
Mel Spectrogram (100Hz, 128 bins)
    ↓
Convolutional Downsampling (8x)
    ↓
Frame Rate: 12.5 Hz
    ↓
Transformer Encoder
    ↓
Frame Embeddings (T×192)
```

## Step-by-Step Processing

### 1. Mel Spectrogram Conversion

```python
# From train_audio_enc.py
mel_spec = torchaudio.transforms.MelSpectrogram(
    sample_rate=16000,
    n_fft=1024,        # FFT window size
    hop_length=160,    # Step size (10ms)
    win_length=400,    # Window size (25ms)
    n_mels=128         # Frequency bins
)

# Convert audio to mel
mel = mel_spec(audio)  # Shape: (128, T)
mel = mel.T            # Shape: (T, 128) - time first
```

**Result**: 
- Input: 1 second audio = 16,000 samples
- Output: 100 frames × 128 mel bins

### 2. Convolutional Downsampling

Reduces temporal resolution from 100 Hz to 12.5 Hz (8x reduction):

```python
# Simplified downsampling
def downsample_8x(mel):
    # First 2x: (T, 128) → (T/2, 128)
    x = conv2d(mel, stride=2)
    # Second 2x: (T/2, 128) → (T/4, 128)
    x = conv2d(x, stride=2)
    # Third 2x: (T/4, 128) → (T/8, 128)
    x = conv2d(x, stride=2)
    return x  # 8x reduction total
```

**Why 12.5 Hz?**
- Matches Qwen3 Omni's frame rate
- Good balance: enough detail, not too slow
- ~80ms per frame (human speech is ~100-200ms per phoneme)

### 3. Projection to Model Dimension

```python
# Flatten and project
x = x.flatten()  # (T/8, H*W)
x = linear(x, d_model)  # (T/8, 192)
```

### 4. Transformer Encoder

Processes sequence of frames:

```python
# Encoder blocks (similar to Thinker but no masking)
for block in encoder_blocks:
    x = block(x)  # Self-attention + MLP
```

**Key Difference from Thinker**:
- **Encoder**: Can see all positions (bidirectional)
- **Decoder**: Only sees previous positions (causal)

### 5. Final Normalization

```python
x = rms_norm(x)  # Final normalization
```

## Training: ASR (Automatic Speech Recognition)

The audio encoder is trained to transcribe speech:

```
Audio → Encoder → CTC Head → Text
```

### CTC (Connectionist Temporal Classification)

Handles alignment between audio frames and text:

**Problem**: Audio has many frames, text has few tokens
- Audio: 100 frames
- Text: "hello" (5 characters)

**Solution**: CTC allows multiple frames per character and handles alignment automatically.

```python
# Simplified CTC
frames = encoder(audio)  # (100, 192)
logits = ctc_head(frames)  # (100, vocab_size)

# CTC loss handles alignment
loss = ctc_loss(logits, "hello")
```

### Training Data

From `data/audio/asr.csv`:
```csv
wav,text
data/audio/wav/000000.wav,"hello world"
data/audio/wav/000001.wav,"how are you"
```

## Code Structure

```python
# From omni/audio_encoder.py

class AudioEncoderTiny(nn.Module):
    def __init__(self, d_model, n_heads, ...):
        # Convolutional downsampling
        self.conv_down = ConvDown(downsample_factor)
        
        # Projection
        self.proj = nn.Linear(flattened_size, d_model)
        
        # Transformer encoder blocks
        self.blocks = nn.ModuleList([
            EncoderBlock(d_model, n_heads, ...)
            for _ in range(n_layers)
        ])
        
        # Final norm
        self.norm = RMSNorm(d_model)
    
    def forward(self, mel):
        # Downsample
        x = self.conv_down(mel)
        
        # Project
        x = self.proj(x.flatten(...))
        
        # Encode
        for block in self.blocks:
            x = block(x)
        
        # Normalize
        x = self.norm(x)
        return x
```

## Configuration

From `configs/audio_enc_tiny.json`:

```json
{
  "sample_rate": 16000,
  "mel_bins": 128,
  "downsample_time": 8,
  "target_hz": 12.5,
  "d_model": 192,
  "n_layers": 4,
  "n_heads": 3,
  "d_ff": 768
}
```

## Frame Rate Calculation

```
Input: 16,000 samples/second
Mel: 100 frames/second (hop_length=160)
After 8x downsample: 12.5 frames/second
```

**Why this matters**: 
- Lower frame rate = fewer tokens for Thinker
- Saves context length
- Still captures speech content

## Multimodal Integration

After encoding, audio embeddings are projected to Thinker's dimension:

```python
# Audio encoder output
audio_emb = audio_encoder(mel)  # (T, 192)

# Project to Thinker dimension
audio_emb = audio_projector(audio_emb)  # (T, 256)

# Now Thinker can process it!
thinker_input = torch.cat([image_emb, audio_emb, text_emb], dim=1)
output = thinker(thinker_input)
```

## Common Issues

### 1. Audio Too Long

**Problem**: Audio exceeds context length

**Solution**: Truncate or chunk
```python
max_audio_tokens = ctx_len // 4
audio_emb = audio_emb[:max_audio_tokens]
```

### 2. Sample Rate Mismatch

**Problem**: Audio not 16kHz

**Solution**: Resample
```python
if sr != 16000:
    audio = torchaudio.functional.resample(audio, sr, 16000)
```

### 3. Silent Audio

**Problem**: No speech detected

**Solution**: Check amplitude
```python
if audio.abs().max() < 0.01:
    print("Warning: Audio too quiet")
```

## Visual Guide

```
Audio File (WAV)
    ↓
[Load] → Waveform (16,000 samples/sec)
    ↓
[Mel Spectrogram] → (100 frames/sec, 128 bins)
    ↓
[Conv Downsample 8x] → (12.5 frames/sec, 128 bins)
    ↓
[Flatten & Project] → (T frames, 192 dims)
    ↓
[Transformer Encoder] → (T frames, 192 dims)
    ↓
[RMSNorm] → Final embeddings
    ↓
[Projector] → (T frames, 256 dims) → Thinker
```

## Training Example

```python
# Load audio and text
audio, sr = torchaudio.load("example.wav")
text = "hello world"

# Convert to mel
mel = mel_spec(audio)

# Encode
embeddings = audio_encoder(mel)

# For ASR training: predict text
logits = ctc_head(embeddings)
loss = ctc_loss(logits, text)
```

## Performance Tips

1. **Batch Processing**: Process multiple audio files together
2. **Padding**: Pad to same length in batch
3. **Normalization**: Normalize audio amplitude
4. **Augmentation**: Add noise, speed changes for robustness

---

**Next:**
- [05_Vision_Encoder.md](05_Vision_Encoder.md) - How images are processed
- [06_Talker_Codec.md](06_Talker_Codec.md) - How speech is generated
- [07_Training_Workflow.md](07_Training_Workflow.md) - Training the encoder

**See Also:**
- [Architecture Overview](02_Architecture_Overview.md)
- [Inference Guide](08_Inference_Guide.md)

