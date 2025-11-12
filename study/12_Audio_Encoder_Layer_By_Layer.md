# Audio Encoder: Complete Layer-by-Layer Breakdown

## Overview

This document explains **every single layer** in the Audio Encoder, combining **deep theoretical understanding** with **practical implementation**. We'll explore the **why** behind each design choice and **what value** it provides.

## Theoretical Foundation: Why This Architecture?

### The Audio Understanding Problem

Audio is fundamentally different from text:
- **Continuous**: Waveform has infinite possible values
- **Temporal**: Time dimension is critical
- **High-dimensional**: Raw audio is very high-dimensional (16,000 samples/second)
- **Variable length**: Different audio clips have different durations

### Why Mel Spectrograms?

**Raw waveform problems**:
- Too high-dimensional (16,000 values per second)
- Phase information is complex
- Hard for neural networks to process directly

**Mel spectrogram benefits**:
- **Lower dimensionality**: 128 frequency bins vs 16,000 samples
- **Phase removed**: Only magnitude (easier to learn)
- **Human-aligned**: Mel scale matches human perception
- **Time-frequency representation**: Captures both temporal and spectral patterns

### Why Downsampling?

**The frame rate problem**:
- Mel spectrogram: 100 frames/second (from 16kHz audio, 160 sample hop)
- Too many frames for transformer (would consume all context)
- Need to reduce temporal resolution

**The solution**: Convolutional downsampling
- Reduces frames from 100 Hz → 12.5 Hz (8× reduction)
- Maintains frequency information
- Aligns with Qwen3 Omni's frame rate

### Why Transformer Encoder?

**Encoder vs Decoder**:
- **Encoder**: Bidirectional (can see all frames)
- **Decoder**: Causal (only previous frames)
- For ASR: Need to see full audio context → Encoder

**Why not CNN?**
- CNNs have limited receptive field
- Transformers have global receptive field (all frames)
- Better for long-range dependencies in speech

## Layer-by-Layer Deep Dive

## Complete Architecture

```
Input: Audio Waveform (16kHz)
    ↓
[Mel Spectrogram] → (B, T, 128)
    ↓
[Reshape] → (B, 1, T, 128)  # Add channel dimension
    ↓
┌─────────────────────────────────┐
│  ConvDown (Downsampling)        │
│  ┌───────────────────────────┐ │
│  │ Conv2D (stride=2)         │ │
│  │ GELU Activation           │ │
│  │ Conv2D (stride=2)         │ │
│  │ GELU Activation           │ │
│  │ [Optional: Conv2D stride=2]│ │
│  └───────────────────────────┘ │
└───────────────┬─────────────────┘
                ↓
    (B, 64, T/8, 128/8)  # 8x downsample
    ↓
[Flatten & Reshape] → (B, T/8, 64*16)
    ↓
[Projection] → (B, T/8, 192)
    ↓
┌─────────────────────────────────┐
│  Encoder Block 1                │
│  ┌───────────────────────────┐ │
│  │ RMSNorm                   │ │
│  │ Multi-Head Self-Attention │ │
│  │ Residual                  │ │
│  │ RMSNorm                   │ │
│  │ MLP (GELU)                │ │
│  │ Residual                  │ │
│  └───────────────────────────┘ │
└───────────────┬─────────────────┘
                ↓
    [Repeat for N blocks (default: 4)]
                ↓
[Final RMSNorm] → (B, T/8, 192)
    ↓
Output: Frame Embeddings
```

## Layer 1: Mel Spectrogram Conversion

### Purpose
Convert raw audio waveform to frequency representation.

### Implementation (in training script)

```python
# From train_audio_enc.py
mel_spec = torchaudio.transforms.MelSpectrogram(
    sample_rate=16000,
    n_fft=1024,        # FFT window size
    hop_length=160,    # Step size (10ms)
    win_length=400,    # Window size (25ms)
    n_mels=128         # Number of mel bins
)

# Convert audio
mel = mel_spec(audio)[0].T  # (T, 128)
```

### What Happens

1. **Input**: Audio waveform `(1, 16000)` for 1 second
2. **STFT**: Short-Time Fourier Transform
3. **Mel Scale**: Convert to mel scale (human perception)
4. **Output**: Mel spectrogram `(100, 128)`
   - 100 frames (at 100 Hz: 16000/160)
   - 128 frequency bins

### Frame Rate Calculation

```
Sample rate: 16,000 samples/second
Hop length: 160 samples
Frame rate: 16,000 / 160 = 100 frames/second
```

## Layer 2: Reshape for Convolution

```python
# From AudioEncoderTiny.forward
def forward(self, mel):  # mel: (B, T, 128)
    # Add channel dimension for Conv2D
    x = mel[:, None, :, :]  # (B, 1, T, 128)
    # B = batch size
    # 1 = channel (grayscale)
    # T = time frames
    # 128 = frequency bins
```

## Layer 3: Convolutional Downsampling

### Purpose
Reduce temporal resolution from 100 Hz to 12.5 Hz (8x reduction).

### Implementation

```python
# From omni/audio_encoder.py
class ConvDown(nn.Module):
    def __init__(self, in_ch=1, mid=64):
        super().__init__()
        self.net = nn.Sequential(
            # First 2x downsampling
            nn.Conv2d(in_ch, mid, 3, stride=2, padding=1),
            nn.GELU(),
            # Second 2x downsampling
            nn.Conv2d(mid, mid, 3, stride=2, padding=1),
            nn.GELU()
        )
```

### Step-by-Step

#### Step 3.1: First Conv2D

```python
# Input: (B, 1, T, 128)
conv1 = nn.Conv2d(
    in_channels=1,
    out_channels=64,
    kernel_size=3,
    stride=2,      # 2x downsampling
    padding=1
)

# Forward
x = conv1(x)  # (B, 1, T, 128) → (B, 64, T/2, 64)
# Time: T → T/2 (2x reduction)
# Frequency: 128 → 64 (2x reduction)
```

**What Conv2D Does**:
```
Input: (B, 1, 100, 128)
       ↓
Apply 3×3 filter with stride=2
       ↓
Output: (B, 64, 50, 64)
```

#### Step 3.2: GELU Activation

```python
# GELU: Gaussian Error Linear Unit
def gelu(x):
    return 0.5 * x * (1 + torch.erf(x / math.sqrt(2)))

x = gelu(x)  # (B, 64, T/2, 64)
```

#### Step 3.3: Second Conv2D

```python
conv2 = nn.Conv2d(
    in_channels=64,
    out_channels=64,
    kernel_size=3,
    stride=2,
    padding=1
)

x = conv2(x)  # (B, 64, T/2, 64) → (B, 64, T/4, 32)
# Time: T/2 → T/4 (another 2x)
# Frequency: 64 → 32 (another 2x)
```

#### Step 3.4: Optional Third Conv2D (for 8x total)

```python
# For 8x downsample (12.5 Hz target)
if downsample_factor == 8:
    conv3 = nn.Conv2d(64, 64, 3, stride=2, padding=1)
    x = gelu(conv3(x))  # (B, 64, T/4, 32) → (B, 64, T/8, 16)
```

### Total Downsampling

```
Input:  (B, 1, 100, 128)  # 100 Hz, 128 bins
After 4x: (B, 64, 25, 32)   # 25 Hz, 32 bins
After 8x: (B, 64, 12.5, 16) # 12.5 Hz, 16 bins
```

## Layer 4: Flatten and Reshape

```python
# From AudioEncoderTiny.forward
B, C, T, F = x.shape  # (B, 64, T/8, 16)

# Flatten channels and frequency
x = x.permute(0, 2, 1, 3)  # (B, T/8, 64, 16)
x = x.contiguous().view(B, T, C * F)  # (B, T/8, 64*16)
# Result: (B, T/8, 1024)
```

### What Happens

- **Before**: `(B, 64, T/8, 16)` - 4D tensor
- **After**: `(B, T/8, 1024)` - 3D tensor
- Each time frame now has 1024 features (64 channels × 16 frequency bins)

## Layer 5: Projection to Model Dimension

```python
# From AudioEncoderTiny.__init__
self.proj = nn.Linear(64 * (128 // downsample_factor), d)
# For 8x downsample: 64 * 16 = 1024 → 192

# Forward
x = self.proj(x)  # (B, T/8, 1024) → (B, T/8, 192)
```

### Linear Layer Details

```python
# Linear transformation: y = xW^T + b
# x: (B, T/8, 1024)
# W: (192, 1024) - weight matrix
# b: (192,) - bias (if used)
# y: (B, T/8, 192)
```

## Layer 6: Encoder Blocks

### Block Structure

```python
# From omni/audio_encoder.py
class EncoderBlock(nn.Module):
    def __init__(self, d, heads, ff, dropout=0.1):
        super().__init__()
        self.norm1 = RMSNorm(d)
        # Multi-head self-attention (encoder style - bidirectional)
        self.attn = nn.MultiheadAttention(
            d, heads, 
            dropout=dropout, 
            batch_first=True
        )
        self.norm2 = RMSNorm(d)
        # MLP with GELU
        self.mlp = nn.Sequential(
            nn.Linear(d, ff),
            nn.GELU(),
            nn.Linear(ff, d)
        )
        self.drop = nn.Dropout(dropout)
```

### Deep Dive: Why Bidirectional Attention?

**Encoder vs Decoder attention**:

**Decoder (causal)**:
- Can only see previous positions
- Realistic for generation
- But can't use future context

**Encoder (bidirectional)**:
- Can see all positions (past and future)
- Better for understanding
- Uses full context

**Why bidirectional for audio?**
- **ASR task**: Need full audio context to transcribe
- **No generation**: Not generating audio, just understanding
- **Better accuracy**: Future context helps disambiguate

**Example**:
- Audio: "I can't hear you"
- Without future context: "I can't" might be misheard
- With future context: "hear you" helps disambiguate "can't"

### Why GELU Instead of SwiGLU?

**Audio encoder uses GELU, Thinker uses SwiGLU**:

**GELU** (Gaussian Error Linear Unit):
- Simpler (single projection)
- Sufficient for encoder task
- Less parameters (efficiency)

**SwiGLU** (used in Thinker):
- More expressive (gating)
- Better for generation tasks
- More parameters

**Why different?**
- **Encoder**: Understanding task, GELU sufficient
- **Decoder**: Generation task, SwiGLU better
- **Efficiency**: Encoder can be simpler

### What Value Do We Get from Encoder Blocks?

1. **Bidirectional Understanding**: Uses full audio context
2. **Long-Range Dependencies**: Captures relationships across time
3. **Parallel Processing**: All frames processed simultaneously
4. **Efficiency**: Simpler than decoder (no causal masking)
5. **Modularity**: Can stack multiple blocks for depth

### Block Forward Pass

```python
def forward(self, x):
    # x shape: (B, T, d_model)
    
    # Pre-norm + Attention + Residual
    h = self.attn(
        self.norm1(x),  # Query
        self.norm1(x),  # Key (same as query for self-attention)
        self.norm1(x),  # Value (same as query)
        need_weights=False
    )[0]
    x = x + self.drop(h)  # Residual connection
    
    # Pre-norm + MLP + Residual
    x = x + self.drop(self.mlp(self.norm2(x)))
    
    return x
```

### Step-by-Step: Attention

#### 6.1: Normalize

```python
x_norm = self.norm1(x)  # (B, T, 192)
```

#### 6.2: Self-Attention

```python
# Multi-head self-attention
# Q = K = V = x_norm (self-attention)
attn_output, _ = self.attn(
    x_norm,  # Query
    x_norm,  # Key
    x_norm,  # Value
    need_weights=False
)
# Output: (B, T, 192)
```

**What Self-Attention Does**:
- Each frame can attend to **all other frames** (bidirectional)
- Learns relationships between different time frames
- Example: Frame at 0.5s can attend to frame at 1.0s

#### 6.3: Residual Connection

```python
x = x + self.drop(attn_output)  # (B, T, 192)
```

#### 6.4: MLP

```python
# MLP: Linear → GELU → Linear
mlp_output = self.mlp(self.norm2(x))  # (B, T, 192)
x = x + self.drop(mlp_output)  # Residual
```

### MLP Details

```python
# MLP structure
self.mlp = nn.Sequential(
    nn.Linear(192, 768),  # Expand
    nn.GELU(),            # Activation
    nn.Linear(768, 192)   # Contract
)

# Forward
x = linear1(x)    # (B, T, 192) → (B, T, 768)
x = gelu(x)       # (B, T, 768)
x = linear2(x)    # (B, T, 768) → (B, T, 192)
```

## Layer 7: Repeat Blocks

```python
# From AudioEncoderTiny.__init__
self.blocks = nn.ModuleList([
    EncoderBlock(d, heads, ff, dropout) 
    for _ in range(layers)  # Default: 4 blocks
])

# Forward
for blk in self.blocks:
    x = blk(x)  # Each block: (B, T/8, 192) → (B, T/8, 192)
```

## Layer 8: Final Normalization

```python
# From AudioEncoderTiny.forward
x = self.norm(x)  # Final RMSNorm
# Output: (B, T/8, 192)
```

## Complete Forward Pass with Shapes

```python
# Input: Audio waveform
audio = torch.randn(1, 16000)  # 1 second at 16kHz

# Step 1: Mel spectrogram
mel = mel_spec(audio)[0].T  # (100, 128)
mel = mel.unsqueeze(0)      # (1, 100, 128) - add batch

# Step 2: Reshape
x = mel[:, None, :, :]  # (1, 1, 100, 128)

# Step 3: ConvDown (8x)
x = conv_down(x)  # (1, 1, 100, 128) → (1, 64, 12.5, 16)
# Note: 12.5 frames (rounded to 13 in practice)

# Step 4: Flatten
x = x.permute(0, 2, 1, 3).contiguous()  # (1, 13, 64, 16)
x = x.view(1, 13, 64 * 16)  # (1, 13, 1024)

# Step 5: Project
x = proj(x)  # (1, 13, 1024) → (1, 13, 192)

# Step 6: Encoder blocks
x = block1(x)  # (1, 13, 192)
x = block2(x)  # (1, 13, 192)
x = block3(x)  # (1, 13, 192)
x = block4(x)  # (1, 13, 192)

# Step 7: Final norm
x = norm(x)  # (1, 13, 192)

# Output: Frame embeddings at 12.5 Hz
```

## Key Parameters

From `configs/audio_enc_tiny.json`:

```json
{
  "d_model": 192,          // Embedding dimension
  "n_layers": 4,           // Number of encoder blocks
  "n_heads": 3,            // Attention heads
  "d_ff": 768,             // MLP feedforward dimension
  "downsample_time": 8,    // Temporal downsampling factor
  "target_hz": 12.5        // Target frame rate
}
```

## Frame Rate Calculation

```
Input mel: 100 Hz (16000 / 160)
After 8x downsample: 100 / 8 = 12.5 Hz
Output: 12.5 frames per second
```

**Why 12.5 Hz?**
- Matches Qwen3 Omni's frame rate
- Good balance: enough detail, not too many tokens
- ~80ms per frame (human speech phonemes are ~100-200ms)

## Memory and Computation

### Per Block:
- **Parameters**: ~0.5M (d_model=192, n_heads=3, d_ff=768)
- **Memory**: O(B * T * d_model) for activations
- **Computation**: O(B * T² * d_model) for attention

### Total Model:
- **Parameters**: ~20M (4 blocks + conv + proj)
- **Input**: 100 frames/second
- **Output**: 12.5 frames/second (8x reduction)

---

**Next:**
- [11_Thinker_Layer_By_Layer.md](11_Thinker_Layer_By_Layer.md) - Thinker layers
- [13_Vision_Encoder_Layer_By_Layer.md](13_Vision_Encoder_Layer_By_Layer.md) - Vision layers
- [04_Audio_Encoder.md](04_Audio_Encoder.md) - Audio encoder overview

