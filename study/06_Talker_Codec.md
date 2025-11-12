# Talker & Codec: Generating Speech

## Overview

**Talker** generates speech from text by predicting audio codes, which are then decoded into audio waveforms.

**RVQ Codec** converts between continuous audio (mel spectrograms) and discrete codes (like text tokens).

Together, they enable **Text-to-Speech (TTS)**.

## Why Discrete Codes?

Just like text uses tokens (words), speech uses codes:

```
Text:  "Hello" → [1234, 5678]
Speech: Audio → [45, 23, 67, 12, ...]
```

**Benefits**:
- Thinker can generate codes (like text tokens)
- Efficient representation
- Enables autoregressive generation

## RVQ Codec

### What is RVQ?

**Residual Vector Quantization** - A multi-stage quantization method:

```
Stage 1: Quantize input
Stage 2: Quantize residual (error)
Stage 3: Quantize residual of residual
...
```

### Architecture

```
Mel Spectrogram (T×128)
    ↓
Project to Codebook Dimension (T×64)
    ↓
┌─────────────────────────┐
│ Codebook 0 (128 codes)  │ ← Quantize input
└───────────┬─────────────┘
            │
      Residual (error)
            ↓
┌─────────────────────────┐
│ Codebook 1 (128 codes)  │ ← Quantize residual
└───────────┬─────────────┘
            │
      Codes: [code0, code1]
```

### Encoding Process

```python
# Simplified RVQ encoding
def encode(mel_frame):
    # Project to codebook dimension
    x = project(mel_frame)  # (64,)
    
    # Stage 1: Find closest code in codebook 0
    code0 = find_nearest(x, codebook0)  # Index: 0-127
    residual = x - codebook0[code0]
    
    # Stage 2: Find closest code in codebook 1 for residual
    code1 = find_nearest(residual, codebook1)  # Index: 0-127
    
    return [code0, code1]  # Two codes per frame
```

### Decoding Process

```python
# Simplified RVQ decoding
def decode(codes):
    code0, code1 = codes
    
    # Lookup embeddings
    emb0 = codebook0[code0]
    emb1 = codebook1[code1]
    
    # Sum embeddings
    quantized = emb0 + emb1
    
    # Project back to mel
    mel = project_back(quantized)  # (128,)
    return mel
```

### Why Residual?

**Residual quantization** captures fine details:

```
Input:  [1.0, 2.0, 3.0, 4.0]
Code0:  [1.0, 2.0, 3.0, 3.9]  ← Approximate
Residual: [0.0, 0.0, 0.0, 0.1] ← Fine detail
Code1:  [0.0, 0.0, 0.0, 0.1]  ← Captures detail
```

## Talker Model

### Architecture

**Talker** is similar to Thinker but predicts audio codes instead of text tokens:

```
Previous Codes (B, T, 2)
    ↓
Code Embeddings
    ↓
Transformer Blocks (same as Thinker)
    ↓
Output Heads (Base + Residual)
    ↓
Code Predictions (B, T, 2)
```

### Code Embeddings

```python
# Embed each codebook separately
base_emb = base_embedding(codes[:, :, 0])    # (B, T, d_model)
res_emb = res_embedding(codes[:, :, 1])      # (B, T, d_model)

# Sum embeddings
x = base_emb + res_emb
```

### Autoregressive Generation

Generate codes one frame at a time:

```python
def generate_audio(max_frames=200):
    codes = torch.zeros(1, 1, 2)  # Start with zeros
    
    for _ in range(max_frames):
        # Predict next codes
        base_logits, res_logits = talker(codes)
        
        # Get most likely codes
        base_code = argmax(base_logits[:, -1, :])
        res_code = argmax(res_logits[:, -1, :])
        
        # Append to sequence
        next_codes = [[[base_code, res_code]]]
        codes = torch.cat([codes, next_codes], dim=1)
    
    return codes
```

### Training

**Teacher Forcing**: Use ground truth previous codes:

```python
# Training: use actual codes
mel = load_audio("example.wav")
codes = rvq.encode(mel)  # Ground truth codes

# Shift by one position (predict current from previous)
prev_codes = torch.roll(codes, 1, dims=1)
prev_codes[:, 0, :] = 0  # First frame is zero

# Predict
base_logits, res_logits = talker(prev_codes)

# Loss on both codebooks
loss = cross_entropy(base_logits, codes[:, :, 0]) + \
       cross_entropy(res_logits, codes[:, :, 1])
```

## Vocoder: Griffin-Lim

### What is a Vocoder?

Converts mel spectrogram to audio waveform.

**Griffin-Lim** is a classical (non-neural) vocoder:

```
Mel Spectrogram (T×128)
    ↓
Convert to Linear Spectrogram
    ↓
Initialize Random Phase
    ↓
Iterative Refinement (32 iterations)
    ↓
Inverse STFT
    ↓
Audio Waveform
```

### Why Griffin-Lim?

- **No training required** - works out of the box
- **Simple** - easy to understand
- **Fast** - quick generation
- **Trade-off**: Lower quality than neural vocoders

### Process

```python
def griffin_lim(mel, n_iter=32):
    # Convert mel to linear spectrogram
    mag_spec = mel_to_linear(mel)
    
    # Initialize random phase
    phase = random_phase(mag_spec.shape)
    
    # Iterative refinement
    for _ in range(n_iter):
        # Phase → Time domain
        audio = istft(mag_spec, phase)
        # Time domain → Phase
        _, phase = stft(audio)
    
    return audio
```

## Complete TTS Pipeline

```
Text: "Hello world"
    ↓
Tokenizer
    ↓
Token IDs: [1234, 5678]
    ↓
Thinker (optional - can condition on text)
    ↓
Talker (autoregressive)
    ↓
RVQ Codes: [[45, 23], [67, 12], ...]
    ↓
RVQ Decode
    ↓
Mel Spectrogram (T×128)
    ↓
Griffin-Lim Vocoder
    ↓
Audio Waveform
    ↓
Save as WAV file
```

## Code Structure

### RVQ Codec

```python
# From omni/codec.py

class RVQ(nn.Module):
    def __init__(self, num_codebooks, codebook_size, d):
        # Codebooks (embeddings)
        self.codebooks = nn.ModuleList([
            nn.Embedding(codebook_size, d)
            for _ in range(num_codebooks)
        ])
        
        # Projections
        self.proj_in = nn.Linear(128, d)
        self.proj_out = nn.Linear(d, 128)
    
    def encode(self, mel):
        # Project
        x = self.proj_in(mel)
        
        # Greedy quantization
        codes = []
        residual = x
        for codebook in self.codebooks:
            # Find nearest
            distances = (residual.unsqueeze(-2) - codebook.weight).norm(dim=-1)
            code = distances.argmin(dim=-1)
            codes.append(code)
            
            # Update residual
            residual = residual - codebook(code)
        
        return torch.stack(codes, dim=-1)  # (B, T, num_codebooks)
    
    def decode(self, codes):
        # Sum codebook embeddings
        quantized = sum(
            codebook(codes[:, :, i])
            for i, codebook in enumerate(self.codebooks)
        )
        
        # Project back
        mel = self.proj_out(quantized)
        return mel
```

### Talker

```python
# From omni/talker.py

class TalkerTiny(nn.Module):
    def __init__(self, d_model, n_layers, ...):
        # Code embeddings
        self.base_emb = nn.Embedding(codebook_size, d_model)
        self.res_emb = nn.Embedding(codebook_size, d_model)
        
        # Start token
        self.start_token = nn.Parameter(torch.randn(1, 1, d_model))
        
        # Transformer blocks (same as Thinker)
        self.blocks = nn.ModuleList([...])
        
        # Output heads
        self.base_head = nn.Linear(d_model, codebook_size)
        self.res_head = nn.Linear(d_model, codebook_size)
    
    def forward(self, codes):
        # Embed codes
        base_emb = self.base_emb(codes[:, :, 0])
        res_emb = self.res_emb(codes[:, :, 1])
        x = base_emb + res_emb
        
        # Add start token
        x = torch.cat([self.start_token.expand(B, -1, -1), x], dim=1)
        
        # Process
        for block in self.blocks:
            x = block(x)
        
        # Predict
        base_logits = self.base_head(x)
        res_logits = self.res_head(x)
        
        return base_logits, res_logits
```

## Configuration

From `configs/talker_tiny.json`:

```json
{
  "d_model": 192,
  "n_layers": 4,
  "n_heads": 3,
  "codebooks": 2,
  "codebook_size": 128,
  "frame_rate": 12.5
}
```

## Frame Rate

At 12.5 Hz frame rate:
- 1 second = 12.5 frames
- 10 seconds = 125 frames
- Each frame = 2 codes (base + residual)

**Code sequence length**:
- 10 seconds audio = 125 frames = 250 codes total

## Training Data

From `data/audio/tts.csv`:
```csv
text,wav
"Hello world",data/audio/wav/000000.wav
"How are you",data/audio/wav/000001.wav
```

**Process**:
1. Load audio → mel spectrogram
2. Encode mel → RVQ codes
3. Train Talker to predict codes autoregressively

## Common Issues

### 1. Repetitive Speech

**Problem**: Model generates same codes repeatedly

**Solution**: 
- Better training data
- Temperature sampling (add randomness)
- Longer training

### 2. Audio Quality

**Problem**: Griffin-Lim produces robotic speech

**Solution**: 
- Use neural vocoder (requires training)
- Better codebook training
- More codebooks

### 3. Generation Speed

**Problem**: Autoregressive generation is slow

**Solution**:
- KV caching
- Parallel decoding (for some models)
- Shorter sequences

## Performance Tips

1. **Batch Encoding**: Encode multiple frames at once
2. **KV Caching**: Cache attention during generation
3. **Codebook Size**: Balance quality vs. memory
4. **Frame Rate**: Lower = fewer codes but less detail

---

**Next:**
- [07_Training_Workflow.md](07_Training_Workflow.md) - How to train Talker
- [08_Inference_Guide.md](08_Inference_Guide.md) - Using TTS in inference
- [09_Hands_On_Exercises.md](09_Hands_On_Exercises.md) - Practice exercises

**See Also:**
- [Architecture Overview](02_Architecture_Overview.md)
- [Audio Encoder](04_Audio_Encoder.md)

