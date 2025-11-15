# Unified Dry Run Computation: μOmni Pipeline

## 🎯 Key Takeaways (TL;DR)

- **What**: Step-by-step trace of data flow through entire μOmni pipeline
- **Why**: Understand exactly how inputs transform at each stage (crucial for debugging)
- **How**: All modalities → unified tokens → Thinker → output generation
- **Key Insight**: Everything becomes tokens in 256-dim space - image (1 token), audio (25 tokens), text (variable)
- **Common Mistake**: Not tracking token counts or shape mismatches between stages
- **Shape Tracking**: Always verify `(batch, tokens, dim)` format matches expected

**📖 Reading Guide**:
- **Quick Read**: 20 minutes (overview + one pipeline example)
- **Standard Read**: 60 minutes (all pipelines)
- **Deep Dive**: 2 hours (read + trace through code with debugger)

## Overview: Token-Based Processing Pipeline

The μOmni system processes multimodal inputs by converting everything to a **unified token representation**:

```
Text:   Direct tokenization → token IDs
Image:  Image → ViT patches → CLS token → projected to token space
Audio:  Audio → Mel → AudioEncoder → projected to token space
All:    Combined into single sequence → Thinker processes → generates output
```

### Core Principle: Unified Token Space

All modalities are converted to **256-dimensional embeddings** that Thinker can process uniformly:

| Modality | Input | Processing | Output Tokens | Dimension |
|----------|-------|------------|---------------|-----------|
| **Text** | String | Tokenize → Embed | Variable (e.g., 6) | 256 |
| **Image** | (224×224×3) | ViT → CLS → Project | 1 | 256 |
| **Audio** | (16k samples) | Mel → Encoder → Project | ~25 | 256 |

**Key Point**: Once in token space, Thinker treats all modalities the same!

---

## Part 1: Text-Only Pipeline

**Use Case**: Simple text generation (chat, completion)

### Input: Text String
```
Input: "Hello, how are you?"
```

### Shape Flow Diagram

```
Text: "Hello, how are you?"
  ↓
Tokenize: [1, 234, 567, 890, 123, 456]  (6 tokens)
  ↓
Embed: (1, 6, 256)  [batch=1, tokens=6, dim=256]
  ↓
Transformer Blocks (4 layers): (1, 6, 256) → (1, 6, 256)
  ↓
LM Head: (1, 6, 256) → (1, 6, 5000)
  ↓
Generate: Extract last position → 1 new token
  ↓
Autoregressive: 6 → 7 → 8 → ... → 70 tokens
  ↓
Decode: Text string output
```

### Step 1: Tokenization (Text → Token IDs)
**What it does**: Converts text string to sequence of integer token IDs

```python
# Input: "Hello, how are you?"
# Process: BPETokenizer.encode()
ids = [1] + tok.encode("Hello, how are you?")
# Output: [1, 234, 567, 890, 123, 456]
#         ↑  ↑    ↑    ↑    ↑    ↑
#        BOS token1 token2 token3 token4 token5
# Total: 6 token IDs
```

**Transformation**: 
- **Input**: String `"Hello, how are you?"` (variable length)
- **Output**: List of integers `[1, 234, 567, 890, 123, 456]` (6 tokens)
- **BOS token** (1) prepended for sequence start
- **Shape**: `(6,)` - 1D list of token IDs

**Try It Yourself**:
```python
from omni.tokenizer import BPETokenizer
tok = BPETokenizer("checkpoints/thinker_tiny/tokenizer.model")
text = "Hello, how are you?"
ids = tok.encode(text)
print(f"Text: {text}")
print(f"Token IDs: {ids}")
print(f"Number of tokens: {len(ids)}")
# Expected: Token IDs: [234, 567, 890, 123, 456] (example)
# Expected: Number of tokens: 5 (plus BOS = 6 total)
```

### Step 2: Token Embedding (Token IDs → Embeddings)
**What it does**: Maps each token ID to a dense vector representation

```python
# Input: [1, 234, 567, 890, 123, 456] (6 token IDs)
# Process: Embedding lookup
x = think.tok_emb(torch.tensor([[1, 234, 567, 890, 123, 456]]))
# Output: (1, 6, 256)
#         ↑  ↑  ↑
#        batch tokens d_model
```

**Transformation**:
- **Input**: `(1, 6)` token IDs - batch of 1, 6 tokens
- **Output**: `(1, 6, 256)` embeddings - same batch, same tokens, now 256-dim vectors
- Each token ID → 256-dimensional vector (learned embedding)
- **6 input tokens → 6 embedding vectors** (count preserved, dimension added)

**Key Point**: Token count stays the same (6), we just add the embedding dimension (256)

### Step 3: Position Embeddings (RoPE)
**What it does**: Adds positional information to embeddings using Rotary Position Embedding

```python
# Input: (1, 6, 256) embeddings
# Process: Generate positions and apply RoPE
pos = [0, 1, 2, 3, 4, 5]  # Position indices for 6 tokens
# RoPE rotates Q and K by frequency-dependent angles based on position
# Applied in each attention block
```

**Transformation**:
- Input: `(1, 6, 256)` embeddings (no position info)
- Output: `(1, 6, 256)` embeddings (with position encoded via RoPE)
- **6 tokens maintain their positions** through rotation matrices

### Step 4: Transformer Blocks (4 layers)
**What it does**: Processes embeddings through attention and MLP layers

**Note**: In training, forward passes use AMP (automatic mixed precision) for faster computation:
```python
# Training mode (with AMP):
with autocast():  # FP16 forward pass
    x = block(x)  # Faster, uses less memory

# Inference mode (with AMP):
with autocast():  # FP16 forward pass
    logits = model(x)  # Faster inference
```

#### Layer 1:
```python
# Input: (1, 6, 256)
# Block 1 (with AMP in training/inference):
#   - RMSNorm: (1, 6, 256) → (1, 6, 256) [normalize]
#   - Attention: (1, 6, 256) → (1, 6, 256)
#     * Q, K, V: (1, 6, 256) → (1, 4 heads, 6, 64)
#     * Attention scores: (1, 4, 6, 6) [each token attends to all 6]
#     * Output: (1, 6, 256) [weighted combination]
#   - MLP (SwiGLU): (1, 6, 256) → (1, 6, 256)
# Output: (1, 6, 256)
# Note: AMP uses FP16 internally, but output is cast back to FP32 if needed
```

**Transformation per layer**:
- Input: `(1, 6, 256)` - 6 token embeddings
- Attention: Each of 6 tokens attends to all 6 tokens (including itself)
- MLP: Each token processed independently
- Output: `(1, 6, 256)` - 6 transformed embeddings
- **6 input tokens → 6 output embeddings** (same count, transformed)

#### Layers 2-4: Same process
```python
# Each layer: (1, 6, 256) → (1, 6, 256)
# After 4 layers: (1, 6, 256)
```

**Cumulative transformation**:
- **6 input tokens → 6 embeddings** (count preserved)
- Each token's representation enriched through 4 layers of attention

### Step 5: Final Norm and LM Head
**What it does**: Normalizes and projects to vocabulary logits

```python
# Input: (1, 6, 256) from transformer
# Process:
x = think.norm(x)  # (1, 6, 256) [final normalization]
logits = think.lm_head(x)  # (1, 6, 5000)
#                          ↑  ↑  ↑
#                         batch tokens vocab_size

# Numerical Stability Check (automatic)
# Checks for NaN/Inf in logits and raises RuntimeError if detected
```

**Transformation**:
- Input: `(1, 6, 256)` embeddings
- Output: `(1, 6, 5000)` logits
- **6 input tokens → 6 sets of vocabulary logits**
- Each position predicts probability distribution over 5000 tokens
- **Safety**: Automatic NaN/Inf detection raises error if numerical instability detected

### Step 6: Next Token Prediction
**What it does**: Samples next token from last position's logits

```python
# Input: (1, 6, 5000) logits
# Process: Extract last token's logits and sample
next_token_logits = logits[0, -1, :]  # (5000,) - logits for position 5
next_token_id = next_token_logits.argmax()  # e.g., 789
# Generated sequence: [1, 234, 567, 890, 123, 456, 789]
#                     ↑  ↑    ↑    ↑    ↑    ↑    ↑
#                    input tokens (6)    new token (1)
```

**Transformation**:
- Input: `(1, 6, 5000)` - 6 positions with vocab logits
- Output: 1 new token ID (789)
- **6 input tokens → 1 new token generated**

### Step 7: Autoregressive Generation (with KV Cache)
**What it does**: Continues generating tokens one at a time, reusing cached attention states

```python
# First call: Process all 6 input tokens
# Cache stores K/V for positions 0-5
logits = think([1, 234, 567, 890, 123, 456])  # Full forward pass
next_id = 789

# Second call: Only process new token (position 6)
# Reuse cached K/V for positions 0-5
logits = think([789])  # Only forward pass for position 6
# Cache now has K/V for positions 0-6
next_id = 101

# Continue...
# Generated: [1, 234, 567, 890, 123, 456, 789, 101, 202, ...]
#            ↑  ↑    ↑    ↑    ↑    ↑    ↑    ↑    ↑
#           6 input tokens    generated tokens
```

**Transformation**:
- **6 input tokens** processed once (full forward pass)
- **Each new token** processed incrementally (using KV cache)
- **Total sequence grows**: 6 → 7 → 8 → ... → 70 tokens

### Step 8: Decoding (Token IDs → Text)
**What it does**: Converts generated token IDs back to text string

```python
# Input: [1, 234, 567, 890, 123, 456, 789, 101, 202, ...]
# Process: BPETokenizer.decode()
output_text = tok.decode([1, 234, 567, 890, 123, 456, 789, 101, 202, ...])
# Output: "Hello, how are you? I'm doing well, thank you!"
```

**Transformation**:
- Input: List of token IDs
- Output: Text string
- **All tokens (input + generated) → final text output**

---

## Part 2: Image + Text Pipeline

**Use Case**: Visual question answering, image captioning

### Input: Image + Text String
```
Image: RGB image (224×224 pixels)
Text: "What do you see in this image?"
```

### Shape Flow Diagram

```
Image (224×224×3 = 150,528 pixels)
  ↓
Patches: (1, 196, 128)  [196 patches]
  ↓
Add CLS: (1, 197, 128)  [1 CLS + 196 patches]
  ↓
ViT Transformer: (1, 197, 128) → (1, 197, 128)
  ↓
Extract CLS: (1, 1, 128)  [Only CLS token]
  ↓
Project: (1, 1, 128) → (1, 1, 256)  [Image token]

Text: "What do you see?"
  ↓
Tokenize: [1, 123, 456, 789, 234, 567, 890, 345]  (8 tokens)
  ↓
Embed: (1, 8, 256)  [8 text tokens]

Combined:
  ↓
Concat: (1, 1, 256) + (1, 8, 256) → (1, 9, 256)  [1 img + 8 text]
  ↓
Thinker: (1, 9, 256) → (1, 9, 5000)
  ↓
Generate: Text response
```

### Step 1: Image Processing (Image → Token Embedding)
**What it does**: Converts image to a single token embedding

#### 1.1: Patch Embedding
```python
# Input: (1, 3, 224, 224) RGB image
# Process: Conv2d with kernel=16, stride=16
x = vis.proj(img)  # Conv2d(3, 128, kernel=16, stride=16)
# Output: (1, 128, 14, 14)
#         ↑  ↑    ↑  ↑
#        batch channels patches patches
# 14×14 = 196 patches
```

**Transformation**:
- Input: `(1, 3, 224, 224)` - 150,528 pixels (224×224×3)
- Output: `(1, 128, 14, 14)` - 196 patches, each 128-dimensional
- **150,528 pixels → 196 patch embeddings**

#### 1.2: Reshape and Add CLS Token
```python
# Input: (1, 128, 14, 14) - 196 patches
# Process: Reshape and add CLS token
x = rearrange(x, "b d h w -> b (h w) d")  # (1, 196, 128)
cls = vis.cls.expand(1, -1, -1)  # (1, 1, 128) - CLS token
x = torch.cat([cls, x], dim=1)  # (1, 197, 128)
#         ↑   ↑
#       CLS 196 patches
```

**Transformation**:
- Input: `(1, 196, 128)` - 196 patch embeddings
- Output: `(1, 197, 128)` - 1 CLS + 196 patches
- **196 patches → 197 tokens** (added CLS token)

#### 1.3: Position Embeddings
```python
# Input: (1, 197, 128)
# Process: Add learnable position embeddings
x = x + vis.pos[:, :197, :]  # (1, 197, 128)
```

**Transformation**:
- Input: `(1, 197, 128)` - tokens without position
- Output: `(1, 197, 128)` - tokens with position
- **197 tokens maintain positions**

#### 1.4: Transformer Processing
```python
# Input: (1, 197, 128)
# Process: 4 encoder layers
for blk in vis.blocks:
    x = blk(x)  # Self-attention + MLP
# Output: (1, 197, 128)
```

**Transformation**:
- Input: `(1, 197, 128)` - 197 token embeddings
- Output: `(1, 197, 128)` - 197 transformed embeddings
- **197 tokens → 197 embeddings** (count preserved)

#### 1.5: Extract CLS Token
```python
# Input: (1, 197, 128)
# Process: Extract first token (CLS)
cls = x[:, :1, :]  # (1, 1, 128)
# Discard: x[:, 1:, :]  # (1, 196, 128) - patch tokens
```

**Transformation**:
- Input: `(1, 197, 128)` - CLS + 196 patches
- Output: `(1, 1, 128)` - CLS token only
- **197 tokens → 1 token** (CLS represents entire image)

#### 1.6: Project to Thinker Dimension
```python
# Input: (1, 1, 128) - CLS token from ViT
# Process: Linear projection
img_emb = proj_v(cls)  # Linear(128, 256)
# Output: (1, 1, 256)
```

**Transformation**:
- Input: `(1, 1, 128)` - ViT CLS token
- Output: `(1, 1, 256)` - Thinker-compatible embedding
- **1 image → 1 token embedding** (same as text token dimension)

### Step 2: Text Processing (Same as Part 1, Steps 1-2)
```python
# Input: "What do you see in this image?"
# Tokenize: [1, 123, 456, 789, 234, 567, 890, 345]  # 8 tokens
# Embed: (1, 8, 256)
text_emb = think.tok_emb(torch.tensor([[1, 123, 456, 789, 234, 567, 890, 345]]))
```

**Transformation**:
- **8 text tokens → 8 embeddings** `(1, 8, 256)`

### Step 3: Multimodal Combination
**What it does**: Concatenates image and text embeddings into single sequence

```python
# Input: 
#   img_emb: (1, 1, 256)  - 1 image token
#   text_emb: (1, 8, 256) - 8 text tokens
# Process: Concatenate along time dimension
multimodal_emb = torch.cat([img_emb, text_emb], dim=1)
# Output: (1, 9, 256)
#         ↑  ↑  ↑
#        batch tokens d_model
#        1 image + 8 text = 9 total tokens
```

**Transformation**:
- Input: `(1, 1, 256)` image + `(1, 8, 256)` text = 9 separate embeddings
- Output: `(1, 9, 256)` - unified sequence
- **1 image token + 8 text tokens → 9 total tokens**

### Step 4: Thinker Processing (Same as Part 1, Steps 3-5)
**What it does**: Processes unified multimodal sequence through Thinker

```python
# Input: (1, 9, 256) - 9 tokens (1 image + 8 text)
# Process: 4 transformer layers + norm + LM head
logits = think(embeddings=multimodal_emb)
# Output: (1, 9, 5000)
#         ↑  ↑  ↑
#        batch tokens vocab_logits
```

**Transformation**:
- Input: `(1, 9, 256)` - 9 token embeddings
- Attention: Each of 9 tokens attends to all 9 tokens
- Output: `(1, 9, 5000)` - 9 positions with vocab logits
- **9 input tokens → 9 output logit positions**

### Step 5: Generation (Same as Part 1, Steps 6-8)
**What it does**: Generates text response based on multimodal context

```python
# Input: (1, 9, 5000) logits
# Process: Sample from last position (position 8, which is text)
next_token = logits[0, -1].argmax()  # e.g., 678
# Generated: [1, 123, ..., 345, 678, ...]
#            ↑  ↑         ↑    ↑
#           8 input tokens   new token
```

**Transformation**:
- **9 input tokens (1 image + 8 text) → generates text tokens**
- Image token provides visual context for text generation

---

## Part 3: Audio + Text Pipeline

**Use Case**: Speech recognition, audio question answering

### Input: Audio + Text String
```
Audio: Waveform (1 second @ 16kHz = 16,000 samples)
Text: "What did you hear?"
```

### Shape Flow Diagram

```
Audio (16,000 samples @ 16kHz)
  ↓
Mel Spectrogram: (1, 100, 128)  [100 frames @ 100Hz]
  ↓
Downsample 4x: (1, 100, 128) → (1, 25, 32)  [25 frames @ 25Hz]
  ↓
Reshape & Project: (1, 25, 192)  [25 audio embeddings]
  ↓
Audio Encoder: (1, 25, 192) → (1, 25, 192)
  ↓
Project: (1, 25, 192) → (1, 25, 256)  [25 audio tokens]

Text: "What did you hear?"
  ↓
Tokenize: [1, 234, 567, 890, 123]  (5 tokens)
  ↓
Embed: (1, 5, 256)  [5 text tokens]

Combined:
  ↓
Concat: (1, 25, 256) + (1, 5, 256) → (1, 30, 256)  [25 aud + 5 text]
  ↓
Thinker: (1, 30, 256) → (1, 30, 5000)
  ↓
Generate: Text response
```

### Step 1: Audio Processing (Audio → Token Embeddings)
**What it does**: Converts audio waveform to sequence of token embeddings

#### 1.1: Mel Spectrogram
```python
# Input: (1, 16000) - 1 second @ 16kHz
# Process: STFT + Mel filterbank
mel_spec = MelSpectrogram(sample_rate=16000, hop_length=160, n_mels=128)
mel = mel_spec(wav)[0].T.unsqueeze(0)
# Output: (1, 100, 128)
#         ↑  ↑   ↑
#        batch frames mel_bins
# 16000 / 160 = 100 frames @ 100 Hz
```

**Transformation**:
- Input: `(1, 16000)` - 16,000 audio samples
- Output: `(1, 100, 128)` - 100 time frames, 128 mel bins
- **16,000 samples → 100 mel frames**

#### 1.2: Audio Encoder (Conv2D Downsampling)
```python
# Input: (1, 100, 128) mel
# Process: Add channel dim and downsample
x = mel[:, None, :, :]  # (1, 1, 100, 128)
x = aud.down(x)  # Conv2D with stride=2 twice
# Conv1: (1, 1, 100, 128) → (1, 64, 50, 64)  [2x down]
# Conv2: (1, 64, 50, 64) → (1, 64, 25, 32)   [2x down]
# Total: 4x downsample in time, 4x in frequency
# Output: (1, 64, 25, 32)
```

**Transformation**:
- Input: `(1, 100, 128)` - 100 frames @ 100 Hz
- Output: `(1, 64, 25, 32)` - 25 frames @ 25 Hz (4x down)
- **100 frames → 25 frames** (time preserved: 1.0 second)

#### 1.3: Reshape and Project
```python
# Input: (1, 64, 25, 32)
# Process: Reshape and project
x = x.permute(0,2,1,3).contiguous().view(1, 25, 64*32)  # (1, 25, 2048)
x = aud.proj(x)  # Linear(2048, 192)
# Output: (1, 25, 192)
```

**Transformation**:
- Input: `(1, 64, 25, 32)` - spatial features
- Output: `(1, 25, 192)` - sequence embeddings
- **25 frames → 25 embeddings**

#### 1.4: Transformer Processing
```python
# Input: (1, 25, 192)
# Process: 4 encoder blocks
for blk in aud.blocks:
    x = blk(x)  # Self-attention + MLP
# Output: (1, 25, 192)
```

**Transformation**:
- Input: `(1, 25, 192)` - 25 frame embeddings
- Output: `(1, 25, 192)` - 25 transformed embeddings
- **25 frames → 25 embeddings** (count preserved)

#### 1.5: Project to Thinker Dimension
```python
# Input: (1, 25, 192) - AudioEncoder output
# Process: Linear projection
audio_emb = proj_a(x)  # Linear(192, 256)
# Output: (1, 25, 256)
# Limit: min(25, ctx_len//4) = min(25, 128) = 25
audio_emb = audio_emb[:, :25, :]  # (1, 25, 256)
```

**Transformation**:
- Input: `(1, 25, 192)` - audio embeddings
- Output: `(1, 25, 256)` - Thinker-compatible embeddings
- **25 audio frames → 25 token embeddings**

### Step 2: Text Processing
```python
# Input: "What did you hear?"
# Tokenize: [1, 234, 567, 890, 123]  # 5 tokens
# Embed: (1, 5, 256)
text_emb = think.tok_emb(torch.tensor([[1, 234, 567, 890, 123]]))
```

**Transformation**:
- **5 text tokens → 5 embeddings** `(1, 5, 256)`

### Step 3: Multimodal Combination
```python
# Input:
#   audio_emb: (1, 25, 256) - 25 audio tokens
#   text_emb: (1, 5, 256)   - 5 text tokens
# Process: Concatenate
multimodal_emb = torch.cat([audio_emb, text_emb], dim=1)
# Output: (1, 30, 256)
#         1 image + 25 audio + 5 text = 30 total tokens
```

**Transformation**:
- **25 audio tokens + 5 text tokens → 30 total tokens**

### Step 4-5: Thinker Processing and Generation
**Same as Part 2, Steps 4-5**
- **30 input tokens → generates text response**

---

## Part 4: Multimodal (Image + Audio + Text) Pipeline

### Input: Image + Audio + Text
```
Image: (224×224 RGB)
Audio: (1 second waveform)
Text: "Describe what you see and hear."
```

### Step 1-3: Process Each Modality
```python
# Image: (1, 3, 224, 224) → ViT → (1, 1, 256)  [1 token]
img_emb = proj_v(vis(img))  # (1, 1, 256)

# Audio: (1, 16000) → Mel → AudioEncoder → (1, 25, 256)  [25 tokens]
audio_emb = proj_a(aud(mel))  # (1, 25, 256)

# Text: "Describe..." → Tokenize → Embed → (1, 9, 256)  [9 tokens]
text_emb = think.tok_emb(text_ids)  # (1, 9, 256)
```

**Transformation**:
- **1 image → 1 token**
- **25 audio frames → 25 tokens**
- **9 text tokens → 9 embeddings**

### Step 4: Combine All Modalities
```python
# Input:
#   img_emb: (1, 1, 256)    - 1 token
#   audio_emb: (1, 25, 256) - 25 tokens
#   text_emb: (1, 9, 256)   - 9 tokens
# Process: Concatenate in order
multimodal_emb = torch.cat([img_emb, audio_emb, text_emb], dim=1)
# Output: (1, 35, 256)
#         1 + 25 + 9 = 35 total tokens
```

**Transformation**:
- **1 image + 25 audio + 9 text → 35 total tokens**

### Step 5: Thinker Processing
```python
# Input: (1, 35, 256) - 35 tokens
# Process: 4 transformer layers
logits = think(embeddings=multimodal_emb)
# Output: (1, 35, 5000)
```

**Transformation**:
- **35 input tokens → 35 output logit positions**
- Each token attends to all 35 tokens (including itself)
- Image and audio tokens provide context for text generation

### Step 6: Generation
```python
# Generate text based on multimodal context
# 35 input tokens → generates text tokens
```

---

## Part 5: Text-to-Speech (TTS) Pipeline

**Use Case**: Converting text responses to speech

### Input: Text String
```
Input: "This is a test of text to speech."
```

### Shape Flow Diagram

```
Text: "This is a test..."
  ↓
Tokenize: 10 token IDs
  ↓
Estimate Duration: 10 tokens → ~300 audio frames
  ↓
Talker (Autoregressive):
  Start: (1, 1, 2)  [zero codes]
  Frame 0: (1, 1, 2) → predict → (1, 2, 2)
  Frame 1: (1, 2, 2) → predict → (1, 3, 2)
  ...
  Final: (1, 301, 2)  [300 frames + start]
  ↓
RVQ Decode: (1, 301, 2) → (301, 128)  [301 mel frames]
  ↓
Vocoder: (301, 128) → (385,280,)  [audio samples @ 16kHz]
  ↓
Audio: 24 seconds of speech
```

### Step 1: Text Generation (Same as Part 1)
```python
# Generate text response first
response = generate(think, tok, "This is a test...")
# Output: "This is a test of text to speech."
# Tokenize: [234, 567, 890, 123, 456, 789, 234, 567, 890, 123]
#           ↑    ↑    ↑    ↑    ↑    ↑    ↑    ↑    ↑    ↑
#          10 tokens
```

**Transformation**:
- **Text string → 10 token IDs**

### Step 2: Estimate Audio Duration
```python
# Input: 10 text tokens
# Process: Estimate frames needed (~30 frames per token)
max_frames = max(50, min(1000, int(10 * 30)))  # = 300 frames
# Duration: 300 frames @ 12.5 Hz = 24 seconds
```

**Transformation**:
- **10 text tokens → 300 audio frames** (estimated)

### Step 3: Generate Audio Codes (Autoregressive)
**What it does**: Generates RVQ codebook indices frame-by-frame using Talker

```python
# Input: Start with zero codes
codes = torch.zeros(1, 1, 2, dtype=torch.long)  # (1, 1, 2)
#         ↑  ↑  ↑
#        batch frames codebooks

# Frame 0:
base_logit, res_logit = talker(codes)  # (1, 1, 128) each
base_code = base_logit[0, -1].argmax()  # e.g., 45
res_code = res_logit[0, -1].argmax()    # e.g., 12
codes = torch.cat([codes, [[[45, 12]]]], dim=1)  # (1, 2, 2)

# Frame 1:
base_logit, res_logit = talker(codes)  # Uses KV cache
base_code = base_logit[0, -1].argmax()  # e.g., 67
res_code = res_logit[0, -1].argmax()    # e.g., 23
codes = torch.cat([codes, [[[67, 23]]]], dim=1)  # (1, 3, 2)

# Continue for 300 frames...
# Output: (1, 301, 2)  # start token + 300 frames
```

**Transformation**:
- **1 start token → 300 frame codes**
- Each frame: 2 codebook indices (base + residual)
- **Total: 301 positions × 2 codebooks = 602 code indices**

### Step 4: Decode Codes to Mel Spectrogram
**What it does**: Converts codebook indices back to mel spectrogram frames

```python
# Input: (1, 301, 2) - 301 frames, 2 codebooks each
# Process: Decode each frame
mel_frames = []
for t in range(301):
    frame_codes = codes[:, t, :]  # (1, 2) - 2 codebook indices
    # RVQ decode: (1, 2) → (1, 128)
    mel_frame = rvq.decode(frame_codes)  # (1, 128)
    mel_frames.append(mel_frame.squeeze(0))  # (128,)

mel = torch.stack(mel_frames, dim=0)  # (301, 128)
```

**Transformation**:
- Input: `(1, 301, 2)` - 301 frames with 2 code indices each
- Process: Each frame's 2 codes → lookup embeddings → sum → project
- Output: `(301, 128)` - 301 mel spectrogram frames
- **602 code indices → 301 mel frames**

### Step 5: Vocoder (Mel → Audio Waveform)
**What it does**: Converts mel spectrogram to audio waveform using improved Griffin-Lim

```python
# Input: (301, 128) mel spectrogram
# Process: Improved Griffin-Lim with proper mel filterbank inversion
mel_np = mel.cpu().numpy()  # (301, 128)

# Improved implementation:
# 1. Proper mel filterbank inversion (pseudo-inverse approach)
# 2. Automatic domain detection (log vs magnitude)
# 3. Griffin-Lim with momentum (0.99) for better convergence
# 4. Proper normalization to prevent clipping
audio = voc.mel_to_audio(mel_np)  # (audio_samples,)
# Duration: 301 frames @ 12.5 Hz = 24.08 seconds
# Samples: 24.08 * 16000 = 385,280 samples
```

**Improvements**: Uses proper mel filterbank inversion (pseudo-inverse) instead of simple upsampling, handles both log and magnitude domains automatically, includes momentum for better convergence, and proper normalization to prevent clipping.

**Transformation**:
- Input: `(301, 128)` - 301 mel frames
- Output: `(385280,)` - audio waveform samples
- **301 mel frames → 385,280 audio samples**

### Complete TTS Pipeline Summary
```
Text: "This is a test..."
  → 10 token IDs
  → 300 estimated audio frames
  → 300 frame codes (600 code indices)
  → 301 mel frames (301×128 = 38,528 values)
  → 385,280 audio samples
```

---

## Part 6: Training Pipeline (Talker)

**Use Case**: Training Talker to generate audio codes from mel spectrograms

### Input: Mel Spectrogram Batch
```
Input: (B=2, T=50, 128) - 2 samples, 50 frames, 128 mel bins
```

### Shape Flow Diagram

```
Mel Batch: (2, 50, 128)  [2 samples, 50 frames each]
  ↓
RVQ Encode: (2, 50, 128) → (2, 50, 2)  [ground truth codes]
  ↓
Shift for Training: (2, 50, 2) → (2, 50, 2)  [previous frames]
  ↓
Talker Forward: (2, 50, 2) → (2, 50, 128) × 2  [base + residual logits]
  ↓
Loss: Compare predictions vs targets
  ↓
Backward: Update Talker weights
```

### Step 1: RVQ Encoding (Batch)
**What it does**: Encodes all mel frames to codebook indices in batch

```python
# Input: (2, 50, 128) - 2 samples, 50 frames each
# Process: Batch encode all frames
idxs = rvq.encode(mel)  # (2, 50, 2)
#         ↑  ↑  ↑
#        batch frames codebooks
# 2 samples × 50 frames × 2 codebooks = 200 code indices
```

**Transformation**:
- Input: `(2, 50, 128)` - 100 mel frames total (2×50)
- Output: `(2, 50, 2)` - 100 code pairs
- **100 mel frames → 100 code pairs** (200 total indices)

### Step 2: Shift for Autoregressive Training
**What it does**: Creates shifted sequence for next-token prediction

```python
# Input: (2, 50, 2) - ground truth codes
# Process: Shift by 1 position, zero first frame
prev = torch.roll(idxs, 1, dims=1)  # Shift right
prev[:, 0, :] = 0  # Zero first frame
# Output: (2, 50, 2)
# Example:
#   idxs[0]:  [45,12], [67,23], [89,34], [12,45], ...
#   prev[0]:  [0,0],   [45,12], [67,23], [89,34], ...
#             ↑        ↑        ↑        ↑
#            zero    frame0   frame1   frame2
```

**Transformation**:
- Input: `(2, 50, 2)` - ground truth codes
- Output: `(2, 50, 2)` - shifted codes (previous frames)
- **50 frames → 50 shifted frames** (for predicting next frame)

### Step 3: Talker Forward Pass
**What it does**: Predicts next codebook indices from previous codes

```python
# Input: (2, 50, 2) - previous codes
# Process: Talker transformer
base_logit, res_logit = talker(prev)
# Output: 
#   base_logit: (2, 50, 128) - logits for base codebook
#   res_logit: (2, 50, 128)  - logits for residual codebook
```

**Transformation**:
- Input: `(2, 50, 2)` - 50 frames of previous codes
- Output: `(2, 50, 128)` each - 50 positions predicting 128 codebook entries
- **50 input frames → 50 prediction positions**

### Step 4: Loss Calculation
**What it does**: Computes cross-entropy loss between predictions and targets

```python
# Input:
#   base_logit: (2, 50, 128) - predictions
#   res_logit: (2, 50, 128)   - predictions
#   idxs: (2, 50, 2)          - ground truth
# Process: Reshape and compute loss
base_flat = base_logit.reshape(-1, 128)  # (100, 128)
res_flat = res_logit.reshape(-1, 128)    # (100, 128)
base_target = idxs[:, :, 0].reshape(-1)  # (100,) - base codebook targets
res_target = idxs[:, :, 1].reshape(-1)   # (100,) - residual codebook targets

loss = F.cross_entropy(base_flat, base_target) + \
       F.cross_entropy(res_flat, res_target)
```

**Transformation**:
- Input: `(2, 50, 128)` predictions, `(2, 50, 2)` targets
- Reshape: `(100, 128)` predictions, `(100,)` targets
- **100 prediction positions → 100 loss values → 1 scalar loss**

---

## Summary: Token Flow Through Pipeline

### Text-Only:
```
Text string
  → 6 token IDs
  → 6 embeddings (1, 6, 256)
  → 6 embeddings after 4 layers (1, 6, 256)
  → 6 vocab logits (1, 6, 5000)
  → 1 new token generated
  → 7 tokens total
  → ... (autoregressive)
  → 70 tokens total
  → Text string output
```

### Image + Text:
```
Image (224×224×3 = 150,528 pixels)
  → 196 patch embeddings
  → 197 tokens (CLS + patches)
  → 1 CLS token (1, 1, 128)
  → 1 token embedding (1, 1, 256)

Text: "What do you see?"
  → 8 token IDs
  → 8 embeddings (1, 8, 256)

Combined:
  → 9 tokens (1, 9, 256)
  → 9 vocab logits (1, 9, 5000)
  → Generates text response
```

### Audio + Text:
```
Audio (16,000 samples)
  → 100 mel frames (1, 100, 128)
  → 25 encoder frames (1, 25, 192)
  → 25 token embeddings (1, 25, 256)

Text: "What did you hear?"
  → 5 token IDs
  → 5 embeddings (1, 5, 256)

Combined:
  → 30 tokens (1, 30, 256)
  → 30 vocab logits (1, 30, 5000)
  → Generates text response
```

### Multimodal (Image + Audio + Text):
```
Image → 1 token
Audio → 25 tokens
Text → 9 tokens
Combined → 35 tokens (1, 35, 256)
  → 35 vocab logits
  → Generates text response
```

### TTS:
```
Text: "This is a test..."
  → 10 token IDs
  → 300 estimated frames
  → 300 frame codes (600 indices)
  → 301 mel frames (38,528 values)
  → 385,280 audio samples
```

---

## 📊 Summary: Token Flow Through All Pipelines

### Quick Reference Table

| Pipeline | Input | Token Count | Output |
|----------|-------|------------|--------|
| **Text-Only** | 6 text tokens | 6 → 6 → 6 → 1 new | Text response |
| **Image + Text** | 1 image + 8 text | 1 + 8 = 9 → 9 → 1 new | Text response |
| **Audio + Text** | 25 audio + 5 text | 25 + 5 = 30 → 30 → 1 new | Text response |
| **Multimodal** | 1 img + 25 aud + 9 text | 1 + 25 + 9 = 35 → 35 → 1 new | Text response |
| **TTS** | 10 text tokens | 10 → 300 codes → 301 mel → 385k samples | Audio waveform |

### Shape Transformation Summary

```
Text Pipeline:
  (1, 6) IDs → (1, 6, 256) emb → (1, 6, 256) blocks → (1, 6, 5000) logits

Image Pipeline:
  (1, 3, 224, 224) img → (1, 197, 128) patches → (1, 1, 128) CLS → (1, 1, 256) emb

Audio Pipeline:
  (1, 16000) wav → (1, 100, 128) mel → (1, 25, 192) enc → (1, 25, 256) emb

Multimodal:
  (1, 1, 256) img + (1, 25, 256) aud + (1, 9, 256) text → (1, 35, 256) combined
```

## 🔒 Numerical Stability & Safety Checks

All model forward passes now include automatic numerical stability checks:

### NaN/Inf Detection in Forward Passes

**Automatic checks in all models**:
- **ThinkerLM**: Checks logits for NaN/Inf before returning
- **TalkerTiny**: Checks base and residual logits separately
- **AudioEncoderTiny**: Checks encoder output embeddings
- **ViTTiny**: Checks CLS and grid tokens separately

**What happens**:
```python
# In ThinkerLM.forward():
logits = self.lm_head(x)  # (1, 6, 5000)

# Automatic check:
if torch.isnan(logits).any() or torch.isinf(logits).any():
    nan_count = torch.isnan(logits).sum().item()
    inf_count = torch.isinf(logits).sum().item()
    raise RuntimeError(f"Numerical instability in ThinkerLM: NaN={nan_count}, Inf={inf_count}")
```

**Benefits**:
- **Early detection**: Catches numerical issues immediately
- **Clear errors**: Detailed error messages with counts
- **Prevents corruption**: Stops training before weights are corrupted

### Mixed Precision (AMP) Safety

**AMP with automatic stability**:
- **Gradient scaling**: Prevents underflow in FP16
- **Automatic casting**: PyTorch handles precision automatically
- **Loss scaling**: Scales loss before backward to prevent underflow
- **Unscaling**: Unscales gradients before clipping (required for AMP)

**AMP workflow with gradient accumulation**:
```python
# Forward pass in FP16 (automatic)
with autocast():
    logits = model(x)  # Computed in FP16
    loss = loss_fn(logits, targets)

# Scale loss for gradient accumulation
loss = loss / accumulation_steps

# Backward with scaling
scaler.scale(loss).backward()  # Scales gradients, accumulates

# Gradient accumulation: only step every N steps
if (step + 1) % accumulation_steps == 0:
    scaler.unscale_(opt)  # Unscales before clipping
    clip_gradients(model, max_grad_norm)  # Clip in FP32
    scaler.step(opt)  # Updates weights
    scaler.update()  # Updates scaler
    opt.zero_grad()  # Clear gradients after stepping
```

**Benefits of AMP**:
- **Speed**: 1.5-2x faster training and inference
- **Memory**: ~50% less VRAM usage
- **Stability**: Gradient scaling prevents underflow
- **Quality**: Minimal impact on model performance

### Loss Validation in Training

**All training scripts validate losses**:
```python
# In training loops:
loss = loss_fn(logits, targets)

# Automatic validation:
validate_loss(loss, min_loss=-1e6, max_loss=1e6)
# Raises RuntimeError if loss is NaN/Inf or out of bounds
```

**What it checks**:
- Loss is not NaN or Inf
- Loss is within reasonable bounds (default: -1e6 to 1e6)
- Skips batch if invalid (prevents training corruption)

### Gradient Explosion Detection

**All training scripts check gradients** (with proper AMP and gradient accumulation handling):
```python
# Scale loss for gradient accumulation
loss = loss / accumulation_steps

# Backward pass with AMP
if use_amp:
    scaler.scale(loss).backward()
else:
    loss.backward()

# Gradient accumulation: only check/step every N steps
if (step + 1) % accumulation_steps == 0:
    if use_amp:
        scaler.unscale_(opt)  # CRITICAL: Unscale BEFORE checking gradients
    
    # Check for gradient explosion (after unscaling if AMP)
    grad_norm, is_exploded = check_gradient_explosion(model, max_grad_norm=100.0)
    if is_exploded:
        logger.error(f"Gradient explosion: grad_norm={grad_norm:.2f}")
        opt.zero_grad()  # Clear gradients
        if use_amp:
            scaler.update()  # Update scaler (unscale was called, so update is safe)
        continue  # Skip this batch
    
    # Gradient clipping and optimizer step
    clip_gradients(model, max_grad_norm)
    if use_amp:
        scaler.step(opt)
        scaler.update()
    else:
        opt.step()
    opt.zero_grad()  # Clear after stepping
```

**What it checks**:
- Gradient norm exceeds threshold (default: 100.0)
- Gradient norm is NaN
- Automatically skips problematic batches

**Critical for AMP**: 
- **Must unscale BEFORE checking** - Gradients are in scaled space after `scaler.scale(loss).backward()`
- **Scaled gradients appear much larger** - Checking before unscaling causes false positives (e.g., 200k+ instead of <10)
- **Unscale immediately after backward** - Then check gradients in real space
- **Update scaler when skipping** - Since `unscale()` was called, `scaler.update()` is safe and necessary

**Important**: When skipping a batch due to gradient explosion:
- **Do call `scaler.unscale_(opt)`** - Before checking gradients (if using AMP)
- **Do call `scaler.update()`** - After unscaling, even if skipping (unscale records inf checks)
- **Do call `opt.zero_grad()`** - Clear the problematic gradients

**Benefits**:
- **Prevents training collapse**: Catches exploding gradients early
- **Automatic recovery**: Skips bad batches, continues training
- **Detailed logging**: Reports gradient norms for debugging
- **Accurate detection**: Unscaling before checking prevents false positives from scaled gradients
- **Scaler safety**: Proper update sequence prevents AssertionError

## ⚠️ Common Pitfalls

1. **Shape Mismatches**: Always verify dimensions match
   ```python
   # WRONG: Mismatched dimensions
   img_emb = (1, 1, 128)  # ViT output
   thinker(img_emb)  # Expects (1, T, 256) - ERROR!
   
   # CORRECT: Project first
   img_emb = proj_v(img_emb)  # (1, 1, 128) → (1, 1, 256)
   thinker(img_emb)  # Now works!
   ```

2. **Token Count Overflow**: Total tokens must fit in context length
   ```python
   # Check total tokens
   total_tokens = img_tokens + audio_tokens + text_tokens
   assert total_tokens <= ctx_len, f"Too many tokens: {total_tokens} > {ctx_len}"
   ```

3. **Forgetting BOS Token**: Always prepend BOS for text
   ```python
   # WRONG: Missing BOS
   ids = tok.encode(text)  # [234, 567, ...]
   
   # CORRECT: Include BOS
   ids = [1] + tok.encode(text)  # [1, 234, 567, ...]
   ```

4. **Wrong Batch Dimension**: Ensure batch dimension is first
   ```python
   # WRONG: No batch dimension
   x = torch.tensor([1, 234, 567])  # (3,)
   
   # CORRECT: Add batch dimension
   x = torch.tensor([[1, 234, 567]])  # (1, 3)
   ```

## ✅ Understanding Checkpoint

Before moving on, can you answer:

1. **How many tokens does a 224×224 image become?**
   - Answer: 1 token (CLS token after ViT processing)

2. **How many tokens does 1 second of audio become?**
   - Answer: ~25 tokens (after downsampling from 100 mel frames)

3. **What's the maximum total tokens for multimodal input?**
   - Answer: ctx_len (e.g., 512), with audio limited to ctx_len//4

4. **Why do we preserve token count in transformer blocks?**
   - Answer: Attention and MLP operate per-token, preserving sequence length

5. **What happens to token count during generation?**
   - Answer: Grows incrementally (6 → 7 → 8 → ...) as new tokens are generated

6. **When are checkpoints saved during training?**
   - Answer: Every `checkpoint_freq` steps (default: 500), when validation improves (best model), and at end. Training automatically resumes from latest checkpoint.

7. **What training improvements have been made?**
   - Answer: Vision uses contrastive learning (CLIP-style), audio uses full character vocabulary (98 tokens), gradient accumulation support, improved Griffin-Lim vocoder, automatic resume functionality

## Key Insights

1. **Unified Token Representation**: All modalities (text, image, audio) are converted to token embeddings of the same dimension (256), allowing them to be processed together.

2. **Token Count Preservation**: In most transformer operations, the number of tokens is preserved:
   - Attention: N tokens → N tokens
   - MLP: N tokens → N tokens
   - Only pooling/extraction reduces token count (e.g., ViT CLS extraction)

3. **Context Length Limits**: Total tokens (multimodal + text) must fit within `ctx_len` (e.g., 512):
   - Image: 1 token
   - Audio: Limited to `ctx_len // 4` (e.g., 128 tokens max)
   - Text: Remaining space

4. **Autoregressive Generation**: Each new token is generated based on all previous tokens (input + generated), with KV caching for efficiency.

5. **Batch Processing**: Training processes multiple samples in parallel, but token counts per sample are independent.

6. **Mixed Precision (AMP)**: All forward passes use FP16 automatically for 1.5-2x speedup:
   - Training: Forward in FP16, backward with gradient scaling
   - Inference: Forward in FP16 for faster generation
   - Memory: ~50% less VRAM usage
   - Quality: Minimal impact on model performance

7. **Gradient Accumulation**: All training scripts support accumulating gradients over multiple steps before updating weights, allowing larger effective batch sizes when memory is limited.

8. **Automatic Checkpointing**: Checkpoints saved every `checkpoint_freq` steps (default: 500), when validation improves, and at end. Training automatically resumes from latest checkpoint.

9. **Proper Algorithms**: 
   - Vision: Contrastive learning (CLIP-style) with InfoNCE loss for proper image-caption alignment
   - Audio: Full character vocabulary (98 tokens) with proper CTC tokenization
   - Vocoder: Improved Griffin-Lim with proper mel filterbank inversion

**Conclusion**: The entire pipeline operates on a unified token-based representation, where each step transforms tokens while preserving or modifying their count based on the operation type. AMP accelerates all computations with minimal quality impact.

---

**Next Steps**:
- [Thinker Deep Dive](03_Thinker_Deep_Dive.md) - Understand transformer processing
- [Architecture Overview](02_Architecture_Overview.md) - See how components connect
- [Inference Guide](08_Inference_Guide.md) - Use these pipelines in practice

