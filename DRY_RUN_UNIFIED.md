# Unified Dry Run Computation - μOmni Pipeline

This document traces the complete computation pipeline from initial input tokens through all transformations to final output, explaining what each step does with respect to the input.

---

## Overview: Token-Based Processing Pipeline

The μOmni system processes multimodal inputs by converting everything to a unified token representation:
- **Text**: Direct tokenization → token IDs
- **Image**: Image → ViT patches → CLS token → projected to token space
- **Audio**: Audio → Mel → AudioEncoder → projected to token space
- **All modalities**: Combined into a single sequence of "tokens" (embeddings) → Thinker processes → generates text tokens

---

## Part 1: Text-Only Pipeline

### Input: Text String
```
Input: "Hello, how are you?"
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
- Input: String (variable length)
- Output: List of integers `[1, 234, 567, 890, 123, 456]` (6 tokens)
- BOS token (1) prepended for sequence start

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
- Input: `(1, 6)` token IDs
- Output: `(1, 6, 256)` embeddings
- Each token ID → 256-dimensional vector
- **6 input tokens → 6 embedding vectors**

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

#### Layer 1:
```python
# Input: (1, 6, 256)
# Block 1:
#   - RMSNorm: (1, 6, 256) → (1, 6, 256) [normalize]
#   - Attention: (1, 6, 256) → (1, 6, 256)
#     * Q, K, V: (1, 6, 256) → (1, 4 heads, 6, 64)
#     * Attention scores: (1, 4, 6, 6) [each token attends to all 6]
#     * Output: (1, 6, 256) [weighted combination]
#   - MLP (SwiGLU): (1, 6, 256) → (1, 6, 256)
# Output: (1, 6, 256)
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
```

**Transformation**:
- Input: `(1, 6, 256)` embeddings
- Output: `(1, 6, 5000)` logits
- **6 input tokens → 6 sets of vocabulary logits**
- Each position predicts probability distribution over 5000 tokens

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

### Input: Image + Text String
```
Image: RGB image (224×224 pixels)
Text: "What do you see in this image?"
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

### Input: Audio + Text String
```
Audio: Waveform (1 second @ 16kHz = 16,000 samples)
Text: "What did you hear?"
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

### Input: Text String
```
Input: "This is a test of text to speech."
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
**What it does**: Converts mel spectrogram to audio waveform using Griffin-Lim

```python
# Input: (301, 128) mel spectrogram
# Process: Griffin-Lim algorithm
mel_np = mel.cpu().numpy()  # (301, 128)
mel_np = mel_np.T  # (128, 301) - transpose for vocoder
# Upsample to linear spectrogram: (513, 301)
# Griffin-Lim: Iterative phase reconstruction
audio = voc.mel_to_audio(mel_np)  # (audio_samples,)
# Duration: 301 frames @ 12.5 Hz = 24.08 seconds
# Samples: 24.08 * 16000 = 385,280 samples
```

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

### Input: Mel Spectrogram Batch
```
Input: (B=2, T=50, 128) - 2 samples, 50 frames, 128 mel bins
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

**Conclusion**: The entire pipeline operates on a unified token-based representation, where each step transforms tokens while preserving or modifying their count based on the operation type.

