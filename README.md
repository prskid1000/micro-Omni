
# μOmni (Tiny Qwen3-style Omni) — fits 12GB VRAM

A tiny, from-scratch **omni** stack (text + image + speech-in/out) that you can train on a single 12 GB GPU
with **small datasets** (each modality well under **5GB**). Includes:
- Minimal Qwen3-style **Thinker** (decoder-only LLM) with TM-RoPE-lite
- **Audio encoder** (AuT-Tiny) for ASR / audio understanding (12.5 Hz rate)
- **Vision encoder** (ViT-Tiny) for image features
- **RVQ codec** (2 codebooks) + **Talker** (speech code predictor)
- **Griffin-Lim** vocoder by default (no heavy TTS training required)
- Training scripts for each stage + an end-to-end **SFT/omni** trainer
- Simple **inference** CLI for text/chat, image QA, speech chat

> This is a **reference learning repo**—compact and readable. It trades SOTA quality for simplicity and VRAM thrift.

## Architecture

### Overview
μOmni follows a **Thinker-Talker** architecture inspired by Qwen3 Omni:
- **Thinker**: Decoder-only LLM that processes unified multimodal embeddings (text + image + audio)
- **Talker**: Autoregressive speech code predictor that generates audio from Thinker outputs
- **Encoders**: Separate encoders for audio (AuT-Tiny) and vision (ViT-Tiny) that project to Thinker's embedding space
- **Codec**: RVQ (Residual Vector Quantization) for speech representation
- **Vocoder**: Griffin-Lim for waveform reconstruction (no training required)

### 1. Thinker (Decoder-Only LLM)

**Purpose**: Core language model that processes unified multimodal embeddings.

**Components**:
- **Token Embeddings**: `nn.Embedding(vocab_size, d_model)`
- **Transformer Blocks**: Stack of `Block` modules
- **Output Head**: `nn.Linear(d_model, vocab_size)` for next-token prediction

**Block Architecture**:
- **Pre-Normalization**: RMSNorm before attention and MLP
- **Attention**: Multi-head self-attention with optional GQA
- **MLP**: Feedforward network with optional SwiGLU activation
- **MoE** (optional): Mixture-of-Experts replaces MLP when enabled

**Algorithms Used**:
- **RoPE (Rotary Position Embedding)**: Relative positional encoding with configurable `theta` (default: 10000)
- **GQA (Grouped Query Attention)**: Multiple query heads share fewer key/value heads (configurable, default: disabled)
- **SwiGLU**: Swish-gated linear unit activation `x * sigmoid(x)` (default: enabled)
- **MoE (Mixture-of-Experts)**: Router selects top-k experts per token (configurable, default: disabled)
- **KV Caching**: Stores past key/value states for faster autoregressive generation
- **RMSNorm**: Root Mean Square Layer Normalization (pre-norm architecture)

**Key Features**:
- Accepts both token IDs and raw embeddings (for multimodal input)
- KV caching support for efficient inference
- Configurable Qwen3 Omni features (GQA, SwiGLU, MoE)

### 2. Audio Encoder (AuT-Tiny)

**Purpose**: Encodes mel spectrograms into frame-level embeddings for Thinker.

**Components**:
- **ConvDown**: 2D convolutional downsampling (4x or 8x time reduction)
- **Projection**: Linear layer mapping flattened conv features to `d_model`
- **Encoder Blocks**: Stack of transformer encoder layers
- **Final Norm**: RMSNorm

**Architecture**:
```
Mel (B, T, 128) 
  → Conv2D Downsample (4x or 8x) 
  → Flatten & Project 
  → Transformer Encoder Blocks 
  → RMSNorm 
  → Output (B, T/downsample, d_model)
```

**Algorithms Used**:
- **Convolutional Downsampling**: Strided 2D convolutions reduce temporal resolution
  - 4x downsample: 2x stride twice (default)
  - 8x downsample: 2x stride three times (for 12.5 Hz frame rate)
- **Multi-Head Self-Attention**: Standard transformer encoder attention
- **RMSNorm**: Pre-normalization in encoder blocks
- **GELU Activation**: Gaussian Error Linear Unit in MLP

**Frame Rate Calculation**:
- Input mel: `sample_rate / hop_length` Hz (e.g., 16000/160 = 100 Hz)
- After 4x downsample: 25 Hz
- After 8x downsample: 12.5 Hz (target for Qwen3 Omni alignment)

### 3. Vision Encoder (ViT-Tiny)

**Purpose**: Encodes images into patch embeddings with CLS token for Thinker.

**Components**:
- **Patch Embedding**: `nn.Conv2d(3, d_model, kernel_size=patch, stride=patch)` - non-overlapping patches
- **CLS Token**: Learnable classification token `[CLS]`
- **Positional Embeddings**: Learnable position embeddings for patches + CLS
- **Encoder Blocks**: Stack of `nn.TransformerEncoderLayer`
- **Final Norm**: RMSNorm

**Architecture**:
```
Image (B, 3, 224, 224)
  → Patch Embedding (patch_size=16 → 14×14 patches)
  → Add CLS Token + Positional Embeddings
  → Transformer Encoder Blocks
  → RMSNorm
  → Output: CLS token (B, 1, d_model) + patch tokens (B, 196, d_model)
```

**Algorithms Used**:
- **Vision Transformer (ViT)**: Patch-based image encoding
- **CLS Token**: Aggregates global image information
- **Pre-Normalization**: `norm_first=True` in encoder layers
- **RMSNorm**: Final normalization layer

**Output**: Only CLS token is used (projected to Thinker's embedding space)

### 4. Talker (Speech Code Predictor)

**Purpose**: Autoregressively predicts RVQ codebook indices for speech generation.

**Components**:
- **Code Embeddings**: `nn.Embedding(codebook_size, d_model)` for each codebook
- **Start Token**: Learnable start-of-sequence parameter
- **Transformer Blocks**: Same architecture as Thinker (GQA, SwiGLU support)
- **Output Heads**: Separate heads for base and residual codebooks

**Architecture**:
```
Previous Codes (B, T, 2)
  → Embed Base + Residual Codes
  → Add Start Token
  → Transformer Blocks (with KV caching)
  → RMSNorm
  → Base Head + Residual Head
  → Output: (base_logits, res_logits) both (B, T, codebook_size)
```

**Algorithms Used**:
- **Autoregressive Generation**: Predicts next frame codes from previous codes
- **GQA (Grouped Query Attention)**: Optional, same as Thinker
- **SwiGLU**: Optional, same as Thinker
- **RoPE**: Rotary positional embeddings
- **KV Caching**: Stores past attention states for faster generation
- **RMSNorm**: Pre-normalization
- **Multi-Codebook Prediction**: Separate heads for base and residual codebooks

**Training**: Uses teacher forcing with `torch.roll` to shift codes by one position

### 5. RVQ Codec (Residual Vector Quantization)

**Purpose**: Quantizes mel spectrograms into discrete codebook indices for efficient speech representation.

**Components**:
- **Codebooks**: List of `nn.Embedding(codebook_size, d)` - typically 2 codebooks
- **Input Projection**: `nn.Linear(128, d)` - projects mel frames to codebook dimension
- **Output Projection**: `nn.Linear(d, 128)` - reconstructs mel from quantized codes

**Architecture**:
```
Mel Frame (B, 128)
  → Project to d-dim
  → Greedy Residual Quantization:
     1. Find nearest codebook entry (codebook 0)
     2. Compute residual
     3. Find nearest codebook entry for residual (codebook 1)
  → Return indices (B, 2)
```

**Algorithms Used**:
- **Residual Vector Quantization**: Multi-stage quantization where each stage quantizes the residual
- **Greedy Quantization**: Nearest neighbor search in embedding space
- **Euclidean Distance**: `||residual - code||²` for codebook lookup

**Decoding**:
```
Indices (B, 2)
  → Lookup embeddings from each codebook
  → Sum embeddings
  → Project back to mel (B, 128)
```

### 6. Griffin-Lim Vocoder

**Purpose**: Converts mel spectrograms to waveform (no training required).

**Algorithms Used**:
- **Griffin-Lim Algorithm**: Iterative phase reconstruction from magnitude spectrogram
- **STFT/ISTFT**: Short-Time Fourier Transform for spectrogram conversion
- **Iterative Refinement**: Alternates between time and frequency domain constraints

**Process**:
1. Convert mel spectrogram to linear magnitude spectrogram
2. Initialize random phase
3. Iteratively refine phase using ISTFT → STFT (typically 32 iterations)
4. Return final waveform

**Note**: This is a classical vocoder - no neural network training required. Quality is lower than neural vocoders but sufficient for prototyping.

### Multimodal Fusion

**Projection Layers**:
- **Audio Projector**: `nn.Linear(audio_dim, thinker_d_model)` - projects audio encoder output
- **Vision Projector**: `nn.Linear(vision_dim, thinker_d_model)` - projects vision encoder CLS token

**Fusion Strategy**:
1. Encode each modality separately (audio → frames, image → CLS token)
2. Project to Thinker's embedding dimension
3. Concatenate: `[image_tokens] + [audio_tokens] + [text_tokens]`
4. Process through Thinker as unified sequence

**Training**: SFT stage trains projectors + Thinker on mixed multimodal batches

## Training Workflow

### Stage-by-Stage Training

The training follows a **staged approach** to fit within 12GB VRAM and enable efficient learning:

#### Stage A: Thinker Pretraining (Text-Only)
**Script**: `train_text.py`

**Process**:
1. **Data**: Raw text corpus → tokenized sequences
2. **Objective**: Next-token prediction (causal language modeling)
3. **Training**:
   - Input: `[BOS] + tokens[:T-1]`
   - Target: `tokens[1:T]` (shifted by one)
   - Loss: Cross-entropy on target tokens (padding ignored)
4. **Features**:
   - Learning rate scheduling (warmup + cosine decay)
   - Gradient clipping
   - Validation loop with best model saving
   - Periodic checkpointing

**Output**: `checkpoints/thinker_tiny/thinker.pt`

#### Stage B: Audio Encoder Pretraining (ASR)
**Script**: `train_audio_enc.py`

**Process**:
1. **Data**: Audio files (WAV) + transcriptions
2. **Preprocessing**:
   - Audio → Mel spectrogram (128 bins, 100 Hz frame rate)
   - Text → character-level targets (mapped to 1-63)
3. **Objective**: CTC (Connectionist Temporal Classification) loss
4. **Training**:
   - Input: Mel spectrogram `(B, T, 128)`
   - Encoder: ConvDown → Transformer → Linear head
   - Output: Logits `(B, T', vocab_size)` where `T' = T/downsample_factor`
   - Loss: CTC loss between encoder output and text targets
5. **Frame Rate**: 8x downsample → 12.5 Hz (aligned with Qwen3 Omni)

**Output**: `checkpoints/audio_enc_tiny/audio_enc.pt`

#### Stage C: Vision Encoder Training
**Script**: `train_vision.py`

**Process**:
1. **Data**: Images + captions
2. **Objective**: Simple classification task (e.g., predict if caption contains "red")
3. **Training**:
   - Input: Image `(B, 3, 224, 224)`
   - Encoder: ViT-Tiny → CLS token
   - Head: Linear classifier on CLS token
   - Loss: Cross-entropy classification loss
4. **Note**: This is a simplified pretraining task. In production, use contrastive learning (CLIP-style).

**Output**: `checkpoints/vision_tiny/vision.pt`

#### Stage D: Talker + RVQ Codec Training
**Script**: `train_talker.py`

**Process**:
1. **Data**: Audio files (WAV) for TTS
2. **Preprocessing**:
   - Audio → Mel spectrogram
   - Mel frames → RVQ codes (greedy quantization)
3. **Objective**: Autoregressive code prediction
4. **Training**:
   - Input: Previous codes `(B, T, 2)` (shifted by one position)
   - Target: Current codes `(B, T, 2)`
   - Model: Talker predicts base + residual codebook indices
   - Loss: Cross-entropy on both codebooks
5. **RVQ Training**: Codebooks learned end-to-end with Talker

**Output**: `checkpoints/talker_tiny/talker.pt` (includes RVQ codebooks)

#### Stage E: Multimodal SFT (Supervised Fine-Tuning)
**Script**: `sft_omni.py`

**Process**:
1. **Data**: Mixed batches of text, images, and audio
2. **Objective**: Instruction following with multimodal understanding
3. **Training Flow**:
   ```
   For each batch:
     - Extract image features (if present) → Vision Encoder → Project
     - Extract audio features (if present) → Audio Encoder → Project
     - Tokenize text → Thinker Token Embeddings
     - Concatenate: [image_tokens] + [audio_tokens] + [text_tokens]
     - Forward through Thinker
     - Loss: Cross-entropy on text tokens only (multimodal tokens ignored)
   ```
4. **Batch Processing**:
   - Images: Batched together, processed in single forward pass
   - Audio: Batched together (with padding), processed in single forward pass
   - Text: Processed per sample, then combined into batch
   - Final: All samples padded to same length, processed through Thinker
5. **Trainable Components**:
   - Thinker (fine-tuned)
   - Audio projector (trained from scratch)
   - Vision projector (trained from scratch)
   - Encoders (frozen)

**Output**: `checkpoints/omni_sft_tiny/omni.pt` (includes Thinker + projectors)

### Training Best Practices

All training scripts include:
- **Learning Rate Scheduling**: Linear warmup → Cosine decay
- **Gradient Clipping**: Prevents exploding gradients (max_norm=1.0)
- **Random Seed**: Ensures reproducibility (seed=42)
- **Validation Loops**: Periodic validation with best model saving
- **Periodic Checkpointing**: Saves checkpoints every N steps
- **Structured Logging**: Timestamped logs with formatted metrics

## Inference Workflow

### End-to-End Pipeline

The inference system supports **three modes**: text-only, multimodal (image/audio input), and speech output.

#### Mode 1: Text-Only Chat

**Flow**:
```
User Input (text)
  → Tokenizer.encode()
  → Token IDs
  → Thinker (with KV caching)
  → Next-token logits
  → Autoregressive generation (greedy decoding)
  → Generated token IDs
  → Tokenizer.decode()
  → Output Text
```

**Details**:
- **KV Caching**: First forward pass processes full prompt, subsequent passes only process new tokens
- **Autoregressive**: Generates one token at a time until EOS or max length
- **Context Management**: Truncates input if exceeds context length

#### Mode 2: Multimodal Input (Image/Audio + Text)

**Flow**:
```
Image (optional)
  → Vision Encoder
  → CLS Token (1, d_vision)
  → Vision Projector
  → (1, 1, d_thinker)

Audio (optional)
  → Mel Spectrogram
  → Audio Encoder
  → Frame Embeddings (1, T_audio, d_audio)
  → Audio Projector
  → (1, T_audio, d_thinker)

Text
  → Tokenizer
  → Token IDs
  → Thinker Token Embeddings
  → (1, T_text, d_thinker)

Concatenate: [image] + [audio] + [text]
  → Combined Embeddings (1, T_total, d_thinker)
  → Thinker (processes unified sequence)
  → Next-token logits
  → Autoregressive generation
  → Output Text
```

**Details**:
- **Context Allocation**: Multimodal tokens consume context, text tokens fill remaining space
- **Projection**: Both encoders project to Thinker's embedding dimension
- **Unified Processing**: Thinker treats all modalities as a single sequence

#### Mode 3: Speech Output (Text → Speech)

**Flow**:
```
Generated Text
  → Tokenizer.encode()
  → Text Token IDs
  → Thinker (optional: can condition on text)
  → Text Embeddings (optional intermediate step)
  → Talker (autoregressive code prediction)
  → RVQ Code Sequences (B, T_frames, 2)
  → RVQ Decode
  → Mel Spectrogram (T_frames, 128)
  → Griffin-Lim Vocoder
  → Waveform (T_samples)
  → Audio File (WAV)
```

**Details**:
- **Talker Generation**:
  1. Start with zero codes `(1, 1, 2)`
  2. First forward: Predict first frame codes
  3. Incremental: Use KV cache, predict next frame from previous
  4. Continue until max frames or stop condition
- **RVQ Decoding**: Sum codebook embeddings → project to mel
- **Vocoding**: Griffin-Lim reconstructs phase from magnitude spectrogram

### Complete Multimodal Round-Trip Example

**Input**: Audio file + text question
**Output**: Text response + speech output

```
1. Audio Input Processing:
   Audio.wav
     → Mel Spectrogram (1, T, 128)
     → Audio Encoder
     → Audio Embeddings (1, T', d_audio)
     → Audio Projector
     → (1, T', d_thinker)

2. Text Processing:
   "What did you hear?"
     → Tokenizer
     → Text Embeddings (1, T_text, d_thinker)

3. Multimodal Fusion:
   [Audio Embeddings] + [Text Embeddings]
     → Thinker
     → Generated Text: "I heard a dog barking."

4. Speech Generation:
   "I heard a dog barking."
     → Talker (autoregressive)
     → RVQ Codes (1, T_frames, 2)
     → RVQ Decode
     → Mel Spectrogram
     → Griffin-Lim
     → Output.wav
```

### Key Inference Optimizations

1. **KV Caching**:
   - **Thinker**: Caches attention keys/values for all previous tokens
   - **Talker**: Caches attention keys/values for all previous frames
   - **Benefit**: Reduces computation from O(n²) to O(n) for incremental generation

2. **Model Evaluation Mode**:
   - All models set to `.eval()` mode
   - Disables dropout, batch norm updates
   - Ensures consistent inference behavior

3. **Context Management**:
   - Dynamically allocates context between modalities
   - Truncates inputs if total exceeds context length
   - Preserves as much text context as possible

4. **Batch Processing** (Training Only):
   - Efficient batching of images/audio during SFT
   - Reduces GPU memory usage
   - Faster training compared to one-by-one processing

## Environment
```
pip install -r requirements.txt
```

## Datasets (< 5GB per modality)
You have two options:

### Option A: Tiny synthetic samples (included)
Run once to generate toy data that exercises the pipeline:
```
python scripts/make_synthetic_datasets.py
```
This creates:
- `data/text/tiny_corpus.txt` (~2MB)
- `data/images/{images,annotations}.json` (~20MB)
- `data/audio/{wav/*.wav, asr.csv, tts.csv}` (~15MB)

### Option B: Real small datasets (each <5GB locally)
Scripts are provided but commented with URLs (so you can choose mirrors). Examples:
- **Text**: `wikitext-2` (~35MB) or `wikitext-103` (~200MB)
- **Images**: **COCO 2017 val** images (5k imgs, ~1GB) + captions (~250MB)
- **Audio (ASR)**: a **subset** of Common Voice EN (e.g., first ~10–20 hours, ~1–3GB)
- **Audio (TTS)**: a **subset** of LJSpeech (e.g., 3k clips, ~1.7GB)

Use:
```
# edit scripts/download_small_datasets.py for paths you prefer, then
python scripts/download_small_datasets.py --modality text images audio_asr audio_tts
```
All datasets are kept under **5GB per modality** on disk via subsetting.

## Training (stages)
Train in stages to fit 12GB easily.

### A) Text LLM pretrain (Thinker)
```
python train_text.py --config configs/thinker_tiny.json
```

### B) Audio encoder (ASR) pretrain
```
python train_audio_enc.py --config configs/audio_enc_tiny.json
```

### C) Vision encoder train (captioning contrastive pretraining)
```
python train_vision.py --config configs/vision_tiny.json
```

### D) Talker (speech code predictor) + codec pretrain
```
python train_talker.py --config configs/talker_tiny.json
```

### E) Omni SFT/alignment (mix multimodal mini-batches)
```
python sft_omni.py --config configs/omni_sft_tiny.json
```

## Inference
### Text chat:
```
python infer_chat.py --ckpt_dir checkpoints/thinker_tiny
```

### Image QA / caption:
```
python infer_chat.py --ckpt_dir checkpoints/omni_sft_tiny --image path/to.jpg "describe this"
```

### Speech chat (ASR in → Thinker → Talker → Griffin-Lim)
```
python infer_chat.py --ckpt_dir checkpoints/omni_sft_tiny --audio_in path/to.wav
```

## Notes
- Default voice uses **Griffin-Lim** (no vocoder training needed).
- All configs target ~**120–140M** total params across modules and fit a **12GB** GPU with gradient accumulation + checkpointing.
- Use smaller context (`ctx=1024`) during pretrain; go `2048` for SFT if VRAM allows.

## License
MIT for this scaffold. Replace datasets with those compatible with your needs.
