# Chapter 19: μOmni System Architecture

[Back to Index](00-INDEX.md) | [Next: The Thinker →](20-thinker-llm.md)

---

## 🎯 What You'll Learn

- Complete μOmni system architecture
- How all components work together
- Data flow through the system
- Design philosophy and trade-offs

---

## 🏗️ High-Level Architecture

```
┌────────────────────────────────────────────────────────┐
│                    μOmni SYSTEM                        │
│                                                        │
│  ┌──────────────┐  ┌──────────────┐  ┌─────────────┐ │
│  │   INPUTS     │  │  PROCESSING  │  │   OUTPUTS   │ │
│  └──────────────┘  └──────────────┘  └─────────────┘ │
│                                                        │
│   🖼️ Image          ┌──────────┐        📝 Text       │
│      ↓             │ Thinker  │          ↑           │
│   Vision Enc  ────→│  (Core   │──────────┘           │
│      ↓             │   LLM)   │                      │
│   Project          └──────────┘        🔊 Speech     │
│                          ↑                  ↑         │
│   🎤 Audio              │            ┌──────────┐    │
│      ↓                  │            │  Talker  │    │
│   Audio Enc  ───────────┤            │    +     │    │
│      ↓                  │            │   RVQ    │    │
│   Project               │            │    +     │    │
│                         │            │ Vocoder  │    │
│   📝 Text               │            └──────────┘    │
│      ↓                  │                            │
│   Tokenizer  ───────────┘                            │
│                                                       │
└───────────────────────────────────────────────────────┘
```

---

## 🧩 Core Components

### 1. **Thinker** (Decoder-Only LLM)

```
Role: Central reasoning engine
Type: Transformer decoder (GPT-style)
Size: 256-dim, 4 layers, 4 heads
Params: ~20.32M

Input: Multimodal tokens (text + image + audio)
Output: Next-token predictions

Key Features:
✅ Causal attention (autoregressive)
✅ RoPE positional encoding
✅ KV caching for fast generation
✅ Optional GQA, SwiGLU, MoE
✅ **Arthemis Neuromorphic Features**:
   - SpikingAttention (event-driven processing)
   - Liquid Time Constants (adaptive temporal dynamics)
```

---

### 2. **Vision Encoder** (ViT-Tiny)

```
Role: Convert images to embeddings
Type: Vision Transformer
Size: 128-dim, 4 layers

Input: Image (224×224×3)
Process: 196 patch tokens + CLS token
Output: CLS token (1, 128)

→ Vision Projector (128→256)
Final: (1, 256) embedding for Thinker
```

---

### 3. **Audio Encoder** (AuT-Tiny)

```
Role: Convert speech to embeddings
Type: Conv + Transformer encoder
Size: 192-dim, 4 layers

Input: Mel spectrogram (T, 128)
Process: 8x downsample + encoding
Output: Frame embeddings (T/8, 192)

→ Audio Projector (192→256)
Final: (T/8, 256) embeddings for Thinker

Key Features:
✅ Custom transformer blocks with Flash Attention
✅ Arthemis Neuromorphic Features:
   - SpikingAttention in encoder blocks
   - Liquid Time Constants in MLP layers
```

---

### 4. **Talker** (Speech Code Predictor)

```
Role: Generate speech codes autoregressively
Type: Transformer decoder
Size: 192-dim, 4 layers

Input: Previous RVQ codes (or start token)
Output: Next frame codes (base + residual)

Key Features:
✅ Inherits Thinker's transformer architecture
✅ Arthemis Neuromorphic Features (inherited):
   - SpikingAttention in decoder blocks
   - Liquid Time Constants in MLP layers

Works with:
- RVQ Codec (2 codebooks, 128 codes each)
- Griffin-Lim Vocoder (mel → audio)
```

---

### 5. **Projectors**

```
Vision Projector: Linear(128 → 256)
Audio Projector: Linear(192 → 256)

Purpose: Align all modalities to Thinker's dimension
Trainable: Yes (trained during SFT)
```

---

## 🔄 Complete Data Flow

### Input → Processing → Output

```
SCENARIO: Image QA

1. USER INPUT:
   Image: cat_photo.jpg
   Text: "What animal is this?"

2. IMAGE PROCESSING:
   cat_photo.jpg
   → Resize (224×224)
   → Vision Encoder
   → CLS token (1, 128)
   → Vision Projector
   → img_emb (1, 1, 256)

3. TEXT PROCESSING:
   "What animal is this?"
   → Tokenizer: [15, 234, 89, 42, 156]
   → Token Embeddings
   → text_emb (1, 5, 256)

4. FUSION:
   combined = [img_emb, text_emb]
   → Shape: (1, 6, 256)

5. THINKER PROCESSING:
   combined → Thinker (4 transformer blocks)
   → Output logits (1, 6, vocab_size)

6. GENERATION:
   Autoregressive decoding:
   → Next token: "This" (ID: 23)
   → Next token: "is" (ID: 67)
   → Next token: "a" (ID: 12)
   → Next token: "cat" (ID: 234)
   → Next token: "." (ID: 5)
   → Next token: <EOS> (ID: 2)

7. OUTPUT:
   "This is a cat."
```

---

## 📊 Parameter Breakdown

| Component | Parameters | Percentage |
|-----------|-----------|-----------|
| **Thinker** | ~20.32M | ~79.2% |
| **Audio Encoder** | ~2.05M | ~8.0% |
| **Vision Encoder** | ~914K | ~3.6% |
| **Talker** | ~2.24M | ~8.7% |
| **RVQ Codec** | ~49K | ~0.2% |
| **Projectors** | ~82K | ~0.3% |
| **TOTAL** | **~25.65M** | **100%** |

```
For comparison:
- GPT-3: 175 **billion** parameters (6800x larger!)
- LLaMA-7B: 7 **billion** parameters (270x larger)
- BERT-base: 110 **million** parameters (4.3x larger)
- μOmni: 25.65 **million** parameters ✓
```

---

## 🎯 Design Philosophy

### 1. **Efficiency First**

```
Goal: Train on single 12GB GPU

Strategies:
✅ Small vocabulary (5K vs 50K+)
✅ Compact dimensions (256 vs 768+)
✅ Fewer layers (4 vs 12-96)
✅ Efficient attention (Flash Attention)
✅ KV caching for generation
✅ Gradient checkpointing
```

---

### 2. **Modularity**

```
Each component trains independently:

Stage A: Thinker (text-only)
Stage B: Audio Encoder (ASR task)
Stage C: Vision Encoder (vision task)
Stage D: Talker + RVQ (speech generation)
Stage E: Joint fine-tuning (multimodal SFT)

Benefits:
✅ Debug easier (isolate issues)
✅ Parallel development
✅ Replace components independently
```

---

### 3. **Educational Clarity**

```
Priority: Understandable > State-of-the-art

Code choices:
✅ Clear variable names
✅ Comprehensive comments
✅ Standard PyTorch (no custom CUDA)
✅ Minimal dependencies
✅ Well-structured files

Trade-off: ~5-10% performance for 10x readability
```

---

## 🔗 Multimodal Fusion Strategy

### Hybrid Fusion

```
Why not early fusion (concatenate raw inputs)?
❌ Different modalities have different dimensions
❌ Loses specialized processing benefits

Why not late fusion (combine predictions)?
❌ No cross-modal interaction during processing

μOmni's Hybrid Fusion:
1. Specialized encoders per modality
2. Project to common dimension (256)
3. Concatenate embeddings
4. Unified Transformer (Thinker) processes all

Benefits:
✅ Specialized encoding (best of each modality)
✅ Cross-modal attention (interaction during processing)
✅ Flexible (any combination of inputs)
```

---

## 📈 Context Management

### Token Budget

```
Total context: 512-2048 tokens

Allocation example (context=512):
- Image: 1 token (CLS)
- Audio (3s): ~38 tokens (at 12.5Hz)
- Text prompt: ~10 tokens
- Available for generation: 512 - 49 = 463 tokens

Strategies:
1. Skip audio samples exceeding length threshold (via dataset filtering)
2. Sample video frames (1 per second)
3. Prioritize recent text context
4. Use KV caching to extend effective context
```

---

## 💻 Codebase Structure

```
μOmni/
├── omni/                      # Core modules
│   ├── thinker.py            # Decoder-only LLM
│   ├── audio_encoder.py      # AuT-Tiny
│   ├── vision_encoder.py     # ViT-Tiny
│   ├── talker.py             # Speech generator
│   ├── codec.py              # RVQ + vocoder
│   ├── tokenizer.py          # BPE tokenizer
│   └── utils.py              # All utilities (RMSNorm, RoPE, training helpers, datasets, checkpoint loading)
│
├── configs/                   # JSON configs
│   ├── thinker_tiny.json
│   ├── audio_enc_tiny.json
│   ├── vision_tiny.json
│   ├── talker_tiny.json
│   └── omni_sft_tiny.json
│
├── train_text.py             # Stage A training
├── train_audio_enc.py        # Stage B training
├── train_vision.py           # Stage C training
├── train_talker.py           # Stage D training
├── sft_omni.py               # Stage E training
│
├── infer_chat.py             # Inference interface
└── checkpoints/              # Model weights
    ├── thinker_tiny/
    ├── audio_enc_tiny/
    ├── vision_tiny/
    ├── talker_tiny/
    └── omni_sft_tiny/
```

---

## 🚀 Inference Modes

### 1. Text-Only Chat

```python
python infer_chat.py --ckpt_dir checkpoints/thinker_tiny

Input: "What is AI?"
Output: "AI is artificial intelligence..."
```

---

### 2. Image Understanding

```python
python infer_chat.py \
  --ckpt_dir checkpoints/omni_sft_tiny \
  --image cat.jpg \
  --text "Describe this image"

Output: "This is a photo of an orange cat sitting..."
```

---

### 3. Speech Input

```python
python infer_chat.py \
  --ckpt_dir checkpoints/omni_sft_tiny \
  --audio_in speech.wav

Output: Transcription + response
```

---

### 4. Text-to-Speech

```python
python infer_chat.py \
  --ckpt_dir checkpoints/omni_sft_tiny \
  --text "Hello world" \
  --audio_out output.wav

Output: output.wav (synthesized speech)
```

---

## 💡 Key Takeaways

✅ **μOmni** = Tiny multimodal AI (25.65M params)  
✅ **Thinker** = Central decoder-only LLM (GPT-style)  
✅ **Specialized encoders** for each modality  
✅ **Hybrid fusion** via projected embeddings  
✅ **5-stage training** pipeline (modular)  
✅ **Fits 12GB GPU** (efficient by design)  
✅ **Educational focus** (clarity > performance)

---

## 🎓 Self-Check Questions

1. What are the 5 main components of μOmni?
2. How many parameters does μOmni have total?
3. What dimension do all modalities project to?
4. What type of transformer is the Thinker (encoder/decoder)?
5. What fusion strategy does μOmni use?

<details>
<summary>📝 Answers</summary>

1. Thinker (LLM), Vision Encoder, Audio Encoder, Talker, RVQ Codec
2. ~25.65 million parameters
3. 256 dimensions (d_model of Thinker)
4. Decoder-only (autoregressive/causal)
5. Hybrid fusion (specialized encoders → project → unified processing)
</details>

---

[Continue to Chapter 20: The Thinker - Core Language Model →](20-thinker-llm.md)

**Chapter Progress:** μOmni Architecture ●○○○○○○ (1/7 complete)

