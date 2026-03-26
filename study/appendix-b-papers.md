# Appendix B: Key Research Papers

The papers that underpin micro-Omni's architecture and training methods.

---

## Core Papers

| Paper                                | Authors               | Year | micro-Omni Component                |
|--------------------------------------|-----------------------|------|--------------------------------------|
| Attention Is All You Need            | Vaswani et al.        | 2017 | Transformer architecture (Thinker, Talker, all encoders) |
| RoFormer: Enhanced Transformer with Rotary Position Embedding | Su et al. | 2021 | RoPE positional encoding in all attention layers |
| GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints | Ainslie et al. | 2023 | Grouped Query Attention (fewer KV heads) |
| GLU Variants Improve Transformer     | Shazeer               | 2020 | SwiGLU activation in FFN layers      |
| SoundStream: An End-to-End Neural Audio Codec | Zeghidour et al. | 2021 | Residual Vector Quantization (RVQ) for audio codec |
| An Image Is Worth 16x16 Words: Transformers for Image Recognition at Scale | Dosovitskiy et al. | 2020 | Vision Transformer (ViT) architecture |
| Learning Transferable Visual Models From Natural Language Supervision | Radford et al. | 2021 | CLIP-style contrastive image-text training |
| HiFi-GAN: Generative Adversarial Networks for Efficient and High Fidelity Speech Synthesis | Kong et al. | 2020 | Neural vocoder (waveform generation) |
| Connectionist Temporal Classification: Labelling Unsegmented Sequence Data with Recurrent Neural Networks | Graves et al. | 2006 | CTC loss for audio encoder training |
| Thinker-Talker Architecture (multimodal LLM research) | Various teams | 2024 | Thinker-Talker dual-model architecture |
| FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness | Dao et al. | 2022 | Efficient attention computation |

---

## Paper-to-Component Map

```
                    Vaswani 2017 (Transformer)
                    /        |         \
                   v         v          v
            +---------+ +---------+ +----------+
            | Thinker | | Talker  | | Encoders |
            +---------+ +---------+ +----------+
                |            |           |
  Su 2021 (RoPE)    Su 2021 (RoPE)     |
  Ainslie 2023 (GQA)                   |
  Shazeer 2020 (SwiGLU)               |
                                       |
                    +------------------+------------------+
                    |                                     |
              +-----+------+                      +------+------+
              | Audio Enc  |                      | Vision Enc  |
              +------------+                      +-------------+
              Graves 2006 (CTC)                   Dosovitskiy 2020 (ViT)
              Zeghidour 2021 (RVQ)                Radford 2021 (CLIP)
                    |
              +-----+------+
              | Vocoder    |
              +------------+
              Kong 2020 (HiFi-GAN)
```

---

## Brief Summaries

### Attention Is All You Need (Vaswani 2017)

Introduced the Transformer: encoder-decoder architecture using only
self-attention and cross-attention, replacing recurrence entirely. The
scaled dot-product attention mechanism and multi-head attention are the
foundation of every component in micro-Omni.

### RoFormer (Su 2021)

Proposed Rotary Position Embeddings (RoPE): encode absolute position
through rotation matrices applied to Q and K, so the dot product
naturally captures relative position. Enables length extrapolation and
has become standard in modern LLMs.

### GQA (Ainslie 2023)

Grouped Query Attention shares key-value heads across groups of query
heads. Interpolates between multi-head (each query has its own KV) and
multi-query (all queries share one KV). Reduces KV cache size and
speeds up inference with minimal quality loss.

### GLU Variants (Shazeer 2020)

Showed that Gated Linear Unit variants, particularly SwiGLU (Swish
activation + gating), outperform standard ReLU FFNs in Transformers.
The gating mechanism lets the network learn which information to pass
through.

### SoundStream (Zeghidour 2021)

End-to-end neural audio codec using Residual Vector Quantization.
Compresses audio into discrete tokens at various bitrates. micro-Omni
uses RVQ to represent audio as sequences of discrete codes that the
Talker can generate.

### ViT (Dosovitskiy 2020)

Applied Transformers directly to image patches. Split an image into
fixed-size patches, linearly embed them, and process with a standard
Transformer encoder. Scales better than CNNs with sufficient data.

### CLIP (Radford 2021)

Trained image and text encoders jointly with a contrastive objective
(InfoNCE). Learns aligned image-text representations useful for
zero-shot classification, retrieval, and as a foundation for
multimodal models.

### HiFi-GAN (Kong 2020)

GAN-based vocoder that synthesizes high-fidelity audio from mel
spectrograms. Uses multi-scale and multi-period discriminators with
feature matching loss. Fast enough for real-time synthesis.

### CTC (Graves 2006)

Connectionist Temporal Classification enables training sequence models
without pre-aligned labels. Uses a blank token and dynamic programming
to marginalize over all valid alignments. Standard for speech
recognition training.

### Thinker-Talker Architecture (Multimodal LLM Research, 2024)

The Thinker-Talker architecture uses a "thinker" LLM that reasons
over all modalities and generates text tokens plus audio token
placeholders; a "talker" model converts those placeholders into audio
codec codes in streaming fashion. This approach was introduced by
recent multimodal LLM research.

### FlashAttention (Dao 2022)

IO-aware exact attention algorithm that tiles the computation to
minimize HBM reads/writes. Provides 2-4x speedup and significant
memory savings with no approximation. Enables longer context lengths
on the same hardware.
