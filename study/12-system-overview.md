# Chapter 12: System Overview & Multimodal Fusion

You have now learned every building block individually -- transformers, attention, tokenization, mel spectrograms, vision patches, vector quantization, and more. It is time to see how they all fit together inside a single system.

---

## The Orchestra Analogy

Think of a symphony orchestra:

- **Specialized musicians** play different instruments: a violinist reads sheet music differently from a drummer reading a rhythm chart. Each expert translates their unique notation into sound.
- **The conductor** listens to all musicians simultaneously, understands the combined piece, and decides what comes next.
- **The vocalist** takes the conductor's cues and produces the final sung output.

In the same way, the system has:

| Role | Component | What It Does |
|------|-----------|-------------|
| Violinist | **Vision Encoder (ViT-Tiny)** | Reads images, produces visual embeddings |
| Drummer | **Audio Encoder (AuT-Tiny)** | Reads speech/sound, produces audio embeddings |
| Sheet music | **Text Embedding** | Converts text tokens into embeddings |
| Conductor | **Thinker** | Processes all modalities together, generates text responses |
| Vocalist | **Talker** | Takes Thinker's output, produces speech codes |
| Sound engineer | **Vocoder (HiFi-GAN / Griffin-Lim)** | Converts speech codes back to audible waveforms |

---

## Full Architecture Diagram

```
                          INPUT MODALITIES
    +-----------+     +-----------+     +-----------+
    |  Image    |     |  Audio    |     |  Text     |
    | (3,224,   |     | (B,T,128) |     | token IDs |
    |    224)   |     |  mel spec |     | (B, T)    |
    +-----------+     +-----------+     +-----------+
         |                 |                 |
         v                 v                 v
  +-------------+   +-------------+   +-------------+
  |  ViT-Tiny   |   |  AuT-Tiny   |   | tok_emb     |
  |  d=192      |   |  d=384      |   | (32000,384) |
  |  8 layers   |   |  8 layers   |   |             |
  +-------------+   +-------------+   +-------------+
         |                 |                 |
         | CLS (1,192)     | (T/8,384)       | (T,384)
         v                 |                 |
  +-------------+          |                 |
  | proj 192->  |          |                 |
  |   384       |          |                 |
  +-------------+          |                 |
         |                 |                 |
         | (1,384)         | (T/8,384)       | (T,384)
         v                 v                 v
  +----------------------------------------------+
  |          CONCATENATE along sequence dim       |
  |  => (B, 1 + T/8 + T_text, 384)               |
  +----------------------------------------------+
                       |
                       v
            +--------------------+
            |      THINKER       |
            |   d=384, 8 layers  |
            |   6 heads, ff=1536 |
            |   Causal Attention |
            +--------------------+
                       |
                       v
              text logits (B, T, 32000)
                       |
              +--------+--------+
              |                 |
              v                 v
        Text Response      +----------+
        (decoded via       |  TALKER  |
         tokenizer)        |  d=384   |
                           |  8 layers|
                           +----------+
                                |
                                v
                        RVQ codes (B,T,2)
                                |
                                v
                        +-------------+
                        | RVQ Decode  |
                        +-------------+
                                |
                                v
                        mel (B,T,128)
                                |
                                v
                        +-------------+
                        | Vocoder     |
                        | (HiFi-GAN)  |
                        +-------------+
                                |
                                v
                          Audio Waveform
```

---

## Component Summary Table

| Component | Source File | Params | Dimension | Purpose |
|-----------|------------|--------|-----------|---------|
| Thinker (ThinkerLM) | `omni/thinker.py` | ~20.32M | d=384, 8 layers, 6 heads | Core language model; processes all modalities, generates text |
| Audio Encoder (AuT-Tiny) | `omni/audio_encoder.py` | ~2.05M | d=384, 8 layers, 6 heads | Converts mel spectrograms to embeddings for ASR |
| Vision Encoder (ViT-Tiny) | `omni/vision_encoder.py` | ~914K | d=192, 8 layers, 3 heads | Converts images to embeddings via CLIP training |
| Talker (TalkerTiny) | `omni/talker.py` | ~2.24M | d=384, 8 layers, 6 heads | Predicts RVQ speech codes for TTS |
| RVQ Codec | `omni/codec.py` | ~49K | d=64, 2 codebooks x 128 | Quantizes/reconstructs mel frames |
| OCR Model | `omni/ocr_model.py` | varies | ViT d=192 + decoder d=384 | Reads text from images |
| **Total (core)** | | **~25.58M** | | |

---

## The Hybrid Fusion Strategy

There are three classical approaches to combining modalities:

1. **Early fusion**: concatenate raw inputs (pixels + waveform + text) before any processing. Simple but forces all modalities through the same low-level processing, which is inefficient.
2. **Late fusion**: process each modality completely independently, then combine final predictions. Easy to train but cannot capture cross-modal relationships (e.g., "the sound of the object in the image").
3. **Hybrid fusion** (what we use): encode each modality with a specialized encoder, project all encodings into a **shared embedding space**, then concatenate and process jointly through a unified transformer.

```
  EARLY               HYBRID (ours)           LATE
  concat raw    =>    encode separately  =>    combine
  inputs              project to 384-dim       final
  before any          concatenate              predictions
  processing          attend jointly           only
```

Why hybrid is the best of both worlds:

- Each encoder is optimized for its modality (convolutions for audio, patches for images, learned embeddings for text)
- The shared 384-dimensional space lets the Thinker attend across modalities -- an image token can attend to a text token and vice versa
- Any subset of modalities works: text-only, image+text, audio+text, or all three

---

## Token Budget

Every input modality is converted to a sequence of 384-dimensional vectors. The key question is: how many tokens does each modality produce?

| Modality | Token Rate | Example |
|----------|-----------|---------|
| **Image** | 1 token (CLS) | One 224x224 image = 1 token of size 384 (after projection from 192) |
| **Audio** | 12.5 tokens/sec | 8x conv downsample from 100Hz mel frames. 10 seconds of audio = 125 tokens |
| **Text** | 1 token/subword | SentencePiece BPE with vocab=32000. "Hello world" = ~2 tokens |

With a context length of 256 (Thinker training) or 512 (SFT), you can fit:

- Pure text: 256 tokens (~200 words)
- Image + text question: 1 + ~50 = 51 tokens (plenty of room)
- 10s audio + text prompt: 125 + ~20 = 145 tokens

---

## Total Parameter Count

```
Thinker (ThinkerLM)         ~20,320,000   (78.8%)
Audio Encoder (AuT-Tiny)     ~2,050,000   ( 8.0%)
Talker (TalkerTiny)          ~2,240,000   ( 8.7%)
Vision Encoder (ViT-Tiny)      ~914,000   ( 3.6%)
RVQ Codec                       ~49,000   ( 0.2%)
Vision Projection (192->384)    ~73,000   ( 0.3%)
Vocoder (HiFi-GAN)*          ~1,000,000+  (separate)
────────────────────────────────────────
Total (core, excl. vocoder)  ~25.65M
```

*The HiFi-GAN vocoder is trained separately and is not counted in the core model size. Griffin-Lim requires no parameters at all.*

For context: GPT-2 Small is 124M parameters, LLaMA-7B is 7,000M. This entire multimodal system fits in **25.65M** -- small enough to train on a single consumer GPU.

---

## Modality Combinations

The system accepts any subset of input modalities. The Thinker does not care what type of tokens it receives -- they are all 384-dimensional vectors by the time they arrive.

| Input | Output | Task |
|-------|--------|------|
| Text only | Text | Chat, Q&A, completion |
| Image + Text | Text | Visual question answering, image captioning |
| Audio + Text | Text | Speech transcription (ASR), audio Q&A |
| Image + Audio + Text | Text | Full multimodal reasoning |
| Text | Text + Speech | Text-to-speech (Thinker generates text, Talker generates speech) |
| Image | Text | OCR (via separate OCR model) |

---

## What the System CAN Do

- **Text chat**: General-purpose language model conversations
- **Image QA**: Answer questions about images ("What color is the car?")
- **ASR (Automatic Speech Recognition)**: Transcribe speech to text
- **TTS (Text-to-Speech)**: Generate spoken audio from text
- **OCR (Optical Character Recognition)**: Read text from images

## What the System CANNOT Do

- **Image generation**: No decoder that produces pixels (would need a diffusion model or GAN)
- **Video understanding**: Processes single frames only, no temporal video modeling
- **Real-time streaming**: Processes complete inputs, not streaming chunks
- **Music generation**: RVQ codec is optimized for speech, not music
- **Multilingual**: Tokenizer is trained on English data primarily

---

## How the Pieces Connect at Inference

Here is the actual data flow when a user sends an image with a text question:

```
User sends: image.jpg + "What is this?"

1. Image pipeline:
   image.jpg -> resize to (3,224,224)
             -> ViT-Tiny -> CLS token (1,192)
             -> Linear projection -> (1,384)

2. Text pipeline:
   "What is this?" -> tokenizer -> [1724, 338, 445, 29973]
                    -> tok_emb -> (4, 384)

3. Concatenation:
   [img_token, text_token_1, text_token_2, text_token_3, text_token_4]
   => (5, 384)

4. Thinker processes with causal attention:
   => generates text tokens autoregressively
   => "This is a photograph of a cat."

5. (Optional) Talker generates speech:
   => Thinker hidden states -> Talker -> RVQ codes -> vocoder -> audio
```

---

## Key Design Decisions

1. **Single CLS token for images**: Rather than sending all 196 patches (14x14) to the Thinker, we compress each image to 1 token. This saves context length but loses spatial detail. The OCR model uses full patch sequences when spatial detail matters.

2. **Shared dimension (384)**: All modality encoders project to the same 384-dimensional space. The Vision Encoder internally uses d=192 but projects up to 384 before concatenation.

3. **Frozen encoders during SFT**: During supervised fine-tuning (Stage E), the audio and vision encoders are loaded from pretrained checkpoints. Only the Thinker and projection layers are fine-tuned.

4. **Separate OCR model**: Rather than routing OCR through the Thinker, a dedicated encoder-decoder model handles OCR. This gives better accuracy for text extraction because it uses all patch features with cross-attention.

5. **Sliding Window Attention (SWA)**: Even-numbered Thinker layers use sliding window attention (attending only to the last `window_size` tokens), while odd-numbered layers use full attention. This reduces memory from O(T^2) to O(T*W) on half the layers while preserving long-range reasoning on the other half. Controlled via `window_size` in config (0 = disabled).

6. **YaRN RoPE Extension**: The RoPE module supports `scaling_factor` for extending context length beyond training length using NTK-by-parts interpolation with mscale correction. This allows the model to generalize to longer sequences at inference time without retraining.

7. **Multi-Token Prediction (MTP)**: The Thinker includes 2 auxiliary prediction heads that predict tokens t+2 and t+3 in addition to the main next-token (t+1) prediction. Enabled via `use_mtp: true`. During training, the MTP loss is averaged with the main LM loss, providing richer gradient signal and improving sample efficiency.

---

*Next: Chapter 13 dives deep into the Thinker -- the "brain" that makes sense of all these modalities.*
