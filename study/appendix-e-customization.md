# Appendix E: Customization & Future Extensions

How to scale, extend, and adapt micro-Omni for your own needs.

---

## Scaling Model Size

micro-Omni is designed to scale from tiny research models to larger
production-grade systems. Adjust config parameters to move between tiers:

| Size    | Params | d_model | n_layers | n_heads | d_ff  | Notes              |
|---------|--------|---------|----------|---------|-------|--------------------|
| Tiny    | ~25M   | 512     | 8        | 8       | 1376  | Single GPU, fast   |
| Small   | ~50M   | 768     | 12       | 12      | 2048  | Single GPU         |
| Base    | ~150M  | 1024    | 16       | 16      | 2816  | 24GB GPU           |
| Large   | ~500M+ | 2048    | 24       | 32      | 5504  | Multi-GPU or A100  |

### Scaling Rules of Thumb

```
d_ff = ~2.7 * d_model     (SwiGLU optimal ratio)
n_heads = d_model / 64    (64 dims per head)
kv_groups = n_heads / 4   (when using GQA)
```

When scaling up:
1. Increase `d_model` first (biggest impact on capacity)
2. Add layers for deeper reasoning
3. Enable GQA at Base size and above (saves KV cache memory)
4. Enable MoE at Large size for conditional computation

### Memory Estimation

```
Parameters (float32):  params * 4 bytes
Parameters (float16):  params * 2 bytes
Optimizer (AdamW):     params * 12 bytes  (param + m + v, float32)
Activations:           ~2x parameter memory (batch_size dependent)

Example (Tiny, float16, batch_size=32):
  Model:       25M * 2B  =  50 MB
  Optimizer:   25M * 12B = 300 MB
  Activations: ~100 MB
  Total:       ~450 MB   (fits easily on any modern GPU)
```

---

## Adding New Modalities

### Video Understanding

Video is the most natural next modality. Approach:

```
Video (T frames)
    |
    v
Extract frames at N fps
    |
    v
+---+---+---+---+
| F1| F2| F3|...|    Each frame: (3, 224, 224)
+---+---+---+---+
    |
    v (per frame)
ViTTiny encoder
    |
    v
Frame embeddings: (N, embed_dim)
    |
    v
Temporal attention (new module)
    |
    v
Projection to Thinker space
    |
    v
Thinker processes as token sequence
```

Implementation steps:
1. Extract frames: `torchvision.io.read_video()` or `ffmpeg`
2. Encode each frame with the existing ViT encoder
3. Add a temporal transformer (2-4 layers) over frame embeddings
4. Project to Thinker's `d_model` dimension
5. Concatenate with text tokens and pass to Thinker

The temporal transformer is the only new component. Everything else
reuses existing modules.

### Other Modalities

| Modality     | Encoder Strategy                          | Training Signal        |
|--------------|-------------------------------------------|------------------------|
| Video        | ViT per frame + temporal attention        | Video-text pairs       |
| 3D/Point Cloud| PointNet-style encoder                   | Shape-text pairs       |
| Music        | Same audio encoder, different data        | Music-caption pairs    |
| Code         | Same thinker, code-specific tokenizer     | Code completion        |
| Structured Data| Linearize tables to text                | Table QA pairs         |

---

## Arthemis Neuromorphic Extensions

micro-Omni includes experimental brain-inspired computing features.

### Spiking Neurons (use_spiking)

Replaces continuous activations with discrete spike events using
Leaky Integrate-and-Fire (LIF) neurons:

```
Config: "use_spiking": true, "spike_thresh": 1.0

Membrane potential update:
  V[t] = beta * V[t-1] + W * x[t]

Spike output:
  S[t] = 1 if V[t] >= threshold, else 0

After spike:
  V[t] = V[t] - threshold   (soft reset)
```

Where `beta` is a learnable decay factor. Spiking neurons are inserted
in the attention mechanism, making attention weights binary (spike / no
spike) rather than continuous.

Benefits:
- Potential for neuromorphic hardware deployment
- Natural temporal processing for audio/video
- Built-in regularization (sparse activations)

Trade-offs:
- Slightly lower accuracy on text tasks
- Surrogate gradient needed for backpropagation (straight-through estimator)

### Liquid Time-Constant Neurons (use_ltc)

Replaces standard FFN neurons with adaptive-time-constant dynamics:

```
Config: "use_ltc": true, "ltc_tau": 1.0

tau[t] = sigmoid(W_tau * x[t] + b_tau) * tau_max
dh/dt = (-h[t] + f(W * x[t])) / tau[t]
h[t+1] = h[t] + dh/dt * dt
```

Each neuron has its own learned time constant `tau` that adapts based on
input. Neurons processing fast-changing inputs (audio) get small tau
(fast response); slow-changing inputs (text context) get large tau
(long memory).

Benefits:
- Adaptive temporal dynamics per neuron
- Better at multi-timescale tasks
- Principled continuous-time formulation

---

## Domain Fine-Tuning

Use the SFT (Supervised Fine-Tuning) stage to adapt the model:

```bash
python sft_omni.py \
    --base_ckpt checkpoints/        \
    --data data/domain_specific/    \
    --config configs/sft.json       \
    --output_dir checkpoints/sft/
```

### SFT Data Format

Prepare conversation-style data:

```json
[
  {
    "messages": [
      {"role": "user", "content": "What is the diagnosis for this X-ray?", "image": "xray_001.jpg"},
      {"role": "assistant", "content": "The X-ray shows..."}
    ]
  }
]
```

### Domain-Specific Tips

| Domain     | Data Needed         | Recommended Steps | Notes                    |
|------------|---------------------|-------------------|--------------------------|
| Medical    | 10K+ QA pairs       | 5000-10000        | Use domain tokenizer     |
| Legal      | 5K+ doc summaries   | 3000-5000         | Long context important   |
| Code       | 50K+ completions    | 10000-20000       | Add code tokens to vocab |
| Education  | 10K+ explanations   | 5000-10000        | Multi-turn conversations |

### Key SFT Parameters

```json
{
  "lr": 1e-5,
  "warmup_steps": 100,
  "max_steps": 5000,
  "batch_size": 8,
  "gradient_accumulation_steps": 4,
  "freeze_vision": true,
  "freeze_audio": true,
  "lora_rank": 0
}
```

Lower learning rate than pretraining. Optionally freeze encoder
components to prevent catastrophic forgetting.

---

## Implemented Features Summary

| Feature           | Status       | Module              |
|-------------------|-------------|---------------------|
| Text generation   | Implemented | thinker.py          |
| Audio encoding    | Implemented | audio_encoder.py    |
| Vision encoding   | Implemented | vision_encoder.py   |
| Audio generation  | Implemented | talker.py, codec.py |
| HiFi-GAN vocoder  | Implemented | codec.py            |
| OCR               | Implemented | ocr_model.py        |
| Spiking neurons   | Implemented | thinker.py          |
| Liquid time-const | Implemented | thinker.py          |
| GQA               | Implemented | thinker.py          |
| MoE               | Implemented | thinker.py          |
| SwiGLU            | Implemented | thinker.py, utils.py|
| RoPE              | Implemented | utils.py            |
| YaRN RoPE scaling | Implemented | utils.py            |
| FlashAttention    | Implemented | thinker.py          |
| Sliding Window Attn| Implemented | thinker.py         |
| Multi-Token Pred  | Implemented | thinker.py          |
| Export/Deploy     | Implemented | export.py           |

---

## Research Findings: Qwen3.5 Architecture Analysis

Analysis of the Qwen3.5 family reveals several architectural patterns relevant to scaling micro-Omni:

| Feature | Qwen3.5 Approach | micro-Omni Status |
|---------|------------------|-------------------|
| **GQA ratios** | Aggressive KV sharing (e.g., 32 query heads / 8 KV groups) | Implemented (configurable `kv_groups`) |
| **Talker-Thinker connection** | Thinker hidden states directly condition the Talker | Implemented (Stage D) |
| **Audio frame rate** | 12.5Hz (same as our target_hz) | Implemented (8x downsample from 100Hz) |
| **Gated DeltaNet** | Linear attention variant for efficient long-context | Not implemented (future extension) |

Key takeaways for scaling:
- **GQA is essential** at Base size and above. Qwen3.5 uses 4:1 query-to-KV ratios.
- **12.5Hz audio** is validated as the sweet spot for speech -- matches our existing design.
- **Gated DeltaNet** is a promising alternative to standard attention for very long contexts, offering O(T) complexity instead of O(T^2). Could replace sliding window attention on even layers.

---

## Future Ideas

### Streaming Inference

Currently, the model processes complete inputs before generating output.
Streaming would allow:
- Real-time audio transcription (process chunks as they arrive)
- Token-by-token TTS (start speaking before full response is generated)
- Live video understanding (process frames as captured)

Architecture change: add a causal buffer that accumulates encoder outputs
and triggers Thinker generation once enough context is available.

### Speaker Diarization

Identify and separate different speakers in multi-speaker audio:
- Add a speaker embedding head to the audio encoder
- Cluster embeddings to identify speaker turns
- Tag transcription with speaker labels

### Image Generation

Add a visual decoder to generate images from text:
- Train a small diffusion model or VQ-VAE decoder
- Condition on Thinker hidden states (like Talker does for audio)
- Generate image tokens that decode to pixels

### Reinforcement Learning from Human Feedback (RLHF)

Align model outputs with human preferences:
1. Collect preference pairs (response A > response B)
2. Train a reward model
3. Fine-tune with PPO or DPO

### Quantization for Edge Deployment

Reduce model size for mobile/embedded deployment:
- INT8 quantization: ~4x smaller, minimal quality loss
- INT4 quantization: ~8x smaller, some quality loss
- Use `torch.ao.quantization` or GPTQ

```
Tiny (float16):   50 MB
Tiny (INT8):      25 MB
Tiny (INT4):      12 MB   <-- runs on mobile
```
