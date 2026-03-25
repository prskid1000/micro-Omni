# μOmni — Zero to Master Learning Guide

A self-contained guide to understanding and building a multimodal AI system. No external materials needed.

---

## Part 1: Foundations (Start here if new to AI)

| # | Chapter | What You'll Learn |
|---|---------|-------------------|
| 01 | [What is AI?](01-what-is-ai.md) | AI vs ML vs Deep Learning, how machines learn from data |
| 02 | [Neural Networks & Training](02-neural-networks.md) | Neurons, layers, backpropagation, optimizers, loss functions |
| 03 | [The Transformer](03-transformers.md) | Attention, Q/K/V, multi-head attention, transformer blocks |

## Part 2: Building Blocks

| # | Chapter | What You'll Learn |
|---|---------|-------------------|
| 04 | [Tokens, Embeddings & Position](04-tokens-embeddings-position.md) | BPE tokenization, dense embeddings, RoPE positional encoding |
| 05 | [Audio: Sound to Tokens](05-audio-processing.md) | Waveforms, mel spectrograms, convolutional downsampling |
| 06 | [Images: Pixels to Patches](06-image-processing.md) | Vision Transformers, patch embeddings, CLS tokens |
| 07 | [Normalization & Activations](07-normalization-activations.md) | RMSNorm, SwiGLU, GELU — why they matter for training stability |

## Part 3: Advanced Techniques

| # | Chapter | What You'll Learn |
|---|---------|-------------------|
| 08 | [Decoder-Only LLMs & KV Caching](08-decoder-llm-kv-cache.md) | Causal masking, autoregressive generation, KV cache speedup |
| 09 | [Efficient Attention: GQA & Flash](09-efficient-attention.md) | Grouped Query Attention, Flash Attention, memory savings |
| 10 | [Mixture of Experts](10-mixture-of-experts.md) | Sparse routing, expert specialization, load balancing |
| 11 | [Vector Quantization & Speech Codes](11-vector-quantization.md) | VQ, RVQ, codebooks, converting continuous audio to discrete tokens |

## Part 4: μOmni Architecture

| # | Chapter | What You'll Learn |
|---|---------|-------------------|
| 12 | [System Overview & Multimodal Fusion](12-system-overview.md) | Full architecture, how text+image+audio combine in one model |
| 13 | [The Thinker: Core LLM](13-thinker.md) | Decoder-only LM, multimodal input, Arthemis extensions |
| 14 | [Audio Encoder](14-audio-encoder.md) | AuT-Tiny: mel → transformer → CTC for ASR |
| 15 | [Vision Encoder](15-vision-encoder.md) | ViT-Tiny: CLIP contrastive training, image understanding |
| 16 | [Talker & Speech Generation](16-talker-speech.md) | AR code prediction, RVQ codec, HiFi-GAN/Griffin-Lim vocoder |
| 17 | [OCR Model](17-ocr-model.md) | Vision encoder + text decoder for reading text from images |

## Part 5: Training & Running

| # | Chapter | What You'll Learn |
|---|---------|-------------------|
| 18 | [Data Preparation](18-data-preparation.md) | Formats, download scripts, dataset requirements per stage |
| 19 | [Training Pipeline: All Stages](19-training-pipeline.md) | Stage A-E walkthrough, configs, loss curves, checkpointing |
| 20 | [Performance & Optimization](20-performance-optimization.md) | AMP, Flash Attention, gradient accumulation, 16GB VRAM tips |
| 21 | [Debugging & Troubleshooting](21-debugging.md) | NaN fixes, gradient explosion, common errors and solutions |

## Part 6: Deployment & Testing

| # | Chapter | What You'll Learn |
|---|---------|-------------------|
| 22 | [Setup & Environment](22-setup-environment.md) | Installation, dependencies, GPU setup, verification |
| 23 | [Inference & Chat](23-inference-chat.md) | Text chat, image QA, audio transcription, TTS |
| 24 | [Export & Deployment](24-export-deployment.md) | Merging to safetensors, HuggingFace integration, from_pretrained, standalone inference |
| 25 | [Testing & Validation](25-testing-validation.md) | Test scripts, metrics, quality checks |

## Appendices

| # | Appendix | Content |
|---|----------|---------|
| A | [Mathematical Foundations](appendix-a-math.md) | All formulas: attention, RoPE, losses, optimizers |
| B | [Research Papers](appendix-b-papers.md) | Papers that inspired μOmni, mapped to components |
| C | [Configuration Reference](appendix-c-configs.md) | Every JSON config parameter explained |
| D | [Code Structure](appendix-d-code-structure.md) | File map, module responsibilities, class hierarchy |
| E | [Customization & Future](appendix-e-customization.md) | Scaling, new modalities, Arthemis extensions, roadmap |

---

## Learning Paths

**Complete Beginner** (never studied AI): 01 → 02 → 03 → 04 → 12 → 22 → 23
**Know Some Programming**: 03 → 04-07 → 12-17 → 19 → 23
**ML Engineer**: 08-11 → 12-17 → 19-20 → 24
**Just Want to Run It**: 22 → 23 → 24

---

*25 chapters + 5 appendices. ~15 hours total reading. Each chapter is self-contained.*
