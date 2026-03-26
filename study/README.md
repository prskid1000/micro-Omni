# μOmni Study Guide — Zero to Master

Learn how to build a multimodal AI system from scratch. No prior AI knowledge required.

## What is This?

A complete, self-contained learning guide that takes you from "What is AI?" to building and deploying a working multimodal model that understands text, images, and speech — all on a single GPU.

Every concept is explained with real-life analogies, ASCII diagrams, and step-by-step walkthroughs. No external textbooks or courses needed.

## How to Use This Guide

### Pick Your Path

| You are... | Start here | Time |
|------------|------------|------|
| Complete beginner (never studied AI) | Chapter 01 → read sequentially | ~15 hours |
| Know some programming | Chapter 03 (Transformers) → 04-07 → 12-17 → 19 | ~8 hours |
| ML engineer, new to multimodal | Chapter 08-11 → 12-17 → 19-20 → 24 | ~5 hours |
| Just want to run it | Chapter 22 → 23 → 24 | ~1 hour |

### Structure

```
Part 1: Foundations          (Chapters 01-03)  — AI, neural nets, transformers
Part 2: Building Blocks      (Chapters 04-07)  — tokens, audio, images, normalization
Part 3: Advanced Techniques  (Chapters 08-11)  — LLMs, efficient attention, MoE, VQ
Part 4: μOmni Architecture   (Chapters 12-17)  — the actual system components
Part 5: Training & Running   (Chapters 18-21)  — data, training pipeline, optimization
Part 6: Deployment & Testing (Chapters 22-25)  — setup, inference, export, testing
Appendices A-E               — math, papers, configs, code structure, customization
```

See [00-INDEX.md](00-INDEX.md) for the full table of contents.

### Tips

- Each chapter is self-contained — you can skip around
- ASCII diagrams show data flow with actual tensor shapes from μOmni
- Config values shown are from the synthetic configs
- Commands shown are copy-paste ready for Windows (bash syntax)

## What You'll Be Able to Do After

- Understand how transformer-based AI models work from first principles
- Explain multimodal fusion (how one model handles text + images + audio)
- Train each component of μOmni on your own GPU
- Run inference: text chat, image Q&A, speech transcription, text-to-speech
- Export and deploy the model
- Tune performance for your specific hardware

## Quick Facts About μOmni

| | |
|---|---|
| **Total parameters** | ~13.9 million (synthetic config) |
| **GPU requirement** | 12-16GB VRAM (e.g., RTX 3060, 4070, 5070 Ti) |
| **Modalities** | Text + Images + Audio (in and out) |
| **Architecture** | Thinker (LLM) + Talker (speech) + encoders |
| **Training time** | Hours per stage on a single GPU |
| **Based on** | Thinker-Talker architecture (inspired by modern multimodal LLMs) |
