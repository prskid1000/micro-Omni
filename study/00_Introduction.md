# Introduction to μOmni

## What is μOmni?

μOmni (pronounced "micro-omni") is a **multimodal AI model** that can understand and generate:
- **Text** - Read and write sentences
- **Images** - See and describe pictures
- **Audio** - Hear speech and generate speech

Think of it like a human brain that can process multiple types of information at once!

## Why "Tiny"?

The "tiny" in μOmni means it's designed to:
- Fit on a single 12GB GPU (most AI models need much more)
- Train quickly with small datasets (< 5GB each)
- Be easy to understand and modify

This makes it perfect for learning!

## Real-World Analogy

Imagine you're learning a new language:

1. **Thinker** = Your brain that understands and generates language
2. **Audio Encoder** = Your ears that convert sound to meaning
3. **Vision Encoder** = Your eyes that convert images to meaning
4. **Talker** = Your mouth that converts thoughts to speech

μOmni works similarly - it has separate "senses" that feed into a central "brain."

## What Can μOmni Do?

### Input Modes:
- 📝 **Text**: "What is the weather?"
- 🖼️ **Image**: A photo of a cat
- 🎤 **Audio**: A spoken question
- 🎬 **Video**: A short clip

### Output Modes:
- 📝 **Text**: Written responses
- 🔊 **Audio**: Spoken responses (text-to-speech)

### Combined:
- See an image + hear audio → Generate text response
- Read text → Generate spoken audio
- And more combinations!

## Key Concepts You'll Learn

1. **Neural Networks** - How computers "learn"
2. **Transformers** - The architecture powering modern AI
3. **Multimodal Fusion** - Combining different data types
4. **Training** - Teaching the model with examples
5. **Inference** - Using the trained model

## Project Structure

```
μOmni/
├── omni/              # Core model code
│   ├── thinker.py     # Language model
│   ├── audio_encoder.py
│   ├── vision_encoder.py
│   └── talker.py
├── train_*.py         # Training scripts
├── infer_chat.py      # Inference interface
├── configs/           # Configuration files
└── study/             # This guide!
```

## What Makes This Special?

Most AI models are:
- ❌ Only text OR images OR audio
- ❌ Require huge datasets (terabytes)
- ❌ Need expensive hardware
- ❌ Hard to understand

μOmni is:
- ✅ All modalities in one model
- ✅ Works with small datasets
- ✅ Runs on consumer GPUs
- ✅ Code is readable and educational

## Learning Goals

By the end of this guide, you'll understand:
- How neural networks process information
- How μOmni's architecture works
- How to train your own model
- How to use trained models for inference
- How to modify and experiment

## Prerequisites Check

Before continuing, make sure you can:
- ✅ Write a Python function
- ✅ Understand classes and objects
- ✅ Read and write files
- ✅ Use imports

If you're comfortable with these, you're ready!

---

**Next:** [01_Neural_Networks_Basics.md](01_Neural_Networks_Basics.md) - Learn the fundamentals

**See Also:**
- [Architecture Overview](02_Architecture_Overview.md)
- [Main README](../README.md)

