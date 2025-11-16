# μOmni Documentation - Complete Learning Guide

**From Zero AI Knowledge to Full Understanding of the Codebase**

---

## 📚 Table of Contents

### **Part 1: Foundation - Understanding AI Basics (Chapters 1-5)**

| Chapter | Title | Description |
|---------|-------|-------------|
| [01](01-what-is-ai.md) | What is Artificial Intelligence? | Introduction to AI, Machine Learning, Deep Learning concepts |
| [02](02-neural-networks-basics.md) | Neural Networks Fundamentals | Understanding neurons, layers, and basic neural network architecture |
| [03](03-training-basics.md) | How Neural Networks Learn | Backpropagation, loss functions, gradients, and optimization |
| [04](04-transformers-intro.md) | Introduction to Transformers | The revolutionary architecture that changed AI |
| [05](05-multimodal-ai.md) | What is Multimodal AI? | Understanding systems that work with text, images, audio, and video |

---

### **Part 2: Core Concepts (Chapters 6-12)**

| Chapter | Title | Description |
|---------|-------|-------------|
| [06](06-embeddings-explained.md) | Understanding Embeddings | How data is represented as vectors |
| [07](07-attention-mechanism.md) | Attention Mechanism Deep Dive | Self-attention, multi-head attention, and why it matters |
| [08](08-positional-encoding.md) | Positional Encodings (RoPE) | How transformers understand sequence order |
| [09](09-tokenization.md) | Tokenization and Vocabularies | Breaking text into tokens - BPE, WordPiece, SentencePiece |
| [10](10-audio-processing.md) | Audio Processing for AI | Waveforms, spectrograms, mel-spectrograms |
| [11](11-image-processing.md) | Image Processing for AI | Convolutional layers, patches, Vision Transformers |
| [12](12-quantization.md) | Vector Quantization | RVQ, codebooks, and discrete representations |

---

### **Part 3: Advanced Architecture Components (Chapters 13-18)**

| Chapter | Title | Description |
|---------|-------|-------------|
| [13](13-decoder-only-llm.md) | Decoder-Only Language Models | GPT-style architecture and autoregressive generation |
| [14](14-kv-caching.md) | KV Caching Optimization | Making generation fast and efficient |
| [15](15-gqa-attention.md) | Grouped Query Attention (GQA) | Efficient attention mechanism from Qwen |
| [16](16-swiglu-activation.md) | SwiGLU Activation Function | Modern activation for better performance |
| [17](17-mixture-of-experts.md) | Mixture of Experts (MoE) | Scaling models with sparse computation |
| [18](18-normalization.md) | Normalization Techniques | RMSNorm, LayerNorm, and stability |

---

### **Part 4: μOmni Architecture Overview (Chapters 19-25)**

| Chapter | Title | Description |
|---------|-------|-------------|
| [19](19-muomni-overview.md) | μOmni System Architecture | High-level overview of the entire system |
| [20](20-thinker-llm.md) | The Thinker - Core Language Model | Decoder-only LLM that processes multimodal embeddings |
| [21](21-audio-encoder.md) | Audio Encoder (AuT-Tiny) | Converting speech to embeddings |
| [22](22-vision-encoder.md) | Vision Encoder (ViT-Tiny) | Converting images to embeddings |
| [23](23-codec-rvq.md) | RVQ Codec for Speech | Quantizing audio for efficient representation |
| [24](24-talker-speech-gen.md) | The Talker - Speech Generator | Generating speech from text |
| [25](25-multimodal-fusion.md) | Multimodal Fusion Strategy | How different modalities are combined |

---

### **Part 5: Training Pipeline (Chapters 26-31)**

| Chapter | Title | Description |
|---------|-------|-------------|
| [26](26-training-overview.md) | Training Workflow Overview | The 5-stage training pipeline |
| [27](27-stage-a-thinker.md) | Stage A: Thinker Pretraining | Training the core language model on text |
| [28](28-stage-b-audio-encoder.md) | Stage B: Audio Encoder ASR Training | Training audio understanding with CTC loss |
| [29](29-stage-c-vision-encoder.md) | Stage C: Vision Encoder Training | Training image understanding |
| [30](30-stage-d-talker.md) | Stage D: Talker & Codec Training | Training speech generation |
| [31](31-stage-e-sft.md) | Stage E: Multimodal SFT | Supervised fine-tuning for multimodal understanding |

---

### **Part 6: Implementation Details (Chapters 32-37)**

| Chapter | Title | Description |
|---------|-------|-------------|
| [32](32-inference-pipeline.md) | Inference Pipeline | How the system processes inputs and generates outputs |
| [33](33-code-structure.md) | Codebase Structure Guide | File organization and module responsibilities |
| [34](34-configuration-files.md) | Configuration Files Explained | Understanding JSON configs for each component |
| [35](35-data-preparation.md) | Data Preparation and Datasets | How to prepare data for each training stage |
| [36](36-optimization-techniques.md) | Optimization Techniques | Flash Attention, gradient checkpointing, mixed precision |
| [37](37-debugging-troubleshooting.md) | Debugging and Troubleshooting | Common issues and how to fix them |

---

### **Part 7: Practical Usage (Chapters 38-42)**

| Chapter | Title | Description |
|---------|-------|-------------|
| [38](38-setup-environment.md) | Setting Up Your Environment | Step-by-step setup guide |
| [39](39-running-training.md) | Running Training Scripts | How to train each component |
| [40](40-inference-examples.md) | Running Inference Examples | Text chat, image QA, speech I/O |
| [41](41-customization-guide.md) | Customization Guide | Adapting μOmni for your needs |
| [42](42-performance-tuning.md) | Performance Tuning and Scaling | Getting the best performance |

---

### **Part 8: Advanced Topics (Chapters 43-45)**

| Chapter | Title | Description |
|---------|-------|-------------|
| [43](43-mathematical-foundations.md) | Mathematical Foundations | The math behind transformers and attention |
| [44](44-research-papers.md) | Key Research Papers | Papers that influenced this architecture |
| [45](45-future-extensions.md) | Future Extensions and Research | Where to go from here |

---

## 🎯 Learning Paths

### **Beginner Path (Never studied AI)**
Read sequentially: Chapters 1-5 → 6-12 → 19 → 26 → 38-40

### **Intermediate Path (Know basic ML)**
Start with: Chapters 4-5 → 13-18 → 19-25 → 26-31 → 32-37

### **Advanced Path (Experienced ML Engineer)**
Focus on: Chapters 19-25 → 32-37 → 43-45

### **Quick Start Path (Just want to run it)**
Essential chapters: 19 → 38 → 39 → 40

---

## 📊 Visual Learning Elements

Throughout this documentation you will find:

- **📊 Diagrams**: Architecture flowcharts and data flow diagrams
- **📈 Tables**: Comparison tables and parameter specifications  
- **🔢 Math Equations**: Mathematical formulations (when necessary)
- **💻 Code Snippets**: Key code examples from the codebase
- **⚡ Quick Tips**: Practical insights and gotchas
- **🎓 Learning Checkpoints**: Self-assessment questions

---

## 🚀 Getting Started

**New to AI?** → Start with [Chapter 01: What is Artificial Intelligence?](01-what-is-ai.md)

**Want system overview?** → Jump to [Chapter 19: μOmni System Architecture](19-muomni-overview.md)

**Ready to code?** → Go to [Chapter 38: Setting Up Your Environment](38-setup-environment.md)

---

## 📖 Document Conventions

```
📌 **Important Concept**: Key concepts you must understand
⚠️ **Warning**: Common pitfalls and mistakes to avoid
💡 **Pro Tip**: Advanced insights and optimizations
🔍 **Deep Dive**: Links to more detailed explanations
```

---

## 🤝 Contributing to Documentation

This documentation is meant to be living and evolving. If you find errors or want to improve explanations, please contribute!

---

**Last Updated**: November 2025  
**Version**: 1.0  
**Author**: μOmni Documentation Team

