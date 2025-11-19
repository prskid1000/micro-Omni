# μOmni Documentation - Study Guide

Complete learning documentation for the μOmni multimodal AI system, covering everything from AI basics to advanced implementation details.

---

## 📚 Documentation Overview

This comprehensive guide contains **45 chapters** organized into 8 parts, designed to take you from zero AI knowledge to complete understanding of the μOmni codebase.

### ✅ Available Chapters (Complete with Diagrams & Examples)

#### **Part 1: Foundation - Understanding AI Basics**
- ✅ [Chapter 01: What is Artificial Intelligence?](01-what-is-ai.md)
- ✅ [Chapter 02: Neural Networks Fundamentals](02-neural-networks-basics.md)
- ✅ [Chapter 03: How Neural Networks Learn](03-training-basics.md)
- ✅ [Chapter 04: Introduction to Transformers](04-transformers-intro.md)
- ✅ [Chapter 05: What is Multimodal AI?](05-multimodal-ai.md)

#### **Part 2: Core Concepts**
- ✅ [Chapter 06: Understanding Embeddings](06-embeddings-explained.md)
- ✅ [Chapter 07: Attention Mechanism Deep Dive](07-attention-mechanism.md)
- ✅ [Chapter 08: Positional Encodings (RoPE)](08-positional-encoding.md)
- ✅ [Chapter 09: Tokenization and Vocabularies](09-tokenization.md)
- ✅ [Chapter 10: Audio Processing for AI](10-audio-processing.md)
- ✅ [Chapter 11: Image Processing for AI](11-image-processing.md)
- ✅ [Chapter 12: Vector Quantization](12-quantization.md)

#### **Part 4: μOmni Architecture**
- ✅ [Chapter 19: μOmni System Architecture](19-muomni-overview.md)
- ✅ [Chapter 20: The Thinker - Core Language Model](20-thinker-llm.md)

#### **Part 5: Training Pipeline**
- ✅ [Chapter 26: Training Workflow Overview](26-training-overview.md)

#### **Part 7: Practical Usage**
- ✅ [Chapter 38: Setting Up Your Environment](38-setup-environment.md)
- ✅ [Chapter 39: Running Training Scripts](39-running-training.md)
- ✅ [Chapter 40: Running Inference Examples](40-inference-examples.md)

---

## 🎯 Quick Start Guides

### 🆕 For Complete Beginners (Never Studied AI)
**Start here:** [📖 Complete Beginner's Learning Guide](LEARNING-GUIDE.md) ⭐ **HIGHLY RECOMMENDED**

This 4-week structured path takes you from zero to understanding and using μOmni!

**Or start directly:** [Prerequisites](00-prerequisites.md) → [Chapter 01: What is AI?](01-what-is-ai.md)

### For Beginners (Some Tech Background)
**Start here:** [Chapter 01: What is Artificial Intelligence?](01-what-is-ai.md)

Follow the sequential path through Parts 1 and 2 to build foundational understanding.

### For ML Practitioners
**Jump to:** [Chapter 19: μOmni System Architecture](19-muomni-overview.md)

Review system architecture, then explore specific components of interest.

### For Developers (Just Want to Use It)
**Essential chapters:**
1. [Chapter 38: Setting Up Your Environment](38-setup-environment.md)
2. [Chapter 39: Running Training Scripts](39-running-training.md)
3. [Chapter 40: Running Inference Examples](40-inference-examples.md)

---

## 📖 Complete Chapter List

See [00-INDEX.md](00-INDEX.md) for the complete table of contents with all 45 chapters listed.

---

## 🎨 Documentation Features

Each chapter includes:

- **📊 Visual Diagrams**: ASCII art flowcharts and architecture diagrams
- **📈 Tables**: Comparison tables and parameter specifications
- **💻 Code Examples**: Practical code snippets from the codebase
- **🎓 Self-Check Questions**: Test your understanding
- **💡 Key Takeaways**: Summary of main concepts
- **⚡ Pro Tips**: Advanced insights and best practices

## 🚀 Recent Optimizations (2024)

**Memory Efficiency:**
- ✅ **Lazy dataset loading** - All training scripts use file offset indexing (90%+ RAM reduction)
- ✅ **Efficient tokenizer training** - Plain text passed directly to SentencePiece. CSV/JSON streams text extraction to temp file.
- ✅ **Smart temp file usage** - Only used for CSV/JSON text extraction (streams extraction), stored in `data/.temp/` and auto-cleaned
- ✅ **Resumable preprocessing** - Vocabulary building and token counting can resume if interrupted
- ✅ **Automatic checkpointing** - Progress saved every 10K items for safe resumption

**Performance:**
- ✅ **Training loops** - All scripts support resumable training with checkpoints
- ✅ **Mixed precision** - FP16 enabled by default for 2x speedup
- ✅ **Gradient accumulation** - Automatic adjustment based on model size
- ✅ **Fast config updates** - Skip tokenization mode for large datasets (`--skip-text-tokenization --assume-text-tokens N`)

---

## 🔗 Related Resources

### Official Documentation
- [Main README](../README.md) - Project overview and quick start
- [Requirements](../requirements.txt) - Python dependencies
- [Configs](../configs/) - Configuration files for each component

### Code Structure
```
../
├── omni/              # Core modules
│   ├── thinker.py    # Language model
│   ├── audio_encoder.py
│   ├── vision_encoder.py
│   ├── talker.py     # Speech generator
│   └── codec.py      # RVQ + vocoder
├── train_*.py        # Training scripts
├── infer_chat.py     # Inference interface
└── configs/          # JSON configurations
```

---

## 🤝 Contributing

Found an error or want to improve the documentation?

1. Each chapter is a standalone Markdown file
2. Follow the existing format (diagrams, tables, examples)
3. Include self-check questions and key takeaways
4. Update the index if adding new chapters

---

## 📊 Documentation Statistics

- **Total Chapters**: 45
- **Completed**: 16 (comprehensive)
- **Remaining**: 29 (outlines available in index)
- **Total Words**: ~50,000+
- **Code Examples**: 100+
- **Diagrams**: 80+
- **Tables**: 50+

---

## 🎓 Learning Path Recommendations

### Path 1: Complete Beginner (4-6 weeks)
```
Week 1: Chapters 1-5 (Foundation)
Week 2: Chapters 6-9 (Core Concepts Part 1)
Week 3: Chapters 10-12 (Core Concepts Part 2)
Week 4: Chapter 19 (System Overview)
Week 5: Chapters 38-40 (Practical Usage)
Week 6: Hands-on experimentation
```

### Path 2: ML Engineer (1-2 weeks)
```
Day 1-2: Chapters 4-5, 19 (Transformers + μOmni)
Day 3-4: Chapters 6-12 (Core Concepts)
Day 5-7: Chapters 38-40 (Practical)
Week 2: Deep dives into specific components
```

### Path 3: Quick Start (2-3 days)
```
Day 1: Chapter 19 (System Architecture)
Day 2: Chapter 38-39 (Setup + Training)
Day 3: Chapter 40 (Inference)
```

---

## 🚀 Next Steps

1. **Start Learning**: Begin with [Chapter 01](01-what-is-ai.md) or jump to your interest area
2. **Set Up Environment**: Follow [Chapter 38](38-setup-environment.md)
3. **Run Training**: Use [Chapter 39](39-running-training.md) as guide
4. **Try Inference**: Experiment with [Chapter 40](40-inference-examples.md) examples
5. **Explore Code**: Dive into the actual implementation files

---

## 📧 Feedback

Questions or suggestions? This documentation is meant to be comprehensive yet accessible. Feedback helps improve it for everyone!

---

**Last Updated**: November 2025  
**Version**: 1.0  
**Documentation Progress**: 16/45 chapters complete (core topics covered)

---

Happy Learning! 🎉

