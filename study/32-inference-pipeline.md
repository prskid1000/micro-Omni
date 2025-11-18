# Chapter 32: Inference Pipeline

[← Previous: Stage E SFT](31-stage-e-sft.md) | [Back to Index](00-INDEX.md) | [Next: Code Structure →](33-code-structure.md)

---

## 🎯 Using μOmni After Training

This chapter explains how to use the trained μOmni system for inference across different modalities.

---

## 📝 Inference Modes

### 1. Text-Only Chat

```python
from omni import load_model

# Load trained model
model = load_model('checkpoints/omni_sft_tiny/omni_final.pt')

# Chat
response = model.chat("What is AI?")
print(response)  # "AI is artificial intelligence..."
```

### 2. Multimodal Input (Image + Text)

```python
# Image question answering
response = model.chat(
    text="What animal is this?",
    image="examples/cat.jpg"
)
print(response)  # "This is a cat sitting on a couch"
```

### 3. Audio Input (Speech Recognition)

```python
# Transcribe audio
response = model.chat(
    text="Transcribe this audio",
    audio="examples/hello.wav"
)
print(response)  # "hello world"
```

### 4. Text-to-Speech Output

```python
# Generate speech
audio = model.generate_speech("Hello world, how are you?")
# Saves to output.wav
```

---

## ⚡ Performance Optimizations

**Automatic optimizations enabled:**
- **KV Caching:** Reuses computed keys/values (3-5x speedup)
- **Mixed Precision (FP16):** Faster inference
- **Flash Attention:** Memory-efficient attention (if available)

**Typical speeds (12GB GPU):**
- Text generation: ~30 tokens/second
- Image processing: ~50ms per image
- Audio processing: ~2x real-time

---

## 💡 Key Takeaways

✅ **Simple API** for all modality combinations  
✅ **Autoregressive** text generation  
✅ **KV caching** dramatically speeds up generation  
✅ **Multimodal** via embedding concatenation  
✅ **Flexible** - text, speech, or both outputs

---

## 🔗 Related: Model Export

For deployment, you can export all components into a single safetensors file:

- See [Chapter 46: Model Export and Deployment](46-model-export-deployment.md) for detailed export instructions
- See [Chapter 47: Quick Start Export](47-quick-start-export.md) for a quick reference

The exported model works with `infer_safetensors.py` which provides the same functionality as `infer_chat.py`.

---

[Continue to Chapter 33: Code Structure →](33-code-structure.md)

---
