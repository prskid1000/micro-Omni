# Chapter 34: Configuration Files Explained

[← Previous: Code Structure](33-code-structure.md) | [Back to Index](00-INDEX.md) | [Next: Data Preparation →](35-data-preparation.md)

---

## 📋 Overview

All model and training parameters are configured via JSON files in `configs/`.

---

## 🧠 thinker_tiny.json

```json
{
  "vocab_size": 5000,        // Tokenizer vocabulary size
  "n_layers": 4,             // Transformer layers
  "d_model": 256,            // Embedding dimension
  "n_heads": 4,              // Attention heads
  "d_ff": 1024,              // Feedforward dimension
  "dropout": 0.1,            // Dropout rate
  "rope_theta": 10000,       // RoPE theta parameter
  "ctx_len": 512,            // Context length
  
  // Optional optimizations
  "use_gqa": false,          // Grouped Query Attention
  "use_swiglu": true,        // SwiGLU activation
  "use_moe": false,          // Mixture of Experts
  "num_experts": 8,          // If MoE enabled
  
  // Training
  "data_path": "data/text/corpus.txt",
  "batch_size": 16,
  "num_epochs": 10,
  "learning_rate": 3e-4,
  "warmup_steps": 1000,
  "max_grad_norm": 1.0,      // Gradient clipping
  
  // Checkpointing
  "checkpoint_dir": "checkpoints/thinker_tiny",
  "save_every": 1000,
  "eval_every": 500
}
```

### Key Parameters

| Parameter | Effect | Recommendation |
|-----------|--------|----------------|
| **n_layers** | More = deeper model | 4 (12GB GPU) |
| **d_model** | Embedding size | 256 (efficient) |
| **n_heads** | Attention heads | 4 |
| **ctx_len** | Max sequence length | 512-2048 |

---

## 🎤 audio_enc_tiny.json

```json
{
  "d_model": 192,            // Encoder dimension
  "n_layers": 4,             // Transformer layers
  "n_heads": 3,              // Attention heads
  "d_ff": 768,               // Feedforward dimension
  "dropout": 0.1,
  "downsample_time": 8,      // Temporal downsampling (4x or 8x)
  
  // Audio processing
  "sample_rate": 16000,      // Audio sample rate
  "n_fft": 1024,            // FFT size
  "hop_length": 160,         // STFT hop (100Hz frame rate)
  "n_mels": 128,            // Mel bins
  
  // Training
  "data_path": "data/audio/asr.csv",
  "batch_size": 8,
  "num_epochs": 20,
  "learning_rate": 1e-4,
  
  "checkpoint_dir": "checkpoints/audio_enc_tiny"
}
```

### Frame Rate Calculation

```
Input: 16kHz audio
After mel: 16000 / 160 = 100 Hz
After downsample (8x): 100 / 8 = 12.5 Hz

3 seconds audio = 3 × 12.5 = 37.5 ≈ 38 frames
```

---

## 🖼️ vision_tiny.json

```json
{
  "img_size": 224,           // Input image size
  "patch": 16,               // Patch size (16×16)
  "d_model": 128,            // Encoder dimension
  "n_layers": 4,             // Transformer layers
  "n_heads": 2,              // Attention heads
  "d_ff": 512,               // Feedforward dimension
  "dropout": 0.1,
  
  // Training
  "data_path": "data/images/",
  "batch_size": 32,
  "num_epochs": 15,
  "learning_rate": 3e-4,
  
  "checkpoint_dir": "checkpoints/vision_tiny"
}
```

### Patch Calculation

```
Image: 224×224
Patch: 16×16
Grid: 224/16 = 14 patches per side
Total patches: 14×14 = 196
+ 1 CLS token = 197 tokens
```

---

## 🔊 talker_tiny.json

```json
{
  "d_model": 192,            // Decoder dimension
  "n_layers": 4,             // Transformer layers
  "n_heads": 3,              // Attention heads
  "d_ff": 768,               // Feedforward dimension
  "dropout": 0.1,
  
  // RVQ codec
  "codebooks": 2,            // Number of codebooks
  "codebook_size": 128,      // Codes per codebook
  "codebook_dim": 64,        // Codebook embedding dim
  
  // Training
  "data_path": "data/audio/tts/",
  "batch_size": 16,
  "num_epochs": 25,
  "learning_rate": 3e-4,
  
  "checkpoint_dir": "checkpoints/talker_tiny"
}
```

---

## 🌈 omni_sft_tiny.json

```json
{
  // Pretrained checkpoints
  "thinker_ckpt": "checkpoints/thinker_tiny/thinker_best.pt",
  "audio_ckpt": "checkpoints/audio_enc_tiny/audio_enc.pt",
  "vision_ckpt": "checkpoints/vision_tiny/vision.pt",
  "talker_ckpt": "checkpoints/talker_tiny/talker.pt",
  
  // Training strategy
  "freeze_encoders": true,   // Only train Thinker + projectors
  "freeze_thinker": false,   // Allow Thinker to adapt
  
  // Data mixing
  "data_mix": {
    "text_only": 0.4,        // 40% text-only samples
    "image_text": 0.3,       // 30% image + text
    "audio_text": 0.3        // 30% audio + text
  },
  
  // Training
  "batch_size": 8,           // Smaller due to multimodal
  "num_epochs": 5,           // Few epochs (fine-tuning)
  "learning_rate": 1e-4,     // Lower LR for fine-tuning
  "warmup_steps": 500,
  
  "checkpoint_dir": "checkpoints/omni_sft_tiny"
}
```

---

## 🔧 Modifying Configurations

### Increase Model Capacity

```json
{
  "n_layers": 6,      // 4 → 6
  "d_model": 384,     // 256 → 384
  "n_heads": 6,       // 4 → 6
  "d_ff": 1536        // 1024 → 1536
}
```

**Effect**: Better quality, more memory, slower training

### Decrease for Faster Training

```json
{
  "n_layers": 2,      // 4 → 2
  "d_model": 128,     // 256 → 128
  "batch_size": 32    // 16 → 32
}
```

**Effect**: Faster training, less memory, lower quality

### Longer Context

```json
{
  "ctx_len": 2048     // 512 → 2048
}
```

**Effect**: Longer conversations, more memory

### Enable Optimizations

```json
{
  "use_gqa": true,           // Efficient attention
  "use_swiglu": true,        // Better activation
  "use_gradient_checkpointing": true  // Save memory
}
```

---

## 💡 Configuration Best Practices

### 1. Start Small, Scale Up
```json
// Initial test
{"batch_size": 4, "num_epochs": 1}

// Full training
{"batch_size": 16, "num_epochs": 10}
```

### 2. Match Dimensions
```
Thinker d_model: 256
Audio → Project: 192 → 256 ✓
Vision → Project: 128 → 256 ✓
All must align!
```

### 3. Learning Rate Schedule
```json
{
  "learning_rate": 3e-4,     // Pretraining
  "warmup_steps": 1000       // Gradual warmup
}

{
  "learning_rate": 1e-4,     // Fine-tuning (lower!)
  "warmup_steps": 500
}
```

### 4. Memory Management
```json
{
  "batch_size": 8,                    // Reduce if OOM
  "use_gradient_checkpointing": true, // Save memory
  "use_amp": true                     // Mixed precision
}
```

---

## 💡 Key Takeaways

✅ **JSON configs** control all parameters  
✅ **Modular** - each component has own config  
✅ **Easy to modify** - no code changes needed  
✅ **Dimension matching** critical for multimodal

---

[Continue to Chapter 35: Data Preparation →](35-data-preparation.md)

