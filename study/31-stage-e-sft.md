# Chapter 31: Stage E - Multimodal SFT

[← Previous: Stage D Talker](30-stage-d-talker.md) | [Back to Index](00-INDEX.md) | [Next: Inference Pipeline →](32-inference-pipeline.md)

---

## 🎯 Purpose

**Final Stage:** Fine-tune all components together on multimodal data, teaching cross-modal understanding and enabling the system to answer questions about images, transcribe audio, and handle any modality combination.

---

## 📝 The Task

Train on mixed batches with different modality combinations:

- **Text-only:** "What is AI?" → "AI is artificial intelligence..."
- **Image+Text:** [cat image] + "What animal?" → "This is a cat"
- **Audio+Text:** [audio of "hello"] + "Transcribe" → "hello"
- **All modalities:** [image] + [audio] + "Describe" → Multimodal response
- **OCR-enhanced:** [image with text] + OCR extraction → Enhanced understanding with extracted text

---

## 💻 Training Details

### Configuration

```json
{
  // Load pretrained checkpoints
  "thinker_ckpt": "checkpoints/thinker_tiny",
  "audio_ckpt": "checkpoints/audio_enc_tiny",
  "vision_ckpt": "checkpoints/vision_tiny",
  "talker_ckpt": "checkpoints/talker_tiny",
  "ctx_len": 512,
  "lr": 5e-05,
  "wd": 0.01,
  "warmup_steps": 1000,
  "max_steps": 1936100,
  "batch_size": 1,
  "gradient_accumulation_steps": 1,
  "use_amp": true,
  "use_flash": true,
  "use_compile": false,
  "num_workers": 0,
  "drop_last": true,
  "print_freq": 100,
  "max_epochs": 2,
  "val_loss_threshold": 0.05,
  "val_split": 0.1,
  "val_freq": 100,
  "checkpoint_freq": 30,
  "val_batch_size": 2,
  "max_grad_norm": 1.0,
  "use_ema": true,
  "ema_decay": 0.999,
  "seed": 42,
  "shuffle_buffer_size": 100,
  "thinker": {
    "vocab_size": 32000,
    "n_layers": 8,
    "d_model": 384,
    "n_heads": 6,
    "d_ff": 1536,
    "dropout": 0.1,
    "rope_theta": 10000,
    "use_gqa": true,
    "kv_groups": 3,
    "use_swiglu": true,
    "use_moe": false,
    "num_experts": 8,
    "num_experts_per_tok": 2
  },
  "prompt": "You are an omni assistant.",
  "sft_mix": {
    "text_path": "data/text/production_corpus.txt",
    "image_manifest": "data/images/production_annotations.json",
    "image_root": "data/images",
    "asr_csv": "data/audio/production_asr.csv"
  },
  "save_dir": "checkpoints/omni_sft_tiny",
  "use_lr_spike": true,
  "lr_spike_multiplier": 10.0,
  "lr_spike_duration": 100,
  "lr_spike_consecutive_increases": 2
}
```

### Expected Progress

```
Epoch 1/5: loss=2.456, text_acc=45.2%
  → Image QA: 35.8%
  → Audio WER: 25.3%
  (Learning to integrate modalities)

Epoch 5/5: loss=1.123, accuracy=75.6%
  → Image QA: 68.9%
  → Audio WER: 12.7%
  (Good multimodal understanding!)

**Expected Validation Loss:**
- Target Loss: < 2.5 (similar to Thinker)
- Target Perplexity: < 10
- Good: loss < 2.0, perplexity < 8
- Excellent: loss < 1.5, perplexity < 5
```

### Running

```bash
python sft_omni.py --config configs/omni_sft_tiny.json
# Time: ~8 hours
# Output: checkpoints/omni_sft_tiny/model.pt
```

---

## 💡 Key Points

✅ **Final integration** of all components  
✅ **Freeze encoders** (already trained)  
✅ **Mixed batches** teach cross-modal understanding  
✅ **Attention masking** prevents attention to padding tokens  
✅ **5 epochs** sufficient for fine-tuning  
✅ **Output** is the complete μOmni system!

---

[Continue to Chapter 32: Inference Pipeline →](32-inference-pipeline.md)

**Chapter Progress:** Training Pipeline ●●●●●● (6/6 complete!)

---
