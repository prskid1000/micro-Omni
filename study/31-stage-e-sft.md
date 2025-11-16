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

---

## 💻 Training Details

### Configuration

```json
{
  // Load pretrained checkpoints
  "thinker_ckpt": "checkpoints/thinker_tiny/thinker_best.pt",
  "audio_ckpt": "checkpoints/audio_enc_tiny/audio_enc.pt",
  "vision_ckpt": "checkpoints/vision_tiny/vision.pt",
  
  // Training strategy
  "freeze_encoders": true,  // Only fine-tune Thinker
  "batch_size": 8,
  "num_epochs": 5,
  "learning_rate": 1e-4,
  
  // Data mix (balance modalities)
  "data_mix": {
    "text_only": 0.4,    // 40% text
    "image_text": 0.3,   // 30% vision
    "audio_text": 0.3    // 30% audio
  }
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
```

### Running

```bash
python sft_omni.py --config configs/omni_sft_tiny.json
# Time: ~8 hours
# Output: checkpoints/omni_sft_tiny/omni_final.pt
```

---

## 💡 Key Points

✅ **Final integration** of all components  
✅ **Freeze encoders** (already trained)  
✅ **Mixed batches** teach cross-modal understanding  
✅ **5 epochs** sufficient for fine-tuning  
✅ **Output** is the complete μOmni system!

---

[Continue to Chapter 32: Inference Pipeline →](32-inference-pipeline.md)

**Chapter Progress:** Training Pipeline ●●●●●● (6/6 complete!)

---
