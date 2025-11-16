# Chapter 31: Stage E - Multimodal SFT

[← Previous: Stage D Talker](30-stage-d-talker.md) | [Back to Index](00-INDEX.md) | [Next: Inference Pipeline →](32-inference-pipeline.md)

---

## 🎯 Purpose

Fine-tune all components together for multimodal understanding.

## 📝 Task

**Objective**: Joint training on mixed multimodal batches

```
Batch 1: Image + Text → Text response
Batch 2: Audio + Text → Text response
Batch 3: Text only → Text response
Batch 4: Image + Audio + Text → Text response
```

## 💻 Command

```bash
python sft_omni.py --config configs/omni_sft_tiny.json
```

## 📊 Configuration

```json
{
  "thinker_ckpt": "checkpoints/thinker_tiny/thinker_best.pt",
  "audio_ckpt": "checkpoints/audio_enc_tiny/audio_enc.pt",
  "vision_ckpt": "checkpoints/vision_tiny/vision.pt",
  
  "freeze_encoders": true,
  "batch_size": 8,
  "num_epochs": 5,
  "learning_rate": 1e-4,
  "warmup_steps": 500,
  
  "data_mix": {
    "text_only": 0.4,
    "image_text": 0.3,
    "audio_text": 0.3
  }
}
```

## 📁 Data Format

```
data/multimodal/
├── text/
│   └── conversations.json
├── images/
│   └── image_qa.json
└── audio/
    └── audio_qa.json
```

## 📈 Expected Progress

```
Epoch 1/5:
Step 100: loss=2.456 text_acc=45.2%
→ Image QA acc: 35.8%
→ Audio transcription WER: 25.3%

Epoch 3/5:
Step 450: loss=1.678 text_acc=62.8%
→ Image QA acc: 58.3%
→ Audio transcription WER: 18.5%

Epoch 5/5:
Final: loss=1.123 accuracy=75.6%
→ Image QA acc: 68.9%
→ Audio transcription WER: 12.7%
```

## 📊 Key Metrics

**Text Loss**: Next-token prediction
**Image QA Accuracy**: Visual understanding
**Audio WER**: Speech recognition quality

## 💡 Training Strategy

1. **Freeze encoders** (optional)
   - Faster training
   - Focus on projectors + Thinker

2. **Unfreeze all** (optional)
   - Better quality
   - Slower training

3. **Mixed batches**
   - Diverse training signal
   - Better generalization

## 🎯 What Gets Trained?

```
✅ Thinker (fine-tuned)
✅ Vision Projector (trained from scratch)
✅ Audio Projector (trained from scratch)
❌ Vision Encoder (frozen, optional)
❌ Audio Encoder (frozen, optional)
```

## 💡 Tips

1. **Start with frozen encoders** - faster
2. **Lower learning rate** - fine-tuning, not pretraining
3. **Monitor all modalities** - balanced performance
4. **Curriculum learning** - easier → harder data

## 🎓 Output

```
checkpoints/omni_sft_tiny/
├── omni.pt               # Final multimodal model
│   ├── thinker           # Fine-tuned Thinker
│   ├── proj_v            # Vision projector
│   └── proj_a            # Audio projector
└── omni_step_500.pt      # Checkpoints
```

## 🎉 Next Steps

After SFT completes, you have a fully trained multimodal model!

**Ready for inference:**
```bash
python infer_chat.py --ckpt_dir checkpoints/omni_sft_tiny
```

---

[Continue to Chapter 32: Inference Pipeline →](32-inference-pipeline.md)

