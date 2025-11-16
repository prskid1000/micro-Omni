# Chapter 29: Stage C - Vision Encoder Training

[← Previous: Stage B Audio](28-stage-b-audio-encoder.md) | [Back to Index](00-INDEX.md) | [Next: Stage D Talker →](30-stage-d-talker.md)

---

## 🎯 Purpose

Train vision encoder for image understanding.

## 📝 Task

**Objective**: Image classification/understanding with captions

```
Input:  Image (224×224×3)
Output: Classification or caption representation
```

## 💻 Command

```bash
python train_vision.py --config configs/vision_tiny.json
```

## 📊 Configuration

```json
{
  "img_size": 224,
  "patch": 16,
  "d_model": 128,
  "n_layers": 4,
  "n_heads": 2,
  "d_ff": 512,
  "dropout": 0.1,
  
  "data_path": "data/images/",
  "batch_size": 32,
  "num_epochs": 15,
  "learning_rate": 3e-4,
  
  "save_every": 500
}
```

## 📁 Data Format

```json
// data/images/annotations.json
{
  "images": [
    {
      "id": 1,
      "file_name": "image1.jpg",
      "caption": "A cat sitting on a couch"
    }
  ]
}
```

## 📈 Expected Progress

```
Epoch 1/15:
Step 100: loss=2.345 acc=35.2%
Step 500: loss=1.678 acc=58.7%

Epoch 8/15:
loss=1.234 acc=68.9%

Epoch 15/15:
Final: loss=0.987 acc=78.9%
```

## 📊 Key Metrics

**Loss**: Cross-entropy
- Lower is better

**Accuracy**: Classification accuracy
- Higher is better
- Good: >70%

## 💡 Tips

1. **Image augmentation** - random crops, flips
2. **Normalize properly** - ImageNet stats
3. **Patch size matters** - 16×16 is standard
4. **Monitor CLS token** - aggregates image info

## 🎓 Output

```
checkpoints/vision_tiny/
├── vision.pt             # Best model
└── vision_step_500.pt    # Checkpoints
```

---

[Continue to Chapter 30: Stage D - Talker & RVQ →](30-stage-d-talker.md)

