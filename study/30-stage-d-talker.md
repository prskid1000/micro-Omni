# Chapter 30: Stage D - Talker & RVQ Codec Training

[← Previous: Stage C Vision](29-stage-c-vision-encoder.md) | [Back to Index](00-INDEX.md) | [Next: Stage E SFT →](31-stage-e-sft.md)

---

## 🎯 Purpose

Train speech code generator (Talker) and RVQ codec jointly.

## 📝 Task

**Objective**: Predict RVQ codes autoregressively for speech synthesis

```
Input:  Previous codes [[42, 87], [103, 12], ...]
Output: Next codes [67, 91]
```

## 💻 Command

```bash
python train_talker.py --config configs/talker_tiny.json
```

## 📊 Configuration

```json
{
  "d_model": 192,
  "n_layers": 4,
  "n_heads": 3,
  "d_ff": 768,
  "codebooks": 2,
  "codebook_size": 128,
  "dropout": 0.1,
  
  "data_path": "data/audio/tts/",
  "batch_size": 16,
  "num_epochs": 25,
  "learning_rate": 3e-4,
  
  "save_every": 500
}
```

## 📁 Data Format

```
data/audio/tts/
├── audio1.wav
├── audio2.wav
└── audio3.wav

(No transcriptions needed - learns from audio only)
```

## 📈 Expected Progress

```
Epoch 1/25:
Step 100: base_loss=3.456 res_loss=3.234 recon=0.087
Step 500: base_loss=2.123 res_loss=2.087 recon=0.045

Epoch 13/25:
base_loss=1.567 res_loss=1.489 recon=0.023

Epoch 25/25:
Final: base_loss=1.234 res_loss=1.189 recon=0.012
```

## 📊 Key Metrics

**base_loss**: Base codebook prediction
**res_loss**: Residual codebook prediction
**recon**: Mel reconstruction MSE
- All lower is better
- Good recon: <0.02

## 💡 Tips

1. **Joint training** - RVQ and Talker train together
2. **Reconstruction quality** - listen to samples
3. **Codebook usage** - ensure all codes used
4. **Frame rate** - 12.5Hz matches encoder

## 🎓 Output

```
checkpoints/talker_tiny/
├── talker.pt             # Contains both Talker & RVQ
└── talker_step_500.pt    # Checkpoints
```

---

[Continue to Chapter 31: Stage E - Multimodal SFT →](31-stage-e-sft.md)

