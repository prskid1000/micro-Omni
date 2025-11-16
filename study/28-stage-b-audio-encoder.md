# Chapter 28: Stage B - Audio Encoder ASR Training

[← Previous: Stage A Thinker](27-stage-a-thinker.md) | [Back to Index](00-INDEX.md) | [Next: Stage C Vision →](29-stage-c-vision-encoder.md)

---

## 🎯 Purpose

Train audio encoder for speech recognition using CTC loss.

## 📝 Task

**Objective**: Convert audio to text (Automatic Speech Recognition)

```
Input:  Mel spectrogram (T, 128)
Output: Character sequence "hello world"
```

## 💻 Command

```bash
python train_audio_enc.py --config configs/audio_enc_tiny.json
```

## 📊 Configuration

```json
{
  "d_model": 192,
  "n_layers": 4,
  "n_heads": 3,
  "d_ff": 768,
  "dropout": 0.1,
  "downsample_time": 8,
  
  "data_path": "data/audio/asr.csv",
  "batch_size": 8,
  "num_epochs": 20,
  "learning_rate": 1e-4,
  
  "save_every": 500
}
```

## 📁 Data Format

```csv
# data/audio/asr.csv
audio_path,transcription
data/audio/wav/sample1.wav,"hello world"
data/audio/wav/sample2.wav,"how are you"
```

## 📈 Expected Progress

```
Epoch 1/20:
Step 100: ctc_loss=45.23 wer=78.5%
Step 500: ctc_loss=18.67 wer=45.2%

Epoch 10/20:
ctc_loss=12.34 wer=25.8%

Epoch 20/20:
Final: ctc_loss=8.45 wer=12.3%
```

## 📊 Key Metrics

**CTC Loss**: Alignment quality
- Lower is better
- Good: <10

**WER** (Word Error Rate):
- Percentage of wrong words
- Excellent: <10%
- Good: 10-20%

## 💡 Tips

1. **Audio quality matters** - clean recordings work best
2. **Normalize audio** - consistent volume levels
3. **Downsample properly** - 8x for 12.5Hz frame rate
4. **Monitor WER** - more interpretable than CTC loss

## 🎓 Output

```
checkpoints/audio_enc_tiny/
├── audio_enc.pt          # Best model
└── audio_enc_step_500.pt # Checkpoints
```

---

[Continue to Chapter 29: Stage C - Vision Encoder →](29-stage-c-vision-encoder.md)

