# Chapter 27: Stage A - Thinker Pretraining

[← Previous: Training Overview](26-training-overview.md) | [Back to Index](00-INDEX.md) | [Next: Stage B Audio →](28-stage-b-audio-encoder.md)

---

## 🎯 Purpose

Train the core language model on text data using next-token prediction.

## 📝 Task

**Objective**: Learn to predict the next token given previous tokens

```
Input:  "The cat sat on"
Target: "the"

Input:  "The cat sat on the"
Target: "mat"
```

## 💻 Command

```bash
python train_text.py --config configs/thinker_tiny.json
```

## 📊 Configuration

```json
{
  "vocab_size": 5000,
  "n_layers": 4,
  "d_model": 256,
  "n_heads": 4,
  "d_ff": 1024,
  "dropout": 0.1,
  "rope_theta": 10000,
  "ctx_len": 512,
  
  "data_path": "data/text/corpus.txt",
  "batch_size": 16,
  "num_epochs": 10,
  "learning_rate": 3e-4,
  "warmup_steps": 1000,
  "max_grad_norm": 1.0,
  
  "save_every": 1000,
  "eval_every": 500
}
```

## 📈 Expected Progress

```
Epoch 1/10:
Step 100: loss=4.234 ppl=68.9
Step 500: loss=3.156 ppl=23.4
→ Validation: loss=3.201 ppl=24.5

Epoch 5/10:
Step 2500: loss=2.456 ppl=11.7

Epoch 10/10:
Final: loss=1.987 ppl=7.3
```

## 📊 Key Metrics

**Loss**: Cross-entropy loss
- Start: 5-7 (random)
- Target: 1.5-2.5 (good)

**Perplexity**: exp(loss)
- Lower is better
- Good: 5-10 range

## 💡 Tips

1. **Monitor loss curve** - should decrease smoothly
2. **Check perplexity** - more interpretable than loss
3. **Validate regularly** - detect overfitting early
4. **Save checkpoints** - resume if interrupted

## 🎓 Output

```
checkpoints/thinker_tiny/
├── thinker_best.pt       # Best validation loss
├── thinker_step_1000.pt  # Periodic checkpoints
├── thinker_step_2000.pt
└── tokenizer.model       # Tokenizer
```

---

[Continue to Chapter 28: Stage B - Audio Encoder →](28-stage-b-audio-encoder.md)

