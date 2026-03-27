[← Previous: 18-data-preparation](18-data-preparation.md) | [Index](00-INDEX.md) | [Next: 20-performance-optimization →](20-performance-optimization.md)

# Chapter 19: Training Pipeline — All 5 Stages

Training a multimodal model is not a single `python train.py` command. It is
five stages, each building a different capability, composed into a final system
that can think, see, hear, and speak.

---

## 19.1 Why Staged Training?

Think of an orchestra rehearsal. You do not start by having every musician
play simultaneously — that produces noise. Instead:

1. Each soloist practices their part alone (Stages A, B, C)
2. Sections rehearse together once soloists are ready (Stage D)
3. The full orchestra performs together (Stage E)

Staged training works the same way. Each modality expert gets competent on its
own before being asked to cooperate with others.

---

## 19.2 Stage Dependency Diagram

```
 +-------------+    +----------------+    +--------------+
 | Stage A     |    | Stage B        |    | Stage C      |
 | Thinker     |    | Audio Encoder  |    | Vision       |
 | (text LM)   |    | (speech→tokens)|    | (image→embed)|
 +------+------+    +-------+--------+    +------+-------+
        |                   |                     |
        |   +---------------+                     |
        v   v                                     |
 +-------------+                                  |
 | Stage D     |                                  |
 | Talker+RVQ  |                                  |
 | (tokens→wav)|                                  |
 +------+------+                                  |
        |                                         |
        +-------------------+---------------------+
                            |
                            v
                     +-------------+
                     | Stage E     |
                     | SFT         |
                     | (all modes) |
                     +-------------+
```

**Key insight:** Stages A, B, and C have zero dependencies on each other. If
you have multiple GPUs (or multiple machines), run them in parallel and cut
total training time by ~60%.

---

## 19.3 Stage A — Thinker (Language Model)

**What it trains:** The core text transformer — the "brain" that handles
reasoning, generation, and instruction following.

**Why:** Every other stage either feeds into or is controlled by the Thinker.
A strong language model is the foundation.

**Command:**

```bash
python -m train.train_thinker --config configs/synthetic_thinker.json
```

**Key config values:**

```python
vocab_size        = 32000
d_model           = 768
n_heads           = 12
n_layers          = 12
max_seq_len       = 512
batch_size        = 32
learning_rate     = 3e-4
warmup_steps      = 2000
max_steps         = 100000
use_amp           = True
```

**Loss function:** Next-token cross-entropy with optional label smoothing (`label_smoothing` config parameter, default 0.1) and optional Multi-Token Prediction.

```
Loss = -1/T * sum( log P(token_t | token_1..t-1) )
```

When `use_mtp: true`, two auxiliary heads predict t+2 and t+3 tokens. The MTP loss is averaged with the main next-token loss, providing richer gradient signal per training example. Label smoothing is applied to all cross-entropy losses (see Chapter 20 for details).

**What "good" loss looks like:**

| Training Phase | Loss Range | Perplexity |
|---------------|-----------|------------|
| First 1K steps | 8-10 | ~3000 |
| 10K steps | 4-5 | ~50-150 |
| Converged | 2.5-3.5 | ~7-15 |

A final perplexity of 7-15 is realistic for a 25M model on general text. If
perplexity is below 5, suspect overfitting. Above 20, suspect underfitting or
bad data.

**Output files:**

```
checkpoints/
  thinker/
    step_10000/
      model.pt          # full state dict
      optimizer.pt      # optimizer state
      metadata.json     # step, loss, config hash
    step_20000/
      ...
    best/
      model.pt          # lowest validation loss
```

---

## 19.4 Stage B — Audio Encoder (Speech-to-Tokens)

**What it trains:** A convolutional + transformer encoder that converts raw
waveforms into discrete token sequences the Thinker can process.

**Why:** The Thinker only understands tokens. The audio encoder is the
"ears" — it translates continuous audio into the discrete vocabulary.

**Command:**

```bash
python -m train.train_audio_enc --config configs/synthetic_audio_enc.json
```

**Key config values:**

```python
sample_rate       = 16000
n_mels            = 80
encoder_layers    = 6
encoder_dim       = 512
ctc_vocab_size    = 32000    # matches Thinker vocab
batch_size        = 16
learning_rate     = 1e-4
warmup_steps      = 3000
max_steps         = 80000
use_amp           = True
```

**Loss function:** CTC (Connectionist Temporal Classification) loss.

CTC handles the alignment problem — audio frames and text tokens are different
lengths, and CTC learns the mapping without explicit alignment labels.

**What "good" loss looks like:**

| Training Phase | CTC Loss |
|---------------|----------|
| First 1K steps | 50-100 |
| 10K steps | 5-15 |
| Converged | 1-3 |

**Watch for blank collapse:** If loss drops very fast early (to near 0) and
Word Error Rate (WER) is terrible, the model learned to output only CTC blank
tokens. This is the audio encoder equivalent of "mode collapse." Fix by
reducing LR and increasing warmup (see Chapter 21).

**Output files:**

```
checkpoints/
  audio_encoder/
    step_10000/
      model.pt
      optimizer.pt
      metadata.json
    best/
      model.pt
```

---

## 19.5 Stage C — Vision (Image-to-Embedding)

**What it trains:** A vision encoder (typically a small ViT or CNN) plus a
projection layer that maps image features into the Thinker's embedding space.

**Why:** Images live in pixel space. The projection layer translates visual
features into vectors the Thinker's attention layers can attend to, just like
text embeddings.

**Command:**

```bash
python -m train.train_vision
```

**Key config values:**

```python
image_size        = 224
patch_size        = 16
vision_layers     = 6
vision_dim        = 512
projection_dim    = 768       # must match Thinker d_model
temperature       = 0.07      # InfoNCE temperature — critical!
batch_size        = 32
learning_rate     = 1e-4
warmup_steps      = 1000
max_steps         = 50000
use_amp           = True
```

**Loss function:** InfoNCE (contrastive loss).

```
L = -log( exp(sim(img_i, txt_i) / tau) / sum_j(exp(sim(img_i, txt_j) / tau)) )
```

The temperature `tau` (0.07) controls how "peaked" the softmax distribution is.
Too low (0.01) and training becomes unstable. Too high (1.0) and the model
cannot distinguish similar pairs. 0.07 is the sweet spot from CLIP research.

**What "good" loss looks like:**

| Training Phase | InfoNCE Loss |
|---------------|-------------|
| First 1K steps | 4-5 (random chance for BS=32: ln(32) = 3.47) |
| 10K steps | 2-3 |
| Converged | 0.5-1.5 |

**Output files:**

```
checkpoints/
  vision/
    step_10000/
      model.pt
      optimizer.pt
      metadata.json
    best/
      model.pt
```

---

## 19.6 Stage D — Talker + RVQ (Tokens-to-Speech)

**What it trains:** The speech synthesis pipeline — an autoregressive model
that predicts RVQ (Residual Vector Quantization) codes, plus the vocoder that
converts those codes back into waveforms.

**Why:** This is the "voice." The Thinker produces text tokens, Stage D
converts them into natural-sounding audio.

**Dependency:** Requires Stage A (Thinker) checkpoint, because the Talker
conditions on the Thinker's hidden states.

**Command:**

```bash
python -m train.train_talker
```

**Key config values:**

```python
rvq_codebook_size = 1024
rvq_num_quantizers = 8
talker_layers     = 6
talker_dim        = 768
batch_size        = 8
learning_rate     = 5e-5
warmup_steps      = 2000
max_steps         = 60000
use_amp           = True
```

**Loss function:** Two losses combined:

1. **AR cross-entropy** on RVQ code prediction (which codebook entry comes next)
2. **Reconstruction MSE** between predicted and target mel spectrograms

```
L_total = L_ce + lambda * L_mse
```

**What "good" loss looks like:**

| Training Phase | Total Loss | AR CE | Reconstruction MSE |
|---------------|-----------|-------|-------------------|
| First 1K steps | 15-20 | 8-10 | 5-8 |
| 10K steps | 5-8 | 3-5 | 2-3 |
| Converged | 2-4 | 1.5-2.5 | 0.5-1.5 |

**Output files:**

```
checkpoints/
  talker/
    step_10000/
      model.pt          # talker weights
      rvq.pt            # codebook weights
      optimizer.pt
      metadata.json
    best/
      model.pt
      rvq.pt
```

---

## 19.7 Stage E — SFT (Supervised Fine-Tuning)

**What it trains:** The full assembled model — Thinker + Audio Encoder +
Vision + Talker — fine-tuned end-to-end on instruction-following data across
all modalities.

**Why:** Stages A-D trained specialists. Stage E teaches them to work together
on real tasks: "describe this image," "transcribe this audio," "answer this
question and speak the response."

**Dependency:** Requires ALL previous stage checkpoints.

**Command:**

```bash
python -m train.sft_omni --config configs/synthetic_omni_sft.json
```

**Key config values:**

```python
# Loads all stage checkpoints automatically
thinker_checkpoint    = "checkpoints/thinker_tiny"
audio_enc_checkpoint  = "checkpoints/audio_enc_tiny"
vision_checkpoint     = "checkpoints/vision_tiny"
talker_checkpoint     = "checkpoints/talker_tiny"

batch_size            = 4         # small — full model is large
learning_rate         = 5e-5      # low — we're fine-tuning, not training
warmup_steps          = 50
max_steps             = 50000
gradient_accumulation = 2         # effective batch = 8
label_smoothing       = 0.1       # same as pretraining
proj_lr_mult          = 5.0       # projectors learn 5x faster (randomly initialized)
use_amp               = True
```

**Encoder freezing:** Audio and vision encoders are loaded from their pretrained
checkpoints but are **not added to the optimizer** — their weights are frozen.
Only the Thinker, Talker, and projection layers receive gradients. This saves
~3M parameters of gradient compute and prevents catastrophic forgetting of the
encoder representations trained in Stages B and C.

**Implementation details:** The SFT script uses a pre-allocated causal mask
buffer (`self._causal_mask[:, :, :T, :T]`) to avoid per-step GPU allocations,
and a fused AdamW optimizer (`fused=True` on CUDA) for a free 10-20% speedup.

**Loss function:** Joint cross-entropy (with label smoothing 0.1) across all
modalities, with mixed-modality batches. Each batch may contain text-only,
image+text, audio+text, or multimodal samples. The loss is averaged across all
tokens regardless of modality.

**What "good" loss looks like:**

| Training Phase | SFT Loss | Notes |
|---------------|---------|-------|
| First 1K steps | 3-5 | High — model adapting to joint format |
| 5K steps | 1.5-2.5 | Settling |
| Converged | 0.8-1.5 | |

Expect **validation loss spikes** when the batch composition changes (e.g.,
a run of audio samples followed by vision samples). This is normal. The
`TrainingMonitor` (Chapter 21) handles this automatically via LR spikes,
and optionally triggers early stopping if loss fails to recover.

**Output files:**

```
checkpoints/
  sft/
    step_5000/
      model.pt          # full model weights (all components)
      optimizer.pt
      metadata.json
    best/
      model.pt
```

---

## 19.8 Rapid Prototyping: Model Variants

You do not need to train all five stages to get a working model. The staged
design lets you build stripped-down variants in minutes for quick experiments.

### Text-Only Model (Stage A only)

The simplest variant. Train just the Thinker — no audio, no vision.

```bash
python -m train.train_thinker --config configs/synthetic_thinker.json
```

Use `configs/synthetic_thinker.json` for rapid iteration (`max_steps: 8400`,
finishes in ~15 minutes).
Inference works immediately with `test/infer_chat.py --ckpt_dir checkpoints/thinker_tiny`.

### Text + Vision Model (Stages A + C + modified SFT)

Train the Thinker and Vision encoder, then SFT on text+image data only.

```bash
# 1. Train Thinker
python -m train.train_thinker --config configs/synthetic_thinker.json

# 2. Train Vision encoder
python -m train.train_vision --config configs/synthetic_vision.json

# 3. SFT — remove audio from sft_mix
#    In your SFT config, delete the "asr_csv" key from "sft_mix":
#    "sft_mix": {
#        "text_path": "data/text/production_corpus.txt",
#        "image_manifest": "data/images/production_annotations.json",
#        "image_root": "data/images"
#    }
python -m train.sft_omni --config configs/synthetic_omni_sft.json
```

The SFT script gracefully handles missing modalities — if `asr_csv` is absent
from `sft_mix`, it simply skips ASR samples. No code changes required.

### Text + Audio Model (Stages A + B + modified SFT)

Train the Thinker and Audio Encoder, then SFT on text+audio data only.

```bash
# 1. Train Thinker
python -m train.train_thinker --config configs/synthetic_thinker.json

# 2. Train Audio Encoder
python -m train.train_audio_enc --config configs/synthetic_audio_enc.json

# 3. SFT — remove vision from sft_mix
#    In your SFT config, delete "image_manifest" and "image_root":
#    "sft_mix": {
#        "text_path": "data/text/production_corpus.txt",
#        "asr_csv": "data/audio/production_asr.csv"
#    }
python -m train.sft_omni --config configs/synthetic_omni_sft.json
```

### Full Multimodal (All Stages A through E)

The complete pipeline as documented in this chapter. All five stages, all
modalities.

```bash
python -m train.train_thinker --config configs/synthetic_thinker.json      # Stage A
python -m train.train_audio_enc --config configs/synthetic_audio_enc.json  # Stage B
python -m train.train_vision --config configs/synthetic_vision.json        # Stage C
python -m train.train_talker --config configs/synthetic_talker.json        # Stage D
python -m train.sft_omni --config configs/synthetic_omni_sft.json          # Stage E
```

Remember: A, B, C can run in parallel. D requires A. E requires all four.

### Quick Iteration Tips

- All `synthetic_*.json` configs use `max_steps: 2000` — training finishes in
  minutes, not hours. Use these to validate your pipeline before committing to
  a full production run.
- Synthetic configs also use smaller model dimensions (d=128, 4 layers) and
  reduced vocabulary (256 tokens), so they need far less VRAM.
- Copy a synthetic config, tweak one variable at a time, and compare loss
  curves. This is the fastest way to understand what each hyperparameter does.

---

## 19.9 Config Saving

Every training script saves a copy of the training config to the checkpoint
directory at the start of training:

```
checkpoints/thinker_tiny/
  config.json              # Copy of training config, saved at start of training
  thinker.pt               # model + optimizer + scheduler + scaler + monitor states
  thinker_metadata.json    # step, epoch, dataset stats
  tokenizer.model          # BPE tokenizer (if applicable)
  tokenizer.vocab          # vocabulary (if applicable)
```

This ensures that the exact configuration used to train a model is always
co-located with its weights. Test scripts and inference scripts load
`config.json` exclusively from the checkpoint directory — they never read
from the `configs/` directory. This eliminates a common class of bugs where
the config on disk has drifted from the config used during training.

---

## 19.10 Resume Logic

Every stage writes a `metadata.json` alongside each checkpoint:

```json
{
  "step": 10000,
  "loss": 2.34,
  "learning_rate": 2.1e-4,
  "config_hash": "a3f8b2c1",
  "timestamp": "2026-03-25T14:30:00"
}
```

When a training script starts, it:

1. Scans the checkpoint directory for the latest `metadata.json`
2. Verifies the `config_hash` matches the current config
3. Loads the model + optimizer state
4. Resumes from the saved step

This is fully automatic. If training crashes at step 15,432, just re-run the
same command. It picks up where it left off.

If you change the config (e.g., different learning rate), the config hash
will not match and training starts fresh — preventing silent bugs from
mismatched hyperparameters.

---

## 19.11 SFT Encoder Freezing

A common question: why not fine-tune the audio and vision encoders during SFT?

The answer is practical and theoretical:

1. **They are already trained.** Stages B and C produced encoders that map
   audio/images into useful representations. Re-training them risks
   catastrophic forgetting — the encoder "unlearns" its features to overfit
   the smaller SFT dataset.

2. **Gradient savings.** The audio encoder (~2.0M params) and vision encoder
   (~914K params) account for ~3M parameters. Freezing them means no gradient
   computation, no optimizer state, and no gradient accumulation buffers for
   those parameters — roughly 24MB of VRAM saved (3M params x 8 bytes for
   AdamW states).

3. **Projectors bridge the gap.** The projection layers (randomly initialized
   at SFT start) are the only new components. They get a higher learning rate
   (`proj_lr_mult: 5.0`) because they need to learn fast while the frozen
   encoders provide stable input features.

In code, the SFT script simply does not pass encoder parameters to the
optimizer. The encoders still run in forward passes (producing embeddings),
but `requires_grad` is `False` for all their parameters.

---

## 19.12 Checkpoint Frequency

How often to save depends on dataset size and how expensive a crash would be:

| Dataset Size | Recommended `save_every` | Rationale |
|-------------|-------------------------|-----------|
| <50K steps total | Every 3,000 steps | Small dataset, checkpoints are cheap |
| 50K-200K steps | Every 5,000 steps | Balance between safety and disk usage |
| >200K steps | Every 10,000 steps | Disk space matters at scale |

Each checkpoint is roughly 2x the model size (model + optimizer states).
For the 25M Thinker, that is about 200MB per checkpoint.

---

## 19.13 Full Training Timeline

```
Day 1:  Start Stages A, B, C in parallel
        +-----------+  +-----------+  +-----------+
        | Thinker   |  | Audio Enc |  | Vision    |
        | ~10 hours |  | ~8 hours  |  | ~6 hours  |
        +-----------+  +-----------+  +-----------+

Day 2:  Start Stage D (needs A)
        +-----------+
        | Talker    |
        | ~8 hours  |
        +-----------+

Day 3:  Start Stage E (needs all)
        +-----------+
        | SFT       |
        | ~6 hours  |
        +-----------+
```

On a single 16GB GPU, expect about 3 days total. With parallel execution of
A/B/C on separate GPUs, you can finish in under 2 days.

---

## 19.14 Label Smoothing Across Stages

Label smoothing is supported in all training scripts that use cross-entropy loss, controlled by the `label_smoothing` config parameter (default 0.1 for pretraining stages):

| Stage | Script | Label Smoothing |
|-------|--------|----------------|
| A (Thinker) | `train_thinker.py` | Yes (0.1) |
| B (Audio) | CTC loss | N/A (CTC has its own alignment mechanism) |
| C (Vision) | `train_vision.py` | Yes (applied to InfoNCE) |
| D (Talker) | `train_talker.py` | No |
| E (SFT) | `sft_omni.py` | Yes (0.1) |
| OCR | `train_ocr.py` | Yes |

See Chapter 20, Section 20.12 for a detailed explanation of how label smoothing works.

---

## 19.15 Synthetic Data Generation

For reproducible experiments and testing, use the data generation script:

```bash
python -m scripts.generate_synthetic_data
```

This generates synthetic training data for all modalities. Useful for debugging training pipelines and verifying that code changes do not affect model behavior.

**Note:** Synthetic configs use d=128, 4 layers, 4 heads.

**Next:** Chapter 20 covers the optimization techniques that make all of this
fit in 16GB of VRAM.
