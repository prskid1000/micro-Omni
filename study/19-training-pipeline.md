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
python train_thinker.py
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
python train_audio_encoder.py
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
python train_vision.py
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
python train_talker.py
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
python train_sft.py
```

**Key config values:**

```python
# Loads all stage checkpoints automatically
thinker_checkpoint    = "checkpoints/thinker/best/model.pt"
audio_enc_checkpoint  = "checkpoints/audio_encoder/best/model.pt"
vision_checkpoint     = "checkpoints/vision/best/model.pt"
talker_checkpoint     = "checkpoints/talker/best/model.pt"

batch_size            = 4         # small — full model is large
learning_rate         = 1e-5      # low — we're fine-tuning, not training
warmup_steps          = 500
max_steps             = 30000
gradient_accumulation = 8         # effective batch = 32
use_amp               = True
```

**Loss function:** Joint cross-entropy across all modalities, with
mixed-modality batches. Each batch may contain text-only, image+text, audio+text,
or multimodal samples. The loss is averaged across all tokens regardless of
modality.

**What "good" loss looks like:**

| Training Phase | SFT Loss | Notes |
|---------------|---------|-------|
| First 1K steps | 3-5 | High — model adapting to joint format |
| 5K steps | 1.5-2.5 | Settling |
| Converged | 0.8-1.5 | |

Expect **validation loss spikes** when the batch composition changes (e.g.,
a run of audio samples followed by vision samples). This is normal. The LR
spike mechanism (Chapter 21) handles this automatically.

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

## 19.8 Resume Logic

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

## 19.9 Checkpoint Frequency

How often to save depends on dataset size and how expensive a crash would be:

| Dataset Size | Recommended `save_every` | Rationale |
|-------------|-------------------------|-----------|
| <50K steps total | Every 3,000 steps | Small dataset, checkpoints are cheap |
| 50K-200K steps | Every 5,000 steps | Balance between safety and disk usage |
| >200K steps | Every 10,000 steps | Disk space matters at scale |

Each checkpoint is roughly 2x the model size (model + optimizer states).
For the 25M Thinker, that is about 200MB per checkpoint.

---

## 19.10 Full Training Timeline

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

## 19.11 Label Smoothing Across Stages

Label smoothing is supported in all training scripts that use cross-entropy loss, controlled by the `label_smoothing` config parameter (default 0.1):

| Stage | Script | Label Smoothing |
|-------|--------|----------------|
| A (Thinker) | `train_text.py` | Yes |
| B (Audio) | CTC loss | N/A (CTC has its own alignment mechanism) |
| C (Vision) | `train_vision.py` | Yes (applied to InfoNCE) |
| D (Talker) | `train_talker.py` | No |
| E (SFT) | `sft_omni.py` | Yes |
| OCR | `train_ocr.py` | Yes |

See Chapter 20, Section 20.12 for a detailed explanation of how label smoothing works.

---

## 19.12 Deterministic Synthetic Data

For reproducible experiments and testing, use the deterministic data generation script:

```bash
python scripts/make_deterministic_data.py
```

This generates synthetic training data with fixed random seeds, ensuring identical data across runs. Useful for debugging training pipelines and verifying that code changes do not affect model behavior.

**Note:** Synthetic configs use smaller model dimensions (d=128, 4 layers) rather than the full production sizes (d=384, 8 layers). Production configs are backed up as `.bak` files.

**Next:** Chapter 20 covers the optimization techniques that make all of this
fit in 16GB of VRAM.
