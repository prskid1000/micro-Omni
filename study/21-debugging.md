[← Previous: 20-performance-optimization](20-performance-optimization.md) | [Index](00-INDEX.md) | [Next: 22-setup-environment →](22-setup-environment.md)

# Chapter 21: Debugging & Troubleshooting

Things will go wrong during training. This chapter is a reference you come back
to when they do. Start with the symptom table, then read the detailed section
for your specific issue.

---

## 21.1 Quick Reference: Symptom Table

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| Loss is NaN | LR too high, no warmup, data corruption | Reduce LR, add warmup, validate data |
| Loss not decreasing | Bad LR, data not loading, model too small | Check LR schedule, verify batch content |
| Gradient norm >100 after clipping | Corrupted sample or LR spike | Skip batch, reduce LR |
| CTC loss drops to ~0 but WER is high | Blank collapse | Reduce LR, increase warmup steps |
| CUDA out of memory | Batch too large or accumulation leak | Reduce BS, enable AMP, check accum logic |
| `triton` / `ptxas` / `sm_120` errors | torch.compile on RTX 50xx | Set `use_compile = False` |
| `_orig_mod.` prefix in state dict | Compiled model checkpoint | Use `strip_orig_mod()` utility |
| Buzzy/metallic audio output | Vocoder undertrained | Increase `max_audio_length_percentile`, train longer |
| Loss spikes during SFT | Modality transitions in batch | Normal — LR spike mechanism handles it |
| Training hangs on Windows | DataLoader multiprocessing | Add `if __name__ == '__main__':` guard |
| File not found on Windows | Backslash paths | Use forward slashes in all configs |
| Beam search CTC duplicates | Duplicate chars in output | Fixed in beam search decoder -- was a known bug |

---

## 21.2 NaN Loss

NaN (Not a Number) means a floating-point operation produced an undefined
result. Common during the first few hundred steps.

**Cause 1: Learning rate too high**

The model takes a huge gradient step, weights explode, and the next forward
pass produces inf/NaN.

```
Step 0:    loss = 8.5
Step 10:   loss = 45.2      ← already too high
Step 20:   loss = inf
Step 21:   loss = NaN       ← game over
```

Fix: Reduce learning rate by 3-10x.

**Cause 2: No warmup**

Warmup gradually increases LR from 0 to the target over N steps. Without it,
the first gradient update uses the full LR on randomly initialized weights —
almost guaranteed to overshoot.

```python
warmup_steps = 2000    # ramp LR from 0 to target over 2000 steps
```

Fix: Always use warmup. 1000-3000 steps is typical.

**Cause 3: Data corruption**

A single NaN or inf value in the training data propagates through the entire
batch.

```python
# Quick data check
import torch
for batch in dataloader:
    assert not torch.isnan(batch).any(), "NaN in input data!"
    assert not torch.isinf(batch).any(), "Inf in input data!"
```

Fix: Validate data before training (Chapter 18).

---

## 21.3 Gradient Explosion

The training scripts clip gradients to `max_grad_norm=1.0` and return the
pre-clip norm. Monitor it:

```python
grad_norm = clip_gradients(model, max_norm=1.0)

if grad_norm > 100:
    print(f"WARNING: grad_norm={grad_norm:.1f}, skipping batch")
    optimizer.zero_grad(set_to_none=True)
    continue    # skip this batch entirely
```

**When to worry:**

```
grad_norm < 1.0     →  normal, no clipping needed
grad_norm 1.0-10    →  normal, clipping is working
grad_norm 10-100    →  elevated, watch closely
grad_norm > 100     →  skip the batch, something is wrong with this sample
```

A single corrupted audio file or a text sample with unusual Unicode can cause
a gradient spike. Skipping the batch is safer than letting the corrupted
gradient update the weights.

---

## 21.4 Loss Not Decreasing

If loss stays flat for thousands of steps:

**Check 1: Is the data actually loading?**

```python
for batch in dataloader:
    print(batch.shape, batch[:2])    # print first 2 samples
    break
```

A common bug: the dataset returns the same sample every time (shuffling
disabled, or dataset length is 1 due to a path error).

**Check 2: Is the learning rate reasonable?**

```
Too low:   LR = 1e-7  →  loss barely moves
Good:      LR = 3e-4  →  loss decreases steadily
Too high:  LR = 1e-2  →  loss oscillates wildly or NaNs
```

**Check 3: Is the model large enough?**

For a 25M parameter model on 500M tokens, expect loss to decrease. If it does
not, the tokenizer or data format is likely wrong before you suspect model
capacity.

---

## 21.5 CTC Blank Collapse (Audio Encoder)

CTC uses a "blank" token to handle alignment. Blank collapse is when the model
learns that outputting 100% blank tokens minimizes CTC loss trivially — because
the CTC algorithm marginalizes over all valid alignments, and "all blanks" is
technically a valid (if useless) alignment for the empty string.

**Symptoms:**

```
Step 100:   CTC loss = 85.0
Step 500:   CTC loss = 2.1     ← suspiciously fast
Step 1000:  CTC loss = 0.3     ← too good to be true
WER:        99.8%              ← model outputs nothing useful
```

**Fixes:**

1. Reduce learning rate: `1e-4` to `5e-5`
2. Increase warmup: `3000` to `5000` steps
3. Verify audio is not silence (check for all-zero waveforms)
4. Ensure text labels are not empty strings

---

## 21.6 CUDA Out of Memory (OOM)

```
RuntimeError: CUDA out of memory. Tried to allocate 512.00 MiB
(GPU 0; 16.00 GiB total capacity; 14.82 GiB already allocated)
```

**Fix priority (try in order):**

1. **Reduce batch_size** — halving BS halves activation memory
2. **Enable AMP** — `use_amp = True` (if not already)
3. **Enable gradient accumulation** — maintain effective BS with less VRAM
4. **Reduce sequence length** — lower `max_text_length_percentile` from 95 to 90
5. **Reduce num_workers** — each worker holds a pre-loaded batch in RAM

**Memory leak check:**

If OOM happens after training for a while (not at the start), suspect a
gradient accumulation bug:

```python
# WRONG — gradients accumulate forever
loss.backward()

# RIGHT — only step every N batches
loss = loss / accumulation_steps
loss.backward()
if (step + 1) % accumulation_steps == 0:
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)   # ← this frees memory
```

---

## 21.7 Triton Compilation Errors (RTX 50-Series)

```
RuntimeError: Triton compilation failed
  ptxas fatal: Unsupported .target 'sm_120'
```

The RTX 50-series (Blackwell) uses SM 120 architecture. As of early 2026,
Triton's PTX assembler does not fully support this target.

**Fix:**

```python
use_compile = False    # in your training config
```

Or set the environment variable:

```bash
export TORCH_COMPILE_DISABLE=1
python train_thinker.py
```

Everything else (AMP, Flash Attention, cuDNN) works fine on Blackwell. Only
`torch.compile()` is affected.

---

## 21.8 Windows-Specific Issues

### DataLoader Multiprocessing

On Windows, `num_workers > 0` requires the main script to be guarded:

```python
if __name__ == '__main__':
    train()    # all DataLoader creation must happen inside here
```

Without this guard, each worker process re-imports the module and tries to
spawn its own workers, causing an infinite fork bomb.

### Path Separators

Windows uses backslashes (`D:\data\audio\clip.wav`), but Python and most
libraries work with forward slashes. Always use forward slashes in configs
and CSV files:

```
# WRONG (in CSV or config)
D:\data\audio\clip_001.wav

# RIGHT
data/audio/clip_001.wav
```

Python's `pathlib.Path` and `os.path.join` handle this automatically, but
hardcoded strings in CSV files must use forward slashes.

---

## 21.9 Checkpoint Loading: _orig_mod Prefix

When you save a checkpoint from a `torch.compile()`-wrapped model, every key
in the state dict gets an `_orig_mod.` prefix:

```python
# Saved from compiled model:
"_orig_mod.transformer.layers.0.attention.q_proj.weight"

# Expected by non-compiled model:
"transformer.layers.0.attention.q_proj.weight"
```

The training scripts include a `strip_orig_mod()` utility:

```python
def strip_orig_mod(state_dict):
    return {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
```

This is called automatically during checkpoint loading. If you load checkpoints
manually, remember to apply it when the keys do not match.

---

## 21.10 Vocoder Audio Quality

**Buzzy or metallic audio** is the most common vocoder problem. It means the
model has not learned fine spectral details.

**Causes and fixes:**

| Audio Quality | Cause | Fix |
|--------------|-------|-----|
| Very buzzy, robotic | Severely undertrained | Train longer (2-3x more steps) |
| Metallic overtones | Clipped audio in training data | Increase `max_audio_length_percentile` to 98 |
| Muffled, low-fi | Mel spectrogram resolution too low | Increase `n_mels` (if configurable) |
| Random noise bursts | Discriminator overpowering generator | Reduce discriminator LR relative to generator |

The `max_audio_length_percentile` setting is particularly important for the
vocoder. If set too low (e.g., 90), longer utterances get truncated, and the
model never learns to handle sustained sounds. Set it to 95-98 for vocoder
training.

---

## 21.11 LR Spike Mechanism

The training scripts include an automatic LR adjustment for when validation
loss increases:

```
Validation losses over time:
  Step 5000:  2.1
  Step 6000:  2.0     ← decreasing, good
  Step 7000:  2.15    ← increased (1st time)
  Step 8000:  2.3     ← increased again (2nd consecutive time)
                         → LR spike triggered!
```

When validation loss increases for 2+ consecutive evaluations, the LR spike
mechanism temporarily boosts the learning rate to escape a bad region of the
loss landscape.

```
Normal LR:  |---------|
                      |
LR Spike:             |--*--|
                            |
Back to normal:             |---------|
```

This is especially important during SFT (Stage E), where switching between
modalities in the batch can cause temporary loss increases that look like
divergence but are actually just the model readjusting.

**Do not disable the LR spike mechanism** unless you have a specific reason.
If you see "LR spike triggered" in the logs, it means the mechanism is working
as intended.

---

## 21.12 Debugging Flowchart

```
Training problem?
       |
       v
  Is loss NaN?
  YES --> Check LR, warmup, data
  NO  |
       v
  Is loss flat?
  YES --> Check data loading, LR, model
  NO  |
       v
  Is loss spiking?
  YES --> Check grad_norm, enable clipping
  NO  |
       v
  OOM error?
  YES --> Reduce BS, enable AMP, enable accum
  NO  |
       v
  Compilation error?
  YES --> Disable torch.compile (RTX 50xx?)
  NO  |
       v
  Bad output quality?
  YES --> Train longer, check data percentiles
  NO  |
       v
  Check logs for warnings you might have missed
```

---

## 21.13 Logging Checklist

Every training run should log these values. If something goes wrong, they are
the first things to check:

```
Per step:
  - train_loss
  - grad_norm (before clipping)
  - learning_rate (current, after scheduler)
  - batch_size (actual, after any dynamic adjustments)
  - throughput (samples/sec or tokens/sec)

Per validation:
  - val_loss
  - stage-specific metrics (WER for audio, BLEU for text, etc.)

Per checkpoint:
  - step number
  - config_hash
  - best_val_loss so far
```

When reporting a training issue, include the last 50-100 lines of logged
metrics. The pattern in those numbers almost always reveals the cause.
