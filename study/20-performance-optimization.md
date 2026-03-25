# Chapter 20: Performance & Optimization

A 25M-parameter multimodal model should fit comfortably on a 16GB GPU — but
only if you use every optimization available. This chapter covers the techniques
that halve memory usage, double throughput, and keep training stable.

---

## 20.1 Mixed Precision (AMP)

Automatic Mixed Precision stores weights in float32 but runs forward/backward
passes in float16. This halves memory for activations and roughly doubles
throughput.

```python
scaler = torch.cuda.amp.GradScaler()

with torch.cuda.amp.autocast():
    output = model(input)
    loss = criterion(output, target)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

Enabled by default in all training scripts via `use_amp = True`.

**AMP memory impact:**

```
Without AMP:     Model (100MB) + Activations (400MB) = 500MB
With AMP:        Model (100MB) + Activations (200MB) = 300MB
                                                        ^^^^
                                                        40% less
```

---

## 20.2 BFloat16 on Ampere+ GPUs

If you have an RTX 30xx, 40xx, or 50xx GPU, you can use bfloat16 instead of
float16:

```python
with torch.cuda.amp.autocast(dtype=torch.bfloat16):
    ...
```

**Why bfloat16 over float16?**

```
float16:   1 sign | 5 exponent  | 10 mantissa   → range: +-65504
bfloat16:  1 sign | 8 exponent  | 7 mantissa    → range: +-3.4e38
```

BFloat16 has the same exponent range as float32, so it never overflows during
training. Float16 can overflow when loss values or gradients exceed 65504,
causing NaN. With bfloat16, the GradScaler becomes unnecessary.

---

## 20.3 Flash Attention

Standard attention is O(T^2) in both compute and memory. Flash Attention
computes exact attention in O(T) memory and 2-4x faster wall-clock time by
fusing operations and tiling through SRAM.

```
Standard Attention:
  Q, K, V each: [B, H, T, D]
  Attention matrix: [B, H, T, T]  ← this is the memory killer
  Memory: O(T^2)

Flash Attention:
  Never materializes [T, T] matrix
  Tiles computation through GPU SRAM
  Memory: O(T)
  Speed: 2-4x faster
```

In PyTorch 2.0+, Flash Attention is used automatically by
`F.scaled_dot_product_attention()` when inputs meet size requirements. No code
changes needed — it is enabled by default.

---

## 20.4 Gradient Accumulation

When VRAM is too tight for a large batch, simulate it:

```python
accumulation_steps = 8
optimizer.zero_grad()

for i, batch in enumerate(dataloader):
    with torch.cuda.amp.autocast():
        loss = model(batch) / accumulation_steps   # scale loss

    scaler.scale(loss).backward()                  # accumulate gradients

    if (i + 1) % accumulation_steps == 0:
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
```

**Effective batch size = batch_size x accumulation_steps**

```
batch_size=4, accumulation_steps=8  →  effective batch = 32
```

The model sees the same gradient direction as a true batch of 32, but only 4
samples are in VRAM at once. The only cost is wall-clock time (8 forward passes
instead of 1).

---

## 20.5 Gradient Clipping

Large gradients destabilize training. Clipping caps the global gradient norm:

```python
max_grad_norm = 1.0
grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
```

The returned `grad_norm` is the norm **before** clipping. Monitor it:

```
grad_norm = 0.8    →  normal, no clipping applied
grad_norm = 5.2    →  clipped to 1.0, training continues
grad_norm = 150.0  →  something is wrong — see Chapter 21
```

---

## 20.6 Memory Micro-Optimizations

### zero_grad(set_to_none=True)

```python
optimizer.zero_grad(set_to_none=True)   # frees gradient tensors
# vs
optimizer.zero_grad()                    # fills with zeros (still allocated)
```

`set_to_none=True` deallocates gradient memory between steps instead of
writing zeros. Saves 5-10% VRAM on a 25M model. Used by default in all
training scripts.

### pin_memory=True

```python
dataloader = DataLoader(
    dataset,
    batch_size=32,
    num_workers=2,
    pin_memory=True    # 10-30% faster CPU→GPU transfer
)
```

Pinned memory uses page-locked RAM that the GPU can DMA-read directly,
bypassing the CPU-mediated copy. Costs slightly more system RAM but speeds
up every batch transfer.

### cudnn.benchmark=True

```python
torch.backends.cudnn.benchmark = True
```

On the first batch, PyTorch benchmarks every available cuDNN algorithm for
each convolution shape and caches the fastest. Adds ~30 seconds at startup,
saves time on every subsequent batch. Most beneficial for the audio encoder
(heavy convolution use).

---

## 20.7 torch.compile()

PyTorch 2.0's compiler fuses operations, eliminates memory round-trips, and
generates optimized GPU kernels:

```python
model = torch.compile(model)   # 20-50% speedup
```

**EXCEPTION: RTX 50-series (Blackwell architecture)**

As of early 2026, `torch.compile()` triggers Triton compilation errors on
RTX 5070, 5070 Ti, 5080, and 5090 GPUs. The Triton backend does not fully
support the Blackwell SM architecture yet.

```python
# In training configs:
use_compile = True    # RTX 20xx/30xx/40xx — use it
use_compile = False   # RTX 50xx — disable until Triton support lands
```

If you see errors mentioning `triton`, `ptxas`, or `sm_120`, set
`use_compile = False`.

---

## 20.8 Cached RoPE Frequencies

Rotary Position Embeddings require sinusoidal frequencies that depend only on
position and dimension — not on the input. Computing them every forward pass
is wasteful.

```python
# Compute once at init, register as buffer
freqs = precompute_rope_frequencies(max_seq_len, dim)
self.register_buffer("rope_freqs", freqs)

# At runtime, just slice
def forward(self, x):
    seq_len = x.shape[1]
    freqs = self.rope_freqs[:seq_len]   # no recomputation
```

This turns a per-batch trigonometric computation into a single tensor slice.

---

## 20.9 Pre-Allocated Causal Masks

The causal attention mask is the same for every batch of the same sequence
length. Pre-allocate it:

```python
# At init
causal_mask = torch.triu(torch.ones(max_len, max_len), diagonal=1).bool()
self.register_buffer("causal_mask", causal_mask)

# At runtime — slice, don't recreate
def forward(self, x):
    T = x.shape[1]
    mask = self.causal_mask[:T, :T]
```

`register_buffer` ensures the mask:
- Moves to GPU with the model (no manual `.to(device)`)
- Is saved/loaded with checkpoints automatically
- Is not treated as a trainable parameter

---

## 20.10 DataLoader Workers

```python
num_workers = 2
```

Worker processes pre-load and pre-process the next batches while the GPU is
busy with the current one. With 0 workers, the GPU idles during data loading.

```
num_workers=0:   [Load][Train][Load][Train][Load][Train]
num_workers=2:   [Load]
                      [Train][Train][Train][Train]...
                 [Load][Load][Load][Load]...
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^
                 Workers pre-load in parallel
```

Two workers is enough for single-GPU training. More than 4 rarely helps and
increases RAM usage.

---

## 20.11 EMA (Exponential Moving Average)

EMA maintains a smoothed copy of the model weights:

```python
ema_decay = 0.999

# After each optimizer step:
for ema_param, param in zip(ema_model.parameters(), model.parameters()):
    ema_param.data.mul_(ema_decay).add_(param.data, alpha=1 - ema_decay)
```

The EMA model is used for validation and inference. It averages out the noise
from individual gradient steps, producing more stable predictions. The training
model keeps learning aggressively while the EMA model provides a "smoothed
consensus."

---

## 20.12 Label Smoothing

Standard cross-entropy uses "hard" targets: the correct token gets probability 1.0, everything else gets 0.0. Label smoothing softens this by redistributing a small fraction of the probability mass to all other tokens.

With the default `label_smoothing=0.1`:

```
Hard targets:      correct=1.0,  others=0.0
Smoothed targets:  correct=0.9,  others=0.1/31999 each
```

**Why it helps:**

Think of a strict teacher who only accepts one exact answer versus a teacher who says "this answer is best, but those others are not completely worthless." The strict teacher trains students to be extremely confident -- even overconfident. The relaxed teacher produces students who are well-calibrated: confident when they should be, uncertain when the answer is genuinely ambiguous.

In practice, label smoothing:
- **Prevents overconfidence**: The model does not push logits to extreme values
- **Improves calibration**: Predicted probabilities better match actual correctness rates
- **Acts as regularization**: Slightly penalizes the model for being too sure, reducing overfitting

The 0.1 default is used across all training scripts that use cross-entropy loss: `train_text.py`, `train_vision.py`, `train_ocr.py`, and `sft_omni.py`. This is the same value used by most production language models.

```python
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
```

---

## 20.13 VRAM Budget Guide (16GB GPU — RTX 5070 Ti)

Here is what fits in 16GB with all optimizations enabled:

```
+------------------------------------------------------------------+
| Component          | VRAM Usage  | Notes                         |
|--------------------|-------------|-------------------------------|
| PyTorch + CUDA     | ~1.5 GB     | fixed overhead                |
| Available          | ~14.5 GB    | for model + training          |
+------------------------------------------------------------------+

Stage A — Thinker:
  Model (25M params, fp16):        ~50 MB
  Optimizer states:                ~200 MB
  Activations (BS=32, seq=512):    ~2 GB
  Gradients:                       ~100 MB
  Total:                           ~2.4 GB     ← fits easily at BS=32

Stage C — Vision:
  Model + projection:              ~80 MB
  Optimizer states:                ~320 MB
  Activations (BS=32, 224x224):    ~3 GB
  Gradients:                       ~160 MB
  Total:                           ~3.6 GB     ← BS=32 with accum=8

Stage D — Vocoder (largest memory consumer):
  Generator:                       ~200 MB
  Discriminator(s):                ~400 MB
  Optimizer states (both):         ~2.4 GB
  Activations (BS=2):              ~4 GB
  Gradients:                       ~1.2 GB
  Total:                           ~8.2 GB     ← tight at BS=2
```

The vocoder is the memory bottleneck because it trains both a generator and
multiple discriminator networks. Batch size of 2 with gradient accumulation is
the practical limit on 16GB.

```
VRAM usage by stage (16GB budget):

16 |################| ← VRAM limit
   |                |
12 |                |
   |   ########    |
 8 |   # Vocoder   |  ← 8.2 GB
   |   ########    |
 4 |   #  #  ####  |
   | ### ## ## SFT |
 2 | #Th #Vi ##    |
   | ### ## ####   |
 0 +---+--+--+--+--+
    A   B  C  D  E
```

---

## 20.14 Optimization Checklist

```
[x] use_amp = True                     (or bfloat16 on Ampere+)
[x] Flash Attention via SDPA           (automatic in PyTorch 2.0+)
[x] gradient_accumulation_steps set    (effective batch >= 32)
[x] max_grad_norm = 1.0               (gradient clipping)
[x] zero_grad(set_to_none=True)        (free gradient memory)
[x] pin_memory = True                  (faster transfers)
[x] cudnn.benchmark = True             (auto-tune convolutions)
[x] torch.compile() if not RTX 50xx   (20-50% speedup)
[x] Cached RoPE frequencies            (compute once, slice)
[x] Pre-allocated causal masks         (register_buffer)
[x] num_workers = 2                    (keep GPU fed)
[x] EMA for validation model           (smoother weights)
```

Every optimization here is already implemented in the training scripts. This
chapter explains **why** each one matters so you can make informed decisions
when tuning for your specific hardware.

---

## 20.15 Benchmark Results (Synthetic Data)

The following results were measured on synthetic/deterministic test data to validate pipeline correctness:

| Component | Metric | Result |
|-----------|--------|--------|
| Thinker (text) | Top-1 accuracy | 65.25% |
| Thinker (text) | Top-5 accuracy | 92.61% |
| Audio Encoder | CER (Character Error Rate) | 0% |
| Audio Encoder | WER (Word Error Rate) | 0% |
| Vision Encoder | Diversity score | 0.88 |
| Talker | Top-5 accuracy | 90% |

These are on deterministic synthetic data (generated via `scripts/make_deterministic_data.py`) and represent pipeline verification, not production benchmarks. Real-world performance will vary with natural data and larger model sizes.

**Next:** Chapter 21 covers what to do when things go wrong.
