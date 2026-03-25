# Appendix C: Configuration Reference

Complete reference for all JSON configuration parameters used across
micro-Omni training scripts.

---

## Model Architecture

| Parameter      | Type  | Default | Description                               |
|----------------|-------|---------|-------------------------------------------|
| `d_model`      | int   | 512     | Hidden dimension of the transformer       |
| `n_layers`     | int   | 8       | Number of transformer blocks              |
| `n_heads`      | int   | 8       | Number of attention heads                 |
| `d_ff`         | int   | 1376    | Feed-forward intermediate dimension       |
| `vocab_size`   | int   | 32000   | Tokenizer vocabulary size                 |
| `ctx_len`      | int   | 2048    | Maximum sequence length (context window)  |
| `dropout`      | float | 0.0     | Dropout rate (0 = no dropout)             |
| `norm_eps`     | float | 1e-6    | Epsilon for RMSNorm                       |

---

## Attention Features

| Parameter      | Type  | Default | Description                               |
|----------------|-------|---------|-------------------------------------------|
| `use_gqa`      | bool  | false   | Enable Grouped Query Attention            |
| `kv_groups`    | int   | 4       | Number of KV head groups (when GQA on)    |
| `use_flash`    | bool  | true    | Use FlashAttention when available         |
| `rope_base`    | float | 10000   | RoPE base frequency                       |

---

## FFN Features

| Parameter      | Type  | Default | Description                               |
|----------------|-------|---------|-------------------------------------------|
| `use_swiglu`   | bool  | true    | Use SwiGLU activation (vs standard ReLU)  |
| `use_moe`      | bool  | false   | Enable Mixture of Experts in FFN          |
| `moe_experts`  | int   | 4       | Number of experts (when MoE on)           |
| `moe_top_k`    | int   | 2       | Number of active experts per token        |

---

## Neuromorphic Extensions (Arthemis)

| Parameter      | Type  | Default | Description                               |
|----------------|-------|---------|-------------------------------------------|
| `use_spiking`  | bool  | false   | Enable LIF spiking neurons in attention   |
| `spike_thresh` | float | 1.0     | Spiking threshold voltage                 |
| `use_ltc`      | bool  | false   | Enable Liquid Time-Constant neurons in FFN|
| `ltc_tau`      | float | 1.0     | Base time constant for LTC                |

---

## Training

| Parameter                    | Type  | Default  | Description                          |
|------------------------------|-------|----------|--------------------------------------|
| `lr`                         | float | 3e-4     | Peak learning rate                   |
| `wd`                         | float | 0.1      | Weight decay (AdamW)                 |
| `warmup_steps`               | int   | 500      | Linear warmup steps                  |
| `max_steps`                  | int   | 50000    | Total training steps                 |
| `batch_size`                 | int   | 32       | Batch size per step                  |
| `gradient_accumulation_steps`| int   | 1        | Gradient accumulation factor         |
| `max_grad_norm`              | float | 1.0      | Gradient clipping norm               |
| `label_smoothing`            | float | 0.1      | Label smoothing factor (0 = off)     |
| `seed`                       | int   | 42       | Random seed                          |
| `save_every`                 | int   | 1000     | Save checkpoint every N steps        |
| `log_every`                  | int   | 100      | Log metrics every N steps            |

---

## Performance

| Parameter      | Type  | Default | Description                               |
|----------------|-------|---------|-------------------------------------------|
| `use_amp`      | bool  | true    | Automatic Mixed Precision (float16)       |
| `use_flash`    | bool  | true    | FlashAttention (reduces memory)           |
| `use_compile`  | bool  | false   | torch.compile() for graph optimization    |
| `num_workers`  | int   | 0       | DataLoader worker processes               |
| `pin_memory`   | bool  | true    | Pin memory for faster GPU transfer        |

---

## Validation

| Parameter          | Type  | Default | Description                            |
|--------------------|-------|---------|----------------------------------------|
| `val_split`        | float | 0.05    | Fraction of data reserved for validation|
| `val_freq`         | int   | 500     | Run validation every N steps           |
| `val_batches`      | int   | 20      | Number of batches per validation run   |
| `val_loss_threshold`| float| 10.0    | Skip checkpoints above this val loss   |

---

## Audio-Specific (audio_enc, talker, vocoder configs)

| Parameter          | Type  | Default | Description                            |
|--------------------|-------|---------|----------------------------------------|
| `sample_rate`      | int   | 16000   | Audio sample rate in Hz                |
| `n_mels`           | int   | 80      | Number of mel spectrogram bins         |
| `n_fft`            | int   | 1024    | FFT window size                        |
| `hop_length`       | int   | 256     | Hop length for STFT                    |
| `downsample_time`  | int   | 4       | Temporal downsampling factor           |
| `frame_ms`         | float | 16.0    | Frame duration in milliseconds         |
| `n_codebooks`      | int   | 4       | Number of RVQ codebook layers          |
| `codebook_size`    | int   | 1024    | Entries per codebook                   |
| `codebook_dim`     | int   | 256     | Dimension of each codebook entry       |

---

## Vision-Specific (vision configs)

| Parameter      | Type  | Default | Description                               |
|----------------|-------|---------|-------------------------------------------|
| `img_size`     | int   | 224     | Input image size (square)                 |
| `patch`        | int   | 16      | Patch size for ViT                        |
| `embed_dim`    | int   | 384     | Patch embedding dimension                 |
| `temperature`  | float | 0.07    | InfoNCE temperature (learnable init)      |
| `n_layers`     | int   | 6       | Vision transformer depth                  |
| `n_heads`      | int   | 6       | Vision attention heads                    |

---

## Vocoder-Specific (HiFi-GAN)

| Parameter            | Type  | Default | Description                         |
|----------------------|-------|---------|-------------------------------------|
| `lambda_mel`         | float | 45.0    | Mel reconstruction loss weight      |
| `lambda_fm`          | float | 2.0     | Feature matching loss weight        |
| `lambda_adv`         | float | 1.0     | Adversarial loss weight             |
| `upsample_rates`     | list  | [8,8,2,2]| Upsampling factors per layer      |
| `upsample_kernels`   | list  | [16,16,4,4]| Kernel sizes for upsampling      |
| `resblock_channels`  | int   | 256     | Channels in residual blocks         |
| `resblock_dilations` | list  | [[1,3,5],[1,3,5],[1,3,5]]| Dilation patterns |
| `disc_periods`       | list  | [2,3,5,7,11]| Multi-period discriminator periods|
| `disc_lr`            | float | 2e-4    | Discriminator learning rate         |

---

## OCR-Specific

| Parameter      | Type  | Default | Description                               |
|----------------|-------|---------|-------------------------------------------|
| `max_seq_len`  | int   | 256     | Maximum output text length                |
| `decoder_layers`| int  | 4       | OCR decoder depth                         |
| `decoder_heads`| int   | 4       | OCR decoder attention heads               |
| `decoder_dim`  | int   | 256     | OCR decoder hidden dimension              |
| `char_vocab`   | int   | 8000    | Character-level vocabulary size           |

---

## Example Config File

`configs/thinker_tiny.json`:

```json
{
  "d_model": 512,
  "n_layers": 8,
  "n_heads": 8,
  "d_ff": 1376,
  "vocab_size": 32000,
  "ctx_len": 2048,
  "use_gqa": false,
  "use_swiglu": true,
  "use_moe": false,
  "use_spiking": false,
  "use_ltc": false,
  "use_flash": true,
  "use_amp": true,
  "lr": 3e-4,
  "wd": 0.1,
  "warmup_steps": 500,
  "max_steps": 50000,
  "batch_size": 32,
  "gradient_accumulation_steps": 1,
  "val_split": 0.05,
  "val_freq": 500,
  "save_every": 1000,
  "seed": 42
}
```
