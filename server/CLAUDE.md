# server/ — Unified micro-Omni Server

## Overview
Single-page dashboard + REST API for the full training lifecycle: metrics visualization, training management, testing, inference (3 modes), export, config editing, and HP tuning (Optuna). Frontend uses Apache ECharts from CDN. Backend uses Python stdlib only (no Flask/FastAPI). HP tuning requires `optuna` package.

## Entry Point
```bash
python -m server                          # Default: http://127.0.0.1:8000
python -m server --port 9000 --no-open    # Custom port, no auto-open
```

## Architecture
```
server/
  __main__.py          CLI entry point
  app.py               ThreadingHTTPServer + route dispatch
  process_manager.py   Subprocess lifecycle (single-GPU enforcement)
  api/
    metrics.py         GET /api/metrics/* — JSONL metric data + summary
    training.py        POST start/stop/clear, GET status/pipeline/logs
    testing.py         POST run tests, GET status/results
    inference.py       POST chat/standalone/huggingface/unload (3 modes)
    export.py          POST run export, GET status
    system.py          GET/POST gpu/checkpoints/configs (read + write)
    tuning.py          GET search spaces/results, POST start/stop tuning
  static/
    index.html         SPA dashboard shell
    style.css          Dark glassmorphism theme
    app.js             ECharts, polling, pipeline, inference, testing, tuning UI
```

## API Quick Reference

### Metrics
| Method | Endpoint | Purpose |
|--------|----------|---------|
| GET | `/api/metrics/files` | List metric JSONL files |
| GET | `/api/metrics/data?file=X&since=T` | Get metric rows (incremental) |
| GET | `/api/metrics/summary` | Aggregated latest per metric |

### Training
| Method | Endpoint | Purpose |
|--------|----------|---------|
| POST | `/api/training/start` | Start stage `{stage: "A", config?}` |
| POST | `/api/training/stop` | Stop stage `{stage: "A"}` |
| POST | `/api/training/clear` | Delete checkpoint files `{stage: "A"}` |
| GET | `/api/training/status` | All training processes |
| GET | `/api/training/pipeline` | Pipeline status (A-G) with dependencies |
| GET | `/api/training/logs/<stage>` | Tail last 200 log lines |

### Testing & Export
| Method | Endpoint | Purpose |
|--------|----------|---------|
| POST | `/api/testing/run` | Run test `{script, checkpoint, num_samples?}` |
| GET | `/api/testing/status` | Test process status |
| GET | `/api/testing/results/<script>` | Test results from JSONL |
| POST | `/api/export/run` | Trigger model export `{output_dir?}` |
| GET | `/api/export/status` | Export process status |

### Inference (3 modes)
| Method | Endpoint | Purpose |
|--------|----------|---------|
| POST | `/api/inference/chat` | Normal inference from checkpoints (full multimodal: text/image/video/audio/OCR) |
| POST | `/api/inference/standalone` | Standalone export inference (text only) |
| POST | `/api/inference/huggingface` | HuggingFace model inference (text + multimodal) |
| POST | `/api/inference/unload` | Free VRAM |

### System & Config
| Method | Endpoint | Purpose |
|--------|----------|---------|
| GET | `/api/system/gpu` | GPU memory/utilization via nvidia-smi |
| GET | `/api/system/checkpoints` | Checkpoint inventory with metadata |
| GET | `/api/system/configs` | List config files |
| GET | `/api/system/config/<name>` | Read config JSON |
| POST | `/api/system/config/<name>` | Save config JSON `{config: {...}}` |
| GET | `/api/system/checkpoint-config/<name>` | Read checkpoint's config.json |

### HP Tuning (Optuna)
| Method | Endpoint | Purpose |
|--------|----------|---------|
| GET | `/api/tuning/spaces` | Search spaces for all stages (51 unique params) |
| GET | `/api/tuning/results/<stage>` | Study results from SQLite DB |
| GET | `/api/tuning/status` | Tuning process status |
| POST | `/api/tuning/start` | Start tuning `{stage, n_trials, max_steps, params?}` |
| POST | `/api/tuning/stop` | Stop tuning `{stage}` |

## Key Design Decisions
- **No torch import in server** (except inference API, lazy-loaded). GPU stats via `nvidia-smi`.
- **Single-GPU enforcement**: ProcessManager returns HTTP 409 if GPU busy.
- **JSONL source of truth**: No database for metrics. Reads `logs/metrics/*.jsonl` directly.
- **Inference VRAM safety**: Engine auto-unloads before training/testing/tuning starts.
- **Windows-first**: `taskkill /F /PID /T`, `.venv/Scripts/python.exe`, `CREATE_NEW_PROCESS_GROUP`.
- **Config editing**: Configs editable from dashboard with side-by-side checkpoint comparison.
- **HP tuning**: Optuna TPE + MedianPruner. SQLite DB in `logs/hp_tuning_<stage>.db`. Best config auto-saved to `configs/tuned_<config>.json`.

## Frontend
- Apache ECharts for charts (zoom, pan, dual Y-axis, log scale, EMA smoothing)
- 5-second polling with LIVE/PAUSE toggle
- Pipeline cards (A-G) with Start/Stop/Clear buttons
- Summary cards: step, best val loss, LR, ETA, throughput, GPU memory
- Chip-based filters (click-toggle) for file/run/metric selection
- X-axis toggle (step/epoch)
- Tabs: Latest Values, Checkpoints, Hyperparameters, Events, Config Editor, Logs
- Collapsible panels: Inference (3 modes), Testing, HP Tuning
- Config Editor: load/edit/save with side-by-side checkpoint diff

## Tuning Search Spaces (per stage)
| Stage | Params | Key unique ones |
|-------|--------|----------------|
| A - Thinker | 25 | use_mtp, window_size, rope_scaling_factor, kv_groups, use_moe, num_experts |
| B - Audio Enc | 21 | use_attention_pooling, use_augmentation, downsample_time, target_hz |
| C - Vision | 23 | temperature, embed_dim, use_thinker_for_text, use_augmentation |
| D - Talker | 26 | rvq_ema_decay, rvq_gumbel_temp, rvq_reset_threshold, codebooks, frame_rate |
| E - SFT | 17 | proj_lr_mult, val_batch_size |
| F - Vocoder | 22 | lr_g, lr_d, lambda_mel/fm/adv, mpd_periods, msd_num_scales |
| G - OCR | 19 | use_gqa, use_swiglu, use_spiking, use_ltc |

### Metric-Based Optimization
Tuning can optimize real test metrics (CER, perplexity, R@1, etc.) instead of just val_loss.
The UI shows metric selection chips per stage. When metrics are selected, tune.py runs the
test script after each trial and computes a combined objective from the selected metrics.

| Stage | Available Metrics |
|-------|------------------|
| A | val_loss, perplexity, top1/5/10_accuracy |
| B | val_loss, cer, wer, cer_greedy, wer_greedy |
| C | val_loss, diversity_score, i2t_r1/r5, t2i_r1/r5, avg_pairwise_similarity |
| D | val_loss, base_accuracy, res_accuracy, base/res_top5_accuracy, reconstruction_mse, codebook_utilization |
| E | val_loss, perplexity, top1/5_accuracy |
| F | val_loss, mel_mse, mel_mae, spectral_convergence, mcd |
| G | val_loss, cer, wer, exact_match_rate, char_accuracy |
