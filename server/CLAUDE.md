# server/ — Unified micro-Omni Server

## Overview
Single-page dashboard + REST API for training management, metrics visualization, testing, inference, and export. No external Python dependencies — uses stdlib only. Frontend uses Apache ECharts from CDN.

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
    metrics.py         GET /api/metrics/*
    training.py        POST /api/training/start|stop, GET status/pipeline/logs
    testing.py         POST /api/testing/run, GET status/results
    inference.py       POST /api/inference/chat|multimodal|unload
    export.py          POST /api/export/run, GET status
    system.py          GET /api/system/gpu|checkpoints|configs
  static/
    index.html         SPA dashboard shell
    style.css          Dark glassmorphism theme
    app.js             ECharts charts, polling, pipeline UI, inference/testing panels
```

## API Quick Reference

| Method | Endpoint | Purpose |
|--------|----------|---------|
| GET | `/api/metrics/files` | List metric JSONL files |
| GET | `/api/metrics/data?file=X&since=T` | Get metric rows (incremental) |
| GET | `/api/metrics/summary` | Aggregated latest per metric |
| POST | `/api/training/start` | Start training stage `{stage: "A"}` |
| POST | `/api/training/stop` | Stop training `{stage: "A"}` |
| GET | `/api/training/pipeline` | Pipeline status (A-G) |
| GET | `/api/training/logs/<stage>` | Tail log output |
| POST | `/api/testing/run` | Run test `{script, checkpoint}` |
| POST | `/api/inference/chat` | Text inference `{text, ckpt_dir}` |
| POST | `/api/export/run` | Trigger model export |
| GET | `/api/system/gpu` | GPU memory/utilization |
| GET | `/api/system/checkpoints` | Checkpoint inventory |
| GET | `/api/system/configs` | Available training configs |

## Key Design Decisions
- **No torch import in server** (except inference API, lazy-loaded). GPU stats via `nvidia-smi`.
- **Single-GPU enforcement**: ProcessManager returns HTTP 409 if GPU busy.
- **JSONL source of truth**: No database. Reads `logs/metrics/*.jsonl` directly.
- **Inference VRAM safety**: Engine auto-unloads before training/testing starts.
- **Windows-first**: `taskkill /F /PID /T`, `.venv/Scripts/python.exe`, `CREATE_NEW_PROCESS_GROUP`.

## Frontend
- Apache ECharts for charts (zoom, pan, dual Y-axis, log scale, EMA smoothing)
- 5-second polling with live/pause toggle
- Pipeline cards (A-G) with start/stop buttons
- Summary cards: step, best val loss, LR, ETA, throughput, GPU memory
- Tabs: Latest Values, Checkpoints, Hyperparameters, Events, Logs
- Collapsible: Inference panel, Testing panel
