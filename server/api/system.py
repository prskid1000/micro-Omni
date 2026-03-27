"""System API — GPU status, checkpoints, configs."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any


def _get_gpu_info() -> dict[str, Any] | None:
    """Get GPU info via nvidia-smi (no torch import needed)."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.used,memory.total,utilization.gpu,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode != 0:
            return None

        line = result.stdout.strip().split("\n")[0]
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 5:
            return None

        return {
            "name": parts[0],
            "memory_used_mb": int(parts[1]),
            "memory_total_mb": int(parts[2]),
            "memory_percent": round(int(parts[1]) / max(int(parts[2]), 1) * 100, 1),
            "utilization_percent": int(parts[3]),
            "temperature_c": int(parts[4]),
        }
    except Exception:
        return None


def _scan_checkpoints() -> list[dict[str, Any]]:
    """Scan checkpoints/ directory for saved models."""
    ckpt_root = Path("checkpoints")
    if not ckpt_root.exists():
        return []

    results = []
    for d in sorted(ckpt_root.iterdir()):
        if not d.is_dir():
            continue

        info: dict[str, Any] = {
            "name": d.name,
            "path": str(d),
            "has_config": (d / "config.json").exists(),
            "has_model": any(d.glob("*.pt")),
            "metadata": None,
            "config": None,
            "size_mb": 0,
            "modified": None,
        }

        # Read metadata
        for mf in d.glob("*_metadata.json"):
            try:
                info["metadata"] = json.loads(mf.read_text(encoding="utf-8"))
            except Exception:
                pass
            break

        # Read config summary (selected fields only)
        cfg_path = d / "config.json"
        if cfg_path.exists():
            try:
                full_cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
                info["config"] = {
                    k: full_cfg[k] for k in [
                        "d_model", "n_layers", "n_heads", "d_ff", "lr", "batch_size",
                        "max_steps", "max_epochs", "dropout", "use_gqa", "use_moe",
                        "use_amp", "vocab_size", "ctx_len",
                    ] if k in full_cfg
                }
            except Exception:
                pass

        # Total size of .pt files
        total_bytes = sum(f.stat().st_size for f in d.glob("*.pt"))
        info["size_mb"] = round(total_bytes / (1024 * 1024), 1)

        # Latest modification time
        pt_files = list(d.glob("*.pt"))
        if pt_files:
            latest = max(f.stat().st_mtime for f in pt_files)
            from datetime import datetime, timezone
            info["modified"] = datetime.fromtimestamp(latest, tz=timezone.utc).isoformat()

        results.append(info)

    return results


def _list_configs() -> list[str]:
    cfg_dir = Path("configs")
    if not cfg_dir.exists():
        return []
    return sorted(f.name for f in cfg_dir.glob("*.json"))


def handle_get(handler: Any, path: str, query: dict[str, list[str]]) -> None:
    if path == "/api/system/gpu":
        gpu = _get_gpu_info()
        handler.send_json({"ok": True, "gpu": gpu})
        return

    if path == "/api/system/checkpoints":
        checkpoints = _scan_checkpoints()
        handler.send_json({"ok": True, "checkpoints": checkpoints})
        return

    if path == "/api/system/configs":
        configs = _list_configs()
        handler.send_json({"ok": True, "configs": configs})
        return

    if path.startswith("/api/system/config/"):
        name = path.split("/")[-1]
        safe_name = os.path.basename(name)
        if safe_name != name:
            handler.send_error_json(400, "Invalid config name")
            return

        cfg_path = Path("configs") / safe_name
        if not cfg_path.exists():
            handler.send_error_json(404, f"Config not found: {safe_name}")
            return

        try:
            config = json.loads(cfg_path.read_text(encoding="utf-8"))
            handler.send_json({"ok": True, "name": safe_name, "config": config})
        except Exception as e:
            handler.send_error_json(500, f"Error reading config: {e}")
        return

    # Read checkpoint config
    if path.startswith("/api/system/checkpoint-config/"):
        name = path.split("/")[-1]
        safe_name = os.path.basename(name)
        cfg_path = Path("checkpoints") / safe_name / "config.json"
        if not cfg_path.exists():
            handler.send_error_json(404, f"Checkpoint config not found: {safe_name}")
            return
        try:
            config = json.loads(cfg_path.read_text(encoding="utf-8"))
            handler.send_json({"ok": True, "name": safe_name, "source": "checkpoint", "config": config})
        except Exception as e:
            handler.send_error_json(500, f"Error reading checkpoint config: {e}")
        return

    handler.send_error_json(404, f"Unknown system endpoint: {path}")


def handle_post(handler: Any, path: str, body: dict[str, Any]) -> None:
    """Save config changes."""
    if path.startswith("/api/system/config/"):
        name = path.split("/")[-1]
        safe_name = os.path.basename(name)
        if safe_name != name or not safe_name.endswith(".json"):
            handler.send_error_json(400, "Invalid config name")
            return

        config = body.get("config")
        if not config or not isinstance(config, dict):
            handler.send_error_json(400, "Missing 'config' dict in body")
            return

        cfg_path = Path("configs") / safe_name
        try:
            cfg_path.write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            handler.send_json({"ok": True, "name": safe_name, "saved": True})
        except Exception as e:
            handler.send_error_json(500, f"Error saving config: {e}")
        return

    handler.send_error_json(404, f"Unknown system endpoint: {path}")
