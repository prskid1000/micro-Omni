"""Training API — start/stop/resume training stages, query pipeline status."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

STAGE_MAP: dict[str, dict[str, str]] = {
    "A": {"module": "train.train_thinker", "config": "synthetic_thinker.json", "name": "Thinker LLM", "checkpoint_dir": "checkpoints/thinker_tiny"},
    "B": {"module": "train.train_audio_enc", "config": "synthetic_audio_enc.json", "name": "Audio Encoder", "checkpoint_dir": "checkpoints/audio_enc_tiny"},
    "C": {"module": "train.train_vision", "config": "synthetic_vision.json", "name": "Vision Encoder", "checkpoint_dir": "checkpoints/vision_tiny"},
    "D": {"module": "train.train_talker", "config": "synthetic_talker.json", "name": "Talker TTS", "checkpoint_dir": "checkpoints/talker_tiny"},
    "E": {"module": "train.sft_omni", "config": "synthetic_omni_sft.json", "name": "Multimodal SFT", "checkpoint_dir": "checkpoints/omni_sft_tiny"},
    "F": {"module": "train.train_vocoder", "config": "synthetic_vocoder.json", "name": "Vocoder", "checkpoint_dir": "checkpoints/vocoder_tiny"},
    "G": {"module": "train.train_ocr", "config": "synthetic_ocr.json", "name": "OCR", "checkpoint_dir": "checkpoints/ocr_tiny"},
}

# Dependency graph: stage -> list of required predecessor stages
DEPENDENCIES: dict[str, list[str]] = {
    "A": [],
    "B": [],
    "C": [],
    "D": ["A"],
    "E": ["A", "B", "C", "D"],
    "F": [],
    "G": [],
}


def _has_checkpoint(checkpoint_dir: str) -> bool:
    """Check if a checkpoint directory has a trained model."""
    d = Path(checkpoint_dir)
    if not d.exists():
        return False
    return any(d.glob("*.pt"))


def _read_metadata(checkpoint_dir: str) -> dict[str, Any] | None:
    d = Path(checkpoint_dir)
    for f in d.glob("*_metadata.json"):
        try:
            return json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
    return None


def _read_config(checkpoint_dir: str) -> dict[str, Any] | None:
    cfg_path = Path(checkpoint_dir) / "config.json"
    if cfg_path.exists():
        try:
            return json.loads(cfg_path.read_text(encoding="utf-8"))
        except Exception:
            pass
    return None


def _get_pipeline_status(pm: Any) -> dict[str, Any]:
    """Build pipeline status for all stages."""
    stages: dict[str, Any] = {}
    running_processes = pm.get_all(category="training")

    for stage_id, info in STAGE_MAP.items():
        ckpt_dir = info["checkpoint_dir"]
        has_ckpt = _has_checkpoint(ckpt_dir)
        metadata = _read_metadata(ckpt_dir) if has_ckpt else None

        # Check if this stage is currently running
        key = f"training_{stage_id}"
        proc_info = running_processes.get(key)
        is_running = proc_info is not None and proc_info.get("status") == "running"

        # Check dependencies
        deps = DEPENDENCIES.get(stage_id, [])
        missing_deps = [d for d in deps if not _has_checkpoint(STAGE_MAP[d]["checkpoint_dir"])]

        if is_running:
            status = "running"
        elif has_ckpt:
            status = "done"
        elif missing_deps:
            status = "blocked"
        else:
            status = "idle"

        stages[stage_id] = {
            "status": status,
            "name": info["name"],
            "module": info["module"],
            "config": info["config"],
            "checkpoint_dir": ckpt_dir,
            "has_checkpoint": has_ckpt,
            "metadata": metadata,
            "blocked_by": missing_deps,
            "process": proc_info,
        }

    return stages


def handle_get(handler: Any, path: str, query: dict[str, list[str]]) -> None:
    from server.app import get_process_manager
    pm = get_process_manager()

    if path == "/api/training/status":
        processes = pm.get_all(category="training")
        handler.send_json({"ok": True, "processes": processes})
        return

    if path == "/api/training/pipeline":
        stages = _get_pipeline_status(pm)
        handler.send_json({"ok": True, "stages": stages})
        return

    if path.startswith("/api/training/logs/"):
        stage = path.split("/")[-1].upper()
        if stage not in STAGE_MAP:
            handler.send_error_json(400, f"Unknown stage: {stage}")
            return

        # Try server log first, then standard log
        log_candidates = [
            f"logs/server_training_{stage}.log",
            f"logs/train_{STAGE_MAP[stage]['module'].split('.')[-1]}.log",
        ]

        lines: list[str] = []
        log_used = ""
        for log_path in log_candidates:
            if os.path.exists(log_path):
                try:
                    with open(log_path, "r", encoding="utf-8", errors="replace") as f:
                        all_lines = f.readlines()
                        lines = [l.rstrip("\n\r") for l in all_lines[-200:]]
                        log_used = log_path
                except Exception:
                    pass
                break

        handler.send_json({"ok": True, "stage": stage, "lines": lines, "log_file": log_used})
        return

    handler.send_error_json(404, f"Unknown training endpoint: {path}")


def _clear_checkpoint(checkpoint_dir: str) -> dict[str, Any]:
    """Remove all files in a checkpoint directory."""
    d = Path(checkpoint_dir)
    if not d.exists():
        return {"cleared": False, "reason": "Directory does not exist"}

    removed = []
    for f in d.iterdir():
        if f.is_file():
            try:
                f.unlink()
                removed.append(f.name)
            except Exception as e:
                return {"cleared": False, "reason": f"Failed to delete {f.name}: {e}"}

    return {"cleared": True, "removed": removed, "count": len(removed)}


def handle_post(handler: Any, path: str, body: dict[str, Any]) -> None:
    from server.app import get_process_manager
    pm = get_process_manager()

    if path == "/api/training/start":
        stage = str(body.get("stage", "")).upper()
        if stage not in STAGE_MAP:
            handler.send_error_json(400, f"Unknown stage: {stage}. Valid: {list(STAGE_MAP.keys())}")
            return

        config = body.get("config") or STAGE_MAP[stage]["config"]

        # Check dependencies
        deps = DEPENDENCIES.get(stage, [])
        missing = [d for d in deps if not _has_checkpoint(STAGE_MAP[d]["checkpoint_dir"])]
        if missing:
            handler.send_error_json(409, f"Stage {stage} blocked by incomplete stages: {missing}")
            return

        try:
            mp = pm.start(
                category="training",
                stage=stage,
                module=STAGE_MAP[stage]["module"],
                config=config,
            )
            handler.send_json({"ok": True, "pid": mp.pid, "stage": stage, "config": config})
        except RuntimeError as e:
            handler.send_error_json(409, str(e))
        return

    if path == "/api/training/stop":
        stage = str(body.get("stage", "")).upper()
        if stage not in STAGE_MAP:
            handler.send_error_json(400, f"Unknown stage: {stage}")
            return

        key = f"training_{stage}"
        stopped = pm.stop(key)
        handler.send_json({"ok": True, "stage": stage, "stopped": stopped})
        return

    if path == "/api/training/clear":
        stage = str(body.get("stage", "")).upper()
        if stage not in STAGE_MAP:
            handler.send_error_json(400, f"Unknown stage: {stage}")
            return

        # Don't allow clearing while running
        key = f"training_{stage}"
        proc = pm.get_status(key)
        if proc and proc.get("status") == "running":
            handler.send_error_json(409, f"Stage {stage} is currently running — stop it first")
            return

        ckpt_dir = STAGE_MAP[stage]["checkpoint_dir"]
        result = _clear_checkpoint(ckpt_dir)
        handler.send_json({"ok": True, "stage": stage, "checkpoint_dir": ckpt_dir, **result})
        return

    handler.send_error_json(404, f"Unknown training endpoint: {path}")
