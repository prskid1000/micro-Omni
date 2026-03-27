"""Metrics API — serves JSONL training/test metrics."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

METRICS_DIR = Path("logs/metrics")


def _list_files() -> list[str]:
    if not METRICS_DIR.exists():
        return []
    return sorted(p.name for p in METRICS_DIR.glob("*.jsonl") if p.is_file())


def _read_jsonl(file_name: str, since: str | None = None) -> list[dict[str, Any]]:
    path = METRICS_DIR / file_name
    rows: list[dict[str, Any]] = []
    if not path.exists() or not path.is_file():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if not isinstance(obj, dict):
                    continue
                if since and obj.get("timestamp", "") <= since:
                    continue
                rows.append(obj)
            except Exception:
                continue
    return rows


def _build_summary() -> dict[str, Any]:
    """Aggregate latest value per (file, run_id, metric_name)."""
    summary: dict[str, Any] = {}
    for fname in _list_files():
        rows = _read_jsonl(fname)
        if not rows:
            continue

        file_info: dict[str, Any] = {
            "runs": {},
            "total_rows": len(rows),
        }

        for r in rows:
            phase = str(r.get("phase", ""))
            if phase == "event":
                continue
            run_id = str(r.get("run_id", "unknown"))
            metric_name = str(r.get("metric_name", ""))
            step = r.get("step", 0) or 0

            if run_id not in file_info["runs"]:
                file_info["runs"][run_id] = {
                    "latest_step": 0,
                    "latest_epoch": 0,
                    "metrics": {},
                    "event_count": 0,
                }

            run_info = file_info["runs"][run_id]

            if step >= run_info["latest_step"]:
                run_info["latest_step"] = step
                run_info["latest_epoch"] = r.get("epoch", 0) or 0

            existing = run_info["metrics"].get(metric_name)
            if existing is None or step >= existing.get("step", 0):
                run_info["metrics"][metric_name] = {
                    "value": r.get("metric_value"),
                    "step": step,
                    "epoch": r.get("epoch"),
                    "lr": r.get("lr"),
                    "timestamp": r.get("timestamp"),
                }

        # Count events separately
        for r in rows:
            if str(r.get("phase", "")) == "event":
                run_id = str(r.get("run_id", "unknown"))
                if run_id in file_info["runs"]:
                    file_info["runs"][run_id]["event_count"] += 1

        summary[fname] = file_info

    return summary


def handle_get(handler: Any, path: str, query: dict[str, list[str]]) -> None:
    if path == "/api/metrics/files":
        files = _list_files()
        handler.send_json({"ok": True, "files": files})
        return

    if path == "/api/metrics/data":
        file_name = (query.get("file") or [""])[0].strip()
        since = (query.get("since") or [None])[0]

        if not file_name:
            handler.send_error_json(400, "Missing 'file' query parameter")
            return

        if file_name == "__all__":
            file_data: dict[str, list[dict]] = {}
            for name in _list_files():
                file_data[name] = _read_jsonl(name, since=since)
            handler.send_json({"ok": True, "file_data": file_data})
            return

        safe_name = os.path.basename(file_name)
        if safe_name != file_name:
            handler.send_error_json(400, "Invalid file name")
            return

        rows = _read_jsonl(safe_name, since=since)
        handler.send_json({"ok": True, "file": safe_name, "rows": rows})
        return

    if path == "/api/metrics/summary":
        summary = _build_summary()
        handler.send_json({"ok": True, "summary": summary})
        return

    handler.send_error_json(404, f"Unknown metrics endpoint: {path}")
