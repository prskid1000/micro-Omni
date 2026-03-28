"""Metrics API — serves JSONL training/test metrics."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

METRICS_DIR = Path("logs/metrics")

# ── Server-side cache for metrics data ──────────────────────────
_cache: dict[str, dict[str, Any]] = {}  # fname -> {mtime, rows}


def _get_cached_rows(file_name: str) -> list[dict[str, Any]]:
    """Read JSONL with mtime-based caching to avoid re-reading unchanged files."""
    path = METRICS_DIR / file_name
    if not path.exists() or not path.is_file():
        return []

    mtime = path.stat().st_mtime
    cached = _cache.get(file_name)
    if cached and cached["mtime"] == mtime:
        return cached["rows"]

    # Re-read
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if isinstance(obj, dict):
                    rows.append(obj)
            except Exception:
                continue

    _cache[file_name] = {"mtime": mtime, "rows": rows}
    return rows


def _list_files() -> list[str]:
    if not METRICS_DIR.exists():
        return []
    return sorted(p.name for p in METRICS_DIR.glob("*.jsonl") if p.is_file())


def _read_jsonl(file_name: str, since: str | None = None) -> list[dict[str, Any]]:
    all_rows = _get_cached_rows(file_name)
    if not since:
        return list(all_rows)  # return copy
    return [r for r in all_rows if r.get("timestamp", "") > since]


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


def handle_post(handler: Any, path: str, body: dict[str, Any]) -> None:
    if path == "/api/metrics/delete":
        file_name = body.get("file", "")
        if not file_name:
            handler.send_error_json(400, "Missing 'file' field")
            return

        if file_name == "__all__":
            # Delete all metrics files
            removed = []
            for fname in _list_files():
                fpath = METRICS_DIR / fname
                try:
                    fpath.unlink()
                    _cache.pop(fname, None)
                    removed.append(fname)
                except Exception as e:
                    handler.send_error_json(500, f"Failed to delete {fname}: {e}")
                    return
            handler.send_json({"ok": True, "removed": removed, "count": len(removed)})
            return

        safe_name = os.path.basename(file_name)
        if safe_name != file_name:
            handler.send_error_json(400, "Invalid file name")
            return

        fpath = METRICS_DIR / safe_name
        if not fpath.exists():
            handler.send_error_json(404, f"File not found: {safe_name}")
            return

        try:
            fpath.unlink()
            _cache.pop(safe_name, None)
            handler.send_json({"ok": True, "removed": [safe_name], "count": 1})
        except Exception as e:
            handler.send_error_json(500, f"Failed to delete {safe_name}: {e}")
        return

    if path == "/api/metrics/delete-run":
        file_name = body.get("file", "")
        run_id = body.get("run_id", "")
        if not file_name or not run_id:
            handler.send_error_json(400, "Missing 'file' or 'run_id'")
            return

        safe_name = os.path.basename(file_name)
        fpath = METRICS_DIR / safe_name
        if not fpath.exists():
            handler.send_error_json(404, f"File not found: {safe_name}")
            return

        # Read, filter out the run, rewrite
        rows = _get_cached_rows(safe_name)
        kept = [r for r in rows if r.get("run_id") != run_id]
        removed_count = len(rows) - len(kept)

        try:
            with fpath.open("w", encoding="utf-8") as f:
                for r in kept:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            _cache.pop(safe_name, None)  # invalidate cache
            handler.send_json({"ok": True, "run_id": run_id, "removed_rows": removed_count})
        except Exception as e:
            handler.send_error_json(500, f"Failed to rewrite {safe_name}: {e}")
        return

    handler.send_error_json(404, f"Unknown metrics endpoint: {path}")
