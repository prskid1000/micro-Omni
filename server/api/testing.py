"""Testing API — run evaluation scripts, query results."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

TEST_MAP: dict[str, dict[str, str]] = {
    "test_thinker": {"module": "test.test_thinker", "checkpoint": "checkpoints/thinker_tiny"},
    "test_audio_enc": {"module": "test.test_audio_enc", "checkpoint": "checkpoints/audio_enc_tiny"},
    "test_vision": {"module": "test.test_vision", "checkpoint": "checkpoints/vision_tiny"},
    "test_talker": {"module": "test.test_talker", "checkpoint": "checkpoints/talker_tiny"},
    "test_vocoder": {"module": "test.test_vocoder", "checkpoint": "checkpoints/vocoder_tiny"},
    "test_ocr": {"module": "test.test_ocr", "checkpoint": "checkpoints/ocr_tiny"},
    "test_sft": {"module": "test.test_sft", "checkpoint": "checkpoints/omni_sft_tiny"},
}


def handle_get(handler: Any, path: str, query: dict[str, list[str]]) -> None:
    from server.app import get_process_manager
    pm = get_process_manager()

    if path == "/api/testing/status":
        processes = pm.get_all(category="testing")
        handler.send_json({"ok": True, "processes": processes})
        return

    if path.startswith("/api/testing/results/"):
        script = path.split("/")[-1]
        metrics_file = Path("logs/metrics") / f"{script}.jsonl"
        rows: list[dict] = []
        if metrics_file.exists():
            try:
                with metrics_file.open("r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            try:
                                rows.append(json.loads(line))
                            except Exception:
                                continue
            except Exception:
                pass
        handler.send_json({"ok": True, "script": script, "results": rows})
        return

    handler.send_error_json(404, f"Unknown testing endpoint: {path}")


def handle_post(handler: Any, path: str, body: dict[str, Any]) -> None:
    from server.app import get_process_manager
    pm = get_process_manager()

    if path == "/api/testing/run":
        script = str(body.get("script", ""))
        if script not in TEST_MAP:
            handler.send_error_json(400, f"Unknown test script: {script}. Valid: {list(TEST_MAP.keys())}")
            return

        checkpoint = body.get("checkpoint") or TEST_MAP[script]["checkpoint"]
        num_samples = body.get("num_samples")

        extra_args = ["--checkpoint", checkpoint]
        if num_samples:
            extra_args += ["--num_samples", str(num_samples)]

        try:
            mp = pm.start(
                category="testing",
                stage=script,
                module=TEST_MAP[script]["module"],
                extra_args=extra_args,
            )
            handler.send_json({"ok": True, "pid": mp.pid, "script": script, "checkpoint": checkpoint})
        except RuntimeError as e:
            handler.send_error_json(409, str(e))
        return

    handler.send_error_json(404, f"Unknown testing endpoint: {path}")
