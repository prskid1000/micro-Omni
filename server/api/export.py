"""Export API — trigger model export to HF-compatible format."""

from __future__ import annotations

from typing import Any


def handle_get(handler: Any, path: str, query: dict[str, list[str]]) -> None:
    from server.app import get_process_manager
    pm = get_process_manager()

    if path == "/api/export/status":
        processes = pm.get_all(category="export")
        handler.send_json({"ok": True, "processes": processes})
        return

    handler.send_error_json(404, f"Unknown export endpoint: {path}")


def handle_post(handler: Any, path: str, body: dict[str, Any]) -> None:
    from server.app import get_process_manager
    pm = get_process_manager()

    if path == "/api/export/run":
        output_dir = body.get("output_dir", "export/")

        try:
            mp = pm.start(
                category="export",
                stage="export",
                module="scripts.export",
                extra_args=["--output_dir", output_dir],
            )
            handler.send_json({"ok": True, "pid": mp.pid, "output_dir": output_dir})
        except RuntimeError as e:
            handler.send_error_json(409, str(e))
        return

    handler.send_error_json(404, f"Unknown export endpoint: {path}")
