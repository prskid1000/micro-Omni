"""Main HTTP server with route dispatch and static file serving."""

from __future__ import annotations

import json
import mimetypes
import os
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from server.api import metrics as metrics_api
from server.api import training as training_api
from server.api import testing as testing_api
from server.api import inference as inference_api
from server.api import export as export_api
from server.api import system as system_api
from server.api import tuning as tuning_api
from server.process_manager import ProcessManager

# Singleton process manager — initialised in run_server()
_pm: ProcessManager | None = None


def get_process_manager() -> ProcessManager:
    assert _pm is not None, "ProcessManager not initialised"
    return _pm


# ── Helpers ──────────────────────────────────────────────────────

REPO_ROOT = Path(__file__).resolve().parent.parent
STATIC_DIR = Path(__file__).resolve().parent / "static"


def _content_type(path: str) -> str:
    ct, _ = mimetypes.guess_type(path)
    return ct or "application/octet-stream"


# ── Handler ──────────────────────────────────────────────────────

class RequestHandler(BaseHTTPRequestHandler):
    """Route requests to API modules or serve static files."""

    # Silence per-request log lines
    def log_message(self, fmt: str, *args: object) -> None:
        pass

    # ── JSON helpers ─────────────────────────────────────────────

    def send_json(self, payload: dict, status: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def read_body(self) -> dict:
        length = int(self.headers.get("Content-Length", 0))
        if length == 0:
            return {}
        raw = self.rfile.read(length)
        return json.loads(raw.decode("utf-8"))

    def send_error_json(self, status: int, message: str) -> None:
        self.send_json({"ok": False, "error": message}, status=status)

    # ── Static files ─────────────────────────────────────────────

    def _serve_static(self, rel_path: str) -> None:
        if rel_path in ("", "/"):
            rel_path = "index.html"
        rel_path = rel_path.lstrip("/")
        # Strip /static/ prefix since files are already in STATIC_DIR
        if rel_path.startswith("static/"):
            rel_path = rel_path[len("static/"):]
        file_path = STATIC_DIR / rel_path
        # Security: ensure resolved path is inside STATIC_DIR
        try:
            file_path = file_path.resolve()
            if not str(file_path).startswith(str(STATIC_DIR.resolve())):
                self.send_error_json(403, "Forbidden")
                return
        except Exception:
            self.send_error_json(400, "Bad path")
            return

        if not file_path.is_file():
            self.send_error_json(404, f"Not found: {rel_path}")
            return

        ct = _content_type(str(file_path))
        data = file_path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", ct)
        self.send_header("Content-Length", str(len(data)))
        if ct.startswith("text/") or ct == "application/javascript":
            self.send_header("Cache-Control", "no-cache")
        else:
            self.send_header("Cache-Control", "public, max-age=86400")
        self.end_headers()
        self.wfile.write(data)

    # ── Route dispatch ───────────────────────────────────────────

    def _route_get(self, path: str, query: dict[str, list[str]]) -> None:
        if path.startswith("/api/metrics/"):
            metrics_api.handle_get(self, path, query)
        elif path.startswith("/api/training/"):
            training_api.handle_get(self, path, query)
        elif path.startswith("/api/testing/"):
            testing_api.handle_get(self, path, query)
        elif path.startswith("/api/export/"):
            export_api.handle_get(self, path, query)
        elif path.startswith("/api/system/"):
            system_api.handle_get(self, path, query)
        elif path.startswith("/api/tuning/"):
            tuning_api.handle_get(self, path, query)
        elif path.startswith("/api/"):
            self.send_error_json(404, f"Unknown API route: {path}")
        else:
            self._serve_static(path)

    def _route_post(self, path: str, body: dict) -> None:
        if path.startswith("/api/training/"):
            training_api.handle_post(self, path, body)
        elif path.startswith("/api/testing/"):
            testing_api.handle_post(self, path, body)
        elif path.startswith("/api/inference/"):
            inference_api.handle_post(self, path, body)
        elif path.startswith("/api/export/"):
            export_api.handle_post(self, path, body)
        elif path.startswith("/api/system/"):
            system_api.handle_post(self, path, body)
        elif path.startswith("/api/tuning/"):
            tuning_api.handle_post(self, path, body)
        else:
            self.send_error_json(404, f"Unknown API route: {path}")

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        query = parse_qs(parsed.query or "")
        self._route_get(parsed.path, query)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        try:
            body = self.read_body()
        except Exception as e:
            self.send_error_json(400, f"Invalid JSON body: {e}")
            return
        self._route_post(parsed.path, body)

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()


# ── Server bootstrap ─────────────────────────────────────────────

def _find_port(host: str, preferred: int) -> tuple[ThreadingHTTPServer, int]:
    for port in [preferred] + list(range(preferred + 1, preferred + 20)):
        try:
            srv = ThreadingHTTPServer((host, port), RequestHandler)
            return srv, port
        except OSError:
            continue
    raise OSError(f"Could not bind on ports {preferred}-{preferred + 19}")


def run_server(*, host: str = "127.0.0.1", port: int = 8000, auto_open: bool = True) -> None:
    global _pm

    repo_root = str(REPO_ROOT)
    os.chdir(repo_root)

    # Resolve python executable
    venv_python = REPO_ROOT / ".venv" / "Scripts" / "python.exe"
    if not venv_python.exists():
        venv_python = REPO_ROOT / ".venv" / "bin" / "python"
    if not venv_python.exists():
        import sys
        venv_python = Path(sys.executable)

    _pm = ProcessManager(repo_root=repo_root, python_exe=str(venv_python))

    # Register inference unload callback
    inference_api.register_unload_callback(_pm)

    server, actual_port = _find_port(host, port)
    url = f"http://{host}:{actual_port}/"

    print(f"micro-Omni server running at {url}", flush=True)
    print(f"  API:       {url}api/", flush=True)
    print(f"  Dashboard: {url}", flush=True)
    print(f"  Python:    {venv_python}", flush=True)

    if auto_open:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...", flush=True)
    finally:
        _pm.stop_all()
        server.server_close()
