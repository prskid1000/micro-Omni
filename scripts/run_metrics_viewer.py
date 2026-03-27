import json
import os
import argparse
import threading
import webbrowser
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from urllib.parse import parse_qs, urlparse


class MetricsServerHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        self.repo_root = Path(os.getcwd()).resolve()
        self.metrics_dir = self.repo_root / "logs" / "metrics"
        super().__init__(*args, directory=str(self.repo_root), **kwargs)

    def _send_json(self, payload: dict, status: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _list_metric_files(self) -> list[str]:
        if not self.metrics_dir.exists():
            return []
        return sorted([p.name for p in self.metrics_dir.glob("*.jsonl") if p.is_file()])

    def _read_jsonl(self, file_name: str) -> list[dict]:
        path = self.metrics_dir / file_name
        rows: list[dict] = []
        if not path.exists() or not path.is_file():
            return rows
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
        return rows

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/metrics/files":
            files = self._list_metric_files()
            self._send_json({"ok": True, "files": files})
            return

        if parsed.path == "/api/metrics/data":
            qs = parse_qs(parsed.query or "")
            file_name = (qs.get("file") or [""])[0].strip()
            if not file_name:
                self._send_json({"ok": False, "error": "missing 'file' query parameter"}, status=400)
                return
            if file_name == "__all__":
                files = self._list_metric_files()
                out = {}
                for name in files:
                    out[name] = self._read_jsonl(name)
                self._send_json({"ok": True, "file_data": out})
                return
            safe_name = os.path.basename(file_name)
            if safe_name != file_name:
                self._send_json({"ok": False, "error": "invalid file name"}, status=400)
                return
            rows = self._read_jsonl(safe_name)
            self._send_json({"ok": True, "file": safe_name, "rows": rows})
            return

        super().do_GET()


def _start_server(host: str, preferred_port: int) -> tuple[ThreadingHTTPServer, int]:
    # Try preferred port first, then scan a small range.
    for port in [preferred_port] + list(range(preferred_port + 1, preferred_port + 20)):
        try:
            return ThreadingHTTPServer((host, port), MetricsServerHandler), port
        except OSError:
            continue
    raise OSError(f"Could not bind server on ports {preferred_port}-{preferred_port + 19}")


def main() -> None:
    print("DEPRECATED: Use 'python -m server' for the unified dashboard.", flush=True)
    print("  This script still works but will be removed in a future release.\n", flush=True)
    parser = argparse.ArgumentParser(description="Run metrics viewer server with API endpoints")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Preferred port (default: 8000)")
    parser.add_argument("--no-open", action="store_true", help="Do not auto-open browser")
    args = parser.parse_args()

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(repo_root)
    host = args.host
    server, port = _start_server(host, args.port)
    url = f"http://{host}:{port}/scripts/metrics_viewer.html"
    print(f"Serving repo at http://{host}:{port}", flush=True)
    print(f"Metrics API: http://{host}:{port}/api/metrics/files", flush=True)
    if not args.no_open:
        print(f"Opening {url}", flush=True)
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    else:
        print(f"Viewer URL: {url}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
