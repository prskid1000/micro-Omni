"""Subprocess lifecycle manager for training, testing, and export processes."""

from __future__ import annotations

import os
import subprocess
import sys
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


@dataclass
class ManagedProcess:
    """Tracks a single subprocess."""

    pid: int
    process: subprocess.Popen
    category: str                       # "training", "testing", "export"
    stage: str                          # "A"-"G" for training, script name otherwise
    module: str                         # Python module path
    config: str | None
    status: str = "running"             # running / completed / failed / stopped
    start_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    end_time: datetime | None = None
    return_code: int | None = None
    log_file: str = ""

    def to_dict(self) -> dict[str, Any]:
        elapsed = None
        if self.end_time:
            elapsed = (self.end_time - self.start_time).total_seconds()
        elif self.status == "running":
            elapsed = (datetime.now(timezone.utc) - self.start_time).total_seconds()
        return {
            "pid": self.pid,
            "category": self.category,
            "stage": self.stage,
            "module": self.module,
            "config": self.config,
            "status": self.status,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "return_code": self.return_code,
            "elapsed_seconds": round(elapsed, 1) if elapsed is not None else None,
            "log_file": self.log_file,
        }


class ProcessManager:
    """Manages subprocess lifecycle with single-GPU enforcement."""

    MAX_HISTORY = 50

    def __init__(self, repo_root: str, python_exe: str):
        self.repo_root = repo_root
        self.python_exe = python_exe
        self._processes: dict[str, ManagedProcess] = {}
        self._lock = threading.Lock()
        self._on_before_start: list[Callable[[], None]] = []

    def register_before_start(self, callback: Callable[[], None]) -> None:
        self._on_before_start.append(callback)

    # ── Queries ──────────────────────────────────────────────────

    def is_gpu_busy(self) -> bool:
        with self._lock:
            return any(
                p.status == "running"
                for p in self._processes.values()
                if p.category in ("training", "testing", "export")
            )

    def get_running_key(self) -> str | None:
        with self._lock:
            for key, p in self._processes.items():
                if p.status == "running" and p.category in ("training", "testing", "export"):
                    return key
        return None

    def get_status(self, key: str) -> dict[str, Any] | None:
        with self._lock:
            mp = self._processes.get(key)
            return mp.to_dict() if mp else None

    def get_all(self, category: str | None = None) -> dict[str, dict[str, Any]]:
        with self._lock:
            result = {}
            for key, mp in self._processes.items():
                if category is None or mp.category == category:
                    result[key] = mp.to_dict()
            return result

    # ── Lifecycle ────────────────────────────────────────────────

    def start(
        self,
        category: str,
        stage: str,
        module: str,
        config: str | None = None,
        extra_args: list[str] | None = None,
    ) -> ManagedProcess:
        # Fire pre-start callbacks (e.g. unload inference engine)
        for cb in self._on_before_start:
            try:
                cb()
            except Exception:
                pass

        with self._lock:
            if self._is_gpu_busy_unlocked():
                running = self._get_running_info()
                raise RuntimeError(f"GPU busy: {running}")

            # Build command
            cmd = [self.python_exe, "-m", module]
            if config:
                cmd += ["--config", f"configs/{config}"]
            if extra_args:
                cmd += extra_args

            # Log file
            logs_dir = os.path.join(self.repo_root, "logs")
            os.makedirs(logs_dir, exist_ok=True)
            log_file = os.path.join(logs_dir, f"server_{category}_{stage}.log")

            env = {**os.environ, "PYTHONIOENCODING": "utf-8"}

            # Open log file handle
            log_fh = open(log_file, "w", encoding="utf-8")

            # Platform-specific flags
            creation_flags = 0
            if sys.platform == "win32":
                creation_flags = subprocess.CREATE_NEW_PROCESS_GROUP

            proc = subprocess.Popen(
                cmd,
                stdout=log_fh,
                stderr=subprocess.STDOUT,
                env=env,
                cwd=self.repo_root,
                creationflags=creation_flags,
            )

            key = f"{category}_{stage}"
            mp = ManagedProcess(
                pid=proc.pid,
                process=proc,
                category=category,
                stage=stage,
                module=module,
                config=config,
                log_file=log_file,
            )
            self._processes[key] = mp

            # Monitor thread
            t = threading.Thread(
                target=self._monitor,
                args=(key, log_fh),
                daemon=True,
            )
            t.start()

            self._trim_history()
            return mp

    def stop(self, key: str) -> bool:
        with self._lock:
            mp = self._processes.get(key)
            if mp is None or mp.status != "running":
                return False

        # Kill outside lock to avoid deadlock
        try:
            if sys.platform == "win32":
                subprocess.run(
                    ["taskkill", "/F", "/PID", str(mp.pid), "/T"],
                    capture_output=True,
                    timeout=10,
                )
            else:
                mp.process.terminate()
                try:
                    mp.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    mp.process.kill()
        except Exception:
            pass

        with self._lock:
            mp.status = "stopped"
            mp.end_time = datetime.now(timezone.utc)
            mp.return_code = mp.process.returncode
        return True

    def stop_all(self) -> None:
        keys = list(self._processes.keys())
        for key in keys:
            mp = self._processes.get(key)
            if mp is not None and mp.status == "running":
                self.stop(key)

    def clear_record(self, key: str) -> bool:
        """Remove a non-running process record so the stage returns to idle."""
        with self._lock:
            mp = self._processes.get(key)
            if mp is None:
                return False
            if mp.status == "running":
                return False  # Can't clear a running process
            del self._processes[key]
            return True

    # ── Internal ─────────────────────────────────────────────────

    def _is_gpu_busy_unlocked(self) -> bool:
        return any(
            p.status == "running"
            for p in self._processes.values()
            if p.category in ("training", "testing", "export")
        )

    def _get_running_info(self) -> str:
        for key, p in self._processes.items():
            if p.status == "running" and p.category in ("training", "testing", "export"):
                return f"{p.category}/{p.stage} (PID {p.pid})"
        return "unknown"

    def _monitor(self, key: str, log_fh: Any) -> None:
        mp = self._processes.get(key)
        if mp is None:
            return
        try:
            mp.process.wait()
        except Exception:
            pass
        finally:
            try:
                log_fh.close()
            except Exception:
                pass

        with self._lock:
            if mp.status == "running":
                mp.status = "completed" if mp.process.returncode == 0 else "failed"
                mp.end_time = datetime.now(timezone.utc)
                mp.return_code = mp.process.returncode

    def _trim_history(self) -> None:
        """Remove oldest completed entries if history exceeds MAX_HISTORY."""
        completed = [
            (k, p) for k, p in self._processes.items()
            if p.status in ("completed", "failed", "stopped")
        ]
        if len(completed) > self.MAX_HISTORY:
            completed.sort(key=lambda x: x[1].end_time or x[1].start_time)
            for k, _ in completed[: len(completed) - self.MAX_HISTORY]:
                del self._processes[k]
