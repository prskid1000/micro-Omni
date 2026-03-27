import os
import sys
from datetime import datetime
from typing import Optional, TextIO

import torch
import torchaudio

_LOG_TEE_STATE = {
    "enabled": False,
    "path": None,
    "file": None,
    "stdout": None,
    "stderr": None,
}


class _TeeTextIO:
    def __init__(self, a: TextIO, b: TextIO) -> None:
        self._a = a
        self._b = b

    def write(self, s: str) -> int:
        na = self._a.write(s)
        self._a.flush()
        self._b.write(s)
        self._b.flush()
        return na

    def flush(self) -> None:
        self._a.flush()
        self._b.flush()

    def isatty(self) -> bool:
        try:
            return bool(getattr(self._a, "isatty", lambda: False)())
        except Exception:
            return False

    @property
    def encoding(self) -> str:
        return getattr(self._a, "encoding", "utf-8")

    def fileno(self) -> int:
        return self._a.fileno()


def enable_log_file(log_file: Optional[str], *, header: Optional[str] = None) -> Optional[str]:
    if not log_file:
        return None
    if _LOG_TEE_STATE["enabled"]:
        return _LOG_TEE_STATE["path"]

    log_path = os.path.normpath(log_file)
    log_dir = os.path.dirname(log_path) or "."
    os.makedirs(log_dir, exist_ok=True)

    f = open(log_path, "a", encoding="utf-8", buffering=1)
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if header is None:
        header = f"--- log start {ts} ---"
    f.write(header + "\n")

    _LOG_TEE_STATE["enabled"] = True
    _LOG_TEE_STATE["path"] = log_path
    _LOG_TEE_STATE["file"] = f
    _LOG_TEE_STATE["stdout"] = sys.stdout
    _LOG_TEE_STATE["stderr"] = sys.stderr

    sys.stdout = _TeeTextIO(sys.stdout, f)  # type: ignore[assignment]
    sys.stderr = _TeeTextIO(sys.stderr, f)  # type: ignore[assignment]
    return log_path


def default_log_path(script_name: str, *, logs_dir: str = "logs") -> str:
    base = os.path.basename(script_name)
    base = base[:-3] if base.endswith(".py") else base
    return os.path.join(logs_dir, f"{base}.log")


def load_audio(path, target_sr=16000):
    audio, sr, e1 = None, None, None
    try:
        audio, sr = torchaudio.load(path)
        if audio.ndim > 1:
            audio = audio.mean(dim=0, keepdim=True)
    except Exception as _e1:
        e1 = _e1
        try:
            import soundfile as sf
            audio_np, sr = sf.read(path)
            if audio_np.ndim == 1:
                audio = torch.from_numpy(audio_np).float().unsqueeze(0)
            else:
                audio = torch.from_numpy(audio_np.T).float().mean(dim=0, keepdim=True)
        except Exception as e2:
            raise RuntimeError(f"Failed to load audio: {path}. Error: {e1} | {e2}")

    if sr != target_sr and audio is not None:
        audio = torchaudio.functional.resample(audio, sr, target_sr)
        sr = target_sr
    return audio, sr


__all__ = ["enable_log_file", "default_log_path", "load_audio"]
