import json
import os
import random
from typing import Iterable, Sequence

import numpy as np
import torch

from omni.checkpoint_utils import find_checkpoint, strip_orig_mod


def resolve_device(requested: str | None = None) -> str:
    if requested:
        return requested
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_checkpoint_and_config(
    checkpoint_dir: str,
    model_file: str,
    step_prefix: str,
    device: str = "cuda",
    config_path: str | None = None,
):
    """Shared checkpoint+config resolution for test scripts."""
    checkpoint_path, checkpoint = find_checkpoint(checkpoint_dir, model_file, step_prefix, device)
    if checkpoint is None:
        raise FileNotFoundError(f"Checkpoint not found in: {checkpoint_dir}")

    print(f"Loading checkpoint from: {checkpoint_path}")

    if config_path and os.path.exists(config_path):
        print(f"Loading config from: {config_path}")
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
    elif isinstance(checkpoint, dict) and "config" in checkpoint:
        cfg = checkpoint["config"]
    else:
        inferred = os.path.join(checkpoint_dir, "config.json")
        if not os.path.exists(inferred):
            raise FileNotFoundError(f"Config not found: {inferred}. Re-run training to generate it.")
        print(f"Loading config from: {inferred}")
        with open(inferred, "r", encoding="utf-8") as f:
            cfg = json.load(f)

    return checkpoint_path, checkpoint, cfg


def select_state_dict(checkpoint, keys: Sequence[str]):
    if isinstance(checkpoint, dict):
        for key in keys:
            if key in checkpoint:
                return strip_orig_mod(checkpoint[key])
    return strip_orig_mod(checkpoint)


def sample_lines(path: str, n: int, min_len: int = 1, seed: int = 42) -> list[str]:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = [line.strip() for line in f if len(line.strip()) >= min_len]
    if not lines:
        return []
    rng = random.Random(seed)
    return rng.sample(lines, min(n, len(lines)))


def safe_perplexity(avg_loss: float, max_exp_input: float = 20.0) -> float:
    if avg_loss > max_exp_input:
        return float("inf")
    return float(np.exp(avg_loss))


def count_topk_from_logits(logits: torch.Tensor, target: torch.Tensor, ks: Iterable[int] = (1, 5, 10)) -> dict[int, int]:
    """
    logits: (T, V)
    target: (T,)
    """
    max_k = max(ks)
    topk = logits.topk(max_k, dim=-1).indices  # (T, max_k)
    out: dict[int, int] = {}
    for k in ks:
        out[k] = int((topk[:, :k] == target.unsqueeze(-1)).any(dim=-1).sum().item())
    return out
