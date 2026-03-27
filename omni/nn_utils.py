import math
from typing import Optional, Tuple

import torch
from torch import nn


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """Apply RMS normalization to input tensor."""
    x_fp32 = x.float()
    rrms = torch.rsqrt(x_fp32.pow(2).mean(-1, keepdim=True) + eps)
    return (x_fp32 * rrms).to(dtype=x.dtype) * weight


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, d: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return rms_norm(x, self.weight)


class RoPE(nn.Module):
    """
    Rotary Positional Embeddings (RoPE) with cached cos/sin tables.
    TM-RoPE-lite for multimodal: we simply continue positions across modalities.
    """

    def __init__(
        self,
        d_head: int,
        theta: float = 10000.0,
        scaling_factor: float = 1.0,
        yarn_beta_fast: float = 32.0,
        yarn_beta_slow: float = 1.0,
    ) -> None:
        super().__init__()
        if d_head % 2 != 0:
            raise ValueError(f"RoPE requires even head dimension, got {d_head}")
        self.d = d_head
        self.theta = theta
        self.scaling_factor = scaling_factor
        if scaling_factor > 1.0:
            freq = 1.0 / (theta ** (torch.arange(0, d_head, 2).float() / d_head))
            low = max(math.floor(yarn_beta_fast * d_head / (2 * math.pi * yarn_beta_slow)), 1)
            high = min(math.ceil(yarn_beta_slow * d_head / (2 * math.pi * yarn_beta_fast)), d_head // 2 - 1)
            dims = torch.arange(0, d_head // 2).float()
            ramp = (dims - low) / max(high - low, 1)
            ramp = ramp.clamp(0, 1)
            inv_freq_interpolated = freq / scaling_factor
            inv_freq = inv_freq_interpolated * (1 - ramp) + freq * ramp
            self.mscale = 0.1 * math.log(scaling_factor) + 1.0
        else:
            inv_freq = 1.0 / (theta ** (torch.arange(0, d_head, 2).float() / d_head))
            self.mscale = 1.0
        self.register_buffer("inv_freq", inv_freq)
        self._cos_cache: Optional[torch.Tensor] = None
        self._sin_cache: Optional[torch.Tensor] = None
        self._cache_len: int = 0

    def _build_cache(self, T: int, device: torch.device) -> None:
        if T <= self._cache_len and self._cos_cache is not None and self._cos_cache.device == device:
            return
        pos = torch.arange(T, device=device).float()
        freqs = torch.einsum("t,f->tf", pos, self.inv_freq.to(device))
        emb = torch.cat([freqs, freqs], dim=-1)
        self._cos_cache = emb.cos()[None, None, :, :]
        self._sin_cache = emb.sin()[None, None, :, :]
        self._cache_len = T

    def forward(self, q: torch.Tensor, k: torch.Tensor, pos: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        T = pos.shape[0]
        self._build_cache(T, q.device)
        cos = self._cos_cache[:, :, :T, :]
        sin = self._sin_cache[:, :, :T, :]
        q1 = (q * cos) + (rotate_half(q) * sin)
        k1 = (k * cos) + (rotate_half(k) * sin)
        if self.mscale != 1.0:
            q1 = q1 * self.mscale
            k1 = k1 * self.mscale
        return q1, k1


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Helper function for RoPE: rotate half the hidden dimensions."""
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat([-x2, x1], dim=-1)


class LearnableTemperature(nn.Module):
    """Learnable temperature parameter for contrastive learning (CLIP-style)."""

    def __init__(self, init_value: float = 0.07):
        super().__init__()
        self.logit_scale = nn.Parameter(torch.ones([]) * torch.log(torch.tensor(1.0 / init_value)))

    def forward(self):
        return torch.clamp(self.logit_scale.exp(), min=0.01, max=100.0)


def make_positions(T: int, device: torch.device) -> torch.Tensor:
    """Create position indices from 0 to T-1."""
    return torch.arange(T, device=device).long()


class ProjectionHead(nn.Module):
    """MLP projection head for contrastive learning (CLIP-style)."""

    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )
        self.ln = nn.LayerNorm(output_dim)

    def forward(self, x):
        return self.ln(self.net(x))

__all__ = [
    "rms_norm",
    "RMSNorm",
    "RoPE",
    "rotate_half",
    "LearnableTemperature",
    "make_positions",
    "ProjectionHead",
]
