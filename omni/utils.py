
import math
import torch
from torch import nn
from typing import Tuple
from einops import rearrange

def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """Apply RMS normalization to input tensor."""
    return (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)) * weight

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""
    def __init__(self, d: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return rms_norm(x, self.weight)

class RoPE(nn.Module):
    """
    Rotary Positional Embeddings (RoPE).
    TM-RoPE-lite for multimodal: we simply continue positions
    across modalities and allow 2D factorization for vision/audio if desired (kept simple here).
    """
    def __init__(self, d_head: int, theta: float = 10000.0) -> None:
        super().__init__()
        self.d = d_head
        self.theta = theta

    def forward(self, q: torch.Tensor, k: torch.Tensor, pos: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply rotary positional embeddings to queries and keys.
        
        Args:
            q: (B, H, T, D) query tensor
            k: (B, H, T, D) key tensor
            pos: (T,) position indices
        
        Returns:
            Tuple of (q_with_rope, k_with_rope)
        """
        device = q.device
        d = self.d
        T = pos.shape[0]
        inv = 1.0 / (self.theta ** (torch.arange(0, d, 2, device=device).float() / d))
        freqs = torch.einsum('t,f->tf', pos.float(), inv)  # (T, d/2)
        emb = torch.cat([freqs, freqs], dim=-1)  # (T, d)
        cos = emb.cos()[None, None, :, :]
        sin = emb.sin()[None, None, :, :]
        q1 = (q * cos) + (rotate_half(q) * sin)
        k1 = (k * cos) + (rotate_half(k) * sin)
        return q1, k1

def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Helper function for RoPE: rotate half the hidden dimensions."""
    x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
    return torch.cat([-x2, x1], dim=-1)

def make_positions(T: int, device: torch.device) -> torch.Tensor:
    """Create position indices from 0 to T-1."""
    return torch.arange(T, device=device).long()
