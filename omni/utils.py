
import math, torch
from torch import nn
from einops import rearrange

def rms_norm(x, weight, eps=1e-5):
    return (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)) * weight

class RMSNorm(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
    def forward(self, x):
        return rms_norm(x, self.weight)

class RoPE(nn.Module):
    """
    Rotary positional embeddings (text). TM-RoPE-lite for multimodal: we simply continue positions
    across modalities and allow 2D factorization for vision/audio if desired (kept simple here).
    """
    def __init__(self, d_head, theta=10000.0):
        super().__init__()
        self.d = d_head
        self.theta = theta

    def forward(self, q, k, pos):
        # q,k: (B, H, T, D), pos: (T,)
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

def rotate_half(x):
    x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
    return torch.cat([-x2, x1], dim=-1)

def make_positions(T, device):
    return torch.arange(T, device=device).long()
