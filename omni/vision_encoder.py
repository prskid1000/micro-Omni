
import torch
from torch import nn
from omni.utils import RMSNorm
from einops import rearrange

class ViTTiny(nn.Module):
    def __init__(self, img_size=224, patch=16, d=192, layers=12, heads=3, ff=768, dropout=0.1):
        super().__init__()
        self.patch = patch
        self.d = d
        self.proj = nn.Conv2d(3, d, kernel_size=patch, stride=patch)
        num_patches = (img_size//patch) * (img_size//patch)
        self.cls = nn.Parameter(torch.zeros(1,1,d))
        self.pos = nn.Parameter(torch.zeros(1, 1+num_patches, d))
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(d, heads, ff, dropout, batch_first=True, norm_first=True, activation='gelu')
            for _ in range(layers)
        ])
        self.norm = RMSNorm(d)

    def forward(self, x):  # x: (B,3,H,W)
        x = self.proj(x)  # (B,d,H',W')
        x = rearrange(x, "b d h w -> b (h w) d")
        B, N, D = x.shape
        cls = self.cls.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.pos[:, :x.size(1), :]
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        cls = x[:, :1, :]  # (B,1,d)
        grid = x[:, 1:, :] # (B,N,d)
        
        # Check for numerical stability (NaN/Inf detection)
        if torch.isnan(cls).any() or torch.isinf(cls).any():
            nan_count = torch.isnan(cls).sum().item()
            inf_count = torch.isinf(cls).sum().item()
            raise RuntimeError(f"Numerical instability in ViTTiny CLS token: NaN={nan_count}, Inf={inf_count}")
        
        if torch.isnan(grid).any() or torch.isinf(grid).any():
            nan_count = torch.isnan(grid).sum().item()
            inf_count = torch.isinf(grid).sum().item()
            raise RuntimeError(f"Numerical instability in ViTTiny grid tokens: NaN={nan_count}, Inf={inf_count}")
        
        return cls, grid
