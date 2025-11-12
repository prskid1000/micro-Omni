
import torch
from torch import nn
from omni.utils import RMSNorm

class ConvDown(nn.Module):
    def __init__(self, in_ch=1, mid=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, mid, 3, stride=2, padding=1), nn.GELU(),
            nn.Conv2d(mid, mid, 3, stride=2, padding=1), nn.GELU()
        )
    def forward(self, x):
        return self.net(x)

class EncoderBlock(nn.Module):
    def __init__(self, d, heads, ff, dropout=0.1):
        super().__init__()
        self.norm1 = RMSNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)
        self.norm2 = RMSNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, ff), nn.GELU(), nn.Linear(ff, d))
        self.drop = nn.Dropout(dropout)
    def forward(self, x):
        h = self.attn(self.norm1(x), self.norm1(x), self.norm1(x), need_weights=False)[0]
        x = x + self.drop(h)
        x = x + self.drop(self.mlp(self.norm2(x)))
        return x

class AudioEncoderTiny(nn.Module):
    """ AuT-Tiny: mel -> conv2d 4x downsample -> Transformer encoder (d=384, L=8) -> frame seq """
    def __init__(self, d=384, heads=6, ff=1536, layers=8, dropout=0.1):
        super().__init__()
        self.down = ConvDown(1, mid=64)
        self.proj = nn.Linear(64* (128//4), d)  # 128 mel bins -> /4 along freq -> 32*64=2048 -> project to d
        self.blocks = nn.ModuleList([EncoderBlock(d, heads, ff, dropout) for _ in range(layers)])
        self.norm = RMSNorm(d)
    def forward(self, mel):  # mel: (B, T, 128)
        x = mel[:, None, :, :]  # (B,1,T,128)
        x = self.down(x)  # (B,64,T/4,128/4)
        B,C,T,F = x.shape
        x = x.permute(0,2,1,3).contiguous().view(B,T, C*F)  # (B, T/4, C*F)
        x = self.proj(x)  # (B, T/4, d)
        for blk in self.blocks:
            x = blk(x)
        return self.norm(x)  # (B, T/4, d)
