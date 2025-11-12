
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
    """ 
    AuT-Tiny: mel -> conv2d downsample -> Transformer encoder -> frame seq
    
    Frame rate calculation:
    - Input: mel at sample_rate/hop_length Hz (e.g., 16000/160 = 100 Hz)
    - ConvDown: 2x stride twice = 4x downsample in time
    - Output: 100/4 = 25 Hz (or 100/8 = 12.5 Hz if 8x downsample)
    """
    def __init__(self, d=192, heads=3, ff=768, layers=4, dropout=0.1, downsample_factor=4):
        super().__init__()
        self.downsample_factor = downsample_factor
        # ConvDown does 2x stride twice = 4x total, or we can add more
        if downsample_factor == 8:
            # 8x downsample: add extra 2x conv
            self.down = nn.Sequential(
                ConvDown(1, mid=64),
                nn.Conv2d(64, 64, 3, stride=2, padding=1), nn.GELU()  # Extra 2x
            )
        else:
            # 4x downsample (default)
            self.down = ConvDown(1, mid=64)
        
        # Projection: 64 channels * (128 mel bins / freq_downsample) -> d
        freq_downsample = downsample_factor  # Same as time downsample
        self.proj = nn.Linear(64 * (128 // freq_downsample), d)
        self.blocks = nn.ModuleList([EncoderBlock(d, heads, ff, dropout) for _ in range(layers)])
        self.norm = RMSNorm(d)
    
    def forward(self, mel):  # mel: (B, T, 128)
        x = mel[:, None, :, :]  # (B,1,T,128)
        x = self.down(x)  # (B,64,T/downsample_factor,128/downsample_factor)
        B,C,T,F = x.shape
        x = x.permute(0,2,1,3).contiguous().view(B,T, C*F)  # (B, T/downsample_factor, C*F)
        x = self.proj(x)  # (B, T/downsample_factor, d)
        for blk in self.blocks:
            x = blk(x)
        return self.norm(x)  # (B, T/downsample_factor, d)
