
import torch
from torch import nn
from typing import Optional
from omni.utils import RMSNorm
import warnings

# Check for Flash Attention support (PyTorch 2.0+)
HAS_FLASH_ATTENTION = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
if HAS_FLASH_ATTENTION:
    from torch.nn.functional import scaled_dot_product_attention

class ConvDown(nn.Module):
    def __init__(self, in_ch: int = 1, mid: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, mid, 3, stride=2, padding=1), nn.GELU(),
            nn.Conv2d(mid, mid, 3, stride=2, padding=1), nn.GELU()
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class EncoderBlock(nn.Module):
    def __init__(self, d: int, heads: int, ff: int, dropout: float = 0.1, use_flash: bool = True) -> None:
        super().__init__()
        self.d = d
        self.heads = heads
        self.head_dim = d // heads
        self.use_flash = use_flash and HAS_FLASH_ATTENTION
        
        if use_flash and not HAS_FLASH_ATTENTION:
            warnings.warn("Flash Attention requested but not available. Falling back to standard attention.")
            self.use_flash = False
        
        self.norm1 = RMSNorm(d)
        
        # Custom attention for Flash Attention support
        self.qkv_proj = nn.Linear(d, 3 * d, bias=True)
        self.out_proj = nn.Linear(d, d, bias=True)
        
        self.norm2 = RMSNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, ff), nn.GELU(), nn.Linear(ff, d))
        self.drop = nn.Dropout(dropout)
        self.dropout_p = dropout
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-attention with Flash Attention support
        normed = self.norm1(x)
        B, T, D = normed.shape
        
        # QKV projection
        qkv = self.qkv_proj(normed).reshape(B, T, 3, self.heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, heads, T, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Compute attention
        if self.use_flash:
            # Use Flash Attention for 2-4x speedup
            attn_out = scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.dropout_p if self.training else 0.0,
                is_causal=False
            )
        else:
            # Standard attention
            scale = self.head_dim ** -0.5
            attn = (q @ k.transpose(-2, -1)) * scale
            attn = attn.softmax(dim=-1)
            attn = self.drop(attn)
            attn_out = attn @ v
        
        # Reshape and project output
        attn_out = attn_out.transpose(1, 2).reshape(B, T, D)
        h = self.out_proj(attn_out)
        
        # Residual connection
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
    
    Optimized with Flash Attention and torch.compile() support for improved performance.
    """
    def __init__(self, d: int = 192, heads: int = 3, ff: int = 768, layers: int = 4, 
                 dropout: float = 0.1, downsample_factor: int = 4, use_flash: bool = True,
                 compile_model: bool = False) -> None:
        """
        Initialize AudioEncoderTiny with performance optimizations.
        
        Args:
            d: model dimension
            heads: number of attention heads
            ff: feedforward dimension
            layers: number of transformer layers
            dropout: dropout rate
            downsample_factor: temporal downsample factor (4 or 8)
            use_flash: use Flash Attention for 2-4x speedup (default: True)
            compile_model: use torch.compile() for 30-50% speedup (default: False)
        """
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
        self.blocks = nn.ModuleList([EncoderBlock(d, heads, ff, dropout, use_flash=use_flash) for _ in range(layers)])
        self.norm = RMSNorm(d)
        
        # Compilation support
        self._compiled = False
        if compile_model:
            self._apply_compilation()
    
    def _apply_compilation(self) -> None:
        """Apply torch.compile() for 30-50% speedup. Requires PyTorch 2.0+."""
        if not hasattr(torch, 'compile'):
            warnings.warn("torch.compile() not available. Requires PyTorch 2.0+. Skipping compilation.")
            return
        
        try:
            # Compile blocks
            # Using 'cudagraphs' backend to avoid Triton/LLVM compatibility issues
            # Provides 10-20% speedup without requiring Triton compilation
            for i, block in enumerate(self.blocks):
                self.blocks[i] = torch.compile(block, backend='cudagraphs', mode='default')
            
            # Compile conv and projection
            self.down = torch.compile(self.down, backend='cudagraphs', mode='default')
            self.proj = torch.compile(self.proj, backend='cudagraphs', mode='default')
            
            self._compiled = True
            print(f"✓ AudioEncoderTiny compiled successfully with torch.compile()")
        except Exception as e:
            warnings.warn(f"Failed to compile AudioEncoderTiny: {e}. Continuing without compilation.")
    
    def forward(self, mel: torch.Tensor) -> torch.Tensor:  # mel: (B, T, 128)
        x = mel[:, None, :, :]  # (B,1,T,128)
        x = self.down(x)  # (B,64,T/downsample_factor,128/downsample_factor)
        B,C,T,F = x.shape
        x = x.permute(0,2,1,3).contiguous().view(B,T, C*F)  # (B, T/downsample_factor, C*F)
        x = self.proj(x)  # (B, T/downsample_factor, d)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)  # (B, T/downsample_factor, d)
        
        # Check for numerical stability (NaN/Inf detection)
        if torch.isnan(x).any() or torch.isinf(x).any():
            nan_count = torch.isnan(x).sum().item()
            inf_count = torch.isinf(x).sum().item()
            raise RuntimeError(f"Numerical instability in AudioEncoderTiny forward pass: NaN={nan_count}, Inf={inf_count}")
        
        return x
