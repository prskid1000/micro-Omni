
import torch
from torch import nn
from typing import Optional, Tuple, Dict
from omni.utils import RMSNorm, make_positions
from omni.thinker import Attention, MLP, Block
from einops import rearrange
import warnings

class TalkerTiny(nn.Module):
    """ 
    AR Transformer that predicts 2 RVQ codebooks per frame (MTP=2).
    Optimized with GQA, SwiGLU, Flash Attention, KV caching, and torch.compile() support.
    """
    def __init__(self, d: int = 384, n_layers: int = 8, n_heads: int = 6, ff: int = 1536, 
                 codebooks: int = 2, codebook_size: int = 128, dropout: float = 0.1, 
                 use_gqa: bool = False, use_swiglu: bool = True, rope_theta: float = 10000.0,
                 use_flash: bool = True, use_spiking: bool = False, use_ltc: bool = False,
                 compile_model: bool = False) -> None:
        """
        Initialize TalkerTiny with performance optimizations.
        
        Args:
            d: model dimension
            n_layers: number of transformer layers
            n_heads: number of attention heads
            ff: feedforward dimension
            codebooks: number of RVQ codebooks (default: 2)
            codebook_size: size of each codebook (default: 128)
            dropout: dropout rate
            use_gqa: use Grouped Query Attention
            use_swiglu: use SwiGLU activation
            rope_theta: RoPE theta parameter
            use_flash: use Flash Attention for 2-4x speedup (default: True)
            use_spiking: use SpikingAttention (Arthemis) (default: False)
            use_ltc: use Liquid Time Constants in MLP (Arthemis) (default: False)
            compile_model: use torch.compile() for 30-50% speedup (default: False)
        """
        super().__init__()
        
        # Structural check
        if d % n_heads != 0:
            raise ValueError(f"Model dimension d ({d}) must be divisible by number of heads ({n_heads}).")
            
        self.emb = nn.Embedding(codebook_size, d)
        self.start = nn.Parameter(torch.randn(1,1,d) * 0.02)
        self.d = d
        self.n_heads = n_heads
        self.n_layers = n_layers
        
        # Use optimized blocks (GQA + SwiGLU + Flash Attention support)
        self.blocks = nn.ModuleList([
            Block(d, n_heads, ff, rope_theta, dropout, use_gqa=use_gqa, use_swiglu=use_swiglu, 
                  use_flash=use_flash, use_spiking=use_spiking, use_ltc=use_ltc)
            for _ in range(n_layers)
        ])
        self.norm = RMSNorm(d)
        self.base_head = nn.Linear(d, codebook_size)
        self.res_head  = nn.Linear(d, codebook_size)

        # Pre-allocated causal mask (sliced at runtime — no allocation per forward)
        self.register_buffer("_causal_mask", torch.tril(torch.ones(2048, 2048)).unsqueeze(0).unsqueeze(0), persistent=False)

        # KV cache for autoregressive generation
        self.kv_cache = None
        self.use_kv_cache = False
        
        # Compilation support for additional speedup
        self._compiled = False
        if compile_model:
            self._apply_compilation()
    
    def _apply_compilation(self) -> None:
        """Apply torch.compile() for 30-50% speedup. Requires PyTorch 2.0+."""
        if not hasattr(torch, 'compile'):
            warnings.warn("torch.compile() not available. Requires PyTorch 2.0+. Skipping compilation.")
            return
        
        try:
            # Compile individual blocks
            # Using 'inductor' backend for nvFuser optimizations
            # Provides 10-20% speedup without requiring Triton compilation
            for i, block in enumerate(self.blocks):
                self.blocks[i] = torch.compile(block, backend='inductor', mode='default')
            
            # Compile heads
            self.base_head = torch.compile(self.base_head, backend='inductor', mode='default')
            self.res_head = torch.compile(self.res_head, backend='inductor', mode='default')
            
            self._compiled = True
            print(f"✓ TalkerTiny compiled successfully with torch.compile()")
        except Exception as e:
            warnings.warn(f"Failed to compile TalkerTiny: {e}. Continuing without compilation.")

    def reset_kv_cache(self) -> None:
        """Reset KV cache (call before new generation)"""
        self.kv_cache = None

    def enable_kv_cache(self, enable: bool = True) -> None:
        """Enable/disable KV caching for faster autoregressive generation"""
        self.use_kv_cache = enable
        if not enable:
            self.kv_cache = None

    def forward(self, prev_codes: torch.Tensor, use_cache: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with optional KV caching for faster autoregressive generation.
        
        Args:
            prev_codes: (B, T, 2) previous frame codes (base,residual)
            use_cache: If True, use KV caching (only compute attention for new tokens)
        
        Returns:
            logits: (base_logits, res_logits) both (B, T, codebook_size)
        """
        B, T, _ = prev_codes.shape
        token_emb = self.emb(prev_codes[:, :, 0]) + self.emb(prev_codes[:, :, 1])
        
        use_kv = use_cache and self.use_kv_cache
        if use_kv and self.kv_cache is None:
            self.kv_cache = [None] * len(self.blocks)
        
        cache_ready = use_kv and self.kv_cache[0] is not None
        if use_kv and cache_ready and T > 1:
            # New sequence without reset; start over
            self.reset_kv_cache()
            self.kv_cache = [None] * len(self.blocks)
            cache_ready = False
        
        start = self.start.expand(B, -1, -1)
        if not use_kv or not cache_ready:
            x = torch.cat([start, token_emb], dim=1)
            offset = 1
        else:
            x = token_emb
            offset = 0
        
        seq_len = x.shape[1]
        
        if use_kv:
            if cache_ready:
                pos = torch.tensor([self.kv_cache[0]['pos']], device=x.device, dtype=torch.long)
                for i, blk in enumerate(self.blocks):
                    x, new_cache = blk(x, mask=None, pos=pos, cache=self.kv_cache[i], return_cache=True)
                    self.kv_cache[i] = new_cache
            else:
                pos = make_positions(seq_len, x.device)
                mask = self._causal_mask[:, :, :seq_len, :seq_len]
                for i, blk in enumerate(self.blocks):
                    x, new_cache = blk(x, mask=mask, pos=pos, cache=None, return_cache=True)
                    self.kv_cache[i] = new_cache
        else:
            pos = make_positions(seq_len, x.device)
            mask = self._causal_mask[:, :, :seq_len, :seq_len]
            for blk in self.blocks:
                x, _ = blk(x, mask=mask, pos=pos, cache=None, return_cache=False)
        
        x = self.norm(x)
        x = x[:, offset:, :]
        base_logits = self.base_head(x)
        res_logits = self.res_head(x)
        
        return (base_logits, res_logits)
