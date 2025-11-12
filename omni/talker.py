
import torch
from torch import nn
from omni.utils import RMSNorm
from omni.thinker import Attention, MLP, Block
from einops import rearrange

class TalkerTiny(nn.Module):
    """ 
    AR Transformer that predicts 2 RVQ codebooks per frame (MTP=2).
    Optimized with GQA, SwiGLU, and KV caching support for faster inference.
    """
    def __init__(self, d=384, n_layers=8, n_heads=6, ff=1536, codebooks=2, codebook_size=128, 
                 dropout=0.1, use_gqa=False, use_swiglu=True, rope_theta=10000.0):
        super().__init__()
        self.emb = nn.Embedding(codebook_size, d)
        self.start = nn.Parameter(torch.zeros(1,1,d))
        self.d = d
        self.n_heads = n_heads
        self.n_layers = n_layers
        
        # Use optimized blocks (GQA + SwiGLU support)
        self.blocks = nn.ModuleList([
            Block(d, n_heads, ff, rope_theta, dropout, use_gqa=use_gqa, use_swiglu=use_swiglu)
            for _ in range(n_layers)
        ])
        self.norm = RMSNorm(d)
        self.base_head = nn.Linear(d, codebook_size)
        self.res_head  = nn.Linear(d, codebook_size)
        
        # KV cache for autoregressive generation
        self.kv_cache = None
        self.use_kv_cache = False

    def reset_kv_cache(self):
        """Reset KV cache (call before new generation)"""
        self.kv_cache = None

    def enable_kv_cache(self, enable=True):
        """Enable/disable KV caching for faster autoregressive generation"""
        self.use_kv_cache = enable
        if not enable:
            self.kv_cache = None

    def forward(self, prev_codes, use_cache=False):
        """
        Forward pass with optional KV caching for faster autoregressive generation.
        
        Args:
            prev_codes: (B, T, 2) previous frame codes (base,residual)
            use_cache: If True, use KV caching (only compute attention for new tokens)
        
        Returns:
            logits: (base_logits, res_logits) both (B, T, codebook_size)
        """
        B, T, _ = prev_codes.shape
        x = self.emb(prev_codes[:,:,0]) + self.emb(prev_codes[:,:,1])
        start = self.start.expand(B, -1, -1)
        x = torch.cat([start, x], dim=1)  # (B, T+1, d)
        
        # Handle KV caching
        use_kv = use_cache and self.use_kv_cache
        
        if use_kv:
            # Initialize cache if first call
            if self.kv_cache is None:
                self.kv_cache = [None] * len(self.blocks)
            
            # Process through blocks with caching
            for i, blk in enumerate(self.blocks):
                x, self.kv_cache[i] = blk(x, mask=None, pos=None, cache=self.kv_cache[i])
        else:
            # Standard forward pass
            for blk in self.blocks:
                x, _ = blk(x, mask=None, pos=None, cache=None)
        
        x = self.norm(x)
        logits = (self.base_head(x[:,1:,:]), self.res_head(x[:,1:,:]))
        return logits
