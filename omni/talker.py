
import torch
from torch import nn
from omni.utils import RMSNorm, make_positions
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
                mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device)).unsqueeze(0).unsqueeze(0)
                for i, blk in enumerate(self.blocks):
                    x, new_cache = blk(x, mask=mask, pos=pos, cache=None, return_cache=True)
                    self.kv_cache[i] = new_cache
        else:
            pos = make_positions(seq_len, x.device)
            mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device)).unsqueeze(0).unsqueeze(0)
            for blk in self.blocks:
                x, _ = blk(x, mask=mask, pos=pos, cache=None, return_cache=False)
        
        x = self.norm(x)
        x = x[:, offset:, :]
        base_logits = self.base_head(x)
        res_logits = self.res_head(x)
        
        # Check for numerical stability (NaN/Inf detection)
        if torch.isnan(base_logits).any() or torch.isinf(base_logits).any():
            nan_count = torch.isnan(base_logits).sum().item()
            inf_count = torch.isinf(base_logits).sum().item()
            raise RuntimeError(f"Numerical instability in TalkerTiny base_head: NaN={nan_count}, Inf={inf_count}")
        
        if torch.isnan(res_logits).any() or torch.isinf(res_logits).any():
            nan_count = torch.isnan(res_logits).sum().item()
            inf_count = torch.isinf(res_logits).sum().item()
            raise RuntimeError(f"Numerical instability in TalkerTiny res_head: NaN={nan_count}, Inf={inf_count}")
        
        return (base_logits, res_logits)
