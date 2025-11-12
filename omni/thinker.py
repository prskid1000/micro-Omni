
import math, torch
from torch import nn
from omni.utils import RMSNorm, RoPE, make_positions
from einops import rearrange

class SwiGLU(nn.Module):
    """Swish Gated Linear Unit activation (SwiGLU) as used in Qwen3 Omni"""
    def __init__(self, dim):
        super().__init__()
        self.gate_proj = nn.Linear(dim, dim, bias=False)
        self.up_proj = nn.Linear(dim, dim, bias=False)
        self.down_proj = nn.Linear(dim, dim, bias=False)
    
    def forward(self, x):
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        # Swish activation: x * sigmoid(x)
        swish = gate * torch.sigmoid(gate)
        return self.down_proj(swish * up)

class MLP(nn.Module):
    """
    Multi-Layer Perceptron with optional SwiGLU activation.
    
    SwiGLU (Swish-Gated Linear Unit) uses three projections:
    - gate_proj: projects input to gate dimension
    - up_proj: projects input to up dimension  
    - down_proj: projects combined gate*up back to model dimension
    
    Formula: output = down_proj(swish(gate_proj(x)) * up_proj(x))
    where swish(x) = x * sigmoid(x)
    
    Args:
        d: model dimension (input/output size)
        ff: feedforward dimension (gate/up projection size)
        use_swiglu: if True, use SwiGLU; if False, use standard GELU MLP
    """
    def __init__(self, d, ff, use_swiglu=True):
        super().__init__()
        if use_swiglu:
            # SwiGLU implementation: gate + up projections, then down projection
            # Note: Standard SwiGLU uses 2/3 * ff for gate/up, but we use full ff
            # for each to match the specified feedforward dimension
            self.gate_proj = nn.Linear(d, ff, bias=False)
            self.up_proj = nn.Linear(d, ff, bias=False)
            self.down_proj = nn.Linear(ff, d, bias=False)
        else:
            # Standard MLP: two linear layers with GELU activation
            self.fc1 = nn.Linear(d, ff)
            self.act = nn.GELU()
            self.fc2 = nn.Linear(ff, d)
        self.use_swiglu = use_swiglu
    
    def forward(self, x):
        if self.use_swiglu:
            # SwiGLU: gate(x) * swish(gate(x)) * up(x)
            gate = self.gate_proj(x)
            up = self.up_proj(x)
            swish = gate * torch.sigmoid(gate)  # Swish activation
            return self.down_proj(swish * up)
        else:
            return self.fc2(self.act(self.fc1(x)))

class Attention(nn.Module):
    def __init__(self, d, heads, rope_theta=10000.0, dropout=0.0, use_gqa=False, kv_groups=None):
        """
        Attention with optional GQA (Grouped Query Attention).
        
        Args:
            d: model dimension
            heads: number of query heads
            rope_theta: RoPE theta parameter
            dropout: dropout rate
            use_gqa: whether to use GQA
            kv_groups: number of key/value groups (if None and use_gqa=True, uses heads//2)
        """
        super().__init__()
        self.h = heads  # query heads
        self.d = d
        self.dk = d // heads
        self.use_gqa = use_gqa
        
        if use_gqa:
            # GQA: multiple query heads share key/value heads
            self.kv_groups = kv_groups if kv_groups is not None else max(1, heads // 2)
            self.q = nn.Linear(d, heads * self.dk, bias=False)
            self.k = nn.Linear(d, self.kv_groups * self.dk, bias=False)
            self.v = nn.Linear(d, self.kv_groups * self.dk, bias=False)
            # RoPE for queries and keys (both use dk)
            self.rope_q = RoPE(self.dk, theta=rope_theta)
            self.rope_k = RoPE(self.dk, theta=rope_theta)
        else:
            # Standard MHA
            self.qkv = nn.Linear(d, 3*d, bias=False)
            self.rope = RoPE(self.dk, theta=rope_theta)
        
        self.o = nn.Linear(d, d, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, mask=None, pos=None, cache=None):
        """
        Forward pass with optional KV caching.
        
        Args:
            x: (B, T, D) input embeddings
            mask: (B, T, T) attention mask
            pos: (T,) position indices
            cache: Optional dict with 'k' and 'v' keys containing cached K/V from previous forward pass
                   Each should be (B, h, T_cached, dk) or (B, g, T_cached, dk) for GQA
        
        Returns:
            y: (B, T, D) output
            cache: Updated cache dict with new K/V appended
        """
        B, T, D = x.shape
        # is_incremental means we're using caching (cache may be None on first call)
        is_incremental = cache is not None or (cache is None and T == 1)  # Single token suggests incremental
        
        if self.use_gqa:
            # GQA: separate Q, K, V projections
            q = self.q(x)  # (B, T, heads * dk)
            k = self.k(x)  # (B, T, kv_groups * dk)
            v = self.v(x)  # (B, T, kv_groups * dk)
            
            # Reshape
            q = rearrange(q, "b t (h d) -> b h t d", h=self.h)
            k = rearrange(k, "b t (g d) -> b g t d", g=self.kv_groups)
            v = rearrange(v, "b t (g d) -> b g t d", g=self.kv_groups)
            
            # Apply RoPE
            if pos is None:
                if is_incremental:
                    # For incremental, pos is just the current position
                    pos = torch.tensor([cache.get('pos', 0)], device=x.device, dtype=torch.long)
                else:
                    pos = make_positions(T, x.device)
            
            q, _ = self.rope_q(q, q, pos if is_incremental else pos)
            k, _ = self.rope_k(k, k, pos if is_incremental else pos)
            
            # Handle KV cache
            if is_incremental and cache is not None and 'k' in cache and 'v' in cache:
                # Concatenate with cached K/V
                k = torch.cat([cache['k'], k], dim=2)  # (B, g, T_cached+T, dk)
                v = torch.cat([cache['v'], v], dim=2)  # (B, g, T_cached+T, dk)
            
            # Repeat k and v to match query heads (each kv head serves multiple q heads)
            repeat_factor = self.h // self.kv_groups
            k = k.repeat_interleave(repeat_factor, dim=1)  # (B, h, T_total, dk)
            v = v.repeat_interleave(repeat_factor, dim=1)  # (B, h, T_total, dk)
            
            # Update cache (always create cache if in incremental mode)
            if is_incremental:
                # Store K/V for next iteration (before repeating)
                cache_k = k[:, ::repeat_factor, :, :]  # (B, g, T_total, dk) - get one per group
                cache_v = v[:, ::repeat_factor, :, :]  # (B, g, T_total, dk)
                cache = {'k': cache_k, 'v': cache_v, 'pos': pos[-1].item() + 1 if pos.numel() > 0 else 0}
            else:
                cache = None
        else:
            # Standard MHA
            qkv = self.qkv(x).chunk(3, dim=-1)  # 3 * (B,T,D)
            q, k, v = [rearrange(t, "b t (h d) -> b h t d", h=self.h) for t in qkv]
            
            # Apply RoPE
            if pos is None:
                if is_incremental:
                    pos = torch.tensor([cache.get('pos', 0)], device=x.device, dtype=torch.long)
                else:
                    pos = make_positions(T, x.device)
            
            q, k = self.rope(q, k, pos if is_incremental else pos)
            
            # Handle KV cache
            if is_incremental and cache is not None and 'k' in cache and 'v' in cache:
                # Concatenate with cached K/V
                k = torch.cat([cache['k'], k], dim=2)  # (B, h, T_cached+T, dk)
                v = torch.cat([cache['v'], v], dim=2)  # (B, h, T_cached+T, dk)
            
            # Update cache (always create cache if in incremental mode)
            if is_incremental:
                cache = {'k': k, 'v': v, 'pos': pos[-1].item() + 1 if pos.numel() > 0 else 0}
            else:
                cache = None
        
        # Attention computation
        T_total = k.shape[2]
        att = torch.einsum("bhtd,bhTd->bhtT", q, k) / math.sqrt(self.dk)
        
        # Create mask if needed
        if mask is None and is_incremental:
            # For incremental, only attend to all previous tokens (causal)
            mask = torch.ones(1, T_total, device=x.device).tril().unsqueeze(0).unsqueeze(0)  # (1, 1, 1, T_total)
        elif mask is not None:
            # Extend mask if needed
            # mask can be (B, T, T) or (B, 1, T, T) or (1, 1, T, T)
            if len(mask.shape) == 2:
                # (T, T) -> expand to (1, 1, T, T)
                mask = mask.unsqueeze(0).unsqueeze(0)
            elif len(mask.shape) == 3:
                # (B, T, T) -> (B, 1, T, T)
                mask = mask.unsqueeze(1)
            
            # Now mask is (B, 1, T_old, T_old) or (1, 1, T_old, T_old)
            if mask.shape[-1] < T_total:
                # Pad mask for cached tokens - pad on both dimensions
                pad_size = T_total - mask.shape[-1]
                # Pad rows: (B, 1, pad_size, T_old)
                row_pad = torch.ones(mask.shape[0], mask.shape[1], pad_size, mask.shape[-1], device=mask.device)
                mask = torch.cat([row_pad, mask], dim=2)  # (B, 1, T_total, T_old)
                # Pad columns: (B, 1, T_total, pad_size)
                col_pad = torch.ones(mask.shape[0], mask.shape[1], mask.shape[2], pad_size, device=mask.device)
                mask = torch.cat([col_pad, mask], dim=3)  # (B, 1, T_total, T_total)
        
        if mask is not None:
            # att is (B, H, T, T_total), mask should be (B, 1, T, T_total) or broadcastable
            # Ensure mask matches att shape
            if len(mask.shape) == 4:
                # mask is (B, 1, T_mask, T_mask) - need to expand to (B, 1, T, T_total)
                if mask.shape[2] != T or mask.shape[3] != T_total:
                    # Reshape mask to match attention
                    # For causal mask, we want to allow attention to all previous tokens
                    if mask.shape[2] == T and mask.shape[3] < T_total:
                        # Pad mask columns
                        pad_cols = T_total - mask.shape[3]
                        col_pad = torch.ones(mask.shape[0], mask.shape[1], mask.shape[2], pad_cols, device=mask.device)
                        mask = torch.cat([col_pad, mask], dim=3)
                    elif mask.shape[2] < T:
                        # Need to pad rows and columns
                        pad_rows = T - mask.shape[2]
                        pad_cols = T_total - mask.shape[3] if mask.shape[3] < T_total else 0
                        row_pad = torch.ones(mask.shape[0], mask.shape[1], pad_rows, mask.shape[3], device=mask.device)
                        mask = torch.cat([row_pad, mask], dim=2)
                        if pad_cols > 0:
                            col_pad = torch.ones(mask.shape[0], mask.shape[1], mask.shape[2], pad_cols, device=mask.device)
                            mask = torch.cat([col_pad, mask], dim=3)
                # Expand mask to match attention heads: (B, 1, T, T_total) -> (B, H, T, T_total)
                mask = mask.expand(B, self.h, -1, -1)
            att = att.masked_fill(mask==0, float("-inf"))
        
        att = att.softmax(dim=-1)
        att = self.drop(att)
        y = torch.einsum("bhtT,bhTd->bhtd", att, v)
        y = rearrange(y, "b h t d -> b t (h d)")
        
        return self.o(y), cache

class MoE(nn.Module):
    """Mixture of Experts layer"""
    def __init__(self, d, ff, num_experts=8, num_experts_per_tok=2, use_swiglu=True):
        super().__init__()
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.router = nn.Linear(d, num_experts, bias=False)
        self.experts = nn.ModuleList([
            MLP(d, ff, use_swiglu=use_swiglu) for _ in range(num_experts)
        ])
    
    def forward(self, x):
        """
        x: (B, T, D)
        Returns: (B, T, D)
        """
        B, T, D = x.shape
        router_logits = self.router(x)  # (B, T, num_experts)
        router_probs = torch.softmax(router_logits, dim=-1)  # (B, T, num_experts)
        
        # Top-k expert selection
        topk_probs, topk_indices = torch.topk(router_probs, self.num_experts_per_tok, dim=-1)  # (B, T, k)
        topk_probs = topk_probs / (topk_probs.sum(dim=-1, keepdim=True) + 1e-8)  # Normalize
        
        # Efficient expert computation using einsum
        # Flatten batch and time dimensions for processing
        x_flat = x.view(B * T, D)  # (B*T, D)
        topk_indices_flat = topk_indices.view(B * T, self.num_experts_per_tok)  # (B*T, k)
        topk_probs_flat = topk_probs.view(B * T, self.num_experts_per_tok)  # (B*T, k)
        
        # Process through experts (more efficient batch processing)
        output = torch.zeros_like(x_flat)  # (B*T, D)
        for k in range(self.num_experts_per_tok):
            # Get expert indices for this k
            expert_indices = topk_indices_flat[:, k]  # (B*T,)
            weights = topk_probs_flat[:, k:k+1]  # (B*T, 1)
            
            # Process each unique expert
            for expert_idx in range(self.num_experts):
                mask = (expert_indices == expert_idx)  # (B*T,)
                if mask.any():
                    expert_out = self.experts[expert_idx](x_flat[mask])  # (N, D)
                    output[mask] = output[mask] + weights[mask] * expert_out
        
        return output.view(B, T, D)

class Block(nn.Module):
    def __init__(self, d, heads, ff, rope_theta, dropout, use_gqa=False, use_swiglu=True, 
                 use_moe=False, num_experts=8, num_experts_per_tok=2):
        super().__init__()
        self.norm1 = RMSNorm(d)
        self.attn = Attention(d, heads, rope_theta=rope_theta, dropout=dropout, use_gqa=use_gqa)
        self.norm2 = RMSNorm(d)
        self.use_moe = use_moe
        if use_moe:
            self.moe = MoE(d, ff, num_experts, num_experts_per_tok, use_swiglu)
        else:
            self.mlp = MLP(d, ff, use_swiglu=use_swiglu)
        self.drop = nn.Dropout(dropout)
    
    def forward(self, x, mask=None, pos=None, cache=None):
        """
        Forward pass with optional KV caching.
        
        Args:
            x: (B, T, D) input
            mask: attention mask
            pos: position indices
            cache: Optional cache dict for this layer
        
        Returns:
            x: (B, T, D) output
            cache: Updated cache for this layer
        """
        attn_out, cache = self.attn(self.norm1(x), mask=mask, pos=pos, cache=cache)
        x = x + self.drop(attn_out)
        if self.use_moe:
            x = x + self.drop(self.moe(self.norm2(x)))
        else:
            x = x + self.drop(self.mlp(self.norm2(x)))
        return x, cache

class ThinkerLM(nn.Module):
    def __init__(self, vocab, n_layers=16, d=512, heads=8, ff=2048, dropout=0.1, rope_theta=10000, ctx=1024, 
                 use_gqa=False, use_swiglu=True, use_moe=False, num_experts=8, num_experts_per_tok=2):
        """
        ThinkerLM with optional Qwen3 Omni features.
        
        Args:
            vocab: vocabulary size
            n_layers: number of transformer layers
            d: model dimension
            heads: number of attention heads
            ff: feedforward dimension
            dropout: dropout rate
            rope_theta: RoPE theta parameter
            ctx: context length
            use_gqa: use Grouped Query Attention (default: False)
            use_swiglu: use SwiGLU activation (default: True)
            use_moe: use Mixture of Experts (default: False)
            num_experts: number of experts for MoE (default: 8)
            num_experts_per_tok: number of experts to activate per token (default: 2)
        """
        super().__init__()
        self.tok_emb = nn.Embedding(vocab, d)
        self.pos_cache = None
        self.blocks = nn.ModuleList([
            Block(d, heads, ff, rope_theta, dropout, use_gqa=use_gqa, use_swiglu=use_swiglu,
                  use_moe=use_moe, num_experts=num_experts, num_experts_per_tok=num_experts_per_tok) 
            for _ in range(n_layers)
        ])
        self.norm = RMSNorm(d)
        self.lm_head = nn.Linear(d, vocab, bias=False)
        self.ctx = ctx
        
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

    def forward(self, idx=None, embeddings=None, attn_mask=None, pos=None):
        """
        Forward pass supporting both token IDs and raw embeddings for multimodal input.
        
        Args:
            idx: (B, T) token indices (for text-only)
            embeddings: (B, T, D) raw embeddings (for multimodal, can include image/audio features)
            attn_mask: (B, T, T) optional attention mask
            pos: (T,) optional position indices (auto-generated if None)
        
        Either idx or embeddings must be provided, not both.
        """
        if embeddings is not None:
            # Use provided embeddings (multimodal case)
            x = embeddings
            T = embeddings.shape[1]
        elif idx is not None:
            # Use token embeddings (text-only case)
            x = self.tok_emb(idx)
            T = idx.shape[1]
        else:
            raise ValueError("Either idx or embeddings must be provided")
        
        # Handle KV caching
        use_cache = self.use_kv_cache and (self.kv_cache is not None or (idx is not None and T == 1))
        
        if use_cache:
            # Initialize cache if first call
            if self.kv_cache is None:
                self.kv_cache = [None] * len(self.blocks)
                # First call with full prompt - create causal mask
                mask = torch.tril(torch.ones(T, T, device=x.device)).unsqueeze(0).unsqueeze(0)  # (1, 1, T, T)
                if attn_mask is not None:
                    if len(attn_mask.shape) == 3:
                        attn_mask = attn_mask.unsqueeze(1)  # (B, 1, T, T)
                    mask = mask * attn_mask
            else:
                # Incremental call - don't pass mask, let Attention create it
                mask = None
            
            if pos is None:
                if self.kv_cache[0] is not None and 'pos' in self.kv_cache[0]:
                    # Incremental: use cached position
                    pos = torch.tensor([self.kv_cache[0]['pos']], device=x.device, dtype=torch.long)
                else:
                    # First call: generate positions
                    pos = make_positions(T, x.device)
            
            # Process through blocks with caching
            for i, blk in enumerate(self.blocks):
                x, self.kv_cache[i] = blk(x, mask=mask, pos=pos, cache=self.kv_cache[i])
        else:
            # No caching - standard forward
            if pos is None:
                pos = make_positions(T, x.device)
            
            # Create causal mask
            mask = torch.tril(torch.ones(T, T, device=x.device)).unsqueeze(0).unsqueeze(0)  # (1, 1, T, T)
            if attn_mask is not None:
                if len(attn_mask.shape) == 3:
                    attn_mask = attn_mask.unsqueeze(1)  # (B, 1, T, T)
                mask = mask * attn_mask
            
            # Standard forward pass
            for blk in self.blocks:
                x, _ = blk(x, mask=mask, pos=pos, cache=None)
        
        x = self.norm(x)
        logits = self.lm_head(x)
        return logits
