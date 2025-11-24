
import math
import torch
from torch import nn
from typing import Optional, Tuple, Dict, List, Union
from omni.utils import RMSNorm, RoPE, make_positions
from einops import rearrange
import warnings

# Check for Flash Attention support (PyTorch 2.0+)
HAS_FLASH_ATTENTION = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
if HAS_FLASH_ATTENTION:
    from torch.nn.functional import scaled_dot_product_attention

class SwiGLU(nn.Module):
    """Swish Gated Linear Unit activation (SwiGLU) as used in Qwen3 Omni"""
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(dim, dim, bias=False)
        self.up_proj = nn.Linear(dim, dim, bias=False)
        self.down_proj = nn.Linear(dim, dim, bias=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
    def __init__(self, d: int, ff: int, use_swiglu: bool = True) -> None:
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
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_swiglu:
            # SwiGLU: gate(x) * swish(gate(x)) * up(x)
            gate = self.gate_proj(x)
            up = self.up_proj(x)
            swish = gate * torch.sigmoid(gate)  # Swish activation
            return self.down_proj(swish * up)
        else:
            return self.fc2(self.act(self.fc1(x)))

class Attention(nn.Module):
    def __init__(self, d: int, heads: int, rope_theta: float = 10000.0, dropout: float = 0.0, 
                 use_gqa: bool = False, kv_groups: Optional[int] = None, use_flash: bool = True) -> None:
        """
        Attention with optional GQA (Grouped Query Attention) and Flash Attention.
        
        Args:
            d: model dimension
            heads: number of query heads
            rope_theta: RoPE theta parameter
            dropout: dropout rate
            use_gqa: whether to use GQA
            kv_groups: number of key/value groups (if None and use_gqa=True, uses heads//2)
            use_flash: whether to use Flash Attention (PyTorch 2.0+ scaled_dot_product_attention)
        """
        super().__init__()
        self.h = heads  # query heads
        self.d = d
        self.dk = d // heads
        self.use_gqa = use_gqa
        self.use_flash = use_flash and HAS_FLASH_ATTENTION
        
        if use_flash and not HAS_FLASH_ATTENTION:
            warnings.warn("Flash Attention requested but not available. Falling back to standard attention.")
            self.use_flash = False
        
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
        self.dropout_p = dropout

    def _compute_attention_flash(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, 
                                  attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute attention using Flash Attention (scaled_dot_product_attention).
        
        Args:
            q: (B, h, T, dk) query
            k: (B, h, T_total, dk) key
            v: (B, h, T_total, dk) value
            attn_mask: Optional attention mask (bool or float mask)
        
        Returns:
            y: (B, h, T, dk) attention output
        """
        # Convert float mask to bool mask if needed
        if attn_mask is not None:
            if attn_mask.dtype == torch.float32 or attn_mask.dtype == torch.float16:
                # Convert float mask (1s and 0s) to bool mask
                # 0 means "mask out", so we want False where mask==0
                attn_mask = attn_mask.bool()
        
        # Use Flash Attention - this is 2-4x faster and more memory efficient
        y = scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=False  # We handle causality through the mask
        )
        return y
    
    def _compute_attention_manual(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                                   mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute attention manually (fallback when Flash Attention is not available).
        
        Args:
            q: (B, h, T, dk) query
            k: (B, h, T_total, dk) key
            v: (B, h, T_total, dk) value
            mask: Optional attention mask
        
        Returns:
            y: (B, h, T, dk) attention output
        """
        B = q.shape[0]
        
        # Check for NaN in q, k before attention computation
        if torch.isnan(q).any() or torch.isnan(k).any():
            raise RuntimeError("NaN detected in attention queries or keys")
        
        att = torch.einsum("bhtd,bhTd->bhtT", q, k) / math.sqrt(self.dk)
        
        # Clamp attention scores to prevent extreme values that could cause NaN in softmax
        att = torch.clamp(att, min=-50.0, max=50.0)
        
        if mask is not None:
            # att is (B, H, T, T_total), mask should be broadcastable
            if len(mask.shape) == 4:
                # Expand mask to match attention heads: (B, 1, T, T_total) -> (B, H, T, T_total)
                mask = mask.expand(B, self.h, -1, -1)
            att = att.masked_fill(mask == 0, float("-inf"))
        
        # Check for NaN before softmax
        if torch.isnan(att).any():
            raise RuntimeError("NaN detected in attention scores before softmax")
        
        att = att.softmax(dim=-1)
        
        # Check for NaN after softmax
        if torch.isnan(att).any():
            raise RuntimeError("NaN detected in attention probabilities after softmax")
        
        att = self.drop(att)
        
        # Check for NaN in values
        if torch.isnan(v).any():
            raise RuntimeError("NaN detected in attention values")
        
        y = torch.einsum("bhtT,bhTd->bhtd", att, v)
        
        # Check for NaN in attention output
        if torch.isnan(y).any():
            raise RuntimeError("NaN detected in attention output")
        
        return y

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None, 
                pos: Optional[torch.Tensor] = None, cache: Optional[Dict[str, torch.Tensor]] = None, 
                need_cache: bool = False) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
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
        store_cache = need_cache or cache is not None
        
        if self.use_gqa:
            # GQA: separate Q, K, V projections
            q = self.q(x)  # (B, T, heads * dk)
            k = self.k(x)  # (B, T, kv_groups * dk)
            v = self.v(x)  # (B, T, kv_groups * dk)
            
            # Reshape
            q = rearrange(q, "b t (h d) -> b h t d", h=self.h)
            k_new = rearrange(k, "b t (g d) -> b g t d", g=self.kv_groups)
            v_new = rearrange(v, "b t (g d) -> b g t d", g=self.kv_groups)
            
            # Apply RoPE
            if pos is None:
                pos = make_positions(T, x.device)
            
            q, _ = self.rope_q(q, q, pos)
            k_new, _ = self.rope_k(k_new, k_new, pos)
            
            if cache is not None:
                k_combined = torch.cat([cache['k'], k_new], dim=2)  # (B, g, T_total, dk)
                v_combined = torch.cat([cache['v'], v_new], dim=2)  # (B, g, T_total, dk)
            else:
                k_combined = k_new
                v_combined = v_new
            
            # Repeat k and v to match query heads (each kv head serves multiple q heads)
            repeat_factor = self.h // self.kv_groups
            k = k_combined.repeat_interleave(repeat_factor, dim=1)  # (B, h, T_total, dk)
            v = v_combined.repeat_interleave(repeat_factor, dim=1)  # (B, h, T_total, dk)
            
            next_cache = None
            if store_cache:
                if pos is not None and pos.numel() > 0:
                    next_pos = pos[-1].item() + 1
                elif cache is not None and 'pos' in cache:
                    next_pos = cache['pos'] + T
                else:
                    next_pos = k_combined.shape[2]
                next_cache = {'k': k_combined, 'v': v_combined, 'pos': next_pos}
        else:
            # Standard MHA
            qkv = self.qkv(x).chunk(3, dim=-1)  # 3 * (B,T,D)
            q, k, v = [rearrange(t, "b t (h d) -> b h t d", h=self.h) for t in qkv]
            
            # Apply RoPE
            if pos is None:
                pos = make_positions(T, x.device)
            
            q, k = self.rope(q, k, pos)
            
            if cache is not None and 'k' in cache and 'v' in cache:
                k = torch.cat([cache['k'], k], dim=2)  # (B, h, T_cached+T, dk)
                v = torch.cat([cache['v'], v], dim=2)  # (B, h, T_cached+T, dk)
            
            next_cache = None
            if store_cache:
                if pos is not None and pos.numel() > 0:
                    next_pos = pos[-1].item() + 1
                elif cache is not None and 'pos' in cache:
                    next_pos = cache['pos'] + T
                else:
                    next_pos = k.shape[2]
                next_cache = {'k': k, 'v': v, 'pos': next_pos}
        
        # Attention computation
        T_total = k.shape[2]
        
        # Prepare mask for attention
        is_incremental = cache is not None
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
            
            # Ensure mask matches attention shape
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
        
        # Compute attention using Flash Attention or manual implementation
        if self.use_flash:
            y = self._compute_attention_flash(q, k, v, mask)
        else:
            y = self._compute_attention_manual(q, k, v, mask)
        
        y = rearrange(y, "b h t d -> b t (h d)")
        
        return self.o(y), next_cache

class MoE(nn.Module):
    """Mixture of Experts layer"""
    def __init__(self, d: int, ff: int, num_experts: int = 8, num_experts_per_tok: int = 2, 
                 use_swiglu: bool = True) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.router = nn.Linear(d, num_experts, bias=False)
        self.experts = nn.ModuleList([
            MLP(d, ff, use_swiglu=use_swiglu) for _ in range(num_experts)
        ])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
    def __init__(self, d: int, heads: int, ff: int, rope_theta: float, dropout: float, 
                 use_gqa: bool = False, use_swiglu: bool = True, use_moe: bool = False, 
                 num_experts: int = 8, num_experts_per_tok: int = 2, use_flash: bool = True) -> None:
        super().__init__()
        self.norm1 = RMSNorm(d)
        self.attn = Attention(d, heads, rope_theta=rope_theta, dropout=dropout, use_gqa=use_gqa, use_flash=use_flash)
        self.norm2 = RMSNorm(d)
        self.use_moe = use_moe
        if use_moe:
            self.moe = MoE(d, ff, num_experts, num_experts_per_tok, use_swiglu)
        else:
            self.mlp = MLP(d, ff, use_swiglu=use_swiglu)
        self.drop = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None, 
                pos: Optional[torch.Tensor] = None, cache: Optional[Dict[str, torch.Tensor]] = None, 
                return_cache: bool = False) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
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
        attn_out, cache = self.attn(self.norm1(x), mask=mask, pos=pos, cache=cache, need_cache=return_cache)
        x = x + self.drop(attn_out)
        if self.use_moe:
            x = x + self.drop(self.moe(self.norm2(x)))
        else:
            x = x + self.drop(self.mlp(self.norm2(x)))
        return x, cache

class ThinkerLM(nn.Module):
    def __init__(self, vocab: int, n_layers: int = 16, d: int = 512, heads: int = 8, ff: int = 2048, 
                 dropout: float = 0.1, rope_theta: float = 10000, ctx: int = 1024, 
                 use_gqa: bool = False, use_swiglu: bool = True, use_moe: bool = False, 
                 num_experts: int = 8, num_experts_per_tok: int = 2, use_flash: bool = True,
                 compile_model: bool = False) -> None:
        """
        ThinkerLM with optional Qwen3 Omni features and performance optimizations.
        
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
            use_flash: use Flash Attention for 2-4x speedup (default: True, requires PyTorch 2.0+)
            compile_model: use torch.compile() for 30-50% speedup (default: False, requires PyTorch 2.0+)
        """
        super().__init__()
        self.tok_emb = nn.Embedding(vocab, d)
        self.pos_cache = None
        self.blocks = nn.ModuleList([
            Block(d, heads, ff, rope_theta, dropout, use_gqa=use_gqa, use_swiglu=use_swiglu,
                  use_moe=use_moe, num_experts=num_experts, num_experts_per_tok=num_experts_per_tok,
                  use_flash=use_flash) 
            for _ in range(n_layers)
        ])
        self.norm = RMSNorm(d)
        self.lm_head = nn.Linear(d, vocab, bias=False)
        self.ctx = ctx
        
        # KV cache for autoregressive generation
        self.kv_cache = None
        self.use_kv_cache = False
        
        # Compilation support for additional speedup
        self._compiled = False
        if compile_model:
            self._apply_compilation()
    
    def _apply_compilation(self) -> None:
        """
        Apply torch.compile() to the model for 30-50% speedup.
        Requires PyTorch 2.0+.
        """
        if not hasattr(torch, 'compile'):
            warnings.warn("torch.compile() not available. Requires PyTorch 2.0+. Skipping compilation.")
            return
        
        try:
            # Compile individual blocks for better compilation efficiency
            # Using 'cudagraphs' backend to avoid Triton/LLVM compatibility issues
            # Provides 10-20% speedup without requiring Triton compilation
            for i, block in enumerate(self.blocks):
                self.blocks[i] = torch.compile(block, backend='cudagraphs', mode='default', fullgraph=False)
            
            # Compile embedding and output head
            self.tok_emb = torch.compile(self.tok_emb, backend='cudagraphs', mode='default', fullgraph=False)
            self.lm_head = torch.compile(self.lm_head, backend='cudagraphs', mode='default', fullgraph=False)
            
            self._compiled = True
            print(f"✓ Model compiled successfully with torch.compile()")
        except Exception as e:
            warnings.warn(f"Failed to compile model: {e}. Continuing without compilation.")
            warnings.warn("If you encounter Triton compilation errors during training, set 'use_compile': false in your config.")
    
    def reset_kv_cache(self) -> None:
        """Reset KV cache (call before new generation)"""
        self.kv_cache = None
    
    def enable_kv_cache(self, enable: bool = True) -> None:
        """Enable/disable KV caching for faster autoregressive generation"""
        self.use_kv_cache = enable
        if not enable:
            self.kv_cache = None
    
    def check_weights_stability(self) -> Tuple[bool, bool, int, int]:
        """
        Check if any model weights contain NaN or Inf values.
        
        Returns:
            tuple: (has_nan, has_inf, nan_count, inf_count)
        """
        has_nan = False
        has_inf = False
        nan_count = 0
        inf_count = 0
        
        for name, param in self.named_parameters():
            if param.requires_grad:
                param_nan = torch.isnan(param).any().item()
                param_inf = torch.isinf(param).any().item()
                if param_nan:
                    has_nan = True
                    nan_count += torch.isnan(param).sum().item()
                if param_inf:
                    has_inf = True
                    inf_count += torch.isinf(param).sum().item()
        
        return has_nan, has_inf, nan_count, inf_count

    def forward(self, idx: Optional[torch.Tensor] = None, embeddings: Optional[torch.Tensor] = None, 
                attn_mask: Optional[torch.Tensor] = None, pos: Optional[torch.Tensor] = None) -> torch.Tensor:
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
        
        # Early NaN detection after embedding
        if torch.isnan(x).any() or torch.isinf(x).any():
            nan_count = torch.isnan(x).sum().item()
            inf_count = torch.isinf(x).sum().item()
            raise RuntimeError(f"Numerical instability detected after embedding: NaN={nan_count}, Inf={inf_count}")
        
        using_cache = self.use_kv_cache
        if using_cache and self.kv_cache is None:
            self.kv_cache = [None] * len(self.blocks)
        
        cache_ready = using_cache and self.kv_cache[0] is not None
        if using_cache and cache_ready and T > 1:
            # New sequence provided without explicit reset
            self.reset_kv_cache()
            self.kv_cache = [None] * len(self.blocks)
            cache_ready = False
        
        if using_cache:
            if cache_ready:
                # Incremental step
                if pos is None:
                    pos = torch.tensor([self.kv_cache[0]['pos']], device=x.device, dtype=torch.long)
                mask = None
                for i, blk in enumerate(self.blocks):
                    x, new_cache = blk(x, mask=mask, pos=pos, cache=self.kv_cache[i], return_cache=True)
                    self.kv_cache[i] = new_cache
                    if torch.isnan(x).any() or torch.isinf(x).any():
                        nan_count = torch.isnan(x).sum().item()
                        inf_count = torch.isinf(x).sum().item()
                        raise RuntimeError(f"Numerical instability detected after block {i}: NaN={nan_count}, Inf={inf_count}")
            else:
                # Prefill step: build caches from full sequence
                if pos is None:
                    pos = make_positions(T, x.device)
                mask = torch.tril(torch.ones(T, T, device=x.device)).unsqueeze(0).unsqueeze(0)
                if attn_mask is not None:
                    if len(attn_mask.shape) == 3:
                        attn_mask = attn_mask.unsqueeze(1)
                    mask = mask * attn_mask
                for i, blk in enumerate(self.blocks):
                    x, new_cache = blk(x, mask=mask, pos=pos, cache=None, return_cache=True)
                    self.kv_cache[i] = new_cache
                    if torch.isnan(x).any() or torch.isinf(x).any():
                        nan_count = torch.isnan(x).sum().item()
                        inf_count = torch.isinf(x).sum().item()
                        raise RuntimeError(f"Numerical instability detected after block {i}: NaN={nan_count}, Inf={inf_count}")
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
            for i, blk in enumerate(self.blocks):
                x, _ = blk(x, mask=mask, pos=pos, cache=None, return_cache=False)
                # Check for NaN after each block
                if torch.isnan(x).any() or torch.isinf(x).any():
                    nan_count = torch.isnan(x).sum().item()
                    inf_count = torch.isinf(x).sum().item()
                    raise RuntimeError(f"Numerical instability detected after block {i}: NaN={nan_count}, Inf={inf_count}")
        
        x = self.norm(x)
        # Check after final norm
        if torch.isnan(x).any() or torch.isinf(x).any():
            nan_count = torch.isnan(x).sum().item()
            inf_count = torch.isinf(x).sum().item()
            raise RuntimeError(f"Numerical instability detected after final norm: NaN={nan_count}, Inf={inf_count}")
        
        logits = self.lm_head(x)
        
        # Check for numerical stability (NaN/Inf detection)
        if torch.isnan(logits).any() or torch.isinf(logits).any():
            nan_count = torch.isnan(logits).sum().item()
            inf_count = torch.isinf(logits).sum().item()
            raise RuntimeError(f"Numerical instability in ThinkerLM forward pass: NaN={nan_count}, Inf={inf_count}")
        
        return logits
