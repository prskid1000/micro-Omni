"""
HuggingFace-compatible model wrapper for μOmni.

Usage:
    from modeling_muomni import MuOmniForCausalLM, MuOmniConfig

    # Load from local export
    model = MuOmniForCausalLM.from_pretrained("./export")

    # Or from HuggingFace Hub
    model = MuOmniForCausalLM.from_pretrained("prskid1000/micro-Omni")
"""

import math
import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict
from transformers import PreTrainedModel, PretrainedConfig, GenerationMixin


class MuOmniConfig(PretrainedConfig):
    """Configuration for μOmni model."""
    model_type = "muomni"

    def __init__(
        self,
        vocab_size: int = 256,
        d_model: int = 128,
        n_layers: int = 4,
        n_heads: int = 4,
        d_ff: int = 344,
        dropout: float = 0.05,
        rope_theta: float = 10000.0,
        ctx_len: int = 64,
        use_gqa: bool = True,
        kv_groups: int = 2,
        use_swiglu: bool = True,
        use_moe: bool = False,
        num_experts: int = 8,
        num_experts_per_tok: int = 2,
        pad_token_id: int = 0,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        **kwargs
    ):
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            **kwargs
        )
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.dropout = dropout
        self.rope_theta = rope_theta
        self.ctx_len = ctx_len
        self.use_gqa = use_gqa
        self.kv_groups = kv_groups
        self.use_swiglu = use_swiglu
        self.use_moe = use_moe
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.hidden_size = d_model  # HF standard name
        self.num_hidden_layers = n_layers
        self.num_attention_heads = n_heads
        self.intermediate_size = d_ff
        self.max_position_embeddings = ctx_len


# ---- Lightweight reimplementation of core layers (no omni/ dependency) ----

def rotate_half(x):
    x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
    return torch.cat([-x2, x1], dim=-1)


class RMSNorm(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x):
        x_fp32 = x.float()
        rrms = torch.rsqrt(x_fp32.pow(2).mean(-1, keepdim=True) + 1e-5)
        return (x_fp32 * rrms).to(dtype=x.dtype) * self.weight


class RoPE(nn.Module):
    def __init__(self, d_head, theta=10000.0):
        super().__init__()
        self.d = d_head
        inv_freq = 1.0 / (theta ** (torch.arange(0, d_head, 2).float() / d_head))
        self.register_buffer("inv_freq", inv_freq)
        self._cos_cache = None
        self._sin_cache = None
        self._cache_len = 0

    def _build_cache(self, T, device):
        if T <= self._cache_len and self._cos_cache is not None and self._cos_cache.device == device:
            return
        pos = torch.arange(T, device=device).float()
        freqs = torch.einsum('t,f->tf', pos, self.inv_freq.to(device))
        emb = torch.cat([freqs, freqs], dim=-1)
        self._cos_cache = emb.cos()[None, None, :, :]
        self._sin_cache = emb.sin()[None, None, :, :]
        self._cache_len = T

    def forward(self, q, k, pos_len):
        self._build_cache(pos_len, q.device)
        cos = self._cos_cache[:, :, :pos_len, :]
        sin = self._sin_cache[:, :, :pos_len, :]
        q1 = (q * cos) + (rotate_half(q) * sin)
        k1 = (k * cos) + (rotate_half(k) * sin)
        return q1, k1


class Attention(nn.Module):
    def __init__(self, d, heads, rope_theta=10000.0, use_gqa=False, kv_groups=None):
        super().__init__()
        self.h = heads
        self.dk = d // heads
        self.use_gqa = use_gqa

        if use_gqa:
            self.kv_groups = kv_groups or max(1, heads // 2)
            self.q = nn.Linear(d, heads * self.dk, bias=False)
            self.k = nn.Linear(d, self.kv_groups * self.dk, bias=False)
            self.v = nn.Linear(d, self.kv_groups * self.dk, bias=False)
            self.rope_q = RoPE(self.dk, theta=rope_theta)
            self.rope_k = RoPE(self.dk, theta=rope_theta)
        else:
            self.qkv = nn.Linear(d, 3 * d, bias=False)
            self.rope = RoPE(self.dk, theta=rope_theta)

        self.o = nn.Linear(d, d, bias=False)

    def forward(self, x, mask=None):
        B, T, D = x.shape
        from einops import rearrange

        if self.use_gqa:
            q = rearrange(self.q(x), "b t (h d) -> b h t d", h=self.h)
            k = rearrange(self.k(x), "b t (g d) -> b g t d", g=self.kv_groups)
            v = rearrange(self.v(x), "b t (g d) -> b g t d", g=self.kv_groups)
            q, _ = self.rope_q(q, q, T)
            k, _ = self.rope_k(k, k, T)
            rf = self.h // self.kv_groups
            B_kv, G, T_kv, dk_kv = k.shape
            k = k.unsqueeze(2).expand(B_kv, G, rf, T_kv, dk_kv).reshape(B_kv, -1, T_kv, dk_kv)
            v = v.unsqueeze(2).expand(B_kv, G, rf, T_kv, dk_kv).reshape(B_kv, -1, T_kv, dk_kv)
        else:
            qkv = self.qkv(x).chunk(3, dim=-1)
            q, k, v = [rearrange(t, "b t (h d) -> b h t d", h=self.h) for t in qkv]
            q, k = self.rope(q, k, T)

        y = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, is_causal=(mask is None)
        )
        y = rearrange(y, "b h t d -> b t (h d)")
        return self.o(y)


class MLP(nn.Module):
    def __init__(self, d, ff, use_swiglu=True):
        super().__init__()
        self.use_swiglu = use_swiglu
        if use_swiglu:
            self.gate_proj = nn.Linear(d, ff, bias=False)
            self.up_proj = nn.Linear(d, ff, bias=False)
            self.down_proj = nn.Linear(ff, d, bias=False)
        else:
            self.fc1 = nn.Linear(d, ff)
            self.act = nn.GELU()
            self.fc2 = nn.Linear(ff, d)

    def forward(self, x):
        if self.use_swiglu:
            gate = self.gate_proj(x)
            return self.down_proj(gate * torch.sigmoid(gate) * self.up_proj(x))
        return self.fc2(self.act(self.fc1(x)))


class Block(nn.Module):
    def __init__(self, d, heads, ff, rope_theta, use_gqa=False, kv_groups=None, use_swiglu=True):
        super().__init__()
        self.norm1 = RMSNorm(d)
        self.attn = Attention(d, heads, rope_theta, use_gqa, kv_groups)
        self.norm2 = RMSNorm(d)
        self.mlp = MLP(d, ff, use_swiglu)

    def forward(self, x, mask=None):
        x = x + self.attn(self.norm1(x), mask)
        x = x + self.mlp(self.norm2(x))
        return x


class MuOmniForCausalLM(PreTrainedModel, GenerationMixin):
    """μOmni language model compatible with HuggingFace transformers."""
    config_class = MuOmniConfig
    _no_split_modules = ["Block"]
    _tied_weights_keys = []

    @property
    def all_tied_weights_keys(self):
        return {}

    def __init__(self, config: MuOmniConfig):
        super().__init__(config)
        d = config.d_model
        self.tok_emb = nn.Embedding(config.vocab_size, d)
        self.blocks = nn.ModuleList([
            Block(d, config.n_heads, config.d_ff, config.rope_theta,
                  config.use_gqa, config.kv_groups, config.use_swiglu)
            for _ in range(config.n_layers)
        ])
        self.norm = RMSNorm(d)
        self.lm_head = nn.Linear(d, config.vocab_size, bias=False)

        # Causal mask buffer
        self.register_buffer(
            "_causal_mask",
            torch.tril(torch.ones(config.ctx_len, config.ctx_len)).unsqueeze(0).unsqueeze(0),
            persistent=False
        )

    def get_input_embeddings(self):
        return self.tok_emb

    def set_input_embeddings(self, value):
        self.tok_emb = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        x = self.tok_emb(input_ids)
        T = x.shape[1]

        mask = self._causal_mask[:, :, :T, :T]
        if attention_mask is not None:
            pad_mask = attention_mask.unsqueeze(1).unsqueeze(2).float()
            mask = mask * pad_mask

        for blk in self.blocks:
            x = blk(x, mask)

        x = self.norm(x)
        logits = self.lm_head(x)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = nn.CrossEntropyLoss(ignore_index=-100)(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )

        from transformers.modeling_outputs import CausalLMOutputWithPast
        return CausalLMOutputWithPast(loss=loss, logits=logits)

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        return {"input_ids": input_ids}


# Auto-registration so AutoModel can discover this model type
MuOmniConfig.register_for_auto_class()
MuOmniForCausalLM.register_for_auto_class("AutoModelForCausalLM")
