"""
HuggingFace-compatible model wrapper for μOmni.

Usage:
    from modeling_muomni import MuOmniForCausalLM, MuOmniConfig

    # Text-only (thinker)
    model = MuOmniForCausalLM.from_pretrained("./export")

    # Full multimodal (image + audio + text + speech)
    model = MuOmniMultimodalModel.from_pretrained("./export", safetensors_file="model_full.safetensors")

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

        if mask is not None:
            # Convert float mask to bool for SDPA compatibility
            attn_mask = mask.bool() if mask.dtype != torch.bool else mask
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        else:
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
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

        mask = self._causal_mask[:, :, :T, :T].to(dtype=x.dtype)
        if attention_mask is not None:
            pad_mask = attention_mask.unsqueeze(1).unsqueeze(2).to(dtype=x.dtype)
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


# ===========================================================================
# Multimodal components — inference-only reimplementations
# ===========================================================================

class MuOmniMultimodalConfig(PretrainedConfig):
    """Configuration for the full multimodal μOmni model."""
    model_type = "muomni_multimodal"

    def __init__(
        self,
        # Thinker (decoder LLM)
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
        # Audio encoder
        audio_d: int = 128,
        audio_layers: int = 4,
        audio_heads: int = 4,
        audio_ff: int = 512,
        audio_mel_bins: int = 128,
        audio_downsample: int = 8,
        # Vision encoder
        vision_d: int = 128,
        vision_layers: int = 4,
        vision_heads: int = 4,
        vision_ff: int = 344,
        vision_img_size: int = 224,
        vision_patch: int = 16,
        # Talker
        talker_d: int = 128,
        talker_layers: int = 4,
        talker_heads: int = 4,
        talker_ff: int = 344,
        talker_codebooks: int = 2,
        talker_codebook_size: int = 128,
        # RVQ codec
        rvq_codebooks: int = 2,
        rvq_codebook_size: int = 128,
        rvq_d: int = 64,
        # Token IDs
        pad_token_id: int = 0,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        **kwargs,
    ):
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            **kwargs,
        )
        # Thinker
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
        # Audio
        self.audio_d = audio_d
        self.audio_layers = audio_layers
        self.audio_heads = audio_heads
        self.audio_ff = audio_ff
        self.audio_mel_bins = audio_mel_bins
        self.audio_downsample = audio_downsample
        # Vision
        self.vision_d = vision_d
        self.vision_layers = vision_layers
        self.vision_heads = vision_heads
        self.vision_ff = vision_ff
        self.vision_img_size = vision_img_size
        self.vision_patch = vision_patch
        # Talker
        self.talker_d = talker_d
        self.talker_layers = talker_layers
        self.talker_heads = talker_heads
        self.talker_ff = talker_ff
        self.talker_codebooks = talker_codebooks
        self.talker_codebook_size = talker_codebook_size
        # RVQ
        self.rvq_codebooks = rvq_codebooks
        self.rvq_codebook_size = rvq_codebook_size
        self.rvq_d = rvq_d
        # HF standard aliases
        self.hidden_size = d_model
        self.num_hidden_layers = n_layers
        self.num_attention_heads = n_heads
        self.intermediate_size = d_ff
        self.max_position_embeddings = ctx_len


# ---- Audio Encoder (matches omni/audio_encoder.py AudioEncoderTiny) ----

class AudioConvDown(nn.Module):
    """Conv2d 4x downsample (stride-2 twice). Matches ConvDown weight names."""
    def __init__(self, in_ch: int = 1, mid: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, mid, 3, stride=2, padding=1), nn.GELU(),
            nn.Conv2d(mid, mid, 3, stride=2, padding=1), nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class AudioEncoderBlock(nn.Module):
    """
    Single encoder block matching omni/audio_encoder.py EncoderBlock weight names.
    Uses combined qkv_proj + out_proj (no RoPE, non-causal) and standard MLP.
    """
    def __init__(self, d: int, heads: int, ff: int):
        super().__init__()
        self.d = d
        self.heads = heads
        self.head_dim = d // heads
        self.norm1 = RMSNorm(d)
        self.qkv_proj = nn.Linear(d, 3 * d, bias=True)
        self.out_proj = nn.Linear(d, d, bias=True)
        self.norm2 = RMSNorm(d)
        self.mlp = MLP(d, ff, use_swiglu=False)  # EncoderBlock uses GELU MLP
        self.drop = nn.Dropout(0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        normed = self.norm1(x)
        qkv = self.qkv_proj(normed).reshape(B, T, 3, self.heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, h, T, dk)
        q, k, v = qkv[0], qkv[1], qkv[2]
        y = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=False)
        y = y.transpose(1, 2).reshape(B, T, D)
        x = x + self.out_proj(y)
        x = x + self.mlp(self.norm2(x))
        return x


class AudioEncoder(nn.Module):
    """
    Minimal reimplementation of AudioEncoderTiny for inference.
    mel (B, T, 128) -> conv downsample -> transformer encoder -> (B, T', d)
    Weight names: down.*, proj.*, blocks.N.*, norm.*
    """
    def __init__(self, d: int = 128, heads: int = 4, ff: int = 512,
                 layers: int = 4, downsample_factor: int = 8, mel_bins: int = 128):
        super().__init__()
        self.downsample_factor = downsample_factor
        if downsample_factor == 8:
            self.down = nn.Sequential(
                AudioConvDown(1, mid=64),
                nn.Conv2d(64, 64, 3, stride=2, padding=1), nn.GELU(),
            )
        else:
            self.down = AudioConvDown(1, mid=64)
        # Conv output: 64 channels * (mel_bins / freq_downsample)
        freq_ds = downsample_factor
        self.proj = nn.Linear(64 * (mel_bins // freq_ds), d)
        self.blocks = nn.ModuleList([AudioEncoderBlock(d, heads, ff) for _ in range(layers)])
        self.norm = RMSNorm(d)

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """
        Args:
            mel: (B, T, mel_bins) mel spectrogram
        Returns:
            (B, T/downsample_factor, d) frame embeddings
        """
        x = mel[:, None, :, :]          # (B, 1, T, mel_bins)
        x = self.down(x)                # (B, 64, T', F')
        B, C, T, F = x.shape
        x = x.permute(0, 2, 1, 3).contiguous().view(B, T, C * F)
        x = self.proj(x)
        for blk in self.blocks:
            x = blk(x)
        return self.norm(x)


# ---- Vision Encoder (matches omni/vision_encoder.py ViTTiny) ----

class VisionEncoder(nn.Module):
    """
    Minimal reimplementation of ViTTiny for inference.
    Uses nn.TransformerEncoderLayer to match weight names
    (self_attn.in_proj_weight, self_attn.out_proj, linear1, linear2, norm1, norm2).
    Weight names: proj.*, cls, pos, blocks.N.*, norm.*
    """
    def __init__(self, img_size: int = 224, patch: int = 16, d: int = 128,
                 layers: int = 4, heads: int = 4, ff: int = 344):
        super().__init__()
        self.patch = patch
        self.d = d
        self.proj = nn.Conv2d(3, d, kernel_size=patch, stride=patch)
        num_patches = (img_size // patch) ** 2
        self.cls = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.pos = nn.Parameter(torch.randn(1, 1 + num_patches, d) * 0.02)
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d, heads, ff, dropout=0.0, batch_first=True,
                norm_first=True, activation='gelu',
            )
            for _ in range(layers)
        ])
        self.norm = RMSNorm(d)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, 3, H, W) image tensor
        Returns:
            cls: (B, 1, d) CLS token embedding
            grid: (B, N, d) patch embeddings
        """
        from einops import rearrange
        x = self.proj(x)                              # (B, d, H', W')
        x = rearrange(x, "b d h w -> b (h w) d")
        B = x.shape[0]
        cls = self.cls.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.pos[:, : x.size(1), :]
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x[:, :1, :], x[:, 1:, :]


# ---- Talker (matches omni/talker.py TalkerTiny) ----

class Talker(nn.Module):
    """
    Minimal reimplementation of TalkerTiny for inference.
    AR transformer that predicts 2 RVQ codebook codes per frame.
    Uses the same Block class as the Thinker (GQA + SwiGLU + RoPE).
    Weight names: emb.*, start, blocks.N.*, norm.*, base_head.*, res_head.*
    """
    def __init__(self, d: int = 128, n_layers: int = 4, n_heads: int = 4,
                 ff: int = 344, codebooks: int = 2, codebook_size: int = 128,
                 rope_theta: float = 10000.0, use_gqa: bool = True,
                 kv_groups: int = 2, use_swiglu: bool = True):
        super().__init__()
        self.emb = nn.Embedding(codebook_size, d)
        self.start = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.blocks = nn.ModuleList([
            Block(d, n_heads, ff, rope_theta, use_gqa=use_gqa,
                  kv_groups=kv_groups, use_swiglu=use_swiglu)
            for _ in range(n_layers)
        ])
        self.norm = RMSNorm(d)
        self.base_head = nn.Linear(d, codebook_size)
        self.res_head = nn.Linear(d, codebook_size)
        self.register_buffer(
            "_causal_mask",
            torch.tril(torch.ones(2048, 2048)).unsqueeze(0).unsqueeze(0),
            persistent=False,
        )

    def forward(self, prev_codes: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            prev_codes: (B, T, 2) previous frame codes [base, residual]
        Returns:
            (base_logits, res_logits) each (B, T, codebook_size)
        """
        B, T, _ = prev_codes.shape
        tok = self.emb(prev_codes[:, :, 0]) + self.emb(prev_codes[:, :, 1])
        x = torch.cat([self.start.expand(B, -1, -1), tok], dim=1)
        S = x.shape[1]
        mask = self._causal_mask[:, :, :S, :S]
        for blk in self.blocks:
            x = blk(x, mask)
        x = self.norm(x)
        x = x[:, 1:, :]  # drop start token
        return self.base_head(x), self.res_head(x)


# ---- RVQ codec (matches omni/codec.py RVQ) ----

class RVQCodec(nn.Module):
    """
    Two-level residual vector quantizer for 128-bin mel frames.
    Weight names: codebooks.N.weight, proj_in.*, proj_out.*
    """
    def __init__(self, codebooks: int = 2, codebook_size: int = 128, d: int = 64,
                 mel_bins: int = 128):
        super().__init__()
        self.codebooks = nn.ParameterList([
            nn.Embedding(codebook_size, d) for _ in range(codebooks)
        ])
        self.proj_in = nn.Linear(mel_bins, d)
        self.proj_out = nn.Linear(d, mel_bins)

    def encode(self, mel_frame: torch.Tensor) -> torch.Tensor:
        """
        Args:
            mel_frame: (B, mel_bins) or (B, T, mel_bins)
        Returns:
            indices: (B, codebooks) or (B, T, codebooks)
        """
        if mel_frame.dim() == 3:
            B, T, _ = mel_frame.shape
            flat = mel_frame.reshape(B * T, -1)
            idxs = self._encode_single(flat)
            return idxs.view(B, T, -1)
        return self._encode_single(mel_frame)

    def _encode_single(self, mel_frame: torch.Tensor) -> torch.Tensor:
        z = self.proj_in(mel_frame)
        residual = z
        idxs = []
        for cb in self.codebooks:
            dist = torch.cdist(residual.unsqueeze(1), cb.weight.unsqueeze(0)).squeeze(1)
            ind = dist.argmin(dim=-1)
            idxs.append(ind)
            residual = residual - cb(ind)
        return torch.stack(idxs, dim=-1)

    def decode(self, idxs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            idxs: (B, codebooks) or (B, T, codebooks)
        Returns:
            mel: (B, mel_bins) or (B, T, mel_bins)
        """
        if idxs.dim() == 3:
            B, T, C = idxs.shape
            flat = idxs.reshape(B * T, C)
            mel = self._decode_single(flat)
            return mel.view(B, T, -1)
        return self._decode_single(idxs)

    def _decode_single(self, idxs: torch.Tensor) -> torch.Tensor:
        z = sum(cb(idxs[:, i]) for i, cb in enumerate(self.codebooks))
        return self.proj_out(z)


# ---- Full Multimodal Model ----

class MuOmniMultimodalModel(PreTrainedModel):
    """
    Full multimodal μOmni model: image + audio + text input, text + speech output.

    Components (matching prefixed keys in model_full.safetensors):
        thinker.*          - decoder-only LLM
        audio_encoder.*    - mel -> transformer encoder
        vision_encoder.*   - image -> ViT
        talker.*           - AR speech code predictor
        rvq.*              - RVQ codec (encode/decode mel)
        proj_a.*           - Linear(audio_d, d_model)
        proj_v.*           - Linear(vision_d, d_model)
    """
    config_class = MuOmniMultimodalConfig
    _no_split_modules = ["Block", "AudioEncoderBlock"]

    def __init__(self, config: MuOmniMultimodalConfig):
        super().__init__(config)
        d = config.d_model

        # --- Thinker (decoder LLM) ---
        self.thinker = nn.ModuleDict({
            "tok_emb": nn.Embedding(config.vocab_size, d),
            "blocks": nn.ModuleList([
                Block(d, config.n_heads, config.d_ff, config.rope_theta,
                      use_gqa=config.use_gqa, kv_groups=config.kv_groups,
                      use_swiglu=config.use_swiglu)
                for _ in range(config.n_layers)
            ]),
            "norm": RMSNorm(d),
            "lm_head": nn.Linear(d, config.vocab_size, bias=False),
        })
        self.register_buffer(
            "_causal_mask",
            torch.tril(torch.ones(config.ctx_len, config.ctx_len)).unsqueeze(0).unsqueeze(0),
            persistent=False,
        )

        # --- Audio Encoder ---
        self.audio_encoder = AudioEncoder(
            d=config.audio_d, heads=config.audio_heads, ff=config.audio_ff,
            layers=config.audio_layers, downsample_factor=config.audio_downsample,
            mel_bins=config.audio_mel_bins,
        )

        # --- Vision Encoder ---
        self.vision_encoder = VisionEncoder(
            img_size=config.vision_img_size, patch=config.vision_patch,
            d=config.vision_d, layers=config.vision_layers,
            heads=config.vision_heads, ff=config.vision_ff,
        )

        # --- Talker ---
        self.talker = Talker(
            d=config.talker_d, n_layers=config.talker_layers,
            n_heads=config.talker_heads, ff=config.talker_ff,
            codebooks=config.talker_codebooks,
            codebook_size=config.talker_codebook_size,
            rope_theta=config.rope_theta,
            use_gqa=config.use_gqa, kv_groups=config.kv_groups,
            use_swiglu=config.use_swiglu,
        )

        # --- RVQ ---
        self.rvq = RVQCodec(
            codebooks=config.rvq_codebooks,
            codebook_size=config.rvq_codebook_size,
            d=config.rvq_d,
            mel_bins=config.audio_mel_bins,
        )

        # --- Projectors ---
        self.proj_a = nn.Linear(config.audio_d, d)
        self.proj_v = nn.Linear(config.vision_d, d)

    # ---- Convenience encode/decode methods ----

    @torch.no_grad()
    def encode_image(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Encode an image into thinker-space embeddings.

        Args:
            pixel_values: (B, 3, H, W) normalised image tensor
        Returns:
            (B, 1, d_model) projected CLS embedding
        """
        cls, _grid = self.vision_encoder(pixel_values)
        return self.proj_v(cls)

    @torch.no_grad()
    def encode_audio(self, mel_spectrogram: torch.Tensor) -> torch.Tensor:
        """
        Encode a mel spectrogram into thinker-space embeddings.

        Args:
            mel_spectrogram: (B, T, mel_bins)
        Returns:
            (B, T', d_model) projected frame embeddings
        """
        frames = self.audio_encoder(mel_spectrogram)
        return self.proj_a(frames)

    @torch.no_grad()
    def generate_text(self, embeddings: torch.Tensor, max_new_tokens: int = 32,
                      temperature: float = 1.0, top_k: int = 0) -> torch.Tensor:
        """
        Autoregressively generate text tokens from prefix embeddings.

        Args:
            embeddings: (B, S, d_model) prefix embeddings (from encode_image/audio or tok_emb)
            max_new_tokens: number of tokens to generate
            temperature: sampling temperature
            top_k: if >0, restrict sampling to top-k logits
        Returns:
            (B, max_new_tokens) generated token IDs
        """
        tok_emb = self.thinker["tok_emb"]
        blocks = self.thinker["blocks"]
        norm = self.thinker["norm"]
        lm_head = self.thinker["lm_head"]

        x = embeddings
        generated = []
        for _ in range(max_new_tokens):
            T = x.shape[1]
            # Expand causal mask if needed
            if T > self._causal_mask.shape[-1]:
                new_mask = torch.tril(torch.ones(T, T, device=x.device)).unsqueeze(0).unsqueeze(0)
            else:
                new_mask = self._causal_mask[:, :, :T, :T]
            h = x
            for blk in blocks:
                h = blk(h, new_mask)
            h = norm(h)
            logits = lm_head(h[:, -1:, :])  # (B, 1, vocab)
            logits = logits[:, 0, :] / max(temperature, 1e-5)
            if top_k > 0:
                topk_vals, _ = logits.topk(top_k, dim=-1)
                logits[logits < topk_vals[:, -1:]] = -float("inf")
            probs = torch.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, 1)  # (B, 1)
            generated.append(next_id)
            next_emb = tok_emb(next_id)  # (B, 1, d)
            x = torch.cat([x, next_emb], dim=1)
        return torch.cat(generated, dim=1)

    @torch.no_grad()
    def generate_speech(self, text_ids: torch.Tensor, max_frames: int = 64,
                        temperature: float = 1.0) -> torch.Tensor:
        """
        Generate speech codes from text token IDs using the Talker.

        Args:
            text_ids: (B, T) text token IDs (feed through thinker first to get context,
                      then use talker autoregressively)
            max_frames: maximum number of speech frames to generate
            temperature: sampling temperature
        Returns:
            (B, max_frames, 2) generated RVQ codes [base, residual]
        """
        B = text_ids.shape[0]
        device = text_ids.device
        # Start with a zero-code frame
        codes = torch.zeros(B, 1, 2, dtype=torch.long, device=device)
        for _ in range(max_frames):
            base_logits, res_logits = self.talker(codes)
            # Sample from the last frame prediction
            b_logits = base_logits[:, -1, :] / max(temperature, 1e-5)
            r_logits = res_logits[:, -1, :] / max(temperature, 1e-5)
            base_id = torch.multinomial(torch.softmax(b_logits, -1), 1)
            res_id = torch.multinomial(torch.softmax(r_logits, -1), 1)
            new_code = torch.stack([base_id, res_id], dim=-1)  # (B, 1, 2)
            codes = torch.cat([codes, new_code], dim=1)
        return codes[:, 1:, :]  # drop initial zero frame

    # ---- Forward (flexible multimodal) ----

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        mel_spectrogram: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        """
        Multimodal forward pass.  Embeds each modality, concatenates along the
        sequence dimension (vision | audio | text), and runs the thinker LLM.

        Args:
            input_ids: (B, T_text) text token IDs
            pixel_values: (B, 3, H, W) image tensor
            mel_spectrogram: (B, T_mel, mel_bins) mel spectrogram
            attention_mask: optional mask for text tokens
            labels: optional text labels for loss computation
        """
        parts = []

        # 1. Vision
        if pixel_values is not None:
            parts.append(self.encode_image(pixel_values))

        # 2. Audio
        if mel_spectrogram is not None:
            parts.append(self.encode_audio(mel_spectrogram))

        # 3. Text
        if input_ids is not None:
            parts.append(self.thinker["tok_emb"](input_ids))

        if not parts:
            raise ValueError("At least one of input_ids, pixel_values, or mel_spectrogram must be provided.")

        x = torch.cat(parts, dim=1)
        T = x.shape[1]

        # Build causal mask
        if T > self._causal_mask.shape[-1]:
            mask = torch.tril(torch.ones(T, T, device=x.device)).unsqueeze(0).unsqueeze(0)
        else:
            mask = self._causal_mask[:, :, :T, :T]

        if attention_mask is not None:
            # Pad attention_mask to match full sequence length
            prefix_len = T - attention_mask.shape[1]
            if prefix_len > 0:
                prefix_mask = torch.ones(
                    attention_mask.shape[0], prefix_len,
                    device=attention_mask.device, dtype=attention_mask.dtype,
                )
                full_mask = torch.cat([prefix_mask, attention_mask], dim=1)
            else:
                full_mask = attention_mask
            pad_mask = full_mask.unsqueeze(1).unsqueeze(2).float()
            mask = mask * pad_mask

        for blk in self.thinker["blocks"]:
            x = blk(x, mask)

        x = self.thinker["norm"](x)
        logits = self.thinker["lm_head"](x)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = nn.CrossEntropyLoss(ignore_index=-100)(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )

        from transformers.modeling_outputs import CausalLMOutputWithPast
        return CausalLMOutputWithPast(loss=loss, logits=logits)

    # ---- Custom loading from model_full.safetensors ----

    @classmethod
    def from_pretrained_safetensors(
        cls,
        model_dir: str,
        config: Optional[MuOmniMultimodalConfig] = None,
        safetensors_file: str = "model_full.safetensors",
        device: str = "cpu",
        **kwargs,
    ):
        """
        Load the full multimodal model from a single safetensors file.

        Args:
            model_dir: directory containing the safetensors file (and optionally config.json)
            config: model config; if None, attempts to load from model_dir/config.json
            safetensors_file: filename of the merged safetensors file
            device: device to load onto
        Returns:
            MuOmniMultimodalModel with loaded weights
        """
        import json
        from pathlib import Path
        from safetensors.torch import load_file

        model_dir = Path(model_dir)

        # Load or create config
        if config is None:
            config_path = model_dir / "config.json"
            if config_path.exists():
                with open(config_path, "r") as f:
                    cfg_dict = json.load(f)
                config = MuOmniMultimodalConfig(**{
                    k: v for k, v in cfg_dict.items()
                    if k in MuOmniMultimodalConfig().__dict__
                })
            else:
                config = MuOmniMultimodalConfig()

        model = cls(config)

        # Load safetensors
        sf_path = model_dir / safetensors_file
        if not sf_path.exists():
            raise FileNotFoundError(f"Safetensors file not found: {sf_path}")

        state_dict = load_file(str(sf_path), device=device)

        # Map prefixed keys to model submodule state dicts
        _load_prefix(model.thinker, state_dict, "thinker.")
        _load_prefix(model.audio_encoder, state_dict, "audio_encoder.")
        _load_prefix(model.vision_encoder, state_dict, "vision_encoder.")
        _load_prefix(model.talker, state_dict, "talker.")
        _load_prefix(model.rvq, state_dict, "rvq.")
        _load_prefix(model.proj_a, state_dict, "proj_a.")
        _load_prefix(model.proj_v, state_dict, "proj_v.")

        model.eval()
        return model


def _load_prefix(module: nn.Module, state_dict: Dict[str, torch.Tensor], prefix: str):
    """Load weights from *state_dict* whose keys start with *prefix* into *module*."""
    sub = {k[len(prefix):]: v for k, v in state_dict.items() if k.startswith(prefix)}
    if sub:
        missing, unexpected = module.load_state_dict(sub, strict=False)
        if missing:
            print(f"[{prefix.rstrip('.')}] missing keys: {missing}")
        if unexpected:
            print(f"[{prefix.rstrip('.')}] unexpected keys: {unexpected}")
