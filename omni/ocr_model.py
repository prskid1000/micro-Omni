
"""
OCR (Optical Character Recognition) Model for extracting text from images.

Architecture:
- Vision Encoder (ViT): Processes image patches
- Text Decoder: Autoregressively generates text tokens from visual features
- Similar to ASR but for images instead of audio
"""

import torch
from torch import nn
from typing import Optional, Tuple
from omni.vision_encoder import ViTTiny
from omni.utils import RMSNorm, make_positions
from omni.thinker import Attention, MLP
import warnings

# Check for Flash Attention support (PyTorch 2.0+)
HAS_FLASH_ATTENTION = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
if HAS_FLASH_ATTENTION:
    from torch.nn.functional import scaled_dot_product_attention


class OCRDecoderBlock(nn.Module):
    """
    Decoder block with self-attention, cross-attention, and feedforward.
    Follows Thinker's Block pattern with separate norm instances.
    """
    def __init__(self, d: int, heads: int, ff: int, rope_theta: float, dropout: float,
                 use_gqa: bool = False, use_swiglu: bool = True, use_flash: bool = True) -> None:
        super().__init__()
        # Separate norm instances for each sub-layer (like Thinker's Block)
        self.norm1 = RMSNorm(d)  # For self-attention
        self.norm2 = RMSNorm(d)  # For cross-attention
        self.norm3 = RMSNorm(d)  # For feedforward
        
        # Self-attention with RoPE (causal)
        self.self_attn = Attention(d, heads, rope_theta=rope_theta, dropout=dropout,
                                   use_gqa=use_gqa, use_flash=use_flash)
        
        # Cross-attention to image features (no RoPE needed for cross-attention)
        self.cross_attn = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)
        
        # Feedforward network
        if use_swiglu:
            self.mlp = MLP(d, ff, use_swiglu=True)
        else:
            self.mlp = MLP(d, ff, use_swiglu=False)
        
        self.drop = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, img_features: torch.Tensor,
                mask: Optional[torch.Tensor] = None,
                pos: Optional[torch.Tensor] = None,
                cache: Optional[dict] = None,
                return_cache: bool = False) -> Tuple[torch.Tensor, Optional[dict]]:
        """
        Forward pass through decoder block.
        
        Args:
            x: (B, T, D) text token embeddings
            img_features: (B, N, D) image features (already projected)
            mask: Optional causal mask for self-attention
            pos: Optional position indices for RoPE
            cache: Optional KV cache for self-attention
            return_cache: Whether to return updated cache
        
        Returns:
            x: (B, T, D) output
            cache: Updated cache (if return_cache=True)
        """
        # Self-attention (causal) with RoPE
        attn_out, cache = self.self_attn(self.norm1(x), mask=mask, pos=pos,
                                         cache=cache, need_cache=return_cache)
        x = x + self.drop(attn_out)
        
        # Cross-attention to image features (no RoPE, no causal mask)
        residual = x
        x_norm = self.norm2(x)
        cross_out, _ = self.cross_attn(x_norm, img_features, img_features)
        x = residual + self.drop(cross_out)
        
        # Feedforward
        x = x + self.drop(self.mlp(self.norm3(x)))
        
        return x, cache


class OCRDecoder(nn.Module):
    """
    Text decoder for OCR - generates text tokens from visual features.
    Similar to Talker but processes image features instead of speech codes.
    """
    def __init__(self, d_model: int = 256, n_layers: int = 4, n_heads: int = 4,
                 d_ff: int = 1024, vocab_size: int = 128, dropout: float = 0.1,
                 max_seq_len: int = 256, use_gqa: bool = False, kv_groups: int = 1,
                 use_swiglu: bool = True, rope_theta: float = 10000.0,
                 use_flash: bool = True, compile_model: bool = False) -> None:
        """
        Initialize OCR decoder.
        
        Args:
            d_model: Model dimension
            n_layers: Number of decoder layers
            n_heads: Number of attention heads
            d_ff: Feedforward dimension
            vocab_size: Character vocabulary size
            dropout: Dropout rate
            max_seq_len: Maximum sequence length
            use_gqa: Use Grouped Query Attention
            kv_groups: Number of KV groups for GQA
            use_swiglu: Use SwiGLU activation
            rope_theta: RoPE theta parameter
            use_flash: Use Flash Attention for speedup
            compile_model: Use torch.compile()
        """
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        
        # Character embedding
        self.char_embed = nn.Embedding(vocab_size, d_model)
        
        # Decoder blocks (each with separate norms)
        self.blocks = nn.ModuleList([
            OCRDecoderBlock(d_model, n_heads, d_ff, rope_theta, dropout,
                           use_gqa=use_gqa, use_swiglu=use_swiglu, use_flash=use_flash)
            for _ in range(n_layers)
        ])
        
        # Output projection
        self.norm = RMSNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        
        # Image feature projection (from vision encoder output)
        self.img_proj = nn.Linear(128, d_model)  # ViT outputs 128-dim, project to d_model
        
        # KV cache for autoregressive generation
        self.kv_cache = None
        self.use_kv_cache = False
        
        # Compilation support
        self._compiled = False
        if compile_model:
            self._apply_compilation()
    
    def _apply_compilation(self) -> None:
        """Apply torch.compile() for speedup."""
        if not hasattr(torch, 'compile'):
            warnings.warn("torch.compile() not available. Requires PyTorch 2.0+. Skipping compilation.")
            return
        try:
            for i, block in enumerate(self.blocks):
                self.blocks[i] = torch.compile(block, backend='cudagraphs', mode='default')
            self.head = torch.compile(self.head, backend='cudagraphs', mode='default')
            self._compiled = True
            print(f"✓ OCRDecoder compiled successfully with torch.compile()")
        except Exception as e:
            warnings.warn(f"Failed to compile OCRDecoder: {e}. Continuing without compilation.")
    
    def reset_kv_cache(self) -> None:
        """Reset KV cache (call before new generation)"""
        self.kv_cache = None
    
    def enable_kv_cache(self, enable: bool = True) -> None:
        """Enable/disable KV caching for faster autoregressive generation"""
        self.use_kv_cache = enable
        if not enable:
            self.kv_cache = None
    
    def forward(self, text_ids: torch.Tensor, img_features: torch.Tensor,
                use_cache: bool = False) -> torch.Tensor:
        """
        Forward pass: decode text from image features.
        
        Args:
            text_ids: (B, T) character token IDs (for teacher forcing)
            img_features: (B, N, 128) image features from vision encoder
            use_cache: Enable KV caching (for inference)
        
        Returns:
            logits: (B, T, vocab_size) character prediction logits
        """
        B, T = text_ids.shape
        
        # Project image features to decoder dimension
        img_proj = self.img_proj(img_features)  # (B, N, d_model)
        
        # Character embeddings
        x = self.char_embed(text_ids)  # (B, T, d_model)
        
        # Prepare position indices for RoPE
        pos = make_positions(T, x.device)
        
        # Prepare causal mask for self-attention
        mask = torch.tril(torch.ones(T, T, device=x.device)).unsqueeze(0).unsqueeze(0)  # (1, 1, T, T)
        
        # KV caching setup
        use_kv = use_cache and self.use_kv_cache
        if use_kv and self.kv_cache is None:
            self.kv_cache = [None] * len(self.blocks)
        
        cache_ready = use_kv and self.kv_cache[0] is not None
        
        if use_kv and cache_ready:
            # Incremental decoding: only process new tokens
            pos = torch.tensor([self.kv_cache[0]['pos']], device=x.device, dtype=torch.long)
            for i, block in enumerate(self.blocks):
                x, new_cache = block(x, img_proj, mask=None, pos=pos,
                                    cache=self.kv_cache[i], return_cache=True)
                self.kv_cache[i] = new_cache
        else:
            # Full forward pass
            for i, block in enumerate(self.blocks):
                x, new_cache = block(x, img_proj, mask=mask, pos=pos,
                                    cache=None, return_cache=use_kv)
                if use_kv:
                    self.kv_cache[i] = new_cache
        
        # Output projection
        x = self.norm(x)
        logits = self.head(x)  # (B, T, vocab_size)
        
        # Check for numerical stability
        if torch.isnan(logits).any() or torch.isinf(logits).any():
            nan_count = torch.isnan(logits).sum().item()
            inf_count = torch.isinf(logits).sum().item()
            raise RuntimeError(f"Numerical instability in OCRDecoder: NaN={nan_count}, Inf={inf_count}")
        
        return logits


class OCRModel(nn.Module):
    """
    Complete OCR model: Vision Encoder + Text Decoder.
    """
    def __init__(self, img_size: int = 224, patch: int = 16,
                 vision_d_model: int = 128, vision_layers: int = 4,
                 vision_heads: int = 2, vision_d_ff: int = 512,
                 decoder_d_model: int = 256, decoder_layers: int = 4,
                 decoder_heads: int = 4, decoder_d_ff: int = 1024,
                 vocab_size: int = 128, dropout: float = 0.1,
                 use_gqa: bool = False, use_swiglu: bool = True,
                 use_flash: bool = True, compile_model: bool = False) -> None:
        """
        Initialize complete OCR model.
        
        Args:
            img_size: Input image size
            patch: Patch size for ViT
            vision_d_model: Vision encoder dimension
            vision_layers: Vision encoder layers
            vision_heads: Vision encoder attention heads
            vision_d_ff: Vision encoder FFN dimension
            decoder_d_model: Text decoder dimension
            decoder_layers: Text decoder layers
            decoder_heads: Text decoder attention heads
            decoder_d_ff: Text decoder FFN dimension
            vocab_size: Character vocabulary size
            dropout: Dropout rate
            use_gqa: Use Grouped Query Attention in decoder
            use_swiglu: Use SwiGLU activation in decoder
            use_flash: Use Flash Attention in decoder
            compile_model: Use torch.compile()
        """
        super().__init__()
        
        # Vision encoder (ViT)
        self.vision_encoder = ViTTiny(
            img_size=img_size,
            patch=patch,
            d=vision_d_model,
            layers=vision_layers,
            heads=vision_heads,
            ff=vision_d_ff,
            dropout=dropout,
            compile_model=compile_model
        )
        
        # Text decoder
        self.decoder = OCRDecoder(
            d_model=decoder_d_model,
            n_layers=decoder_layers,
            n_heads=decoder_heads,
            d_ff=decoder_d_ff,
            vocab_size=vocab_size,
            dropout=dropout,
            use_gqa=use_gqa,
            use_swiglu=use_swiglu,
            use_flash=use_flash,
            compile_model=compile_model
        )
    
    def forward(self, image: torch.Tensor, text_ids: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: extract text from image.
        
        Args:
            image: (B, 3, H, W) input image
            text_ids: (B, T) target text token IDs (for teacher forcing)
        
        Returns:
            logits: (B, T, vocab_size) character prediction logits
        """
        # Encode image
        cls, grid = self.vision_encoder(image)  # cls: (B, 1, 128), grid: (B, N, 128)
        
        # Use grid features (spatial patches) for OCR (better for text detection)
        # Can also use CLS token or combine both
        img_features = grid  # (B, N, 128)
        
        # Decode text from image features
        logits = self.decoder(text_ids, img_features)  # (B, T, vocab_size)
        
        return logits
