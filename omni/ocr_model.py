
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
from omni.utils import RMSNorm


class OCRDecoder(nn.Module):
    """
    Text decoder for OCR - generates text tokens from visual features.
    Similar to Talker but processes image features instead of speech codes.
    """
    def __init__(self, d_model: int = 256, n_layers: int = 4, n_heads: int = 4,
                 d_ff: int = 1024, vocab_size: int = 128, dropout: float = 0.1,
                 max_seq_len: int = 256, use_gqa: bool = False, kv_groups: int = 1,
                 use_swiglu: bool = True, rope_theta: float = 10000.0,
                 compile_model: bool = False) -> None:
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
            compile_model: Use torch.compile()
        """
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        
        # Character embedding
        self.char_embed = nn.Embedding(vocab_size, d_model)
        
        # Positional encoding (RoPE)
        self.rope_theta = rope_theta
        
        # Decoder layers
        self.layers = nn.ModuleList([
            self._make_decoder_layer(d_model, n_heads, d_ff, dropout, use_gqa, kv_groups, use_swiglu)
            for _ in range(n_layers)
        ])
        
        # Output projection
        self.norm = RMSNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        
        # Cross-attention to image features
        self.cross_attn_layers = nn.ModuleList([
            nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
            for _ in range(n_layers)
        ])
        
        # Image feature projection (from vision encoder output)
        self.img_proj = nn.Linear(128, d_model)  # ViT outputs 128-dim, project to d_model
        
        # Compilation support
        self._compiled = False
        if compile_model:
            self._apply_compilation()
    
    def _make_decoder_layer(self, d_model, n_heads, d_ff, dropout, use_gqa, kv_groups, use_swiglu):
        """Create a decoder layer with self-attention and cross-attention."""
        layers = []
        
        # Self-attention
        if use_gqa:
            # Grouped Query Attention
            kv_heads = n_heads // kv_groups
            q_proj = nn.Linear(d_model, d_model)
            k_proj = nn.Linear(d_model, d_model // kv_groups)
            v_proj = nn.Linear(d_model, d_model // kv_groups)
            layers.append(('q_proj', q_proj))
            layers.append(('k_proj', k_proj))
            layers.append(('v_proj', v_proj))
        else:
            q_proj = nn.Linear(d_model, d_model)
            k_proj = nn.Linear(d_model, d_model)
            v_proj = nn.Linear(d_model, d_model)
            layers.append(('q_proj', q_proj))
            layers.append(('k_proj', k_proj))
            layers.append(('v_proj', v_proj))
        
        # Feedforward
        if use_swiglu:
            gate_proj = nn.Linear(d_model, d_ff)
            up_proj = nn.Linear(d_model, d_ff)
            down_proj = nn.Linear(d_ff, d_model)
            layers.append(('gate_proj', gate_proj))
            layers.append(('up_proj', up_proj))
            layers.append(('down_proj', down_proj))
        else:
            ffn = nn.Sequential(
                nn.Linear(d_model, d_ff),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_ff, d_model),
                nn.Dropout(dropout)
            )
            layers.append(('ffn', ffn))
        
        return nn.ModuleDict(layers)
    
    def _apply_compilation(self) -> None:
        """Apply torch.compile() for speedup."""
        if not hasattr(torch, 'compile'):
            return
        try:
            for i, layer in enumerate(self.layers):
                self.layers[i] = torch.compile(layer, backend='cudagraphs', mode='default')
            self.head = torch.compile(self.head, backend='cudagraphs', mode='default')
            self._compiled = True
        except Exception:
            pass
    
    def _apply_rope(self, x: torch.Tensor, pos: int) -> torch.Tensor:
        """Apply Rotary Position Embedding (RoPE)."""
        # Simplified RoPE - can be enhanced
        # For now, just return x (positional info from embeddings)
        return x
    
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
        
        # Process through decoder layers
        for i, layer in enumerate(self.layers):
            # Self-attention (causal)
            residual = x
            x = self.norm(x)  # Use shared norm instance
            
            # Simplified self-attention (can use proper causal mask)
            q = layer['q_proj'](x)
            k = layer['k_proj'](x)
            v = layer['v_proj'](x)
            
            # Causal mask
            mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
            attn_out, _ = nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=mask, is_causal=True
            )
            x = residual + attn_out
            
            # Cross-attention to image features
            residual = x
            x = self.norm(x)  # Use shared norm instance
            cross_out, _ = self.cross_attn_layers[i](x, img_proj, img_proj)
            x = residual + cross_out
            
            # Feedforward
            residual = x
            x = self.norm(x)  # Use shared norm instance
            if 'gate_proj' in layer:
                # SwiGLU
                gate = layer['gate_proj'](x)
                up = layer['up_proj'](x)
                ffn_out = layer['down_proj'](torch.nn.functional.silu(gate) * up)
            else:
                ffn_out = layer['ffn'](x)
            x = residual + ffn_out
        
        # Output projection
        x = self.norm(x)
        logits = self.head(x)  # (B, T, vocab_size)
        
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
                 compile_model: bool = False) -> None:
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

