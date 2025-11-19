"""
Calculate exact parameter counts for μOmni models based on config files.
Uses mathematical formulas instead of instantiating models.

All formulas are based on:
1. Standard transformer architecture (Vaswani et al., 2017)
2. Actual implementation in omni/*.py modules
3. PyTorch default parameter configurations (bias usage)

Formula References:
- Transformer: Vaswani et al. "Attention is All You Need" (2017)
- SwiGLU: Shazeer "GLU Variants Improve Transformer" (2020)
- GQA: Ainslie et al. "GQA: Training Generalized Multi-Query Transformer Models" (2023)
- ViT: Dosovitskiy et al. "An Image is Worth 16x16 Words" (2020)
- MoE: Shazeer et al. "Outrageously Large Neural Networks" (2017)

Standard Parameter Formulas:
- Linear layer: (in_dim × out_dim) + bias (if bias=True)
- Conv2d: (in_ch × out_ch × kernel_h × kernel_w) + bias (if bias=True)
- Embedding: vocab_size × embed_dim
- RMSNorm: embed_dim (weight only, no bias)
- LayerNorm: 2 × embed_dim (weight + bias)
"""

import json
import os

def format_size(num_params):
    """Format parameter count in readable format"""
    if num_params >= 1_000_000_000:
        return f"{num_params / 1_000_000_000:.2f}B"
    elif num_params >= 1_000_000:
        return f"{num_params / 1_000_000:.2f}M"
    elif num_params >= 1_000:
        return f"{num_params / 1_000:.2f}K"
    else:
        return str(num_params)

def calculate_moe_params(d_model, d_ff, num_experts, use_swiglu=True):
    """
    Calculate MoE (Mixture of Experts) parameters.
    
    Based on actual implementation in omni/thinker.py:
    - Router: d_model → num_experts (bias=False)
    - Experts: num_experts × MLP(d_model, d_ff, use_swiglu)
    """
    params = 0
    
    # Router: d_model → num_experts (bias=False)
    params += d_model * num_experts
    
    # Each expert is an MLP
    if use_swiglu:
        # SwiGLU: gate, up, down (all bias=False)
        expert_params = d_model * d_ff + d_model * d_ff + d_ff * d_model
    else:
        # Standard MLP: fc1 (bias=True), fc2 (bias=True)
        expert_params = (d_model * d_ff + d_ff) + (d_ff * d_model + d_model)
    
    # num_experts experts
    params += expert_params * num_experts
    
    return params

def calculate_thinker_params(vocab_size, n_layers, d_model, n_heads, d_ff, use_gqa=False, kv_groups=2, use_swiglu=True, use_moe=False, num_experts=8):
    """
    Calculate Thinker parameters using standard transformer formulas.
    
    Based on actual implementation in omni/thinker.py:
    - Attention: Q, K, V, O projections (all bias=False)
    - MLP: SwiGLU (gate, up, down, all bias=False) or standard (fc1 with bias, fc2 with bias)
    - MoE: Optional Mixture of Experts (replaces MLP when enabled)
    - RMSNorm: d_model params per layer (weight only, no bias)
    
    Formula references:
    - Standard transformer: Vaswani et al. "Attention is All You Need" (2017)
    - SwiGLU: Shazeer "GLU Variants Improve Transformer" (2020)
    - GQA: Ainslie et al. "GQA: Training Generalized Multi-Query Transformer Models" (2023)
    """
    params = 0
    
    # Token embeddings: vocab_size × d_model
    params += vocab_size * d_model
    
    # Per transformer block
    # Attention: QKV projections (all bias=False in implementation)
    if use_gqa:
        # GQA: Q has n_heads, KV has kv_groups
        d_head = d_model // n_heads
        q_params = d_model * (n_heads * d_head)  # Q projection: d_model → n_heads * d_head
        kv_params = d_model * (kv_groups * d_head) * 2  # K and V: d_model → kv_groups * d_head each
        out_params = d_model * d_model  # Output projection: d_model → d_model
        attn_params = q_params + kv_params + out_params
    else:
        # Standard MHA: Q, K, V combined projection, then output
        # Implementation uses qkv = nn.Linear(d, 3*d, bias=False)
        qkv_params = d_model * (3 * d_model)  # QKV combined: d_model → 3*d_model
        out_params = d_model * d_model  # Output: d_model → d_model
        attn_params = qkv_params + out_params
    
    # MLP or MoE
    if use_moe:
        # MoE: Router + num_experts × MLP
        mlp_params = calculate_moe_params(d_model, d_ff, num_experts, use_swiglu)
    elif use_swiglu:
        # SwiGLU: gate_proj, up_proj, down_proj (all bias=False in implementation)
        # gate: d_model → d_ff, up: d_model → d_ff, down: d_ff → d_model
        mlp_params = d_model * d_ff + d_model * d_ff + d_ff * d_model  # gate + up + down
    else:
        # Standard MLP: fc1 (with bias), fc2 (with bias)
        # fc1: d_model → d_ff (bias=True), fc2: d_ff → d_model (bias=True)
        mlp_params = (d_model * d_ff + d_ff) + (d_ff * d_model + d_model)  # fc1 + bias + fc2 + bias
    
    # Normalization (RMSNorm: weight only, d_model params per layer)
    norm_params = d_model * 2  # 2 RMSNorm layers per block (pre-attn, pre-mlp)
    
    # Per block total
    block_params = attn_params + mlp_params + norm_params
    
    # All blocks
    params += block_params * n_layers
    
    # Final norm (RMSNorm)
    params += d_model
    
    # Output head (LM head: bias=False in implementation)
    params += d_model * vocab_size
    
    return params

def calculate_audio_encoder_params(d_model, n_layers, n_heads, d_ff, downsample_factor=8):
    """
    Calculate Audio Encoder parameters using standard formulas.
    
    Based on actual implementation in omni/audio_encoder.py:
    - ConvDown: 2 conv layers (bias=True by default)
    - Projection: Linear with bias (default)
    - Attention: QKV projection (bias=True), output (bias=True)
    - MLP: fc1 (bias=True), fc2 (bias=True)
    - RMSNorm: weight only (d_model params per layer)
    """
    params = 0
    
    # ConvDown layers (for 4x or 8x downsample)
    # Conv1: 1 -> 64, kernel 3, stride 2 (bias=True by default)
    params += 1 * 64 * 3 * 3 + 64  # weights + bias
    # Conv2: 64 -> 64, kernel 3, stride 2 (bias=True by default)
    params += 64 * 64 * 3 * 3 + 64  # weights + bias
    # Extra conv for 8x: 64 -> 64, kernel 3, stride 2 (bias=True by default)
    if downsample_factor == 8:
        params += 64 * 64 * 3 * 3 + 64  # weights + bias
    
    # Projection: 64 * (128 // downsample_factor) -> d_model (bias=True by default)
    freq_dim = 64 * (128 // downsample_factor)
    params += freq_dim * d_model + d_model  # weights + bias
    
    # Transformer blocks
    # Attention: QKV projection (bias=True in implementation)
    # qkv_proj: d_model → 3*d_model (bias=True)
    qkv_params = d_model * (3 * d_model) + (3 * d_model)  # weights + bias
    # out_proj: d_model → d_model (bias=True)
    out_params = d_model * d_model + d_model  # weights + bias
    attn_params = qkv_params + out_params
    
    # MLP: fc1 (bias=True), fc2 (bias=True)
    # fc1: d_model → d_ff (bias=True)
    # fc2: d_ff → d_model (bias=True)
    mlp_params = (d_model * d_ff + d_ff) + (d_ff * d_model + d_model)  # fc1 + bias + fc2 + bias
    
    # Normalization (RMSNorm: weight only, d_model params per layer)
    norm_params = d_model * 2  # 2 RMSNorm per block (pre-attn, pre-mlp)
    
    block_params = attn_params + mlp_params + norm_params
    params += block_params * n_layers
    
    # Final norm (RMSNorm)
    params += d_model
    
    return params

def calculate_vision_encoder_params(img_size, patch, d_model, n_layers, n_heads, d_ff):
    """
    Calculate Vision Encoder parameters using standard ViT formulas.
    
    Based on actual implementation in omni/vision_encoder.py:
    - Uses nn.TransformerEncoderLayer which has:
      - Attention: Q, K, V, output projections (bias=True by default in PyTorch)
      - MLP: fc1 (bias=True), fc2 (bias=True)
      - LayerNorm: weight + bias (2*d_model per layer)
    - Final norm: RMSNorm (weight only, d_model params)
    """
    params = 0
    
    # Patch embedding: Conv2d(3, d_model, kernel=patch, stride=patch)
    # No bias in Conv2d patch embedding (default)
    params += 3 * d_model * patch * patch
    
    # CLS token (learnable parameter)
    params += d_model
    
    # Position embeddings (learnable parameters)
    num_patches = (img_size // patch) ** 2
    params += (1 + num_patches) * d_model  # CLS + patches
    
    # Transformer blocks (using nn.TransformerEncoderLayer)
    # PyTorch TransformerEncoderLayer uses:
    # - Self-attention: Q, K, V, output projections (all with bias=True by default)
    # - MLP: fc1 (bias=True), fc2 (bias=True)
    # - LayerNorm: weight + bias (2*d_model per layer, 2 layers per block)
    
    # Attention: Q, K, V, output (all with bias in PyTorch default)
    # Standard implementation: d_model → d_model for each of Q, K, V, output
    # But PyTorch uses MultiheadAttention which internally handles this
    # Approximate: 4 * (d_model * d_model + d_model) for Q, K, V, output with bias
    attn_params = 4 * (d_model * d_model + d_model)  # Q, K, V, output (weights + bias each)
    
    # MLP: fc1 (bias=True), fc2 (bias=True)
    mlp_params = (d_model * d_ff + d_ff) + (d_ff * d_model + d_model)  # fc1 + bias + fc2 + bias
    
    # Normalization: LayerNorm has weight + bias (2*d_model per layer)
    # 2 LayerNorm per block (pre-attn, pre-mlp)
    norm_params = 2 * (d_model + d_model)  # 2 LayerNorm × (weight + bias)
    
    block_params = attn_params + mlp_params + norm_params
    params += block_params * n_layers
    
    # Final norm (RMSNorm: weight only)
    params += d_model
    
    return params

def calculate_talker_params(d_model, n_layers, n_heads, d_ff, codebooks, codebook_size, use_gqa=False, kv_groups=1, use_swiglu=True):
    """
    Calculate Talker parameters using standard formulas.
    
    Based on actual implementation in omni/talker.py:
    - Uses same Block architecture as Thinker (bias=False for attention/MLP)
    - Output heads: base_head and res_head (bias=True by default)
    """
    params = 0
    
    # Codebook embeddings: codebook_size × d_model
    params += codebook_size * d_model
    
    # Start token (learnable parameter)
    params += d_model
    
    # Transformer blocks (same as Thinker - all bias=False for attention/MLP)
    if use_gqa:
        d_head = d_model // n_heads
        q_params = d_model * (n_heads * d_head)  # Q projection
        kv_params = d_model * (kv_groups * d_head) * 2  # K and V
        out_params = d_model * d_model  # Output projection
        attn_params = q_params + kv_params + out_params
    else:
        # Standard MHA: QKV combined + output
        qkv_params = d_model * (3 * d_model)  # QKV combined
        out_params = d_model * d_model  # Output
        attn_params = qkv_params + out_params
    
    if use_swiglu:
        # SwiGLU: gate, up, down (all bias=False)
        mlp_params = d_model * d_ff + d_model * d_ff + d_ff * d_model
    else:
        # Standard MLP: fc1 (bias=True), fc2 (bias=True)
        mlp_params = (d_model * d_ff + d_ff) + (d_ff * d_model + d_model)
    
    norm_params = d_model * 2  # 2 RMSNorm per block
    block_params = attn_params + mlp_params + norm_params
    params += block_params * n_layers
    
    # Final norm (RMSNorm)
    params += d_model
    
    # Output heads (bias=True by default in nn.Linear)
    params += (d_model * codebook_size + codebook_size)  # base_head (weights + bias)
    params += (d_model * codebook_size + codebook_size)  # res_head (weights + bias)
    
    return params

def calculate_codec_params(codebooks, codebook_size, dim):
    """Calculate RVQ Codec parameters"""
    params = 0
    # Each codebook: codebook_size vectors of dimension dim
    params += codebooks * codebook_size * dim
    return params

def calculate_ocr_params(img_size, patch, vision_d_model, vision_layers, vision_heads, vision_d_ff,
                        decoder_d_model, decoder_layers, decoder_heads, decoder_d_ff, vocab_size):
    """
    Calculate OCR model parameters using standard formulas.
    
    Based on actual implementation in omni/ocr_model.py:
    - Vision encoder: ViT (same as vision encoder)
    - Decoder: Self-attention (Q, K, V), Cross-attention (MultiheadAttention), MLP (SwiGLU)
    - Output head: bias=False
    - Image projection: vision_d_model → decoder_d_model (bias=True by default)
    """
    params = 0
    
    # Vision encoder (ViT) - same as vision encoder
    params += calculate_vision_encoder_params(img_size, patch, vision_d_model, vision_layers, vision_heads, vision_d_ff)
    
    # Text decoder
    # Character embeddings: vocab_size × decoder_d_model
    params += vocab_size * decoder_d_model
    
    # Decoder layers
    # Self-attention: Q, K, V projections (bias=True by default in nn.Linear)
    # Implementation uses separate q_proj, k_proj, v_proj
    q_params = decoder_d_model * decoder_d_model + decoder_d_model  # Q (weights + bias)
    k_params = decoder_d_model * decoder_d_model + decoder_d_model  # K (weights + bias)
    v_params = decoder_d_model * decoder_d_model + decoder_d_model  # V (weights + bias)
    attn_params = q_params + k_params + v_params
    
    # Cross-attention: MultiheadAttention (Q, K, V + output projection, all with bias)
    # PyTorch MultiheadAttention: 4 projections (Q, K, V, output) with bias
    cross_attn_params = 4 * (decoder_d_model * decoder_d_model + decoder_d_model)  # Q, K, V, output
    
    # MLP (SwiGLU): gate, up, down (bias=True by default in nn.Linear)
    gate_params = decoder_d_model * decoder_d_ff + decoder_d_ff  # gate (weights + bias)
    up_params = decoder_d_model * decoder_d_ff + decoder_d_ff  # up (weights + bias)
    down_params = decoder_d_ff * decoder_d_model + decoder_d_model  # down (weights + bias)
    mlp_params = gate_params + up_params + down_params
    
    # Normalization (RMSNorm: weight only, d_model params per layer)
    # 3 RMSNorm per block (self-attn, cross-attn, MLP)
    norm_params = decoder_d_model * 3
    
    block_params = attn_params + cross_attn_params + mlp_params + norm_params
    params += block_params * decoder_layers
    
    # Final norm (RMSNorm)
    params += decoder_d_model
    
    # Output head (bias=False in implementation)
    params += decoder_d_model * vocab_size
    
    # Image feature projection (vision_d_model -> decoder_d_model, bias=True by default)
    params += vision_d_model * decoder_d_model + decoder_d_model  # weights + bias
    
    return params

def calculate_model_sizes():
    """Calculate sizes for all models based on config files"""
    
    print("="*60)
    print("μOmni Model Size Calculator")
    print("="*60)
    
    total_params = 0
    results = {}
    
    # 1. Thinker (LLM)
    print("\n1. Thinker (Decoder-Only LLM)")
    print("-" * 60)
    thinker_cfg_path = "configs/thinker_tiny.json"
    if os.path.exists(thinker_cfg_path):
        with open(thinker_cfg_path, 'r') as f:
            cfg = json.load(f)
        
        params = calculate_thinker_params(
            vocab_size=cfg["vocab_size"],
            n_layers=cfg["n_layers"],
            d_model=cfg["d_model"],
            n_heads=cfg["n_heads"],
            d_ff=cfg["d_ff"],
            use_gqa=cfg.get("use_gqa", False),
            kv_groups=cfg.get("kv_groups", 2),
            use_swiglu=cfg.get("use_swiglu", True),
            use_moe=cfg.get("use_moe", False),
            num_experts=cfg.get("num_experts", 8)
        )
        total_params += params
        results["thinker"] = params
        print(f"  Config: {cfg['n_layers']} layers, d_model={cfg['d_model']}, n_heads={cfg['n_heads']}, d_ff={cfg['d_ff']}")
        print(f"  Vocab size: {cfg['vocab_size']:,}")
        print(f"  GQA: {cfg.get('use_gqa', False)}, SwiGLU: {cfg.get('use_swiglu', True)}, MoE: {cfg.get('use_moe', False)}")
        if cfg.get('use_moe', False):
            print(f"  MoE experts: {cfg.get('num_experts', 8)}")
        print(f"  Parameters: {params:,} ({format_size(params)})")
    else:
        print("  Config file not found")
    
    # 2. Audio Encoder
    print("\n2. Audio Encoder (AuT-Tiny)")
    print("-" * 60)
    audio_cfg_path = "configs/audio_enc_tiny.json"
    if os.path.exists(audio_cfg_path):
        with open(audio_cfg_path, 'r') as f:
            cfg = json.load(f)
        
        params = calculate_audio_encoder_params(
            d_model=cfg["d_model"],
            n_layers=cfg["n_layers"],
            n_heads=cfg["n_heads"],
            d_ff=cfg["d_ff"],
            downsample_factor=cfg.get("downsample_time", 8)
        )
        total_params += params
        results["audio_encoder"] = params
        print(f"  Config: {cfg['n_layers']} layers, d_model={cfg['d_model']}, n_heads={cfg['n_heads']}, d_ff={cfg['d_ff']}")
        print(f"  Downsample factor: {cfg.get('downsample_time', 8)}x")
        print(f"  Parameters: {params:,} ({format_size(params)})")
    else:
        print("  Config file not found")
    
    # 3. Vision Encoder
    print("\n3. Vision Encoder (ViT-Tiny)")
    print("-" * 60)
    vision_cfg_path = "configs/vision_tiny.json"
    if os.path.exists(vision_cfg_path):
        with open(vision_cfg_path, 'r') as f:
            cfg = json.load(f)
        
        params = calculate_vision_encoder_params(
            img_size=cfg["img_size"],
            patch=cfg["patch"],
            d_model=cfg["d_model"],
            n_layers=cfg["n_layers"],
            n_heads=cfg["n_heads"],
            d_ff=cfg["d_ff"]
        )
        total_params += params
        results["vision_encoder"] = params
        print(f"  Config: {cfg['n_layers']} layers, d_model={cfg['d_model']}, n_heads={cfg['n_heads']}, d_ff={cfg['d_ff']}")
        print(f"  Image size: {cfg['img_size']}×{cfg['img_size']}, patch size: {cfg['patch']}")
        print(f"  Parameters: {params:,} ({format_size(params)})")
    else:
        print("  Config file not found")
    
    # 4. Talker
    print("\n4. Talker (Speech Code Predictor)")
    print("-" * 60)
    talker_cfg_path = "configs/talker_tiny.json"
    if os.path.exists(talker_cfg_path):
        with open(talker_cfg_path, 'r') as f:
            cfg = json.load(f)
        
        params = calculate_talker_params(
            d_model=cfg["d_model"],
            n_layers=cfg["n_layers"],
            n_heads=cfg["n_heads"],
            d_ff=cfg["d_ff"],
            codebooks=cfg["codebooks"],
            codebook_size=cfg["codebook_size"],
            use_gqa=cfg.get("use_gqa", False),
            kv_groups=cfg.get("kv_groups", 1),
            use_swiglu=cfg.get("use_swiglu", True)
        )
        total_params += params
        results["talker"] = params
        print(f"  Config: {cfg['n_layers']} layers, d_model={cfg['d_model']}, n_heads={cfg['n_heads']}, d_ff={cfg['d_ff']}")
        print(f"  Codebooks: {cfg['codebooks']}, codebook_size: {cfg['codebook_size']}")
        print(f"  GQA: {cfg.get('use_gqa', False)}, SwiGLU: {cfg.get('use_swiglu', True)}")
        print(f"  Parameters: {params:,} ({format_size(params)})")
    else:
        print("  Config file not found")
    
    # 5. RVQ Codec
    print("\n5. RVQ Codec")
    print("-" * 60)
    params = calculate_codec_params(codebooks=2, codebook_size=128, dim=192)
    total_params += params
    results["codec"] = params
    print(f"  Codebooks: 2, codebook_size: 128, dim: 192")
    print(f"  Parameters: {params:,} ({format_size(params)})")
    
    # 6. OCR Model
    print("\n6. OCR Model (Vision Encoder + Text Decoder)")
    print("-" * 60)
    ocr_cfg_path = "configs/ocr_tiny.json"
    if os.path.exists(ocr_cfg_path):
        with open(ocr_cfg_path, 'r') as f:
            cfg = json.load(f)
        
        params = calculate_ocr_params(
            img_size=cfg.get("img_size", 224),
            patch=cfg.get("patch", 16),
            vision_d_model=cfg.get("vision_d_model", 128),
            vision_layers=cfg.get("vision_layers", 4),
            vision_heads=cfg.get("vision_heads", 2),
            vision_d_ff=cfg.get("vision_d_ff", 512),
            decoder_d_model=cfg.get("decoder_d_model", 256),
            decoder_layers=cfg.get("decoder_layers", 4),
            decoder_heads=cfg.get("decoder_heads", 4),
            decoder_d_ff=cfg.get("decoder_d_ff", 1024),
            vocab_size=cfg.get("vocab_size", 128)
        )
        total_params += params
        results["ocr"] = params
        print(f"  Vision: {cfg.get('vision_layers', 4)} layers, d_model={cfg.get('vision_d_model', 128)}")
        print(f"  Decoder: {cfg.get('decoder_layers', 4)} layers, d_model={cfg.get('decoder_d_model', 256)}")
        print(f"  Vocab size: {cfg.get('vocab_size', 128)}")
        print(f"  Parameters: {params:,} ({format_size(params)})")
    else:
        print("  Config file not found (optional)")
    
    # 7. Projectors (for SFT)
    print("\n7. Projectors (Vision & Audio)")
    print("-" * 60)
    # Vision projector: 128 → 256
    vision_proj_params = 128 * 256 + 256  # Linear with bias
    # Audio projector: 192 → 256
    audio_proj_params = 192 * 256 + 256  # Linear with bias
    projector_params = vision_proj_params + audio_proj_params
    total_params += projector_params
    results["projectors"] = projector_params
    print(f"  Vision projector: 128 → 256")
    print(f"  Audio projector: 192 → 256")
    print(f"  Total projector parameters: {projector_params:,} ({format_size(projector_params)})")
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"{'Component':<20} {'Parameters':>15} {'Percentage':>12}")
    print("-" * 60)
    
    for name, params in results.items():
        percentage = (params / total_params) * 100
        print(f"{name.replace('_', ' ').title():<20} {params:>15,} {percentage:>11.1f}%")
    
    print("-" * 60)
    print(f"{'TOTAL':<20} {total_params:>15,} ({format_size(total_params)}) {'100.0%':>12}")
    
    # Memory size (assuming float32)
    memory_mb = (total_params * 4) / (1024 * 1024)  # 4 bytes per float32
    memory_gb = memory_mb / 1024
    
    print("\n" + "="*60)
    print("MEMORY REQUIREMENTS (float32)")
    print("="*60)
    print(f"Model size: {memory_mb:.2f} MB ({memory_gb:.2f} GB)")
    print(f"With optimizer (AdamW): ~{memory_mb * 3:.2f} MB ({memory_gb * 3:.2f} GB)")
    print(f"With gradients: ~{memory_mb * 2:.2f} MB ({memory_gb * 2:.2f} GB)")
    print(f"Total training (model + optimizer + gradients): ~{memory_mb * 6:.2f} MB ({memory_gb * 6:.2f} GB)")
    
    print("\n" + "="*60)
    print("COMPARISON")
    print("="*60)
    print(f"μOmni:        {format_size(total_params):>8} parameters")
    print(f"BERT-base:    ~110M parameters")
    print(f"GPT-2 small: ~124M parameters")
    print(f"LLaMA-7B:     7B parameters ({7_000_000_000 / total_params:.0f}x larger)")
    print(f"GPT-3:        175B parameters ({175_000_000_000 / total_params:.0f}x larger)")
    
    return results, total_params

if __name__ == "__main__":
    try:
        results, total = calculate_model_sizes()
    except Exception as e:
        print(f"\nError calculating model sizes: {e}")
        import traceback
        traceback.print_exc()
