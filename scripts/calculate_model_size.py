"""
Calculate exact parameter counts for μOmni models based on config files
Uses mathematical formulas instead of instantiating models
"""

import json
import os
from pathlib import Path

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

def calculate_thinker_params(vocab_size, n_layers, d_model, n_heads, d_ff, use_gqa=False, kv_groups=2, use_swiglu=True):
    """Calculate Thinker parameters"""
    params = 0
    
    # Token embeddings
    params += vocab_size * d_model
    
    # Per transformer block
    # Attention: QKV projections
    if use_gqa:
        # GQA: Q has n_heads, KV has kv_groups
        q_params = d_model * d_model  # Q projection
        kv_params = d_model * (d_model // n_heads * kv_groups) * 2  # K and V
        attn_params = q_params + kv_params + d_model * d_model  # output projection
    else:
        # Standard: Q, K, V each have n_heads
        attn_params = d_model * d_model * 4  # Q, K, V, output
    
    # MLP
    if use_swiglu:
        # SwiGLU: gate_proj, up_proj, down_proj
        mlp_params = d_model * d_ff * 3  # gate, up, down
    else:
        # Standard MLP: fc1, fc2
        mlp_params = d_model * d_ff * 2 + d_ff  # fc1 (with bias), fc2
    
    # Normalization (RMSNorm has d_model params)
    norm_params = d_model * 2  # 2 RMSNorm layers per block
    
    # Per block total
    block_params = attn_params + mlp_params + norm_params
    
    # All blocks
    params += block_params * n_layers
    
    # Final norm
    params += d_model
    
    # Output head (LM head)
    params += d_model * vocab_size
    
    return params

def calculate_audio_encoder_params(d_model, n_layers, n_heads, d_ff, downsample_factor=8):
    """Calculate Audio Encoder parameters"""
    params = 0
    
    # ConvDown layers (for 8x downsample)
    # Conv1: 1 -> 64, kernel 3, stride 2
    params += 1 * 64 * 3 * 3  # weights
    params += 64  # bias
    # Conv2: 64 -> 64, kernel 3, stride 2
    params += 64 * 64 * 3 * 3
    params += 64
    # Extra conv for 8x: 64 -> 64, kernel 3, stride 2
    if downsample_factor == 8:
        params += 64 * 64 * 3 * 3
        params += 64
    
    # Projection: 64 * (128 // downsample_factor) -> d_model
    freq_dim = 64 * (128 // downsample_factor)
    params += freq_dim * d_model
    params += d_model  # bias
    
    # Transformer blocks
    # Attention: QKV projection
    attn_params = d_model * d_model * 4  # Q, K, V, output (all with bias)
    # MLP
    mlp_params = d_model * d_ff + d_ff  # fc1 with bias
    mlp_params += d_ff * d_model  # fc2
    # Normalization
    norm_params = d_model * 2  # 2 RMSNorm per block
    
    block_params = attn_params + mlp_params + norm_params
    params += block_params * n_layers
    
    # Final norm
    params += d_model
    
    return params

def calculate_vision_encoder_params(img_size, patch, d_model, n_layers, n_heads, d_ff):
    """Calculate Vision Encoder parameters"""
    params = 0
    
    # Patch embedding: Conv2d(3, d_model, kernel=patch, stride=patch)
    params += 3 * d_model * patch * patch
    
    # CLS token
    params += d_model
    
    # Position embeddings
    num_patches = (img_size // patch) ** 2
    params += (1 + num_patches) * d_model
    
    # Transformer blocks (using TransformerEncoderLayer)
    # Attention: QKV projection
    attn_params = d_model * d_model * 4  # Q, K, V, output
    # MLP
    mlp_params = d_model * d_ff + d_ff  # fc1 with bias
    mlp_params += d_ff * d_model  # fc2
    # Normalization (LayerNorm has 2*d_model params)
    norm_params = d_model * 4  # 2 LayerNorm per block (weight + bias each)
    
    block_params = attn_params + mlp_params + norm_params
    params += block_params * n_layers
    
    # Final norm (RMSNorm)
    params += d_model
    
    return params

def calculate_talker_params(d_model, n_layers, n_heads, d_ff, codebooks, codebook_size, use_gqa=False, kv_groups=1, use_swiglu=True):
    """Calculate Talker parameters"""
    params = 0
    
    # Codebook embeddings
    params += codebook_size * d_model
    
    # Start token
    params += d_model
    
    # Transformer blocks (same as Thinker)
    if use_gqa:
        q_params = d_model * d_model
        kv_params = d_model * (d_model // n_heads * kv_groups) * 2
        attn_params = q_params + kv_params + d_model * d_model
    else:
        attn_params = d_model * d_model * 4
    
    if use_swiglu:
        mlp_params = d_model * d_ff * 3
    else:
        mlp_params = d_model * d_ff * 2 + d_ff
    
    norm_params = d_model * 2
    block_params = attn_params + mlp_params + norm_params
    params += block_params * n_layers
    
    # Final norm
    params += d_model
    
    # Output heads
    params += d_model * codebook_size  # base_head
    params += d_model * codebook_size  # res_head
    
    return params

def calculate_codec_params(codebooks, codebook_size, dim):
    """Calculate RVQ Codec parameters"""
    params = 0
    # Each codebook: codebook_size vectors of dimension dim
    params += codebooks * codebook_size * dim
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
            use_swiglu=cfg.get("use_swiglu", True)
        )
        total_params += params
        results["thinker"] = params
        print(f"  Config: {cfg['n_layers']} layers, d_model={cfg['d_model']}, n_heads={cfg['n_heads']}, d_ff={cfg['d_ff']}")
        print(f"  Vocab size: {cfg['vocab_size']:,}")
        print(f"  GQA: {cfg.get('use_gqa', False)}, SwiGLU: {cfg.get('use_swiglu', True)}")
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
    
    # 6. Projectors (for SFT)
    print("\n6. Projectors (Vision & Audio)")
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
