#!/usr/bin/env python3
"""
Learning Rate Finder Utility

Automatically discovers optimal learning rate for your training run using the
LR range test method from "Cyclical Learning Rates for Training Neural Networks" (Smith, 2017).

Usage:
    python find_lr.py --config configs/vision_tiny.json --start_lr 1e-7 --end_lr 1 --num_iter 100
    python find_lr.py --config configs/audio_enc_tiny.json --plot lr_finder.png
    python find_lr.py --config configs/talker_tiny.json --num_iter 200

The suggested learning rate will be printed and optionally plotted.
You can then use this LR in your main training config.
"""

import argparse
import json
import os
import sys
import torch
from torch import nn
from torch.utils.data import DataLoader

# Import all model architectures and utilities
from omni.vision_encoder import ViTTiny
from omni.audio_encoder import AudioEncoderTiny
from omni.talker import TalkerTiny
from omni.codec import RVQ
from omni.ocr_model import OCRModel
from omni.thinker import ThinkerLM
from omni.tokenizer import BPETokenizer
from omni.utils import (
    set_seed, ImgCapDataset, ASRDataset, TTSDataset, OCRDataset, TextDataset,
    collate_mel_text_fn, collate_mel_fn, LRFinder
)


def load_config(config_path):
    """Load configuration from JSON file."""
    with open(config_path, 'r') as f:
        return json.load(f)


def setup_vision_model(cfg, device):
    """Setup vision encoder model and data."""
    d_model = cfg.get("d_model", 128)
    vit = ViTTiny(
        cfg.get("img_size", 224), 
        cfg.get("patch", 16), 
        d_model, 
        cfg.get("n_layers", 4), 
        cfg.get("n_heads", 2), 
        cfg.get("d_ff", 512), 
        cfg.get("dropout", 0.1),
        compile_model=False
    ).to(device)
    
    embed_dim = cfg.get("embed_dim", d_model)
    img_proj = nn.Sequential(
        nn.Linear(d_model, embed_dim),
        nn.LayerNorm(embed_dim)
    ).to(device)
    text_proj = nn.Sequential(
        nn.Linear(d_model, embed_dim),
        nn.LayerNorm(embed_dim)
    ).to(device)
    
    # Initialize projections with smaller weights
    for module in [img_proj, text_proj]:
        for m in module.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    # Simple embedding for text (avoid loading full Thinker for LR finder)
    text_embed = nn.Embedding(cfg.get("vocab_size", 32000), d_model).to(device)
    
    model = nn.ModuleDict({
        'vit': vit,
        'img_proj': img_proj,
        'text_proj': text_proj,
        'text_embed': text_embed
    })
    
    # Dataset
    train_manifest = cfg.get("train_manifest", "data/images/production_annotations.json")
    image_root = cfg.get("image_root", "data/images")
    dataset = ImgCapDataset(
        train_manifest, image_root, cfg.get("img_size", 224),
        shuffle_buffer_size=100, seed=42, skip_samples=0
    )
    
    dataloader = DataLoader(
        dataset, 
        batch_size=cfg.get("batch_size", 8), 
        shuffle=False, 
        num_workers=2
    )
    
    # Loss function
    temperature = cfg.get("temperature", 0.07)
    
    def criterion(outputs, targets):
        # Contrastive loss placeholder (simplified for LR finder)
        # In real training, this would be InfoNCE
        return nn.functional.mse_loss(outputs, targets)
    
    return model, dataloader, criterion


def setup_audio_enc_model(cfg, device):
    """Setup audio encoder model and data."""
    d_model = cfg.get("d_model", 192)
    model = AudioEncoderTiny(
        d_model,
        cfg.get("n_heads", 3),
        cfg.get("d_ff", 768),
        cfg.get("n_layers", 4),
        cfg.get("dropout", 0.1),
        downsample_factor=cfg.get("downsample_time", 8),
        compile_model=False
    ).to(device)
    
    head = nn.Linear(d_model, cfg.get("ctc_vocab_size", 100)).to(device)
    
    combined = nn.ModuleDict({'encoder': model, 'head': head})
    
    # Dataset
    train_csv = cfg.get("train_csv", "data/audio/production_asr.csv")
    dataset = ASRDataset(
        train_csv,
        cfg.get("sample_rate", 16000),
        cfg.get("mel_bins", 128),
        shuffle_buffer_size=100,
        seed=42,
        skip_samples=0
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.get("batch_size", 4),
        shuffle=False,
        num_workers=2,
        collate_fn=collate_mel_text_fn
    )
    
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    
    return combined, dataloader, criterion


def setup_talker_model(cfg, device):
    """Setup talker (TTS) model and data."""
    codebooks = cfg.get("codebooks", 2)
    codebook_size = cfg.get("codebook_size", 128)
    
    rvq = RVQ(codebooks, codebook_size, d=64, compile_model=False).to(device)
    talker = TalkerTiny(
        cfg.get("d_model", 384),
        cfg.get("n_layers", 8),
        cfg.get("n_heads", 6),
        cfg.get("d_ff", 1536),
        codebooks,
        codebook_size,
        cfg.get("dropout", 0.1),
        use_gqa=cfg.get("use_gqa", False),
        use_swiglu=cfg.get("use_swiglu", True),
        rope_theta=cfg.get("rope_theta", 10000.0),
        compile_model=False
    ).to(device)
    
    combined = nn.ModuleDict({'rvq': rvq, 'talker': talker})
    
    # Dataset
    train_csv = cfg.get("train_csv", "data/audio/production_tts.csv")
    dataset = TTSDataset(
        train_csv,
        cfg.get("sample_rate", 16000),
        cfg.get("n_mels", 128),
        cfg.get("frame_ms", 80),
        shuffle_buffer_size=100,
        seed=42,
        skip_samples=0
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.get("batch_size", 4),
        shuffle=False,
        num_workers=2,
        collate_fn=collate_mel_fn
    )
    
    criterion = nn.CrossEntropyLoss(reduction='none')
    
    return combined, dataloader, criterion


def setup_ocr_model(cfg, device):
    """Setup OCR model and data."""
    model = OCRModel(
        cfg.get("d_model", 192),
        cfg.get("n_heads", 3),
        cfg.get("d_ff", 768),
        cfg.get("n_layers", 4),
        cfg.get("dropout", 0.1),
        cfg.get("img_height", 64),
        cfg.get("img_width", 256),
        cfg.get("patch_height", 8),
        cfg.get("patch_width", 8),
        cfg.get("vocab_size", 100),
        compile_model=False
    ).to(device)
    
    # OCRModel has built-in head, no separate head needed
    
    # Dataset
    train_csv = cfg.get("train_csv", "data/ocr/production_ocr.csv")
    dataset = OCRDataset(
        train_csv,
        cfg.get("img_height", 64),
        cfg.get("img_width", 256),
        shuffle_buffer_size=100,
        seed=42,
        skip_samples=0
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.get("batch_size", 4),
        shuffle=False,
        num_workers=2
    )
    
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    
    return model, dataloader, criterion


def setup_text_model(cfg, device):
    """Setup text (Thinker LM) model and data."""
    # Load or create tokenizer
    save_dir = cfg.get("save_dir", "checkpoints/thinker_tiny")
    tok_model_path = os.path.join(save_dir, "tokenizer.model")
    
    if not os.path.exists(tok_model_path):
        print(f"Warning: Tokenizer not found at {tok_model_path}")
        print("LR Finder needs an existing tokenizer. Please train the text model first or provide a tokenizer.")
        sys.exit(1)
    
    tok = BPETokenizer(tok_model_path)
    vocab_size = len(tok.sp)
    
    model = ThinkerLM(
        vocab_size,
        cfg.get("n_layers", 4),
        cfg.get("d_model", 256),
        cfg.get("n_heads", 4),
        cfg.get("d_ff", 1024),
        cfg.get("dropout", 0.1),
        cfg.get("rope_theta", 10000),
        cfg.get("ctx_len", 512),
        use_gqa=cfg.get("use_gqa", False),
        use_swiglu=cfg.get("use_swiglu", True),
        use_moe=cfg.get("use_moe", False),
        num_experts=cfg.get("num_experts", 8),
        num_experts_per_tok=cfg.get("num_experts_per_tok", 2),
        compile_model=False
    ).to(device)
    
    # Dataset
    train_csv = cfg.get("train_csv", "data/text/production_text.csv")
    ctx_len = cfg.get("ctx_len", 512)
    dataset = TextDataset(
        train_csv,
        tok,
        ctx_len,
        shuffle_buffer_size=100,
        seed=42,
        skip_samples=0
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.get("batch_size", 4),
        shuffle=False,
        num_workers=2
    )
    
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    
    return model, dataloader, criterion


def detect_model_type(config_path):
    """Detect which model type based on config filename."""
    basename = os.path.basename(config_path).lower()
    
    if 'vision' in basename:
        return 'vision'
    elif 'audio_enc' in basename or 'audio' in basename:
        return 'audio_enc'
    elif 'talker' in basename or 'tts' in basename:
        return 'talker'
    elif 'ocr' in basename:
        return 'ocr'
    elif 'thinker' in basename or 'text' in basename:
        return 'text'
    else:
        raise ValueError(f"Cannot detect model type from config: {config_path}")


def main():
    parser = argparse.ArgumentParser(description="Learning Rate Finder")
    parser.add_argument('--config', type=str, required=True, help='Path to training config JSON')
    parser.add_argument('--start_lr', type=float, default=1e-7, help='Starting learning rate')
    parser.add_argument('--end_lr', type=float, default=1.0, help='Ending learning rate')
    parser.add_argument('--num_iter', type=int, default=100, help='Number of iterations to test')
    parser.add_argument('--smooth_f', type=float, default=0.05, help='Smoothing factor for loss')
    parser.add_argument('--plot', type=str, default=None, help='Path to save plot (optional)')
    parser.add_argument('--model_type', type=str, default=None, 
                       choices=['vision', 'audio_enc', 'talker', 'ocr', 'text'],
                       help='Model type (auto-detected if not specified)')
    
    args = parser.parse_args()
    
    # Load config
    cfg = load_config(args.config)
    
    # Detect or use specified model type
    model_type = args.model_type or detect_model_type(args.config)
    print(f"\n🔍 LR Finder for {model_type.upper()} model")
    print(f"   Config: {args.config}")
    print(f"   LR range: {args.start_lr:.2e} to {args.end_lr:.2e}")
    print(f"   Iterations: {args.num_iter}\n")
    
    # Set seed
    set_seed(cfg.get("seed", 42))
    
    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Setup model and data based on type
    setup_funcs = {
        'vision': setup_vision_model,
        'audio_enc': setup_audio_enc_model,
        'talker': setup_talker_model,
        'ocr': setup_ocr_model,
        'text': setup_text_model
    }
    
    if model_type not in setup_funcs:
        print(f"❌ Unsupported model type: {model_type}")
        sys.exit(1)
    
    print(f"\n📦 Loading {model_type} model and dataset...")
    model, dataloader, criterion = setup_funcs[model_type](cfg, device)
    
    # Create optimizer
    lr = cfg.get("lr", 3e-4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=cfg.get("wd", 0.01))
    
    # Run LR Finder
    print(f"\n🔬 Running LR Finder...\n")
    lr_finder = LRFinder(model, optimizer, criterion, device=device)
    
    try:
        lr_finder.range_test(
            dataloader, 
            start_lr=args.start_lr, 
            end_lr=args.end_lr, 
            num_iter=args.num_iter,
            smooth_f=args.smooth_f
        )
    except Exception as e:
        print(f"\n❌ LR Finder failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Get suggested LR
    print("\n" + "="*60)
    suggested_lr = lr_finder.suggest_lr()
    print("="*60)
    
    # Plot if requested
    if args.plot:
        lr_finder.plot(save_path=args.plot)
    
    # Print usage instructions
    print(f"\n📝 To use this learning rate in your training:")
    print(f"   Update '{args.config}' with:")
    print(f'   "lr": {suggested_lr:.2e}')
    print()


if __name__ == "__main__":
    main()
