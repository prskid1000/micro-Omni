"""
Complete test script to validate Vision Encoder accuracy.
Measures image-text alignment, retrieval metrics, and embedding quality.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import os
import argparse
import random
import numpy as np
from PIL import Image
from torchvision import transforms
from omni.vision_encoder import ViTTiny, TransformerTextEncoder
from omni.utils import ImgCapDataset, find_checkpoint, strip_orig_mod
from tqdm import tqdm

# Import your custom tokenizer and Thinker model
from omni.tokenizer import BPETokenizer
from omni.thinker import ThinkerLM

torch.set_float32_matmul_precision('high')


def load_model_and_head(checkpoint_dir, device="cuda"):
    """Load Vision Encoder model, projection heads, and text encoder from checkpoint."""
    checkpoint_path, checkpoint = find_checkpoint(checkpoint_dir, "vision.pt", "vision_step_", device)
    if checkpoint is None:
        raise FileNotFoundError(f"Checkpoint not found in: {checkpoint_dir}")
    
    print(f"Loading checkpoint from: {checkpoint_path}")
    
    # Get config from checkpoint or load from config file
    if "config" in checkpoint:
        cfg = checkpoint["config"]
    else:
        # Load config from JSON file based on checkpoint directory
        checkpoint_name = os.path.basename(checkpoint_dir)
        config_path = f"configs/{checkpoint_name}.json"
        
        if os.path.exists(config_path):
            print(f"Loading config from: {config_path}")
            with open(config_path, 'r') as f:
                cfg = json.load(f)
        else:
            raise FileNotFoundError(
                f"Config not found in checkpoint and config file not found: {config_path}"
            )
    
    d_model = cfg.get("d_model", 128)
    embed_dim = cfg.get("embed_dim", d_model)
    
    # Initialize vision model
    model = ViTTiny(
        img_size=cfg.get("img_size", 224),
        patch=cfg.get("patch", 16),
        d=d_model,
        layers=cfg.get("n_layers", 4),
        heads=cfg.get("n_heads", 2),
        ff=cfg.get("d_ff", 512),
        dropout=cfg.get("dropout", 0.1),
        compile_model=False
    ).to(device)
    
    # Load model weights
    if "vit" in checkpoint:
        state_dict = checkpoint["vit"]
    elif "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint
    
    state_dict = strip_orig_mod(state_dict)
    model.load_state_dict(state_dict, strict=False)
    
    # Load image projection head (matches training script structure)
    img_proj = nn.Sequential(
        nn.Linear(d_model, embed_dim),
        nn.Dropout(cfg.get("dropout", 0.1)),
        nn.LayerNorm(embed_dim)
    ).to(device)
    if "img_proj" in checkpoint:
        img_proj_state = checkpoint["img_proj"]
        img_proj_state = strip_orig_mod(img_proj_state)
        img_proj.load_state_dict(img_proj_state, strict=False)
        print("✓ Loaded image projection head")
    
    # Load text projection head (matches training script structure)
    text_proj = nn.Sequential(
        nn.Linear(d_model, embed_dim),
        nn.Dropout(cfg.get("dropout", 0.1)),
        nn.LayerNorm(embed_dim)
    ).to(device)
    if "text_proj" in checkpoint:
        text_proj_state = checkpoint["text_proj"]
        text_proj_state = strip_orig_mod(text_proj_state)
        text_proj.load_state_dict(text_proj_state, strict=False)
        print("✓ Loaded text projection head")
    
    # Load tokenizer
    use_thinker = cfg.get("use_thinker_for_text", True)
    thinker_ckpt_dir = cfg.get("thinker_ckpt", "checkpoints/thinker_tiny")
    tok_model_path = os.path.join(thinker_ckpt_dir, "tokenizer.model")
    
    if not os.path.exists(tok_model_path):
        # Try checkpoint directory as fallback
        tok_model_path = os.path.join(checkpoint_dir, "tokenizer.model")
    
    if not os.path.exists(tok_model_path):
        print(f"⚠️  Warning: Tokenizer not found at {tok_model_path}")
        print("   Text encoding will not be available for retrieval metrics")
        tok = None
        text_encoder = None
    else:
        tok = BPETokenizer(tok_model_path)
        vocab_size = tok.sp.get_piece_size()
        print(f"✓ Loaded tokenizer (vocab size: {vocab_size})")
        
        # Load text encoder (Thinker or embedding layer)
        if use_thinker:
            # Load Thinker model for text encoding
            thinker_cfg = cfg.get("thinker", {})
            thinker_d_model = thinker_cfg.get("d_model", 256)
            ctx_len = cfg.get("ctx_len", 512)
            
            text_encoder = ThinkerLM(
                thinker_cfg.get("vocab_size", vocab_size),
                thinker_cfg.get("n_layers", 4),
                thinker_d_model,
                thinker_cfg.get("n_heads", 4),
                thinker_cfg.get("d_ff", 1024),
                thinker_cfg.get("dropout", 0.1),
                thinker_cfg.get("rope_theta", 10000),
                ctx_len,
                use_gqa=thinker_cfg.get("use_gqa", False),
                use_swiglu=thinker_cfg.get("use_swiglu", True),
                use_moe=thinker_cfg.get("use_moe", False),
                num_experts=thinker_cfg.get("num_experts", 8),
                num_experts_per_tok=thinker_cfg.get("num_experts_per_tok", 2),
                compile_model=False,
                use_spiking=thinker_cfg.get("use_spiking", False),
                use_ltc=thinker_cfg.get("use_ltc", False)
            ).to(device)
            
            # Load Thinker weights
            thinker_path, thinker_ckpt = find_checkpoint(thinker_ckpt_dir, "thinker.pt", "thinker_step_", device)
            if thinker_ckpt is not None:
                if isinstance(thinker_ckpt, dict):
                    if "model" in thinker_ckpt:
                        text_encoder.load_state_dict(thinker_ckpt["model"])
                    elif "thinker" in thinker_ckpt:
                        text_encoder.load_state_dict(thinker_ckpt["thinker"])
                    else:
                        text_encoder.load_state_dict(thinker_ckpt)
                else:
                    text_encoder.load_state_dict(thinker_ckpt)
                print(f"✓ Loaded Thinker text encoder from {thinker_path}")
            else:
                print("⚠️  Warning: Thinker checkpoint not found")
            
            text_encoder.eval()
            # Update text_proj input dimension to match Thinker output
            text_proj = nn.Sequential(
                nn.Linear(thinker_d_model, embed_dim),
                nn.Dropout(cfg.get("dropout", 0.1)),
                nn.LayerNorm(embed_dim)
            ).to(device)
            if "text_proj" in checkpoint:
                text_proj_state = checkpoint["text_proj"]
                text_proj_state = strip_orig_mod(text_proj_state)
                text_proj.load_state_dict(text_proj_state, strict=False)
        else:
            # Use TransformerTextEncoder (CLIP-style)
            ctx_len = cfg.get("text_max_len", 77)
            text_encoder = TransformerTextEncoder(
                vocab_size, 
                d_model=cfg.get("text_d_model", d_model), 
                n_layers=cfg.get("text_n_layers", 6),
                n_heads=cfg.get("text_n_heads", 8),
                d_ff=cfg.get("text_d_ff", 2048),
                max_len=ctx_len,
                dropout=cfg.get("dropout", 0.1)
            ).to(device)
            # Try loading from both text_encoder and text_embed (backward compatibility)
            if "text_encoder" in checkpoint:
                text_encoder.load_state_dict(checkpoint["text_encoder"])
            elif "text_embed" in checkpoint:
                # Try to load old text_embed weights (only embedding layer)
                try:
                    text_encoder.embedding.load_state_dict(checkpoint["text_embed"])
                except:
                    pass
            text_encoder.eval()
            print(f"✓ Using TransformerTextEncoder for text encoding")
    
    model.eval()
    img_proj.eval()
    text_proj.eval()
    
    print("✓ Model loaded successfully")
    print(f"  Image size: {cfg.get('img_size', 224)}")
    print(f"  Model dimension: {d_model}")
    print(f"  Embedding dimension: {embed_dim}")
    
    return model, img_proj, text_proj, text_encoder, tok, cfg


def encode_caption(caption, tok, text_encoder, cfg, device="cuda", use_thinker=True):
    """Encode caption using tokenizer and either Thinker model or TransformerTextEncoder"""
    ctx_len = cfg.get("text_max_len", 77)  # CLIP standard context length
    
    # Tokenize caption
    token_ids = tok.encode(caption)
    # Truncate to context length
    token_ids = token_ids[:ctx_len-1]  # -1 for BOS/CLS token
    
    # Add BOS/CLS token at the beginning (token ID 1)
    token_ids = [1] + token_ids  # BOS/CLS=1
    if len(token_ids) == 0:
        token_ids = [1]  # At least BOS/CLS token
    
    # Convert to tensor
    token_tensor = torch.tensor(token_ids, device=device, dtype=torch.long)
    
    if use_thinker:
        # Use Thinker model for contextual embeddings (better quality)
        token_tensor = token_tensor.unsqueeze(0)  # (1, T)
        with torch.inference_mode():
            # Use Thinker to get contextual embeddings
            text_emb = text_encoder(idx=token_tensor)  # (1, T, thinker_d_model)
        # Use mean pooling for Thinker (final token pooling for TransformerTextEncoder)
        return text_emb.squeeze(0).mean(dim=0)  # (thinker_d_model,)
    else:
        # Use TransformerTextEncoder (CLIP-style)
        with torch.inference_mode():
            text_emb = text_encoder(token_tensor, return_cls=True)  # (d_model,)
        return text_emb


def compute_cosine_similarity(img_embeds, text_embeds):
    """Compute cosine similarity between image and text embeddings."""
    img_embeds = F.normalize(img_embeds, dim=-1)
    text_embeds = F.normalize(text_embeds, dim=-1)
    return (img_embeds * text_embeds).sum(dim=-1)


def compute_retrieval_metrics(image_embeds, text_embeds, batch_size):
    """
    Compute image-to-text and text-to-image retrieval metrics.
    
    Args:
        image_embeds: (N, D) tensor of image embeddings
        text_embeds: (N, D) tensor of text embeddings
        batch_size: number of samples
    
    Returns:
        Dictionary with R@1, R@5, R@10 for both directions
    """
    # Normalize embeddings
    image_embeds = F.normalize(image_embeds, dim=-1)
    text_embeds = F.normalize(text_embeds, dim=-1)
    
    # Compute similarity matrix (N x N)
    similarity = image_embeds @ text_embeds.T  # (N, N)
    
    # Image-to-Text Retrieval (for each image, find matching text)
    i2t_ranks = []
    for i in range(batch_size):
        # Get similarities for this image to all texts
        sims = similarity[i]  # (N,)
        # Rank by similarity (descending)
        ranking = torch.argsort(sims, descending=True)
        # Find where the correct match (index i) is in the ranking
        rank = (ranking == i).nonzero(as_tuple=True)[0].item()
        i2t_ranks.append(rank)
    
    # Text-to-Image Retrieval (for each text, find matching image)
    t2i_ranks = []
    for i in range(batch_size):
        # Get similarities for this text to all images
        sims = similarity[:, i]  # (N,)
        # Rank by similarity (descending)
        ranking = torch.argsort(sims, descending=True)
        # Find where the correct match (index i) is in the ranking
        rank = (ranking == i).nonzero(as_tuple=True)[0].item()
        t2i_ranks.append(rank)
    
    # Compute recall@k
    i2t_r1 = sum(1 for r in i2t_ranks if r < 1) / batch_size
    i2t_r5 = sum(1 for r in i2t_ranks if r < 5) / batch_size
    i2t_r10 = sum(1 for r in i2t_ranks if r < 10) / batch_size
    
    t2i_r1 = sum(1 for r in t2i_ranks if r < 1) / batch_size
    t2i_r5 = sum(1 for r in t2i_ranks if r < 5) / batch_size
    t2i_r10 = sum(1 for r in t2i_ranks if r < 10) / batch_size
    
    return {
        'i2t_r1': i2t_r1,
        'i2t_r5': i2t_r5,
        'i2t_r10': i2t_r10,
        't2i_r1': t2i_r1,
        't2i_r5': t2i_r5,
        't2i_r10': t2i_r10,
        'avg_i2t_rank': np.mean(i2t_ranks),
        'avg_t2i_rank': np.mean(t2i_ranks),
    }

def evaluate_embedding_quality(model, proj_head, cfg, tok, device="cuda", num_samples=100, verbose=True):
    """
    Evaluate vision encoder embedding quality.
    
    Metrics:
    - Embedding statistics (norm, std)
    - Intra-class consistency (if captions are similar)
    - Embedding diversity
    """
    model.eval()
    if proj_head is not None:
        proj_head.eval()
    
    manifest_path = cfg.get("train_manifest", "data/images/production_annotations.json")
    image_root = cfg.get("image_root", "data/images")
    
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Image manifest not found: {manifest_path}")
    
    # Create dataset (requires tokenizer and ctx_len for caption tokenization)
    ctx_len = cfg.get("ctx_len", cfg.get("text_max_len", 77))
    if tok is None:
        raise ValueError("Tokenizer required for ImgCapDataset. Ensure tokenizer.model exists in thinker checkpoint.")
    dataset = ImgCapDataset(
        manifest=manifest_path,
        image_root=image_root,
        tokenizer=tok,
        ctx_len=ctx_len,
        img_size=cfg.get("img_size", 224),
        shuffle_buffer_size=10000,
        seed=42,  # Fixed seed for reproducibility
        skip_samples=0
    )
    
    all_cls_embeds = []
    all_grid_embeds = []
    all_captions = []
    
    if verbose:
        print(f"\nExtracting embeddings from {num_samples} samples...")
        iterator = tqdm(dataset, total=num_samples, desc="Processing")
    else:
        iterator = iter(dataset)
    
    with torch.inference_mode():
        for i, (img_tensor, caption) in enumerate(iterator):
            if i >= num_samples:
                break
            
            try:
                img_tensor = img_tensor.unsqueeze(0).to(device)
                
                # Forward pass
                cls, grid = model(img_tensor)  # cls: (1, 1, d), grid: (1, num_patches, d)
                
                # Apply projection if available (squeeze sequence dim: (1, 1, d) -> (1, d))
                if proj_head is not None:
                    cls = proj_head(cls.squeeze(1))
                else:
                    cls = cls.squeeze(1)
                
                all_cls_embeds.append(cls.cpu())
                all_grid_embeds.append(grid.cpu())
                all_captions.append(caption)
                
            except Exception as e:
                if verbose:
                    print(f"\n⚠️  Error processing sample {i}: {e}")
                continue
    
    if len(all_cls_embeds) == 0:
        raise ValueError("No valid samples processed!")
    
    # Stack embeddings
    cls_embeds = torch.cat(all_cls_embeds, dim=0)  # (N, d)
    
    # Compute statistics
    cls_norm_mean = cls_embeds.norm(dim=-1).mean().item()
    cls_norm_std = cls_embeds.norm(dim=-1).std().item()
    cls_std_mean = cls_embeds.std(dim=0).mean().item()
    
    # Compute embedding diversity (average pairwise distance)
    # Sample pairs to avoid O(N^2) computation
    num_pairs = min(1000, len(cls_embeds) * (len(cls_embeds) - 1) // 2)
    if num_pairs > 0:
        indices = torch.randperm(len(cls_embeds))[:min(100, len(cls_embeds))]
        sample_embeds = cls_embeds[indices]
        # Normalize for cosine similarity
        sample_embeds_norm = F.normalize(sample_embeds, dim=-1)
        # Compute pairwise similarities (N x d) @ (d x N) = (N x N)
        pairwise_sim = sample_embeds_norm @ sample_embeds_norm.t()
        # Get upper triangle (excluding diagonal)
        mask = torch.triu(torch.ones_like(pairwise_sim), diagonal=1).bool()
        similarities = pairwise_sim[mask]
        avg_similarity = similarities.mean().item()
        diversity_score = 1.0 - avg_similarity  # Higher is more diverse
    else:
        diversity_score = 0.0
    
    # Check for embedding collapse (all embeddings very similar)
    collapse_threshold = 0.95
    is_collapsed = avg_similarity > collapse_threshold if num_pairs > 0 else False
    
    return {
        'num_samples': len(cls_embeds),
        'cls_norm_mean': cls_norm_mean,
        'cls_norm_std': cls_norm_std,
        'cls_std_mean': cls_std_mean,
        'diversity_score': diversity_score,
        'avg_pairwise_similarity': avg_similarity if num_pairs > 0 else 0.0,
        'is_collapsed': is_collapsed,
        'embeddings': cls_embeds,
        'captions': all_captions,
    }


def evaluate_retrieval(model, img_proj, text_proj, text_encoder, tok, cfg, device="cuda", num_samples=100, verbose=True):
    """
    Evaluate image-text retrieval performance using the trained text encoder.
    """
    model.eval()
    if img_proj is not None:
        img_proj.eval()
    if text_proj is not None:
        text_proj.eval()
    if text_encoder is not None:
        text_encoder.eval()
    
    manifest_path = cfg.get("train_manifest", "data/images/production_annotations.json")
    image_root = cfg.get("image_root", "data/images")
    
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Image manifest not found: {manifest_path}")
    
    # Create dataset (requires tokenizer and ctx_len for caption tokenization)
    ctx_len = cfg.get("ctx_len", cfg.get("text_max_len", 77))
    if tok is None:
        raise ValueError("Tokenizer required for ImgCapDataset. Ensure tokenizer.model exists in thinker checkpoint.")
    dataset = ImgCapDataset(
        manifest=manifest_path,
        image_root=image_root,
        tokenizer=tok,
        ctx_len=ctx_len,
        img_size=cfg.get("img_size", 224),
        shuffle_buffer_size=10000,
        seed=42,
        skip_samples=0
    )
    
    all_img_embeds = []
    all_text_embeds = []
    
    if verbose:
        print(f"\nEvaluating retrieval on {num_samples} samples...")
        iterator = tqdm(dataset, total=num_samples, desc="Processing")
    else:
        iterator = iter(dataset)
    
    use_thinker = cfg.get("use_thinker_for_text", True)
    
    # Check if text encoder is available
    if text_encoder is None or tok is None:
        print("⚠️  Warning: Text encoder or tokenizer not available. Cannot evaluate retrieval.")
        return None
    
    with torch.inference_mode():
        for i, (img_tensor, caption) in enumerate(iterator):
            if i >= num_samples:
                break
            
            try:
                img_tensor = img_tensor.unsqueeze(0).to(device)
                
                # Encode image
                cls, _ = model(img_tensor)
                if img_proj is not None:
                    cls = img_proj(cls.squeeze(1))  # Remove sequence dim: (1, 1, d) -> (1, d)
                
                # Encode text using the trained text encoder
                text_emb = encode_caption(caption, tok, text_encoder, cfg, device, use_thinker)
                if text_proj is not None:
                    text_emb = text_proj(text_emb.unsqueeze(0))  # (d,) -> (1, d)
                else:
                    text_emb = text_emb.unsqueeze(0)  # (d,) -> (1, d)
                
                all_img_embeds.append(cls.cpu())
                all_text_embeds.append(text_emb.cpu())
                
            except Exception as e:
                if verbose:
                    print(f"\n⚠️  Error processing sample {i}: {e}")
                continue
    
    if len(all_img_embeds) == 0:
        raise ValueError("No valid samples processed!")
    
    # Stack embeddings
    img_embeds = torch.cat(all_img_embeds, dim=0)  # (N, d)
    text_embeds = torch.cat(all_text_embeds, dim=0)  # (N, d)
    
    # Compute retrieval metrics
    retrieval_metrics = compute_retrieval_metrics(img_embeds, text_embeds, len(img_embeds))
    
    return retrieval_metrics


def print_results(embedding_metrics, retrieval_metrics=None):
    """Pretty print evaluation results."""
    print(f"\n{'='*70}")
    print("VISION ENCODER EVALUATION RESULTS")
    print(f"{'='*70}")
    
    print(f"\nEMBEDDING QUALITY METRICS:")
    print(f"  Samples Evaluated: {embedding_metrics['num_samples']}")
    print(f"  CLS Norm Mean: {embedding_metrics['cls_norm_mean']:.4f}")
    print(f"  CLS Norm Std: {embedding_metrics['cls_norm_std']:.4f}")
    print(f"  CLS Feature Std: {embedding_metrics['cls_std_mean']:.4f}")
    print(f"  Diversity Score: {embedding_metrics['diversity_score']:.4f}")
    print(f"  Avg Pairwise Similarity: {embedding_metrics['avg_pairwise_similarity']:.4f}")
    
    # Check for collapse
    if embedding_metrics['is_collapsed']:
        print(f"\n  ⚠️  WARNING: Embeddings may be collapsed (very similar)")
    else:
        print(f"\n  ✓ Embeddings are diverse (good)")
    
    # Interpretation
    print(f"\n{'='*70}")
    print("INTERPRETATION:")
    print(f"{'='*70}")
    
    # Check embedding quality
    norm_mean = embedding_metrics['cls_norm_mean']
    if 5.0 < norm_mean < 15.0:
        norm_status = "✓ GOOD - Embeddings have reasonable magnitude"
    elif 2.0 < norm_mean < 20.0:
        norm_status = "⚠ ACCEPTABLE - Embeddings magnitude is okay"
    else:
        norm_status = "✗ POOR - Embeddings magnitude may be problematic"
    
    diversity = embedding_metrics['diversity_score']
    if diversity > 0.3:
        diversity_status = "✓ EXCELLENT - High embedding diversity"
    elif diversity > 0.15:
        diversity_status = "✓ GOOD - Reasonable embedding diversity"
    elif diversity > 0.05:
        diversity_status = "⚠ ACCEPTABLE - Low diversity, may need more training"
    else:
        diversity_status = "✗ POOR - Embeddings are collapsed or too similar"
    
    print(f"Embedding Norm: {norm_status}")
    print(f"Embedding Diversity: {diversity_status}")
    
    # Retrieval metrics (if available)
    if retrieval_metrics is not None:
        print(f"\nIMAGE-TEXT RETRIEVAL METRICS:")
        print(f"\n  Image-to-Text:")
        print(f"    R@1:  {retrieval_metrics['i2t_r1']*100:.1f}%")
        print(f"    R@5:  {retrieval_metrics['i2t_r5']*100:.1f}%")
        print(f"    R@10: {retrieval_metrics['i2t_r10']*100:.1f}%")
        print(f"    Avg Rank: {retrieval_metrics['avg_i2t_rank']:.1f}")
        
        print(f"\n  Text-to-Image:")
        print(f"    R@1:  {retrieval_metrics['t2i_r1']*100:.1f}%")
        print(f"    R@5:  {retrieval_metrics['t2i_r5']*100:.1f}%")
        print(f"    R@10: {retrieval_metrics['t2i_r10']*100:.1f}%")
        print(f"    Avg Rank: {retrieval_metrics['avg_t2i_rank']:.1f}")
    
    print(f"\n{'='*70}")
    
    # Overall assessment
    if not embedding_metrics['is_collapsed'] and diversity > 0.15:
        print("✓ Vision encoder is working properly!")
        print("  Embeddings are diverse and have good quality.")
    elif embedding_metrics['is_collapsed']:
        print("⚠ Vision encoder may need more training.")
        print("  Embeddings are collapsed - model needs to learn discriminative features.")
    else:
        print("⚠ Vision encoder is learning but could improve.")
        print("  Consider training longer or adjusting hyperparameters.")
    
    print(f"{'='*70}")


def test_single_image(model, proj_head, image_path, cfg, device="cuda"):
    """Test on a single image."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    print(f"\nTesting single image: {image_path}")
    
    # Load and preprocess image
    img = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((cfg.get("img_size", 224), cfg.get("img_size", 224))),
        transforms.ToTensor()
    ])
    img_tensor = transform(img).unsqueeze(0).to(device)
    
    # Forward pass
    model.eval()
    if proj_head is not None:
        proj_head.eval()
    
    with torch.inference_mode():
        cls, grid = model(img_tensor)
        if proj_head is not None:
            cls_proj = proj_head(cls)
        else:
            cls_proj = cls
    
    print(f"\nEmbedding Statistics:")
    print(f"  CLS shape: {cls.shape}")
    print(f"  CLS norm: {cls.norm().item():.4f}")
    print(f"  Grid shape: {grid.shape}")
    print(f"  Grid norm: {grid.norm().item():.4f}")
    if proj_head is not None:
        print(f"  Projected CLS shape: {cls_proj.shape}")
        print(f"  Projected CLS norm: {cls_proj.norm().item():.4f}")
    
    return cls, grid


def main():
    parser = argparse.ArgumentParser(description="Test Vision Encoder with accuracy metrics")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/vision_tiny",
                       help="Path to Vision Encoder checkpoint directory")
    parser.add_argument("--num_samples", type=int, default=100,
                       help="Number of samples to evaluate (default: 100)")
    parser.add_argument("--image", type=str, default=None,
                       help="Path to single image file to test (optional)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                       help="Device to use (cuda/cpu)")
    parser.add_argument("--quick", action="store_true",
                       help="Quick test with 10 samples")
    parser.add_argument("--retrieval", action="store_true",
                       help="Also evaluate retrieval metrics (requires more samples)")
    args = parser.parse_args()
    
    if args.quick:
        args.num_samples = 10
    
    print("=" * 70)
    print("VISION ENCODER ACCURACY TEST")
    print("=" * 70)

    if args.device == "cuda" and torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
    
    # Load model
    try:
        model, img_proj, text_proj, text_encoder, tok, cfg = load_model_and_head(args.checkpoint, args.device)
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Test single image if provided
    if args.image:
        try:
            test_single_image(model, img_proj, args.image, cfg, args.device)
        except Exception as e:
            print(f"✗ Error testing image file: {e}")
            import traceback
            traceback.print_exc()
        return
    
    # Evaluate embedding quality
    try:
        embedding_metrics = evaluate_embedding_quality(
            model, img_proj, cfg, tok,
            device=args.device,
            num_samples=args.num_samples,
            verbose=True
        )
        
        # Optionally evaluate retrieval
        retrieval_metrics = None
        if args.retrieval and args.num_samples >= 20:
            retrieval_metrics = evaluate_retrieval(
                model, img_proj, text_proj, text_encoder, tok, cfg,
                device=args.device,
                num_samples=min(args.num_samples, 100),
                verbose=True
            )
            if retrieval_metrics is None:
                print("⚠️  Retrieval evaluation skipped (text encoder not available)")
        
        print_results(embedding_metrics, retrieval_metrics)
        
    except Exception as e:
        print(f"✗ Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
        return


if __name__ == "__main__":
    main()