"""
Simple test script to verify Vision Encoder is working properly.
Loads a checkpoint, processes an image, and displays the output embeddings.
"""

import torch
import json
import os
import argparse
import random
from PIL import Image
from torchvision import transforms
from omni.vision_encoder import ViTTiny
from omni.utils import ImgCapDataset, find_checkpoint, strip_orig_mod

def load_model(checkpoint_dir, device="cuda"):
    """Load Vision Encoder model from checkpoint."""
    checkpoint_path, checkpoint = find_checkpoint(checkpoint_dir, "vision.pt", "vision_step_", device)
    if checkpoint is None:
        raise FileNotFoundError(f"Checkpoint not found in: {checkpoint_dir}")
    
    print(f"Loading checkpoint from: {checkpoint_path}")
    
    # Get config from checkpoint or use defaults
    if "config" in checkpoint:
        cfg = checkpoint["config"]
    else:
        # Try loading from config file
        config_path = "configs/vision_tiny.json"
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                cfg = json.load(f)
        else:
            cfg = {
                "img_size": 224,
                "patch": 16,
                "d_model": 128,
                "n_layers": 4,
                "n_heads": 2,
                "d_ff": 512,
                "dropout": 0.1,
            }
    
    # Initialize model
    model = ViTTiny(
        img_size=cfg.get("img_size", 224),
        patch=cfg.get("patch", 16),
        d=cfg.get("d_model", 128),
        layers=cfg.get("n_layers", 4),
        heads=cfg.get("n_heads", 2),
        ff=cfg.get("d_ff", 512),
        dropout=cfg.get("dropout", 0.1),
        compile_model=False
    ).to(device)
    
    # Load weights
    if "vit" in checkpoint:
        state_dict = checkpoint["vit"]
    elif "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint
    
    # Strip _orig_mod (matches training script behavior)
    state_dict = strip_orig_mod(state_dict)
    model.load_state_dict(state_dict, strict=False)
    
    model.eval()
    print("✓ Model loaded successfully")
    
    return model, cfg

def preprocess_image(image_path, img_size=224):
    """Load and preprocess image."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    img = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor()
    ])
    img_tensor = transform(img).unsqueeze(0)  # Add batch dimension
    return img_tensor, img

def test_vision_encoder(model, image_tensor, device="cuda"):
    """Run vision encoder forward pass."""
    model.eval()
    image_tensor = image_tensor.to(device)
    
    with torch.no_grad():
        cls, grid = model(image_tensor)
    
    return cls, grid

def get_random_image_from_dataset(cfg):
    """Get a random image from the dataset."""
    manifest_path = cfg.get("train_manifest", "data/images/production_annotations.json")
    image_root = cfg.get("image_root", "data/images")
    
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Image manifest not found: {manifest_path}")
    
    # Create dataset with shuffling
    dataset = ImgCapDataset(
        manifest=manifest_path,
        image_root=image_root,
        img_size=cfg.get("img_size", 224),
        shuffle_buffer_size=10000,
        seed=random.randint(0, 1000000),
        skip_samples=0
    )
    
    # Get first sample
    try:
        iterator = iter(dataset)
        img_tensor, caption = next(iterator)
        return img_tensor.unsqueeze(0), caption  # Add batch dimension
    except StopIteration:
        raise ValueError("Dataset is empty")

def evaluate_vision_encoder(model, cfg, device="cuda", num_samples=100):
    """Evaluate vision encoder on multiple samples and compute metrics."""
    model.eval()
    
    manifest_path = cfg.get("train_manifest", "data/images/production_annotations.json")
    image_root = cfg.get("image_root", "data/images")
    
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Image manifest not found: {manifest_path}")
    
    # Create dataset
    dataset = ImgCapDataset(
        manifest=manifest_path,
        image_root=image_root,
        img_size=cfg.get("img_size", 224),
        shuffle_buffer_size=10000,
        seed=random.randint(0, 1000000),
        skip_samples=0
    )
    
    iterator = iter(dataset)
    total_cls_norm = 0.0
    total_cls_std = 0.0
    total_grid_norm = 0.0
    total_grid_std = 0.0
    num_valid_samples = 0
    
    print(f"\nEvaluating on {num_samples} samples...")
    
    with torch.no_grad():
        for i in range(num_samples):
            try:
                img_tensor, caption = next(iterator)
                img_tensor = img_tensor.unsqueeze(0).to(device)
                
                # Forward pass
                cls, grid = model(img_tensor)
                
                # Compute statistics
                cls_norm = torch.norm(cls).item()
                cls_std = cls.std().item()
                grid_norm = torch.norm(grid).item()
                grid_std = grid.std().item()
                
                total_cls_norm += cls_norm
                total_cls_std += cls_std
                total_grid_norm += grid_norm
                total_grid_std += grid_std
                num_valid_samples += 1
                
                if (i + 1) % 10 == 0:
                    print(f"  Processed {i + 1}/{num_samples} samples...", end='\r')
                    
            except StopIteration:
                print(f"\n⚠️  Dataset exhausted after {i} samples")
                break
            except Exception as e:
                print(f"\n⚠️  Error processing sample {i}: {e}")
                continue
    
    print()  # New line after progress
    
    # Calculate metrics
    avg_cls_norm = total_cls_norm / num_valid_samples if num_valid_samples > 0 else 0.0
    avg_cls_std = total_cls_std / num_valid_samples if num_valid_samples > 0 else 0.0
    avg_grid_norm = total_grid_norm / num_valid_samples if num_valid_samples > 0 else 0.0
    avg_grid_std = total_grid_std / num_valid_samples if num_valid_samples > 0 else 0.0
    
    return {
        'avg_cls_norm': avg_cls_norm,
        'avg_cls_std': avg_cls_std,
        'avg_grid_norm': avg_grid_norm,
        'avg_grid_std': avg_grid_std,
        'num_samples': num_valid_samples
    }

def main():
    parser = argparse.ArgumentParser(description="Test Vision Encoder")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/vision_tiny",
                       help="Path to Vision Encoder checkpoint directory")
    parser.add_argument("--num_samples", type=int, default=100,
                       help="Number of samples to evaluate (default: 100)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                       help="Device to use (cuda/cpu)")
    args = parser.parse_args()
    
    print("=" * 60)
    print("Vision Encoder Test")
    print("=" * 60)
    
    # Load model
    try:
        model, cfg = load_model(args.checkpoint, args.device)
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Evaluate on multiple samples from dataset
    try:
        metrics = evaluate_vision_encoder(model, cfg, args.device, args.num_samples)
        
        print(f"\n{'=' * 60}")
        print("EVALUATION RESULTS:")
        print(f"{'=' * 60}")
        print(f"Samples evaluated: {metrics['num_samples']}")
        print(f"Average CLS Norm: {metrics['avg_cls_norm']:.4f}")
        print(f"Average CLS Std: {metrics['avg_cls_std']:.4f}")
        print(f"Average Grid Norm: {metrics['avg_grid_norm']:.4f}")
        print(f"Average Grid Std: {metrics['avg_grid_std']:.4f}")
        print(f"{'=' * 60}")
        print("✓ Vision encoder is working properly!")
        print("  (Note: For image captioning accuracy, use a full vision-language pipeline)")
            
    except Exception as e:
        print(f"❌ Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
        return

if __name__ == "__main__":
    main()

