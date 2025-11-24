"""
Simple test script to verify Talker (TTS) is working properly.
Loads a checkpoint, processes text/mel, and displays the output codes.
"""

import torch
import json
import os
import argparse
import random
from omni.talker import TalkerTiny
from omni.codec import RVQ
from omni.utils import TTSDataset, find_checkpoint, strip_orig_mod

def load_model(checkpoint_dir, device="cuda"):
    """Load Talker model from checkpoint."""
    checkpoint_path, checkpoint = find_checkpoint(checkpoint_dir, "talker.pt", "talker_step_", device)
    if checkpoint is None:
        raise FileNotFoundError(f"Checkpoint not found in: {checkpoint_dir}")
    
    print(f"Loading checkpoint from: {checkpoint_path}")
    
    # Get config from checkpoint or use defaults
    if "config" in checkpoint:
        cfg = checkpoint["config"]
    else:
        # Try loading from config file
        config_path = "configs/talker_tiny.json"
        if os.path.exists(config_path):
            cfg = json.load(open(config_path))
        else:
            cfg = {
                "d_model": 256,
                "n_layers": 4,
                "n_heads": 4,
                "d_ff": 1024,
                "codebooks": 8,
                "codebook_size": 1024,
                "dropout": 0.1,
                "sample_rate": 16000,
                "n_mels": 128,
                "frame_ms": 80,
            }
    
    # Initialize models
    rvq = RVQ(
        codebooks=cfg.get("codebooks", 8),
        codebook_size=cfg.get("codebook_size", 1024),
        d=64,
        compile_model=False
    ).to(device)
    
    talker = TalkerTiny(
        d_model=cfg.get("d_model", 256),
        n_layers=cfg.get("n_layers", 4),
        n_heads=cfg.get("n_heads", 4),
        d_ff=cfg.get("d_ff", 1024),
        codebooks=cfg.get("codebooks", 8),
        codebook_size=cfg.get("codebook_size", 1024),
        dropout=cfg.get("dropout", 0.1),
        use_gqa=cfg.get("use_gqa", False),
        use_swiglu=cfg.get("use_swiglu", True),
        rope_theta=cfg.get("rope_theta", 10000.0),
        compile_model=False
    ).to(device)
    
    # Load weights (strip _orig_mod, matches training script behavior)
    if "rvq" in checkpoint and "talker" in checkpoint:
        rvq_state = strip_orig_mod(checkpoint["rvq"])
        talker_state = strip_orig_mod(checkpoint["talker"])
        rvq.load_state_dict(rvq_state, strict=False)
        talker.load_state_dict(talker_state, strict=False)
    elif "model" in checkpoint:
        # Try to split model state dict
        model_state = strip_orig_mod(checkpoint["model"])
        # This is tricky - might need to filter by prefix
        rvq_state = {k.replace("rvq.", ""): v for k, v in model_state.items() if k.startswith("rvq.")}
        talker_state = {k.replace("talker.", ""): v for k, v in model_state.items() if k.startswith("talker.")}
        if rvq_state:
            rvq.load_state_dict(rvq_state, strict=False)
        if talker_state:
            talker.load_state_dict(talker_state, strict=False)
    else:
        # Try loading as-is
        state_dict = strip_orig_mod(checkpoint)
        # Try to load what we can
        try:
            rvq.load_state_dict({k: v for k, v in state_dict.items() if "rvq" in k or "codebook" in k}, strict=False)
        except:
            pass
        try:
            talker.load_state_dict({k: v for k, v in state_dict.items() if "talker" in k or ("decoder" in k and "rvq" not in k)}, strict=False)
        except:
            pass
    
    rvq.eval()
    talker.eval()
    print("✓ Models loaded successfully")
    
    return rvq, talker, cfg

def test_talker(rvq, talker, mel, device="cuda"):
    """Run Talker forward pass."""
    rvq.eval()
    talker.eval()
    mel = mel.to(device)
    
    with torch.no_grad():
        # Encode mel to codes
        codes = rvq.encode(mel)  # (B, T, codebooks)
        
        # Decode codes to mel (for testing)
        mel_recon = rvq.decode(codes)  # (B, T, d)
        
        # Talker forward pass (generate codes from mel)
        # This is a simplified test - actual usage would be text-to-codes
        logits = talker(mel_recon)  # (B, T, codebooks, codebook_size)
    
    return codes, mel_recon, logits

def get_random_mel_from_dataset(cfg):
    """Get a random mel spectrogram from the dataset."""
    csv_path = cfg.get("tts_csv", "data/audio/production_tts.csv")
    
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"TTS CSV not found: {csv_path}")
    
    # Create dataset with shuffling
    dataset = TTSDataset(
        csv_path=csv_path,
        sr=cfg.get("sample_rate", 16000),
        n_mels=cfg.get("n_mels", 128),
        frame_ms=cfg.get("frame_ms", 80),
        cfg=cfg,
        shuffle_buffer_size=10000,
        seed=random.randint(0, 1000000),
        skip_samples=0
    )
    
    # Get first sample
    try:
        iterator = iter(dataset)
        mel = next(iterator)
        return mel.unsqueeze(0)  # Add batch dimension
    except StopIteration:
        raise ValueError("Dataset is empty")

def evaluate_talker(rvq, talker, cfg, device="cuda", num_samples=100):
    """Evaluate Talker on multiple samples and compute metrics."""
    rvq.eval()
    talker.eval()
    
    csv_path = cfg.get("train_csv", "data/tts/production_tts.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"TTS CSV not found: {csv_path}")
    
    # Create dataset
    dataset = TTSDataset(
        csv_path=csv_path,
        sr=cfg.get("sample_rate", 16000),
        n_mels=cfg.get("n_mels", 128),
        frame_ms=cfg.get("frame_ms", 80),
        cfg=cfg,
        shuffle_buffer_size=10000,
        seed=random.randint(0, 1000000),
        skip_samples=0
    )
    
    iterator = iter(dataset)
    total_recon_error = 0.0
    num_valid_samples = 0
    
    print(f"\nEvaluating on {num_samples} samples...")
    
    with torch.no_grad():
        for i in range(num_samples):
            try:
                mel = next(iterator)
                mel = mel.unsqueeze(0).to(device)
                
                # Forward pass
                codes, mel_recon, logits = test_talker(rvq, talker, mel, device)
                
                # Compute reconstruction error
                recon_error = torch.mean((mel - mel_recon) ** 2).item()
                total_recon_error += recon_error
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
    avg_recon_error = total_recon_error / num_valid_samples if num_valid_samples > 0 else float('inf')
    
    return {
        'avg_reconstruction_error': avg_recon_error,
        'num_samples': num_valid_samples
    }

def main():
    parser = argparse.ArgumentParser(description="Test Talker (TTS)")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/talker_tiny",
                       help="Path to Talker checkpoint directory")
    parser.add_argument("--num_samples", type=int, default=100,
                       help="Number of samples to evaluate (default: 100)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                       help="Device to use (cuda/cpu)")
    args = parser.parse_args()
    
    print("=" * 60)
    print("Talker (TTS) Test")
    print("=" * 60)
    
    # Load model
    try:
        rvq, talker, cfg = load_model(args.checkpoint, args.device)
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Evaluate on multiple samples
    try:
        metrics = evaluate_talker(rvq, talker, cfg, args.device, args.num_samples)
        
        print(f"\n{'=' * 60}")
        print("EVALUATION RESULTS:")
        print(f"{'=' * 60}")
        print(f"Samples evaluated: {metrics['num_samples']}")
        print(f"Average Reconstruction Error (MSE): {metrics['avg_reconstruction_error']:.6f}")
        print(f"{'=' * 60}")
        
        # Interpretation
        if metrics['avg_reconstruction_error'] < 0.01:
            print("✓ Excellent reconstruction quality!")
        elif metrics['avg_reconstruction_error'] < 0.1:
            print("✓ Good reconstruction quality")
        elif metrics['avg_reconstruction_error'] < 0.5:
            print("⚠️  Moderate quality - model may need more training")
        else:
            print("⚠️  Poor quality - model needs significant training")
            
    except Exception as e:
        print(f"❌ Error during evaluation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

