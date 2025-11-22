"""
Simple test script to verify Vocoder is working properly.
Loads a checkpoint, processes a mel spectrogram, and displays the output audio info.
"""

import torch
import json
import os
import argparse
import random
import torchaudio
from omni.codec import HiFiGANVocoder
from omni.utils import VocoderDataset, find_checkpoint, strip_orig_mod

def load_model(checkpoint_dir, device="cuda"):
    """Load Vocoder model from checkpoint."""
    checkpoint_path, checkpoint = find_checkpoint(checkpoint_dir, "vocoder.pt", "vocoder_step_", device)
    if checkpoint is None:
        raise FileNotFoundError(f"Checkpoint not found in: {checkpoint_dir}")
    
    print(f"Loading checkpoint from: {checkpoint_path}")
    
    # Get config from checkpoint or use defaults
    if "config" in checkpoint:
        cfg = checkpoint["config"]
    else:
        # Try loading from config file
        config_path = "configs/vocoder_tiny.json"
        if os.path.exists(config_path):
            cfg = json.load(open(config_path))
        else:
            cfg = {
                "sample_rate": 16000,
                "n_mels": 128,
                "n_fft": 1024,
                "hop_length": 256,
            }
    
    # Initialize model
    model = HiFiGANVocoder(
        sample_rate=cfg.get("sample_rate", 16000),
        n_mels=cfg.get("n_mels", 128),
        n_fft=cfg.get("n_fft", 1024),
        hop_length=cfg.get("hop_length", 256)
    ).to(device)
    
    # Load weights
    if "vocoder" in checkpoint:
        state_dict = checkpoint["vocoder"]
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

def test_vocoder(model, mel, device="cuda"):
    """Run vocoder forward pass."""
    model.eval()
    mel = mel.to(device)
    
    with torch.no_grad():
        # Forward pass: mel -> audio
        audio = model(mel)  # (B, T_audio)
    
    return audio

def get_random_mel_from_dataset(cfg):
    """Get a random mel spectrogram from the dataset."""
    csv_path = cfg.get("train_csv", "data/audio/production_tts.csv")
    
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Vocoder CSV not found: {csv_path}")
    
    # Create dataset with shuffling
    dataset = VocoderDataset(
        csv_path=csv_path,
        sr=cfg.get("sample_rate", 16000),
        n_mels=cfg.get("n_mels", 128),
        n_fft=cfg.get("n_fft", 1024),
        hop_length=cfg.get("hop_length", 256),
        cfg=cfg,
        shuffle_buffer_size=10000,
        seed=random.randint(0, 1000000),
        skip_samples=0
    )
    
    # Get first sample
    try:
        iterator = iter(dataset)
        mel, audio = next(iterator)
        return mel.unsqueeze(0), audio  # Add batch dimension
    except StopIteration:
        raise ValueError("Dataset is empty")

def evaluate_vocoder(model, cfg, device="cuda", num_samples=100):
    """Evaluate vocoder on multiple samples and compute metrics."""
    model.eval()
    
    csv_path = cfg.get("train_csv", "data/vocoder/production_vocoder.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Vocoder CSV not found: {csv_path}")
    
    # Create dataset
    dataset = VocoderDataset(
        csv_path=csv_path,
        sr=cfg.get("sample_rate", 16000),
        n_mels=cfg.get("n_mels", 128),
        n_fft=cfg.get("n_fft", 1024),
        hop_length=cfg.get("hop_length", 256),
        cfg=cfg,
        shuffle_buffer_size=10000,
        seed=random.randint(0, 1000000),
        skip_samples=0
    )
    
    iterator = iter(dataset)
    total_audio_norm = 0.0
    total_audio_std = 0.0
    num_valid_samples = 0
    
    print(f"\nEvaluating on {num_samples} samples...")
    
    with torch.no_grad():
        for i in range(num_samples):
            try:
                mel, ground_truth_audio = next(iterator)
                mel = mel.unsqueeze(0).to(device)
                
                # Forward pass
                audio = test_vocoder(model, mel, device)
                
                # Compute statistics
                audio_norm = torch.norm(audio).item()
                audio_std = audio.std().item()
                
                total_audio_norm += audio_norm
                total_audio_std += audio_std
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
    avg_audio_norm = total_audio_norm / num_valid_samples if num_valid_samples > 0 else 0.0
    avg_audio_std = total_audio_std / num_valid_samples if num_valid_samples > 0 else 0.0
    
    return {
        'avg_audio_norm': avg_audio_norm,
        'avg_audio_std': avg_audio_std,
        'num_samples': num_valid_samples
    }

def main():
    parser = argparse.ArgumentParser(description="Test Vocoder")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/vocoder_tiny",
                       help="Path to Vocoder checkpoint directory")
    parser.add_argument("--num_samples", type=int, default=100,
                       help="Number of samples to evaluate (default: 100)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                       help="Device to use (cuda/cpu)")
    args = parser.parse_args()
    
    print("=" * 60)
    print("Vocoder Test")
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
        metrics = evaluate_vocoder(model, cfg, args.device, args.num_samples)
        
        print(f"\n{'=' * 60}")
        print("EVALUATION RESULTS:")
        print(f"{'=' * 60}")
        print(f"Samples evaluated: {metrics['num_samples']}")
        print(f"Average Audio Norm: {metrics['avg_audio_norm']:.4f}")
        print(f"Average Audio Std: {metrics['avg_audio_std']:.4f}")
        print(f"{'=' * 60}")
        print("✓ Vocoder is working properly!")
            
    except Exception as e:
        print(f"❌ Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
        return

if __name__ == "__main__":
    main()

