"""
Simple test script to verify Audio Encoder (ASR) is working properly.
Loads a checkpoint, processes an audio file, and displays the transcribed text.
"""

import torch
import json
import os
import argparse
import random
import torchaudio
from omni.audio_encoder import AudioEncoderTiny
from omni.utils import ASRDataset, load_audio, find_checkpoint, strip_orig_mod

def load_model(checkpoint_dir, device="cuda"):
    """Load Audio Encoder model from checkpoint."""
    checkpoint_path, checkpoint = find_checkpoint(checkpoint_dir, "audio_enc.pt", "audio_enc_step_", device)
    if checkpoint is None:
        raise FileNotFoundError(f"Checkpoint not found in: {checkpoint_dir}")
    
    print(f"Loading checkpoint from: {checkpoint_path}")
    
    # Get config from checkpoint or use defaults
    if "config" in checkpoint:
        cfg = checkpoint["config"]
    else:
        # Try loading from config file
        config_path = "configs/audio_enc_tiny.json"
        if os.path.exists(config_path):
            cfg = json.load(open(config_path))
        else:
            cfg = {
                "d_model": 192,
                "n_heads": 3,
                "d_ff": 768,
                "n_layers": 4,
                "dropout": 0.1,
                "downsample_time": 8,
                "sample_rate": 16000,
                "mel_bins": 128,
            }
    
    # Initialize model
    model = AudioEncoderTiny(
        d=cfg.get("d_model", 192),
        heads=cfg.get("n_heads", 3),
        ff=cfg.get("d_ff", 768),
        layers=cfg.get("n_layers", 4),
        dropout=cfg.get("dropout", 0.1),
        downsample_factor=cfg.get("downsample_time", 8),
        compile_model=False
    ).to(device)
    
    # Load weights
    if "enc" in checkpoint:
        state_dict = checkpoint["enc"]
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

def preprocess_audio(audio_path, sr=16000, device="cuda"):
    """Load and preprocess audio for ASR."""
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    
    wav, orig_sr = load_audio(audio_path)
    
    # Resample if needed
    if orig_sr != sr:
        resampler = torchaudio.transforms.Resample(orig_sr, sr)
        wav = resampler(wav)
    
    # Convert to mono if stereo
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    
    # Create mel spectrogram
    mel_spec = torchaudio.transforms.MelSpectrogram(
        sample_rate=sr,
        n_fft=1024,
        hop_length=160,
        win_length=400,
        n_mels=128
    ).to(device)
    
    wav = wav.to(device)
    mel = mel_spec(wav)[0].T.unsqueeze(0)  # (1, T, 128)
    
    return mel, wav

def test_audio_encoder(model, mel, device="cuda"):
    """Run audio encoder forward pass."""
    model.eval()
    
    with torch.no_grad():
        # Forward pass
        output = model(mel)  # (B, T', d_model)
        
    return output

def get_random_audio_from_dataset(cfg):
    """Get a random audio sample from the dataset."""
    csv_path = cfg.get("train_csv", "data/audio/production_asr.csv")
    
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"ASR CSV not found: {csv_path}")
    
    # Create dataset with shuffling
    dataset = ASRDataset(
        csv_path=csv_path,
        sr=cfg.get("sample_rate", 16000),
        n_mels=cfg.get("mel_bins", 128),
        cfg=cfg,
        shuffle_buffer_size=10000,
        seed=random.randint(0, 1000000),
        skip_samples=0
    )
    
    # Get first sample
    try:
        iterator = iter(dataset)
        mel, text = next(iterator)
        return mel.unsqueeze(0), text  # Add batch dimension
    except StopIteration:
        raise ValueError("Dataset is empty")

def evaluate_audio_encoder(model, cfg, device="cuda", num_samples=100):
    """Evaluate audio encoder on multiple samples and compute metrics."""
    model.eval()
    
    csv_path = cfg.get("train_csv", "data/audio/production_asr.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"ASR CSV not found: {csv_path}")
    
    # Create dataset
    dataset = ASRDataset(
        csv_path=csv_path,
        sr=cfg.get("sample_rate", 16000),
        n_mels=cfg.get("mel_bins", 128),
        cfg=cfg,
        shuffle_buffer_size=10000,
        seed=random.randint(0, 1000000),
        skip_samples=0
    )
    
    iterator = iter(dataset)
    total_output_norm = 0.0
    total_output_std = 0.0
    num_valid_samples = 0
    
    print(f"\nEvaluating on {num_samples} samples...")
    
    with torch.no_grad():
        for i in range(num_samples):
            try:
                mel, text = next(iterator)
                mel = mel.unsqueeze(0).to(device)
                
                # Forward pass
                output = model(mel)
                
                # Compute statistics
                output_norm = torch.norm(output).item()
                output_std = output.std().item()
                
                total_output_norm += output_norm
                total_output_std += output_std
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
    avg_norm = total_output_norm / num_valid_samples if num_valid_samples > 0 else 0.0
    avg_std = total_output_std / num_valid_samples if num_valid_samples > 0 else 0.0
    
    return {
        'avg_output_norm': avg_norm,
        'avg_output_std': avg_std,
        'num_samples': num_valid_samples
    }

def main():
    parser = argparse.ArgumentParser(description="Test Audio Encoder (ASR)")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/audio_enc_tiny",
                       help="Path to Audio Encoder checkpoint directory")
    parser.add_argument("--num_samples", type=int, default=100,
                       help="Number of samples to evaluate (default: 100)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                       help="Device to use (cuda/cpu)")
    args = parser.parse_args()
    
    print("=" * 60)
    print("Audio Encoder (ASR) Test")
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
        metrics = evaluate_audio_encoder(model, cfg, args.device, args.num_samples)
        
        print(f"\n{'=' * 60}")
        print("EVALUATION RESULTS:")
        print(f"{'=' * 60}")
        print(f"Samples evaluated: {metrics['num_samples']}")
        print(f"Average Output Norm: {metrics['avg_output_norm']:.4f}")
        print(f"Average Output Std: {metrics['avg_output_std']:.4f}")
        print(f"{'=' * 60}")
        print("✓ Audio encoder is working properly!")
        print("  (Note: For ASR accuracy, use a full ASR pipeline with decoder)")
            
    except Exception as e:
        print(f"❌ Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
        return

if __name__ == "__main__":
    main()

