"""
Complete test script to validate Vocoder (HiFi-GAN) accuracy.
Measures mel reconstruction, audio quality metrics (PESQ, MCD), and perceptual quality.
"""

import torch
import torch.nn.functional as F
import json
import os
import argparse
import random
import numpy as np
import torchaudio
from omni.codec import HiFiGANVocoder
from omni.utils import VocoderDataset, find_checkpoint, strip_orig_mod
from tqdm import tqdm

# Try to import optional audio quality metrics
try:
    import librosa
    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False
    print("⚠️  librosa not available. Some metrics will be disabled.")

try:
    from pesq import pesq as pesq_metric
    HAS_PESQ = True
except ImportError:
    HAS_PESQ = False
    print("⚠️  pesq not available. Install with: pip install pesq")


def load_model_and_config(checkpoint_dir, device="cuda"):
    """Load Vocoder model from checkpoint."""
    checkpoint_path, checkpoint = find_checkpoint(checkpoint_dir, "vocoder.pt", "vocoder_step_", device)
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
    
    sr = cfg.get("sample_rate", 16000)
    n_mels = cfg.get("n_mels", 128)
    n_fft = cfg.get("n_fft", 1024)
    hop_length = cfg.get("hop_length", 256)
    
    # Initialize model
    model = HiFiGANVocoder(
        sample_rate=sr,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length,
        upsample_initial_channel=cfg.get("upsample_initial_channel", 256),
        resblock_kernel_sizes=cfg.get("resblock_kernel_sizes", [3, 5, 7]),
        resblock_dilation_sizes=cfg.get("resblock_dilation_sizes", [[1, 2], [1, 2], [1, 2]]),
        compile_model=False
    ).to(device)
    
    # Load weights
    if "generator" in checkpoint:
        state_dict = checkpoint["generator"]
    elif "vocoder" in checkpoint:
        state_dict = checkpoint["vocoder"]
    elif "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint
    
    # Strip _orig_mod
    state_dict = strip_orig_mod(state_dict)
    model.load_state_dict(state_dict, strict=False)
    
    model.eval()
    
    print("✓ Vocoder loaded successfully")
    print(f"  Sample rate: {sr} Hz")
    print(f"  Mel bins: {n_mels}")
    print(f"  Hop length: {hop_length}")
    print(f"  Upsampling factor: {hop_length}x")
    
    return model, cfg


def compute_mel_reconstruction_error(mel_real, mel_fake):
    """
    Compute mel spectrogram reconstruction error.
    
    Args:
        mel_real: (T, n_mels) ground truth mel
        mel_fake: (T, n_mels) generated mel
    
    Returns:
        Dictionary with MSE, MAE, and spectral convergence
    """
    # Ensure same length
    min_len = min(mel_real.shape[0], mel_fake.shape[0])
    mel_real = mel_real[:min_len]
    mel_fake = mel_fake[:min_len]
    
    # MSE and MAE
    mse = torch.mean((mel_real - mel_fake) ** 2).item()
    mae = torch.mean(torch.abs(mel_real - mel_fake)).item()
    
    # Spectral convergence (normalized error)
    numerator = torch.norm(mel_real - mel_fake, p='fro')
    denominator = torch.norm(mel_real, p='fro')
    spec_convergence = (numerator / denominator).item() if denominator > 0 else float('inf')
    
    return {
        'mse': mse,
        'mae': mae,
        'spectral_convergence': spec_convergence,
    }


def compute_mel_cepstral_distortion(mel_real, mel_fake):
    """
    Compute Mel-Cepstral Distortion (MCD) - standard TTS metric.
    Lower is better. Good vocoders achieve MCD < 3.0 dB.
    
    Requires librosa.
    """
    if not HAS_LIBROSA:
        return None
    
    # Ensure same length
    min_len = min(mel_real.shape[0], mel_fake.shape[0])
    mel_real = mel_real[:min_len].cpu().numpy()
    mel_fake = mel_fake[:min_len].cpu().numpy()
    
    # Convert to mel-cepstral coefficients
    try:
        # Take log mel (if not already in log scale)
        log_mel_real = np.log(np.maximum(mel_real, 1e-10))
        log_mel_fake = np.log(np.maximum(mel_fake, 1e-10))
        
        # DCT to get cepstral coefficients
        from scipy.fftpack import dct
        mfcc_real = dct(log_mel_real, type=2, axis=1, norm='ortho')
        mfcc_fake = dct(log_mel_fake, type=2, axis=1, norm='ortho')
        
        # MCD formula: 10/ln(10) * sqrt(2 * sum((c_real - c_fake)^2))
        # Typically use first 13 coefficients (exclude 0th)
        mfcc_real = mfcc_real[:, 1:14]  # Use coef 1-13
        mfcc_fake = mfcc_fake[:, 1:14]
        
        diff = mfcc_real - mfcc_fake
        mcd = (10.0 / np.log(10.0)) * np.sqrt(2 * np.mean(diff ** 2))
        
        return mcd
    except Exception as e:
        print(f"Warning: Failed to compute MCD: {e}")
        return None


def compute_pesq(audio_real, audio_fake, sr=16000):
    """
    Compute PESQ (Perceptual Evaluation of Speech Quality).
    Range: -0.5 to 4.5, higher is better. Good vocoders achieve > 3.5.
    
    Requires pesq package.
    """
    if not HAS_PESQ:
        return None
    
    # Ensure same length
    min_len = min(len(audio_real), len(audio_fake))
    audio_real = audio_real[:min_len].cpu().numpy()
    audio_fake = audio_fake[:min_len].cpu().numpy()
    
    # PESQ requires specific sample rates (8000 or 16000)
    if sr not in [8000, 16000]:
        print(f"Warning: PESQ requires 8kHz or 16kHz, got {sr}Hz. Skipping PESQ.")
        return None
    
    try:
        # PESQ mode: 'wb' for 16kHz, 'nb' for 8kHz
        mode = 'wb' if sr == 16000 else 'nb'
        pesq_score = pesq_metric(sr, audio_real, audio_fake, mode)
        return pesq_score
    except Exception as e:
        print(f"Warning: Failed to compute PESQ: {e}")
        return None


def compute_stoi(audio_real, audio_fake, sr=16000):
    """
    Compute STOI (Short-Time Objective Intelligibility).
    Range: 0 to 1, higher is better. Good vocoders achieve > 0.90.
    
    Requires pystoi package (optional).
    """
    try:
        from pystoi import stoi
        
        # Ensure same length
        min_len = min(len(audio_real), len(audio_fake))
        audio_real = audio_real[:min_len].cpu().numpy()
        audio_fake = audio_fake[:min_len].cpu().numpy()
        
        stoi_score = stoi(audio_real, audio_fake, sr, extended=False)
        return stoi_score
    except ImportError:
        return None
    except Exception as e:
        print(f"Warning: Failed to compute STOI: {e}")
        return None


def evaluate_vocoder_quality(model, cfg, device="cuda", num_samples=100, verbose=True):
    """
    Comprehensive vocoder evaluation.
    
    Metrics:
    - Mel reconstruction (MSE, MAE, spectral convergence)
    - Mel-Cepstral Distortion (MCD)
    - PESQ (perceptual quality)
    - STOI (intelligibility)
    - Audio statistics
    """
    model.eval()
    
    csv_path = cfg.get("train_csv", "data/audio/production_tts.csv")
    if not os.path.exists(csv_path):
        # Try ASR CSV as fallback
        csv_path = cfg.get("train_csv", "data/audio/production_asr.csv")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Audio CSV not found: {csv_path}")
    
    sr = cfg.get("sample_rate", 16000)
    n_mels = cfg.get("n_mels", 128)
    n_fft = cfg.get("n_fft", 1024)
    hop_length = cfg.get("hop_length", 256)
    
    # Create dataset
    dataset = VocoderDataset(
        csv_path=csv_path,
        sr=sr,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length,
        cfg=cfg,
        shuffle_buffer_size=10000,
        seed=42,  # Fixed seed for reproducibility
        skip_samples=0
    )
    
    # Create mel spectrogram transform for reconstructing mel from generated audio
    melspec_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=n_fft,
        n_mels=n_mels
    ).to(device)
    
    # Accumulators
    total_mel_mse = 0.0
    total_mel_mae = 0.0
    total_spec_conv = 0.0
    total_mcd = 0.0
    total_pesq = 0.0
    total_stoi = 0.0
    total_audio_mse = 0.0
    mcd_count = 0
    pesq_count = 0
    stoi_count = 0
    num_valid = 0
    
    if verbose:
        print(f"\nEvaluating vocoder quality on {num_samples} samples...")
        iterator = tqdm(dataset, total=num_samples, desc="Processing")
    else:
        iterator = iter(dataset)
    
    with torch.no_grad():
        for i, (mel_real, audio_real) in enumerate(iterator):
            if i >= num_samples:
                break
            
            try:
                mel_real = mel_real.unsqueeze(0).to(device)  # (1, T_mel, n_mels)
                audio_real = audio_real.to(device)  # (T_audio,)
                
                # Transpose mel for generator: (1, n_mels, T_mel)
                mel_input = mel_real.transpose(1, 2)
                
                # Generate audio
                audio_fake = model(mel_input)  # (1, T_audio)
                audio_fake = audio_fake.squeeze(0)  # (T_audio,)
                
                # Ensure same length
                min_audio_len = min(audio_real.shape[0], audio_fake.shape[0])
                audio_real_trimmed = audio_real[:min_audio_len]
                audio_fake_trimmed = audio_fake[:min_audio_len]
                
                # 1. Mel reconstruction metrics
                # Compute mel from generated audio
                mel_fake_raw = melspec_transform(audio_fake_trimmed.unsqueeze(0))  # (1, n_mels, T_mel)
                mel_fake = mel_fake_raw.squeeze(0).transpose(0, 1)  # (T_mel, n_mels)
                mel_real_squeezed = mel_real.squeeze(0)  # (T_mel, n_mels)
                
                mel_metrics = compute_mel_reconstruction_error(mel_real_squeezed, mel_fake)
                total_mel_mse += mel_metrics['mse']
                total_mel_mae += mel_metrics['mae']
                total_spec_conv += mel_metrics['spectral_convergence']
                
                # 2. Mel-Cepstral Distortion
                mcd = compute_mel_cepstral_distortion(mel_real_squeezed, mel_fake)
                if mcd is not None:
                    total_mcd += mcd
                    mcd_count += 1
                
                # 3. PESQ (perceptual quality)
                pesq_score = compute_pesq(audio_real_trimmed, audio_fake_trimmed, sr)
                if pesq_score is not None:
                    total_pesq += pesq_score
                    pesq_count += 1
                
                # 4. STOI (intelligibility)
                stoi_score = compute_stoi(audio_real_trimmed, audio_fake_trimmed, sr)
                if stoi_score is not None:
                    total_stoi += stoi_score
                    stoi_count += 1
                
                # 5. Audio MSE
                audio_mse = torch.mean((audio_real_trimmed - audio_fake_trimmed) ** 2).item()
                total_audio_mse += audio_mse
                
                num_valid += 1
                
            except Exception as e:
                if verbose:
                    print(f"\n⚠️  Error processing sample {i}: {e}")
                continue
    
    if num_valid == 0:
        raise ValueError("No valid samples processed!")
    
    # Calculate averages
    results = {
        'num_samples': num_valid,
        # Mel reconstruction
        'mel_mse': total_mel_mse / num_valid,
        'mel_mae': total_mel_mae / num_valid,
        'spectral_convergence': total_spec_conv / num_valid,
        # Advanced metrics
        'mcd': total_mcd / mcd_count if mcd_count > 0 else None,
        'pesq': total_pesq / pesq_count if pesq_count > 0 else None,
        'stoi': total_stoi / stoi_count if stoi_count > 0 else None,
        # Audio MSE
        'audio_mse': total_audio_mse / num_valid,
    }
    
    return results


def print_results(metrics):
    """Pretty print evaluation results."""
    print(f"\n{'='*70}")
    print("VOCODER (HiFi-GAN) EVALUATION RESULTS")
    print(f"{'='*70}")
    
    print(f"\nSamples Evaluated: {metrics['num_samples']}")
    
    # Mel reconstruction quality
    print(f"\nMEL SPECTROGRAM RECONSTRUCTION:")
    print(f"  MSE: {metrics['mel_mse']:.6f}")
    print(f"  MAE: {metrics['mel_mae']:.6f}")
    print(f"  Spectral Convergence: {metrics['spectral_convergence']:.6f}")
    
    # Advanced metrics
    print(f"\nADVANCED AUDIO QUALITY METRICS:")
    if metrics['mcd'] is not None:
        print(f"  MCD (Mel-Cepstral Distortion): {metrics['mcd']:.4f} dB")
        print(f"    (Lower is better. Good: < 3.0 dB)")
    else:
        print(f"  MCD: Not available (install librosa)")
    
    if metrics['pesq'] is not None:
        print(f"  PESQ (Perceptual Quality): {metrics['pesq']:.4f}")
        print(f"    (Range: -0.5 to 4.5. Good: > 3.5)")
    else:
        print(f"  PESQ: Not available (install pesq: pip install pesq)")
    
    if metrics['stoi'] is not None:
        print(f"  STOI (Intelligibility): {metrics['stoi']:.4f}")
        print(f"    (Range: 0 to 1. Good: > 0.90)")
    else:
        print(f"  STOI: Not available (install pystoi: pip install pystoi)")
    
    print(f"\nAUDIO WAVEFORM:")
    print(f"  MSE: {metrics['audio_mse']:.6f}")
    
    # Interpretation
    print(f"\n{'='*70}")
    print("INTERPRETATION:")
    print(f"{'='*70}")
    
    # Mel reconstruction assessment
    mel_mse = metrics['mel_mse']
    if mel_mse < 0.01:
        mel_status = "✓ EXCELLENT - High fidelity mel reconstruction"
    elif mel_mse < 0.05:
        mel_status = "✓ GOOD - Reasonable mel reconstruction"
    elif mel_mse < 0.1:
        mel_status = "⚠ ACCEPTABLE - Moderate quality"
    else:
        mel_status = "✗ POOR - Needs more training"
    
    # MCD assessment
    if metrics['mcd'] is not None:
        mcd = metrics['mcd']
        if mcd < 3.0:
            mcd_status = "✓ EXCELLENT - Professional quality"
        elif mcd < 4.5:
            mcd_status = "✓ GOOD - High quality"
        elif mcd < 6.0:
            mcd_status = "⚠ ACCEPTABLE - Usable quality"
        else:
            mcd_status = "✗ POOR - Needs improvement"
    else:
        mcd_status = "? Unknown (install librosa for MCD)"
    
    # PESQ assessment
    if metrics['pesq'] is not None:
        pesq = metrics['pesq']
        if pesq > 3.5:
            pesq_status = "✓ EXCELLENT - Excellent perceptual quality"
        elif pesq > 3.0:
            pesq_status = "✓ GOOD - Good perceptual quality"
        elif pesq > 2.5:
            pesq_status = "⚠ ACCEPTABLE - Fair quality"
        else:
            pesq_status = "✗ POOR - Poor perceptual quality"
    else:
        pesq_status = "? Unknown (install pesq for PESQ)"
    
    print(f"Mel Reconstruction: {mel_status}")
    print(f"MCD: {mcd_status}")
    print(f"PESQ: {pesq_status}")
    
    # Overall assessment
    print(f"\n{'='*70}")
    
    # Determine overall quality based on available metrics
    excellent_criteria = []
    good_criteria = []
    
    if mel_mse < 0.05:
        excellent_criteria.append("mel reconstruction")
    elif mel_mse < 0.1:
        good_criteria.append("mel reconstruction")
    
    if metrics['mcd'] is not None and metrics['mcd'] < 4.5:
        excellent_criteria.append("MCD")
    elif metrics['mcd'] is not None and metrics['mcd'] < 6.0:
        good_criteria.append("MCD")
    
    if metrics['pesq'] is not None and metrics['pesq'] > 3.0:
        excellent_criteria.append("PESQ")
    elif metrics['pesq'] is not None and metrics['pesq'] > 2.5:
        good_criteria.append("PESQ")
    
    if len(excellent_criteria) >= 2:
        print("✓ Vocoder is working excellently!")
        print(f"  Strong performance in: {', '.join(excellent_criteria)}")
        print("  Ready for high-quality TTS synthesis.")
    elif len(excellent_criteria) + len(good_criteria) >= 2:
        print("✓ Vocoder is working well!")
        print("  Good quality for most TTS applications.")
        if good_criteria:
            print(f"  Could improve: {', '.join(good_criteria)}")
    else:
        print("⚠ Vocoder needs more training.")
        print("  Current quality is not sufficient for production.")
        print("  Recommendation: Train for more steps or check hyperparameters.")
    
    print(f"{'='*70}")


def test_single_mel(model, mel, cfg, device="cuda"):
    """Test on a single mel spectrogram."""
    print(f"\nTesting single mel spectrogram...")
    print(f"  Input mel shape: {mel.shape}")
    
    mel = mel.unsqueeze(0).to(device)
    
    # Transpose for generator
    mel_input = mel.transpose(1, 2)
    
    # Generate audio
    model.eval()
    with torch.no_grad():
        audio = model(mel_input)
    
    audio = audio.squeeze(0)
    
    print(f"  Generated audio shape: {audio.shape}")
    print(f"  Audio length: {len(audio)} samples ({len(audio)/cfg.get('sample_rate', 16000):.2f} seconds)")
    print(f"  Audio range: [{audio.min().item():.4f}, {audio.max().item():.4f}]")
    print(f"  Audio mean: {audio.mean().item():.4f}")
    print(f"  Audio std: {audio.std().item():.4f}")
    
    return audio


def main():
    parser = argparse.ArgumentParser(description="Test Vocoder with audio quality metrics")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/vocoder_tiny",
                       help="Path to Vocoder checkpoint directory")
    parser.add_argument("--num_samples", type=int, default=100,
                       help="Number of samples to evaluate (default: 100)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                       help="Device to use (cuda/cpu)")
    parser.add_argument("--quick", action="store_true",
                       help="Quick test with 10 samples")
    args = parser.parse_args()
    
    if args.quick:
        args.num_samples = 10
    
    print("=" * 70)
    print("VOCODER (HiFi-GAN) ACCURACY TEST")
    print("=" * 70)
    
    # Load model
    try:
        model, cfg = load_model_and_config(args.checkpoint, args.device)
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Evaluate on dataset
    try:
        metrics = evaluate_vocoder_quality(
            model, cfg,
            device=args.device,
            num_samples=args.num_samples,
            verbose=True
        )
        
        print_results(metrics)
        
    except Exception as e:
        print(f"✗ Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
        return


if __name__ == "__main__":
    main()