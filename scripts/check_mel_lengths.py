"""
Utility script to check the actual mel spectrogram lengths in your dataset.
This helps determine the optimal max_mel_length for CUDA graphs compatibility.

Usage:
    # For audio encoder (default: hop=160, 100 frames/sec)
    python scripts/check_mel_lengths.py --csv data/audio/production_asr.csv
    
    # For talker (frame_ms=80, 12.5 frames/sec)
    python scripts/check_mel_lengths.py --csv data/audio/production_tts.csv --talker
    
    # Sample subset for faster analysis
    python scripts/check_mel_lengths.py --csv data/audio/production_asr.csv --sample 1000
"""

import argparse
import csv
import torch
import torchaudio
from collections import Counter
import numpy as np
from omni.utils import load_audio

def analyze_mel_lengths(csv_path, sample_size=None, sr=16000, hop_length=None, frame_ms=None, is_talker=False):
    """
    Analyze mel spectrogram lengths in the dataset.
    
    Args:
        csv_path: Path to CSV file with 'wav' column
        sample_size: Number of samples to check (None = all)
        sr: Sample rate (default: 16000)
        hop_length: Hop length for mel spectrogram (overrides frame_ms if provided)
        frame_ms: Frame duration in milliseconds (for talker, default: 80ms)
        is_talker: Whether this is for talker training (uses frame_ms calculation)
    """
    # Determine hop_length based on configuration
    if hop_length is None:
        if is_talker or frame_ms is not None:
            # Talker configuration: frame_ms=80
            frame_ms = frame_ms or 80
            hop_length = int(sr * frame_ms / 1000)
            win_length = min(1024, hop_length * 4)
            config_type = "Talker"
        else:
            # Audio encoder configuration: hop=160
            hop_length = 160
            win_length = 400
            config_type = "Audio Encoder"
    else:
        win_length = 400
        config_type = "Custom"
    
    frame_rate = sr / hop_length
    
    print(f"Analyzing mel spectrogram lengths in: {csv_path}")
    print(f"Configuration: {config_type}")
    print(f"Sample rate: {sr} Hz")
    if is_talker or frame_ms:
        print(f"Frame duration: {frame_ms}ms")
    print(f"Hop length: {hop_length}")
    print(f"Frame rate: {frame_rate:.1f} frames/second")
    print(f"60 seconds = {60 * frame_rate:.0f} frames")
    print("-" * 60)
    
    melspec = torchaudio.transforms.MelSpectrogram(
        sample_rate=sr, 
        n_fft=1024, 
        hop_length=hop_length, 
        win_length=win_length, 
        n_mels=128
    )
    
    lengths = []
    errors = 0
    
    with open(csv_path, 'r', encoding='utf-8', errors='ignore') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        
        if sample_size:
            import random
            rows = random.sample(rows, min(sample_size, len(rows)))
        
        total = len(rows)
        print(f"Checking {total} audio files...")
        
        for idx, row in enumerate(rows):
            if (idx + 1) % 100 == 0:
                print(f"  Processed {idx + 1}/{total} files...", end='\r')
            
            try:
                wav_path = row.get("wav") or row.get("audio")
                if not wav_path:
                    continue
                
                wav, file_sr = load_audio(wav_path)
                if file_sr != sr:
                    wav = torchaudio.transforms.Resample(file_sr, sr)(wav)
                
                mel = melspec(wav)[0].T  # (T, 128)
                length = mel.shape[0]
                lengths.append(length)
                
            except Exception as e:
                errors += 1
                continue
        
        print(f"\n  Processed {total} files ({errors} errors)")
    
    if not lengths:
        print("No valid audio files found!")
        return
    
    lengths = np.array(lengths)
    
    # Statistics
    print("\n" + "=" * 60)
    print("MEL SPECTROGRAM LENGTH STATISTICS")
    print("=" * 60)
    print(f"Total samples analyzed: {len(lengths)}")
    print(f"\nFrame length statistics:")
    print(f"  Min:     {lengths.min():.0f} frames ({lengths.min() / frame_rate:.2f} seconds)")
    print(f"  Max:     {lengths.max():.0f} frames ({lengths.max() / frame_rate:.2f} seconds)")
    print(f"  Mean:    {lengths.mean():.1f} frames ({lengths.mean() / frame_rate:.2f} seconds)")
    print(f"  Median:  {np.median(lengths):.1f} frames ({np.median(lengths) / frame_rate:.2f} seconds)")
    print(f"  Std:     {lengths.std():.1f} frames ({lengths.std() / frame_rate:.2f} seconds)")
    
    # Percentiles
    print(f"\nPercentiles:")
    for p in [50, 75, 90, 95, 99, 99.5, 99.9]:
        val = np.percentile(lengths, p)
        print(f"  {p:5.1f}%: {val:7.0f} frames ({val / frame_rate:6.2f} seconds)")
    
    # Distribution
    print(f"\nLength distribution (rounded to nearest 100):")
    bins = np.arange(0, lengths.max() + 200, 200)
    hist, _ = np.histogram(lengths, bins=bins)
    for i, count in enumerate(hist[:20]):  # Show first 20 bins
        if count > 0:
            frame_range = f"{bins[i]:.0f}-{bins[i+1]:.0f}"
            sec_range = f"{bins[i]/frame_rate:.1f}-{bins[i+1]/frame_rate:.1f}s"
            print(f"  {frame_range:>12} frames ({sec_range:>10}): {count:>6} files")
    
    # Recommendations
    print("\n" + "=" * 60)
    print("RECOMMENDATIONS FOR max_mel_length")
    print("=" * 60)
    
    max_length = lengths.max()
    p99 = np.percentile(lengths, 99)
    p99_5 = np.percentile(lengths, 99.5)
    p99_9 = np.percentile(lengths, 99.9)
    
    # Round up to nearest 256 for better memory alignment
    def round_up_to_256(x):
        return int(np.ceil(x / 256) * 256)
    
    rec_min = round_up_to_256(max_length)
    rec_p99 = round_up_to_256(p99)
    rec_p99_5 = round_up_to_256(p99_5)
    rec_p99_9 = round_up_to_256(p99_9)
    
    # Determine default based on configuration type
    if is_talker:
        default_max = 750  # 60 seconds at 12.5 Hz
    else:
        default_max = 6000  # 60 seconds at 100 Hz
    
    print(f"\nCurrent max_mel_length: {default_max} frames ({default_max / frame_rate:.2f} seconds)")
    print(f"\nBased on your data:")
    print(f"  Maximum found:     {max_length:.0f} frames ({max_length / frame_rate:.2f} seconds)")
    print(f"  99th percentile:   {p99:.0f} frames ({p99 / frame_rate:.2f} seconds)")
    print(f"  99.5th percentile: {p99_5:.0f} frames ({p99_5 / frame_rate:.2f} seconds)")
    print(f"  99.9th percentile: {p99_9:.0f} frames ({p99_9 / frame_rate:.2f} seconds)")
    
    print(f"\nRecommended values (rounded to 256 for memory alignment):")
    print(f"  Conservative (covers 100%): {rec_min} frames ({rec_min / frame_rate:.2f} seconds)")
    print(f"  Balanced (covers 99.5%):    {rec_p99_5} frames ({rec_p99_5 / frame_rate:.2f} seconds)")
    print(f"  Aggressive (covers 99%):     {rec_p99} frames ({rec_p99 / frame_rate:.2f} seconds)")
    
    # For 1 minute coverage
    one_minute_frames = int(60 * frame_rate)
    print(f"\nFor 1 minute coverage:")
    print(f"  Required: {one_minute_frames} frames (60 seconds × {frame_rate:.1f} fps)")
    rec_one_min = round_up_to_256(one_minute_frames)
    print(f"  Recommended (aligned): {rec_one_min} frames")
    
    if max_length > default_max:
        print(f"\n⚠️  WARNING: Maximum length ({max_length:.0f} frames) exceeds current max_mel_length ({default_max})")
        print(f"   You should increase max_mel_length to at least {rec_min} frames")
    elif p99_5 > default_max:
        print(f"\n⚠️  WARNING: 99.5th percentile ({p99_5:.0f} frames) exceeds current max_mel_length ({default_max})")
        print(f"   Consider increasing to {rec_p99_5} frames to avoid truncating 0.5% of data")
    else:
        print(f"\n✓ Current max_mel_length ({default_max}) should be sufficient for your dataset")
        if p99 < default_max * 0.9:
            print(f"   Note: You could reduce to {rec_p99} frames to save memory (covers 99% of data)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check mel spectrogram lengths in dataset")
    parser.add_argument("--csv", required=True, help="Path to CSV file with audio paths")
    parser.add_argument("--sample", type=int, default=None, help="Sample size (check only N files, default: all)")
    parser.add_argument("--sr", type=int, default=16000, help="Sample rate (default: 16000)")
    parser.add_argument("--hop", type=int, default=None, help="Hop length (overrides --talker if provided)")
    parser.add_argument("--talker", action="store_true", help="Use talker configuration (frame_ms=80, hop=1280)")
    parser.add_argument("--frame-ms", type=int, default=None, help="Frame duration in ms (for talker, default: 80)")
    
    args = parser.parse_args()
    analyze_mel_lengths(args.csv, args.sample, args.sr, args.hop, args.frame_ms, args.talker)

