"""
Analyze vocoder dataset to determine optimal max_audio_length.

This script analyzes your audio dataset and calculates the optimal max_audio_length
using percentile-based approach to minimize padding while covering most of your data.

Usage:
    python scripts/analyze_audio_lengths.py --csv data/audio/production_tts.csv
    python scripts/analyze_audio_lengths.py --csv data/audio/production_tts.csv --percentile 99
    python scripts/analyze_audio_lengths.py --csv data/audio/production_tts.csv --samples 1000
"""

import argparse
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from omni.utils import analyze_vocoder_dataset

def main():
    parser = argparse.ArgumentParser(description="Analyze vocoder dataset audio lengths")
    parser.add_argument("--csv", type=str, required=True, help="Path to CSV file with 'wav' column")
    parser.add_argument("--sr", type=int, default=16000, help="Sample rate (default: 16000)")
    parser.add_argument("--percentile", type=float, default=95.0, 
                       help="Percentile to use for max_audio_length (default: 95.0)")
    parser.add_argument("--samples", type=int, default=None,
                       help="Number of samples to analyze (default: all)")
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("VOCODER DATASET AUDIO LENGTH ANALYSIS")
    print("=" * 70)
    print(f"CSV Path: {args.csv}")
    print(f"Sample Rate: {args.sr} Hz")
    print(f"Percentile: {args.percentile}%")
    if args.samples:
        print(f"Analyzing: {args.samples} samples (random subset)")
    else:
        print(f"Analyzing: All samples")
    print("=" * 70)
    print()
    
    # Analyze dataset
    max_audio_length = analyze_vocoder_dataset(
        csv_path=args.csv,
        sr=args.sr,
        sample_size=args.samples,
        audio_percentile=args.percentile
    )
    
    print("\n" + "=" * 70)
    print("RECOMMENDATIONS")
    print("=" * 70)
    print(f"\n✅ Add to your vocoder config (configs/vocoder_tiny.json):")
    print(f'   "max_audio_length": {max_audio_length},')
    print(f'   "max_audio_length_percentile": {args.percentile},')
    
    print(f"\n💡 Memory Considerations:")
    memory_per_sample = max_audio_length * 4 / 1024 / 1024  # 4 bytes per float32, convert to MB
    print(f"   • Each audio sample: ~{memory_per_sample:.2f} MB")
    print(f"   • Batch size 16: ~{16 * memory_per_sample:.2f} MB")
    
    print(f"\n🎯 Percentile Options:")
    print(f"   • 90th percentile: Less padding, more truncation (10% truncated)")
    print(f"   • 95th percentile: Balanced (5% truncated) ⭐ RECOMMENDED")
    print(f"   • 99th percentile: Minimal truncation, more padding (1% truncated)")
    print(f"   • 100th percentile: No truncation, maximum padding")
    
    print("\n" + "=" * 70)

if __name__ == "__main__":
    main()
