"""
Update training configs based on actual dataset sizes
Analyzes data folders and adjusts epochs, max_steps, warmup, and other parameters
"""

import os
import json
import csv
import argparse
from pathlib import Path
from typing import Dict, Tuple, Optional

# Best practices for training duration based on dataset size
# (samples, min_epochs, max_epochs, recommended_epochs)
EPOCH_RECOMMENDATIONS = [
    (1000000, 1, 3, 2),      # Very large (>1M): 1-3 epochs
    (500000, 2, 4, 3),       # Large (500K-1M): 2-4 epochs
    (100000, 3, 6, 4),       # Medium (100K-500K): 3-6 epochs
    (50000, 5, 10, 7),       # Small (50K-100K): 5-10 epochs
    (0, 10, 20, 15),         # Very small (<50K): 10-20 epochs
]

def count_text_samples(text_path: str) -> int:
    """Count lines in text file"""
    if not os.path.exists(text_path):
        return 0
    count = 0
    try:
        with open(text_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                if line.strip():
                    count += 1
    except Exception as e:
        print(f"Warning: Could not count text samples from {text_path}: {e}")
    return count

def count_image_samples(manifest_path: str) -> int:
    """Count entries in image JSON manifest"""
    if not os.path.exists(manifest_path):
        return 0
    try:
        with open(manifest_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if isinstance(data, list):
                return len(data)
            elif isinstance(data, dict) and 'images' in data:
                return len(data['images'])
            return 0
    except Exception as e:
        print(f"Warning: Could not count image samples from {manifest_path}: {e}")
        return 0

def count_audio_samples(csv_path: str) -> int:
    """Count rows in audio CSV file"""
    if not os.path.exists(csv_path):
        return 0
    count = 0
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            count = sum(1 for _ in reader)
    except Exception as e:
        print(f"Warning: Could not count audio samples from {csv_path}: {e}")
        return 0
    return count

def get_recommended_epochs(num_samples: int) -> Tuple[int, int, int]:
    """Get recommended epoch range based on dataset size"""
    for threshold, min_epochs, max_epochs, recommended in EPOCH_RECOMMENDATIONS:
        if num_samples >= threshold:
            return min_epochs, max_epochs, recommended
    return 10, 20, 15  # Default for very small datasets

def calculate_training_params(
    num_samples: int,
    batch_size: int,
    gradient_accumulation: int = 1,
    val_split: float = 0.1
) -> Dict:
    """Calculate training parameters based on dataset size"""
    # Effective batch size
    effective_batch = batch_size * gradient_accumulation
    
    # Training samples (excluding validation)
    train_samples = int(num_samples * (1 - val_split))
    
    # Steps per epoch
    steps_per_epoch = max(1, train_samples // effective_batch)
    
    # Get recommended epochs
    min_epochs, max_epochs, recommended_epochs = get_recommended_epochs(num_samples)
    
    # Calculate max_steps (use recommended epochs)
    max_steps = steps_per_epoch * recommended_epochs
    
    # Warmup steps: 5-10% of total steps, but at least 100, max 10% of max_steps
    warmup_steps = max(100, min(int(max_steps * 0.1), int(max_steps * 0.05)))
    warmup_steps = min(warmup_steps, 10000)  # Cap at 10K
    
    # Validation frequency: every 500-1000 steps, or 10% of steps per epoch
    val_freq = max(100, min(1000, steps_per_epoch // 10))
    
    # Checkpoint frequency: every 5000-10000 steps, or 1 per epoch
    checkpoint_freq = max(1000, min(10000, steps_per_epoch))
    
    return {
        "num_samples": num_samples,
        "train_samples": train_samples,
        "steps_per_epoch": steps_per_epoch,
        "min_epochs": min_epochs,
        "max_epochs": max_epochs,
        "recommended_epochs": recommended_epochs,
        "max_steps": max_steps,
        "warmup_steps": warmup_steps,
        "val_freq": val_freq,
        "checkpoint_freq": checkpoint_freq,
    }

def update_data_paths(config: Dict) -> Dict:
    """Update data paths to use production or synthetic files if they exist"""
    # Text paths - only production or synthetic
    if "train_text" in config:
        if not os.path.exists(config["train_text"]):
            # Try production first, then synthetic
            if os.path.exists("data/text/production_corpus.txt"):
                print(f"  → Updating train_text: {config['train_text']} → data/text/production_corpus.txt")
                config["train_text"] = "data/text/production_corpus.txt"
            elif os.path.exists("data/text/tiny_corpus.txt"):
                print(f"  → Updating train_text: {config['train_text']} → data/text/tiny_corpus.txt")
                config["train_text"] = "data/text/tiny_corpus.txt"
    
    # Image paths - only production or synthetic
    if "train_manifest" in config:
        if not os.path.exists(config["train_manifest"]):
            # Try production first, then synthetic
            if os.path.exists("data/images/production_annotations.json"):
                print(f"  → Updating train_manifest: {config['train_manifest']} → data/images/production_annotations.json")
                config["train_manifest"] = "data/images/production_annotations.json"
            elif os.path.exists("data/images/annotations.json"):
                print(f"  → Updating train_manifest: {config['train_manifest']} → data/images/annotations.json")
                config["train_manifest"] = "data/images/annotations.json"
    
    # Audio ASR paths - only production or synthetic
    if "train_csv" in config:
        if not os.path.exists(config["train_csv"]):
            # Try production first, then synthetic
            if os.path.exists("data/audio/production_asr.csv"):
                print(f"  → Updating train_csv: {config['train_csv']} → data/audio/production_asr.csv")
                config["train_csv"] = "data/audio/production_asr.csv"
            elif os.path.exists("data/audio/asr.csv"):
                print(f"  → Updating train_csv: {config['train_csv']} → data/audio/asr.csv")
                config["train_csv"] = "data/audio/asr.csv"
    
    # Audio TTS paths - only production or synthetic
    if "tts_csv" in config:
        if not os.path.exists(config["tts_csv"]):
            # Try production first, then synthetic
            if os.path.exists("data/audio/production_tts.csv"):
                print(f"  → Updating tts_csv: {config['tts_csv']} → data/audio/production_tts.csv")
                config["tts_csv"] = "data/audio/production_tts.csv"
            elif os.path.exists("data/audio/tts.csv"):
                print(f"  → Updating tts_csv: {config['tts_csv']} → data/audio/tts.csv")
                config["tts_csv"] = "data/audio/tts.csv"
    
    # Multimodal paths - only production or synthetic
    if "sft_mix" in config:
        sft = config["sft_mix"]
        if "text_path" in sft:
            if not os.path.exists(sft["text_path"]):
                # Try production first, then synthetic
                if os.path.exists("data/text/production_corpus.txt"):
                    print(f"  → Updating sft_mix.text_path: {sft['text_path']} → data/text/production_corpus.txt")
                    sft["text_path"] = "data/text/production_corpus.txt"
                elif os.path.exists("data/text/tiny_corpus.txt"):
                    print(f"  → Updating sft_mix.text_path: {sft['text_path']} → data/text/tiny_corpus.txt")
                    sft["text_path"] = "data/text/tiny_corpus.txt"
        if "image_manifest" in sft:
            if not os.path.exists(sft["image_manifest"]):
                # Try production first, then synthetic
                if os.path.exists("data/images/production_annotations.json"):
                    print(f"  → Updating sft_mix.image_manifest: {sft['image_manifest']} → data/images/production_annotations.json")
                    sft["image_manifest"] = "data/images/production_annotations.json"
                elif os.path.exists("data/images/annotations.json"):
                    print(f"  → Updating sft_mix.image_manifest: {sft['image_manifest']} → data/images/annotations.json")
                    sft["image_manifest"] = "data/images/annotations.json"
        if "asr_csv" in sft:
            if not os.path.exists(sft["asr_csv"]):
                # Try production first, then synthetic
                if os.path.exists("data/audio/production_asr.csv"):
                    print(f"  → Updating sft_mix.asr_csv: {sft['asr_csv']} → data/audio/production_asr.csv")
                    sft["asr_csv"] = "data/audio/production_asr.csv"
                elif os.path.exists("data/audio/asr.csv"):
                    print(f"  → Updating sft_mix.asr_csv: {sft['asr_csv']} → data/audio/asr.csv")
                    sft["asr_csv"] = "data/audio/asr.csv"
    
    return config

def update_config_file(
    config_path: str,
    params: Dict,
    dataset_type: str = "general",
    preserve_keys: Optional[list] = None,
    update_paths: bool = True
):
    """Update config file with calculated parameters"""
    if not os.path.exists(config_path):
        print(f"Warning: Config file not found: {config_path}")
        return False
    
    # Load existing config
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Preserve certain keys if specified
    if preserve_keys:
        preserved = {k: config.get(k) for k in preserve_keys if k in config}
    else:
        preserved = {}
    
    # Update data paths if requested
    if update_paths:
        config = update_data_paths(config)
    
    # Update training parameters
    config["max_steps"] = params["max_steps"]
    config["warmup_steps"] = params["warmup_steps"]
    config["max_epochs"] = params["recommended_epochs"]
    config["val_freq"] = params["val_freq"]
    config["checkpoint_freq"] = params["checkpoint_freq"]
    
    # Restore preserved keys (but keep updated paths)
    for key in preserved:
        if key in config and isinstance(config[key], dict) and isinstance(preserved[key], dict):
            # Merge dicts (e.g., sft_mix)
            config[key].update(preserved[key])
        else:
            config[key] = preserved[key]
    
    # Save updated config
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"✓ Updated {config_path}")
    return True

def analyze_and_update_all_configs(data_dir: str = "data", configs_dir: str = "configs", dry_run: bool = False):
    """Analyze all datasets and update all config files"""
    print("="*60)
    print("Analyzing Datasets and Updating Configs")
    print("="*60)
    
    # Analyze text dataset - only production and synthetic files
    print("\n[1] Analyzing Text Dataset...")
    text_files = [
        "data/text/production_corpus.txt",  # Production file
        "data/text/tiny_corpus.txt",        # Synthetic file from make_synthetic_datasets.py
    ]
    text_samples = 0
    text_path = None
    for tf in text_files:
        if os.path.exists(tf):
            count = count_text_samples(tf)
            if count > text_samples:
                text_samples = count
                text_path = tf
    print(f"  Text samples: {text_samples:,} (from {text_path or 'N/A'})")
    
    # Analyze image dataset - only production and synthetic files
    print("\n[2] Analyzing Image Dataset...")
    image_files = [
        "data/images/production_annotations.json",  # Production file
        "data/images/annotations.json",           # Synthetic file from make_synthetic_datasets.py
    ]
    image_samples = 0
    image_manifest = None
    for imf in image_files:
        if os.path.exists(imf):
            count = count_image_samples(imf)
            if count > image_samples:
                image_samples = count
                image_manifest = imf
    print(f"  Image samples: {image_samples:,} (from {image_manifest or 'N/A'})")
    
    # Analyze audio dataset - only production and synthetic files
    print("\n[3] Analyzing Audio Dataset...")
    audio_files = [
        "data/audio/production_asr.csv",  # Production file
        "data/audio/asr.csv",             # Synthetic file from make_synthetic_datasets.py
    ]
    audio_samples = 0
    audio_csv = None
    for af in audio_files:
        if os.path.exists(af):
            count = count_audio_samples(af)
            if count > audio_samples:
                audio_samples = count
                audio_csv = af
    print(f"  Audio samples: {audio_samples:,} (from {audio_csv or 'N/A'})")
    
    # Calculate parameters for each training stage
    print("\n" + "="*60)
    print("Calculating Training Parameters")
    print("="*60)
    
    # Stage A: Text-only (thinker_tiny.json)
    print("\n[Stage A] Text-only Training (thinker_tiny.json)")
    if text_samples > 0:
        text_params = calculate_training_params(
            text_samples,
            batch_size=8,  # From config
            gradient_accumulation=1,
            val_split=0.1
        )
        print(f"  Samples: {text_params['num_samples']:,}")
        print(f"  Steps/epoch: {text_params['steps_per_epoch']:,}")
        print(f"  Recommended epochs: {text_params['recommended_epochs']}")
        print(f"  Max steps: {text_params['max_steps']:,}")
        print(f"  Warmup steps: {text_params['warmup_steps']:,}")
        
        if not dry_run:
            update_config_file(
                "configs/thinker_tiny.json",
                text_params,
                preserve_keys=["train_text"]  # Preserve data path
            )
    else:
        print("  ⚠ No text data found, skipping...")
    
    # Stage B: Audio encoder (audio_enc_tiny.json)
    print("\n[Stage B] Audio Encoder Training (audio_enc_tiny.json)")
    if audio_samples > 0:
        audio_params = calculate_training_params(
            audio_samples,
            batch_size=4,  # From config
            gradient_accumulation=1,
            val_split=0.1
        )
        print(f"  Samples: {audio_params['num_samples']:,}")
        print(f"  Steps/epoch: {audio_params['steps_per_epoch']:,}")
        print(f"  Recommended epochs: {audio_params['recommended_epochs']}")
        print(f"  Max steps: {audio_params['max_steps']:,}")
        print(f"  Warmup steps: {audio_params['warmup_steps']:,}")
        
        if not dry_run:
            update_config_file(
                "configs/audio_enc_tiny.json",
                audio_params,
                preserve_keys=["train_csv"]  # Preserve data path
            )
    else:
        print("  ⚠ No audio data found, skipping...")
    
    # Stage C: Vision encoder (vision_tiny.json)
    print("\n[Stage C] Vision Encoder Training (vision_tiny.json)")
    if image_samples > 0:
        vision_params = calculate_training_params(
            image_samples,
            batch_size=8,  # From config
            gradient_accumulation=1,
            val_split=0.1
        )
        print(f"  Samples: {vision_params['num_samples']:,}")
        print(f"  Steps/epoch: {vision_params['steps_per_epoch']:,}")
        print(f"  Recommended epochs: {vision_params['recommended_epochs']}")
        print(f"  Max steps: {vision_params['max_steps']:,}")
        print(f"  Warmup steps: {vision_params['warmup_steps']:,}")
        
        if not dry_run:
            update_config_file(
                "configs/vision_tiny.json",
                vision_params,
                preserve_keys=["train_manifest", "image_root"]  # Preserve data paths
            )
    else:
        print("  ⚠ No image data found, skipping...")
    
    # Stage D: Talker (talker_tiny.json) - uses TTS data
    print("\n[Stage D] Talker Training (talker_tiny.json)")
    # Only check production and synthetic TTS files
    tts_files = [
        "data/audio/production_tts.csv",  # Production file
        "data/audio/tts.csv",             # Synthetic file from make_synthetic_datasets.py
    ]
    tts_samples = 0
    tts_csv = None
    for ttf in tts_files:
        if os.path.exists(ttf):
            count = count_audio_samples(ttf)
            if count > tts_samples:
                tts_samples = count
                tts_csv = ttf
    if tts_samples > 0:
        talker_params = calculate_training_params(
            tts_samples,
            batch_size=4,  # From config
            gradient_accumulation=1,
            val_split=0.1
        )
        print(f"  Samples: {talker_params['num_samples']:,}")
        print(f"  Steps/epoch: {talker_params['steps_per_epoch']:,}")
        print(f"  Recommended epochs: {talker_params['recommended_epochs']}")
        print(f"  Max steps: {talker_params['max_steps']:,}")
        print(f"  Warmup steps: {talker_params['warmup_steps']:,}")
        
        if not dry_run:
            update_config_file(
                "configs/talker_tiny.json",
                talker_params,
                preserve_keys=["tts_csv"]  # Preserve data path
            )
    else:
        print("  ⚠ No TTS data found, skipping...")
    
    # Stage E: Multimodal SFT (omni_sft_tiny.json)
    print("\n[Stage E] Multimodal SFT Training (omni_sft_tiny.json)")
    # Use the maximum of all modalities for multimodal training
    multimodal_samples = max(text_samples, image_samples, audio_samples)
    if multimodal_samples > 0:
        sft_params = calculate_training_params(
            multimodal_samples,
            batch_size=2,  # From config
            gradient_accumulation=4,  # From config
            val_split=0.1
        )
        print(f"  Samples (max across modalities): {sft_params['num_samples']:,}")
        print(f"  Steps/epoch: {sft_params['steps_per_epoch']:,}")
        print(f"  Recommended epochs: {sft_params['recommended_epochs']}")
        print(f"  Max steps: {sft_params['max_steps']:,}")
        print(f"  Warmup steps: {sft_params['warmup_steps']:,}")
        
        if not dry_run:
            update_config_file(
                "configs/omni_sft_tiny.json",
                sft_params,
                preserve_keys=["sft_mix"]  # Preserve data paths
            )
    else:
        print("  ⚠ No multimodal data found, skipping...")
    
    print("\n" + "="*60)
    if dry_run:
        print("DRY RUN: No files were modified")
        print("Run without --dry-run to apply changes")
    else:
        print("✓ All configs updated successfully!")
    print("="*60)
    
    # Print summary
    print("\nSummary:")
    print(f"  Text samples: {text_samples:,}")
    print(f"  Image samples: {image_samples:,}")
    print(f"  Audio samples: {audio_samples:,}")
    print(f"  Multimodal (max): {multimodal_samples:,}")

def main():
    parser = argparse.ArgumentParser(
        description="Update training configs based on actual dataset sizes"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Directory containing datasets (default: data)"
    )
    parser.add_argument(
        "--configs-dir",
        type=str,
        default="configs",
        help="Directory containing config files (default: configs)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be changed without modifying files"
    )
    args = parser.parse_args()
    
    analyze_and_update_all_configs(
        data_dir=args.data_dir,
        configs_dir=args.configs_dir,
        dry_run=args.dry_run
    )

if __name__ == "__main__":
    main()

