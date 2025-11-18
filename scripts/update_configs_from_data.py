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

# Import tokenizer (required)
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from omni.tokenizer import BPETokenizer

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

def get_or_create_tokenizer(text_path: str, tokenizer_path: Optional[str] = None) -> BPETokenizer:
    """Get existing tokenizer or create one from text data"""
    # Try to find existing tokenizer
    tokenizer_candidates = [
        tokenizer_path,  # Explicitly provided
        "checkpoints/thinker_tiny/tokenizer.model",  # Default location
        "tokenizer.model",  # Current directory
    ]
    
    for candidate in tokenizer_candidates:
        if candidate and os.path.exists(candidate):
            return BPETokenizer(candidate)
    
    # No tokenizer found - create one from text data
    if not os.path.exists(text_path):
        raise FileNotFoundError(f"Cannot create tokenizer: text file not found: {text_path}")
    
    print(f"  No tokenizer found. Creating tokenizer from {text_path}...")
    os.makedirs("checkpoints/thinker_tiny", exist_ok=True)
    tokenizer_model = "checkpoints/thinker_tiny/tokenizer.model"
    BPETokenizer.train_new(text_path, tokenizer_model, vocab_size=32000)
    print(f"  ✓ Tokenizer created: {tokenizer_model}")
    return BPETokenizer(tokenizer_model)

def count_text_tokens(text_path: str, tokenizer_path: Optional[str] = None) -> int:
    """Count actual tokens in text file using tokenizer"""
    if not os.path.exists(text_path):
        return 0
    
    try:
        tokenizer = get_or_create_tokenizer(text_path, tokenizer_path)
        total_tokens = 0
        sample_count = 0
        
        with open(text_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                line = line.strip()
                if line:
                    tokens = tokenizer.encode(line)
                    total_tokens += len(tokens)
                    sample_count += 1
                    
                    # Progress indicator for large files
                    if sample_count % 10000 == 0:
                        print(f"  Counting tokens: {sample_count:,} samples, {total_tokens:,} tokens so far...")
        
        return total_tokens
    except Exception as e:
        print(f"Error: Could not count tokens from {text_path}: {e}")
        raise

def count_csv_tokens(csv_path: str, text_column: str = 'text', tokenizer_path: Optional[str] = None) -> int:
    """Count tokens from CSV file (for audio transcriptions or OCR text)"""
    if not os.path.exists(csv_path):
        return 0
    
    try:
        # Get tokenizer (use first text file we can find, or create from CSV text)
        text_files = [
            "data/text/production_corpus.txt",
            "data/text/tiny_corpus.txt",
        ]
        tokenizer = None
        for tf in text_files:
            if os.path.exists(tf):
                tokenizer = get_or_create_tokenizer(tf, tokenizer_path)
                break
        
        if not tokenizer:
            # Create tokenizer from CSV text
            print(f"  Creating tokenizer from CSV text...")
            # Extract all text to temp file
            temp_text = f"data/.temp_csv_{text_column}.txt"
            with open(csv_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                with open(temp_text, 'w', encoding='utf-8') as out:
                    for row in reader:
                        text = row.get(text_column, '').strip()
                        if text:
                            out.write(text + '\n')
            os.makedirs("checkpoints/thinker_tiny", exist_ok=True)
            tokenizer_model = "checkpoints/thinker_tiny/tokenizer.model"
            BPETokenizer.train_new(temp_text, tokenizer_model, vocab_size=32000)
            tokenizer = BPETokenizer(tokenizer_model)
            os.remove(temp_text)  # Clean up
        
        total_tokens = 0
        sample_count = 0
        
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                text = row.get(text_column, '').strip()
                if text:
                    tokens = tokenizer.encode(text)
                    total_tokens += len(tokens)
                    sample_count += 1
                    
                    if sample_count % 10000 == 0:
                        print(f"  Counting tokens: {sample_count:,} samples, {total_tokens:,} tokens so far...")
        
        return total_tokens
    except Exception as e:
        print(f"Error: Could not count tokens from {csv_path}: {e}")
        raise

def count_audio_tokens(csv_path: str, tokenizer_path: Optional[str] = None) -> int:
    """Count tokens from audio transcriptions (text column in CSV)"""
    return count_csv_tokens(csv_path, 'text', tokenizer_path)

def count_ocr_tokens(csv_path: str, tokenizer_path: Optional[str] = None) -> int:
    """Count tokens from OCR text (text column in CSV)"""
    return count_csv_tokens(csv_path, 'text', tokenizer_path)

def count_image_tokens(manifest_path: str, tokenizer_path: Optional[str] = None) -> int:
    """Count tokens from image captions"""
    if not os.path.exists(manifest_path):
        return 0
    
    try:
        # Get tokenizer (use first text file we can find, or create from captions)
        text_files = [
            "data/text/production_corpus.txt",
            "data/text/tiny_corpus.txt",
        ]
        tokenizer = None
        for tf in text_files:
            if os.path.exists(tf):
                tokenizer = get_or_create_tokenizer(tf, tokenizer_path)
                break
        
        if not tokenizer:
            # Create tokenizer from image captions
            print(f"  Creating tokenizer from image captions...")
            temp_text = "data/.temp_image_captions.txt"
            with open(manifest_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                with open(temp_text, 'w', encoding='utf-8') as out:
                    if isinstance(data, list):
                        for item in data:
                            caption = item.get('caption', '').strip()
                            if caption:
                                out.write(caption + '\n')
                    elif isinstance(data, dict) and 'images' in data:
                        for item in data['images']:
                            caption = item.get('caption', '').strip()
                            if caption:
                                out.write(caption + '\n')
            os.makedirs("checkpoints/thinker_tiny", exist_ok=True)
            tokenizer_model = "checkpoints/thinker_tiny/tokenizer.model"
            BPETokenizer.train_new(temp_text, tokenizer_model, vocab_size=32000)
            tokenizer = BPETokenizer(tokenizer_model)
            os.remove(temp_text)  # Clean up
        
        total_tokens = 0
        sample_count = 0
        
        with open(manifest_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            items = data if isinstance(data, list) else data.get('images', [])
            for item in items:
                caption = item.get('caption', '').strip()
                if caption:
                    tokens = tokenizer.encode(caption)
                    total_tokens += len(tokens)
                    sample_count += 1
                    
                    if sample_count % 10000 == 0:
                        print(f"  Counting tokens: {sample_count:,} samples, {total_tokens:,} tokens so far...")
        
        return total_tokens
    except Exception as e:
        print(f"Error: Could not count tokens from {manifest_path}: {e}")
        raise

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

def update_data_paths(config: Dict, config_path: str = "") -> Dict:
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
    
    # OCR paths - only production or synthetic
    # Check if this is an OCR config by checking config_path or save_dir
    is_ocr_config = False
    if config_path:
        is_ocr_config = "ocr" in config_path.lower()
    elif "save_dir" in config:
        is_ocr_config = "ocr" in config["save_dir"].lower()
    
    if is_ocr_config and "train_csv" in config:
        if not os.path.exists(config["train_csv"]):
            # Try production first, then synthetic
            if os.path.exists("data/ocr/production_ocr.csv"):
                print(f"  → Updating train_csv: {config['train_csv']} → data/ocr/production_ocr.csv")
                config["train_csv"] = "data/ocr/production_ocr.csv"
            elif os.path.exists("data/ocr/ocr_train.csv"):
                print(f"  → Updating train_csv: {config['train_csv']} → data/ocr/ocr_train.csv")
                config["train_csv"] = "data/ocr/ocr_train.csv"
    
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
        # Pass config_path to update_data_paths for OCR detection
        config = update_data_paths(config, config_path)
    
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
    text_tokens = 0
    text_path = None
    for tf in text_files:
        if os.path.exists(tf):
            count = count_text_samples(tf)
            if count > text_samples:
                text_samples = count
                text_path = tf
    
    # Count tokens (required)
    if text_path:
        print(f"  Counting tokens (this may take a while for large files)...")
        text_tokens = count_text_tokens(text_path)
        avg_tokens_per_sample = text_tokens / text_samples if text_samples > 0 else 0
        print(f"  Text samples: {text_samples:,} (from {text_path})")
        print(f"  Text tokens: {text_tokens:,} (~{text_tokens/1_000_000:.1f}M tokens)")
        print(f"  Average tokens per sample: {avg_tokens_per_sample:.1f}")
    else:
        print(f"  ⚠ No text data found")
    
    # Analyze image dataset - only production and synthetic files
    print("\n[2] Analyzing Image Dataset...")
    image_files = [
        "data/images/production_annotations.json",  # Production file
        "data/images/annotations.json",           # Synthetic file from make_synthetic_datasets.py
    ]
    image_samples = 0
    image_tokens = 0
    image_manifest = None
    for imf in image_files:
        if os.path.exists(imf):
            count = count_image_samples(imf)
            if count > image_samples:
                image_samples = count
                image_manifest = imf
    
    # Count tokens from captions (required)
    if image_manifest:
        print(f"  Counting tokens from captions (this may take a while for large files)...")
        image_tokens = count_image_tokens(image_manifest)
        avg_tokens_per_sample = image_tokens / image_samples if image_samples > 0 else 0
        print(f"  Image samples: {image_samples:,} (from {image_manifest})")
        print(f"  Caption tokens: {image_tokens:,} (~{image_tokens/1_000_000:.1f}M tokens)")
        print(f"  Average tokens per caption: {avg_tokens_per_sample:.1f}")
    else:
        print(f"  ⚠ No image data found")
    
    # Analyze audio dataset - only production and synthetic files
    print("\n[3] Analyzing Audio Dataset...")
    audio_files = [
        "data/audio/production_asr.csv",  # Production file
        "data/audio/asr.csv",             # Synthetic file from make_synthetic_datasets.py
    ]
    audio_samples = 0
    audio_tokens = 0
    audio_csv = None
    for af in audio_files:
        if os.path.exists(af):
            count = count_audio_samples(af)
            if count > audio_samples:
                audio_samples = count
                audio_csv = af
    
    # Count tokens from transcriptions (required)
    if audio_csv:
        print(f"  Counting tokens from transcriptions (this may take a while for large files)...")
        audio_tokens = count_audio_tokens(audio_csv)
        avg_tokens_per_sample = audio_tokens / audio_samples if audio_samples > 0 else 0
        print(f"  Audio samples: {audio_samples:,} (from {audio_csv})")
        print(f"  Transcription tokens: {audio_tokens:,} (~{audio_tokens/1_000_000:.1f}M tokens)")
        print(f"  Average tokens per transcription: {avg_tokens_per_sample:.1f}")
    else:
        print(f"  ⚠ No audio data found")
    
    # Calculate parameters for each training stage
    print("\n" + "="*60)
    print("Calculating Training Parameters")
    print("="*60)
    
    # Helper function to calculate params from tokens
    def calculate_params_from_tokens(num_tokens: int, batch_size: int, ctx_len: int, gradient_accumulation: int = 1, val_split: float = 0.1):
        """Calculate training parameters from token count"""
        effective_batch = batch_size * gradient_accumulation
        train_tokens = int(num_tokens * (1 - val_split))
        tokens_per_step = effective_batch * ctx_len
        steps_per_epoch = max(1, train_tokens // tokens_per_step)
        
        # Epoch recommendations based on token count
        if num_tokens >= 100_000_000:
            min_epochs, max_epochs, recommended_epochs = 1, 3, 2
        elif num_tokens >= 50_000_000:
            min_epochs, max_epochs, recommended_epochs = 2, 4, 3
        elif num_tokens >= 10_000_000:
            min_epochs, max_epochs, recommended_epochs = 3, 6, 4
        else:
            min_epochs, max_epochs, recommended_epochs = 5, 10, 7
        
        max_steps = steps_per_epoch * recommended_epochs
        warmup_steps = max(100, min(int(max_steps * 0.1), int(max_steps * 0.05)))
        warmup_steps = min(warmup_steps, 10000)
        val_freq = max(100, min(1000, steps_per_epoch // 10))
        checkpoint_freq = max(1000, min(10000, steps_per_epoch))
        
        return {
            "num_tokens": num_tokens,
            "train_tokens": train_tokens,
            "steps_per_epoch": steps_per_epoch,
            "min_epochs": min_epochs,
            "max_epochs": max_epochs,
            "recommended_epochs": recommended_epochs,
            "max_steps": max_steps,
            "warmup_steps": warmup_steps,
            "val_freq": val_freq,
            "checkpoint_freq": checkpoint_freq,
        }
    
    # Stage A: Text-only (thinker_tiny.json)
    print("\n[Stage A] Text-only Training (thinker_tiny.json)")
    if text_tokens > 0:
        # Load config to get ctx_len if available
        config_path = "configs/thinker_tiny.json"
        ctx_len = 512  # Default
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                cfg = json.load(f)
                ctx_len = cfg.get("ctx_len", 512)
        
        text_params = calculate_params_from_tokens(text_tokens, batch_size=8, ctx_len=ctx_len, gradient_accumulation=1)
        text_params["num_samples"] = text_samples  # Keep for reference
        
        print(f"  Tokens: {text_params['num_tokens']:,} (~{text_params['num_tokens']/1_000_000:.1f}M tokens)")
        print(f"  Steps/epoch: {text_params['steps_per_epoch']:,}")
        print(f"  Recommended epochs: {text_params['recommended_epochs']}")
        print(f"  Max steps: {text_params['max_steps']:,}")
        print(f"  Warmup steps: {text_params['warmup_steps']:,}")
        
        if not dry_run:
            update_config_file(
                config_path,
                text_params,
                preserve_keys=["train_text"]  # Preserve data path
            )
    else:
        print("  ⚠ No text data found, skipping...")
    
    # Stage B: Audio encoder (audio_enc_tiny.json)
    print("\n[Stage B] Audio Encoder Training (audio_enc_tiny.json)")
    if audio_tokens > 0:
        # Audio encoder processes transcription tokens
        # Use ctx_len from config or default
        config_path = "configs/audio_enc_tiny.json"
        ctx_len = 256  # Default for audio (shorter sequences)
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                cfg = json.load(f)
                ctx_len = cfg.get("ctx_len", 256)
        
        audio_params = calculate_params_from_tokens(audio_tokens, batch_size=4, ctx_len=ctx_len, gradient_accumulation=1)
        audio_params["num_samples"] = audio_samples  # Keep for reference
        
        print(f"  Tokens: {audio_params['num_tokens']:,} (~{audio_params['num_tokens']/1_000_000:.1f}M tokens)")
        print(f"  Steps/epoch: {audio_params['steps_per_epoch']:,}")
        print(f"  Recommended epochs: {audio_params['recommended_epochs']}")
        print(f"  Max steps: {audio_params['max_steps']:,}")
        print(f"  Warmup steps: {audio_params['warmup_steps']:,}")
        
        if not dry_run:
            update_config_file(
                config_path,
                audio_params,
                preserve_keys=["train_csv"]  # Preserve data path
            )
    else:
        print("  ⚠ No audio data found, skipping...")
    
    # Stage C: Vision encoder (vision_tiny.json)
    print("\n[Stage C] Vision Encoder Training (vision_tiny.json)")
    if image_tokens > 0:
        # Vision encoder processes caption tokens
        config_path = "configs/vision_tiny.json"
        ctx_len = 128  # Default for captions (shorter sequences)
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                cfg = json.load(f)
                ctx_len = cfg.get("ctx_len", 128)
        
        vision_params = calculate_params_from_tokens(image_tokens, batch_size=8, ctx_len=ctx_len, gradient_accumulation=1)
        vision_params["num_samples"] = image_samples  # Keep for reference
        
        print(f"  Tokens: {vision_params['num_tokens']:,} (~{vision_params['num_tokens']/1_000_000:.1f}M tokens)")
        print(f"  Steps/epoch: {vision_params['steps_per_epoch']:,}")
        print(f"  Recommended epochs: {vision_params['recommended_epochs']}")
        print(f"  Max steps: {vision_params['max_steps']:,}")
        print(f"  Warmup steps: {vision_params['warmup_steps']:,}")
        
        if not dry_run:
            update_config_file(
                config_path,
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
    tts_tokens = 0
    tts_csv = None
    for ttf in tts_files:
        if os.path.exists(ttf):
            count = count_audio_samples(ttf)
            if count > tts_samples:
                tts_samples = count
                tts_csv = ttf
    
    # Count tokens from TTS transcriptions
    if tts_csv:
        print(f"  Counting tokens from TTS transcriptions...")
        tts_tokens = count_audio_tokens(tts_csv)
    
    if tts_tokens > 0:
        config_path = "configs/talker_tiny.json"
        ctx_len = 256  # Default for TTS
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                cfg = json.load(f)
                ctx_len = cfg.get("ctx_len", 256)
        
        talker_params = calculate_params_from_tokens(tts_tokens, batch_size=4, ctx_len=ctx_len, gradient_accumulation=1)
        talker_params["num_samples"] = tts_samples  # Keep for reference
        
        print(f"  Tokens: {talker_params['num_tokens']:,} (~{talker_params['num_tokens']/1_000_000:.1f}M tokens)")
        print(f"  Steps/epoch: {talker_params['steps_per_epoch']:,}")
        print(f"  Recommended epochs: {talker_params['recommended_epochs']}")
        print(f"  Max steps: {talker_params['max_steps']:,}")
        print(f"  Warmup steps: {talker_params['warmup_steps']:,}")
        
        if not dry_run:
            update_config_file(
                config_path,
                talker_params,
                preserve_keys=["tts_csv"]  # Preserve data path
            )
    else:
        print("  ⚠ No TTS data found, skipping...")
    
    # Stage E: Multimodal SFT (omni_sft_tiny.json)
    print("\n[Stage E] Multimodal SFT Training (omni_sft_tiny.json)")
    # Use the maximum of all token counts for multimodal training
    multimodal_tokens = max(text_tokens, image_tokens, audio_tokens)
    if multimodal_tokens > 0:
        config_path = "configs/omni_sft_tiny.json"
        ctx_len = 512  # Default for multimodal
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                cfg = json.load(f)
                ctx_len = cfg.get("ctx_len", 512)
        
        sft_params = calculate_params_from_tokens(multimodal_tokens, batch_size=2, ctx_len=ctx_len, gradient_accumulation=4)
        sft_params["num_samples"] = max(text_samples, image_samples, audio_samples)  # Keep for reference
        
        print(f"  Tokens (max across modalities): {sft_params['num_tokens']:,} (~{sft_params['num_tokens']/1_000_000:.1f}M tokens)")
        print(f"  Steps/epoch: {sft_params['steps_per_epoch']:,}")
        print(f"  Recommended epochs: {sft_params['recommended_epochs']}")
        print(f"  Max steps: {sft_params['max_steps']:,}")
        print(f"  Warmup steps: {sft_params['warmup_steps']:,}")
        
        if not dry_run:
            update_config_file(
                config_path,
                sft_params,
                preserve_keys=["sft_mix"]  # Preserve data paths
            )
    else:
        print("  ⚠ No multimodal data found, skipping...")
    
    # OCR Training (ocr_tiny.json)
    print("\n[OCR] OCR Training (ocr_tiny.json)")
    ocr_files = [
        "data/ocr/production_ocr.csv",  # Production file
        "data/ocr/ocr_train.csv",        # Synthetic file from make_synthetic_datasets.py
    ]
    ocr_samples = 0
    ocr_tokens = 0
    ocr_csv = None
    for ocrf in ocr_files:
        if os.path.exists(ocrf):
            count = count_audio_samples(ocrf)  # OCR uses same CSV format as audio
            if count > ocr_samples:
                ocr_samples = count
                ocr_csv = ocrf
    
    # Count tokens from OCR text
    if ocr_csv:
        print(f"  Counting tokens from OCR text...")
        ocr_tokens = count_ocr_tokens(ocr_csv)
    
    if ocr_tokens > 0:
        config_path = "configs/ocr_tiny.json"
        ctx_len = 128  # Default for OCR (short text sequences)
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                cfg = json.load(f)
                ctx_len = cfg.get("ctx_len", 128)
        
        ocr_params = calculate_params_from_tokens(ocr_tokens, batch_size=4, ctx_len=ctx_len, gradient_accumulation=2)
        ocr_params["num_samples"] = ocr_samples  # Keep for reference
        
        print(f"  Tokens: {ocr_params['num_tokens']:,} (~{ocr_params['num_tokens']/1_000_000:.1f}M tokens)")
        print(f"  Steps/epoch: {ocr_params['steps_per_epoch']:,}")
        print(f"  Recommended epochs: {ocr_params['recommended_epochs']}")
        print(f"  Max steps: {ocr_params['max_steps']:,}")
        print(f"  Warmup steps: {ocr_params['warmup_steps']:,}")
        
        if not dry_run:
            update_config_file(
                config_path,
                ocr_params,
                preserve_keys=["train_csv", "image_root"],  # Preserve data paths
                update_paths=True
            )
    else:
        print("  ⚠ No OCR data found, skipping...")
    
    print("\n" + "="*60)
    if dry_run:
        print("DRY RUN: No files were modified")
        print("Run without --dry-run to apply changes")
    else:
        print("✓ All configs updated successfully!")
    print("="*60)
    
    # Print summary
    print("\nSummary:")
    print(f"  Text tokens: {text_tokens:,} (~{text_tokens/1_000_000:.1f}M)")
    print(f"  Image caption tokens: {image_tokens:,} (~{image_tokens/1_000_000:.1f}M)")
    print(f"  Audio transcription tokens: {audio_tokens:,} (~{audio_tokens/1_000_000:.1f}M)")
    if ocr_tokens > 0:
        print(f"  OCR tokens: {ocr_tokens:,} (~{ocr_tokens/1_000_000:.1f}M)")
    multimodal_tokens = max(text_tokens, image_tokens, audio_tokens)
    if multimodal_tokens > 0:
        print(f"  Multimodal (max tokens): {multimodal_tokens:,} (~{multimodal_tokens/1_000_000:.1f}M)")

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

