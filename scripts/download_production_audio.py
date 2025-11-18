"""
Download and prepare production-grade audio datasets for μOmni training
Target: Under 30GB, millions of samples
Includes: General speech datasets

Supports:
- General Speech: LibriSpeech, LJSpeech
"""

import os
import json
import argparse
import subprocess
import requests
from pathlib import Path
from tqdm import tqdm
import tarfile
import csv
import shutil

# State file to track progress
STATE_FILE = "data/.audio_download_state.json"

def load_state():
    """Load download/conversion state"""
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, 'r') as f:
            state = json.load(f)
            # Remove old entries if they exist (migration from old version)
            removed = False
            for old_key in ["musan", "urbansound"]:
                if old_key in state:
                    del state[old_key]
                    removed = True
            if removed:
                # Save cleaned state
                save_state(state)
            return state
    return {
        # General Speech
        "librispeech": {"downloaded": False, "extracted": False, "converted": False, "samples": 0},
        "ljspeech": {"downloaded": False, "extracted": False, "converted": False, "samples": 0}
    }

def print_progress_with_remaining(current, max_count, label="samples", report_interval=100):
    """Print progress with remaining count and percentage"""
    if current % report_interval == 0 or current >= max_count:
        remaining = max_count - current
        percent = (current / max_count * 100) if max_count > 0 else 0
        print(f"Progress: {current:,} {label} ({percent:.1f}%) - Remaining: ~{remaining:,} {label}")

def save_state(state):
    """Save download/conversion state"""
    os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)
    with open(STATE_FILE, 'w') as f:
        json.dump(state, f, indent=2)

def save_checkpoint(dataset_name, checkpoint_data):
    """Save fine-grained checkpoint for resuming"""
    checkpoint_file = f"data/.checkpoint_audio_{dataset_name}.json"
    os.makedirs(os.path.dirname(checkpoint_file), exist_ok=True)
    with open(checkpoint_file, 'w') as f:
        json.dump(checkpoint_data, f, indent=2)

def load_checkpoint(dataset_name):
    """Load fine-grained checkpoint for resuming"""
    checkpoint_file = f"data/.checkpoint_audio_{dataset_name}.json"
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'r') as f:
            return json.load(f)
    return None

def get_directory_size(path):
    """Calculate total size of a directory recursively in bytes"""
    if not os.path.exists(path):
        return 0
    total_size = 0
    try:
        for dirpath, dirnames, filenames in os.walk(path):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                if os.path.exists(filepath):
                    total_size += os.path.getsize(filepath)
    except (OSError, PermissionError):
        pass
    return total_size

def get_audio_dataset_size(ds_name):
    """Get actual disk size of audio dataset folder"""
    # Map dataset names to their actual data folders
    folder_map = {
        "librispeech": "data/audio/librispeech",
        "ljspeech": "data/audio/ljspeech",
    }
    
    folder_path = folder_map.get(ds_name)
    if folder_path and os.path.exists(folder_path):
        return get_directory_size(folder_path)
    return 0

def download_file(url, output_path, resume=True):
    """Download file with resume support"""
    if os.path.exists(output_path):
        if resume:
            print(f"File exists, resuming: {output_path}")
            resume_header = {'Range': f'bytes={os.path.getsize(output_path)}-'}
        else:
            print(f"File exists, skipping: {output_path}")
            return True
    else:
        resume_header = {}
    
    try:
        response = requests.get(url, headers=resume_header, stream=True, timeout=30)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        if resume and os.path.exists(output_path):
            mode = 'ab'
            initial_pos = os.path.getsize(output_path)
        else:
            mode = 'wb'
            initial_pos = 0
        
        with open(output_path, mode) as f:
            with tqdm(total=total_size, initial=initial_pos, unit='B', unit_scale=True, desc=os.path.basename(output_path)) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
        
        return True
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        return False

def download_librispeech_subset(state):
    """Download LibriSpeech subset (multiple splits to stay under 30GB)"""
    print("\n" + "="*60)
    print("Downloading LibriSpeech Subset")
    print("="*60)
    
    if state["librispeech"]["downloaded"]:
        print("LibriSpeech already downloaded, skipping...")
        return True
    
    base_url = "https://www.openslr.org/resources/12"
    download_dir = "data/audio_downloads/librispeech"
    os.makedirs(download_dir, exist_ok=True)
    
    # LibriSpeech splits:
    # - train-clean-100: 6.3GB, ~100 hours (recommended for ~50GB total budget)
    # - train-clean-360: 23GB, ~360 hours (optional, large)
    # - dev-clean: 337MB
    # - test-clean: 315MB
    
    # We'll download train-clean-100 only (~6.3GB) to keep audio dataset manageable
    # This provides ~100 hours of high-quality speech data
    downloads = [
        {
            "url": f"{base_url}/train-clean-100.tar.gz",
            "file": os.path.join(download_dir, "train-clean-100.tar.gz"),
            "size_gb": 6.3,
            "desc": "LibriSpeech train-clean-100 (6.3GB)"
        },
        {
            "url": f"{base_url}/dev-clean.tar.gz",
            "file": os.path.join(download_dir, "dev-clean.tar.gz"),
            "size_gb": 0.337,
            "desc": "LibriSpeech dev-clean (337MB)"
        },
        {
            "url": f"{base_url}/test-clean.tar.gz",
            "file": os.path.join(download_dir, "test-clean.tar.gz"),
            "size_gb": 0.315,
            "desc": "LibriSpeech test-clean (315MB)"
        }
    ]
    
    total_size = sum(d["size_gb"] for d in downloads)
    print(f"Total download size: ~{total_size:.2f} GB")
    print("This will provide ~100 hours of speech data")
    print("Note: train-clean-360 (23GB) is available but not downloaded by default to save space")
    
    for dl in downloads:
        if os.path.exists(dl["file"]):
            print(f"✓ {dl['desc']} already downloaded, skipping...")
            continue
        
        print(f"\nDownloading {dl['desc']}...")
        print(f"URL: {dl['url']}")
        print(f"Output: {dl['file']}")
        print("This may take 30-60 minutes depending on your connection...")
        
        if download_file(dl["url"], dl["file"], resume=True):
            print(f"✓ {dl['desc']} downloaded successfully")
        else:
            print(f"✗ Failed to download {dl['desc']}")
            return False
    
    state["librispeech"]["downloaded"] = True
    save_state(state)
    print("\n✓ All LibriSpeech files downloaded")
    return True

def extract_librispeech_subset(state):
    """Extract LibriSpeech archives with fine-grained resuming"""
    print("\n" + "="*60)
    print("Extracting LibriSpeech Archives")
    print("="*60)
    
    if state["librispeech"]["extracted"]:
        print("LibriSpeech already extracted, skipping...")
        return True
    
    download_dir = "data/audio_downloads/librispeech"
    extract_dir = "data/audio/librispeech"
    os.makedirs(extract_dir, exist_ok=True)
    
    # Load checkpoint
    checkpoint = load_checkpoint("librispeech_extract")
    if checkpoint:
        print(f"Resuming extraction: {checkpoint.get('extracted_files', [])} already extracted")
        extracted_files = set(checkpoint.get('extracted_files', []))
    else:
        extracted_files = set()
    
    tar_files = [
        "train-clean-100.tar.gz",
        "dev-clean.tar.gz",
        "test-clean.tar.gz"
    ]
    
    for tar_file in tar_files:
        # Check if already extracted
        if tar_file in extracted_files:
            print(f"✓ {tar_file} already extracted, skipping...")
            continue
        
        tar_path = os.path.join(download_dir, tar_file)
        if not os.path.exists(tar_path):
            print(f"WARNING: {tar_file} not found, skipping...")
            continue
        
        print(f"\nExtracting {tar_file} (this may take 10-15 minutes)...")
        try:
            with tarfile.open(tar_path, 'r:gz') as tar:
                tar.extractall(extract_dir)
            print(f"✓ {tar_file} extracted")
            extracted_files.add(tar_file)
            
            # Save checkpoint after each file
            save_checkpoint("librispeech_extract", {
                'extracted_files': list(extracted_files)
            })
            save_state(state)
        except Exception as e:
            print(f"ERROR extracting {tar_file}: {e}")
            # Save checkpoint on error
            save_checkpoint("librispeech_extract", {
                'extracted_files': list(extracted_files)
            })
            return False
    
    state["librispeech"]["extracted"] = True
    save_state(state)
    
    # Clean up checkpoint on success
    checkpoint_file = "data/.checkpoint_audio_librispeech_extract.json"
    if os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)
    
    print("\n✓ All LibriSpeech archives extracted")
    return True

def convert_librispeech_to_csv(state, max_samples=1000000):
    """Convert LibriSpeech to ASR CSV format with fine-grained resuming"""
    print("\n" + "="*60)
    print("Converting LibriSpeech to ASR CSV")
    print("="*60)
    
    if state["librispeech"]["converted"] and state["librispeech"]["samples"] >= max_samples:
        print(f"LibriSpeech already converted ({state['librispeech']['samples']:,} samples), skipping...")
        return True
    
    base_dir = "data/audio/librispeech/LibriSpeech"
    
    if not os.path.exists(base_dir):
        print(f"ERROR: LibriSpeech not extracted. Run extract first.")
        return False
    
    output_file = "data/audio/librispeech_asr.csv"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Load checkpoint
    checkpoint = load_checkpoint("librispeech")
    if checkpoint:
        print(f"Resuming from checkpoint: {checkpoint.get('last_split', 'start')}")
        processed_splits = set(checkpoint.get('processed_splits', []))
        rows = checkpoint.get('rows', [])
        mode = 'a'  # Append mode
    else:
        processed_splits = set()
        rows = []
        mode = 'w'  # Write mode
    
    print("Scanning LibriSpeech directory structure...")
    
    # Process all splits
    splits = ["train-clean-100", "dev-clean", "test-clean"]
    
    with open(output_file, mode, encoding='utf-8', newline='') as csv_f:
        writer = csv.DictWriter(csv_f, fieldnames=['wav', 'text'])
        if mode == 'w':
            writer.writeheader()
        
        for split_name in splits:
            if split_name in processed_splits:
                print(f"Skipping already processed {split_name}...")
                continue
            
            split_dir = os.path.join(base_dir, split_name)
            if not os.path.exists(split_dir):
                print(f"WARNING: {split_name} not found, skipping...")
                continue
            
            print(f"\nProcessing {split_name}...")
            
            # LibriSpeech structure: split/speaker/chapter/...
            for speaker_dir in sorted(Path(split_dir).iterdir()):
                if not speaker_dir.is_dir():
                    continue
                
                for chapter_dir in sorted(speaker_dir.iterdir()):
                    if not chapter_dir.is_dir():
                        continue
                    
                    # Find .txt transcription file
                    txt_files = list(chapter_dir.glob("*.txt"))
                    if not txt_files:
                        continue
                    
                    txt_file = txt_files[0]
                    
                    # Load checkpoint for this chapter
                    checkpoint_chapter = checkpoint.get('last_chapter', {}) if checkpoint else {}
                    last_line_chapter = checkpoint_chapter.get(f"{split_name}/{speaker_dir.name}/{chapter_dir.name}", 0)
                    catching_up_chapter = (last_line_chapter > 0)
                    
                    if catching_up_chapter:
                        print(f"  Fast-forwarding {speaker_dir.name}/{chapter_dir.name}: {last_line_chapter:,} lines...")
                    
                    with open(txt_file, 'r', encoding='utf-8') as f:
                        line_num = 0
                        for line in tqdm(f, desc=f"  {speaker_dir.name}/{chapter_dir.name}", leave=False):
                            line_num += 1
                            
                            # Fast-forward: skip already processed lines
                            if catching_up_chapter and line_num <= last_line_chapter:
                                if line_num % 10000 == 0:
                                    print(f"    Fast-forward: {line_num:,} / {last_line_chapter:,} lines...")
                                continue
                            
                            # We've caught up - start processing normally
                            if catching_up_chapter and line_num > last_line_chapter:
                                catching_up_chapter = False
                                print(f"    Caught up! Starting from line {line_num:,}...")
                            
                            parts = line.strip().split(' ', 1)
                            if len(parts) == 2:
                                audio_id, text = parts
                                # LibriSpeech uses .flac files
                                flac_path = chapter_dir / f"{audio_id}.flac"
                                if flac_path.exists():
                                    # Use path relative to project root (include data/audio prefix)
                                    rel_path = os.path.relpath(str(flac_path), ".")
                                    writer.writerow({"wav": rel_path, "text": text})
                                    rows.append({"wav": rel_path, "text": text})
                                    
                                    # Save checkpoint every 1000 entries
                                    if len(rows) % 1000 == 0:
                                        chapter_key = f"{split_name}/{speaker_dir.name}/{chapter_dir.name}"
                                        checkpoint_chapter[chapter_key] = line_num
                                        save_checkpoint("librispeech", {
                                            'processed_splits': list(processed_splits),
                                            'rows': rows[-1000:],  # Keep last 1000 for reference
                                            'last_split': split_name,
                                            'last_chapter': checkpoint_chapter,
                                            'count': len(rows)
                                        })
                                        save_state(state)
                                    
                                    # Stop if we've reached sample limit
                                    if len(rows) >= max_samples:
                                        print(f"\nReached sample limit ({max_samples:,}), stopping...")
                                        break
                                    
                                    # Print progress with remaining
                                    if len(rows) % 1000 == 0:
                                        print_progress_with_remaining(len(rows), max_samples, "samples", report_interval=1000)
                        
                        if len(rows) >= max_samples:
                            break
                    
                    if len(rows) >= max_samples:
                        break
                
                if len(rows) >= max_samples:
                    break
            
            processed_splits.add(split_name)
            save_checkpoint("librispeech", {
                'processed_splits': list(processed_splits),
                'rows': [],
                'last_split': split_name,
                'count': len(rows)
            })
            save_state(state)
    
    state["librispeech"]["converted"] = True
    state["librispeech"]["samples"] = len(rows)
    save_state(state)
    
    # Clean up checkpoint file (state file only, actual data is never deleted)
    checkpoint_file = "data/.checkpoint_audio_librispeech.json"
    if os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)
    
    print(f"\n✓ Created ASR CSV with {len(rows):,} entries")
    print(f"  Saved to: {output_file} (ready to use)")
    return True

def download_ljspeech(state, max_samples=1000000):
    """Download LJSpeech dataset - single speaker TTS dataset"""
    print("\n" + "="*60)
    print("Downloading LJSpeech Dataset")
    print("="*60)
    
    if state["ljspeech"]["downloaded"] and state["ljspeech"]["samples"] >= max_samples:
        print(f"LJSpeech already downloaded ({state['ljspeech']['samples']:,} samples), skipping...")
        return True
    
    # LJSpeech has direct download
    url = "https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2"
    download_dir = "data/audio_downloads"
    os.makedirs(download_dir, exist_ok=True)
    tar_file = os.path.join(download_dir, "LJSpeech-1.1.tar.bz2")
    
    print("Downloading LJSpeech (~2.6GB, 13k+ clips)...")
    if download_file(url, tar_file, resume=True):
        extract_dir = "data/audio/ljspeech"
        os.makedirs(extract_dir, exist_ok=True)
        
        print("Extracting LJSpeech...")
        with tarfile.open(tar_file, 'r:bz2') as tar:
            tar.extractall(extract_dir)
        
        # Convert to CSV
        output_file = "data/audio/ljspeech_asr.csv"
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # LJSpeech structure: LJSpeech-1.1/wavs/*.wav and metadata.csv
        metadata_file = os.path.join(extract_dir, "LJSpeech-1.1", "metadata.csv")
        wavs_dir = os.path.join(extract_dir, "LJSpeech-1.1", "wavs")
        
        rows = []
        count = 0
        
        # Load checkpoint for LJSpeech
        checkpoint = load_checkpoint("ljspeech")
        last_line = checkpoint.get('last_line', 0) if checkpoint else 0
        count = checkpoint.get('count', 0) if checkpoint else 0
        catching_up = (last_line > 0)
        
        if catching_up:
            print(f"Fast-forwarding through {last_line:,} already-processed lines...")
        
        if os.path.exists(metadata_file):
            with open(metadata_file, 'r', encoding='utf-8') as f:
                line_num = 0
                for line in tqdm(f, desc="Processing LJSpeech"):
                    line_num += 1
                    
                    # Fast-forward: skip already processed lines
                    if catching_up and line_num <= last_line:
                        if line_num % 10000 == 0:
                            print(f"Fast-forward: {line_num:,} / {last_line:,} lines...")
                        continue
                    
                    # We've caught up - start processing normally
                    if catching_up and line_num > last_line:
                        catching_up = False
                        print(f"Caught up! Starting processing from line {line_num:,}...")
                    
                    parts = line.strip().split('|')
                    if len(parts) >= 2:
                        wav_name = parts[0]
                        text = parts[2] if len(parts) > 2 else parts[1]
                        
                        wav_path = os.path.join(wavs_dir, f"{wav_name}.wav")
                        if os.path.exists(wav_path):
                            rel_path = os.path.relpath(wav_path, ".")
                            rows.append({"wav": rel_path, "text": text})
                            count += 1
                            
                            # Print progress with remaining
                            print_progress_with_remaining(count, max_samples, "samples", report_interval=100)
                            
                            # Save checkpoint
                            if count % 100 == 0:
                                save_checkpoint("ljspeech", {
                                    'last_line': line_num,
                                    'count': count
                                })
                            
                            if count >= max_samples:
                                break
            
            with open(output_file, 'w', encoding='utf-8', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=['wav', 'text'])
                writer.writeheader()
                writer.writerows(rows)
            
            # Only mark as downloaded if we reached max_samples
            if count >= max_samples:
                state["ljspeech"]["downloaded"] = True
                state["ljspeech"]["extracted"] = True
                state["ljspeech"]["converted"] = True
            state["ljspeech"]["samples"] = count
            save_state(state)
            
            if count >= max_samples:
                print(f"✓ Created CSV with {count:,} entries")
            else:
                print(f"⚠ Created CSV with {count:,} entries (target: {max_samples:,})")
            return True
    
    return False

def combine_audio_csvs():
    """Combine all audio CSVs into one file (ASR format: wav,text)"""
    print("\n" + "="*60)
    print("Combining Audio CSVs (ASR format)")
    print("="*60)
    
    output_file = "data/audio/production_asr.csv"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    input_files = [
        "data/audio/librispeech_asr.csv",
        "data/audio/ljspeech_asr.csv"
    ]
    
    all_rows = []
    
    for input_file in input_files:
        if os.path.exists(input_file):
            print(f"Loading {os.path.basename(input_file)}...")
            with open(input_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                all_rows.extend(rows)
                print(f"  Added {len(rows):,} entries")
    
    with open(output_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['wav', 'text'])
        writer.writeheader()
        writer.writerows(all_rows)
    
    print(f"\n✓ Combined {len(all_rows):,} audio entries")
    print(f"  Saved to: {output_file}")
    
    # Also create TTS format (text,wav) for train_talker.py
    print("\n" + "="*60)
    print("Creating TTS format CSV (text,wav)")
    print("="*60)
    
    tts_output_file = "data/audio/production_tts.csv"
    with open(tts_output_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['text', 'wav'])
        writer.writeheader()
        for row in all_rows:
            writer.writerow({'text': row['text'], 'wav': row['wav']})
    
    print(f"✓ Created TTS CSV: {tts_output_file}")
    print(f"  Format: text,wav (for train_talker.py)")


def main():
    parser = argparse.ArgumentParser(description="Download production-grade audio datasets for μOmni")
    parser.add_argument("--dataset", 
                       choices=["all", "librispeech", "ljspeech", "general"], 
                       default="all",
                       help="Which dataset to download (default: all)")
    parser.add_argument("--skip-download", action="store_true",
                       help="Skip download, only extract/convert existing data")
    parser.add_argument("--skip-extract", action="store_true",
                       help="Skip extraction, only convert")
    parser.add_argument("--skip-convert", action="store_true",
                       help="Skip conversion, only download/extract")
    parser.add_argument("--combine", action="store_true",
                       help="Combine all downloaded datasets into one CSV (outputs to data/audio/production_asr.csv)")
    parser.add_argument("--reset", action="store_true",
                       help="Reset state and re-download everything")
    parser.add_argument("--max-samples", type=int, default=1000000,
                       help="Maximum number of samples per dataset (default: 1000000)")
    
    args = parser.parse_args()
    
    # Load or reset state
    if args.reset:
        print("Resetting state...")
        state = load_state()
        for key in state:
            for subkey in state[key]:
                if subkey != "samples":
                    state[key][subkey] = False
        save_state(state)
    else:
        state = load_state()
    
    print("="*60)
    print("μOmni Production Audio Dataset Downloader")
    print("="*60)
    print(f"State file: {STATE_FILE}")
    print(f"Dataset: {args.dataset}")
    print("="*60)
    
    success = True
    
    # LibriSpeech
    if args.dataset in ["all", "librispeech", "general"]:
        if not args.skip_download:
            success = download_librispeech_subset(state) and success
        if not args.skip_extract:
            success = extract_librispeech_subset(state) and success
        if not args.skip_convert:
            success = convert_librispeech_to_csv(state, args.max_samples) and success
    
    # LJSpeech
    if args.dataset in ["all", "ljspeech", "general"]:
        if not args.skip_download:
            success = download_ljspeech(state, args.max_samples) and success
    
    # Combine if requested
    if args.combine:
        combine_audio_csvs()
    
    print("\n" + "="*60)
    if success:
        print("✓ All operations completed successfully!")
        print("\nOutput files (ready to use, no formatting needed):")
        print("  - Individual datasets: data/audio/*_asr.csv")
        if args.combine or args.dataset == "all":
            print("  - Combined ASR CSV: data/audio/production_asr.csv")
            print("  - Combined TTS CSV: data/audio/production_tts.csv")
        print("\nNext steps:")
        print("1. CSVs are already in final format in data/audio/")
        print("2. Update config files to point to:")
        if args.combine or args.dataset == "all":
            print("   ASR: data/audio/production_asr.csv (for train_audio_enc.py)")
            print("   TTS: data/audio/production_tts.csv (for train_talker.py)")
        else:
            print("   data/audio/[dataset_name]_asr.csv")
        print("3. Run training:")
        print("   python train_audio_enc.py --config configs/audio_enc_tiny.json")
        print("   python train_talker.py --config configs/talker_tiny.json")
    else:
        print("✗ Some operations failed. Check errors above.")
        print("You can resume by running the script again (it will skip completed steps)")
        print("Fine-grained checkpoints saved - will resume from exact position")
    print("="*60)

if __name__ == "__main__":
    main()

