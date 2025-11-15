"""
Download and format datasets for μOmni training (Quick Start Option A)
Supports resuming downloads and conversion progress
"""

import os
import json
import csv
import argparse
import subprocess
import requests
from pathlib import Path
from tqdm import tqdm
import zipfile
import tarfile
import shutil

# State file to track progress
STATE_FILE = "data/.download_state.json"

def load_state():
    """Load download/conversion state"""
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, 'r') as f:
            return json.load(f)
    return {
        "dialogstudio": {"downloaded": False, "converted": False, "sub_datasets_loaded": 0, "total_sub_datasets": 86, "conversations_count": 0},
        "coco": {"images_downloaded": False, "annotations_downloaded": False, "converted": False},
        "librispeech": {"downloaded": False, "converted": False}
    }

def save_state(state):
    """Save download/conversion state"""
    os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)
    with open(STATE_FILE, 'w') as f:
        json.dump(state, f, indent=2)

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

def download_dialogstudio(state):
    """Download DialogStudio from HuggingFace (mark as downloaded, actual conversion happens in convert)"""
    print("\n" + "="*60)
    print("Preparing DialogStudio Download")
    print("="*60)
    
    if state["dialogstudio"]["downloaded"]:
        print("DialogStudio download marked as complete, skipping...")
        return True
    
    try:
        from datasets import load_dataset
        print("Note: DialogStudio will be downloaded during conversion step")
        print("This is because DialogStudio has a complex structure that's better handled directly")
        
        # Just mark as downloaded - actual download happens in convert step
        state["dialogstudio"]["downloaded"] = True
        save_state(state)
        return True
    except ImportError:
        print("ERROR: 'datasets' package not installed. Install with: pip install datasets")
        return False
    except Exception as e:
        print(f"ERROR: {e}")
        return False

def convert_dialogstudio(state, load_all_sub_datasets=True):
    """Download and convert DialogStudio to text format"""
    print("\n" + "="*60)
    print("Downloading and Converting DialogStudio to Text Format")
    print("="*60)
    
    if state["dialogstudio"]["converted"]:
        print("DialogStudio already converted, skipping...")
        return True
    
    try:
        from datasets import load_dataset
        
        output_file = "data/text/dialogstudio.txt"
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        print("Loading DialogStudio from HuggingFace (this downloads it)...")
        print("This may take 10-30 minutes depending on your connection...")
        print("\nNOTE: DialogStudio is a gated dataset.")
        print("You need to:")
        print("1. Accept the license at: https://huggingface.co/datasets/Salesforce/dialogstudio")
        print("2. Login to HuggingFace: huggingface-cli login")
        print("   Or set HF_TOKEN environment variable")
        
        # Check for authentication
        try:
            from huggingface_hub import whoami
            user = whoami()
            if user:
                print(f"✓ Authenticated as: {user.get('name', 'unknown')}")
        except Exception:
            print("⚠ WARNING: Not authenticated with HuggingFace")
            print("  Run: huggingface-cli login")
            print("  Or set HF_TOKEN environment variable")
        
        # Load dataset directly (downloads if not cached)
        # DialogStudio contains 86 sub-datasets
        # Note: DialogStudio uses a loading script which requires datasets <3.0.0
        # and trust_remote_code=True to execute the custom loading script
        try:
            if load_all_sub_datasets:
                from datasets import get_dataset_config_names
                
                print("Getting list of DialogStudio sub-datasets...")
                config_names = get_dataset_config_names('Salesforce/dialogstudio')
                print(f"Found {len(config_names)} sub-datasets")
                print("⚠ WARNING: Loading all sub-datasets will take 30-60 minutes and download ~10GB")
                print("This will take a while - loading all sub-datasets...")
                
                # Load all sub-datasets and combine them
                all_datasets = []
                for i, config_name in enumerate(tqdm(config_names, desc="Loading sub-datasets")):
                    try:
                        ds_config = load_dataset('Salesforce/dialogstudio', config_name, trust_remote_code=True)
                        all_datasets.append((config_name, ds_config))
                    except Exception as e:
                        print(f"  ⚠ Skipping {config_name}: {e}")
                        continue
                
                if not all_datasets:
                    raise RuntimeError("Failed to load any DialogStudio sub-datasets")
                
                print(f"\n✓ Loaded {len(all_datasets)} out of {len(config_names)} sub-datasets")
                if len(all_datasets) < len(config_names):
                    print(f"⚠ WARNING: {len(config_names) - len(all_datasets)} sub-datasets failed to load or were skipped")
                
                # Store metadata in state
                state["dialogstudio"]["sub_datasets_loaded"] = len(all_datasets)
                state["dialogstudio"]["total_sub_datasets"] = len(config_names)
                save_state(state)
                
                # Store as a list of (name, dataset) tuples for processing
                ds = all_datasets
            else:
                # Load combined dataset (faster, but may be a sample)
                # The "combined" dataset is HuggingFace's default view, which typically contains
                # only a subset or sample of the full 86 sub-datasets, not all of them.
                # This is why it's smaller - it doesn't include all the individual dialog datasets.
                print("Loading DialogStudio (combined dataset)...")
                print("Note: The combined dataset is smaller because it only includes a subset")
                print("      of the 86 sub-datasets, not all of them. For full training data,")
                print("      use the default (loads all 86 sub-datasets, ~10GB, 30-60 min).")
                ds = load_dataset('Salesforce/dialogstudio', trust_remote_code=True)
                
                # Mark as combined dataset in state
                state["dialogstudio"]["sub_datasets_loaded"] = 0  # 0 means combined mode
                state["dialogstudio"]["total_sub_datasets"] = 1
                save_state(state)
        except Exception as load_error:
            error_str = str(load_error).lower()
            
            # Check for trust_remote_code requirement
            if "trust_remote_code" in error_str:
                print("\n" + "="*60)
                print("TRUST_REMOTE_CODE REQUIRED")
                print("="*60)
                print("DialogStudio requires trust_remote_code=True to execute its custom loading script.")
                print("This should already be set in the script, but if you see this error,")
                print("please report it as a bug.")
                print("="*60)
                raise RuntimeError(
                    "DialogStudio requires trust_remote_code=True. "
                    "This should be handled automatically by the script."
                ) from load_error
            
            # Check for loading script compatibility issue
            elif "dataset scripts are no longer supported" in error_str or "loading script" in error_str:
                print("\n" + "="*60)
                print("COMPATIBILITY ISSUE DETECTED")
                print("="*60)
                print("DialogStudio requires a datasets library version that supports loading scripts.")
                print("Your current version doesn't support this.")
                print("\nSOLUTION: Downgrade datasets library")
                print("\nRun this command:")
                print("  pip install 'datasets<3.0.0'")
                print("\nThen run this script again.")
                print("\nNote: This is a known issue with DialogStudio and newer datasets versions.")
                print("="*60)
                raise RuntimeError(
                    "DialogStudio requires datasets<3.0.0. "
                    "Please run: pip install 'datasets<3.0.0'"
                ) from load_error
            
            # Check for authentication error
            elif "gated" in error_str or "authenticated" in error_str:
                print("\n" + "="*60)
                print("AUTHENTICATION REQUIRED")
                print("="*60)
                print("DialogStudio requires HuggingFace authentication.")
                print("\nSteps to fix:")
                print("1. Visit: https://huggingface.co/datasets/Salesforce/dialogstudio")
                print("2. Accept the dataset license")
                print("3. Run: huggingface-cli login")
                print("   Or set environment variable: HF_TOKEN=your_token_here")
                print("="*60)
                raise
            else:
                raise
        
        print("\nConverting conversations to text format...")
        
        count = 0
        total_sub_datasets = len(ds) if isinstance(ds, list) else 1
        
        with open(output_file, 'w', encoding='utf-8') as f:
            # Process all sub-datasets
            if isinstance(ds, list):
                # ds is a list of (config_name, dataset) tuples
                for config_name, dataset in ds:
                    print(f"\nProcessing sub-dataset: {config_name}")
                    
                    # Each dataset is a DatasetDict with train/validation/test splits
                    if hasattr(dataset, 'keys') and callable(getattr(dataset, 'keys', None)):
                        splits = list(dataset.keys())
                        for split_name in splits:
                            split_data = dataset[split_name]
                            split_count = len(split_data)
                            print(f"  Processing {split_name} split ({split_count} items)...")
                            for item in tqdm(split_data, desc=f"    Converting", leave=False, total=split_count):
                                text = extract_text_from_item(item)
                                if text:
                                    f.write(text + '\n')
                                    count += 1
                    else:
                        # Fallback: try direct iteration
                        try:
                            total = len(dataset)
                            print(f"  Processing all items ({total} items)...")
                            for item in tqdm(dataset, desc=f"    Converting", total=total):
                                text = extract_text_from_item(item)
                                if text:
                                    f.write(text + '\n')
                                    count += 1
                        except Exception as e:
                            print(f"  ⚠ Error processing {config_name}: {e}")
                            continue
            else:
                # Fallback: original logic for single dataset
                print(f"Dataset type: {type(ds)}")
                print(f"Dataset keys: {list(ds.keys()) if hasattr(ds, 'keys') else 'N/A'}")
                
                for dataset_name, dataset in ds.items():
                    print(f"\nProcessing dataset: {dataset_name}")
                    
                    if hasattr(dataset, 'keys') and callable(getattr(dataset, 'keys', None)):
                        splits = list(dataset.keys())
                        for split_name in splits:
                            split_data = dataset[split_name]
                            print(f"  Processing split: {split_name} ({len(split_data)} items)...")
                            for item in tqdm(split_data, desc=f"    Converting", leave=False):
                                text = extract_text_from_item(item)
                                if text:
                                    f.write(text + '\n')
                                    count += 1
                    else:
                        # Try direct iteration
                        try:
                            total = len(dataset)
                            print(f"  Processing all items ({total} items)...")
                            for item in tqdm(dataset, desc=f"    Converting", total=total):
                                text = extract_text_from_item(item)
                                if text:
                                    f.write(text + '\n')
                                    count += 1
                        except Exception as e:
                            print(f"  ERROR: {e}")
                            continue
        
        state["dialogstudio"]["converted"] = True
        state["dialogstudio"]["conversations_count"] = count
        save_state(state)
        
        # Print summary
        print(f"\n✓ Converted {count:,} conversations to {output_file}")
        if isinstance(ds, list):
            print(f"   Processed {len(ds)} sub-datasets")
            if state["dialogstudio"].get("total_sub_datasets", 0) > len(ds):
                print(f"   ⚠ WARNING: Only {len(ds)}/{state['dialogstudio'].get('total_sub_datasets', 86)} sub-datasets were processed")
        else:
            print(f"   Used combined dataset (not all sub-datasets)")
        
        return True
    except ImportError:
        print("ERROR: 'datasets' package not installed. Install with: pip install datasets")
        return False
    except Exception as e:
        print(f"ERROR converting DialogStudio: {e}")
        import traceback
        traceback.print_exc()
        return False

def extract_text_from_item(item):
    """Extract text from a DialogStudio item (handles various formats)"""
    if isinstance(item, dict):
        # DialogStudio format: 'log' field contains conversation turns
        if 'log' in item and isinstance(item['log'], list):
            conv_parts = []
            for turn in item['log']:
                if isinstance(turn, dict):
                    # Extract user utterance
                    user_utt = turn.get('user utterance') or turn.get('user_utterance') or turn.get('user')
                    if user_utt:
                        conv_parts.append(str(user_utt).strip())
                    
                    # Extract system response
                    sys_resp = turn.get('system response') or turn.get('system_response') or turn.get('system') or turn.get('assistant')
                    if sys_resp:
                        conv_parts.append(str(sys_resp).strip())
            
            if conv_parts:
                return ' '.join(conv_parts)
        
        # Try conversations format (other datasets)
        if 'conversations' in item:
            conv_text = ' '.join([
                str(turn.get('content', '')) or str(turn.get('value', '')) or str(turn.get('text', ''))
                for turn in item['conversations']
                if isinstance(turn, dict) and (turn.get('content') or turn.get('value') or turn.get('text'))
            ])
            if conv_text.strip():
                return conv_text.strip()
        
        # Try instruction/input/output format
        parts = []
        for key in ['instruction', 'input', 'output', 'query', 'response', 'context', 'answer', 'prompt']:
            if key in item and item[key]:
                val = item[key]
                if isinstance(val, str):
                    parts.append(val.strip())
                elif isinstance(val, list):
                    # If it's a list, try to extract text from it
                    for elem in val:
                        if isinstance(elem, str) and elem.strip():
                            parts.append(elem.strip())
        
        if parts:
            return ' '.join(parts)
        
        # Try to extract all string values (fallback)
        text_parts = []
        for key, value in item.items():
            if isinstance(value, str) and value.strip() and key not in ['id', 'dataset', 'split', 'original dialog id', 'new dialog id', 'dialog index']:
                text_parts.append(value.strip())
        if text_parts:
            return ' '.join(text_parts)
    
    return None

def download_coco(state):
    """Download COCO 2017 dataset"""
    print("\n" + "="*60)
    print("Downloading COCO 2017 (Image-Caption Data)")
    print("="*60)
    
    base_url = "http://images.cocodataset.org"
    coco_dir = "data/coco_downloads"
    os.makedirs(coco_dir, exist_ok=True)
    
    downloads = [
        {
            "url": f"{base_url}/zips/train2017.zip",
            "file": os.path.join(coco_dir, "train2017.zip"),
            "key": "images_downloaded",
            "desc": "COCO Train Images (18GB)"
        },
        {
            "url": f"{base_url}/zips/val2017.zip",
            "file": os.path.join(coco_dir, "val2017.zip"),
            "key": "val_images_downloaded",
            "desc": "COCO Val Images (1GB)"
        },
        {
            "url": f"{base_url}/annotations/annotations_trainval2017.zip",
            "file": os.path.join(coco_dir, "annotations_trainval2017.zip"),
            "key": "annotations_downloaded",
            "desc": "COCO Annotations (241MB)"
        }
    ]
    
    for dl in downloads:
        if state["coco"].get(dl["key"], False):
            print(f"✓ {dl['desc']} already downloaded, skipping...")
            continue
        
        print(f"\nDownloading {dl['desc']}...")
        print(f"URL: {dl['url']}")
        print(f"Output: {dl['file']}")
        
        if download_file(dl["url"], dl["file"], resume=True):
            state["coco"][dl["key"]] = True
            save_state(state)
            print(f"✓ {dl['desc']} downloaded successfully")
        else:
            print(f"✗ Failed to download {dl['desc']}")
            return False
    
    return True

def extract_coco(state):
    """Extract COCO archives"""
    print("\n" + "="*60)
    print("Extracting COCO Archives")
    print("="*60)
    
    coco_dir = "data/coco_downloads"
    images_dir = "data/images/coco_images"
    annotations_dir = "data/images/coco_annotations"
    
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(annotations_dir, exist_ok=True)
    
    # Extract train images
    train_zip = os.path.join(coco_dir, "train2017.zip")
    train_extracted = os.path.join(images_dir, "train2017")
    if os.path.exists(train_zip) and not os.path.exists(train_extracted):
        print("Extracting train2017.zip (this may take 10-15 minutes)...")
        with zipfile.ZipFile(train_zip, 'r') as zip_ref:
            zip_ref.extractall(images_dir)
        print("✓ Train images extracted")
    
    # Extract val images
    val_zip = os.path.join(coco_dir, "val2017.zip")
    val_extracted = os.path.join(images_dir, "val2017")
    if os.path.exists(val_zip) and not os.path.exists(val_extracted):
        print("Extracting val2017.zip...")
        with zipfile.ZipFile(val_zip, 'r') as zip_ref:
            zip_ref.extractall(images_dir)
        print("✓ Val images extracted")
    
    # Extract annotations
    ann_zip = os.path.join(coco_dir, "annotations_trainval2017.zip")
    ann_extracted = os.path.join(annotations_dir, "annotations")
    if os.path.exists(ann_zip) and not os.path.exists(ann_extracted):
        print("Extracting annotations...")
        with zipfile.ZipFile(ann_zip, 'r') as zip_ref:
            zip_ref.extractall(annotations_dir)
        print("✓ Annotations extracted")
    
    return True

def convert_coco(state):
    """Convert COCO to image manifest"""
    print("\n" + "="*60)
    print("Converting COCO to Image Manifest")
    print("="*60)
    
    if state["coco"]["converted"]:
        print("COCO already converted, skipping...")
        return True
    
    annotations_file = "data/images/coco_annotations/annotations/captions_train2017.json"
    val_annotations_file = "data/images/coco_annotations/annotations/captions_val2017.json"
    
    if not os.path.exists(annotations_file):
        print(f"ERROR: COCO annotations not found at {annotations_file}")
        print("Please download and extract COCO annotations first.")
        return False
    
    print("Loading COCO annotations...")
    manifest = []
    
    # Process train annotations
    if os.path.exists(annotations_file):
        with open(annotations_file, 'r') as f:
            coco_data = json.load(f)
        
        # Create image ID to filename mapping
        img_map = {img['id']: img['file_name'] for img in coco_data['images']}
        
        print("Processing train annotations...")
        for ann in tqdm(coco_data['annotations'], desc="Converting train"):
            img_id = ann['image_id']
            if img_id in img_map:
                img_path = os.path.join("coco_images/train2017", img_map[img_id])
                manifest.append({
                    "image": img_path,
                    "caption": ann['caption']
                })
    
    # Process val annotations
    if os.path.exists(val_annotations_file):
        with open(val_annotations_file, 'r') as f:
            coco_data = json.load(f)
        
        img_map = {img['id']: img['file_name'] for img in coco_data['images']}
        
        print("Processing val annotations...")
        for ann in tqdm(coco_data['annotations'], desc="Converting val"):
            img_id = ann['image_id']
            if img_id in img_map:
                img_path = os.path.join("coco_images/val2017", img_map[img_id])
                manifest.append({
                    "image": img_path,
                    "caption": ann['caption']
                })
    
    # Save manifest
    output_file = "data/images/annotations.json"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    
    state["coco"]["converted"] = True
    save_state(state)
    print(f"✓ Created manifest with {len(manifest)} image-caption pairs")
    print(f"  Saved to: {output_file}")
    return True

def download_librispeech(state):
    """Download LibriSpeech train-clean-100"""
    print("\n" + "="*60)
    print("Downloading LibriSpeech train-clean-100 (Audio Data)")
    print("="*60)
    
    if state["librispeech"]["downloaded"]:
        print("LibriSpeech already downloaded, skipping...")
        return True
    
    url = "https://www.openslr.org/resources/12/train-clean-100.tar.gz"
    output_file = "data/librispeech_downloads/train-clean-100.tar.gz"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    print(f"Downloading LibriSpeech train-clean-100 (6.3 GB)...")
    print(f"URL: {url}")
    print(f"Output: {output_file}")
    print("This may take 30-60 minutes depending on your connection...")
    
    if download_file(url, output_file, resume=True):
        state["librispeech"]["downloaded"] = True
        save_state(state)
        print("✓ LibriSpeech downloaded successfully")
        return True
    else:
        print("✗ Failed to download LibriSpeech")
        return False

def extract_librispeech(state):
    """Extract LibriSpeech archive"""
    print("\n" + "="*60)
    print("Extracting LibriSpeech Archive")
    print("="*60)
    
    tar_file = "data/librispeech_downloads/train-clean-100.tar.gz"
    extract_dir = "data/audio/librispeech"
    
    if not os.path.exists(tar_file):
        print(f"ERROR: LibriSpeech archive not found at {tar_file}")
        return False
    
    if os.path.exists(os.path.join(extract_dir, "LibriSpeech", "train-clean-100")):
        print("LibriSpeech already extracted, skipping...")
        return True
    
    os.makedirs(extract_dir, exist_ok=True)
    
    print("Extracting LibriSpeech (this may take 10-15 minutes)...")
    try:
        with tarfile.open(tar_file, 'r:gz') as tar:
            tar.extractall(extract_dir)
        print("✓ LibriSpeech extracted successfully")
        return True
    except Exception as e:
        print(f"ERROR extracting LibriSpeech: {e}")
        return False

def convert_librispeech(state):
    """Convert LibriSpeech to ASR CSV"""
    print("\n" + "="*60)
    print("Converting LibriSpeech to ASR CSV")
    print("="*60)
    
    if state["librispeech"]["converted"]:
        print("LibriSpeech already converted, skipping...")
        return True
    
    base_dir = "data/audio/librispeech/LibriSpeech/train-clean-100"
    
    if not os.path.exists(base_dir):
        print(f"ERROR: LibriSpeech not extracted. Run extract first.")
        return False
    
    print("Scanning LibriSpeech directory structure...")
    rows = []
    
    # LibriSpeech structure: train-clean-100/speaker/chapter/...
    for speaker_dir in sorted(Path(base_dir).iterdir()):
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
            print(f"Processing {txt_file.parent.name}/{txt_file.name}...")
            
            with open(txt_file, 'r', encoding='utf-8') as f:
                for line in tqdm(f, desc=f"  Converting", leave=False):
                    parts = line.strip().split(' ', 1)
                    if len(parts) == 2:
                        audio_id, text = parts
                        # LibriSpeech uses .flac files
                        wav_path = chapter_dir / f"{audio_id}.flac"
                        if wav_path.exists():
                            # Use relative path from data/audio
                            rel_path = os.path.relpath(str(wav_path), "data/audio")
                            rows.append({"wav": rel_path, "text": text})
    
    # Save CSV
    output_file = "data/audio/asr.csv"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    print(f"\nWriting {len(rows)} entries to CSV...")
    with open(output_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['wav', 'text'])
        writer.writeheader()
        writer.writerows(rows)
    
    state["librispeech"]["converted"] = True
    save_state(state)
    print(f"✓ Created ASR CSV with {len(rows)} entries")
    print(f"  Saved to: {output_file}")
    return True

def main():
    parser = argparse.ArgumentParser(description="Download and format datasets for μOmni training")
    parser.add_argument("--dataset", choices=["all", "text", "images", "audio"], default="all",
                       help="Which dataset to download/convert (default: all)")
    parser.add_argument("--skip-download", action="store_true",
                       help="Skip download, only convert existing data")
    parser.add_argument("--skip-convert", action="store_true",
                       help="Skip conversion, only download")
    parser.add_argument("--reset", action="store_true",
                       help="Reset state and re-download/convert everything")
    parser.add_argument("--dialogstudio-combined", action="store_true",
                       help="Load combined DialogStudio dataset instead of all 86 sub-datasets (faster but smaller)")
    
    args = parser.parse_args()
    
    # Load or reset state
    if args.reset:
        print("Resetting state...")
        state = {
            "dialogstudio": {"downloaded": False, "converted": False, "sub_datasets_loaded": 0, "total_sub_datasets": 86, "conversations_count": 0},
            "coco": {"images_downloaded": False, "val_images_downloaded": False, "annotations_downloaded": False, "converted": False},
            "librispeech": {"downloaded": False, "converted": False}
        }
        save_state(state)
    else:
        state = load_state()
    
    print("="*60)
    print("μOmni Dataset Download and Format Script")
    print("="*60)
    print(f"State file: {STATE_FILE}")
    print(f"Dataset: {args.dataset}")
    print("="*60)
    
    success = True
    
    # Text dataset (DialogStudio)
    if args.dataset in ["all", "text"]:
        if not args.skip_download:
            success = download_dialogstudio(state) and success
        if not args.skip_convert:
            success = convert_dialogstudio(state, load_all_sub_datasets=not args.dialogstudio_combined) and success
    
    # Image dataset (COCO)
    if args.dataset in ["all", "images"]:
        if not args.skip_download:
            success = download_coco(state) and success
            if success:
                success = extract_coco(state) and success
        if not args.skip_convert:
            success = convert_coco(state) and success
    
    # Audio dataset (LibriSpeech)
    if args.dataset in ["all", "audio"]:
        if not args.skip_download:
            success = download_librispeech(state) and success
            if success:
                success = extract_librispeech(state) and success
        if not args.skip_convert:
            success = convert_librispeech(state) and success
    
    print("\n" + "="*60)
    if success:
        print("✓ All operations completed successfully!")
        print("\nNext steps:")
        print("1. Update config files with correct paths")
        print("2. Run training scripts:")
        print("   python train_text.py --config configs/thinker_tiny.json")
        print("   python train_vision.py --config configs/vision_tiny.json")
        print("   python train_audio_enc.py --config configs/audio_enc_tiny.json")
        print("   python train_talker.py --config configs/talker_tiny.json")
        print("   python sft_omni.py --config configs/omni_sft_tiny.json")
    else:
        print("✗ Some operations failed. Check errors above.")
        print("You can resume by running the script again (it will skip completed steps)")
    print("="*60)

if __name__ == "__main__":
    main()

