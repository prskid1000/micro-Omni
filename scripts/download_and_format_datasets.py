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
        "dialogstudio": {"downloaded": False, "converted": False},
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
    """Download DialogStudio from HuggingFace"""
    print("\n" + "="*60)
    print("Downloading DialogStudio (Text Data)")
    print("="*60)
    
    if state["dialogstudio"]["downloaded"]:
        print("DialogStudio already downloaded, skipping...")
        return True
    
    try:
        from datasets import load_dataset
        print("Loading DialogStudio from HuggingFace...")
        
        output_dir = "data/dialogstudio_raw"
        os.makedirs(output_dir, exist_ok=True)
        
        # Load dataset (this downloads it)
        print("Downloading dataset (this may take a while)...")
        ds = load_dataset('Salesforce/dialogstudio', trust_remote_code=True)
        
        # Save to disk for later use
        ds.save_to_disk(output_dir)
        
        state["dialogstudio"]["downloaded"] = True
        save_state(state)
        print("✓ DialogStudio downloaded successfully")
        return True
    except ImportError:
        print("ERROR: 'datasets' package not installed. Install with: pip install datasets")
        return False
    except Exception as e:
        print(f"ERROR downloading DialogStudio: {e}")
        return False

def convert_dialogstudio(state):
    """Convert DialogStudio to text format"""
    print("\n" + "="*60)
    print("Converting DialogStudio to Text Format")
    print("="*60)
    
    if state["dialogstudio"]["converted"]:
        print("DialogStudio already converted, skipping...")
        return True
    
    try:
        from datasets import load_from_disk
        
        input_dir = "data/dialogstudio_raw"
        if not os.path.exists(input_dir):
            print(f"ERROR: DialogStudio not downloaded. Run download first.")
            return False
        
        print("Loading DialogStudio from disk...")
        ds = load_from_disk(input_dir)
        
        output_file = "data/text/dialogstudio.txt"
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        print("Converting conversations to text format...")
        count = 0
        with open(output_file, 'w', encoding='utf-8') as f:
            for split_name in ['train', 'validation', 'test']:
                if split_name in ds:
                    print(f"Processing {split_name} split...")
                    for item in tqdm(ds[split_name], desc=f"Converting {split_name}"):
                        # Extract conversation text
                        if 'conversations' in item:
                            conv_text = ' '.join([
                                turn.get('content', '') or turn.get('value', '')
                                for turn in item['conversations']
                                if isinstance(turn, dict)
                            ])
                            if conv_text.strip():
                                f.write(conv_text.strip() + '\n')
                                count += 1
                        elif 'instruction' in item or 'input' in item or 'output' in item:
                            # Alternative format
                            parts = []
                            if 'instruction' in item:
                                parts.append(str(item['instruction']))
                            if 'input' in item:
                                parts.append(str(item['input']))
                            if 'output' in item:
                                parts.append(str(item['output']))
                            if parts:
                                f.write(' '.join(parts).strip() + '\n')
                                count += 1
        
        state["dialogstudio"]["converted"] = True
        save_state(state)
        print(f"✓ Converted {count} conversations to {output_file}")
        return True
    except Exception as e:
        print(f"ERROR converting DialogStudio: {e}")
        import traceback
        traceback.print_exc()
        return False

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
    
    args = parser.parse_args()
    
    # Load or reset state
    if args.reset:
        print("Resetting state...")
        state = {
            "dialogstudio": {"downloaded": False, "converted": False},
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
            success = convert_dialogstudio(state) and success
    
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

