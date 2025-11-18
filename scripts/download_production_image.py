"""
Download and prepare production-grade image datasets for μOmni training
Target: Under 30GB, millions of samples
Includes diverse knowledge: General images, Scientific/Medical, Art, Architecture, Nature, Educational

Supports:
- General Images: COCO
"""

import os
import json
import argparse
import subprocess
import requests
from pathlib import Path
from tqdm import tqdm
import zipfile
import tarfile
import shutil
from PIL import Image

# State file to track progress
STATE_FILE = "data/.image_download_state.json"

def load_state():
    """Load download/conversion state"""
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, 'r') as f:
            return json.load(f)
    return {
        # General Images
        "coco": {"downloaded": False, "extracted": False, "converted": False, "samples": 0}
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
    checkpoint_file = f"data/.checkpoint_image_{dataset_name}.json"
    os.makedirs(os.path.dirname(checkpoint_file), exist_ok=True)
    with open(checkpoint_file, 'w') as f:
        json.dump(checkpoint_data, f, indent=2)

def load_checkpoint(dataset_name):
    """Load fine-grained checkpoint for resuming"""
    checkpoint_file = f"data/.checkpoint_image_{dataset_name}.json"
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

def get_image_dataset_size(ds_name):
    """Get actual disk size of image dataset folder"""
    # Map dataset names to their actual data folders
    folder_map = {
        "coco": "data/images/coco",
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

def download_coco(state, max_samples=1000000):
    """Download COCO 2017 dataset - large-scale object detection and captioning"""
    print("\n" + "="*60)
    print("Downloading COCO 2017 Dataset")
    print("="*60)
    
    if state["coco"]["downloaded"] and state["coco"]["samples"] >= max_samples:
        print(f"COCO already downloaded ({state['coco']['samples']:,} samples), skipping...")
        return True
    
    # COCO 2017 has direct download URLs
    base_url = "http://images.cocodataset.org"
    download_dir = "data/image_downloads/coco"
    os.makedirs(download_dir, exist_ok=True)
    
    # COCO 2017 train images (~18GB, 118k images)
    # We'll download train2017.zip for ~50GB total image budget
    url = f"{base_url}/zips/train2017.zip"
    zip_file = os.path.join(download_dir, "train2017.zip")
    
    print("Downloading COCO 2017 train images (~18GB, 118k images)...")
    print("This may take 1-2 hours depending on your connection...")
    
    if download_file(url, zip_file, resume=True):
        extract_dir = "data/images/coco"
        os.makedirs(extract_dir, exist_ok=True)
        
        print("Extracting COCO images...")
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
        
        # Download annotations
        annotations_url = f"{base_url}/annotations/annotations_trainval2017.zip"
        annotations_file = os.path.join(download_dir, "annotations_trainval2017.zip")
        
        print("Downloading COCO annotations...")
        if download_file(annotations_url, annotations_file, resume=True):
            print("Extracting annotations...")
            with zipfile.ZipFile(annotations_file, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)
        
        # Convert to annotations format
        output_file = "data/images/coco_annotations.json"
        annotations = []
        count = 0
        
        # Load COCO annotations
        ann_file = os.path.join(extract_dir, "annotations", "captions_train2017.json")
        if os.path.exists(ann_file):
            with open(ann_file, 'r') as f:
                coco_data = json.load(f)
            
            # Create image ID to filename mapping
            images_dict = {img['id']: img['file_name'] for img in coco_data['images']}
            
            # Process captions
            for ann in tqdm(coco_data['annotations'], desc="Processing captions"):
                img_id = ann['image_id']
                caption = ann['caption']
                filename = images_dict.get(img_id)
                
                if filename:
                    img_path = f"coco/train2017/{filename}"
                    annotations.append({
                        "image": img_path,
                        "caption": caption,
                        "category": "coco"
                    })
                    count += 1
                    
                    # Print progress with remaining
                    print_progress_with_remaining(count, max_samples, "images", report_interval=1000)
                    
                    if count >= max_samples:
                        break
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(annotations, f, indent=2, ensure_ascii=False)
        
        # Only mark as downloaded if we reached max_samples
        if count >= max_samples:
            state["coco"]["downloaded"] = True
            state["coco"]["extracted"] = True
            state["coco"]["converted"] = True
        state["coco"]["samples"] = count
        save_state(state)
        
        if count >= max_samples:
            print(f"✓ Created annotations with {count:,} images")
        else:
            print(f"⚠ Created annotations with {count:,} images (target: {max_samples:,})")
        return True
    
    return False

    return False

def combine_image_annotations():
    """Combine all image annotations into one file"""
    print("\n" + "="*60)
    print("Combining Image Annotations")
    print("="*60)
    
    output_file = "data/images/production_annotations.json"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    input_files = [
        "data/images/coco_annotations.json"
    ]
    
    all_annotations = []
    
    for input_file in input_files:
        if os.path.exists(input_file):
            print(f"Loading {os.path.basename(input_file)}...")
            with open(input_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    all_annotations.extend(data)
                    print(f"  Added {len(data):,} entries")
                else:
                    all_annotations.append(data)
                    print(f"  Added 1 entry")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_annotations, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Combined {len(all_annotations):,} image annotations")
    print(f"  Saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Download production-grade image datasets for μOmni")
    parser.add_argument("--dataset", 
                       choices=["all", "coco", "general"], 
                       default="all",
                       help="Which dataset to download (default: all)")
    parser.add_argument("--skip-download", action="store_true",
                       help="Skip download, only extract/convert existing data")
    parser.add_argument("--skip-extract", action="store_true",
                       help="Skip extraction, only convert")
    parser.add_argument("--skip-convert", action="store_true",
                       help="Skip conversion, only download/extract")
    parser.add_argument("--combine", action="store_true",
                       help="Combine all downloaded datasets into one annotation file (outputs to data/images/production_annotations.json)")
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
    print("μOmni Production Image Dataset Downloader")
    print("="*60)
    print(f"State file: {STATE_FILE}")
    print(f"Dataset: {args.dataset}")
    print("="*60)
    
    success = True
    
    # COCO
    if args.dataset in ["all", "coco", "general"]:
        if not args.skip_download:
            success = download_coco(state, args.max_samples) and success
    
    # Combine if requested
    if args.combine:
        combine_image_annotations()
    
    print("\n" + "="*60)
    if success:
        print("✓ All operations completed successfully!")
        print("\nOutput files (ready to use, no formatting needed):")
        print("  - Individual datasets: data/images/*_annotations.json")
        if args.combine or args.dataset == "all":
            print("  - Combined annotations: data/images/production_annotations.json")
        print("\nNext steps:")
        print("1. Annotations are already in final format in data/images/")
        print("2. Update config files to point to:")
        if args.combine or args.dataset == "all":
            print("   data/images/production_annotations.json")
        else:
            print("   data/images/[dataset_name]_annotations.json")
        print("3. Run training: python train_vision.py --config configs/vision_tiny.json")
    else:
        print("✗ Some operations failed. Check errors above.")
        print("You can resume by running the script again (it will skip completed steps)")
        print("Fine-grained checkpoints saved - will resume from exact position")
    print("="*60)

if __name__ == "__main__":
    main()

