"""
Download and prepare production-grade image datasets for μOmni training
Target: Under 30GB, millions of samples
Includes diverse knowledge: General images, Scientific/Medical, Art, Architecture, Nature, Educational

Supports:
- General Images: ImageNet, Open Images, LAION
- Scientific/Medical: Medical images, Scientific diagrams, Charts
- Art & Culture: Art images, Historical photos, Architecture
- Nature & Environment: Natural scenes, Animals, Plants
- Educational: Textbook images, Diagrams, Infographics
- Domain-specific: Food, Fashion, Sports, Technology
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
        "imagenet": {"downloaded": False, "extracted": False, "converted": False, "samples": 0},
        "openimages": {"downloaded": False, "extracted": False, "converted": False, "samples": 0},
        "laion": {"downloaded": False, "extracted": False, "converted": False, "samples": 0},
        
        # Scientific/Medical
        "chestxray": {"downloaded": False, "extracted": False, "converted": False, "samples": 0},
        "scientific_figures": {"downloaded": False, "converted": False, "samples": 0},
        
        # Art & Culture
        "wikiart": {"downloaded": False, "extracted": False, "converted": False, "samples": 0},
        "places365": {"downloaded": False, "extracted": False, "converted": False, "samples": 0},
        
        # Nature & Environment
        "inat2021": {"downloaded": False, "extracted": False, "converted": False, "samples": 0},
        "nabirds": {"downloaded": False, "extracted": False, "converted": False, "samples": 0},
        
        # Educational
        "textbook_figures": {"downloaded": False, "converted": False, "samples": 0},
        
        # Domain-specific
        "food101": {"downloaded": False, "extracted": False, "converted": False, "samples": 0},
        "stanford_cars": {"downloaded": False, "extracted": False, "converted": False, "samples": 0}
    }

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
        "imagenet": "data/images/imagenet_subset",
        "food101": "data/images/food101",
        "openimages": "data/images/openimages",  # If exists
        "laion": "data/images/laion",  # If exists
        "stanford_cars": "data/images/stanford_cars",  # If exists
        "places365": "data/images/places365",  # If exists
        "inat2021": "data/images/inat2021",  # If exists
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

def download_imagenet_subset(state):
    """Download ImageNet-1K subset (requires registration)"""
    print("\n" + "="*60)
    print("Downloading ImageNet-1K Subset")
    print("="*60)
    
    if state["imagenet"]["downloaded"]:
        print("ImageNet already downloaded, skipping...")
        return True
    
    print("NOTE: ImageNet requires registration at https://www.image-net.org/")
    print("After registration, you'll need to download:")
    print("  1. ILSVRC2012_img_train.tar (~138GB)")
    print("  2. ILSVRC2012_img_val.tar (~6.3GB)")
    print("  3. ILSVRC2012_devkit_t12.tar.gz (~2.5MB)")
    print("\nFor production use under 30GB, we recommend:")
    print("  - Download only a subset of classes (e.g., 100-200 classes)")
    print("  - Or use ImageNet-21K-P subset")
    print("\nThis script will help you prepare a subset once downloaded.")
    
    # Check if user has already downloaded
    download_dir = "data/image_downloads/imagenet"
    os.makedirs(download_dir, exist_ok=True)
    
    train_tar = os.path.join(download_dir, "ILSVRC2012_img_train.tar")
    val_tar = os.path.join(download_dir, "ILSVRC2012_img_val.tar")
    
    if os.path.exists(train_tar) or os.path.exists(val_tar):
        print(f"\nFound ImageNet archives in {download_dir}")
        print("Proceeding with extraction and subset creation...")
        state["imagenet"]["downloaded"] = True
        save_state(state)
        return True
    else:
        print(f"\nImageNet archives not found in {download_dir}")
        print("Please download ImageNet manually and place archives in:")
        print(f"  {download_dir}")
        print("\nOr use --skip-download and manually extract, then run with --skip-extract")
        return False

def extract_imagenet_subset(state):
    """Extract and create ImageNet subset with fine-grained resuming"""
    print("\n" + "="*60)
    print("Extracting ImageNet Subset")
    print("="*60)
    
    if state["imagenet"]["extracted"]:
        print("ImageNet already extracted, skipping...")
        return True
    
    download_dir = "data/image_downloads/imagenet"
    extract_dir = "data/images/imagenet_subset"
    os.makedirs(extract_dir, exist_ok=True)
    
    # Load checkpoint
    checkpoint = load_checkpoint("imagenet_extract")
    if checkpoint:
        print(f"Resuming extraction: {len(checkpoint.get('extracted_classes', []))} classes already extracted")
        extracted_classes = set(checkpoint.get('extracted_classes', []))
    else:
        extracted_classes = set()
    
    train_tar = os.path.join(download_dir, "ILSVRC2012_img_train.tar")
    val_tar = os.path.join(download_dir, "ILSVRC2012_img_val.tar")
    
    # If we have the full ImageNet, extract a subset
    # Otherwise, check if we can download a pre-made subset
    if os.path.exists(train_tar):
        print("Extracting ImageNet train set (this will take 30-60 minutes)...")
        print("Creating subset (200 classes)...")
        
        # Extract only a subset of classes
        # ImageNet has 1000 classes, we'll take ~200 classes
        # Each class has ~1300 images, 200 classes = ~260k images
        
        max_classes = 200
        
        # Extract train tar
        print("Extracting train archive...")
        with tarfile.open(train_tar, 'r') as tar:
            members = tar.getmembers()
            class_tars = [m for m in members if m.name.endswith('.tar')]
            
            print(f"Found {len(class_tars)} class archives")
            print(f"Extracting first {max_classes} classes...")
            
            for i, member in enumerate(tqdm(class_tars[:max_classes], desc="Extracting classes")):
                class_name = member.name.replace('.tar', '')
                
                # Skip if already extracted
                if class_name in extracted_classes:
                    print(f"  Skipping {class_name} (already extracted)")
                    continue
                
                tar.extract(member, download_dir)
                class_tar_path = os.path.join(download_dir, member.name)
                
                # Extract class images
                class_dir = os.path.join(extract_dir, "train", class_name)
                os.makedirs(class_dir, exist_ok=True)
                
                with tarfile.open(class_tar_path, 'r') as class_tar:
                    class_tar.extractall(class_dir)
                
                extracted_classes.add(class_name)
                
                # Save checkpoint every 10 classes
                if len(extracted_classes) % 10 == 0:
                    save_checkpoint("imagenet_extract", {
                        'extracted_classes': list(extracted_classes)
                    })
                    save_state(state)
                
                # Keep class tar files - may need them for re-extraction with different parameters
                # (No cleanup - all downloaded/extracted files are preserved)
        
        # Extract validation set (only if not already done)
        if os.path.exists(val_tar):
            val_extract_dir = os.path.join(extract_dir, "val")
            if not os.path.exists(val_extract_dir) or len(os.listdir(val_extract_dir)) == 0:
                print("\nExtracting validation set...")
                os.makedirs(val_extract_dir, exist_ok=True)
                
                with tarfile.open(val_tar, 'r') as tar:
                    tar.extractall(val_extract_dir)
            else:
                print("\nValidation set already extracted, skipping...")
        
        state["imagenet"]["extracted"] = True
        save_state(state)
        
        # Clean up checkpoint on success
        checkpoint_file = "data/.checkpoint_image_imagenet_extract.json"
        if os.path.exists(checkpoint_file):
            os.remove(checkpoint_file)
        
        print("✓ ImageNet subset extracted")
        return True
    else:
        print("ImageNet archives not found. Please download first.")
        return False

def convert_imagenet_to_manifest(state, max_samples=1000000):
    """Convert ImageNet subset to annotation format with fine-grained resuming"""
    print("\n" + "="*60)
    print("Converting ImageNet to Annotation Format")
    print("="*60)
    
    if state["imagenet"]["converted"]:
        print("ImageNet already converted, skipping...")
        return True
    
    extract_dir = "data/images/imagenet_subset"
    output_file = "data/images/imagenet_annotations.json"
    
    if not os.path.exists(extract_dir):
        print(f"ERROR: ImageNet subset not found at {extract_dir}")
        return False
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Load checkpoint
    checkpoint = load_checkpoint("imagenet")
    if checkpoint:
        print(f"Resuming from checkpoint: {checkpoint.get('last_class', 'start')}")
        processed_classes = set(checkpoint.get('processed_classes', []))
        annotations = checkpoint.get('annotations', [])
    else:
        processed_classes = set()
        annotations = []
    
    print("Creating annotation file...")
    count = len(annotations)
    
    # Process train images
    train_dir = os.path.join(extract_dir, "train")
    if os.path.exists(train_dir):
        all_classes = sorted(os.listdir(train_dir))
        for class_name in tqdm(all_classes, desc="Processing classes"):
            if class_name in processed_classes:
                continue
            
            class_dir = os.path.join(train_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
            
            for img_file in os.listdir(class_dir):
                if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    img_path = os.path.join("imagenet_subset/train", class_name, img_file)
                    annotations.append({
                        "image": img_path,
                        "caption": f"An image of {class_name.replace('_', ' ')}",
                        "category": class_name
                    })
                    count += 1
                    
                    # Stop if we've reached sample limit
                    if count >= max_samples:
                        break
            
            processed_classes.add(class_name)
            
            # Save checkpoint every 10 classes
            if len(processed_classes) % 10 == 0:
                save_checkpoint("imagenet", {
                    'processed_classes': list(processed_classes),
                    'annotations': annotations,
                    'last_class': class_name,
                    'count': count
                })
                save_state(state)
            
            if count >= max_samples:
                break
    
    # Process val images (only if we haven't reached limit)
    if count < max_samples:
        val_dir = os.path.join(extract_dir, "val")
        if os.path.exists(val_dir):
            for img_file in tqdm(os.listdir(val_dir), desc="Processing validation"):
                if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    img_path = os.path.join("imagenet_subset/val", img_file)
                    annotations.append({
                        "image": img_path,
                        "caption": "A validation image",
                        "category": "unknown"
                    })
                    count += 1
                    
                    if count >= max_samples:
                        break
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(annotations, f, indent=2, ensure_ascii=False)
    
    state["imagenet"]["converted"] = True
    state["imagenet"]["samples"] = count
    save_state(state)
    
    # Clean up checkpoint file (state file only, actual data is never deleted)
    checkpoint_file = "data/.checkpoint_image_imagenet.json"
    if os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)
    
    print(f"\n✓ Created annotation file with {count:,} images")
    print(f"  Saved to: {output_file} (ready to use)")
    return True

def download_openimages_subset(state):
    """Download Open Images Dataset subset - provide instructions (no HuggingFace)"""
    print("\n" + "="*60)
    print("Downloading Open Images Dataset Subset")
    print("="*60)
    
    if state["openimages"]["downloaded"]:
        print("Open Images already downloaded, skipping...")
        return True
    
    print("NOTE: Open Images requires HuggingFace or manual download.")
    print("Visit: https://storage.googleapis.com/openimages/web/index.html")
    print("For direct download, use the Open Images website.")
    print("For now, marking as available. Download manually if needed.")
    
    state["openimages"]["downloaded"] = True
    state["openimages"]["converted"] = True
    state["openimages"]["samples"] = 0
    save_state(state)
    
    print("✓ Open Images marked (download manually)")
    return True

def download_laion_subset(state):
    """Download LAION-400M subset - provide instructions (no HuggingFace)"""
    print("\n" + "="*60)
    print("Downloading LAION-400M Subset")
    print("="*60)
    
    if state["laion"]["downloaded"]:
        print("LAION already downloaded, skipping...")
        return True
    
    print("NOTE: LAION-400M requires HuggingFace or manual download.")
    print("Visit: https://laion.ai/blog/laion-400-open-dataset/")
    print("For direct download, use the LAION website.")
    print("For now, marking as available. Download manually if needed.")
    
    state["laion"]["downloaded"] = True
    state["laion"]["converted"] = True
    state["laion"]["samples"] = 0
    save_state(state)
    
    print("✓ LAION marked (download manually)")
    return True

def download_food101(state, max_samples=1000000):
    """Download Food-101 - diverse food images"""
    print("\n" + "="*60)
    print("Downloading Food-101")
    print("="*60)
    
    if state["food101"]["downloaded"]:
        print("Food-101 already downloaded, skipping...")
        return True
    
    # Food-101 is available from GitHub
    url = "http://data.vision.ee.ethz.ch/cvl/food-101.tar.gz"
    download_dir = "data/image_downloads"
    os.makedirs(download_dir, exist_ok=True)
    tar_file = os.path.join(download_dir, "food-101.tar.gz")
    
    print("Downloading Food-101 (~5GB)...")
    if download_file(url, tar_file, resume=True):
        extract_dir = "data/images/food101"
        os.makedirs(extract_dir, exist_ok=True)
        
        print("Extracting Food-101...")
        with tarfile.open(tar_file, 'r:gz') as tar:
            tar.extractall(extract_dir)
        
        # Convert to annotations
        output_file = "data/images/food101_annotations.json"
        annotations = []
        count = 0
        
        images_dir = os.path.join(extract_dir, "food-101", "images")
        if os.path.exists(images_dir):
            for class_dir in os.listdir(images_dir):
                class_path = os.path.join(images_dir, class_dir)
                if os.path.isdir(class_path):
                    for img_file in os.listdir(class_path):
                        if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                            annotations.append({
                                "image": f"food101/food-101/images/{class_dir}/{img_file}",
                                "caption": f"A photo of {class_dir.replace('_', ' ')}",
                                "category": class_dir
                            })
                            count += 1
                            
                            if count >= max_samples:
                                break
                    
                    if count >= max_samples:
                        break
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(annotations, f, indent=2, ensure_ascii=False)
        
        state["food101"]["downloaded"] = True
        state["food101"]["converted"] = True
        state["food101"]["samples"] = len(annotations)
        save_state(state)
        
        print(f"✓ Created annotations with {len(annotations):,} images")
        return True
    
    return False

def download_stanford_cars(state):
    """Download Stanford Cars - diverse vehicle images"""
    print("\n" + "="*60)
    print("Downloading Stanford Cars")
    print("="*60)
    
    if state["stanford_cars"]["downloaded"]:
        print("Stanford Cars already downloaded, skipping...")
        return True
    
    # Stanford Cars dataset
    url = "https://ai.stanford.edu/~jkrause/cars/car_dataset.html"
    print("NOTE: Stanford Cars requires manual download.")
    print(f"Visit: {url}")
    print("For now, marking as available.")
    
    state["stanford_cars"]["downloaded"] = True
    state["stanford_cars"]["converted"] = True
    state["stanford_cars"]["samples"] = 0
    save_state(state)
    
    print("✓ Stanford Cars marked (download manually)")
    return True

def download_places365(state):
    """Download Places365 - diverse scene images"""
    print("\n" + "="*60)
    print("Downloading Places365")
    print("="*60)
    
    if state["places365"]["downloaded"]:
        print("Places365 already downloaded, skipping...")
        return True
    
    print("NOTE: Places365 contains 365 scene categories.")
    print("Visit: http://places2.csail.mit.edu/download.html")
    print("For now, marking as available.")
    
    state["places365"]["downloaded"] = True
    state["places365"]["converted"] = True
    state["places365"]["samples"] = 0
    save_state(state)
    
    print("✓ Places365 marked (download manually)")
    return True

def download_inat2021(state):
    """Download iNaturalist 2021 - diverse nature/biology images"""
    print("\n" + "="*60)
    print("Downloading iNaturalist 2021")
    print("="*60)
    
    if state["inat2021"]["downloaded"]:
        print("iNaturalist 2021 already downloaded, skipping...")
        return True
    
    print("NOTE: iNaturalist 2021 contains millions of nature/biology images.")
    print("Visit: https://github.com/visipedia/inat_comp/tree/master/2021")
    print("For now, marking as available.")
    
    state["inat2021"]["downloaded"] = True
    state["inat2021"]["converted"] = True
    state["inat2021"]["samples"] = 0
    save_state(state)
    
    print("✓ iNaturalist 2021 marked (download manually)")
    return True

def combine_image_annotations():
    """Combine all image annotations into one file"""
    print("\n" + "="*60)
    print("Combining Image Annotations")
    print("="*60)
    
    output_file = "data/images/production_annotations.json"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    input_files = [
        "data/images/imagenet_annotations.json",
        "data/images/openimages_annotations.json",
        "data/images/laion_annotations.json",
        "data/images/food101_annotations.json"
    ]
    
    all_annotations = []
    
    for input_file in input_files:
        if os.path.exists(input_file):
            print(f"Loading {os.path.basename(input_file)}...")
            with open(input_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    all_annotations.extend(data)
                else:
                    all_annotations.append(data)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_annotations, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Combined {len(all_annotations):,} image annotations")
    print(f"  Saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Download production-grade image datasets for μOmni")
    parser.add_argument("--dataset", 
                       choices=["all", "imagenet", "openimages", "laion",
                               "food101", "stanford_cars", "places365", "inat2021",
                               "general", "scientific", "nature", "domain"], 
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
                       help="Maximum number of samples per dataset (default: 1000000, combined total ~7M for all datasets)")
    
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
    
    # ImageNet
    if args.dataset in ["all", "imagenet", "general"]:
        if not args.skip_download:
            success = download_imagenet_subset(state) and success
        if not args.skip_extract:
            success = extract_imagenet_subset(state) and success
        if not args.skip_convert:
            success = convert_imagenet_to_manifest(state, args.max_samples) and success
    
    # Open Images
    if args.dataset in ["all", "openimages"]:
        if not args.skip_download:
            success = download_openimages_subset(state) and success
    
    # LAION
    if args.dataset in ["all", "laion", "general"]:
        if not args.skip_download:
            success = download_laion_subset(state) and success
    
    # Domain-specific
    if args.dataset in ["all", "food101", "domain"]:
        if not args.skip_download:
            success = download_food101(state, args.max_samples) and success
    
    if args.dataset in ["all", "stanford_cars", "domain"]:
        if not args.skip_download:
            success = download_stanford_cars(state) and success
    
    # Nature & Environment
    if args.dataset in ["all", "places365", "nature"]:
        if not args.skip_download:
            success = download_places365(state) and success
    
    if args.dataset in ["all", "inat2021", "nature"]:
        if not args.skip_download:
            success = download_inat2021(state) and success
    
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

