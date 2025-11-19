
"""
Download and prepare OCR (Optical Character Recognition) datasets for μOmni training.

Supported datasets:
- SynthText: Synthetic text in natural scenes (large, ~40GB)
- MJSynth: Synthetic text dataset (medium, ~10GB)
- TextOCR: Real-world text in images (medium, ~5GB)
- ICDAR 2015: Scene text detection and recognition (small, ~1GB)

Note: Some datasets require manual download or have limited availability.
This script focuses on publicly available OCR datasets.
"""

import os
import json
import argparse
import requests
from pathlib import Path
from tqdm import tqdm
import zipfile
import tarfile
import shutil
from PIL import Image
import csv

# State file to track progress
STATE_FILE = "data/.ocr_download_state.json"

def load_state():
    """Load download/conversion state"""
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, 'r') as f:
            return json.load(f)
    return {
        "mjsynth": {"downloaded": False, "extracted": False, "converted": False, "samples": 0},
        "textocr": {"downloaded": False, "extracted": False, "converted": False, "samples": 0},
    }

def save_state(state):
    """Save download/conversion state"""
    os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)
    with open(STATE_FILE, 'w') as f:
        json.dump(state, f, indent=2)

def download_file(url, filepath, resume=True):
    """Download file with progress bar and resume support"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    if os.path.exists(filepath):
        if resume:
            # Resume download
            headers = {}
            file_size = os.path.getsize(filepath)
            if file_size > 0:
                headers['Range'] = f'bytes={file_size}-'
            
            try:
                response = requests.get(url, headers=headers, stream=True, timeout=30)
                if response.status_code == 206:  # Partial content
                    mode = 'ab'
                    total_size = int(response.headers.get('content-length', 0)) + file_size
                elif response.status_code == 200:
                    mode = 'wb'
                    total_size = int(response.headers.get('content-length', 0))
                else:
                    print(f"Error: HTTP {response.status_code}")
                    return False
                
                with open(filepath, mode) as f:
                    if mode == 'ab':
                        f.seek(file_size)
                    with tqdm(total=total_size, initial=file_size, unit='B', unit_scale=True, desc=os.path.basename(filepath)) as pbar:
                        for chunk in response.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
                                pbar.update(len(chunk))
                return True
            except Exception as e:
                print(f"Resume failed: {e}, downloading from scratch...")
                os.remove(filepath)
        
        # If resume failed or not enabled, download from scratch
        os.remove(filepath)
    
    # Download from scratch
    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        total_size = int(response.headers.get('content-length', 0))
        
        with open(filepath, 'wb') as f:
            with tqdm(total=total_size, unit='B', unit_scale=True, desc=os.path.basename(filepath)) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
        return True
    except Exception as e:
        print(f"Download failed: {e}")
        return False

def print_progress_with_remaining(current, max_count, label="samples", report_interval=100):
    """Print progress with remaining count and percentage"""
    if current % report_interval == 0 or current >= max_count:
        remaining = max_count - current
        percent = (current / max_count * 100) if max_count > 0 else 0
        print(f"Progress: {current:,} {label} ({percent:.1f}%) - Remaining: ~{remaining:,} {label}")

def download_mjsynth(state, max_samples=1000000):
    """Download MJSynth (Synth90k) dataset - synthetic text in natural scenes"""
    print("\n" + "="*60)
    print("Downloading MJSynth (Synth90k) Dataset")
    print("="*60)
    
    if state["mjsynth"]["downloaded"] and state["mjsynth"]["extracted"] and state["mjsynth"]["converted"] and state["mjsynth"]["samples"] >= max_samples:
        print(f"MJSynth already downloaded and converted ({state['mjsynth']['samples']:,} samples), skipping...")
        return True
    
    # Check if we already have converted data
    output_csv = "data/ocr/mjsynth_ocr.csv"
    if os.path.exists(output_csv) and state["mjsynth"]["converted"]:
        print("MJSynth CSV already exists, skipping conversion...")
        state["mjsynth"]["converted"] = True
        with open(output_csv, 'r') as f:
            state["mjsynth"]["samples"] = sum(1 for _ in f) - 1  # Subtract header
        save_state(state)
        return True
    
    # MJSynth direct download links (Synth90k dataset)
    # Using publicly available mirrors
    download_dir = "data/ocr_downloads/mjsynth"
    os.makedirs(download_dir, exist_ok=True)
    
    # MJSynth dataset parts - downloading annotation file and images
    # Annotation file (small, contains all text labels)
    annotation_url = "https://www.robots.ox.ac.uk/~vgg/data/text/mjsynth.tar.gz"
    annotation_file = os.path.join(download_dir, "mjsynth.tar.gz")
    
    # Alternative: Direct link to annotation file if available
    # If the main link doesn't work, we'll try alternative sources
    alt_urls = [
        "https://www.robots.ox.ac.uk/~vgg/data/text/mjsynth.tar.gz",
        # Add more mirrors if needed
    ]
    
    if not state["mjsynth"]["downloaded"]:
        print("Downloading MJSynth dataset...")
        print(f"URL: {annotation_url}")
        print("Note: This is a large dataset (~10GB). Download may take time.")
        
        downloaded = False
        for url in alt_urls:
            if download_file(url, annotation_file, resume=True):
                downloaded = True
                break
        
        if not downloaded:
            print("⚠️  Failed to download MJSynth from direct links.")
            print("   Please download manually from: https://www.robots.ox.ac.uk/~vgg/data/text/")
            print("   Extract to: data/ocr_downloads/mjsynth/")
            return False
        
        state["mjsynth"]["downloaded"] = True
        save_state(state)
        print("✓ MJSynth downloaded successfully")
    
    # Extract if needed
    if not state["mjsynth"]["extracted"]:
        print("Extracting MJSynth dataset...")
        extract_dir = os.path.join(download_dir, "extracted")
        os.makedirs(extract_dir, exist_ok=True)
        
        try:
            if annotation_file.endswith('.tar.gz'):
                with tarfile.open(annotation_file, 'r:gz') as tar:
                    tar.extractall(extract_dir)
            elif annotation_file.endswith('.zip'):
                with zipfile.ZipFile(annotation_file, 'r') as zip_ref:
                    zip_ref.extractall(extract_dir)
            
            state["mjsynth"]["extracted"] = True
            save_state(state)
            print("✓ MJSynth extracted successfully")
        except Exception as e:
            print(f"ERROR extracting MJSynth: {e}")
            return False
    
    # Convert to CSV format
    if not state["mjsynth"]["converted"]:
        print("Converting MJSynth to CSV format...")
        extract_dir = os.path.join(download_dir, "extracted")
        output_dir = "data/ocr/mjsynth"
        os.makedirs(output_dir, exist_ok=True)
        
        # Find annotation files (typically annotation.txt or similar)
        annotation_files = list(Path(extract_dir).rglob("*.txt"))
        annotation_files.extend(list(Path(extract_dir).rglob("annotation*")))
        
        samples = []
        for ann_file in annotation_files[:1]:  # Use first annotation file found
            print(f"Processing annotation file: {ann_file}")
            with open(ann_file, 'r', encoding='utf-8') as f:
                for line in tqdm(f, desc="Reading annotations"):
                    line = line.strip()
                    if not line:
                        continue
                    
                    # MJSynth format: path/to/image.jpg "text"
                    # Or: path/to/image.jpg text
                    parts = line.split()
                    if len(parts) >= 2:
                        img_path = parts[0]
                        # Remove quotes if present
                        text = ' '.join(parts[1:]).strip('"\'')
                        
                        # Find actual image file
                        img_full_path = Path(extract_dir) / img_path
                        if img_full_path.exists():
                            # Copy image to output directory
                            rel_path = os.path.relpath(img_full_path, extract_dir)
                            dst_img = os.path.join(output_dir, rel_path)
                            os.makedirs(os.path.dirname(dst_img), exist_ok=True)
                            if not os.path.exists(dst_img):
                                shutil.copy2(img_full_path, dst_img)
                            
                            samples.append({
                                "image": f"mjsynth/{rel_path}",
                                "text": text
                            })
                            
                            if len(samples) >= max_samples:
                                break
        
        # Write CSV
        with open(output_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["image", "text"])
            for sample in samples:
                writer.writerow([sample["image"], sample["text"]])
        
        state["mjsynth"]["converted"] = True
        state["mjsynth"]["samples"] = len(samples)
        save_state(state)
        
        print(f"✓ Created MJSynth CSV with {len(samples):,} samples")
    
    return True

def download_textocr(state, max_samples=1000000):
    """Download TextOCR dataset - real-world text in images"""
    print("\n" + "="*60)
    print("Downloading TextOCR Dataset")
    print("="*60)
    
    if state["textocr"]["downloaded"] and state["textocr"]["extracted"] and state["textocr"]["converted"] and state["textocr"]["samples"] >= max_samples:
        print(f"TextOCR already downloaded and converted ({state['textocr']['samples']:,} samples), skipping...")
        return True
    
    # Check if we already have converted data
    output_csv = "data/ocr/textocr_ocr.csv"
    if os.path.exists(output_csv) and state["textocr"]["converted"]:
        print("TextOCR CSV already exists, skipping conversion...")
        state["textocr"]["converted"] = True
        with open(output_csv, 'r') as f:
            state["textocr"]["samples"] = sum(1 for _ in f) - 1  # Subtract header
        save_state(state)
        return True
    
    download_dir = "data/ocr_downloads/textocr"
    os.makedirs(download_dir, exist_ok=True)
    
    # TextOCR direct download links (Google Cloud Storage)
    # Annotation file
    annotation_url = "https://dl.fbaipublicfiles.com/textvqa/data/textocr/TextOCR_0.1_train.json"
    annotation_file = os.path.join(download_dir, "TextOCR_train.json")
    
    # Alternative URLs (try multiple sources)
    alt_annotation_urls = [
        "https://dl.fbaipublicfiles.com/textvqa/data/textocr/TextOCR_0.1_train.json",
        "https://textvqa.org/textocr/dataset",
    ]
    
    # Download annotation file
    if not state["textocr"]["downloaded"]:
        print("Downloading TextOCR annotation file...")
        print(f"URL: {annotation_url}")
        
        downloaded = False
        for url in alt_annotation_urls:
            try:
                if download_file(url, annotation_file, resume=True):
                    downloaded = True
                    break
            except:
                continue
        
        if not downloaded:
            # Try to download images zip if available
            images_url = "https://dl.fbaipublicfiles.com/textvqa/data/textocr/TextOCR_0.1_train.zip"
            images_file = os.path.join(download_dir, "TextOCR_images.zip")
            print("Trying to download images archive...")
            if download_file(images_url, images_file, resume=True):
                state["textocr"]["downloaded"] = True
                state["textocr"]["extracted"] = False  # Will extract next
                save_state(state)
            else:
                print("⚠️  Failed to download TextOCR from direct links.")
                print("   Please download manually from: https://textvqa.org/textocr/dataset")
                print("   Extract to: data/ocr_downloads/textocr/")
                return False
        else:
            state["textocr"]["downloaded"] = True
            save_state(state)
            print("✓ TextOCR annotation file downloaded")
    
    # Extract images if we have a zip file
    textocr_dir = download_dir
    images_zip = os.path.join(download_dir, "TextOCR_0.1_train.zip")
    if os.path.exists(images_zip) and not state["textocr"]["extracted"]:
        print("Extracting TextOCR images...")
        extract_dir = os.path.join(download_dir, "images")
        os.makedirs(extract_dir, exist_ok=True)
        
        try:
            with zipfile.ZipFile(images_zip, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)
            state["textocr"]["extracted"] = True
            save_state(state)
            print("✓ TextOCR images extracted")
        except Exception as e:
            print(f"ERROR extracting TextOCR: {e}")
            # Continue anyway - might have images already
    
    # Convert to CSV format
    if not state["textocr"]["converted"]:
        print("Converting TextOCR to CSV format...")
        output_dir = "data/ocr/textocr"
        os.makedirs(output_dir, exist_ok=True)
        
        # Look for annotation files
        ann_files = []
        if os.path.exists(annotation_file):
            ann_files.append(annotation_file)
        ann_files.extend(list(Path(textocr_dir).glob("*.json")))
        
        if not ann_files:
            print("⚠️  TextOCR annotation file not found.")
            print("   Expected: TextOCR_0.1_train.json or similar JSON file")
            return False
        
        ann_file = ann_files[0]
        print(f"Processing annotation file: {ann_file}")
        
        with open(ann_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # TextOCR format: {"images": [...], "annotations": [...]}
        images_dict = {}
        if "images" in data:
            for img in data["images"]:
                images_dict[img["id"]] = img
        
        annotations = []
        images_dir = os.path.join(textocr_dir, "images")
        if not os.path.exists(images_dir):
            images_dir = textocr_dir  # Fallback
        
        if "annotations" in data:
            for ann in tqdm(data["annotations"], desc="Processing TextOCR"):
                if ann.get("image_id") in images_dict:
                    img_info = images_dict[ann["image_id"]]
                    img_path = img_info.get("file_name", f"{ann['image_id']}.jpg")
                    text = ann.get("utf8_string", "")
                    
                    if text and img_path:
                        # Find image file
                        src_img = os.path.join(images_dir, img_path)
                        if not os.path.exists(src_img):
                            # Try alternative paths
                            for alt_path in [img_path, f"train_{img_path}", os.path.basename(img_path)]:
                                alt_src = os.path.join(images_dir, alt_path)
                                if os.path.exists(alt_src):
                                    src_img = alt_src
                                    img_path = alt_path
                                    break
                        
                        if os.path.exists(src_img):
                            # Copy image to output directory
                            dst_img = os.path.join(output_dir, os.path.basename(img_path))
                            os.makedirs(os.path.dirname(dst_img), exist_ok=True)
                            if not os.path.exists(dst_img):
                                shutil.copy2(src_img, dst_img)
                            
                            annotations.append({
                                "image": f"textocr/{os.path.basename(img_path)}",
                                "text": text
                            })
                            
                            if len(annotations) >= max_samples:
                                break
        
        # Write CSV
        with open(output_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["image", "text"])
            for ann in annotations:
                writer.writerow([ann["image"], ann["text"]])
        
        state["textocr"]["converted"] = True
        state["textocr"]["samples"] = len(annotations)
        save_state(state)
        
        print(f"✓ Created TextOCR CSV with {len(annotations):,} samples")
    
    return True

def combine_ocr_csvs():
    """Combine all OCR CSV files into one production file"""
    print("\n" + "="*60)
    print("Combining OCR Datasets")
    print("="*60)
    
    output_file = "data/ocr/production_ocr.csv"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    input_files = [
        "data/ocr/mjsynth_ocr.csv",
        "data/ocr/textocr_ocr.csv",
    ]
    
    all_rows = []
    header_written = False
    
    for csv_file in input_files:
        if os.path.exists(csv_file):
            print(f"Reading {csv_file}...")
            with open(csv_file, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                header = next(reader)  # Skip header
                if not header_written:
                    all_rows.append(header)
                    header_written = True
                
                row_count = 0
                for row in reader:
                    if len(row) >= 2:  # Ensure we have image and text
                        all_rows.append(row)
                        row_count += 1
                print(f"  Added {row_count:,} rows")
        else:
            print(f"  Skipping {csv_file} (not found)")
    
    # Write combined CSV
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerows(all_rows)
    
    total_samples = len(all_rows) - 1  # Subtract header
    print(f"\n✓ Combined {total_samples:,} OCR samples into {output_file}")
    return total_samples

def main():
    parser = argparse.ArgumentParser(description="Download OCR datasets for μOmni training")
    parser.add_argument("--dataset", type=str, default="all",
                       choices=["all", "mjsynth", "textocr"],
                       help="Dataset to download")
    parser.add_argument("--max-samples", type=int, default=1000000,
                       help="Maximum samples per dataset (default: 1,000,000)")
    parser.add_argument("--combine", action="store_true",
                       help="Combine all downloaded datasets into one CSV file")
    parser.add_argument("--reset", action="store_true",
                       help="Reset download state and re-download everything")
    
    args = parser.parse_args()
    
    state = load_state()
    
    if args.reset:
        print("Resetting download state...")
        state = {
            "mjsynth": {"downloaded": False, "extracted": False, "converted": False, "samples": 0},
            "textocr": {"downloaded": False, "extracted": False, "converted": False, "samples": 0},
        }
        save_state(state)
    
    success = True
    
    # MJSynth
    if args.dataset in ["all", "mjsynth"]:
        success = download_mjsynth(state, args.max_samples) and success
    
    # TextOCR
    if args.dataset in ["all", "textocr"]:
        success = download_textocr(state, args.max_samples) and success
    
    save_state(state)
    
    # Combine datasets
    if args.combine:
        combine_ocr_csvs()
        print("\n" + "="*60)
        print("Download Summary")
        print("="*60)
        print("  - Individual datasets: data/ocr/*_ocr.csv")
        if os.path.exists("data/ocr/production_ocr.csv"):
            print("  - Combined OCR: data/ocr/production_ocr.csv")
    
    if success:
        print("\n✓ OCR dataset download complete!")
    else:
        print("\n⚠️  Some datasets failed to download.")
        print("   Check the error messages above for details.")

if __name__ == "__main__":
    main()

