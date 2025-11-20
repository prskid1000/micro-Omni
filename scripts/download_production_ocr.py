
"""
Download and prepare OCR (Optical Character Recognition) datasets for μOmni training.

Supported datasets:
- MJSynth (Synth90k): Synthetic text dataset (medium, ~10GB)

Note: MJSynth may require manual download if automatic download fails.
See script output for instructions.
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
    
    # Extract directly to final location (like image script does)
    output_dir = "data/ocr/mjsynth"
    if not state["mjsynth"]["extracted"]:
        print("Extracting MJSynth dataset directly to data/ocr/mjsynth...")
        os.makedirs(output_dir, exist_ok=True)
        
        try:
            if annotation_file.endswith('.tar.gz'):
                with tarfile.open(annotation_file, 'r:gz') as tar:
                    tar.extractall(output_dir)
            elif annotation_file.endswith('.zip'):
                with zipfile.ZipFile(annotation_file, 'r') as zip_ref:
                    zip_ref.extractall(output_dir)
            
            state["mjsynth"]["extracted"] = True
            save_state(state)
            print("✓ MJSynth extracted successfully")
        except Exception as e:
            print(f"ERROR extracting MJSynth: {e}")
            return False
    
    # Convert to CSV format (images are already in the right place)
    if not state["mjsynth"]["converted"]:
        print("Converting MJSynth to CSV format...")
        
        # Find annotation files (typically annotation.txt or similar)
        # Use rglob to search recursively, but filter to only annotation files
        annotation_files = []
        annotation_files.extend(list(Path(output_dir).rglob("annotation*.txt")))
        annotation_files.extend(list(Path(output_dir).rglob("annotation.txt")))
        
        # Filter out non-annotation files (lexicon.txt, imlist.txt, etc.)
        annotation_files = [f for f in annotation_files 
                          if f.name.lower().startswith('annotation') 
                          and f.name.lower() not in ['lexicon.txt', 'imlist.txt']]
        
        if not annotation_files:
            print("⚠️  No annotation files found in", output_dir)
            print("   Looking for files matching: annotation*.txt")
            return False
        
        samples = []
        for ann_file in annotation_files[:1]:  # Use first annotation file found
            print(f"Processing annotation file: {ann_file}")
            # Annotation file directory - paths in annotation are relative to this
            ann_file_dir = ann_file.parent
            
            with open(ann_file, 'r', encoding='utf-8') as f:
                for line in tqdm(f, desc="Reading annotations"):
                    line = line.strip()
                    if not line:
                        continue
                    
                    # MJSynth format: ./path/to/image.jpg number
                    # The text is actually in the filename, e.g., "182_slinking_71711.jpg" -> "slinking"
                    parts = line.split()
                    if len(parts) >= 2:
                        img_path = parts[0].lstrip('./')  # Remove leading ./
                        
                        # Extract text from filename
                        # Format: number_text_number.jpg -> extract "text"
                        filename = os.path.basename(img_path)
                        if '_' in filename:
                            # Split by underscore and take middle parts (skip first and last which are numbers)
                            name_parts = filename.replace('.jpg', '').split('_')
                            if len(name_parts) >= 3:
                                # Join all parts except first and last (which are numbers)
                                text = '_'.join(name_parts[1:-1])
                            else:
                                # Fallback: use the number as text (not ideal but better than nothing)
                                text = parts[1]
                        else:
                            # Fallback: use the number from annotation
                            text = parts[1]
                        
                        # Find actual image file - paths are relative to annotation file directory
                        # Try multiple path resolution strategies (most common first)
                        img_full_path = None
                        
                        # Strategy 1: Direct path from annotation file directory (most common case)
                        candidate = ann_file_dir / img_path
                        if candidate.exists():
                            img_full_path = candidate
                        else:
                            # Strategy 2: Try with ./ prefix (in case annotation has explicit ./)
                            candidate = ann_file_dir / f"./{img_path}"
                            if candidate.exists():
                                img_full_path = candidate
                            else:
                                # Strategy 3: Try relative to output_dir root (fallback)
                                candidate = Path(output_dir) / img_path
                                if candidate.exists():
                                    img_full_path = candidate
                        
                        if img_full_path and img_full_path.exists():
                            # Calculate relative path from output_dir for CSV
                            # OCRDataset expects: image_root (e.g., "data/ocr/") + img_path from CSV (e.g., "mjsynth/path/to/img.jpg")
                            # So we store paths as "mjsynth/{rel_path}" where rel_path is relative to output_dir
                            try:
                                rel_path = os.path.relpath(img_full_path, output_dir)
                            except ValueError:
                                # If paths are on different drives (Windows), use absolute path
                                # But this shouldn't happen in normal usage
                                rel_path = str(img_full_path)
                                print(f"Warning: Using absolute path for {img_path} (different drive)")
                            
                            # Normalize path separators to forward slashes (works on all platforms)
                            rel_path = rel_path.replace('\\', '/')
                            
                            # Ensure path doesn't start with / (should be relative)
                            if rel_path.startswith('/'):
                                rel_path = rel_path[1:]
                            
                            # Store in CSV format: "mjsynth/{rel_path}"
                            # When loaded: os.path.join("data/ocr", "mjsynth/{rel_path}") = "data/ocr/mjsynth/{rel_path}"
                            samples.append({
                                "image": f"mjsynth/{rel_path}",
                                "text": text
                            })
                            
                            if len(samples) >= max_samples:
                                break
                        else:
                            # Log missing files occasionally (not every one to avoid spam)
                            if len(samples) % 10000 == 0 and len(samples) > 0:
                                print(f"Warning: Could not find image: {img_path} (relative to {ann_file_dir})")
                            # Skip this sample - don't add to CSV
        
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

def combine_ocr_csvs():
    """Combine all OCR CSV files into one production file"""
    print("\n" + "="*60)
    print("Combining OCR Datasets")
    print("="*60)
    
    output_file = "data/ocr/production_ocr.csv"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    input_files = [
        "data/ocr/mjsynth_ocr.csv",
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
                       choices=["all", "mjsynth"],
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
        }
        save_state(state)
    
    success = True
    
    # MJSynth
    if args.dataset in ["all", "mjsynth"]:
        success = download_mjsynth(state, args.max_samples) and success
    
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

