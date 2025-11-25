"""
Download and prepare production-grade text datasets for μOmni training
Target: Under 30GB, millions of samples
Includes: English Learning

Supports:
- English Learning: Books
"""

import os
import json
import argparse
import requests
from pathlib import Path
from tqdm import tqdm
import bz2
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# State file to track progress
STATE_FILE = "data/.text_download_state.json"

def load_state():
    """Load download/conversion state"""
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, 'r') as f:
            state = json.load(f)
            return state
    return {
        "books": {"downloaded": False, "converted": False, "samples": 0}
    }

def save_state(state):
    """Save download/conversion state"""
    os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)
    with open(STATE_FILE, 'w') as f:
        json.dump(state, f, indent=2)

def print_progress_with_remaining(current, max_count, label="samples", report_interval=100):
    """Print progress with remaining count and percentage"""
    if current % report_interval == 0 or current >= max_count:
        remaining = max_count - current
        percent = (current / max_count * 100) if max_count > 0 else 0
        print(f"Progress: {current:,} {label} ({percent:.1f}%) - Remaining: ~{remaining:,} {label}")

def save_checkpoint(dataset_name, checkpoint_data):
    """Save fine-grained checkpoint for resuming"""
    checkpoint_file = f"data/.checkpoint_{dataset_name}.json"
    os.makedirs(os.path.dirname(checkpoint_file), exist_ok=True)
    with open(checkpoint_file, 'w') as f:
        json.dump(checkpoint_data, f, indent=2)

def load_checkpoint(dataset_name):
    """Load fine-grained checkpoint for resuming"""
    checkpoint_file = f"data/.checkpoint_{dataset_name}.json"
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'r') as f:
            return json.load(f)
    return None

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
        
        # Handle 416 Range Not Satisfiable (file already complete)
        if response.status_code == 416:
            return True
        
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        if resume and os.path.exists(output_path) and resume_header:
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

def _download_single_book(book_id, base_url, processed_books_set):
    """Download and process a single book. Returns (book_id, book_text, passage_count, success)"""
    # Try different file formats
    for suffix in ['-0.txt', '-8.txt', '.txt']:
        url = f"{base_url}/{book_id}/{book_id}{suffix}"
        try:
            response = requests.get(url, timeout=30, stream=True)
            if response.status_code == 200:
                # Read and process book
                content = response.text
                # Remove Project Gutenberg headers/footers
                lines = content.split('\n')
                start_idx_text = 0
                end_idx_text = len(lines)
                
                # Find start (skip header)
                for i, line in enumerate(lines):
                    if 'START OF THIS PROJECT GUTENBERG' in line.upper() or 'START OF THE PROJECT GUTENBERG' in line.upper():
                        start_idx_text = i + 1
                        break
                
                # Find end (skip footer)
                for i in range(len(lines)-1, -1, -1):
                    if 'END OF THIS PROJECT GUTENBERG' in lines[i].upper() or 'END OF THE PROJECT GUTENBERG' in lines[i].upper():
                        end_idx_text = i
                        break
                
                book_text = '\n'.join(lines[start_idx_text:end_idx_text])
                
                # Count passages by double newlines (filter out empty passages for accurate count)
                # Split by \n\n and count non-empty passages
                passages = [p.strip() for p in book_text.split('\n\n') if p.strip()]
                passage_count = len(passages)
                
                if len(book_text) > 0:
                    return (book_id, book_text, passage_count, True)
                else:
                    return (book_id, '', 0, False)
        except Exception as e:
            continue
    
    return (book_id, '', 0, False)

def download_books(state, max_samples=50000):
    """Download books corpus from Project Gutenberg with parallel downloads"""
    print("\n" + "="*60)
    print("Downloading Books Corpus")
    print("="*60)
    
    if state["books"]["downloaded"] and state["books"]["samples"] >= max_samples:
        print(f"Books already downloaded ({state['books']['samples']:,} samples), skipping...")
        return True
    elif state["books"]["samples"] > 0 and state["books"]["samples"] < max_samples:
        print(f"Resuming books download: {state['books']['samples']:,} / {max_samples:,} passages")
    
    print("Downloading books from Project Gutenberg...")
    print("NOTE: Project Gutenberg provides free books in plain text format.")
    print("Downloading 15 books in parallel until sample limit is reached...")
    
    output_file = "data/text/books.txt"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Load checkpoint for resuming
    checkpoint = load_checkpoint("books")
    if checkpoint:
        print(f"Resuming from checkpoint: {checkpoint.get('count', 0)} passages")
        count = checkpoint.get('count', 0)
        processed_books = set(checkpoint.get('processed_books', []))
        last_tried_id = checkpoint.get('last_tried_id', 0)
        mode = 'a'  # Append mode
        resume = True
    else:
        count = 0
        processed_books = set()
        last_tried_id = 0
        mode = 'w'  # Write mode
        resume = False
    
    base_url = "https://www.gutenberg.org/files"
    
    # Generate random book IDs (Project Gutenberg has books from ID 1 to ~70,000+)
    import random
    
    # Start from a random point if resuming, otherwise start from 1
    if resume:
        current_id = last_tried_id + 1
    else:
        current_id = 1
    
    # Create progress bar that tracks passages
    pbar = tqdm(total=max_samples, desc="Downloading passages", unit="passage", initial=count)
    
    # Thread-safe locks for file writing and state updates
    file_lock = threading.Lock()
    state_lock = threading.Lock()
    count_lock = threading.Lock()  # Lock for count variable
    
    # Use ThreadPoolExecutor for parallel downloads (15 workers)
    max_workers = 15
    
    with open(output_file, mode, encoding='utf-8') as f:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Keep track of submitted tasks
            futures = {}
            consecutive_failures = 0
            max_consecutive_failures = 100
            
            while True:
                # Check count with lock
                with count_lock:
                    if count >= max_samples:
                        break
                # Submit new tasks if we have room and haven't reached max_samples
                with count_lock:
                    can_continue = count < max_samples
                while len(futures) < max_workers and can_continue:
                    # Try random book IDs, but also try sequential to find available books
                    if random.random() < 0.3:  # 30% chance to try random ID
                        book_id = random.randint(1, 70000)
                    else:
                        book_id = current_id
                        current_id += 1
                    
                    # Skip if already processed
                    if book_id in processed_books:
                        continue
                    
                    # Submit download task
                    future = executor.submit(_download_single_book, book_id, base_url, processed_books)
                    futures[future] = book_id
                
                # Process completed downloads
                for future in as_completed(futures):
                    book_id = futures.pop(future)
                    try:
                        result_book_id, book_text, passage_count, success = future.result()
                        
                        if success and len(book_text) > 0:
                            # Thread-safe write and count update
                            with file_lock, count_lock:
                                # Check if we've reached max_samples before writing
                                if count >= max_samples:
                                    continue
                                
                                # Write entire book content (thread-safe)
                                f.write(book_text + '\n\n')
                                f.flush()
                                
                                # Update count (thread-safe)
                                prev_count = count
                                count += passage_count
                                
                                # Update progress bar (cap at max_samples)
                                update_amount = min(passage_count, max_samples - prev_count)
                                if update_amount > 0:
                                    pbar.update(update_amount)
                                
                                # Update postfix with current count
                                current_count = count
                            
                            # Update state (outside file lock for better performance)
                            with state_lock:
                                processed_books.add(book_id)
                                pbar.set_postfix({'books': len(processed_books), 'passages': min(current_count, max_samples)})
                            
                            # Save checkpoint every 50 passages (thread-safe)
                            with count_lock:
                                if count % 50 == 0:
                                    with state_lock:
                                        save_checkpoint("books", {
                                            'count': count,
                                            'last_tried_id': book_id,
                                            'processed_books': list(processed_books)
                                        })
                            
                            consecutive_failures = 0
                        else:
                            # Mark as tried to avoid retrying
                            with state_lock:
                                processed_books.add(book_id)
                            consecutive_failures += 1
                            
                    except Exception as e:
                        # Mark as tried on error
                        with state_lock:
                            processed_books.add(book_id)
                        consecutive_failures += 1
                
                # If too many consecutive failures, try more random IDs
                if consecutive_failures >= max_consecutive_failures:
                    print(f"\nWarning: {max_consecutive_failures} consecutive failures. Trying more random IDs...")
                    consecutive_failures = 0
                    current_id = random.randint(1, 70000)
                
                # Check if we should continue
                with count_lock:
                    if count >= max_samples:
                        break
                    can_continue = count < max_samples
    
    pbar.close()
    
    # Get final count (thread-safe)
    with count_lock:
        final_count = count
    
    # Only mark as downloaded if we reached max_samples
    if final_count >= max_samples:
        state["books"]["downloaded"] = True
        state["books"]["converted"] = True
    state["books"]["samples"] = final_count
    save_state(state)
    
    # Clean up checkpoint file on success (only if reached max_samples)
    checkpoint_file = "data/.checkpoint_books.json"
    if os.path.exists(checkpoint_file) and count >= max_samples:
        os.remove(checkpoint_file)
    
    if count >= max_samples:
        print(f"\n✓ Downloaded {count:,} book passages to {output_file}")
    else:
        print(f"\n⚠ Downloaded {count:,} book passages (target: {max_samples:,})")
        print("   Some books may have failed to download or book list may be exhausted.")
        print("   You can resume by running the script again.")
    
    return True

def combine_text_datasets():
    """Combine all downloaded text datasets into one corpus"""
    print("\n" + "="*60)
    print("Combining Text Datasets")
    print("="*60)
    
    output_file = "data/text/production_corpus.txt"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    input_files = [
        # General knowledge & English learning
        "data/text/books.txt"
    ]
    
    total_samples = 0
    
    with open(output_file, 'w', encoding='utf-8') as out_f:
        for input_file in input_files:
            if os.path.exists(input_file):
                print(f"Adding {os.path.basename(input_file)}...")
                
                with open(input_file, 'r', encoding='utf-8', errors='ignore') as in_f:
                    count = 0
                    for line in tqdm(in_f, desc=f"  Processing {os.path.basename(input_file)}", leave=False):
                        if line.strip():
                            out_f.write(line)
                            count += 1
                    total_samples += count
    
    print(f"\n✓ Combined corpus created: {output_file}")
    print(f"  Total samples: {total_samples:,}")


def main():
    parser = argparse.ArgumentParser(description="Download production-grade text datasets for μOmni")
    parser.add_argument("--dataset", 
                       choices=["all", "books"], 
                       default="all",
                       help="Which dataset to download (default: all)")
    parser.add_argument("--skip-download", action="store_true",
                       help="Skip download, only extract/convert existing data")
    parser.add_argument("--skip-extract", action="store_true",
                       help="Skip extraction, only convert")
    parser.add_argument("--skip-convert", action="store_true",
                       help="Skip conversion, only download/extract")
    parser.add_argument("--combine", action="store_true",
                       help="Combine all downloaded datasets into one corpus (outputs to data/text/production_corpus.txt)")
    parser.add_argument("--reset", action="store_true",
                       help="Reset state and re-download everything")
    parser.add_argument("--max-samples", type=int, default=500000,
                       help="Maximum number of samples per dataset (default: 500000, combined total ~12M for all datasets)")
    parser.add_argument("--parallel-datasets", action="store_true",
                       help="Download multiple datasets in parallel (when using 'all' or multiple datasets)")
    
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
    print("μOmni Production Text Dataset Downloader")
    print("="*60)
    print(f"State file: {STATE_FILE}")
    print(f"Dataset: {args.dataset}")
    print("="*60)
    
    success = True
    
    if args.dataset in ["all", "books", "general"]:
        if not args.skip_download:
            success = download_books(state, args.max_samples) and success
    
    # Combine if requested
    if args.combine:
        combine_text_datasets()
    
    print("\n" + "="*60)
    if success:
        print("✓ All operations completed successfully!")
        print("\nOutput files (ready to use, no formatting needed):")
        print("  - Individual datasets: data/text/*.txt")
        if args.combine or args.dataset == "all":
            print("  - Combined corpus: data/text/production_corpus.txt")
        print("\nNext steps:")
        print("1. Datasets are already in final format in data/text/")
        print("2. Update config files to point to:")
        if args.combine or args.dataset == "all":
            print("   data/text/production_corpus.txt")
        else:
            print("   data/text/[dataset_name].txt")
        print("3. Run training: python train_text.py --config configs/thinker_tiny.json")
    else:
        print("✗ Some operations failed. Check errors above.")
        print("You can resume by running the script again (it will skip completed steps)")
        print("Fine-grained checkpoints saved - will resume from exact position")
    print("="*60)

if __name__ == "__main__":
    main()

