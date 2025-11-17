"""
Download and prepare production-grade text datasets for μOmni training
Target: Under 30GB, millions of samples
Includes: Conversations, Instruction Following, Tool Calls, Scientific Content (Physics, Chemistry, Math, Biology), English Learning

Supports:
- Conversations: Alpaca
- Scientific: ArXiv (physics, chemistry, math, biology), PubMed, Math datasets
- English Learning: Books, Wikipedia
- Science Education: ScienceQA
"""

import os
import json
import argparse
import subprocess
import sys
import requests
from pathlib import Path
from tqdm import tqdm
import bz2
import gzip
import shutil

# State file to track progress
STATE_FILE = "data/.text_download_state.json"

def load_state():
    """Load download/conversion state"""
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, 'r') as f:
            return json.load(f)
    return {
        # General knowledge & English learning
        "wikipedia": {"downloaded": False, "extracted": False, "converted": False, "samples": 0, "last_processed_file": "", "last_offset": 0},
        "books": {"downloaded": False, "extracted": False, "converted": False, "samples": 0, "last_book_id": 0, "last_paragraph": 0},
        
        # Conversations & Instruction Following
        "alpaca": {"downloaded": False, "converted": False, "samples": 0, "last_line": 0},
        
        # Scientific Content
        "arxiv_physics": {"downloaded": False, "converted": False, "samples": 0, "last_start": 0, "last_batch": 0},
        "arxiv_chemistry": {"downloaded": False, "converted": False, "samples": 0, "last_start": 0, "last_batch": 0},
        "arxiv_math": {"downloaded": False, "converted": False, "samples": 0, "last_start": 0, "last_batch": 0},
        "arxiv_biology": {"downloaded": False, "converted": False, "samples": 0, "last_start": 0, "last_batch": 0},
        "pubmed": {"downloaded": False, "converted": False, "samples": 0, "last_batch": 0, "last_id_index": 0},
        "math_datasets": {"downloaded": False, "converted": False, "samples": 0, "last_dataset": "", "last_item": 0},
        "scienceqa": {"downloaded": False, "converted": False, "samples": 0, "last_item": 0}
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

def download_wikipedia(state):
    """Download Wikipedia English dump (latest, ~24GB compressed)"""
    print("\n" + "="*60)
    print("Downloading Wikipedia English Dump")
    print("="*60)
    
    if state["wikipedia"]["downloaded"]:
        print("Wikipedia already downloaded, skipping...")
        return True
    
    # Wikipedia dumps are large, we'll download the latest pages-articles
    # This is ~24GB compressed, expands to ~100GB+ but we'll extract only text
    url = "https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2"
    output_dir = "data/text_downloads"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "enwiki-latest-pages-articles.xml.bz2")
    
    print(f"Downloading Wikipedia dump (~24GB compressed)...")
    print(f"URL: {url}")
    print(f"Output: {output_file}")
    print("This may take several hours depending on your connection...")
    
    if download_file(url, output_file, resume=True):
        state["wikipedia"]["downloaded"] = True
        save_state(state)
        print("✓ Wikipedia downloaded successfully")
        return True
    else:
        print("✗ Failed to download Wikipedia")
        return False

def extract_wikipedia_text_custom(input_file, output_dir, min_text_length=100, checkpoint=None):
    """Custom Wikipedia XML parser - extracts text without external dependencies with fine-grained resuming"""
    import xml.etree.ElementTree as ET
    import re
    
    def clean_wiki_text(text):
        """Simple wiki text cleaning"""
        if not text:
            return ""
        
        # Remove wiki markup patterns
        text = re.sub(r'\[\[([^\]]+)\]\]', r'\1', text)  # [[link]] -> link
        text = re.sub(r'\[([^\]]+)\]', r'\1', text)  # [url] -> url
        text = re.sub(r'{{[^}]+}}', '', text)  # Remove templates
        text = re.sub(r'<ref[^>]*>.*?</ref>', '', text, flags=re.DOTALL)  # Remove refs
        text = re.sub(r'<[^>]+>', '', text)  # Remove HTML tags
        text = re.sub(r"''+", '', text)  # Remove bold/italic markers
        text = re.sub(r'={2,}', '', text)  # Remove section headers
        text = re.sub(r'\n{3,}', '\n\n', text)  # Normalize newlines
        text = text.strip()
        
        return text
    
    def process_page(elem):
        """Extract text from a single page element"""
        try:
            title_elem = elem.find('{http://www.mediawiki.org/xml/export-0.10/}title')
            ns_elem = elem.find('{http://www.mediawiki.org/xml/export-0.10/}ns')
            text_elem = elem.find('{http://www.mediawiki.org/xml/export-0.10/}revision/{http://www.mediawiki.org/xml/export-0.10/}text')
            
            if title_elem is None or text_elem is None:
                return None
            
            title = title_elem.text or ""
            ns = ns_elem.text if ns_elem is not None else "0"
            text = text_elem.text or ""
            
            # Skip non-article namespaces (0 = main namespace)
            if ns != "0":
                return None
            
            # Skip disambiguation pages
            if "disambiguation" in title.lower() or "(disambiguation)" in title.lower():
                return None
            
            # Skip redirects
            if text.strip().startswith("#REDIRECT") or text.strip().startswith("#redirect"):
                return None
            
            # Clean the text
            cleaned = clean_wiki_text(text)
            
            if len(cleaned) < min_text_length:
                return None
            
            return (title, cleaned)
        except Exception as e:
            return None
    
    # Load checkpoint if resuming
    processed_titles = set()
    article_count = 0
    page_count = 0
    files_per_dir = 100
    file_count = 0
    subdir = None
    
    if checkpoint:
        processed_titles = set(checkpoint.get('processed_titles', []))
        article_count = checkpoint.get('article_count', 0)
        page_count = checkpoint.get('page_count', 0)
        file_count = checkpoint.get('file_count', 0)
        print(f"Resuming: {page_count} pages processed, {article_count} articles extracted, {len(processed_titles)} unique titles")
    
    print("Parsing Wikipedia XML dump (this may take 30-60 minutes)...")
    print("Using custom XML parser (no external dependencies required)...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Parse XML incrementally
    print("Reading and parsing XML file...")
    try:
        # Handle bz2 compression - need to open in text mode for XML parsing
        if input_file.endswith('.bz2'):
            file_handle = bz2.open(input_file, 'rt', encoding='utf-8', errors='ignore')
        else:
            file_handle = open(input_file, 'rt', encoding='utf-8', errors='ignore')
        
        # Detect namespace from root element
        namespace = None
        for event, elem in ET.iterparse(file_handle, events=('start',)):
            if elem.tag.startswith('{'):
                # Extract namespace from first element
                namespace = elem.tag[:elem.tag.index('}') + 1]
            break
        
        # Close and reopen to start from beginning
        file_handle.close()
        if input_file.endswith('.bz2'):
            file_handle = bz2.open(input_file, 'rt', encoding='utf-8', errors='ignore')
        else:
            file_handle = open(input_file, 'rt', encoding='utf-8', errors='ignore')
        
        # Update process_page to use detected namespace
        def process_page_with_ns(elem):
            """Extract text from a single page element using detected namespace"""
            try:
                ns_prefix = namespace if namespace else ''
                title_elem = elem.find(f'{ns_prefix}title')
                ns_elem = elem.find(f'{ns_prefix}ns')
                revision_elem = elem.find(f'{ns_prefix}revision')
                if revision_elem is not None:
                    text_elem = revision_elem.find(f'{ns_prefix}text')
                else:
                    text_elem = None
                
                if title_elem is None or text_elem is None:
                    return None
                
                title = title_elem.text or ""
                ns = ns_elem.text if ns_elem is not None else "0"
                text = text_elem.text or ""
                
                # Skip non-article namespaces (0 = main namespace)
                if ns != "0":
                    return None
                
                # Skip disambiguation pages
                if "disambiguation" in title.lower() or "(disambiguation)" in title.lower():
                    return None
                
                # Skip redirects
                if text.strip().startswith("#REDIRECT") or text.strip().startswith("#redirect"):
                    return None
                
                # Clean the text
                cleaned = clean_wiki_text(text)
                
                if len(cleaned) < min_text_length:
                    return None
                
                return (title, cleaned)
            except Exception as e:
                return None
        
        # Use iterparse for memory-efficient parsing
        context = ET.iterparse(file_handle, events=('start', 'end'))
        context = iter(context)
        event, root = next(context)
        
        # Open output file for appending
        if file_count > 0 or article_count > 0:
            subdir = os.path.join(output_dir, f"AA")
            os.makedirs(subdir, exist_ok=True)
            output_file = os.path.join(subdir, f"wiki_{file_count:02d}")
            out_f = open(output_file, 'a', encoding='utf-8')
        else:
            out_f = None
        
        max_articles = 1000000  # Limit to prevent memory issues
        
        # Track progress for reporting
        last_report_pages = page_count
        last_report_articles = article_count
        checkpoint_interval = 50  # Save checkpoint every 50 pages
        report_interval = 1000  # Report progress every 1000 pages
        
        # Track if we're still catching up to checkpoint
        resume_page_count = page_count
        catching_up = (resume_page_count > 0)
        current_page_num = 0  # Track current page number in XML
        
        if catching_up:
            print(f"Fast-forwarding through {resume_page_count:,} already-processed pages...")
            print("(This may take a few minutes - skipping page processing to speed up)")
        
        print(f"Starting extraction from page {page_count}...")
        
        for event, elem in context:
            # Check if this is a page element (handle both with and without namespace)
            tag_name = elem.tag
            if '}' in tag_name:
                tag_name = tag_name.split('}')[1]  # Remove namespace prefix
            
            if event == 'end' and tag_name == 'page':
                current_page_num += 1
                
                # Fast-forward: skip processing until we reach checkpoint
                if catching_up and current_page_num <= resume_page_count:
                    page_count = current_page_num  # Update for checkpoint saving
                    # Clear element quickly without processing
                    elem.clear()
                    root.clear()
                    # Report progress every 10000 pages during fast-forward
                    if current_page_num % 10000 == 0:
                        print(f"Fast-forward: {current_page_num:,} / {resume_page_count:,} pages...")
                    continue
                
                # We've caught up - start processing normally
                if catching_up and current_page_num > resume_page_count:
                    catching_up = False
                    print(f"Caught up! Starting processing from page {current_page_num:,}...")
                
                page_count = current_page_num
                article = process_page_with_ns(elem)
                
                if article:
                    title, cleaned = article
                    # Skip if already processed
                    if title not in processed_titles:
                        # Open new file if needed
                        if out_f is None or (article_count % files_per_dir == 0 and article_count > 0):
                            if out_f:
                                out_f.flush()  # Ensure data is written
                                out_f.close()
                            subdir = os.path.join(output_dir, f"AA")
                            os.makedirs(subdir, exist_ok=True)
                            file_count = article_count // files_per_dir
                            output_file = os.path.join(subdir, f"wiki_{file_count:02d}")
                            out_f = open(output_file, 'a', encoding='utf-8')
                        
                        out_f.write(f"{title}\n\n{cleaned}\n\n")
                        out_f.flush()  # Flush after each write for fine-grained resumption
                        processed_titles.add(title)
                        article_count += 1
                        
                        # Report progress periodically
                        if page_count - last_report_pages >= report_interval:
                            remaining_articles = max_articles - article_count
                            article_percent = (article_count / max_articles * 100) if max_articles > 0 else 0
                            
                            # Estimate remaining pages based on current extraction rate
                            if article_count > 0 and page_count > 0:
                                articles_per_page = article_count / page_count
                                if articles_per_page > 0:
                                    estimated_remaining_pages = int(remaining_articles / articles_per_page)
                                    print(f"Progress: {page_count:,} pages processed, {article_count:,} articles extracted ({article_percent:.1f}%)")
                                    print(f"  Remaining: ~{remaining_articles:,} articles (~{estimated_remaining_pages:,} pages estimated)")
                                else:
                                    print(f"Progress: {page_count:,} pages processed, {article_count:,} articles extracted ({article_percent:.1f}%)")
                                    print(f"  Remaining: ~{remaining_articles:,} articles")
                            else:
                                print(f"Progress: {page_count:,} pages processed, {article_count:,} articles extracted ({article_percent:.1f}%)")
                                print(f"  Remaining: ~{remaining_articles:,} articles")
                            
                            last_report_pages = page_count
                            last_report_articles = article_count
                        
                        # Save checkpoint periodically (every N pages)
                        if page_count % checkpoint_interval == 0:
                            save_checkpoint("wikipedia_extract", {
                                'processed_titles': list(processed_titles),  # Keep all for deduplication
                                'article_count': article_count,
                                'page_count': page_count,
                                'file_count': file_count
                            })
                        
                        if article_count >= max_articles:
                            print(f"Reached maximum article limit ({max_articles})")
                            break
                
                # Clear element to free memory after processing each page
                elem.clear()
                root.clear()
                
                # Save final checkpoint after each page (for fine-grained resumption)
                # But only every checkpoint_interval pages to avoid too frequent I/O
                if page_count % checkpoint_interval == 0:
                    save_checkpoint("wikipedia_extract", {
                        'processed_titles': list(processed_titles),
                        'article_count': article_count,
                        'page_count': page_count,
                        'file_count': file_count
                    })
        
        if out_f:
            out_f.flush()
            out_f.close()
        file_handle.close()
        
        # Save final checkpoint
        save_checkpoint("wikipedia_extract", {
            'processed_titles': list(processed_titles),
            'article_count': article_count,
            'page_count': page_count,
            'file_count': file_count
        })
        
    except ET.ParseError as e:
        print(f"XML parsing error: {e}")
        # Save checkpoint on error for resumption
        if out_f:
            out_f.flush()
            out_f.close()
        save_checkpoint("wikipedia_extract", {
            'processed_titles': list(processed_titles),
            'article_count': article_count,
            'page_count': page_count,
            'file_count': file_count
        })
        print(f"Checkpoint saved. Resumed at page {page_count}, {article_count} articles extracted.")
        return False
    except Exception as e:
        print(f"Error reading file: {e}")
        import traceback
        traceback.print_exc()
        # Save checkpoint on error for resumption
        if out_f:
            out_f.flush()
            out_f.close()
        save_checkpoint("wikipedia_extract", {
            'processed_titles': list(processed_titles),
            'article_count': article_count,
            'page_count': page_count,
            'file_count': file_count
        })
        print(f"Checkpoint saved. Resumed at page {page_count}, {article_count} articles extracted.")
        return False
    
    # Clean up checkpoint on success
    checkpoint_file = "data/.checkpoint_wikipedia_extract.json"
    if os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)
    
    print(f"✓ Extracted {article_count:,} articles from {page_count:,} pages")
    return True

def extract_wikipedia_text(state):
    """Extract text from Wikipedia XML dump using custom parser with fine-grained resuming"""
    print("\n" + "="*60)
    print("Extracting Text from Wikipedia Dump")
    print("="*60)
    
    if state["wikipedia"]["extracted"]:
        print("Wikipedia already extracted, skipping...")
        return True
    
    input_file = "data/text_downloads/enwiki-latest-pages-articles.xml.bz2"
    output_dir = "data/text_downloads/wikipedia_extracted"
    
    if not os.path.exists(input_file):
        print(f"ERROR: Wikipedia dump not found at {input_file}")
        return False
    
    # Load checkpoint for extraction
    checkpoint = load_checkpoint("wikipedia_extract")
    if checkpoint:
        print(f"Resuming extraction: {checkpoint.get('article_count', 0)} articles already processed")
    
    # Handle bz2 compression - parse directly from compressed file
    import bz2
    if input_file.endswith('.bz2'):
        print("Parsing directly from bz2 compressed file...")
        # We'll handle decompression in the parser
    
    try:
        success = extract_wikipedia_text_custom(input_file, output_dir, min_text_length=100, checkpoint=checkpoint)
        
        if success:
            state["wikipedia"]["extracted"] = True
            save_state(state)
            print("✓ Wikipedia text extracted successfully")
            return True
        else:
            print("ERROR: Extraction failed (checkpoint saved, can resume)")
            return False
    except Exception as e:
        print(f"ERROR extracting Wikipedia: {e}")
        import traceback
        traceback.print_exc()
        return False

def convert_wikipedia_to_text(state, max_samples=1000000):
    """Convert extracted Wikipedia files to single text file with fine-grained resuming"""
    print("\n" + "="*60)
    print("Converting Wikipedia to Text Format")
    print("="*60)
    
    extracted_dir = "data/text_downloads/wikipedia_extracted"
    output_file = "data/text/wikipedia.txt"
    
    if not os.path.exists(extracted_dir):
        print(f"ERROR: Extracted Wikipedia not found at {extracted_dir}")
        return False
    
    # Check if all files are already processed (files should match extraction sample size)
    all_files_check = []
    for root, dirs, files in os.walk(extracted_dir):
        for file in files:
            file_path = os.path.join(root, file)
            rel_path = os.path.relpath(file_path, extracted_dir)
            all_files_check.append(rel_path)
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Load checkpoint if exists
    checkpoint = load_checkpoint("wikipedia")
    if checkpoint:
        processed_files_check = set(checkpoint.get('processed_files', []))
        if len(processed_files_check) >= len(all_files_check):
            print(f"Wikipedia already converted ({len(processed_files_check):,} files, {state['wikipedia']['samples']:,} articles), skipping...")
            return True
        
        print(f"Resuming from checkpoint: {checkpoint.get('last_file', 'start')}")
        processed_files = set(checkpoint.get('processed_files', []))
        count = checkpoint.get('count', 0)
        last_file = checkpoint.get('last_file', '')
        mode = 'a'  # Append mode
    else:
        processed_files = set()
        count = 0
        last_file = ''
        mode = 'w'  # Write mode
    resume_from_last = (last_file != '')
    
    print("Combining Wikipedia articles into single text file...")
    print("This may take 10-20 minutes...")
    if resume_from_last:
        print(f"Resuming from last processed file: {last_file}")
    
    with open(output_file, mode, encoding='utf-8') as out_f:
        # WikiExtractor creates subdirectories with files
        all_files = []
        for root, dirs, files in os.walk(extracted_dir):
            for file in files:
                file_path = os.path.join(root, file)
                rel_path = os.path.relpath(file_path, extracted_dir)
                all_files.append((rel_path, file_path))
        
        # Sort for consistent processing
        all_files.sort()
        
        # Skip already processed files
        if resume_from_last:
            skip_until = last_file
            skipping = True
        else:
            skipping = False
            skip_until = None
        
        # Create progress bar that tracks files (since we process all files to match extraction)
        # The extraction created files_per_dir files for max_samples articles
        # So we should process all files to get all extracted articles
        total_files = len(all_files)
        pbar = tqdm(total=total_files, desc="Processing files", unit="file", initial=len(processed_files))
        
        for rel_path, file_path in all_files:
            # Resume logic: skip until we reach last processed file
            if skipping:
                if rel_path == skip_until:
                    skipping = False
                else:
                    continue
            
            if rel_path in processed_files:
                continue
            
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as in_f:
                    content = in_f.read().strip()
                    if len(content) > 100:  # Skip very short files
                        # Count articles in file
                        # Each article is formatted as: title\n\ncontent\n\n
                        # Split by double newlines to get alternating titles and contents
                        parts = [p.strip() for p in content.split('\n\n') if p.strip()]
                        # Each article has 2 parts (title + content), so count = parts / 2
                        # Handle edge case: if only 1 part, it's still 1 article
                        article_count_in_file = max(1, len(parts) // 2) if len(parts) > 1 else 1
                        
                        out_f.write(content + '\n\n')
                        count += article_count_in_file
                        processed_files.add(rel_path)
                        
                        # Update progress bar (track files, show articles in postfix)
                        pbar.update(1)
                        pbar.set_postfix({'articles': count})
                        
                        # Save checkpoint every 100 files
                        if len(processed_files) % 100 == 0:
                            save_checkpoint("wikipedia", {
                                'processed_files': list(processed_files),
                                'count': count,
                                'last_file': rel_path
                            })
                            save_state(state)
            except Exception as e:
                continue
        
        pbar.close()
    
    state["wikipedia"]["converted"] = True
    state["wikipedia"]["samples"] = count
    save_state(state)
    
    # Clean up checkpoint file (state file only, actual data is never deleted)
    checkpoint_file = "data/.checkpoint_wikipedia.json"
    if os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)
    
    print(f"\n✓ Converted {count:,} Wikipedia articles to {output_file}")
    return True


def download_json_dataset_from_url(state, dataset_name, url, output_file, max_samples=500000, extract_func=None):
    """Download JSON dataset from direct URL with fine-grained resuming"""
    print(f"\n{'='*60}")
    print(f"Downloading {dataset_name}")
    print("="*60)
    
    if state[dataset_name]["downloaded"] and state[dataset_name]["samples"] >= max_samples:
        print(f"{dataset_name} already downloaded ({state[dataset_name]['samples']:,} samples), skipping...")
        return True
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Download the file
    download_dir = "data/text_downloads"
    os.makedirs(download_dir, exist_ok=True)
    temp_file = os.path.join(download_dir, f"{dataset_name}.json")
    
    print(f"Downloading from: {url}")
    if not download_file(url, temp_file, resume=True):
        print(f"ERROR: Failed to download {dataset_name}")
        return False
    
    # Load checkpoint for processing
    checkpoint = load_checkpoint(dataset_name)
    if checkpoint:
        print(f"Resuming processing from line {checkpoint.get('last_line', 0)}")
        last_line = checkpoint.get('last_line', 0)
        count = checkpoint.get('count', 0)
        mode = 'a'  # Append mode
        catching_up = (last_line > 0)
    else:
        last_line = 0
        count = 0
        mode = 'w'  # Write mode
        catching_up = False
    
    # Process JSON file
    print("Processing JSON file...")
    
    if catching_up:
        print(f"Fast-forwarding through {last_line:,} already-processed lines...")
        print("(Skipping processing to speed up resumption)")
    
    with open(output_file, mode, encoding='utf-8') as f:
        # Try to read as JSONL (one JSON per line) or regular JSON array
        try:
            with open(temp_file, 'r', encoding='utf-8') as in_f:
                # Try JSONL first
                line_num = 0
                for line in tqdm(in_f, desc=f"Processing {dataset_name}"):
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
                    
                    try:
                        item = json.loads(line.strip())
                        if extract_func:
                            text = extract_func(item)
                        else:
                            text = extract_conversation_text(item)
                        
                        if text and len(text) > 50:
                            f.write(text + '\n\n')
                            count += 1
                            
                            # Print progress with remaining
                            print_progress_with_remaining(count, max_samples, "samples", report_interval=100)
                            
                            # Save checkpoint every 100 items
                            if count % 100 == 0:
                                save_checkpoint(dataset_name, {
                                    'last_line': line_num,
                                    'count': count
                                })
                                save_state(state)
                            
                            if count >= max_samples:
                                break
                    except json.JSONDecodeError:
                        # Not JSONL, try as array
                        break
                    except Exception as e:
                        continue
                
                # If JSONL didn't work, try as JSON array
                if count == 0 and last_line == 0:
                    in_f.seek(0)
                    data = json.load(in_f)
                    if isinstance(data, list):
                        catching_up_array = (last_line > 0)
                        if catching_up_array:
                            print(f"Fast-forwarding through {last_line:,} already-processed items...")
                        
                        for idx, item in enumerate(tqdm(data, desc=f"Processing {dataset_name}")):
                            # Fast-forward: skip already processed items
                            if catching_up_array and idx < last_line:
                                if idx % 10000 == 0:
                                    print(f"Fast-forward: {idx:,} / {last_line:,} items...")
                                continue
                            
                            # We've caught up - start processing normally
                            if catching_up_array and idx >= last_line:
                                catching_up_array = False
                                print(f"Caught up! Starting processing from item {idx:,}...")
                            
                            try:
                                if extract_func:
                                    text = extract_func(item)
                                else:
                                    text = extract_conversation_text(item)
                                
                                if text and len(text) > 50:
                                    f.write(text + '\n\n')
                                    count += 1
                                    
                                    # Print progress with remaining
                                    print_progress_with_remaining(count, max_samples, "samples", report_interval=100)
                                    
                                    # Save checkpoint every 100 items
                                    if count % 100 == 0:
                                        save_checkpoint(dataset_name, {
                                            'last_line': idx + 1,
                                            'count': count
                                        })
                                        save_state(state)
                                    
                                    if count >= max_samples:
                                        break
                            except Exception as e:
                                continue
        except Exception as e:
            print(f"ERROR processing {dataset_name}: {e}")
            # Save checkpoint on error
            save_checkpoint(dataset_name, {
                'last_line': last_line,
                'count': count
            })
            save_state(state)
            return False
    
    state[dataset_name]["downloaded"] = True
    state[dataset_name]["converted"] = True
    state[dataset_name]["samples"] = count
    save_state(state)
    
    # Clean up checkpoint file (state file only, actual data is never deleted)
    checkpoint_file = f"data/.checkpoint_{dataset_name}.json"
    if os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)
    
    print(f"\n✓ Downloaded {count:,} samples to {output_file}")
    return True

def extract_conversation_text(item):
    """Extract text from various conversation formats"""
    # Try different conversation formats
    if 'conversations' in item:
        # ShareGPT/OpenAssistant format
        parts = []
        for turn in item['conversations']:
            if isinstance(turn, dict):
                role = turn.get('from', turn.get('role', ''))
                content = turn.get('value', turn.get('content', ''))
                if content:
                    parts.append(f"{role}: {content}")
        return '\n'.join(parts) if parts else None
    
    if 'messages' in item:
        # Messages format
        parts = []
        for msg in item['messages']:
            role = msg.get('role', '')
            content = msg.get('content', '')
            if content:
                parts.append(f"{role}: {content}")
        return '\n'.join(parts) if parts else None
    
    if 'instruction' in item or 'input' in item or 'output' in item:
        # Alpaca format
        instruction = item.get('instruction', '')
        input_text = item.get('input', '')
        output = item.get('output', '')
        parts = []
        if instruction:
            parts.append(f"Instruction: {instruction}")
        if input_text:
            parts.append(f"Input: {input_text}")
        if output:
            parts.append(f"Output: {output}")
        return '\n'.join(parts) if parts else None
    
    if 'log' in item:
        # DialogStudio format
        parts = []
        for turn in item['log']:
            if isinstance(turn, dict):
                user = turn.get('user utterance', turn.get('user_utterance', ''))
                system = turn.get('system response', turn.get('system_response', ''))
                if user:
                    parts.append(f"User: {user}")
                if system:
                    parts.append(f"Assistant: {system}")
        return '\n'.join(parts) if parts else None
    
    # Fallback: try to extract any text fields
    text_parts = []
    for key in ['text', 'content', 'prompt', 'response', 'question', 'answer']:
        if key in item and item[key]:
            text_parts.append(str(item[key]))
    
    return '\n'.join(text_parts) if text_parts else None

def download_alpaca(state, max_samples=500000):
    """Download Alpaca instruction following dataset from GitHub and extended variants"""
    print("\n" + "="*60)
    print("Downloading Alpaca")
    print("="*60)
    
    if state["alpaca"]["downloaded"] and state["alpaca"]["samples"] >= max_samples:
        print(f"Alpaca already downloaded ({state['alpaca']['samples']:,} samples), skipping...")
        return True
    elif state["alpaca"]["samples"] > 0 and state["alpaca"]["samples"] < max_samples:
        print(f"Resuming Alpaca download: {state['alpaca']['samples']:,} / {max_samples:,} samples")
    
    output_file = "data/text/alpaca.txt"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Download multiple Alpaca variants for more data
    alpaca_urls = [
        # Original Alpaca (52k samples)
        ("https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/main/alpaca_data.json", "original"),
        # Try Alpaca-GPT4 if available
        ("https://raw.githubusercontent.com/Instruction-Tuning-with-GPT-4/GPT-4-LLM/main/data/alpaca_gpt4_data.json", "gpt4"),
    ]
    
    download_dir = "data/text_downloads"
    os.makedirs(download_dir, exist_ok=True)
    
    count = 0
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for url, variant_name in alpaca_urls:
            try:
                print(f"\nDownloading Alpaca variant: {variant_name} from {url}")
                temp_file = os.path.join(download_dir, f"alpaca_{variant_name}.json")
                
                if not download_file(url, temp_file, resume=True):
                    print(f"Warning: Failed to download {variant_name}, trying next variant...")
                    continue
                
                # Process the JSON file
                print(f"Processing {variant_name}...")
                try:
                    with open(temp_file, 'r', encoding='utf-8') as in_f:
                        data = json.load(in_f)
                        if isinstance(data, list):
                            for item in tqdm(data, desc=f"Processing {variant_name}"):
                                try:
                                    text = extract_conversation_text(item)
                                    if text and len(text) > 50:
                                        f.write(text + '\n\n')
                                        count += 1
                                        
                                        if count >= max_samples:
                                            print(f"\nReached sample limit ({max_samples:,}), stopping...")
                                            break
                                except Exception as e:
                                    continue
                        else:
                            print(f"Warning: {variant_name} is not a JSON array, skipping...")
                except json.JSONDecodeError as e:
                    print(f"Warning: Failed to parse {variant_name} JSON: {e}")
                    continue
                except Exception as e:
                    print(f"Warning: Error processing {variant_name}: {e}")
                    continue
                
                if count >= max_samples:
                    break
                    
            except Exception as e:
                print(f"Warning: Error downloading {variant_name}: {e}")
                continue
    
    if count > 0:
        # Only mark as downloaded if we reached max_samples, otherwise keep trying
        if count >= max_samples:
            state["alpaca"]["downloaded"] = True
            state["alpaca"]["converted"] = True
        state["alpaca"]["samples"] = count
        save_state(state)
        
        if count >= max_samples:
            print(f"\n✓ Downloaded {count:,} Alpaca samples to {output_file}")
            return True
        else:
            print(f"\n⚠ Downloaded {count:,} Alpaca samples (target: {max_samples:,}), but data source exhausted.")
            print("   To get more samples, you may need additional data sources.")
            return True  # Return True so script continues, but mark as incomplete
    else:
        # Fallback to original method if all variants fail
        print("Falling back to original Alpaca dataset...")
        fallback_url = "https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/main/alpaca_data.json"
        return download_json_dataset_from_url(
            state, "alpaca", fallback_url,
            "data/text/alpaca.txt", max_samples=max_samples
        )



def download_arxiv_by_category(state, category, category_key, max_samples=100000):
    """Download ArXiv papers for a specific category using ArXiv API with fine-grained resuming"""
    print(f"\n{'='*60}")
    print(f"Downloading ArXiv {category} Papers")
    print("="*60)
    
    if state[category_key]["downloaded"] and state[category_key]["samples"] >= max_samples:
        print(f"ArXiv {category} already downloaded ({state[category_key]['samples']:,} samples), skipping...")
        return True
    
    import time
    import xml.etree.ElementTree as ET
    
    output_file = f"data/text/arxiv_{category.lower()}.txt"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Load checkpoint for resuming
    checkpoint = load_checkpoint(category_key)
    if checkpoint:
        print(f"Resuming from checkpoint: batch {checkpoint.get('last_batch', 0)}, {checkpoint.get('count', 0)} papers")
        count = checkpoint.get('count', 0)
        start = checkpoint.get('last_start', 0)
        mode = 'a'  # Append mode
        catching_up = (count > 0)
    else:
        count = 0
        start = 0
        mode = 'w'  # Write mode
        catching_up = False
    
    if catching_up:
        print(f"Fast-forwarding: Already have {count:,} papers, starting from batch {start // batch_size + 1}...")
    
    # Category to ArXiv category code mapping
    category_map = {
        "physics": ["physics", "quant-ph", "astro-ph", "cond-mat", "hep-ph", "hep-th", "hep-ex", "hep-lat"],
        "chemistry": ["physics.chem-ph", "cond-mat.mtrl-sci"],
        "math": ["math", "math-ph", "stat"],
        "biology": ["q-bio", "q-bio.BM", "q-bio.CB", "q-bio.GN", "q-bio.MN", "q-bio.NC", "q-bio.OT", "q-bio.PE", "q-bio.QM", "q-bio.SC", "q-bio.TO"]
    }
    
    arxiv_categories = category_map.get(category.lower(), [category.lower()])
    
    print(f"Using ArXiv API to fetch {category} papers...")
    print(f"Categories: {', '.join(arxiv_categories)}")
    print("This may take a while due to API rate limits...")
    if checkpoint:
        print(f"Resuming from start={start}, already have {count} papers")
    
    base_url = "http://export.arxiv.org/api/query"
    batch_size = 100  # ArXiv API limit
    
    with open(output_file, mode, encoding='utf-8') as f:
        while count < max_samples:
            # Build query
            query_parts = [f"cat:{cat}" for cat in arxiv_categories]
            query = " OR ".join(query_parts)
            
            params = {
                "search_query": query,
                "start": start,
                "max_results": batch_size,
                "sortBy": "submittedDate",
                "sortOrder": "descending"
            }
            
            try:
                response = requests.get(base_url, params=params, timeout=30)
                response.raise_for_status()
                
                # Parse XML response
                root = ET.fromstring(response.content)
                ns = {'atom': 'http://www.w3.org/2005/Atom'}
                
                entries = root.findall('atom:entry', ns)
                if not entries:
                    print(f"\nNo more papers found (fetched {count} so far)")
                    break
                
                for entry in tqdm(entries, desc=f"Processing batch {start//batch_size + 1}", leave=False):
                    try:
                        title_elem = entry.find('atom:title', ns)
                        summary_elem = entry.find('atom:summary', ns)
                        
                        title = title_elem.text.strip() if title_elem is not None and title_elem.text else ""
                        abstract = summary_elem.text.strip() if summary_elem is not None and summary_elem.text else ""
                        
                        if title and abstract:
                            text = f"Title: {title}\n\nAbstract: {abstract}"
                            if len(text) > 200:
                                f.write(text + '\n\n')
                                count += 1
                                
                                # Print progress with remaining
                                print_progress_with_remaining(count, max_samples, "papers", report_interval=50)
                                
                                if count >= max_samples:
                                    break
                    except Exception as e:
                        continue
                
                start += batch_size
                
                # Print progress with remaining
                print_progress_with_remaining(count, max_samples, "papers", report_interval=50)
                
                # Save checkpoint after each batch
                save_checkpoint(category_key, {
                    'count': count,
                    'last_start': start,
                    'last_batch': start // batch_size
                })
                save_state(state)
                
                # Rate limiting - be polite to ArXiv API
                time.sleep(3)  # 3 second delay between requests
                
                if count >= max_samples:
                    break
                    
            except Exception as e:
                print(f"\nError fetching batch: {e}")
                print(f"Progress saved: {count} papers, batch {start // batch_size}")
                # Save checkpoint on error so we can resume
                save_checkpoint(category_key, {
                    'count': count,
                    'last_start': start,
                    'last_batch': start // batch_size
                })
                save_state(state)
                break
    
    state[category_key]["downloaded"] = True
    state[category_key]["converted"] = True
    state[category_key]["samples"] = count
    save_state(state)
    
    # Clean up checkpoint file (state file only, actual data is never deleted)
    checkpoint_file = f"data/.checkpoint_{category_key}.json"
    if os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)
    
    print(f"\n✓ Downloaded {count:,} {category} papers to {output_file}")
    return True

def download_pubmed(state, max_samples=500000):
    """Download PubMed abstracts using PubMed API with fine-grained resuming"""
    print("\n" + "="*60)
    print("Downloading PubMed Abstracts")
    print("="*60)
    
    if state["pubmed"]["downloaded"] and state["pubmed"]["samples"] >= max_samples:
        print(f"PubMed already downloaded ({state['pubmed']['samples']:,} samples), skipping...")
        return True
    
    import time
    import xml.etree.ElementTree as ET
    
    output_file = "data/text/pubmed.txt"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Load checkpoint
    checkpoint = load_checkpoint("pubmed")
    if checkpoint:
        print(f"Resuming from checkpoint: batch {checkpoint.get('last_batch', 0)}, {checkpoint.get('count', 0)} abstracts")
        count = checkpoint.get('count', 0)
        start_index = checkpoint.get('last_id_index', 0)
        id_list = checkpoint.get('id_list', [])
        mode = 'a'  # Append mode
        resume = True
    else:
        count = 0
        start_index = 0
        id_list = []
        mode = 'w'  # Write mode
        resume = False
    
    max_abstracts = max_samples
    
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    fetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    
    print("Using PubMed API to fetch abstracts...")
    print("This may take a while due to API rate limits...")
    
    # Get paper IDs if not resuming
    if not resume or not id_list:
        # Search for recent papers
        search_params = {
            "db": "pubmed",
            "term": "2023:2024[PDAT]",  # Papers from 2023-2024
            "retmax": 10000,  # Get up to 10k IDs per search
            "retmode": "json"
        }
        
        try:
            response = requests.get(base_url, params=search_params, timeout=30)
            response.raise_for_status()
            data = response.json()
            id_list = data.get('esearchresult', {}).get('idlist', [])
            print(f"Found {len(id_list)} papers. Fetching abstracts...")
            
            # Save ID list to checkpoint
            save_checkpoint("pubmed", {
                'id_list': id_list,
                'count': 0,
                'last_id_index': 0,
                'last_batch': 0
            })
        except Exception as e:
            print(f"ERROR getting paper IDs: {e}")
            return False
    else:
        print(f"Resuming with {len(id_list)} paper IDs, starting from index {start_index}")
    
    batch_size = 100
    
    with open(output_file, mode, encoding='utf-8') as f:
        try:
            # Fetch in batches, starting from checkpoint
            for i in range(start_index, min(len(id_list), max_abstracts), batch_size):
                batch_ids = id_list[i:i+batch_size]
                ids_str = ','.join(batch_ids)
                
                fetch_params = {
                    "db": "pubmed",
                    "id": ids_str,
                    "retmode": "xml",
                    "rettype": "abstract"
                }
                
                try:
                    response = requests.get(fetch_url, params=fetch_params, timeout=30)
                    response.raise_for_status()
                    
                    # Parse XML
                    root = ET.fromstring(response.content)
                    for article in root.findall('.//PubmedArticle'):
                        try:
                            title_elem = article.find('.//ArticleTitle')
                            abstract_elem = article.find('.//AbstractText')
                            
                            title = title_elem.text if title_elem is not None and title_elem.text else ""
                            abstract = abstract_elem.text if abstract_elem is not None and abstract_elem.text else ""
                            
                            if title and abstract:
                                text = f"Title: {title}\n\nAbstract: {abstract}"
                                if len(text) > 100:
                                    f.write(text + '\n\n')
                                    count += 1
                                    
                                    # Print progress with remaining
                                    print_progress_with_remaining(count, max_abstracts, "abstracts", report_interval=100)
                                    
                                    if count >= max_abstracts:
                                        break
                        except Exception as e:
                            continue
                    
                    # Print progress with remaining
                    print_progress_with_remaining(count, max_abstracts, "abstracts", report_interval=100)
                    
                    # Save checkpoint after each batch
                    save_checkpoint("pubmed", {
                        'id_list': id_list,
                        'count': count,
                        'last_id_index': i + batch_size,
                        'last_batch': i // batch_size
                    })
                    save_state(state)
                    
                    # Rate limiting
                    time.sleep(0.34)  # PubMed allows 3 requests/second
                    
                    if count >= max_abstracts:
                        break
                        
                except Exception as e:
                    print(f"Error fetching batch: {e}")
                    # Save checkpoint on error
                    save_checkpoint("pubmed", {
                        'id_list': id_list,
                        'count': count,
                        'last_id_index': i,
                        'last_batch': i // batch_size
                    })
                    save_state(state)
                    continue
                    
        except Exception as e:
            print(f"ERROR downloading PubMed: {e}")
            import traceback
            traceback.print_exc()
            # Save checkpoint on error
            save_checkpoint("pubmed", {
                'id_list': id_list,
                'count': count,
                'last_id_index': start_index,
                'last_batch': start_index // batch_size
            })
            save_state(state)
            return False
    
    state["pubmed"]["downloaded"] = True
    state["pubmed"]["converted"] = True
    state["pubmed"]["samples"] = count
    save_state(state)
    
    # Clean up checkpoint file (state file only, actual data is never deleted)
    checkpoint_file = "data/.checkpoint_pubmed.json"
    if os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)
    
    print(f"\n✓ Downloaded {count:,} PubMed abstracts to {output_file}")
    return True

def download_math_datasets(state, max_samples=100000):
    """Download math problem datasets (GSM8K, MATH, etc.) from GitHub"""
    print("\n" + "="*60)
    print("Downloading Math Datasets")
    print("="*60)
    
    if state["math_datasets"]["downloaded"] and state["math_datasets"]["samples"] >= max_samples:
        print(f"Math datasets already downloaded ({state['math_datasets']['samples']:,} samples), skipping...")
        return True
    
    output_file = "data/text/math_datasets.txt"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    count = 0
    
    # GSM8K from GitHub
    gsm8k_url = "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/train.jsonl"
    
    print("Downloading GSM8K from GitHub...")
    download_dir = "data/text_downloads"
    os.makedirs(download_dir, exist_ok=True)
    gsm8k_file = os.path.join(download_dir, "gsm8k.jsonl")
    
    # Load checkpoint for math datasets
    checkpoint = load_checkpoint("math_datasets")
    last_line_gsm8k = checkpoint.get('last_line_gsm8k', 0) if checkpoint else 0
    catching_up_gsm8k = (last_line_gsm8k > 0)
    
    if catching_up_gsm8k:
        print(f"Fast-forwarding through {last_line_gsm8k:,} already-processed GSM8K lines...")
    
    if download_file(gsm8k_url, gsm8k_file, resume=True):
        with open(output_file, 'a' if catching_up_gsm8k else 'w', encoding='utf-8') as f:
            with open(gsm8k_file, 'r', encoding='utf-8') as in_f:
                line_num = 0
                for line in tqdm(in_f, desc="Processing GSM8K"):
                    line_num += 1
                    
                    # Fast-forward: skip already processed lines
                    if catching_up_gsm8k and line_num <= last_line_gsm8k:
                        if line_num % 10000 == 0:
                            print(f"Fast-forward: {line_num:,} / {last_line_gsm8k:,} lines...")
                        continue
                    
                    # We've caught up - start processing normally
                    if catching_up_gsm8k and line_num > last_line_gsm8k:
                        catching_up_gsm8k = False
                        print(f"Caught up! Starting processing from line {line_num:,}...")
                    
                    try:
                        item = json.loads(line.strip())
                        question = item.get('question', '')
                        answer = item.get('answer', '')
                        
                        if question and answer:
                            text = f"Math Problem: {question}\n\nSolution: {answer}"
                            if len(text) > 50:
                                f.write(text + '\n\n')
                                count += 1
                                
                                # Print progress with remaining
                                print_progress_with_remaining(count, max_samples, "problems", report_interval=100)
                                
                                # Save checkpoint
                                if count % 100 == 0:
                                    save_checkpoint("math_datasets", {
                                        'last_line_gsm8k': line_num,
                                        'count': count
                                    })
                                
                                if count >= max_samples:
                                    break
                    except Exception as e:
                        continue
    
    # MATH competition dataset - download from GitHub
    math_url = "https://raw.githubusercontent.com/hendrycks/math/main/data/math.json"
    math_file = os.path.join(download_dir, "math.json")
    
    if download_file(math_url, math_file, resume=True) and count < max_samples:
        print("Downloading MATH competition dataset...")
        checkpoint_math = load_checkpoint("math_datasets")
        last_idx_math = checkpoint_math.get('last_idx_math', 0) if checkpoint_math else 0
        catching_up_math = (last_idx_math > 0)
        
        if catching_up_math:
            print(f"Fast-forwarding through {last_idx_math:,} already-processed MATH items...")
        
        with open(output_file, 'a', encoding='utf-8') as f:
            with open(math_file, 'r', encoding='utf-8') as in_f:
                data = json.load(in_f)
                for idx, item in enumerate(tqdm(data, desc="Processing MATH")):
                    # Fast-forward: skip already processed items
                    if catching_up_math and idx < last_idx_math:
                        if idx % 10000 == 0:
                            print(f"Fast-forward: {idx:,} / {last_idx_math:,} items...")
                        continue
                    
                    # We've caught up - start processing normally
                    if catching_up_math and idx >= last_idx_math:
                        catching_up_math = False
                        print(f"Caught up! Starting processing from item {idx:,}...")
                    
                    try:
                        problem = item.get('problem', '')
                        solution = item.get('solution', '')
                        
                        if problem and solution:
                            text = f"Math Problem: {problem}\n\nSolution: {solution}"
                            if len(text) > 50:
                                f.write(text + '\n\n')
                                count += 1
                                
                                # Print progress with remaining
                                print_progress_with_remaining(count, max_samples, "problems", report_interval=100)
                                
                                # Save checkpoint
                                if count % 100 == 0:
                                    save_checkpoint("math_datasets", {
                                        'last_line_gsm8k': last_line_gsm8k,
                                        'last_idx_math': idx + 1,
                                        'count': count
                                    })
                                
                                if count >= max_samples:
                                    break
                    except Exception as e:
                        continue
    
    state["math_datasets"]["downloaded"] = True
    state["math_datasets"]["converted"] = True
    state["math_datasets"]["samples"] = count
    save_state(state)
    
    print(f"\n✓ Downloaded {count:,} math problems to {output_file}")
    return True

def download_scienceqa(state, max_samples=50000):
    """Download ScienceQA dataset from GitHub"""
    print("\n" + "="*60)
    print("Downloading ScienceQA")
    print("="*60)
    
    if state["scienceqa"]["downloaded"] and state["scienceqa"]["samples"] >= max_samples:
        print(f"ScienceQA already downloaded ({state['scienceqa']['samples']:,} samples), skipping...")
        return True
    
    # ScienceQA is on GitHub
    scienceqa_url = "https://raw.githubusercontent.com/lil-lab/scienceqa/main/data/scienceqa/train.json"
    
    output_file = "data/text/scienceqa.txt"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    download_dir = "data/text_downloads"
    os.makedirs(download_dir, exist_ok=True)
    scienceqa_file = os.path.join(download_dir, "scienceqa.json")
    
    # Load checkpoint for ScienceQA
    checkpoint = load_checkpoint("scienceqa")
    last_idx = checkpoint.get('last_idx', 0) if checkpoint else 0
    count = checkpoint.get('count', 0) if checkpoint else 0
    catching_up = (last_idx > 0)
    
    if catching_up:
        print(f"Resuming: Already have {count:,} samples, fast-forwarding through {last_idx:,} items...")
    
    if download_file(scienceqa_url, scienceqa_file, resume=True):
        with open(output_file, 'a' if catching_up else 'w', encoding='utf-8') as f:
            with open(scienceqa_file, 'r', encoding='utf-8') as in_f:
                data = json.load(in_f)
                for idx, item in enumerate(tqdm(data, desc="Processing ScienceQA")):
                    # Fast-forward: skip already processed items
                    if catching_up and idx < last_idx:
                        if idx % 10000 == 0:
                            print(f"Fast-forward: {idx:,} / {last_idx:,} items...")
                        continue
                    
                    # We've caught up - start processing normally
                    if catching_up and idx >= last_idx:
                        catching_up = False
                        print(f"Caught up! Starting processing from item {idx:,}...")
                    try:
                        question = item.get('question', '')
                        choices = item.get('choices', {})
                        answer = item.get('answer', '')
                        explanation = item.get('explanation', '')
                        
                        if question:
                            text_parts = [f"Question: {question}"]
                            if choices:
                                text_parts.append(f"Choices: {choices}")
                            if answer:
                                text_parts.append(f"Answer: {answer}")
                            if explanation:
                                text_parts.append(f"Explanation: {explanation}")
                            
                            text = '\n\n'.join(text_parts)
                            if len(text) > 50:
                                f.write(text + '\n\n')
                                count += 1
                                
                                # Print progress with remaining
                                print_progress_with_remaining(count, max_samples, "samples", report_interval=100)
                                
                                # Save checkpoint
                                if count % 100 == 0:
                                    save_checkpoint("scienceqa", {
                                        'last_idx': idx + 1,
                                        'count': count
                                    })
                                
                                if count >= max_samples:
                                    break
                    except Exception as e:
                        continue
        
        state["scienceqa"]["downloaded"] = True
        state["scienceqa"]["converted"] = True
        state["scienceqa"]["samples"] = count
        save_state(state)
        
        print(f"\n✓ Downloaded {count:,} ScienceQA samples to {output_file}")
        return True
    else:
        print("ERROR: Failed to download ScienceQA")
        return False

def download_books(state, max_samples=500000):
    """Download books corpus from Project Gutenberg with fine-grained resuming"""
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
    print("Downloading random books until sample limit is reached...")
    
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
    # We'll try random IDs and skip ones we've already processed
    import random
    
    # Start from a random point if resuming, otherwise start from 1
    if resume:
        current_id = last_tried_id + 1
    else:
        current_id = 1
    
    # Create progress bar that tracks passages
    pbar = tqdm(total=max_samples, desc="Downloading passages", unit="passage", initial=count)
    
    consecutive_failures = 0
    max_consecutive_failures = 100  # Stop trying if 100 consecutive failures
    
    with open(output_file, mode, encoding='utf-8') as f:
        while count < max_samples:
            # Try random book IDs, but also try sequential to find available books
            # Mix of sequential and random to find books efficiently
            if random.random() < 0.3:  # 30% chance to try random ID
                book_id = random.randint(1, 70000)
            else:
                book_id = current_id
                current_id += 1
            
            # Skip if already processed
            if book_id in processed_books:
                continue
            
            try:
                # Try different file formats
                downloaded = False
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
                            
                            # Split into paragraphs
                            paragraphs = book_text.split('\n\n')
                            passages_added = 0
                            for para in paragraphs:
                                para = para.strip()
                                if len(para) > 200:
                                    f.write(para + '\n\n')
                                    f.flush()  # Flush for fine-grained resumption
                                    prev_count = count
                                    count += 1
                                    passages_added += 1
                                    
                                    # Update progress bar
                                    update_amount = min(1, max_samples - prev_count)
                                    if update_amount > 0:
                                        pbar.update(update_amount)
                                    pbar.set_postfix({'books': len(processed_books), 'passages': min(count, max_samples)})
                                    
                                    # Save checkpoint every 50 passages
                                    if count % 50 == 0:
                                        save_checkpoint("books", {
                                            'count': count,
                                            'last_tried_id': book_id,
                                            'processed_books': list(processed_books)
                                        })
                                    
                                    if count >= max_samples:
                                        break
                            
                            if passages_added > 0:
                                processed_books.add(book_id)
                                downloaded = True
                                consecutive_failures = 0  # Reset failure counter on success
                            
                            if count >= max_samples:
                                break
                            break  # Successfully downloaded this book
                    except Exception as e:
                        continue
                
                if downloaded:
                    consecutive_failures = 0
                else:
                    consecutive_failures += 1
                    # Mark as tried to avoid infinite loops
                    processed_books.add(book_id)
                
                # If too many consecutive failures, try more random IDs
                if consecutive_failures >= max_consecutive_failures:
                    print(f"\nWarning: {max_consecutive_failures} consecutive failures. Trying more random IDs...")
                    consecutive_failures = 0
                    # Jump to a random range
                    current_id = random.randint(1, 70000)
                
                if count >= max_samples:
                    break
                    
            except Exception as e:
                consecutive_failures += 1
                continue
    
    pbar.close()
    
    # Only mark as downloaded if we reached max_samples
    if count >= max_samples:
        state["books"]["downloaded"] = True
        state["books"]["converted"] = True
    state["books"]["samples"] = count
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
        "data/text/wikipedia.txt",
        "data/text/books.txt",
        
        # Conversations & Instruction Following
        "data/text/alpaca.txt",
        
        # Scientific Content
        "data/text/arxiv_physics.txt",
        "data/text/arxiv_chemistry.txt",
        "data/text/arxiv_math.txt",
        "data/text/arxiv_biology.txt",
        "data/text/pubmed.txt",
        "data/text/math_datasets.txt",
        "data/text/scienceqa.txt"
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
                       choices=["all", "wikipedia", "books",
                               "alpaca",
                               "arxiv_physics", "arxiv_chemistry", "arxiv_math", "arxiv_biology",
                               "pubmed", "math_datasets", "scienceqa",
                               "conversations", "scientific", "general"], 
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
    parser.add_argument("--max-samples", type=int, default=1000000,
                       help="Maximum number of samples per dataset (default: 1000000, combined total ~12M for all datasets)")
    
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
    
    # General knowledge & English learning
    if args.dataset in ["all", "wikipedia", "general"]:
        if not args.skip_download:
            success = download_wikipedia(state) and success
        if not args.skip_extract:
            success = extract_wikipedia_text(state) and success
        if not args.skip_convert:
            success = convert_wikipedia_to_text(state, args.max_samples) and success
    
    if args.dataset in ["all", "books", "general"]:
        if not args.skip_download:
            success = download_books(state, args.max_samples) and success
    
    # Conversations & Instruction Following
    if args.dataset in ["all", "alpaca", "conversations"]:
        if not args.skip_download:
            success = download_alpaca(state, args.max_samples) and success
    
    # Scientific Content
    if args.dataset in ["all", "arxiv_physics", "scientific"]:
        if not args.skip_download:
            success = download_arxiv_by_category(state, "physics", "arxiv_physics", args.max_samples) and success
    
    if args.dataset in ["all", "arxiv_chemistry", "scientific"]:
        if not args.skip_download:
            success = download_arxiv_by_category(state, "chemistry", "arxiv_chemistry", args.max_samples) and success
    
    if args.dataset in ["all", "arxiv_math", "scientific"]:
        if not args.skip_download:
            success = download_arxiv_by_category(state, "math", "arxiv_math", args.max_samples) and success
    
    if args.dataset in ["all", "arxiv_biology", "scientific"]:
        if not args.skip_download:
            success = download_arxiv_by_category(state, "biology", "arxiv_biology", args.max_samples) and success
    
    if args.dataset in ["all", "pubmed", "scientific"]:
        if not args.skip_download:
            success = download_pubmed(state, args.max_samples) and success
    
    if args.dataset in ["all", "math_datasets", "scientific"]:
        if not args.skip_download:
            success = download_math_datasets(state, args.max_samples) and success
    
    if args.dataset in ["all", "scienceqa", "scientific"]:
        if not args.skip_download:
            success = download_scienceqa(state, args.max_samples) and success
    
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

