"""
Download and prepare production-grade text datasets for μOmni training
Target: Under 30GB, millions of samples
Includes: Conversations, Instruction Following, Tool Calls, Scientific Content (Physics, Chemistry, Math, Biology), English Learning

Supports:
- Conversations: DialogStudio, Alpaca
- Tool Calls: ToolBench
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
        "dialogstudio": {"downloaded": False, "converted": False, "samples": 0},
        "alpaca": {"downloaded": False, "converted": False, "samples": 0, "last_line": 0},
        
        # Tool Calls
        "toolbench": {"downloaded": False, "converted": False, "samples": 0},
        
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

def extract_wikipedia_text_custom(input_file, output_dir, min_text_length=100):
    """Custom Wikipedia XML parser - extracts text without external dependencies"""
    import xml.etree.ElementTree as ET
    import re
    from multiprocessing import Pool
    
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
            
            return f"{title}\n\n{cleaned}"
        except Exception as e:
            return None
    
    print("Parsing Wikipedia XML dump (this may take 30-60 minutes)...")
    print("Using custom XML parser (no external dependencies required)...")
    
    articles = []
    article_count = 0
    max_articles = 1000000  # Limit to prevent memory issues
    
    # Parse XML incrementally
    print("Reading and parsing XML file...")
    try:
        # Handle bz2 compression
        if input_file.endswith('.bz2'):
            file_handle = bz2.open(input_file, 'rb')
        else:
            file_handle = open(input_file, 'rb')
        
        # Use iterparse for memory-efficient parsing
        context = ET.iterparse(file_handle, events=('start', 'end'))
        context = iter(context)
        event, root = next(context)
        
        for event, elem in tqdm(context, desc="Processing pages"):
            if event == 'end' and elem.tag.endswith('page'):
                article = process_page(elem)
                if article:
                    articles.append(article)
                    article_count += 1
                    if article_count >= max_articles:
                        break
                # Clear element to free memory
                elem.clear()
                root.clear()
        
        file_handle.close()
    except ET.ParseError as e:
        print(f"XML parsing error: {e}")
        return False
    except Exception as e:
        print(f"Error reading file: {e}")
        return False
    
    # Write articles to files
    print(f"Writing {len(articles)} articles to output directory...")
    os.makedirs(output_dir, exist_ok=True)
    
    # Split into multiple files (similar to WikiExtractor output format)
    files_per_dir = 100
    file_count = 0
    
    for i, article in enumerate(tqdm(articles, desc="Writing articles")):
        if i % files_per_dir == 0:
            subdir = os.path.join(output_dir, f"AA")
            os.makedirs(subdir, exist_ok=True)
            file_count = 0
        
        output_file = os.path.join(subdir, f"wiki_{file_count:02d}")
        with open(output_file, 'a', encoding='utf-8') as f:
            f.write(article + '\n\n')
        
        file_count += 1
    
    print(f"✓ Extracted {len(articles)} articles")
    return True

def extract_wikipedia_text(state):
    """Extract text from Wikipedia XML dump using custom parser"""
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
    
    # Handle bz2 compression - parse directly from compressed file
    import bz2
    if input_file.endswith('.bz2'):
        print("Parsing directly from bz2 compressed file...")
        # We'll handle decompression in the parser
    
    try:
        success = extract_wikipedia_text_custom(input_file, output_dir, min_text_length=100)
        
        if success:
            state["wikipedia"]["extracted"] = True
            save_state(state)
            print("✓ Wikipedia text extracted successfully")
            return True
        else:
            print("ERROR: Extraction failed")
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
    
    if state["wikipedia"]["converted"]:
        print("Wikipedia already converted, skipping...")
        return True
    
    extracted_dir = "data/text_downloads/wikipedia_extracted"
    output_file = "data/text/wikipedia.txt"
    
    if not os.path.exists(extracted_dir):
        print(f"ERROR: Extracted Wikipedia not found at {extracted_dir}")
        return False
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Load checkpoint if exists
    checkpoint = load_checkpoint("wikipedia")
    if checkpoint:
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
        
        for rel_path, file_path in tqdm(all_files, desc="Processing files"):
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
                    if len(content) > 100:  # Skip very short articles
                        out_f.write(content + '\n\n')
                        count += 1
                        processed_files.add(rel_path)
                        
                        # Save checkpoint every 1000 files
                        if count % 1000 == 0:
                            save_checkpoint("wikipedia", {
                                'processed_files': list(processed_files),
                                'count': count,
                                'last_file': rel_path
                            })
                            save_state(state)
                        
                        # Stop if we've reached sample limit
                        if count >= max_samples:
                            print(f"\nReached sample limit ({max_samples:,}), stopping...")
                            break
            except Exception as e:
                continue
            
            if count >= max_samples:
                break
    
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
    
    if state[dataset_name]["downloaded"]:
        print(f"{dataset_name} already downloaded, skipping...")
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
    else:
        last_line = 0
        count = 0
        mode = 'w'  # Write mode
    
    # Process JSON file
    print("Processing JSON file...")
    
    with open(output_file, mode, encoding='utf-8') as f:
        # Try to read as JSONL (one JSON per line) or regular JSON array
        try:
            with open(temp_file, 'r', encoding='utf-8') as in_f:
                # Try JSONL first
                line_num = 0
                for line in tqdm(in_f, desc=f"Processing {dataset_name}"):
                    line_num += 1
                    
                    # Skip already processed lines
                    if line_num <= last_line:
                        continue
                    
                    try:
                        item = json.loads(line.strip())
                        if extract_func:
                            text = extract_func(item)
                        else:
                            text = extract_conversation_text(item)
                        
                        if text and len(text) > 50:
                            f.write(text + '\n\n')
                            count += 1
                            
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
                        for idx, item in enumerate(tqdm(data, desc=f"Processing {dataset_name}")):
                            # Skip already processed items
                            if idx < last_line:
                                continue
                            
                            try:
                                if extract_func:
                                    text = extract_func(item)
                                else:
                                    text = extract_conversation_text(item)
                                
                                if text and len(text) > 50:
                                    f.write(text + '\n\n')
                                    count += 1
                                    
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

def download_dialogstudio(state, max_samples=100000):
    """Download DialogStudio conversations from GitHub"""
    print("\n" + "="*60)
    print("Downloading DialogStudio")
    print("="*60)
    
    if state["dialogstudio"]["downloaded"]:
        print("DialogStudio already downloaded, skipping...")
        return True
    
    output_file = "data/text/dialogstudio.txt"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Download from GitHub (direct links)
    github_urls = [
        "https://github.com/Salesforce/DialogStudio/raw/main/data/Conversational_Task_Oriented/ConvAI2/train.json",
        "https://github.com/Salesforce/DialogStudio/raw/main/data/Conversational_Task_Oriented/MultiWOZ_2.1/train.json",
    ]
    
    count = 0
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for url in github_urls:
            try:
                print(f"Downloading from {url}...")
                response = requests.get(url, timeout=60, stream=True)
                if response.status_code == 200:
                    data = response.json()
                    if isinstance(data, list):
                        for item in tqdm(data, desc="Processing items"):
                            text = extract_conversation_text(item)
                            if text and len(text) > 50:
                                f.write(text + '\n\n')
                                count += 1
                                if count >= max_samples:
                                    break
                    if count >= max_samples:
                        break
            except Exception as e:
                print(f"Error downloading from {url}: {e}")
                continue
    
    if count > 0:
        state["dialogstudio"]["downloaded"] = True
        state["dialogstudio"]["converted"] = True
        state["dialogstudio"]["samples"] = count
        save_state(state)
        print(f"\n✓ Downloaded {count:,} DialogStudio samples to {output_file}")
        return True
    else:
        print("ERROR: Could not download DialogStudio from any source")
        return False

def download_alpaca(state, max_samples=500000):
    """Download Alpaca instruction following dataset from GitHub and extended variants"""
    print("\n" + "="*60)
    print("Downloading Alpaca")
    print("="*60)
    
    if state["alpaca"]["downloaded"]:
        print("Alpaca already downloaded, skipping...")
        return True
    
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
        state["alpaca"]["downloaded"] = True
        state["alpaca"]["converted"] = True
        state["alpaca"]["samples"] = count
        save_state(state)
        
        print(f"\n✓ Downloaded {count:,} Alpaca samples to {output_file}")
        return True
    else:
        # Fallback to original method if all variants fail
        print("Falling back to original Alpaca dataset...")
        url = "https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/main/alpaca_data.json"
        return download_json_dataset_from_url(
            state, "alpaca", url,
            "data/text/alpaca.txt", max_samples=max_samples
        )


def download_toolbench(state, max_samples=50000):
    """Download ToolBench tool calling dataset from GitHub"""
    print("\n" + "="*60)
    print("Downloading ToolBench")
    print("="*60)
    
    if state["toolbench"]["downloaded"]:
        print("ToolBench already downloaded, skipping...")
        return True
    
    output_file = "data/text/toolbench.txt"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Download from GitHub (direct links)
    github_urls = [
        "https://raw.githubusercontent.com/OpenBMB/ToolBench/main/data/train.json",
    ]
    
    count = 0
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for url in github_urls:
            try:
                print(f"Downloading from {url}...")
                response = requests.get(url, timeout=60, stream=True)
                if response.status_code == 200:
                    data = response.json()
                    if isinstance(data, list):
                        for item in tqdm(data, desc="Processing items"):
                            text = extract_tool_call_text(item)
                            if text and len(text) > 50:
                                f.write(text + '\n\n')
                                count += 1
                                if count >= max_samples:
                                    break
                    elif isinstance(data, dict):
                        # Handle dict format
                        for key, items in data.items():
                            if isinstance(items, list):
                                for item in items:
                                    text = extract_tool_call_text(item)
                                    if text and len(text) > 50:
                                        f.write(text + '\n\n')
                                        count += 1
                                        if count >= max_samples:
                                            break
                    if count >= max_samples:
                        break
            except Exception as e:
                print(f"Error downloading from {url}: {e}")
                continue
    
    if count > 0:
        state["toolbench"]["downloaded"] = True
        state["toolbench"]["converted"] = True
        state["toolbench"]["samples"] = count
        save_state(state)
        print(f"\n✓ Downloaded {count:,} ToolBench samples to {output_file}")
        return True
    else:
        print("ERROR: Could not download ToolBench from any source")
        return False

def extract_tool_call_text(item):
    """Extract tool call information from dataset item"""
    parts = []
    
    if 'query' in item:
        parts.append(f"Query: {item['query']}")
    if 'tool_name' in item or 'function' in item:
        tool = item.get('tool_name', item.get('function', ''))
        parts.append(f"Tool: {tool}")
    if 'parameters' in item or 'arguments' in item:
        params = item.get('parameters', item.get('arguments', ''))
        parts.append(f"Parameters: {params}")
    if 'result' in item or 'output' in item:
        result = item.get('result', item.get('output', ''))
        parts.append(f"Result: {result}")
    
    return '\n'.join(parts) if parts else None

def download_arxiv_by_category(state, category, category_key, max_samples=100000):
    """Download ArXiv papers for a specific category using ArXiv API with fine-grained resuming"""
    print(f"\n{'='*60}")
    print(f"Downloading ArXiv {category} Papers")
    print("="*60)
    
    if state[category_key]["downloaded"]:
        print(f"ArXiv {category} already downloaded, skipping...")
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
    else:
        count = 0
        start = 0
        mode = 'w'  # Write mode
    
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
                                
                                if count >= max_samples:
                                    break
                    except Exception as e:
                        continue
                
                start += batch_size
                
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
    
    if state["pubmed"]["downloaded"]:
        print("PubMed already downloaded, skipping...")
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
                                    
                                    if count >= max_abstracts:
                                        break
                        except Exception as e:
                            continue
                    
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
    
    if state["math_datasets"]["downloaded"]:
        print("Math datasets already downloaded, skipping...")
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
    
    if download_file(gsm8k_url, gsm8k_file, resume=True):
        with open(output_file, 'w', encoding='utf-8') as f:
            with open(gsm8k_file, 'r', encoding='utf-8') as in_f:
                for line in tqdm(in_f, desc="Processing GSM8K"):
                    try:
                        item = json.loads(line.strip())
                        question = item.get('question', '')
                        answer = item.get('answer', '')
                        
                        if question and answer:
                            text = f"Math Problem: {question}\n\nSolution: {answer}"
                            if len(text) > 50:
                                f.write(text + '\n\n')
                                count += 1
                                
                                if count >= max_samples:
                                    break
                    except Exception as e:
                        continue
    
    # MATH competition dataset - download from GitHub
    math_url = "https://raw.githubusercontent.com/hendrycks/math/main/data/math.json"
    math_file = os.path.join(download_dir, "math.json")
    
    if download_file(math_url, math_file, resume=True) and count < max_samples:
        print("Downloading MATH competition dataset...")
        with open(output_file, 'a', encoding='utf-8') as f:
            with open(math_file, 'r', encoding='utf-8') as in_f:
                data = json.load(in_f)
                for item in tqdm(data, desc="Processing MATH"):
                    try:
                        problem = item.get('problem', '')
                        solution = item.get('solution', '')
                        
                        if problem and solution:
                            text = f"Math Problem: {problem}\n\nSolution: {solution}"
                            if len(text) > 50:
                                f.write(text + '\n\n')
                                count += 1
                                
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
    
    if state["scienceqa"]["downloaded"]:
        print("ScienceQA already downloaded, skipping...")
        return True
    
    # ScienceQA is on GitHub
    scienceqa_url = "https://raw.githubusercontent.com/lil-lab/scienceqa/main/data/scienceqa/train.json"
    
    output_file = "data/text/scienceqa.txt"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    download_dir = "data/text_downloads"
    os.makedirs(download_dir, exist_ok=True)
    scienceqa_file = os.path.join(download_dir, "scienceqa.json")
    
    if download_file(scienceqa_url, scienceqa_file, resume=True):
        count = 0
        
        with open(output_file, 'w', encoding='utf-8') as f:
            with open(scienceqa_file, 'r', encoding='utf-8') as in_f:
                data = json.load(in_f)
                for item in tqdm(data, desc="Processing ScienceQA"):
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
    """Download books corpus from Project Gutenberg"""
    print("\n" + "="*60)
    print("Downloading Books Corpus")
    print("="*60)
    
    if state["books"]["downloaded"]:
        print("Books already downloaded, skipping...")
        return True
    
    print("Downloading books from Project Gutenberg...")
    print("NOTE: Project Gutenberg provides free books in plain text format.")
    
    output_file = "data/text/books.txt"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Project Gutenberg book IDs (popular books)
    # You can expand this list
    book_ids = [
        1342,  # Pride and Prejudice
        84,    # Frankenstein
        11,    # Alice's Adventures in Wonderland
        2701,  # Moby Dick
        74,    # The Adventures of Tom Sawyer
        98,    # A Tale of Two Cities
        5200,  # Metamorphosis
        1661,  # The Adventures of Sherlock Holmes
    ]
    
    count = 0
    
    base_url = "https://www.gutenberg.org/files"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for book_id in tqdm(book_ids, desc="Downloading books"):
            try:
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
                            start_idx = 0
                            end_idx = len(lines)
                            
                            # Find start (skip header)
                            for i, line in enumerate(lines):
                                if 'START OF THIS PROJECT GUTENBERG' in line.upper():
                                    start_idx = i + 1
                                    break
                            
                            # Find end (skip footer)
                            for i in range(len(lines)-1, -1, -1):
                                if 'END OF THIS PROJECT GUTENBERG' in lines[i].upper():
                                    end_idx = i
                                    break
                            
                            book_text = '\n'.join(lines[start_idx:end_idx])
                            
                            # Split into paragraphs
                            paragraphs = book_text.split('\n\n')
                            for para in paragraphs:
                                para = para.strip()
                                if len(para) > 200:
                                    f.write(para + '\n\n')
                                    count += 1
                                    
                                    if count >= max_samples:
                                        break
                            
                            if count >= max_samples:
                                break
                            break  # Successfully downloaded this book
                    except Exception as e:
                        continue
                
                if count >= max_samples:
                    break
                    
            except Exception as e:
                continue
    
    state["books"]["downloaded"] = True
    state["books"]["converted"] = True
    state["books"]["samples"] = count
    save_state(state)
    
    print(f"\n✓ Downloaded {count:,} book passages to {output_file}")
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
        "data/text/dialogstudio.txt",
        "data/text/alpaca.txt",
        
        # Tool Calls
        "data/text/toolbench.txt",
        
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
                               "dialogstudio", "alpaca",
                               "toolbench",
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
    if args.dataset in ["all", "dialogstudio", "conversations"]:
        if not args.skip_download:
            success = download_dialogstudio(state, args.max_samples) and success
    
    if args.dataset in ["all", "alpaca", "conversations"]:
        if not args.skip_download:
            success = download_alpaca(state, args.max_samples) and success
    
    # Tool Calls
    if args.dataset in ["all", "toolbench"]:
        if not args.skip_download:
            success = download_toolbench(state, args.max_samples) and success
    
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

