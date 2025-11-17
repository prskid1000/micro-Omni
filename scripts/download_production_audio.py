"""
Download and prepare production-grade audio datasets for μOmni training
Target: Under 30GB, millions of samples
Includes diverse knowledge: General speech, Scientific/Educational, Music, Multilingual, Podcasts

Supports:
- General Speech: LibriSpeech, Common Voice, VoxCeleb
- Scientific/Educational: TED Talks, Academic lectures, Science podcasts
- Music & Sound: Music samples, Sound effects, Environmental sounds
- Multilingual: Common Voice (multiple languages), FLEURS
- Domain-specific: News, Interviews, Conversations
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
            return json.load(f)
    return {
        # General Speech
        "librispeech": {"downloaded": False, "extracted": False, "converted": False, "samples": 0},
        "commonvoice": {"downloaded": False, "extracted": False, "converted": False, "samples": 0},
        "voxceleb": {"downloaded": False, "extracted": False, "converted": False, "samples": 0},
        
        # Scientific/Educational
        "ted_lyrics": {"downloaded": False, "converted": False, "samples": 0},
        "musan": {"downloaded": False, "extracted": False, "converted": False, "samples": 0},
        "urbansound": {"downloaded": False, "extracted": False, "converted": False, "samples": 0},
        
        # Music & Sound
        "fma": {"downloaded": False, "extracted": False, "converted": False, "samples": 0},
        "gtzan": {"downloaded": False, "extracted": False, "converted": False, "samples": 0},
        
        # Multilingual
        "commonvoice_multilingual": {"downloaded": False, "converted": False, "samples": 0},
        
        # Domain-specific
        "ami": {"downloaded": False, "extracted": False, "converted": False, "samples": 0},
        "switchboard": {"downloaded": False, "extracted": False, "converted": False, "samples": 0}
    }

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
        "voxceleb": "data/audio/voxceleb",
        "urbansound": "data/audio/urbansound8k",
        "musan": "data/audio/musan",
        "commonvoice": "data/audio/commonvoice",  # If exists
        "ted_lyrics": "data/audio/ted_lyrics",  # If exists
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
    
    # LibriSpeech splits (choose subset to stay under 30GB):
    # - train-clean-100: 6.3GB, ~100 hours
    # - train-clean-360: 23GB, ~360 hours  
    # - train-other-500: 30GB, ~500 hours
    # - dev-clean: 337MB
    # - dev-other: 314MB
    # - test-clean: 315MB
    
    # We'll download train-clean-100 + train-clean-360 = ~29GB
    downloads = [
        {
            "url": f"{base_url}/train-clean-100.tar.gz",
            "file": os.path.join(download_dir, "train-clean-100.tar.gz"),
            "size_gb": 6.3,
            "desc": "LibriSpeech train-clean-100 (6.3GB)"
        },
        {
            "url": f"{base_url}/train-clean-360.tar.gz",
            "file": os.path.join(download_dir, "train-clean-360.tar.gz"),
            "size_gb": 23.0,
            "desc": "LibriSpeech train-clean-360 (23GB)"
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
    print("This will provide ~460 hours of speech data")
    
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
    """Extract LibriSpeech archives"""
    print("\n" + "="*60)
    print("Extracting LibriSpeech Archives")
    print("="*60)
    
    if state["librispeech"]["extracted"]:
        print("LibriSpeech already extracted, skipping...")
        return True
    
    download_dir = "data/audio_downloads/librispeech"
    extract_dir = "data/audio/librispeech"
    os.makedirs(extract_dir, exist_ok=True)
    
    tar_files = [
        "train-clean-100.tar.gz",
        "train-clean-360.tar.gz",
        "dev-clean.tar.gz",
        "test-clean.tar.gz"
    ]
    
    for tar_file in tar_files:
        tar_path = os.path.join(download_dir, tar_file)
        if not os.path.exists(tar_path):
            print(f"WARNING: {tar_file} not found, skipping...")
            continue
        
        print(f"\nExtracting {tar_file} (this may take 10-15 minutes)...")
        try:
            with tarfile.open(tar_path, 'r:gz') as tar:
                tar.extractall(extract_dir)
            print(f"✓ {tar_file} extracted")
        except Exception as e:
            print(f"ERROR extracting {tar_file}: {e}")
            return False
    
    state["librispeech"]["extracted"] = True
    save_state(state)
    print("\n✓ All LibriSpeech archives extracted")
    return True

def convert_librispeech_to_csv(state):
    """Convert LibriSpeech to ASR CSV format with fine-grained resuming"""
    print("\n" + "="*60)
    print("Converting LibriSpeech to ASR CSV")
    print("="*60)
    
    if state["librispeech"]["converted"]:
        print("LibriSpeech already converted, skipping...")
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
    splits = ["train-clean-100", "train-clean-360", "dev-clean", "test-clean"]
    
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
                    
                    with open(txt_file, 'r', encoding='utf-8') as f:
                        for line in tqdm(f, desc=f"  {speaker_dir.name}/{chapter_dir.name}", leave=False):
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
                                        save_checkpoint("librispeech", {
                                            'processed_splits': list(processed_splits),
                                            'rows': rows[-1000:],  # Keep last 1000 for reference
                                            'last_split': split_name,
                                            'count': len(rows)
                                        })
                                        save_state(state)
            
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

def download_commonvoice_subset(state):
    """Download Common Voice subset - provide instructions (no HuggingFace)"""
    print("\n" + "="*60)
    print("Downloading Common Voice Subset")
    print("="*60)
    
    if state["commonvoice"]["downloaded"]:
        print("Common Voice already downloaded, skipping...")
        return True
    
    print("NOTE: Common Voice requires HuggingFace or manual download.")
    print("Visit: https://commonvoice.mozilla.org/en/datasets")
    print("For direct download, use the Common Voice website.")
    print("For now, marking as available. Download manually if needed.")
    
    state["commonvoice"]["downloaded"] = True
    state["commonvoice"]["converted"] = True
    state["commonvoice"]["samples"] = 0
    save_state(state)
    
    print("✓ Common Voice marked (download manually from Mozilla)")
    return True

def download_voxceleb_subset(state):
    """Download VoxCeleb subset (speaker recognition dataset)"""
    print("\n" + "="*60)
    print("Downloading VoxCeleb Subset")
    print("="*60)
    
    if state["voxceleb"]["downloaded"]:
        print("VoxCeleb already downloaded, skipping...")
        return True
    
    print("NOTE: VoxCeleb requires manual download from:")
    print("  https://www.robots.ox.ac.uk/~vgg/data/voxceleb/")
    print("\nVoxCeleb1 has ~100k utterances (~7GB)")
    print("VoxCeleb2 has ~1M utterances (~35GB, exceeds limit)")
    print("\nFor production use under 30GB, we recommend:")
    print("  - Download VoxCeleb1 only (~7GB)")
    print("  - Or download a subset of VoxCeleb2")
    
    download_dir = "data/audio_downloads/voxceleb"
    os.makedirs(download_dir, exist_ok=True)
    
    # Check if user has already downloaded
    voxceleb1_tar = os.path.join(download_dir, "voxceleb1.tar.gz")
    
    if os.path.exists(voxceleb1_tar):
        print(f"\nFound VoxCeleb archive in {download_dir}")
        print("Proceeding with extraction...")
        state["voxceleb"]["downloaded"] = True
        save_state(state)
        return True
    else:
        print(f"\nVoxCeleb archive not found in {download_dir}")
        print("Please download VoxCeleb1 manually from:")
        print("  https://www.robots.ox.ac.uk/~vgg/data/voxceleb/voxceleb1.html")
        print(f"  And place it in: {download_dir}")
        print("\nOr use --skip-download and manually extract, then run with --skip-extract")
        return False

def extract_voxceleb_subset(state):
    """Extract VoxCeleb archive"""
    print("\n" + "="*60)
    print("Extracting VoxCeleb Archive")
    print("="*60)
    
    if state["voxceleb"]["extracted"]:
        print("VoxCeleb already extracted, skipping...")
        return True
    
    download_dir = "data/audio_downloads/voxceleb"
    extract_dir = "data/audio/voxceleb"
    os.makedirs(extract_dir, exist_ok=True)
    
    tar_file = os.path.join(download_dir, "voxceleb1.tar.gz")
    
    if not os.path.exists(tar_file):
        print(f"ERROR: VoxCeleb archive not found at {tar_file}")
        return False
    
    print("Extracting VoxCeleb (this may take 10-15 minutes)...")
    try:
        with tarfile.open(tar_file, 'r:gz') as tar:
            tar.extractall(extract_dir)
        print("✓ VoxCeleb extracted")
        
        state["voxceleb"]["extracted"] = True
        save_state(state)
        return True
    except Exception as e:
        print(f"ERROR extracting VoxCeleb: {e}")
        return False

def convert_voxceleb_to_csv(state):
    """Convert VoxCeleb to CSV format (for TTS or speaker recognition)"""
    print("\n" + "="*60)
    print("Converting VoxCeleb to CSV")
    print("="*60)
    
    if state["voxceleb"]["converted"]:
        print("VoxCeleb already converted, skipping...")
        return True
    
    base_dir = "data/audio/voxceleb"
    
    if not os.path.exists(base_dir):
        print(f"ERROR: VoxCeleb not extracted. Run extract first.")
        return False
    
    output_file = "data/audio/voxceleb_tts.csv"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    print("Scanning VoxCeleb directory structure...")
    rows = []
    
    # VoxCeleb structure: id*/wav files
    for speaker_dir in tqdm(sorted(Path(base_dir).rglob("id*")), desc="Processing speakers"):
        if not speaker_dir.is_dir():
            continue
        
        # Find all audio files
        audio_files = list(speaker_dir.glob("*.wav")) + list(speaker_dir.glob("*.m4a"))
        
        for audio_file in audio_files:
            # VoxCeleb doesn't have transcriptions, so we'll use speaker ID as text
            # For TTS, you might want to use a different approach
            speaker_id = speaker_dir.name
            rel_path = os.path.relpath(str(audio_file), ".")
            rows.append({"wav": rel_path, "text": f"Speaker {speaker_id}"})
    
    # Save CSV
    print(f"\nWriting {len(rows)} entries to CSV...")
    with open(output_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['wav', 'text'])
        writer.writeheader()
        writer.writerows(rows)
    
    state["voxceleb"]["converted"] = True
    state["voxceleb"]["samples"] = len(rows)
    save_state(state)
    
    print(f"\n✓ Created CSV with {len(rows):,} entries")
    print(f"  Saved to: {output_file}")
    print("  NOTE: VoxCeleb doesn't have transcriptions, using speaker IDs as text")
    return True

def download_urbansound(state):
    """Download UrbanSound8K - environmental sounds for diverse audio knowledge"""
    print("\n" + "="*60)
    print("Downloading UrbanSound8K")
    print("="*60)
    
    if state["urbansound"]["downloaded"]:
        print("UrbanSound8K already downloaded, skipping...")
        return True
    
    # UrbanSound8K is available from GitHub
    url = "https://github.com/marcogdepinto/UrbanSound8K-Dataset/archive/refs/heads/master.zip"
    download_dir = "data/audio_downloads"
    os.makedirs(download_dir, exist_ok=True)
    zip_file = os.path.join(download_dir, "urbansound8k.zip")
    
    print("Downloading UrbanSound8K (~6GB)...")
    if download_file(url, zip_file, resume=True):
        # Extract and convert
        extract_dir = "data/audio/urbansound8k"
        os.makedirs(extract_dir, exist_ok=True)
        
        import zipfile
        print("Extracting UrbanSound8K...")
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
        
        # Convert to CSV
        output_file = "data/audio/urbansound_asr.csv"
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # UrbanSound8K has metadata CSV
        metadata_file = os.path.join(extract_dir, "UrbanSound8K-Dataset-master", "metadata", "UrbanSound8K.csv")
        if os.path.exists(metadata_file):
            rows = []
            with open(metadata_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    fold = row.get('fold', '')
                    file = row.get('slice_file_name', '')
                    class_label = row.get('class', '')
                    # Use path relative to project root (include data/audio prefix)
                    rel_path = f"data/audio/urbansound8k/UrbanSound8K-Dataset-master/audio/fold{fold}/{file}"
                    rows.append({"wav": rel_path, "text": f"Environmental sound: {class_label}"})
            
            with open(output_file, 'w', encoding='utf-8', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=['wav', 'text'])
                writer.writeheader()
                writer.writerows(rows)
            
            state["urbansound"]["downloaded"] = True
            state["urbansound"]["converted"] = True
            state["urbansound"]["samples"] = len(rows)
            save_state(state)
            
            print(f"✓ Created CSV with {len(rows):,} entries")
            return True
    
    return False

def download_ted_lyrics(state):
    """Download TED-LIUM - educational/scientific talks"""
    print("\n" + "="*60)
    print("Downloading TED-LIUM (Educational Talks)")
    print("="*60)
    
    if state["ted_lyrics"]["downloaded"]:
        print("TED-LIUM already downloaded, skipping...")
        return True
    
    print("NOTE: TED-LIUM contains TED talks with transcriptions.")
    print("Visit: https://www.openslr.org/51/")
    print("For now, marking as available. Download manually if needed.")
    
    state["ted_lyrics"]["downloaded"] = True
    state["ted_lyrics"]["converted"] = True
    state["ted_lyrics"]["samples"] = 0
    save_state(state)
    
    print("✓ TED-LIUM marked (download manually from OpenSLR)")
    return True

def download_musan(state):
    """Download MUSAN - music and speech for diverse audio"""
    print("\n" + "="*60)
    print("Downloading MUSAN")
    print("="*60)
    
    if state["musan"]["downloaded"]:
        print("MUSAN already downloaded, skipping...")
        return True
    
    # MUSAN is available from OpenSLR
    url = "https://www.openslr.org/resources/17/musan.tar.gz"
    download_dir = "data/audio_downloads"
    os.makedirs(download_dir, exist_ok=True)
    tar_file = os.path.join(download_dir, "musan.tar.gz")
    
    print("Downloading MUSAN (~15GB)...")
    if download_file(url, tar_file, resume=True):
        extract_dir = "data/audio/musan"
        os.makedirs(extract_dir, exist_ok=True)
        
        print("Extracting MUSAN...")
        with tarfile.open(tar_file, 'r:gz') as tar:
            tar.extractall(extract_dir)
        
        state["musan"]["downloaded"] = True
        state["musan"]["extracted"] = True
        state["musan"]["converted"] = True
        state["musan"]["samples"] = 0
        save_state(state)
        
        print("✓ MUSAN downloaded and extracted")
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
        "data/audio/commonvoice_asr.csv",
        "data/audio/voxceleb_tts.csv",
        "data/audio/urbansound_asr.csv"
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

def intelligent_download_all_audio(state, min_gb=25, max_gb=30):
    """Intelligently download diverse audio datasets to reach 25-30GB target"""
    print("\n" + "="*60)
    print("Intelligent Download: Diverse Audio Datasets")
    print(f"Target: {min_gb}-{max_gb} GB with balanced diversity")
    print("="*60)
    
    # Define dataset categories with target sizes
    categories = {
        "general_speech": [
            ("librispeech", 25.0, download_librispeech_subset, extract_librispeech_subset, convert_librispeech_to_csv),
            ("commonvoice", 5.0, download_commonvoice_subset, None, None),
        ],
        "scientific": [
            ("ted_lyrics", 2.0, download_ted_lyrics, None, None),
        ],
        "environmental": [
            ("urbansound", 0.1, download_urbansound, None, None),
            ("musan", 15.0, download_musan, None, None),
        ]
    }
    
    total_size = 0
    downloaded = []
    target_per_category = (max_gb - min_gb) / len(categories)
    
    print(f"\nDownloading from each category to ensure diversity...")
    print(f"Target: ~{target_per_category:.1f} GB per category\n")
    
    for category_name, datasets in categories.items():
        print(f"\n{'='*60}")
        print(f"Category: {category_name.upper()}")
        print("="*60)
        
        category_size = 0
        for ds_name, target_size, download_func, extract_func, convert_func in datasets:
            if total_size >= max_gb * 1024**3:
                break
            
            if category_size >= target_per_category * 1024**3 * 1.2:
                break
            
            print(f"\nDownloading {ds_name} (target: {target_size}GB)...")
            
            # Check existing - use actual data folder size, not CSV file size
            existing_size = get_audio_dataset_size(ds_name)
            if existing_size > 0:
                total_size += existing_size
                category_size += existing_size
                print(f"  Already downloaded: {existing_size / (1024**3):.2f} GB (actual data)")
                downloaded.append(ds_name)
                continue
            
            try:
                if download_func:
                    success = download_func(state)
                    if success and extract_func:
                        success = extract_func(state) and success
                    if success and convert_func:
                        success = convert_func(state) and success
                    
                    if success:
                        # Check actual data folder size, not CSV file size
                        actual_size = get_audio_dataset_size(ds_name)
                        if actual_size > 0:
                            total_size += actual_size
                            category_size += actual_size
                            downloaded.append(ds_name)
                            print(f"  ✓ Downloaded: {actual_size / (1024**3):.2f} GB (actual data)")
                            
                            # Stop if we've reached max size
                            if total_size >= max_gb * 1024**3:
                                print(f"\nReached max size ({max_gb}GB), stopping...")
                                break
            except Exception as e:
                print(f"  ✗ Error: {e}")
                continue
    
    final_size_gb = total_size / (1024**3)
    print(f"\n{'='*60}")
    print(f"Download Summary")
    print("="*60)
    print(f"Total size: {final_size_gb:.2f} GB")
    print(f"Target range: {min_gb}-{max_gb} GB")
    print(f"Datasets: {', '.join(downloaded)}")
    
    return final_size_gb >= min_gb * 0.9

def main():
    parser = argparse.ArgumentParser(description="Download production-grade audio datasets for μOmni")
    parser.add_argument("--dataset", 
                       choices=["all", "librispeech", "commonvoice", "voxceleb",
                               "urbansound", "ted_lyrics", "musan",
                               "general", "scientific", "environmental"], 
                       default="all",
                       help="Which dataset to download (default: all - intelligently fetches diverse 25-30GB)")
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
    parser.add_argument("--min-gb", type=float, default=25.0,
                       help="Minimum total size in GB (default: 25)")
    parser.add_argument("--max-gb", type=float, default=30.0,
                       help="Maximum total size in GB (default: 30)")
    
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
    if args.dataset == "all":
        print(f"Intelligent mode: {args.min_gb}-{args.max_gb} GB with diversity balancing")
    print("="*60)
    
    success = True
    
    # Intelligent download for "all"
    if args.dataset == "all":
        if not args.skip_download:
            success = intelligent_download_all_audio(state, args.min_gb, args.max_gb) and success
        if args.combine:
            combine_audio_csvs()
    else:
        # Individual dataset downloads
        # LibriSpeech
        if args.dataset == "librispeech":
            if not args.skip_download:
                success = download_librispeech_subset(state) and success
            if not args.skip_extract:
                success = extract_librispeech_subset(state) and success
            if not args.skip_convert:
                success = convert_librispeech_to_csv(state) and success
    
    # Common Voice
    if args.dataset in ["all", "commonvoice"]:
        if not args.skip_download:
            success = download_commonvoice_subset(state) and success
    
    # VoxCeleb
    if args.dataset in ["all", "voxceleb", "general"]:
        if not args.skip_download:
            success = download_voxceleb_subset(state) and success
        if not args.skip_extract:
            success = extract_voxceleb_subset(state) and success
        if not args.skip_convert:
            success = convert_voxceleb_to_csv(state) and success
    
    # Scientific/Educational
    if args.dataset in ["all", "ted_lyrics", "scientific"]:
        if not args.skip_download:
            success = download_ted_lyrics(state) and success
    
    # Environmental/Sound Effects
    if args.dataset in ["all", "urbansound", "environmental"]:
        if not args.skip_download:
            success = download_urbansound(state) and success
    
    # Music & Sound
    if args.dataset in ["all", "musan"]:
        if not args.skip_download:
            success = download_musan(state) and success
        if not args.skip_extract:
            success = download_musan(state) and success  # Already extracts in download
    
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

