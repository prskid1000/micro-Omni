"""
Data preparation script for post-training datasets.

This script helps prepare datasets in the correct format for post-training:
- Text data: Converts raw text files to training format
- Audio data: Prepares ASR/TTS CSV files
- Image data: Creates image-caption manifest files

Usage:
    # Text data
    python scripts/prep_post_training_data.py \
        --input data/raw/my_text.txt \
        --output data/post_training/text.txt \
        --format text

    # Audio ASR data
    python scripts/prep_post_training_data.py \
        --input data/raw/audio/ \
        --output data/post_training/asr.csv \
        --format audio_asr \
        --sample_rate 16000

    # Audio TTS data
    python scripts/prep_post_training_data.py \
        --input data/raw/audio/ \
        --output data/post_training/tts.csv \
        --format audio_tts \
        --sample_rate 16000

    # Image data
    python scripts/prep_post_training_data.py \
        --input data/raw/images/ \
        --output data/post_training/images.json \
        --format images \
        --caption_file data/raw/captions.txt
"""

import argparse
import os
import json
import csv
import glob
from pathlib import Path

def prepare_text_data(input_path, output_path, encoding='utf-8'):
    """
    Prepare text data for post-training.
    
    Args:
        input_path: Path to input text file or directory
        output_path: Path to output text file
        encoding: File encoding (default: utf-8)
    """
    print(f"Preparing text data from {input_path}...")
    
    # Collect all text files
    text_files = []
    if os.path.isfile(input_path):
        text_files = [input_path]
    elif os.path.isdir(input_path):
        # Find all .txt files in directory
        text_files = glob.glob(os.path.join(input_path, "**/*.txt"), recursive=True)
        if not text_files:
            text_files = glob.glob(os.path.join(input_path, "*.txt"))
    else:
        raise ValueError(f"Input path not found: {input_path}")
    
    if not text_files:
        raise ValueError(f"No text files found in {input_path}")
    
    print(f"Found {len(text_files)} text file(s)")
    
    # Read and combine all text files
    lines = []
    for text_file in text_files:
        try:
            with open(text_file, 'r', encoding=encoding, errors='ignore') as f:
                file_lines = [l.strip() for l in f if l.strip()]
                lines.extend(file_lines)
                print(f"  Read {len(file_lines)} lines from {text_file}")
        except Exception as e:
            print(f"  Warning: Could not read {text_file}: {e}")
    
    # Remove duplicates while preserving order
    seen = set()
    unique_lines = []
    for line in lines:
        if line not in seen:
            seen.add(line)
            unique_lines.append(line)
    
    print(f"Total lines: {len(lines)}, Unique lines: {len(unique_lines)}")
    
    # Write output
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for line in unique_lines:
            f.write(line + '\n')
    
    print(f"✓ Saved {len(unique_lines)} lines to {output_path}")
    return len(unique_lines)

def prepare_audio_asr_data(input_path, output_path, sample_rate=16000):
    """
    Prepare audio ASR data for post-training.
    
    Args:
        input_path: Path to directory containing audio files
        output_path: Path to output CSV file
        sample_rate: Expected sample rate (default: 16000)
    """
    print(f"Preparing audio ASR data from {input_path}...")
    
    if not os.path.isdir(input_path):
        raise ValueError(f"Input path must be a directory: {input_path}")
    
    # Find all audio files
    audio_extensions = ['.wav', '.flac', '.mp3', '.m4a']
    audio_files = []
    for ext in audio_extensions:
        audio_files.extend(glob.glob(os.path.join(input_path, "**/*" + ext), recursive=True))
        audio_files.extend(glob.glob(os.path.join(input_path, "*" + ext)))
    
    if not audio_files:
        raise ValueError(f"No audio files found in {input_path}")
    
    print(f"Found {len(audio_files)} audio file(s)")
    
    # Check for transcript file
    transcript_file = os.path.join(input_path, "transcripts.txt")
    transcripts = {}
    
    if os.path.exists(transcript_file):
        print(f"Loading transcripts from {transcript_file}...")
        with open(transcript_file, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t', 1)
                if len(parts) == 2:
                    audio_id, text = parts
                    transcripts[audio_id] = text
        print(f"  Loaded {len(transcripts)} transcripts")
    
    # Create CSV rows
    rows = []
    for audio_file in audio_files:
        # Get audio ID (filename without extension)
        audio_id = os.path.splitext(os.path.basename(audio_file))[0]
        
        # Get transcript
        if audio_id in transcripts:
            text = transcripts[audio_id]
        elif os.path.exists(transcript_file):
            # Try to find transcript by filename
            text = transcripts.get(audio_id, "")
        else:
            # No transcript file - use placeholder
            text = f"audio transcription for {audio_id}"
            print(f"  Warning: No transcript for {audio_id}, using placeholder")
        
        rows.append({
            "wav": os.path.abspath(audio_file),
            "text": text
        })
    
    # Write CSV
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['wav', 'text'])
        writer.writeheader()
        writer.writerows(rows)
    
    print(f"✓ Saved {len(rows)} audio entries to {output_path}")
    return len(rows)

def prepare_audio_tts_data(input_path, output_path, sample_rate=16000):
    """
    Prepare audio TTS data for post-training.
    
    Args:
        input_path: Path to directory containing audio files
        output_path: Path to output CSV file
        sample_rate: Expected sample rate (default: 16000)
    """
    print(f"Preparing audio TTS data from {input_path}...")
    
    if not os.path.isdir(input_path):
        raise ValueError(f"Input path must be a directory: {input_path}")
    
    # Find all audio files
    audio_extensions = ['.wav', '.flac', '.mp3', '.m4a']
    audio_files = []
    for ext in audio_extensions:
        audio_files.extend(glob.glob(os.path.join(input_path, "**/*" + ext), recursive=True))
        audio_files.extend(glob.glob(os.path.join(input_path, "*" + ext)))
    
    if not audio_files:
        raise ValueError(f"No audio files found in {input_path}")
    
    print(f"Found {len(audio_files)} audio file(s)")
    
    # Check for transcript file
    transcript_file = os.path.join(input_path, "transcripts.txt")
    transcripts = {}
    
    if os.path.exists(transcript_file):
        print(f"Loading transcripts from {transcript_file}...")
        with open(transcript_file, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t', 1)
                if len(parts) == 2:
                    audio_id, text = parts
                    transcripts[audio_id] = text
        print(f"  Loaded {len(transcripts)} transcripts")
    
    # Create CSV rows (TTS format: text, wav)
    rows = []
    for audio_file in audio_files:
        # Get audio ID (filename without extension)
        audio_id = os.path.splitext(os.path.basename(audio_file))[0]
        
        # Get transcript
        if audio_id in transcripts:
            text = transcripts[audio_id]
        elif os.path.exists(transcript_file):
            text = transcripts.get(audio_id, "")
        else:
            text = f"audio transcription for {audio_id}"
            print(f"  Warning: No transcript for {audio_id}, using placeholder")
        
        rows.append({
            "text": text,
            "wav": os.path.abspath(audio_file)
        })
    
    # Write CSV
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['text', 'wav'])
        writer.writeheader()
        writer.writerows(rows)
    
    print(f"✓ Saved {len(rows)} TTS entries to {output_path}")
    return len(rows)

def prepare_image_data(input_path, output_path, caption_file=None):
    """
    Prepare image data for post-training.
    
    Args:
        input_path: Path to directory containing images
        output_path: Path to output JSON manifest file
        caption_file: Optional path to caption file (one caption per line)
    """
    print(f"Preparing image data from {input_path}...")
    
    if not os.path.isdir(input_path):
        raise ValueError(f"Input path must be a directory: {input_path}")
    
    # Find all image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(input_path, "**/*" + ext), recursive=True))
        image_files.extend(glob.glob(os.path.join(input_path, "*" + ext)))
        # Also check uppercase
        image_files.extend(glob.glob(os.path.join(input_path, "**/*" + ext.upper()), recursive=True))
        image_files.extend(glob.glob(os.path.join(input_path, "*" + ext.upper())))
    
    if not image_files:
        raise ValueError(f"No image files found in {input_path}")
    
    print(f"Found {len(image_files)} image file(s)")
    
    # Load captions if provided
    captions = {}
    if caption_file and os.path.exists(caption_file):
        print(f"Loading captions from {caption_file}...")
        with open(caption_file, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                captions[idx] = line.strip()
        print(f"  Loaded {len(captions)} captions")
    
    # Create manifest
    manifest = []
    for idx, image_file in enumerate(image_files):
        # Get image ID (filename without extension)
        image_id = os.path.splitext(os.path.basename(image_file))[0]
        
        # Get caption
        if idx in captions:
            caption = captions[idx]
        elif image_id in captions:
            caption = captions[image_id]
        else:
            caption = f"Image {image_id}"
            if idx < 10:  # Only warn for first few
                print(f"  Warning: No caption for {image_id}, using placeholder")
        
        manifest.append({
            "image": os.path.abspath(image_file),
            "caption": caption
        })
    
    # Write JSON
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Saved {len(manifest)} image entries to {output_path}")
    return len(manifest)

def main():
    parser = argparse.ArgumentParser(
        description="Prepare datasets for post-training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Text data
  python scripts/prep_post_training_data.py \\
      --input data/raw/my_text.txt \\
      --output data/post_training/text.txt \\
      --format text

  # Audio ASR data
  python scripts/prep_post_training_data.py \\
      --input data/raw/audio/ \\
      --output data/post_training/asr.csv \\
      --format audio_asr

  # Image data with captions
  python scripts/prep_post_training_data.py \\
      --input data/raw/images/ \\
      --output data/post_training/images.json \\
      --format images \\
      --caption_file data/raw/captions.txt
        """
    )
    
    parser.add_argument(
        "--input",
        required=True,
        help="Input path (file for text, directory for audio/images)"
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output path (file path where prepared data will be saved)"
    )
    parser.add_argument(
        "--format",
        required=True,
        choices=["text", "audio_asr", "audio_tts", "images"],
        help="Data format: text, audio_asr, audio_tts, or images"
    )
    parser.add_argument(
        "--sample_rate",
        type=int,
        default=16000,
        help="Sample rate for audio (default: 16000)"
    )
    parser.add_argument(
        "--caption_file",
        type=str,
        help="Path to caption file for images (one caption per line)"
    )
    parser.add_argument(
        "--encoding",
        type=str,
        default="utf-8",
        help="Text file encoding (default: utf-8)"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Post-Training Data Preparation")
    print("=" * 60)
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    print(f"Format: {args.format}")
    print("=" * 60)
    print()
    
    try:
        if args.format == "text":
            count = prepare_text_data(args.input, args.output, args.encoding)
        elif args.format == "audio_asr":
            count = prepare_audio_asr_data(args.input, args.output, args.sample_rate)
        elif args.format == "audio_tts":
            count = prepare_audio_tts_data(args.input, args.output, args.sample_rate)
        elif args.format == "images":
            count = prepare_image_data(args.input, args.output, args.caption_file)
        
        print()
        print("=" * 60)
        print(f"✓ Successfully prepared {count:,} entries")
        print(f"✓ Output saved to: {args.output}")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())

