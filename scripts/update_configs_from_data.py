"""
Update training configs based on actual dataset sizes
Analyzes data folders and adjusts epochs, max_steps, warmup, and other parameters
"""

import os
import json
import csv
import io
import tempfile
import argparse
from typing import Dict, Tuple, Optional

# Import tokenizer (required)
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from omni.tokenizer import BPETokenizer

# Import model size calculation functions
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)
from calculate_model_size import (
    calculate_thinker_params,
    calculate_audio_encoder_params,
    calculate_vision_encoder_params,
    calculate_talker_params,
    calculate_ocr_params
)

# Best practices for training duration based on dataset size
# (samples, min_epochs, max_epochs, recommended_epochs)
EPOCH_RECOMMENDATIONS = [
    (1000000, 1, 3, 2),      # Very large (>1M): 1-3 epochs
    (500000, 2, 4, 3),       # Large (500K-1M): 2-4 epochs
    (100000, 3, 6, 4),       # Medium (100K-500K): 3-6 epochs
    (50000, 5, 10, 7),       # Small (50K-100K): 5-10 epochs
    (0, 10, 20, 15),         # Very small (<50K): 10-20 epochs
]

def count_text_samples(text_path: str) -> int:
    """Count lines in text file"""
    if not os.path.exists(text_path):
        return 0
    count = 0
    try:
        with open(text_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                if line.strip():
                    count += 1
    except Exception as e:
        print(f"Warning: Could not count text samples from {text_path}: {e}")
    return count

def stream_text_file(text_path: str, chunk_size_mb: int = 100) -> str:
    """
    Stream entire text file in chunks to a temporary file (memory efficient).
    Processes the entire corpus without loading it all into memory.
    Returns path to temporary file with all data.
    """
    import tempfile
    temp_file = tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', delete=False, suffix='.txt')
    temp_path = temp_file.name
    temp_file.close()
    
    lines_read = 0
    
    try:
        with open(text_path, 'rb') as infile, open(temp_path, 'w', encoding='utf-8') as outfile:
            # Read file in chunks to avoid loading entire file into memory
            while True:
                line_bytes = infile.readline()
                if not line_bytes:
                    break
                
                try:
                    line = line_bytes.decode('utf-8')
                    if line.strip():
                        outfile.write(line)
                        lines_read += 1
                        
                        # Progress indicator for large files
                        if lines_read % 100000 == 0:
                            print(f"  Streaming corpus: {lines_read:,} lines processed...")
                except UnicodeDecodeError:
                    pass  # Skip invalid UTF-8 lines
    except Exception as e:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise
    
    print(f"  Streamed entire corpus: {lines_read:,} lines")
    return temp_path

def get_or_create_tokenizer(text_path: str, tokenizer_path: Optional[str] = None) -> BPETokenizer:
    """Get existing tokenizer or create one from text data (streams entire corpus in chunks)"""
    # Try to find existing tokenizer
    tokenizer_candidates = [
        tokenizer_path,  # Explicitly provided
        "checkpoints/thinker_tiny/tokenizer.model",  # Default location
        "tokenizer.model",  # Current directory
    ]
    
    for candidate in tokenizer_candidates:
        if candidate and os.path.exists(candidate):
            return BPETokenizer(candidate)
    
    # No tokenizer found - create one from text data
    if not os.path.exists(text_path):
        raise FileNotFoundError(f"Cannot create tokenizer: text file not found: {text_path}")
    
    print(f"  No tokenizer found. Creating tokenizer from {text_path} (streaming entire corpus in chunks)...")
    os.makedirs("checkpoints/thinker_tiny", exist_ok=True)
    tokenizer_model = "checkpoints/thinker_tiny/tokenizer.model"
    
    # Stream entire corpus in chunks instead of loading entire file
    temp_streamed = None
    try:
        temp_streamed = stream_text_file(text_path, chunk_size_mb=100)
        print(f"  Training tokenizer on entire corpus...")
        BPETokenizer.train_new(temp_streamed, tokenizer_model, vocab_size=32000)
        print(f"  ✓ Tokenizer created: {tokenizer_model}")
    finally:
        if temp_streamed and os.path.exists(temp_streamed):
            os.remove(temp_streamed)
    
    return BPETokenizer(tokenizer_model)

def count_text_tokens(text_path: str, tokenizer_path: Optional[str] = None) -> int:
    """Count actual tokens in text file using tokenizer (streams line-by-line, resumable)"""
    if not os.path.exists(text_path):
        return 0
    
    try:
        tokenizer = get_or_create_tokenizer(text_path, tokenizer_path)
        
        # Check for existing count checkpoint (resumable)
        checkpoint_path = f"{text_path}.token_count_checkpoint.json"
        total_tokens = 0
        start_line_idx = 0
        
        if os.path.exists(checkpoint_path):
            try:
                checkpoint_data = json.load(open(checkpoint_path, 'r'))
                total_tokens = checkpoint_data.get("total_tokens", 0)
                start_line_idx = checkpoint_data.get("last_processed_line", 0)
                print(f"  Resuming token count from line {start_line_idx:,} (already counted {total_tokens:,} tokens)...")
            except Exception as e:
                print(f"  Warning: Could not load checkpoint: {e}, starting from beginning")
                total_tokens = 0
                start_line_idx = 0
        
        # Build line offset index (same approach as train_text.py)
        line_offsets = []
        with open(text_path, 'rb') as f:
            offset = 0
            while True:
                line_start = offset
                line_bytes = f.readline()
                if not line_bytes:
                    break
                try:
                    decoded = line_bytes.decode('utf-8')
                    if decoded.strip():
                        line_offsets.append(line_start)
                except UnicodeDecodeError:
                    pass  # Skip invalid UTF-8 lines
                offset += len(line_bytes)  # Manually track offset
        
        sample_count = start_line_idx
        checkpoint_freq = 10000  # Save checkpoint every N lines
        
        # Process lines one at a time using offsets (memory efficient)
        with open(text_path, 'rb') as f:
            for line_idx, line_start in enumerate(line_offsets[start_line_idx:], start_line_idx):
                f.seek(line_start)
                line_bytes = f.readline()
                text = line_bytes.decode('utf-8').strip()
                if text:
                    tokens = tokenizer.encode(text)
                    total_tokens += len(tokens)
                    sample_count += 1
                    
                    # Save checkpoint periodically (resumable)
                    if sample_count % checkpoint_freq == 0:
                        try:
                            checkpoint_data = {
                                "total_tokens": total_tokens,
                                "last_processed_line": line_idx + 1
                            }
                            json.dump(checkpoint_data, open(checkpoint_path, 'w'))
                        except Exception:
                            pass
                    
                    # Progress indicator for large files
                    if sample_count % 10000 == 0:
                        print(f"  Counting tokens: {sample_count:,} samples, {total_tokens:,} tokens so far...")
        
        # Clean up checkpoint after successful completion
        if os.path.exists(checkpoint_path):
            try:
                os.remove(checkpoint_path)
            except:
                pass
        
        return total_tokens
    except Exception as e:
        print(f"Error: Could not count tokens from {text_path}: {e}")
        raise

def count_csv_tokens(csv_path: str, text_column: str = 'text', tokenizer_path: Optional[str] = None) -> int:
    """Count tokens from CSV file (streams row-by-row like train_audio_enc.py)"""
    if not os.path.exists(csv_path):
        return 0
    
    try:
        # Get tokenizer (use first text file we can find, or create from CSV text)
        text_files = [
            "data/text/production_corpus.txt",
            "data/text/tiny_corpus.txt",
        ]
        tokenizer = None
        for tf in text_files:
            if os.path.exists(tf):
                tokenizer = get_or_create_tokenizer(tf, tokenizer_path)
                break
        
        if not tokenizer:
            # Create tokenizer from CSV text (stream through file)
            print(f"  Creating tokenizer from CSV text...")
            # Use proper tempfile for security and cleanup
            temp_text_file = tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', delete=False, suffix='.txt')
            temp_text = temp_text_file.name
            temp_text_file.close()
            
            temp_text_created = False
            try:
                # Build row offset index first
                row_offsets = []
                fieldnames = None
                with open(csv_path, 'rb') as f:
                    header_line = f.readline()
                    if not header_line:
                        raise ValueError("CSV file is empty")
                    fieldnames = header_line.decode('utf-8').strip().split(',')
                    offset = f.tell()
                    while True:
                        line_start = offset
                        line_bytes = f.readline()
                        if not line_bytes:
                            break
                        try:
                            decoded = line_bytes.decode('utf-8').strip()
                            if decoded:
                                row_offsets.append(line_start)
                        except UnicodeDecodeError:
                            pass
                        offset += len(line_bytes)
                
                # Stream through rows to extract text (entire dataset)
                rows_written = 0
                with open(temp_text, 'w', encoding='utf-8') as out:
                    with open(csv_path, 'rb') as f:
                        for row_start in row_offsets:
                            f.seek(row_start)
                            line_bytes = f.readline()
                            line = line_bytes.decode('utf-8').strip()
                            reader = csv.DictReader(io.StringIO(line), fieldnames=fieldnames)
                            row = next(reader)
                            text = row.get(text_column, '').strip()
                            if text:
                                out.write(text + '\n')
                                rows_written += 1
                                
                                # Progress indicator
                                if rows_written % 100000 == 0:
                                    print(f"  Extracting text: {rows_written:,} rows...")
                
                temp_text_created = True
                
                # Stream the temp file in chunks for tokenizer training
                temp_streamed = None
                try:
                    print(f"  Streaming {rows_written:,} rows for tokenizer training...")
                    temp_streamed = stream_text_file(temp_text, chunk_size_mb=100)
                    os.makedirs("checkpoints/thinker_tiny", exist_ok=True)
                    tokenizer_model = "checkpoints/thinker_tiny/tokenizer.model"
                    print(f"  Training tokenizer on entire dataset...")
                    BPETokenizer.train_new(temp_streamed, tokenizer_model, vocab_size=32000)
                    tokenizer = BPETokenizer(tokenizer_model)
                finally:
                    if temp_streamed and os.path.exists(temp_streamed):
                        os.remove(temp_streamed)
            finally:
                # Clean up temp text file
                if temp_text_created and os.path.exists(temp_text):
                    os.remove(temp_text)
        
        # Build row offset index (same approach as train_audio_enc.py)
        row_offsets = []
        fieldnames = None
        with open(csv_path, 'rb') as f:
            header_line = f.readline()
            if not header_line:
                raise ValueError("CSV file is empty")
            fieldnames = header_line.decode('utf-8').strip().split(',')
            offset = f.tell()
            while True:
                line_start = offset
                line_bytes = f.readline()
                if not line_bytes:
                    break
                try:
                    decoded = line_bytes.decode('utf-8').strip()
                    if decoded:
                        row_offsets.append(line_start)
                except UnicodeDecodeError:
                    pass
                offset += len(line_bytes)  # Manually track offset
        
        # Check for existing count checkpoint (resumable)
        checkpoint_path = f"{csv_path}.token_count_checkpoint.json"
        total_tokens = 0
        start_row_idx = 0
        
        if os.path.exists(checkpoint_path):
            try:
                checkpoint_data = json.load(open(checkpoint_path, 'r'))
                total_tokens = checkpoint_data.get("total_tokens", 0)
                start_row_idx = checkpoint_data.get("last_processed_row", 0)
                print(f"  Resuming token count from row {start_row_idx:,} (already counted {total_tokens:,} tokens)...")
            except Exception as e:
                print(f"  Warning: Could not load checkpoint: {e}, starting from beginning")
                total_tokens = 0
                start_row_idx = 0
        
        sample_count = start_row_idx
        checkpoint_freq = 10000  # Save checkpoint every N rows
        
        # Process rows one at a time using offsets (memory efficient)
        with open(csv_path, 'rb') as f:
            for row_idx, row_start in enumerate(row_offsets[start_row_idx:], start_row_idx):
                f.seek(row_start)
                line_bytes = f.readline()
                line = line_bytes.decode('utf-8').strip()
                # Parse CSV row properly (handles quoted fields)
                reader = csv.DictReader(io.StringIO(line), fieldnames=fieldnames)
                row = next(reader)
                text = row.get(text_column, '').strip()
                if text:
                    tokens = tokenizer.encode(text)
                    total_tokens += len(tokens)
                    sample_count += 1
                    
                    # Save checkpoint periodically (resumable)
                    if sample_count % checkpoint_freq == 0:
                        try:
                            checkpoint_data = {
                                "total_tokens": total_tokens,
                                "last_processed_row": row_idx + 1
                            }
                            json.dump(checkpoint_data, open(checkpoint_path, 'w'))
                        except Exception:
                            pass
                    
                    if sample_count % 10000 == 0:
                        print(f"  Counting tokens: {sample_count:,} samples, {total_tokens:,} tokens so far...")
        
        # Clean up checkpoint after successful completion
        if os.path.exists(checkpoint_path):
            try:
                os.remove(checkpoint_path)
            except:
                pass
        
        return total_tokens
    except Exception as e:
        print(f"Error: Could not count tokens from {csv_path}: {e}")
        raise

def count_audio_tokens(csv_path: str, tokenizer_path: Optional[str] = None) -> int:
    """
    Count tokens from audio transcriptions (text column in CSV).
    
    Note: This counts tokens from the transcription text in the CSV file,
    NOT from the actual audio waveforms. For audio training, training steps
    are calculated based on transcription tokens (used in CTC loss).
    
    The actual audio files are processed as mel spectrograms by the audio encoder,
    but training step calculation uses transcription tokens.
    """
    return count_csv_tokens(csv_path, 'text', tokenizer_path)

def count_ocr_tokens(csv_path: str, tokenizer_path: Optional[str] = None) -> int:
    """Count tokens from OCR text (text column in CSV)"""
    return count_csv_tokens(csv_path, 'text', tokenizer_path)

def _build_json_offset_index(manifest_path: str) -> Tuple[list, list, bool]:
    """Build an index of JSON object offsets without loading entire file into memory.
    Returns (item_offsets, item_lengths, is_dict_format) where is_dict_format indicates
    if the JSON is a dict with 'images' key rather than a direct array."""
    item_offsets = []
    item_lengths = []
    is_dict_format = False
    
    try:
        with open(manifest_path, 'rb') as f:
            content = f.read()
            
            # Find array start
            start_pos = content.find(b'[')
            if start_pos == -1:
                # Try dict format with 'images' key
                dict_start = content.find(b'{')
                if dict_start == -1:
                    raise ValueError("JSON manifest must be an array or dict")
                # For dict format, find 'images' array
                images_key_pos = content.find(b'"images"')
                if images_key_pos == -1:
                    raise ValueError("JSON dict must have 'images' key")
                # Find array after 'images' key
                start_pos = content.find(b'[', images_key_pos)
                if start_pos == -1:
                    raise ValueError("Could not find images array")
                is_dict_format = True
            
            # Parse JSON to find object boundaries
            pos = start_pos + 1
            depth = 0
            obj_start = None
            in_string = False
            escape_next = False
            
            while pos < len(content):
                byte = content[pos:pos+1]
                if escape_next:
                    escape_next = False
                elif byte == b'\\':
                    escape_next = True
                elif byte == b'"' and not escape_next:
                    in_string = not in_string
                elif not in_string:
                    if byte == b'{':
                        if depth == 0:
                            obj_start = pos
                        depth += 1
                    elif byte == b'}':
                        depth -= 1
                        if depth == 0 and obj_start is not None:
                            # Found complete object
                            item_offsets.append(obj_start)
                            item_lengths.append(pos - obj_start + 1)
                            obj_start = None
                    elif byte == b']' and depth == 0:
                        # End of array
                        break
                pos += 1
    except Exception as e:
        # If parsing fails, return empty index (will use fallback)
        return [], [], False
    
    return item_offsets, item_lengths, is_dict_format

def _stream_json_items(manifest_path: str):
    """Stream JSON items from a file without loading entire file into memory.
    Uses offset index to read items one at a time."""
    item_offsets, item_lengths, is_dict_format = _build_json_offset_index(manifest_path)
    
    if not item_offsets:
        # Fallback: load entire JSON (for malformed JSON or small files)
        with open(manifest_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            items = data if isinstance(data, list) else data.get('images', [])
            for item in items:
                yield item
        return
    
    # Stream through items using offsets
    with open(manifest_path, 'rb') as f:
        for offset, length in zip(item_offsets, item_lengths):
            f.seek(offset)
            obj_bytes = f.read(length)
            try:
                obj = json.loads(obj_bytes.decode('utf-8'))
                yield obj
            except json.JSONDecodeError:
                # Try reading a bit more if parsing fails (handles trailing commas)
                chunk = obj_bytes
                while True:
                    next_byte = f.read(1)
                    if not next_byte or next_byte in [b'{', b']']:
                        break
                    chunk += next_byte
                # Try to find and parse the JSON object in the chunk
                chunk_str = chunk.decode('utf-8')
                start = chunk_str.find('{')
                if start != -1:
                    depth = 0
                    end = start
                    for j, char in enumerate(chunk_str[start:], start):
                        if char == '{':
                            depth += 1
                        elif char == '}':
                            depth -= 1
                            if depth == 0:
                                end = j + 1
                                break
                    try:
                        obj = json.loads(chunk_str[start:end])
                        yield obj
                    except json.JSONDecodeError:
                        pass  # Skip malformed objects

def count_image_tokens(manifest_path: str, tokenizer_path: Optional[str] = None) -> int:
    """
    Count tokens from image captions (text descriptions).
    
    Note: This counts tokens from the caption text in the JSON manifest,
    NOT from the actual image pixels. For vision training, training steps
    are calculated based on caption tokens (used in contrastive learning).
    
    The actual image files are processed as embeddings by the vision encoder,
    but training step calculation uses caption tokens.
    """
    if not os.path.exists(manifest_path):
        return 0
    
    try:
        # Get tokenizer (use first text file we can find, or create from captions)
        text_files = [
            "data/text/production_corpus.txt",
            "data/text/tiny_corpus.txt",
        ]
        tokenizer = None
        for tf in text_files:
            if os.path.exists(tf):
                tokenizer = get_or_create_tokenizer(tf, tokenizer_path)
                break
        
        if not tokenizer:
            # Create tokenizer from image captions (stream through file, entire dataset)
            print(f"  Creating tokenizer from image captions (streaming entire dataset in chunks)...")
            # Use proper tempfile for security and cleanup
            temp_text_file = tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', delete=False, suffix='.txt')
            temp_text = temp_text_file.name
            temp_text_file.close()
            
            temp_text_created = False
            try:
                captions_written = 0
                with open(temp_text, 'w', encoding='utf-8') as out:
                    for item in _stream_json_items(manifest_path):
                        caption = item.get('caption', '').strip()
                        if caption:
                            out.write(caption + '\n')
                            captions_written += 1
                            
                            # Progress indicator
                            if captions_written % 100000 == 0:
                                print(f"  Extracting captions: {captions_written:,} captions...")
                
                temp_text_created = True
                
                # Stream the temp file in chunks for tokenizer training
                temp_streamed = None
                try:
                    print(f"  Streaming {captions_written:,} captions for tokenizer training...")
                    temp_streamed = stream_text_file(temp_text, chunk_size_mb=100)
                    os.makedirs("checkpoints/thinker_tiny", exist_ok=True)
                    tokenizer_model = "checkpoints/thinker_tiny/tokenizer.model"
                    print(f"  Training tokenizer on entire dataset...")
                    BPETokenizer.train_new(temp_streamed, tokenizer_model, vocab_size=32000)
                    tokenizer = BPETokenizer(tokenizer_model)
                finally:
                    if temp_streamed and os.path.exists(temp_streamed):
                        os.remove(temp_streamed)
            finally:
                # Clean up temp text file
                if temp_text_created and os.path.exists(temp_text):
                    os.remove(temp_text)
        
        # Check for existing count checkpoint (resumable)
        checkpoint_path = f"{manifest_path}.token_count_checkpoint.json"
        total_tokens = 0
        start_sample_count = 0
        
        if os.path.exists(checkpoint_path):
            try:
                checkpoint_data = json.load(open(checkpoint_path, 'r'))
                total_tokens = checkpoint_data.get("total_tokens", 0)
                start_sample_count = checkpoint_data.get("last_processed_sample", 0)
                print(f"  Resuming token count from sample {start_sample_count:,} (already counted {total_tokens:,} tokens)...")
            except Exception as e:
                print(f"  Warning: Could not load checkpoint: {e}, starting from beginning")
                total_tokens = 0
                start_sample_count = 0
        
        sample_count = start_sample_count
        checkpoint_freq = 10000  # Save checkpoint every N samples
        items_processed = 0
        
        # Stream through JSON items without loading entire file
        # Note: For streaming JSON, we can't efficiently skip to a specific index,
        # so we'll process from the beginning but use checkpoint to track progress
        for item in _stream_json_items(manifest_path):
            # Skip items until we reach where we left off
            if items_processed < start_sample_count:
                items_processed += 1
                continue
                
            caption = item.get('caption', '').strip()
            if caption:
                tokens = tokenizer.encode(caption)
                total_tokens += len(tokens)
                sample_count += 1
                items_processed += 1
                
                # Save checkpoint periodically (resumable)
                if sample_count % checkpoint_freq == 0:
                    try:
                        checkpoint_data = {
                            "total_tokens": total_tokens,
                            "last_processed_sample": sample_count
                        }
                        json.dump(checkpoint_data, open(checkpoint_path, 'w'))
                    except Exception:
                        pass
                
                if sample_count % 10000 == 0:
                    print(f"  Counting tokens: {sample_count:,} samples, {total_tokens:,} tokens so far...")
        
        # Clean up checkpoint after successful completion
        if os.path.exists(checkpoint_path):
            try:
                os.remove(checkpoint_path)
            except:
                pass
        
        return total_tokens
    except Exception as e:
        print(f"Error: Could not count tokens from {manifest_path}: {e}")
        raise

def count_image_samples(manifest_path: str) -> int:
    """Count entries in image JSON manifest using streaming parsing"""
    if not os.path.exists(manifest_path):
        return 0
    try:
        # Use offset index to count without loading entire file
        item_offsets, _, _ = _build_json_offset_index(manifest_path)
        if item_offsets:
            return len(item_offsets)
        # Fallback: load entire JSON if offset indexing failed
        with open(manifest_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if isinstance(data, list):
                return len(data)
            elif isinstance(data, dict) and 'images' in data:
                return len(data['images'])
            return 0
    except Exception as e:
        print(f"Warning: Could not count image samples from {manifest_path}: {e}")
        return 0

def count_audio_samples(csv_path: str) -> int:
    """Count rows in audio CSV file"""
    if not os.path.exists(csv_path):
        return 0
    count = 0
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            count = sum(1 for _ in reader)
    except Exception as e:
        print(f"Warning: Could not count audio samples from {csv_path}: {e}")
        return 0
    return count

def get_recommended_epochs(num_samples: int) -> Tuple[int, int, int]:
    """Get recommended epoch range based on dataset size"""
    for threshold, min_epochs, max_epochs, recommended in EPOCH_RECOMMENDATIONS:
        if num_samples >= threshold:
            return min_epochs, max_epochs, recommended
    return 10, 20, 15  # Default for very small datasets

def estimate_training_memory(model_params: int, ctx_len: int, batch_size: int) -> float:
    """
    Estimate training memory requirements in GB (float32).
    
    Formula based on research:
    - Model parameters: params × 4 bytes (float32)
    - Optimizer (AdamW): params × 8 bytes (momentum + variance)
    - Gradients: params × 4 bytes
    - Activations: ~batch_size × ctx_len × d_model × layers (approximate)
    
    Args:
        model_params: Number of model parameters
        ctx_len: Context length (sequence length)
        batch_size: Batch size
    
    Returns:
        Estimated memory in GB
    """
    # Base memory: model + optimizer + gradients
    # Research shows: ~6x model size for full training (model + optimizer + gradients)
    base_memory_gb = (model_params * 6 * 4) / (1024 ** 3)  # 6x for AdamW optimizer
    
    # Activation memory (approximate, depends on architecture)
    # Rough estimate: batch_size × ctx_len × hidden_dim × layers × 4 bytes
    # For simplicity, use a multiplier based on context length
    activation_multiplier = 1.0 + (ctx_len / 512) * 0.5  # Scale with context length
    activation_memory_gb = base_memory_gb * activation_multiplier * 0.3  # ~30% for activations
    
    total_memory_gb = base_memory_gb + activation_memory_gb
    return total_memory_gb

def get_optimal_batch_size_and_grad_accum(
    model_params: int, 
    base_batch_size: int = 8, 
    base_grad_accum: int = 1,
    ctx_len: int = 512
) -> Tuple[int, int]:
    """
    Calculate optimal batch size and gradient accumulation based on model size.
    
    Based on research formulas:
    - Effective Batch Size (EBS) = Micro Batch Size (MBS) × Gradient Accumulation (GA) × Data Parallel (DP)
    - Larger models require smaller micro-batch sizes to fit in memory
    - Gradient accumulation maintains effective batch size while reducing memory
    
    Strategy:
    - Estimate memory requirements
    - Adjust micro-batch size based on model size
    - Increase gradient accumulation to maintain effective batch size
    
    Args:
        model_params: Number of model parameters
        base_batch_size: Base micro-batch size for small models
        base_grad_accum: Base gradient accumulation steps
        ctx_len: Context length (affects memory)
    
    Returns:
        (micro_batch_size, gradient_accumulation_steps)
    """
    # Estimate memory requirements
    estimated_memory_gb = estimate_training_memory(model_params, ctx_len, base_batch_size)
    
    # Model size categories (in parameters)
    # Based on research: adjust batch size to fit in typical GPU memory (8-24GB)
    # Very large (>100M): smaller batch, more accumulation
    # Large (50M-100M): medium batch, some accumulation
    # Medium (10M-50M): normal batch, minimal accumulation
    # Small (<10M): larger batch, no accumulation needed
    
    if model_params >= 100_000_000:  # >100M params
        # Very large models: reduce batch size significantly
        batch_size = max(1, base_batch_size // 4)
        grad_accum = base_grad_accum * 4
    elif model_params >= 50_000_000:  # 50M-100M params
        # Large models: reduce batch size moderately
        batch_size = max(2, base_batch_size // 2)
        grad_accum = base_grad_accum * 2
    elif model_params >= 10_000_000:  # 10M-50M params
        # Medium models: use base batch size
        batch_size = max(4, base_batch_size)
        grad_accum = base_grad_accum
    else:  # <10M params
        # Small models: can use larger batch sizes
        batch_size = base_batch_size
        grad_accum = base_grad_accum
    
    # Ensure effective batch size is maintained
    effective_batch = batch_size * grad_accum
    base_effective_batch = base_batch_size * base_grad_accum
    
    # If effective batch size dropped too much, increase gradient accumulation
    if effective_batch < base_effective_batch * 0.75:
        grad_accum = int(base_effective_batch / batch_size)
        grad_accum = max(grad_accum, base_grad_accum)
    
    return batch_size, grad_accum

def format_model_size(num_params: int) -> str:
    """Format model parameter count in readable format"""
    if num_params >= 1_000_000_000:
        return f"{num_params / 1_000_000_000:.2f}B"
    elif num_params >= 1_000_000:
        return f"{num_params / 1_000_000:.2f}M"
    elif num_params >= 1_000:
        return f"{num_params / 1_000:.2f}K"
    else:
        return str(num_params)

def calculate_training_params(
    num_samples: int,
    batch_size: int,
    gradient_accumulation: int = 1,
    val_split: float = 0.1
) -> Dict:
    """Calculate training parameters based on dataset size"""
    # Effective batch size
    effective_batch = batch_size * gradient_accumulation
    
    # Training samples (excluding validation)
    train_samples = int(num_samples * (1 - val_split))
    
    # Steps per epoch
    steps_per_epoch = max(1, train_samples // effective_batch)
    
    # Get recommended epochs
    min_epochs, max_epochs, recommended_epochs = get_recommended_epochs(num_samples)
    
    # Calculate max_steps (use recommended epochs)
    max_steps = steps_per_epoch * recommended_epochs
    
    # Warmup steps: 5-10% of total steps, but at least 100, max 10% of max_steps
    warmup_steps = max(100, min(int(max_steps * 0.1), int(max_steps * 0.05)))
    warmup_steps = min(warmup_steps, 10000)  # Cap at 10K
    
    # Validation frequency: every 500-1000 steps, or 10% of steps per epoch
    val_freq = max(100, min(1000, steps_per_epoch // 10))
    
    # Checkpoint frequency: every 5000-10000 steps, or 1 per epoch
    checkpoint_freq = max(1000, min(10000, steps_per_epoch))
    
    return {
        "num_samples": num_samples,
        "train_samples": train_samples,
        "steps_per_epoch": steps_per_epoch,
        "min_epochs": min_epochs,
        "max_epochs": max_epochs,
        "recommended_epochs": recommended_epochs,
        "max_steps": max_steps,
        "warmup_steps": warmup_steps,
        "val_freq": val_freq,
        "checkpoint_freq": checkpoint_freq,
    }

def update_data_paths(config: Dict, config_path: str = "") -> Dict:
    """Update data paths to use production or synthetic files if they exist"""
    # Text paths - only production or synthetic
    if "train_text" in config:
        if not os.path.exists(config["train_text"]):
            # Try production first, then synthetic
            if os.path.exists("data/text/production_corpus.txt"):
                print(f"  → Updating train_text: {config['train_text']} → data/text/production_corpus.txt")
                config["train_text"] = "data/text/production_corpus.txt"
            elif os.path.exists("data/text/tiny_corpus.txt"):
                print(f"  → Updating train_text: {config['train_text']} → data/text/tiny_corpus.txt")
                config["train_text"] = "data/text/tiny_corpus.txt"
    
    # Image paths - only production or synthetic
    if "train_manifest" in config:
        if not os.path.exists(config["train_manifest"]):
            # Try production first, then synthetic
            if os.path.exists("data/images/production_annotations.json"):
                print(f"  → Updating train_manifest: {config['train_manifest']} → data/images/production_annotations.json")
                config["train_manifest"] = "data/images/production_annotations.json"
            elif os.path.exists("data/images/annotations.json"):
                print(f"  → Updating train_manifest: {config['train_manifest']} → data/images/annotations.json")
                config["train_manifest"] = "data/images/annotations.json"
    
    # Audio ASR paths - only production or synthetic
    if "train_csv" in config:
        if not os.path.exists(config["train_csv"]):
            # Try production first, then synthetic
            if os.path.exists("data/audio/production_asr.csv"):
                print(f"  → Updating train_csv: {config['train_csv']} → data/audio/production_asr.csv")
                config["train_csv"] = "data/audio/production_asr.csv"
            elif os.path.exists("data/audio/asr.csv"):
                print(f"  → Updating train_csv: {config['train_csv']} → data/audio/asr.csv")
                config["train_csv"] = "data/audio/asr.csv"
    
    # Audio TTS paths - only production or synthetic
    if "tts_csv" in config:
        if not os.path.exists(config["tts_csv"]):
            # Try production first, then synthetic
            if os.path.exists("data/audio/production_tts.csv"):
                print(f"  → Updating tts_csv: {config['tts_csv']} → data/audio/production_tts.csv")
                config["tts_csv"] = "data/audio/production_tts.csv"
            elif os.path.exists("data/audio/tts.csv"):
                print(f"  → Updating tts_csv: {config['tts_csv']} → data/audio/tts.csv")
                config["tts_csv"] = "data/audio/tts.csv"
    
    # OCR paths - only production or synthetic
    # Check if this is an OCR config by checking config_path or save_dir
    is_ocr_config = False
    if config_path:
        is_ocr_config = "ocr" in config_path.lower()
    elif "save_dir" in config:
        is_ocr_config = "ocr" in config["save_dir"].lower()
    
    if is_ocr_config and "train_csv" in config:
        if not os.path.exists(config["train_csv"]):
            # Try production first, then synthetic
            if os.path.exists("data/ocr/production_ocr.csv"):
                print(f"  → Updating train_csv: {config['train_csv']} → data/ocr/production_ocr.csv")
                config["train_csv"] = "data/ocr/production_ocr.csv"
            elif os.path.exists("data/ocr/ocr_train.csv"):
                print(f"  → Updating train_csv: {config['train_csv']} → data/ocr/ocr_train.csv")
                config["train_csv"] = "data/ocr/ocr_train.csv"
    
    # Multimodal paths - only production or synthetic
    if "sft_mix" in config:
        sft = config["sft_mix"]
        if "text_path" in sft:
            if not os.path.exists(sft["text_path"]):
                # Try production first, then synthetic
                if os.path.exists("data/text/production_corpus.txt"):
                    print(f"  → Updating sft_mix.text_path: {sft['text_path']} → data/text/production_corpus.txt")
                    sft["text_path"] = "data/text/production_corpus.txt"
                elif os.path.exists("data/text/tiny_corpus.txt"):
                    print(f"  → Updating sft_mix.text_path: {sft['text_path']} → data/text/tiny_corpus.txt")
                    sft["text_path"] = "data/text/tiny_corpus.txt"
        if "image_manifest" in sft:
            if not os.path.exists(sft["image_manifest"]):
                # Try production first, then synthetic
                if os.path.exists("data/images/production_annotations.json"):
                    print(f"  → Updating sft_mix.image_manifest: {sft['image_manifest']} → data/images/production_annotations.json")
                    sft["image_manifest"] = "data/images/production_annotations.json"
                elif os.path.exists("data/images/annotations.json"):
                    print(f"  → Updating sft_mix.image_manifest: {sft['image_manifest']} → data/images/annotations.json")
                    sft["image_manifest"] = "data/images/annotations.json"
        if "asr_csv" in sft:
            if not os.path.exists(sft["asr_csv"]):
                # Try production first, then synthetic
                if os.path.exists("data/audio/production_asr.csv"):
                    print(f"  → Updating sft_mix.asr_csv: {sft['asr_csv']} → data/audio/production_asr.csv")
                    sft["asr_csv"] = "data/audio/production_asr.csv"
                elif os.path.exists("data/audio/asr.csv"):
                    print(f"  → Updating sft_mix.asr_csv: {sft['asr_csv']} → data/audio/asr.csv")
                    sft["asr_csv"] = "data/audio/asr.csv"
    
    return config

def update_config_file(
    config_path: str,
    params: Dict,
    preserve_keys: Optional[list] = None,
    update_paths: bool = True
):
    """Update config file with calculated parameters"""
    if not os.path.exists(config_path):
        print(f"Warning: Config file not found: {config_path}")
        return False
    
    # Load existing config
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Preserve certain keys if specified
    if preserve_keys:
        preserved = {k: config.get(k) for k in preserve_keys if k in config}
    else:
        preserved = {}
    
    # Update data paths if requested
    if update_paths:
        # Pass config_path to update_data_paths for OCR detection
        config = update_data_paths(config, config_path)
    
    # Update training parameters
    config["max_steps"] = params["max_steps"]
    config["warmup_steps"] = params["warmup_steps"]
    config["max_epochs"] = params["recommended_epochs"]
    config["val_freq"] = params["val_freq"]
    config["checkpoint_freq"] = params["checkpoint_freq"]
    
    # Update batch size and gradient accumulation if calculated from model size
    if "batch_size" in params:
        config["batch_size"] = params["batch_size"]
    if "gradient_accumulation_steps" in params:
        config["gradient_accumulation_steps"] = params["gradient_accumulation_steps"]
    
    # Restore preserved keys (but keep updated paths)
    for key in preserved:
        if key in config and isinstance(config[key], dict) and isinstance(preserved[key], dict):
            # Merge dicts (e.g., sft_mix)
            config[key].update(preserved[key])
        else:
            config[key] = preserved[key]
    
    # Save updated config
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"✓ Updated {config_path}")
    return True

def analyze_and_update_all_configs(dry_run: bool = False):
    """Analyze all datasets and update all config files"""
    print("="*60)
    print("Analyzing Datasets and Updating Configs")
    print("="*60)
    
    # Analyze text dataset - only production and synthetic files
    print("\n[1] Analyzing Text Dataset...")
    text_files = [
        "data/text/production_corpus.txt",  # Production file
        "data/text/tiny_corpus.txt",        # Synthetic file from make_synthetic_datasets.py
    ]
    text_samples = 0
    text_tokens = 0
    text_path = None
    for tf in text_files:
        if os.path.exists(tf):
            count = count_text_samples(tf)
            if count > text_samples:
                text_samples = count
                text_path = tf
    
    # Count tokens (required)
    if text_path:
        print(f"  Counting tokens (this may take a while for large files)...")
        text_tokens = count_text_tokens(text_path)
        avg_tokens_per_sample = text_tokens / text_samples if text_samples > 0 else 0
        print(f"  Text samples: {text_samples:,} (from {text_path})")
        print(f"  Text tokens: {text_tokens:,} (~{text_tokens/1_000_000:.1f}M tokens)")
        print(f"  Average tokens per sample: {avg_tokens_per_sample:.1f}")
    else:
        print(f"  ⚠ No text data found")
    
    # Analyze image dataset - only production and synthetic files
    print("\n[2] Analyzing Image Dataset...")
    image_files = [
        "data/images/production_annotations.json",  # Production file
        "data/images/annotations.json",           # Synthetic file from make_synthetic_datasets.py
    ]
    image_samples = 0
    image_tokens = 0
    image_manifest = None
    for imf in image_files:
        if os.path.exists(imf):
            count = count_image_samples(imf)
            if count > image_samples:
                image_samples = count
                image_manifest = imf
    
    # Count tokens from captions (required)
    # Note: We count tokens from caption text, not from image pixels.
    # Training steps are based on caption tokens (used in contrastive learning).
    if image_manifest:
        print(f"  Counting tokens from captions (this may take a while for large files)...")
        image_tokens = count_image_tokens(image_manifest)
        avg_tokens_per_sample = image_tokens / image_samples if image_samples > 0 else 0
        print(f"  Image samples: {image_samples:,} (from {image_manifest})")
        print(f"  Caption tokens: {image_tokens:,} (~{image_tokens/1_000_000:.1f}M tokens)")
        print(f"  Average tokens per caption: {avg_tokens_per_sample:.1f}")
        print(f"  Note: Tokens counted from caption text, not image pixels")
    else:
        print(f"  ⚠ No image data found")
    
    # Analyze audio dataset - only production and synthetic files
    print("\n[3] Analyzing Audio Dataset...")
    audio_files = [
        "data/audio/production_asr.csv",  # Production file
        "data/audio/asr.csv",             # Synthetic file from make_synthetic_datasets.py
    ]
    audio_samples = 0
    audio_tokens = 0
    audio_csv = None
    for af in audio_files:
        if os.path.exists(af):
            count = count_audio_samples(af)
            if count > audio_samples:
                audio_samples = count
                audio_csv = af
    
    # Count tokens from transcriptions (required)
    # Note: We count tokens from transcription text, not from audio waveforms.
    # Training steps are based on transcription tokens (used in CTC loss).
    if audio_csv:
        print(f"  Counting tokens from transcriptions (this may take a while for large files)...")
        audio_tokens = count_audio_tokens(audio_csv)
        avg_tokens_per_sample = audio_tokens / audio_samples if audio_samples > 0 else 0
        print(f"  Audio samples: {audio_samples:,} (from {audio_csv})")
        print(f"  Transcription tokens: {audio_tokens:,} (~{audio_tokens/1_000_000:.1f}M tokens)")
        print(f"  Average tokens per transcription: {avg_tokens_per_sample:.1f}")
        print(f"  Note: Tokens counted from transcription text, not audio waveforms")
    else:
        print(f"  ⚠ No audio data found")
    
    # Calculate parameters for each training stage
    print("\n" + "="*60)
    print("Calculating Training Parameters")
    print("="*60)
    
    # Helper function to calculate params from samples (for vision/audio training)
    def calculate_params_from_samples(num_samples: int, batch_size: int, gradient_accumulation: int = 1, val_split: float = 0.1):
        """
        Calculate training parameters from sample count.
        
        Used for vision and audio training where steps are based on samples, not tokens.
        - Vision training: Contrastive learning (image-caption pairs) - steps = samples / batch_size
        - Audio training: CTC loss (audio-transcription pairs) - steps = samples / batch_size
        
        Based on research formulas:
        - Effective Batch Size (EBS) = Micro Batch Size (MBS) × Gradient Accumulation (GA)
        - Steps per epoch = Training Samples / Effective Batch Size
        - Total steps = Steps per epoch × Number of epochs
        
        Args:
            num_samples: Total number of training samples
            batch_size: Micro batch size (per device)
            gradient_accumulation: Gradient accumulation steps
            val_split: Validation split ratio (default 0.1 = 10%)
        
        Returns:
            Dictionary with calculated training parameters
        """
        # Effective batch size formula: EBS = MBS × GA × DP
        effective_batch = batch_size * gradient_accumulation
        
        # Training samples (excluding validation)
        train_samples = int(num_samples * (1 - val_split))
        
        # Steps per epoch
        # Formula: steps_per_epoch = training_samples / effective_batch_size
        steps_per_epoch = max(1, train_samples // effective_batch)
        
        # Epoch recommendations based on sample count
        # Research: larger datasets need fewer epochs, smaller datasets need more
        if num_samples >= 1_000_000:  # >1M samples
            min_epochs, max_epochs, recommended_epochs = 1, 3, 2
        elif num_samples >= 500_000:  # 500K-1M samples
            min_epochs, max_epochs, recommended_epochs = 2, 4, 3
        elif num_samples >= 100_000:  # 100K-500K samples
            min_epochs, max_epochs, recommended_epochs = 3, 6, 4
        else:  # <100K samples
            min_epochs, max_epochs, recommended_epochs = 5, 10, 7
        
        # Total training steps
        max_steps = steps_per_epoch * recommended_epochs
        
        # Warmup steps calculation (3-5% of total steps)
        warmup_percentage = 0.04  # 4%
        warmup_steps = max(100, int(max_steps * warmup_percentage))
        warmup_steps = min(warmup_steps, 10000)  # Cap at 10K
        
        # Validation frequency
        val_freq = max(100, min(1000, steps_per_epoch // 10))
        
        # Checkpoint frequency
        checkpoint_freq = max(1000, min(10000, steps_per_epoch))
        
        return {
            "num_samples": num_samples,
            "train_samples": train_samples,
            "effective_batch_size": effective_batch,
            "steps_per_epoch": steps_per_epoch,
            "min_epochs": min_epochs,
            "max_epochs": max_epochs,
            "recommended_epochs": recommended_epochs,
            "max_steps": max_steps,
            "warmup_steps": warmup_steps,
            "warmup_percentage": warmup_steps / max_steps if max_steps > 0 else 0,
            "val_freq": val_freq,
            "checkpoint_freq": checkpoint_freq,
        }
    
    # Helper function to calculate params from tokens (for text training)
    def calculate_params_from_tokens(num_tokens: int, batch_size: int, ctx_len: int, gradient_accumulation: int = 1, val_split: float = 0.1):
        """
        Calculate training parameters from token count.
        
        Based on research formulas:
        - Effective Batch Size (EBS) = Micro Batch Size (MBS) × Gradient Accumulation (GA)
        - Tokens per step = EBS × Context Length
        - Steps per epoch = Training Tokens / Tokens per step
        - Total steps = Steps per epoch × Number of epochs
        
        Warmup steps: Research shows 1-10% of total steps, typically 3-5%
        - Too little warmup: unstable training
        - Too much warmup: slower convergence
        
        Args:
            num_tokens: Total number of training tokens
            batch_size: Micro batch size (per device)
            ctx_len: Context length (sequence length)
            gradient_accumulation: Gradient accumulation steps
            val_split: Validation split ratio (default 0.1 = 10%)
        
        Returns:
            Dictionary with calculated training parameters
        """
        # Effective batch size formula: EBS = MBS × GA × DP
        # (DP = Data Parallel replicas, assumed 1 for single GPU)
        effective_batch = batch_size * gradient_accumulation
        
        # Training tokens (excluding validation)
        train_tokens = int(num_tokens * (1 - val_split))
        
        # Tokens processed per training step
        # Formula: tokens_per_step = effective_batch_size × context_length
        tokens_per_step = effective_batch * ctx_len
        
        # Steps per epoch
        # Formula: steps_per_epoch = training_tokens / tokens_per_step
        steps_per_epoch = max(1, train_tokens // tokens_per_step)
        
        # Epoch recommendations based on token count
        # Research: larger datasets need fewer epochs, smaller datasets need more
        # Based on neural scaling laws and empirical observations
        if num_tokens >= 100_000_000:  # >100M tokens
            min_epochs, max_epochs, recommended_epochs = 1, 3, 2
        elif num_tokens >= 50_000_000:  # 50M-100M tokens
            min_epochs, max_epochs, recommended_epochs = 2, 4, 3
        elif num_tokens >= 10_000_000:  # 10M-50M tokens
            min_epochs, max_epochs, recommended_epochs = 3, 6, 4
        else:  # <10M tokens
            min_epochs, max_epochs, recommended_epochs = 5, 10, 7
        
        # Total training steps
        # Formula: max_steps = steps_per_epoch × recommended_epochs
        max_steps = steps_per_epoch * recommended_epochs
        
        # Warmup steps calculation
        # Research: typically 3-5% of total steps, range 1-10%
        # Minimum 100 steps, maximum 10K steps (to avoid excessive warmup)
        warmup_percentage = 0.04  # 4% (middle of 3-5% range)
        warmup_steps = max(100, int(max_steps * warmup_percentage))
        warmup_steps = min(warmup_steps, 10000)  # Cap at 10K
        
        # Validation frequency
        # Validate every 10% of steps per epoch, but at least every 100 steps
        # and at most every 1000 steps
        val_freq = max(100, min(1000, steps_per_epoch // 10))
        
        # Checkpoint frequency
        # Checkpoint every epoch, but at least every 1000 steps
        # and at most every 10000 steps
        checkpoint_freq = max(1000, min(10000, steps_per_epoch))
        
        return {
            "num_tokens": num_tokens,
            "train_tokens": train_tokens,
            "effective_batch_size": effective_batch,
            "tokens_per_step": tokens_per_step,
            "steps_per_epoch": steps_per_epoch,
            "min_epochs": min_epochs,
            "max_epochs": max_epochs,
            "recommended_epochs": recommended_epochs,
            "max_steps": max_steps,
            "warmup_steps": warmup_steps,
            "warmup_percentage": warmup_steps / max_steps if max_steps > 0 else 0,
            "val_freq": val_freq,
            "checkpoint_freq": checkpoint_freq,
        }
    
    # Stage A: Text-only (thinker_tiny.json)
    print("\n[Stage A] Text-only Training (thinker_tiny.json)")
    if text_tokens > 0:
        # Load config to get ctx_len and model config
        config_path = "configs/thinker_tiny.json"
        ctx_len = 512  # Default
        model_params = 0
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                cfg = json.load(f)
                ctx_len = cfg.get("ctx_len", 512)
                # Calculate model size
                model_params = calculate_thinker_params(
                    vocab_size=cfg.get("vocab_size", 32000),
                    n_layers=cfg.get("n_layers", 4),
                    d_model=cfg.get("d_model", 256),
                    n_heads=cfg.get("n_heads", 4),
                    d_ff=cfg.get("d_ff", 1024),
                    use_gqa=cfg.get("use_gqa", False),
                    kv_groups=cfg.get("kv_groups", 2),
                    use_swiglu=cfg.get("use_swiglu", True)
                )
        
        # Adjust batch size and gradient accumulation based on model size
        # Formula: Effective Batch Size = Micro Batch Size × Gradient Accumulation
        batch_size, grad_accum = get_optimal_batch_size_and_grad_accum(
            model_params, base_batch_size=8, base_grad_accum=1, ctx_len=ctx_len
        )
        text_params = calculate_params_from_tokens(text_tokens, batch_size=batch_size, ctx_len=ctx_len, gradient_accumulation=grad_accum)
        text_params["num_samples"] = text_samples  # Keep for reference
        text_params["model_params"] = model_params  # Keep for reference
        text_params["batch_size"] = batch_size  # Store for config update
        text_params["gradient_accumulation_steps"] = grad_accum  # Store for config update
        
        print(f"  Model size: {model_params:,} params ({format_model_size(model_params)})")
        effective_batch = text_params.get('effective_batch_size', batch_size * grad_accum)
        print(f"  Batch size: {batch_size} (gradient accumulation: {grad_accum}, effective: {effective_batch})")
        print(f"  Tokens: {text_params['num_tokens']:,} (~{text_params['num_tokens']/1_000_000:.1f}M tokens)")
        print(f"  Tokens/step: {text_params.get('tokens_per_step', effective_batch * ctx_len):,}")
        print(f"  Steps/epoch: {text_params['steps_per_epoch']:,}")
        print(f"  Recommended epochs: {text_params['recommended_epochs']}")
        print(f"  Max steps: {text_params['max_steps']:,}")
        warmup_pct = text_params.get('warmup_percentage', 0) * 100
        print(f"  Warmup steps: {text_params['warmup_steps']:,} ({warmup_pct:.1f}% of total)")
        
        if not dry_run:
            update_config_file(
                config_path,
                text_params,
                preserve_keys=["train_text"]  # Preserve data path
            )
    else:
        print("  ⚠ No text data found, skipping...")
    
    # Stage B: Audio encoder (audio_enc_tiny.json)
    print("\n[Stage B] Audio Encoder Training (audio_enc_tiny.json)")
    if audio_tokens > 0:
        # Audio encoder processes transcription tokens
        # Use ctx_len from config or default
        config_path = "configs/audio_enc_tiny.json"
        ctx_len = 256  # Default for audio (shorter sequences)
        model_params = 0
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                cfg = json.load(f)
                ctx_len = cfg.get("ctx_len", 256)
                # Calculate model size
                model_params = calculate_audio_encoder_params(
                    d_model=cfg.get("d_model", 128),
                    n_layers=cfg.get("n_layers", 4),
                    n_heads=cfg.get("n_heads", 2),
                    d_ff=cfg.get("d_ff", 512),
                    downsample_factor=cfg.get("downsample_time", 8)
                )
        
        # Adjust batch size and gradient accumulation based on model size
        # Note: Audio training uses samples, not tokens, for step calculation
        batch_size, grad_accum = get_optimal_batch_size_and_grad_accum(
            model_params, base_batch_size=4, base_grad_accum=1, ctx_len=256  # Use default ctx_len for memory estimation
        )
        # Audio training: steps = samples / batch_size (not tokens / (batch_size * ctx_len))
        audio_params = calculate_params_from_samples(audio_samples, batch_size=batch_size, gradient_accumulation=grad_accum)
        audio_params["num_tokens"] = audio_tokens  # Keep token count for reference
        audio_params["num_samples"] = audio_samples  # Keep for reference
        audio_params["model_params"] = model_params  # Keep for reference
        audio_params["batch_size"] = batch_size  # Store for config update
        audio_params["gradient_accumulation_steps"] = grad_accum  # Store for config update
        
        print(f"  Model size: {model_params:,} params ({format_model_size(model_params)})")
        effective_batch = audio_params.get('effective_batch_size', batch_size * grad_accum)
        print(f"  Batch size: {batch_size} (gradient accumulation: {grad_accum}, effective: {effective_batch})")
        print(f"  Samples: {audio_params['num_samples']:,} (transcription tokens: {audio_params.get('num_tokens', 0):,})")
        print(f"  Steps/epoch: {audio_params['steps_per_epoch']:,} (based on samples, not tokens)")
        print(f"  Recommended epochs: {audio_params['recommended_epochs']}")
        print(f"  Max steps: {audio_params['max_steps']:,}")
        warmup_pct = audio_params.get('warmup_percentage', 0) * 100
        print(f"  Warmup steps: {audio_params['warmup_steps']:,} ({warmup_pct:.1f}% of total)")
        
        if not dry_run:
            update_config_file(
                config_path,
                audio_params,
                preserve_keys=["train_csv"]  # Preserve data path
            )
    else:
        print("  ⚠ No audio data found, skipping...")
    
    # Stage C: Vision encoder (vision_tiny.json)
    print("\n[Stage C] Vision Encoder Training (vision_tiny.json)")
    if image_tokens > 0:
        # Vision encoder processes caption tokens
        config_path = "configs/vision_tiny.json"
        ctx_len = 128  # Default for captions (shorter sequences)
        model_params = 0
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                cfg = json.load(f)
                ctx_len = cfg.get("ctx_len", 128)
                # Calculate model size
                model_params = calculate_vision_encoder_params(
                    img_size=cfg.get("img_size", 224),
                    patch=cfg.get("patch", 16),
                    d_model=cfg.get("d_model", 128),
                    n_layers=cfg.get("n_layers", 4),
                    n_heads=cfg.get("n_heads", 2),
                    d_ff=cfg.get("d_ff", 512)
                )
        
        # Adjust batch size and gradient accumulation based on model size
        # Note: Vision training uses samples, not tokens, for step calculation
        batch_size, grad_accum = get_optimal_batch_size_and_grad_accum(
            model_params, base_batch_size=8, base_grad_accum=1, ctx_len=224  # Use default ctx_len for memory estimation
        )
        # Vision training: steps = samples / batch_size (not tokens / (batch_size * ctx_len))
        vision_params = calculate_params_from_samples(image_samples, batch_size=batch_size, gradient_accumulation=grad_accum)
        vision_params["num_tokens"] = image_tokens  # Keep token count for reference
        vision_params["num_samples"] = image_samples  # Keep for reference
        vision_params["model_params"] = model_params  # Keep for reference
        vision_params["batch_size"] = batch_size  # Store for config update
        vision_params["gradient_accumulation_steps"] = grad_accum  # Store for config update
        
        print(f"  Model size: {model_params:,} params ({format_model_size(model_params)})")
        effective_batch = vision_params.get('effective_batch_size', batch_size * grad_accum)
        print(f"  Batch size: {batch_size} (gradient accumulation: {grad_accum}, effective: {effective_batch})")
        print(f"  Samples: {vision_params['num_samples']:,} (caption tokens: {vision_params.get('num_tokens', 0):,})")
        print(f"  Steps/epoch: {vision_params['steps_per_epoch']:,} (based on samples, not tokens)")
        print(f"  Recommended epochs: {vision_params['recommended_epochs']}")
        print(f"  Max steps: {vision_params['max_steps']:,}")
        warmup_pct = vision_params.get('warmup_percentage', 0) * 100
        print(f"  Warmup steps: {vision_params['warmup_steps']:,} ({warmup_pct:.1f}% of total)")
        
        if not dry_run:
            update_config_file(
                config_path,
                vision_params,
                preserve_keys=["train_manifest", "image_root"]  # Preserve data paths
            )
    else:
        print("  ⚠ No image data found, skipping...")
    
    # Stage D: Talker (talker_tiny.json) - uses TTS data
    print("\n[Stage D] Talker Training (talker_tiny.json)")
    # Only check production and synthetic TTS files
    tts_files = [
        "data/audio/production_tts.csv",  # Production file
        "data/audio/tts.csv",             # Synthetic file from make_synthetic_datasets.py
    ]
    tts_samples = 0
    tts_tokens = 0
    tts_csv = None
    for ttf in tts_files:
        if os.path.exists(ttf):
            count = count_audio_samples(ttf)
            if count > tts_samples:
                tts_samples = count
                tts_csv = ttf
    
    # Count tokens from TTS transcriptions
    if tts_csv:
        print(f"  Counting tokens from TTS transcriptions...")
        tts_tokens = count_audio_tokens(tts_csv)
    
    if tts_tokens > 0:
        config_path = "configs/talker_tiny.json"
        ctx_len = 256  # Default for TTS
        model_params = 0
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                cfg = json.load(f)
                ctx_len = cfg.get("ctx_len", 256)
                # Calculate model size (Talker only, RVQ is separate)
                model_params = calculate_talker_params(
                    d_model=cfg.get("d_model", 256),
                    n_layers=cfg.get("n_layers", 4),
                    n_heads=cfg.get("n_heads", 4),
                    d_ff=cfg.get("d_ff", 1024),
                    codebooks=cfg.get("codebooks", 2),
                    codebook_size=cfg.get("codebook_size", 128),
                    use_gqa=cfg.get("use_gqa", False),
                    kv_groups=cfg.get("kv_groups", 1),
                    use_swiglu=cfg.get("use_swiglu", True)
                )
        
        # Adjust batch size and gradient accumulation based on model size
        # Note: Talker training uses samples, not tokens, for step calculation
        batch_size, grad_accum = get_optimal_batch_size_and_grad_accum(
            model_params, base_batch_size=4, base_grad_accum=1, ctx_len=256  # Use default ctx_len for memory estimation
        )
        # Talker training: steps = samples / batch_size (not tokens / (batch_size * ctx_len))
        talker_params = calculate_params_from_samples(tts_samples, batch_size=batch_size, gradient_accumulation=grad_accum)
        talker_params["num_tokens"] = tts_tokens  # Keep token count for reference
        talker_params["num_samples"] = tts_samples  # Keep for reference
        talker_params["model_params"] = model_params  # Keep for reference
        talker_params["batch_size"] = batch_size  # Store for config update
        talker_params["gradient_accumulation_steps"] = grad_accum  # Store for config update
        
        print(f"  Model size: {model_params:,} params ({format_model_size(model_params)})")
        effective_batch = talker_params.get('effective_batch_size', batch_size * grad_accum)
        print(f"  Batch size: {batch_size} (gradient accumulation: {grad_accum}, effective: {effective_batch})")
        print(f"  Samples: {talker_params['num_samples']:,} (transcription tokens: {talker_params.get('num_tokens', 0):,})")
        print(f"  Steps/epoch: {talker_params['steps_per_epoch']:,} (based on samples, not tokens)")
        print(f"  Recommended epochs: {talker_params['recommended_epochs']}")
        print(f"  Max steps: {talker_params['max_steps']:,}")
        warmup_pct = talker_params.get('warmup_percentage', 0) * 100
        print(f"  Warmup steps: {talker_params['warmup_steps']:,} ({warmup_pct:.1f}% of total)")
        
        if not dry_run:
            update_config_file(
                config_path,
                talker_params,
                preserve_keys=["tts_csv"]  # Preserve data path
            )
    else:
        print("  ⚠ No TTS data found, skipping...")
    
    # Stage E: Multimodal SFT (omni_sft_tiny.json)
    print("\n[Stage E] Multimodal SFT Training (omni_sft_tiny.json)")
    # Use the maximum of all token counts for multimodal training
    multimodal_tokens = max(text_tokens, image_tokens, audio_tokens)
    if multimodal_tokens > 0:
        config_path = "configs/omni_sft_tiny.json"
        ctx_len = 512  # Default for multimodal
        model_params = 0
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                cfg = json.load(f)
                ctx_len = cfg.get("ctx_len", 512)
                # Calculate total model size (Thinker + projectors)
                # Load Thinker config to calculate its size
                thinker_cfg_path = "configs/thinker_tiny.json"
                if os.path.exists(thinker_cfg_path):
                    with open(thinker_cfg_path, 'r') as tf:
                        thinker_cfg = json.load(tf)
                        thinker_params = calculate_thinker_params(
                            vocab_size=thinker_cfg.get("vocab_size", 32000),
                            n_layers=thinker_cfg.get("n_layers", 4),
                            d_model=thinker_cfg.get("d_model", 256),
                            n_heads=thinker_cfg.get("n_heads", 4),
                            d_ff=thinker_cfg.get("d_ff", 1024),
                            use_gqa=thinker_cfg.get("use_gqa", False),
                            kv_groups=thinker_cfg.get("kv_groups", 2),
                            use_swiglu=thinker_cfg.get("use_swiglu", True)
                        )
                        # Add projectors (vision: 128->256, audio: 192->256)
                        projector_params = (128 * 256 + 256) + (192 * 256 + 256)
                        model_params = thinker_params + projector_params
        
        # Multimodal training uses smaller batch size and more gradient accumulation
        # Adjust based on model size (SFT is typically larger due to full model)
        batch_size, grad_accum = get_optimal_batch_size_and_grad_accum(
            model_params, base_batch_size=2, base_grad_accum=4, ctx_len=ctx_len
        )
        sft_params = calculate_params_from_tokens(multimodal_tokens, batch_size=batch_size, ctx_len=ctx_len, gradient_accumulation=grad_accum)
        sft_params["num_samples"] = max(text_samples, image_samples, audio_samples)  # Keep for reference
        sft_params["model_params"] = model_params  # Keep for reference
        sft_params["batch_size"] = batch_size  # Store for config update
        sft_params["gradient_accumulation_steps"] = grad_accum  # Store for config update
        
        print(f"  Model size: {model_params:,} params ({format_model_size(model_params)})")
        effective_batch = sft_params.get('effective_batch_size', batch_size * grad_accum)
        print(f"  Batch size: {batch_size} (gradient accumulation: {grad_accum}, effective: {effective_batch})")
        print(f"  Tokens (max across modalities): {sft_params['num_tokens']:,} (~{sft_params['num_tokens']/1_000_000:.1f}M tokens)")
        print(f"  Tokens/step: {sft_params.get('tokens_per_step', effective_batch * ctx_len):,}")
        print(f"  Steps/epoch: {sft_params['steps_per_epoch']:,}")
        print(f"  Recommended epochs: {sft_params['recommended_epochs']}")
        print(f"  Max steps: {sft_params['max_steps']:,}")
        warmup_pct = sft_params.get('warmup_percentage', 0) * 100
        print(f"  Warmup steps: {sft_params['warmup_steps']:,} ({warmup_pct:.1f}% of total)")
        
        if not dry_run:
            update_config_file(
                config_path,
                sft_params,
                preserve_keys=["sft_mix"]  # Preserve data paths
            )
    else:
        print("  ⚠ No multimodal data found, skipping...")
    
    # OCR Training (ocr_tiny.json)
    print("\n[OCR] OCR Training (ocr_tiny.json)")
    ocr_files = [
        "data/ocr/production_ocr.csv",  # Production file
        "data/ocr/ocr_train.csv",        # Synthetic file from make_synthetic_datasets.py
    ]
    ocr_samples = 0
    ocr_tokens = 0
    ocr_csv = None
    for ocrf in ocr_files:
        if os.path.exists(ocrf):
            count = count_audio_samples(ocrf)  # OCR uses same CSV format as audio
            if count > ocr_samples:
                ocr_samples = count
                ocr_csv = ocrf
    
    # Count tokens from OCR text
    if ocr_csv:
        print(f"  Counting tokens from OCR text...")
        ocr_tokens = count_ocr_tokens(ocr_csv)
    
    if ocr_tokens > 0:
        config_path = "configs/ocr_tiny.json"
        ctx_len = 128  # Default for OCR (short text sequences)
        model_params = 0
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                cfg = json.load(f)
                ctx_len = cfg.get("ctx_len", 128)
                # Calculate model size
                model_params = calculate_ocr_params(
                    img_size=cfg.get("img_size", 224),
                    patch=cfg.get("patch", 16),
                    vision_d_model=cfg.get("vision_d_model", 128),
                    vision_layers=cfg.get("vision_layers", 4),
                    vision_heads=cfg.get("vision_heads", 2),
                    vision_d_ff=cfg.get("vision_d_ff", 512),
                    decoder_d_model=cfg.get("decoder_d_model", 256),
                    decoder_layers=cfg.get("decoder_layers", 4),
                    decoder_heads=cfg.get("decoder_heads", 4),
                    decoder_d_ff=cfg.get("decoder_d_ff", 1024),
                    vocab_size=cfg.get("vocab_size", 128)
                )
        
        # Adjust batch size and gradient accumulation based on model size
        # Note: OCR training uses samples, not tokens, for step calculation
        batch_size, grad_accum = get_optimal_batch_size_and_grad_accum(
            model_params, base_batch_size=4, base_grad_accum=2, ctx_len=128  # Use default ctx_len for memory estimation
        )
        # OCR training: steps = samples / batch_size (not tokens / (batch_size * ctx_len))
        ocr_params = calculate_params_from_samples(ocr_samples, batch_size=batch_size, gradient_accumulation=grad_accum)
        ocr_params["num_tokens"] = ocr_tokens  # Keep token count for reference
        ocr_params["num_samples"] = ocr_samples  # Keep for reference
        ocr_params["model_params"] = model_params  # Keep for reference
        ocr_params["batch_size"] = batch_size  # Store for config update
        ocr_params["gradient_accumulation_steps"] = grad_accum  # Store for config update
        
        print(f"  Model size: {model_params:,} params ({format_model_size(model_params)})")
        effective_batch = ocr_params.get('effective_batch_size', batch_size * grad_accum)
        print(f"  Batch size: {batch_size} (gradient accumulation: {grad_accum}, effective: {effective_batch})")
        print(f"  Samples: {ocr_params['num_samples']:,} (text tokens: {ocr_params.get('num_tokens', 0):,})")
        print(f"  Steps/epoch: {ocr_params['steps_per_epoch']:,} (based on samples, not tokens)")
        print(f"  Recommended epochs: {ocr_params['recommended_epochs']}")
        print(f"  Max steps: {ocr_params['max_steps']:,}")
        warmup_pct = ocr_params.get('warmup_percentage', 0) * 100
        print(f"  Warmup steps: {ocr_params['warmup_steps']:,} ({warmup_pct:.1f}% of total)")
        
        if not dry_run:
            update_config_file(
                config_path,
                ocr_params,
                preserve_keys=["train_csv", "image_root"],  # Preserve data paths
                update_paths=True
            )
    else:
        print("  ⚠ No OCR data found, skipping...")
    
    print("\n" + "="*60)
    if dry_run:
        print("DRY RUN: No files were modified")
        print("Run without --dry-run to apply changes")
    else:
        print("✓ All configs updated successfully!")
    print("="*60)
    
    # Print summary
    print("\nSummary:")
    print(f"  Text tokens: {text_tokens:,} (~{text_tokens/1_000_000:.1f}M)")
    print(f"  Image caption tokens: {image_tokens:,} (~{image_tokens/1_000_000:.1f}M)")
    print(f"  Audio transcription tokens: {audio_tokens:,} (~{audio_tokens/1_000_000:.1f}M)")
    if ocr_tokens > 0:
        print(f"  OCR tokens: {ocr_tokens:,} (~{ocr_tokens/1_000_000:.1f}M)")
    multimodal_tokens = max(text_tokens, image_tokens, audio_tokens)
    if multimodal_tokens > 0:
        print(f"  Multimodal (max tokens): {multimodal_tokens:,} (~{multimodal_tokens/1_000_000:.1f}M)")

def main():
    parser = argparse.ArgumentParser(
        description="Update training configs based on actual dataset sizes"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be changed without modifying files"
    )
    args = parser.parse_args()
    
    analyze_and_update_all_configs(
        dry_run=args.dry_run
    )

if __name__ == "__main__":
    main()

