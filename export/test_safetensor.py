"""
Test script for μOmni merged safetensors model
Tests all media types using the exported model.safetensors file
Picks random samples from actual data folders for testing
Evaluates on 100 samples per test type and reports metrics
Verifies that the merged safetensors model loads and works correctly
"""

import os
import subprocess
import sys
import random
import gc
import time
import argparse
from pathlib import Path
from collections import defaultdict

def run_inference(cmd_args, description, silent=False):
    """Run inference command and return success status and timing"""
    if not silent:
        print(f"  Running: {description}")
    
    start_time = time.time()
    try:
        result = subprocess.run(
            [sys.executable, "infer_chat.py"] + cmd_args,
            capture_output=True,
            text=True,
            timeout=120,
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Run from project root
        )
        elapsed = time.time() - start_time
        
        # Force garbage collection after subprocess to free memory
        gc.collect()
        
        success = result.returncode == 0
        if not silent and not success:
            if result.stderr:
                print(f"    Error: {result.stderr[:200]}")
        
        return success, elapsed
    except subprocess.TimeoutExpired:
        elapsed = time.time() - start_time
        gc.collect()
        if not silent:
            print("    ERROR: Command timed out")
        return False, elapsed
    except Exception as e:
        elapsed = time.time() - start_time
        gc.collect()
        if not silent:
            print(f"    ERROR: {e}")
        return False, elapsed

def find_random_file(directory, extensions, recursive=True, max_scan=10000):
    """Find a random file with given extensions in directory (memory-efficient fallback)"""
    if not os.path.exists(directory):
        return None
    
    path = Path(directory)
    files = []
    count = 0
    
    # Use iterator approach to avoid loading all files into memory
    if recursive:
        for ext in extensions:
            if count >= max_scan:
                break
            for file_path in path.rglob(f"*{ext}"):
                # Skip hidden files and checkpoints
                if not any(part.startswith('.') for part in file_path.parts):
                    files.append(str(file_path))
                    count += 1
                    # Early exit if we have enough samples
                    if len(files) >= 1000:  # Collect up to 1000, then pick randomly
                        break
                if count >= max_scan:
                    break
    else:
        for ext in extensions:
            if count >= max_scan:
                break
            for file_path in path.glob(f"*{ext}"):
                if not any(part.startswith('.') for part in file_path.parts):
                    files.append(str(file_path))
                    count += 1
                    if len(files) >= 1000:
                        break
                if count >= max_scan:
                    break
    
    if files:
        return random.choice(files)
    return None

def get_random_text_sample(text_dir="data/text", max_lines_to_read=1000):
    """Get a random line from text corpus files (memory-efficient)"""
    if not os.path.exists(text_dir):
        return "Hello, how are you?"
    
    # Find .txt files (limit to first 100 to avoid memory issues)
    text_files = []
    for txt_file in Path(text_dir).glob("*.txt"):
        text_files.append(txt_file)
        if len(text_files) >= 100:
            break
    
    if not text_files:
        return "Hello, how are you?"
    
    # Pick random file
    text_file = random.choice(text_files)
    try:
        # Memory-efficient: read lines one at a time, don't load entire file
        lines = []
        with open(text_file, 'r', encoding='utf-8', errors='ignore') as f:
            for i, line in enumerate(f):
                if i >= max_lines_to_read:
                    break
                line = line.strip()
                if line:
                    lines.append(line)
        
        if lines:
            # Return a random line, but limit length for testing
            line = random.choice(lines)
            return line[:200] if len(line) > 200 else line
    except Exception:
        pass
    
    return "Hello, how are you?"

def collect_samples(directory, extensions, num_samples=100, recursive=True, max_scan=10000):
    """Collect multiple random samples from directory"""
    if not os.path.exists(directory):
        return []
    
    path = Path(directory)
    files = []
    count = 0
    
    # Collect files
    if recursive:
        for ext in extensions:
            if count >= max_scan:
                break
            for file_path in path.rglob(f"*{ext}"):
                if not any(part.startswith('.') for part in file_path.parts):
                    files.append(str(file_path))
                    count += 1
                    if count >= max_scan:
                        break
    else:
        for ext in extensions:
            if count >= max_scan:
                break
            for file_path in path.glob(f"*{ext}"):
                if not any(part.startswith('.') for part in file_path.parts):
                    files.append(str(file_path))
                    count += 1
                    if count >= max_scan:
                        break
    
    # Return random sample of requested size
    if len(files) > num_samples:
        return random.sample(files, num_samples)
    return files

def evaluate_test_type(test_name, cmd_template, sample_getter, num_samples=100):
    """Evaluate a test type on multiple samples and return metrics"""
    print(f"\n{'='*60}")
    print(f"Evaluating: {test_name}")
    print(f"{'='*60}")
    print(f"Testing on {num_samples} samples...")
    
    successes = 0
    failures = 0
    times = []
    
    for i in range(num_samples):
        # Get a random sample for this iteration
        sample = sample_getter()
        if not sample:
            failures += 1
            continue
        
        # Build command with this sample
        cmd_args = cmd_template(sample)
        
        # Run inference (silent mode for batch processing)
        success, elapsed = run_inference(cmd_args, f"Sample {i+1}/{num_samples}", silent=True)
        
        if success:
            successes += 1
        else:
            failures += 1
        times.append(elapsed)
        
        if (i + 1) % 10 == 0:
            print(f"  Processed {i + 1}/{num_samples} samples... (Success: {successes}, Failed: {failures})", end='\r')
    
    print()  # New line after progress
    
    # Calculate metrics
    success_rate = (successes / num_samples * 100) if num_samples > 0 else 0.0
    avg_time = sum(times) / len(times) if times else 0.0
    min_time = min(times) if times else 0.0
    max_time = max(times) if times else 0.0
    
    return {
        'success_rate': success_rate,
        'successes': successes,
        'failures': failures,
        'total': num_samples,
        'avg_time': avg_time,
        'min_time': min_time,
        'max_time': max_time
    }

def verify_export_model():
    """Verify that the exported model exists and is accessible"""
    export_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "export")
    model_file = os.path.join(export_dir, "model.safetensors")
    config_file = os.path.join(export_dir, "config.json")
    
    if not os.path.exists(model_file):
        print(f"ERROR: Model file not found: {model_file}")
        print("Please run export.py first to create the merged model.")
        return False
    
    if not os.path.exists(config_file):
        print(f"ERROR: Config file not found: {config_file}")
        return False
    
    # Check file size
    model_size_mb = os.path.getsize(model_file) / (1024 * 1024)
    print(f"✓ Found exported model: {model_file}")
    print(f"  Model size: {model_size_mb:.2f} MB")
    print(f"✓ Found config file: {config_file}")
    
    return True

def main():
    """Run all media type tests on exported safetensors model"""
    parser = argparse.ArgumentParser(description="Test μOmni merged safetensors model on multiple samples")
    parser.add_argument("--num_samples", type=int, default=100,
                       help="Number of samples to test per test type (default: 100)")
    parser.add_argument("--export_dir", type=str, default="export",
                       help="Directory containing exported model (default: export)")
    args = parser.parse_args()
    
    print("μOmni Merged Safetensors Model Test Suite")
    print("=" * 60)
    print(f"Testing exported model from: {args.export_dir}")
    print(f"Testing on {args.num_samples} samples per test type")
    print("Picking random samples from data folders...")
    
    # Verify export model exists
    if not verify_export_model():
        return False
    
    # Get absolute path to export directory
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    export_dir = os.path.join(project_root, args.export_dir)
    
    # Collect sample pools
    print("\nScanning data folders for test samples...")
    print("  (Using memory-efficient scanning, limiting to 10k files per directory)")
    
    # Collect image samples
    image_samples = collect_samples(
        os.path.join(project_root, "data/images"),
        [".jpg", ".jpeg", ".png", ".JPEG", ".PNG"],
        num_samples=args.num_samples,
        recursive=True,
        max_scan=10000
    )
    print(f"  ✓ Found {len(image_samples)} image samples")
    
    # Collect audio samples
    audio_samples = collect_samples(
        os.path.join(project_root, "data/audio"),
        [".wav", ".flac", ".WAV", ".FLAC"],
        num_samples=args.num_samples,
        recursive=True,
        max_scan=10000
    )
    print(f"  ✓ Found {len(audio_samples)} audio samples")
    
    # Force cleanup
    gc.collect()
    
    # Ensure examples directory exists for output files
    examples_dir = Path(project_root) / "examples"
    if not examples_dir.exists():
        examples_dir.mkdir()
    
    # Test results with metrics
    results = []
    
    # Test 1: Text only
    if True:  # Always available
        def get_text_sample():
            return get_random_text_sample(os.path.join(project_root, "data/text"), max_lines_to_read=1000)
        
        def text_cmd(sample):
            return ["--ckpt_dir", export_dir, "--text", sample[:100]]
        
        metrics = evaluate_test_type("Text-only (merged safetensors)", text_cmd, get_text_sample, args.num_samples)
        results.append(("Text-only", metrics))
    
    # Test 2: Image + Text
    if image_samples:
        image_idx = [0]
        def get_image_sample():
            if image_idx[0] >= len(image_samples):
                image_idx[0] = 0
            img = image_samples[image_idx[0]]
            image_idx[0] += 1
            return img if os.path.exists(img) else None
        
        def image_text_cmd(sample):
            return ["--ckpt_dir", export_dir, 
                   "--image", sample, 
                   "--text", "What do you see in this image?"]
        
        metrics = evaluate_test_type("Image+Text (merged safetensors)", image_text_cmd, get_image_sample, min(args.num_samples, len(image_samples)))
        results.append(("Image+Text", metrics))
    else:
        results.append(("Image+Text", None))
    
    # Test 3: Audio + Text
    if audio_samples:
        audio_idx = [0]
        def get_audio_sample():
            if audio_idx[0] >= len(audio_samples):
                audio_idx[0] = 0
            aud = audio_samples[audio_idx[0]]
            audio_idx[0] += 1
            return aud if os.path.exists(aud) else None
        
        def audio_text_cmd(sample):
            return ["--ckpt_dir", export_dir,
                   "--audio_in", sample,
                   "--text", "What did you hear?"]
        
        metrics = evaluate_test_type("Audio+Text (merged safetensors)", audio_text_cmd, get_audio_sample, min(args.num_samples, len(audio_samples)))
        results.append(("Audio+Text", metrics))
    else:
        results.append(("Audio+Text", None))
    
    # Test 4: Image only
    if image_samples:
        image_idx2 = [0]
        def get_image_sample2():
            if image_idx2[0] >= len(image_samples):
                image_idx2[0] = 0
            img = image_samples[image_idx2[0]]
            image_idx2[0] += 1
            return img if os.path.exists(img) else None
        
        def image_only_cmd(sample):
            return ["--ckpt_dir", export_dir, "--image", sample]
        
        metrics = evaluate_test_type("Image-only (merged safetensors)", image_only_cmd, get_image_sample2, min(args.num_samples, len(image_samples)))
        results.append(("Image-only", metrics))
    else:
        results.append(("Image-only", None))
    
    # Test 5: Audio only
    if audio_samples:
        audio_idx2 = [0]
        def get_audio_sample2():
            if audio_idx2[0] >= len(audio_samples):
                audio_idx2[0] = 0
            aud = audio_samples[audio_idx2[0]]
            audio_idx2[0] += 1
            return aud if os.path.exists(aud) else None
        
        def audio_only_cmd(sample):
            return ["--ckpt_dir", export_dir, "--audio_in", sample]
        
        metrics = evaluate_test_type("Audio-only (merged safetensors)", audio_only_cmd, get_audio_sample2, min(args.num_samples, len(audio_samples)))
        results.append(("Audio-only", metrics))
    else:
        results.append(("Audio-only", None))
    
    # Test 6: OCR
    if image_samples:
        image_idx3 = [0]
        def get_image_sample3():
            if image_idx3[0] >= len(image_samples):
                image_idx3[0] = 0
            img = image_samples[image_idx3[0]]
            image_idx3[0] += 1
            return img if os.path.exists(img) else None
        
        def ocr_cmd(sample):
            return ["--ckpt_dir", export_dir, "--image", sample, "--ocr"]
        
        metrics = evaluate_test_type("OCR (merged safetensors)", ocr_cmd, get_image_sample3, min(args.num_samples, len(image_samples)))
        results.append(("OCR", metrics))
    else:
        results.append(("OCR", None))
    
    # Test 7: Text-to-Speech (TTS)
    if True:  # Always available
        def get_text_sample_tts():
            return get_random_text_sample(os.path.join(project_root, "data/text"), max_lines_to_read=1000)
        
        def tts_cmd(sample):
            output_file = os.path.join(project_root, "examples", f"tts_output_{random.randint(1000, 9999)}.wav")
            return ["--ckpt_dir", export_dir, 
                   "--text", sample[:100],
                   "--audio_out", output_file]
        
        metrics = evaluate_test_type("Text-to-Speech (merged safetensors)", tts_cmd, get_text_sample_tts, min(args.num_samples, 50))  # Limit TTS tests
        results.append(("TTS", metrics))
    
    # Summary
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY - Merged Safetensors Model")
    print("=" * 60)
    
    total_tests = 0
    total_successes = 0
    total_failures = 0
    skipped = 0
    
    for test_name, metrics in results:
        if metrics is None:
            print(f"{test_name:30s} - SKIPPED (no samples available)")
            skipped += 1
        else:
            total_tests += metrics['total']
            total_successes += metrics['successes']
            total_failures += metrics['failures']
            
            status = "✓" if metrics['success_rate'] >= 80 else "⚠" if metrics['success_rate'] >= 50 else "✗"
            print(f"{test_name:30s} {status} {metrics['success_rate']:5.1f}% "
                  f"({metrics['successes']}/{metrics['total']}) | "
                  f"Avg: {metrics['avg_time']:.2f}s | "
                  f"Range: {metrics['min_time']:.2f}s-{metrics['max_time']:.2f}s")
    
    print("-" * 60)
    overall_success_rate = (total_successes / total_tests * 100) if total_tests > 0 else 0.0
    print(f"Overall: {overall_success_rate:.1f}% success rate ({total_successes}/{total_tests} samples)")
    print(f"Skipped: {skipped} test types")
    print("=" * 60)
    
    if overall_success_rate >= 80:
        print("✓ Merged safetensors model is working correctly!")
    elif overall_success_rate >= 50:
        print("⚠ Merged safetensors model has some issues")
    else:
        print("✗ Merged safetensors model has significant issues")
    
    return total_failures == 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

