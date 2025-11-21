"""
Test script for μOmni multimodal inference interface
Tests all media types: text, image, audio, and video
Picks random samples from actual data folders for testing
"""

import os
import subprocess
import sys
import random
import gc
from pathlib import Path

def run_inference(cmd_args, description):
    """Run inference command and display results"""
    print(f"\n{'='*60}")
    print(f"Testing: {description}")
    print(f"{'='*60}")
    print(f"Command: python infer_chat.py {' '.join(cmd_args)}")
    print("-" * 60)
    
    try:
        result = subprocess.run(
            [sys.executable, "infer_chat.py"] + cmd_args,
            capture_output=True,
            text=True,
            timeout=120
        )
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        
        # Force garbage collection after subprocess to free memory
        gc.collect()
        
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        print("ERROR: Command timed out")
        gc.collect()
        return False
    except Exception as e:
        print(f"ERROR: {e}")
        gc.collect()
        return False

def find_random_file(directory, extensions, recursive=True, max_scan=10000):
    """Find a random file with given extensions in directory (memory-efficient)"""
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

def main():
    """Run all media type tests"""
    print("μOmni Multimodal Inference Test Suite")
    print("=" * 60)
    print("Picking random samples from data folders...")
    
    # Find random samples from actual data folders
    print("\nScanning data folders for test samples...")
    print("  (Using memory-efficient scanning, limiting to 10k files per directory)")
    
    # Find random image (limit scan to prevent memory issues)
    sample_image = find_random_file(
        "data/images",
        [".jpg", ".jpeg", ".png", ".JPEG", ".PNG"],
        recursive=True,
        max_scan=10000
    )
    if sample_image:
        print(f"  ✓ Found image: {sample_image}")
    else:
        print("  ⚠ No images found in data/images/")
    
    # Force cleanup after file scanning
    gc.collect()
    
    # Find random audio file (limit scan to prevent memory issues)
    sample_audio = find_random_file(
        "data/audio",
        [".wav", ".flac", ".WAV", ".FLAC"],
        recursive=True,
        max_scan=10000
    )
    if sample_audio:
        print(f"  ✓ Found audio: {sample_audio}")
    else:
        print("  ⚠ No audio files found in data/audio/")
    
    # Force cleanup after file scanning
    gc.collect()
    
    # Get random text sample (limit lines read to prevent memory issues)
    sample_text = get_random_text_sample("data/text", max_lines_to_read=1000)
    print(f"  ✓ Using text sample: {sample_text[:50]}...")
    
    # Force cleanup
    gc.collect()
    
    # Ensure examples directory exists for output files
    examples_dir = Path("examples")
    if not examples_dir.exists():
        examples_dir.mkdir()
    
    # Test results
    results = []
    
    # Test 1: Text only (text output) - using random text sample
    print("\n[TEST 1] Text-only inference (text output)")
    test_text = sample_text[:100] if len(sample_text) > 100 else sample_text
    results.append(("Text-only", run_inference(
        ["--ckpt_dir", "checkpoints/thinker_tiny", "--text", test_text],
        f"Text-only chat (sample: {test_text[:30]}...)"
    )))
    
    # Test 1b: Text with audio output (TTS) - using random text sample
    print("\n[TEST 1b] Text with audio output (TTS)")
    audio_out_path = "examples/test_output_text_tts.wav"
    tts_text = sample_text[:150] if len(sample_text) > 150 else sample_text
    results.append(("Text+TTS", run_inference(
        ["--ckpt_dir", "checkpoints/omni_sft_tiny", 
         "--text", tts_text,
         "--audio_out", audio_out_path],
        f"Text input with audio output (TTS) (sample: {tts_text[:30]}...)"
    )))
    
    # Test 2: Image + Text - using random image from data
    print("\n[TEST 2] Image + Text")
    if sample_image and os.path.exists(sample_image):
        results.append(("Image+Text", run_inference(
            ["--ckpt_dir", "checkpoints/omni_sft_tiny", 
             "--image", sample_image, 
             "--text", "What do you see in this image?"],
            f"Image with text prompt (using: {os.path.basename(sample_image)})"
        )))
    else:
        print(f"SKIP: No image files found in data/images/")
        results.append(("Image+Text", None))
    
    # Test 3: Audio + Text - using random audio from data
    print("\n[TEST 3] Audio + Text")
    if sample_audio and os.path.exists(sample_audio):
        results.append(("Audio+Text", run_inference(
            ["--ckpt_dir", "checkpoints/omni_sft_tiny",
             "--audio_in", sample_audio,
             "--text", "What did you hear?"],
            f"Audio with text prompt (using: {os.path.basename(sample_audio)})"
        )))
    else:
        print(f"SKIP: No audio files found in data/audio/")
        results.append(("Audio+Text", None))
    
    # Test 4: Image only (default prompt) - using random image
    print("\n[TEST 4] Image only")
    if sample_image and os.path.exists(sample_image):
        results.append(("Image-only", run_inference(
            ["--ckpt_dir", "checkpoints/omni_sft_tiny",
             "--image", sample_image],
            f"Image with default prompt (using: {os.path.basename(sample_image)})"
        )))
    else:
        print(f"SKIP: No image files found in data/images/")
        results.append(("Image-only", None))
    
    # Test 5: Audio only - using random audio
    print("\n[TEST 5] Audio only")
    if sample_audio and os.path.exists(sample_audio):
        results.append(("Audio-only", run_inference(
            ["--ckpt_dir", "checkpoints/omni_sft_tiny",
             "--audio_in", sample_audio],
            f"Audio with default prompt (using: {os.path.basename(sample_audio)})"
        )))
    else:
        print(f"SKIP: No audio files found in data/audio/")
        results.append(("Audio-only", None))
    
    # Test 6: Image + Audio + Text (multimodal) - using random samples
    print("\n[TEST 6] Image + Audio + Text (Multimodal)")
    if sample_image and sample_audio and os.path.exists(sample_image) and os.path.exists(sample_audio):
        results.append(("Multimodal", run_inference(
            ["--ckpt_dir", "checkpoints/omni_sft_tiny",
             "--image", sample_image,
             "--audio_in", sample_audio,
             "--text", "Describe what you see and hear."],
            f"Combined image, audio, and text (using random samples from data/)"
        )))
    else:
        missing = []
        if not sample_image or not os.path.exists(sample_image):
            missing.append("image")
        if not sample_audio or not os.path.exists(sample_audio):
            missing.append("audio")
        print(f"SKIP: Missing {', '.join(missing)} files in data/")
        results.append(("Multimodal", None))
    
    # Test 6b: Image + Text with audio output - using random image
    print("\n[TEST 6b] Image + Text with audio output")
    if sample_image and os.path.exists(sample_image):
        audio_out_path = "examples/test_output_image_tts.wav"
        results.append(("Image+Text+TTS", run_inference(
            ["--ckpt_dir", "checkpoints/omni_sft_tiny",
             "--image", sample_image,
             "--text", "Describe this image.",
             "--audio_out", audio_out_path],
            f"Image input with text and audio output (using: {os.path.basename(sample_image)})"
        )))
    else:
        print(f"SKIP: No image files found in data/images/")
        results.append(("Image+Text+TTS", None))
    
    # Test 8: OCR (Optical Character Recognition) - extract text from image
    print("\n[TEST 8] OCR - Extract text from image")
    if sample_image and os.path.exists(sample_image):
        results.append(("OCR", run_inference(
            ["--ckpt_dir", "checkpoints/ocr_tiny",
             "--image", sample_image,
             "--ocr"],
            f"OCR text extraction from image (using: {os.path.basename(sample_image)})"
        )))
    else:
        print(f"SKIP: No image files found in data/images/")
        results.append(("OCR", None))
    
    # Test 8b: OCR + Text prompt - extract text and describe
    print("\n[TEST 8b] OCR + Text prompt")
    if sample_image and os.path.exists(sample_image):
        results.append(("OCR+Text", run_inference(
            ["--ckpt_dir", "checkpoints/omni_sft_tiny",
             "--image", sample_image,
             "--text", "What text do you see in this image?",
             "--ocr"],
            f"OCR with text prompt (using: {os.path.basename(sample_image)})"
        )))
    else:
        print(f"SKIP: No image files found in data/images/")
        results.append(("OCR+Text", None))
    
    # Test 7: Video (if available) - check data/images for video or create from images
    print("\n[TEST 7] Video")
    sample_video = find_random_file("data/images", [".mp4", ".avi", ".mov"], recursive=True, max_scan=1000)
    if not sample_video:
        # Try examples as fallback
        sample_video = "examples/sample_video.mp4"
    
    if sample_video and os.path.exists(sample_video):
        results.append(("Video", run_inference(
            ["--ckpt_dir", "checkpoints/omni_sft_tiny",
             "--video", sample_video,
             "--text", "Describe the video."],
            f"Video with text prompt (using: {os.path.basename(sample_video)})"
        )))
    else:
        print(f"SKIP: No video files found")
        print("      Note: Create a test video from images with:")
        print("      ffmpeg -framerate 1 -pattern_type glob -i 'data/images/**/*.jpg' -c:v libx264 -pix_fmt yuv420p examples/sample_video.mp4")
        results.append(("Video", None))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    passed = 0
    failed = 0
    skipped = 0
    
    for test_name, result in results:
        if result is True:
            status = "✓ PASSED"
            passed += 1
        elif result is False:
            status = "✗ FAILED"
            failed += 1
        else:
            status = "- SKIPPED"
            skipped += 1
        print(f"{test_name:20s} {status}")
    
    print("-" * 60)
    print(f"Total: {len(results)} | Passed: {passed} | Failed: {failed} | Skipped: {skipped}")
    print("=" * 60)
    
    return failed == 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

