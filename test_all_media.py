"""
Test script for μOmni multimodal inference interface
Tests all media types: text, image, audio, and video
"""

import os
import subprocess
import sys
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
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        print("ERROR: Command timed out")
        return False
    except Exception as e:
        print(f"ERROR: {e}")
        return False

def main():
    """Run all media type tests"""
    print("μOmni Multimodal Inference Test Suite")
    print("=" * 60)
    
    # Check if example files exist
    examples_dir = Path("examples")
    data_dir = Path("data")
    
    # Ensure we have sample files
    if not examples_dir.exists():
        examples_dir.mkdir()
    
    # Use existing data files or examples
    sample_image = "examples/sample_image.png"
    if not os.path.exists(sample_image):
        sample_image = "data/images/images/000000.png"
    
    sample_audio = "examples/sample_audio.wav"
    if not os.path.exists(sample_audio):
        sample_audio = "data/audio/wav/000000.wav"
    
    sample_text = "examples/sample_text.txt"
    if not os.path.exists(sample_text):
        with open(sample_text, "w") as f:
            f.write("This is a test prompt.")
    
    # Test results
    results = []
    
    # Test 1: Text only (text output)
    print("\n[TEST 1] Text-only inference (text output)")
    results.append(("Text-only", run_inference(
        ["--ckpt_dir", "checkpoints/thinker_tiny", "--text", "Hello, how are you?"],
        "Text-only chat"
    )))
    
    # Test 1b: Text with audio output (TTS)
    print("\n[TEST 1b] Text with audio output (TTS)")
    audio_out_path = "examples/test_output_text_tts.wav"
    results.append(("Text+TTS", run_inference(
        ["--ckpt_dir", "checkpoints/omni_sft_tiny", 
         "--text", "This is a test of text to speech.",
         "--audio_out", audio_out_path],
        "Text input with audio output (TTS)"
    )))
    
    # Test 2: Image + Text
    print("\n[TEST 2] Image + Text")
    if os.path.exists(sample_image):
        results.append(("Image+Text", run_inference(
            ["--ckpt_dir", "checkpoints/omni_sft_tiny", 
             "--image", sample_image, 
             "--text", "What do you see in this image?"],
            "Image with text prompt"
        )))
    else:
        print(f"SKIP: Image file not found: {sample_image}")
        results.append(("Image+Text", False))
    
    # Test 3: Audio + Text
    print("\n[TEST 3] Audio + Text")
    if os.path.exists(sample_audio):
        results.append(("Audio+Text", run_inference(
            ["--ckpt_dir", "checkpoints/omni_sft_tiny",
             "--audio_in", sample_audio,
             "--text", "What did you hear?"],
            "Audio with text prompt"
        )))
    else:
        print(f"SKIP: Audio file not found: {sample_audio}")
        results.append(("Audio+Text", False))
    
    # Test 4: Image only (default prompt)
    print("\n[TEST 4] Image only")
    if os.path.exists(sample_image):
        results.append(("Image-only", run_inference(
            ["--ckpt_dir", "checkpoints/omni_sft_tiny",
             "--image", sample_image],
            "Image with default prompt"
        )))
    else:
        print(f"SKIP: Image file not found: {sample_image}")
        results.append(("Image-only", False))
    
    # Test 5: Audio only
    print("\n[TEST 5] Audio only")
    if os.path.exists(sample_audio):
        results.append(("Audio-only", run_inference(
            ["--ckpt_dir", "checkpoints/omni_sft_tiny",
             "--audio_in", sample_audio],
            "Audio with default prompt"
        )))
    else:
        print(f"SKIP: Audio file not found: {sample_audio}")
        results.append(("Audio-only", False))
    
    # Test 6: Image + Audio + Text (multimodal)
    print("\n[TEST 6] Image + Audio + Text (Multimodal)")
    if os.path.exists(sample_image) and os.path.exists(sample_audio):
        results.append(("Multimodal", run_inference(
            ["--ckpt_dir", "checkpoints/omni_sft_tiny",
             "--image", sample_image,
             "--audio_in", sample_audio,
             "--text", "Describe what you see and hear."],
            "Combined image, audio, and text"
        )))
    else:
        print(f"SKIP: Missing files (image: {os.path.exists(sample_image)}, audio: {os.path.exists(sample_audio)})")
        results.append(("Multimodal", False))
    
    # Test 6b: Image + Text with audio output
    print("\n[TEST 6b] Image + Text with audio output")
    if os.path.exists(sample_image):
        audio_out_path = "examples/test_output_image_tts.wav"
        results.append(("Image+Text+TTS", run_inference(
            ["--ckpt_dir", "checkpoints/omni_sft_tiny",
             "--image", sample_image,
             "--text", "Describe this image.",
             "--audio_out", audio_out_path],
            "Image input with text and audio output"
        )))
    else:
        print(f"SKIP: Image file not found: {sample_image}")
        results.append(("Image+Text+TTS", False))
    
    # Test 7: Video (if available)
    print("\n[TEST 7] Video")
    sample_video = "examples/sample_video.mp4"
    if os.path.exists(sample_video):
        results.append(("Video", run_inference(
            ["--ckpt_dir", "checkpoints/omni_sft_tiny",
             "--video", sample_video,
             "--text", "Describe the video."],
            "Video with text prompt"
        )))
    else:
        print(f"SKIP: Video file not found: {sample_video}")
        print("      Note: Create a test video with:")
        print("      ffmpeg -framerate 1 -i data/images/images/%06d.png -c:v libx264 -pix_fmt yuv420p examples/sample_video.mp4")
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

