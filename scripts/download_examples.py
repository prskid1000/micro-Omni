"""
Download very small example files for testing multimodal inference
"""
import os
import urllib.request
from pathlib import Path

def download_file(url, dest_path):
    """Download a file from URL to destination"""
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    print(f"Downloading {url} to {dest_path}...")
    try:
        urllib.request.urlretrieve(url, dest_path)
        print(f"✓ Downloaded {dest_path}")
        return True
    except Exception as e:
        print(f"✗ Failed to download {url}: {e}")
        return False

def main():
    examples_dir = Path("examples")
    examples_dir.mkdir(exist_ok=True)
    
    # Very small sample files (each < 1MB)
    samples = [
        # Image - small test image
        {
            "url": "https://raw.githubusercontent.com/microsoft/fluentui-emoji/main/assets/Red%20circle/Default/3D/red_circle_3d.png",
            "path": examples_dir / "sample_image.png"
        },
        # Alternative: use a simple test image from a reliable source
        {
            "url": "https://via.placeholder.com/224x224.png",
            "path": examples_dir / "sample_image.png"
        },
        # Audio - very short sample (using a public domain sample)
        {
            "url": "https://www.soundhelix.com/examples/mp3/SoundHelix-Song-1.mp3",
            "path": examples_dir / "sample_audio.mp3"
        },
        # Text - create a simple text file
        {
            "url": None,  # Will create locally
            "path": examples_dir / "sample_text.txt",
            "content": "This is a sample text prompt for testing the multimodal interface."
        },
    ]
    
    # Try to download or create samples
    for sample in samples:
        if sample.get("content"):
            # Create text file
            with open(sample["path"], "w", encoding="utf-8") as f:
                f.write(sample["content"])
            print(f"✓ Created {sample['path']}")
        elif sample.get("url"):
            download_file(sample["url"], sample["path"])
    
    # For video, we'll create a simple synthetic one or use existing data
    # Since downloading videos can be large, we'll use existing image data to create a simple "video"
    print("\nNote: For video testing, you can use existing image sequences or create a simple test video.")
    print("Example: Use ffmpeg to create a video from images:")
    print("  ffmpeg -framerate 1 -i data/images/images/%06d.png -c:v libx264 -pix_fmt yuv420p examples/sample_video.mp4")
    
    print(f"\nExamples directory: {examples_dir.absolute()}")
    print("Sample files ready for testing!")

if __name__ == "__main__":
    main()

