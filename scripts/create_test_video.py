"""Create a simple test video from image frames"""
import os
from PIL import Image
import numpy as np

def create_test_video():
    """Create a simple test video from existing images using PyAV"""
    examples_dir = "examples"
    os.makedirs(examples_dir, exist_ok=True)
    
    # Load a few images
    images = []
    for i in range(5):
        img_path = f"data/images/images/{i:06d}.png"
        if os.path.exists(img_path):
            img = Image.open(img_path).convert("RGB")
            # Convert to numpy array (H, W, C)
            img_array = np.array(img)
            images.append(img_array)
    
    if not images:
        print("No images found to create video")
        return
    
    output_path = os.path.join(examples_dir, "sample_video.mp4")
    
    # Try using PyAV
    try:
        import av
        container = av.open(output_path, mode='w')
        stream = container.add_stream('libx264', rate=1)  # 1 fps
        stream.width = images[0].shape[1]
        stream.height = images[0].shape[0]
        stream.pix_fmt = 'yuv420p'
        
        for img_array in images:
            frame = av.VideoFrame.from_ndarray(img_array, format='rgb24')
            for packet in stream.encode(frame):
                container.mux(packet)
        
        # Flush stream
        for packet in stream.encode():
            container.mux(packet)
        
        container.close()
        print(f"Created test video: {output_path}")
        return True
    except ImportError:
        print("PyAV not installed. Install with: pip install av")
        print("Note: PyAV requires ffmpeg to be installed on your system.")
    except Exception as e:
        print(f"Could not create video with PyAV: {e}")
        print("\nTo enable video support:")
        print("  1. Install PyAV: pip install av")
        print("  2. Install ffmpeg on your system:")
        print("     - Windows: Download from https://ffmpeg.org/download.html")
        print("     - Linux: sudo apt-get install ffmpeg")
        print("     - Mac: brew install ffmpeg")
        print("\nOr create video manually with ffmpeg:")
        print("  ffmpeg -framerate 1 -i data/images/images/%06d.png -c:v libx264 -pix_fmt yuv420p examples/sample_video.mp4")
    
    return False

if __name__ == "__main__":
    create_test_video()

