"""Create a simple test video from image frames"""
import os
from PIL import Image
import torch
import torchvision.io as tvio

def create_test_video():
    """Create a simple test video from existing images"""
    examples_dir = "examples"
    os.makedirs(examples_dir, exist_ok=True)
    
    # Load a few images
    images = []
    for i in range(5):
        img_path = f"data/images/images/{i:06d}.png"
        if os.path.exists(img_path):
            img = Image.open(img_path).convert("RGB")
            # Convert to tensor (C, H, W)
            import torchvision.transforms as T
            transform = T.ToTensor()
            img_tensor = transform(img)
            images.append(img_tensor)
    
    if not images:
        print("No images found to create video")
        return
    
    # Stack images as frames (T, C, H, W)
    video_tensor = torch.stack(images)
    
    # Save as video
    output_path = os.path.join(examples_dir, "sample_video.mp4")
    try:
        tvio.write_video(output_path, video_tensor, fps=1.0)
        print(f"Created test video: {output_path}")
    except Exception as e:
        print(f"Could not create video with torchvision: {e}")
        print("Note: Video creation requires ffmpeg. You can create it manually with:")
        print("  ffmpeg -framerate 1 -i data/images/images/%06d.png -c:v libx264 -pix_fmt yuv420p examples/sample_video.mp4")

if __name__ == "__main__":
    create_test_video()

