# Chapter 40: Running Inference Examples

[← Previous: Running Training](39-running-training.md) | [Back to Index](00-INDEX.md)

---

## 🎯 What You'll Learn

- Running text-only inference
- Image understanding and QA
- Audio input processing
- Text-to-speech generation
- Multimodal combinations
- Practical code examples

---

## 📝 Text-Only Chat

### Basic Usage

```bash
python infer_chat.py --ckpt_dir checkpoints/thinker_tiny
```

### Interactive Session

```
μOmni Setup Complete!
Using device: cuda

Entering interactive chat mode. Type 'exit' to quit.

You: What is artificial intelligence?
μOmni: Artificial intelligence is the simulation of human intelligence 
       processes by machines, especially computer systems.

You: Explain transformers
μOmni: Transformers are neural network architectures that use attention 
       mechanisms to process sequential data in parallel...

You: exit
```

### Programmatic Usage

```python
from omni.thinker import ThinkerLM
from omni.tokenizer import BPETokenizer
import torch

# Load model
device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = BPETokenizer("checkpoints/thinker_tiny/tokenizer.model")
model = ThinkerLM(vocab=5000, n_layers=4, d=256, heads=4, ff=1024)
model.load_state_dict(torch.load("checkpoints/thinker_tiny/thinker.pt"))
model.to(device)
model.eval()

# Generate text
def generate(prompt, max_tokens=50):
    ids = [1] + tokenizer.encode(prompt)  # Add BOS
    x = torch.tensor(ids, device=device).unsqueeze(0)
    
    model.reset_kv_cache()
    model.enable_kv_cache(True)
    
    for _ in range(max_tokens):
        with torch.no_grad():
            logits = model(x if len(ids) == 1 else x[:, -1:])
            next_id = torch.argmax(logits[0, -1]).item()
            if next_id == 2:  # EOS
                break
            ids.append(next_id)
            x = torch.tensor([[next_id]], device=device)
    
    return tokenizer.decode(ids)

# Use it
output = generate("What is machine learning?")
print(output)
```

---

## 🖼️ Image Understanding

### Basic Image Description

```bash
python infer_chat.py \
  --ckpt_dir checkpoints/omni_sft_tiny \
  --image examples/cat.jpg \
  --text "Describe this image"
```

Output:
```
Processing image: examples/cat.jpg
Prompt: Describe this image

μOmni (text): This is a photo of an orange tabby cat sitting on a 
              blue couch. The cat is looking directly at the camera.
```

---

### Visual Question Answering (VQA)

```bash
# Question 1: Color
python infer_chat.py \
  --ckpt_dir checkpoints/omni_sft_tiny \
  --image examples/car.jpg \
  --text "What color is the car?"

Output: The car is red.

# Question 2: Count
python infer_chat.py \
  --ckpt_dir checkpoints/omni_sft_tiny \
  --image examples/park.jpg \
  --text "How many people are in the image?"

Output: There are three people in the image.

# Question 3: Location
python infer_chat.py \
  --ckpt_dir checkpoints/omni_sft_tiny \
  --image examples/beach.jpg \
  --text "Where is this photo taken?"

Output: This photo appears to be taken at a beach during sunset.
```

---

### Batch Image Processing

```python
import os
from PIL import Image
import torch
from torchvision import transforms

# Load model components
from omni.vision_encoder import ViTTiny
from omni.thinker import ThinkerLM

# Setup
device = "cuda"
vision_enc = ViTTiny(img_size=224, patch=16, d=128, layers=4).to(device)
thinker = ThinkerLM(...).to(device)

# Load weights
vision_enc.load_state_dict(torch.load("checkpoints/vision_tiny/vision.pt")["vit"])
thinker.load_state_dict(torch.load("checkpoints/omni_sft_tiny/omni.pt")["thinker"])

vision_enc.eval()
thinker.eval()

# Image transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Process multiple images
image_dir = "examples/images/"
for img_file in os.listdir(image_dir):
    if img_file.endswith(('.jpg', '.png')):
        img_path = os.path.join(image_dir, img_file)
        img = Image.open(img_path).convert("RGB")
        img_tensor = transform(img).unsqueeze(0).to(device)
        
        with torch.no_grad():
            cls, _ = vision_enc(img_tensor)
            # ... process with thinker
            print(f"{img_file}: [description]")
```

---

## 🎤 Audio Input (Speech Recognition + Understanding)

### Speech Transcription

```bash
python infer_chat.py \
  --ckpt_dir checkpoints/omni_sft_tiny \
  --audio_in examples/speech.wav
```

Output:
```
Processing audio: examples/speech.wav
Audio duration: 3.2 seconds

Transcription: "Hello, how are you doing today?"

μOmni (understanding): The speaker is greeting someone and asking 
                        about their well-being.
```

---

### Speech + Text Query

```bash
python infer_chat.py \
  --ckpt_dir checkpoints/omni_sft_tiny \
  --audio_in examples/meeting_audio.wav \
  --text "Summarize the main points"
```

Output:
```
μOmni: The audio discusses three main topics:
       1. Project timeline and deliverables
       2. Budget allocation for Q4
       3. Team role assignments
```

---

## 🔊 Text-to-Speech Generation

### Basic TTS

```bash
python infer_chat.py \
  --ckpt_dir checkpoints/omni_sft_tiny \
  --text "Hello world, this is a test" \
  --audio_out output.wav
```

Output:
```
Generating audio output...
Generating 375 frames for 3 tokens (~6.0s)
Generated audio
Audio saved to: output.wav
```

Play the audio:
```bash
# Linux
aplay output.wav

# Mac
afplay output.wav

# Windows
start output.wav

# Or use Python
import soundfile as sf
import sounddevice as sd
audio, sr = sf.read("output.wav")
sd.play(audio, sr)
sd.wait()
```

---

### Batch TTS Generation

```python
import torch
from omni.talker import TalkerTiny
from omni.codec import RVQ, GriffinLimVocoder
import soundfile as sf

# Load models
talker = TalkerTiny(d=192, layers=4, heads=3, ff=768, 
                    codebooks=2, codebook_size=128).to("cuda")
rvq = RVQ(num_codebooks=2, codebook_size=128, d=64).to("cuda")
talker.load_state_dict(torch.load("checkpoints/talker_tiny/talker.pt")["talker"])
rvq.load_state_dict(torch.load("checkpoints/talker_tiny/talker.pt")["rvq"])
vocoder = GriffinLimVocoder()

talker.eval()
rvq.eval()

# Generate speech for multiple sentences
sentences = [
    "Hello, welcome to μOmni.",
    "This is a text to speech system.",
    "It converts text into natural speech."
]

for i, text in enumerate(sentences):
    # Generate RVQ codes (simplified)
    codes = generate_codes(talker, max_frames=200)  # Your generation logic
    
    # Decode to mel
    mel = []
    for t in range(codes.shape[1]):
        frame = rvq.decode(codes[:, t, :])
        mel.append(frame)
    mel = torch.stack(mel, dim=0)
    
    # Mel to audio
    audio = vocoder.mel_to_audio(mel.cpu().numpy())
    
    # Save
    sf.write(f"output_{i}.wav", audio, 16000)
    print(f"✓ Generated: output_{i}.wav")
```

---

## 🌈 Multimodal Combinations

### Image + Audio + Text

```bash
python infer_chat.py \
  --ckpt_dir checkpoints/omni_sft_tiny \
  --image examples/scene.jpg \
  --audio_in examples/description.wav \
  --text "Combine what you see and hear"
```

Output:
```
Processing image: examples/scene.jpg
Processing audio: examples/description.wav (2.5s)
Prompt: Combine what you see and hear

μOmni: I can see a park with trees and benches. The audio describes 
       people playing frisbee. Together, this shows a recreational 
       park scene with active visitors enjoying outdoor activities.
```

---

### Video Processing (Frame Sampling)

```bash
python infer_chat.py \
  --ckpt_dir checkpoints/omni_sft_tiny \
  --video examples/clip.mp4 \
  --text "What's happening in this video?"
```

```python
# Video processing extracts frames
def extract_video_frames(video_path, num_frames=4):
    import torchvision.io as tvio
    video, audio, info = tvio.read_video(video_path)
    total_frames = video.shape[0]
    indices = [int(i * total_frames / (num_frames + 1)) 
               for i in range(1, num_frames + 1)]
    frames = [video[i] for i in indices if i < total_frames]
    return frames

# Process first frame (representative)
frames = extract_video_frames("examples/clip.mp4")
# ... process with vision encoder
```

Output:
```
Extracted 4 frames from video
Processing frame 1 as representative

μOmni: The video shows a person walking down a city street during 
       sunset. There are buildings and cars in the background.
```

---

## 🎯 Advanced Inference Options

### Temperature Sampling

```python
# Greedy (default): Always pick highest probability
next_id = torch.argmax(logits[0, -1])

# Temperature sampling: Add randomness
def sample_with_temperature(logits, temperature=0.8):
    logits = logits / temperature
    probs = torch.softmax(logits, dim=-1)
    next_id = torch.multinomial(probs, num_samples=1)
    return next_id.item()

# Higher temperature = more creative (random)
# Lower temperature = more conservative (deterministic)
```

---

### Top-k and Top-p Sampling

```python
def sample_top_k(logits, k=50):
    """Sample from top-k most likely tokens"""
    top_k_logits, top_k_indices = torch.topk(logits, k)
    probs = torch.softmax(top_k_logits, dim=-1)
    next_id = top_k_indices[torch.multinomial(probs, 1)]
    return next_id.item()

def sample_top_p(logits, p=0.9):
    """Sample from smallest set of tokens with cumulative prob > p"""
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
    
    # Remove tokens with cumulative probability above threshold
    sorted_indices_to_remove = cumulative_probs > p
    sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
    sorted_indices_to_remove[0] = False
    
    filtered_logits = sorted_logits.clone()
    filtered_logits[sorted_indices_to_remove] = float('-inf')
    
    probs = torch.softmax(filtered_logits, dim=-1)
    next_id = sorted_indices[torch.multinomial(probs, 1)]
    return next_id.item()
```

---

## 📊 Performance Metrics

### Measure Inference Speed

```python
import time
import torch

# Warmup
for _ in range(10):
    _ = model(torch.randn(1, 10).long().to("cuda"))

# Measure
torch.cuda.synchronize()
start = time.time()

for _ in range(100):
    _ = model(torch.randn(1, 10).long().to("cuda"))

torch.cuda.synchronize()
end = time.time()

latency = (end - start) / 100 * 1000  # ms per inference
throughput = 100 / (end - start)      # inferences per second

print(f"Latency: {latency:.2f} ms")
print(f"Throughput: {throughput:.2f} inf/sec")
```

---

### Memory Usage

```python
import torch

# Before inference
torch.cuda.reset_peak_memory_stats()

# Run inference
output = model(input_ids)

# Check memory
max_memory = torch.cuda.max_memory_allocated() / 1e9  # GB
print(f"Peak GPU memory: {max_memory:.2f} GB")
```

---

## 💡 Key Takeaways

✅ **Text-only** mode for language tasks  
✅ **Image understanding** with VQA support  
✅ **Audio input** for speech recognition  
✅ **Text-to-speech** generation  
✅ **Multimodal** combinations supported  
✅ **Sampling strategies** (temperature, top-k, top-p) for variety  
✅ **KV caching** makes generation fast

---

## 🎓 Self-Check Questions

1. How do you run text-only inference?
2. What command generates speech from text?
3. Can μOmni process video directly?
4. What does temperature sampling control?
5. How does KV caching speed up generation?

<details>
<parameter name="summary">📝 Answers
