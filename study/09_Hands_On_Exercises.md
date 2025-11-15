# Hands-On Exercises

## Exercise 1: Understanding Embeddings

### Task
Create a simple embedding layer and see how it works.

### Code
```python
import torch
import torch.nn as nn

# Create embedding layer
vocab_size = 1000
embed_dim = 128
emb = nn.Embedding(vocab_size, embed_dim)

# Test it
token_id = torch.tensor([42])
embedding = emb(token_id)
print(f"Token ID: {token_id.item()}")
print(f"Embedding shape: {embedding.shape}")
print(f"Embedding: {embedding[0, :5]}")  # First 5 values
```

### Questions
1. What is the shape of the embedding?
2. What happens if you use token_id = 2000? (vocab_size is 1000)

## Exercise 2: Simple Attention

### Task
Implement a basic attention mechanism.

### Code
```python
import torch
import torch.nn.functional as F

def simple_attention(query, key, value):
    # Compute attention scores
    scores = torch.matmul(query, key.transpose(-2, -1))
    scores = scores / (query.size(-1) ** 0.5)  # Scale
    
    # Apply softmax
    weights = F.softmax(scores, dim=-1)
    
    # Weighted sum
    output = torch.matmul(weights, value)
    return output, weights

# Test
d = 64  # dimension
seq_len = 10
q = torch.randn(1, seq_len, d)
k = torch.randn(1, seq_len, d)
v = torch.randn(1, seq_len, d)

output, weights = simple_attention(q, k, v)
print(f"Output shape: {output.shape}")
print(f"Attention weights shape: {weights.shape}")
print(f"Attention weights sum: {weights.sum(dim=-1)}")  # Should be 1.0
```

### Questions
1. Why do we divide by sqrt(d)?
2. What does the attention weights matrix represent?

## Exercise 3: Mel Spectrogram

### Task
Convert audio to mel spectrogram and visualize.

### Code
```python
import torch
import torchaudio
import matplotlib.pyplot as plt

# Load audio
audio, sr = torchaudio.load("examples/sample_audio.wav")
print(f"Audio shape: {audio.shape}, Sample rate: {sr}")

# Create mel spectrogram
mel_spec = torchaudio.transforms.MelSpectrogram(
    sample_rate=16000,
    n_fft=1024,
    hop_length=160,
    n_mels=128
)

# Convert
mel = mel_spec(audio)
print(f"Mel shape: {mel.shape}")

# Visualize (optional - requires matplotlib)
# plt.figure(figsize=(12, 4))
# plt.imshow(mel[0].numpy(), aspect='auto', origin='lower')
# plt.xlabel('Time frames')
# plt.ylabel('Mel bins')
# plt.title('Mel Spectrogram')
# plt.colorbar()
# plt.show()
```

### Questions
1. What does each dimension represent?
2. How many frames per second? (hop_length=160, sr=16000)

## Exercise 4: Image Patches

### Task
Split an image into patches like ViT does.

### Code
```python
from PIL import Image
import torch
import torch.nn as nn

# Load image
img = Image.open("examples/sample_image.png")
img_tensor = torch.tensor(img).permute(2, 0, 1).float() / 255.0
print(f"Image shape: {img_tensor.shape}")

# Create patch embedding (like ViT)
patch_size = 16
patch_emb = nn.Conv2d(3, 128, kernel_size=patch_size, stride=patch_size)

# Extract patches
patches = patch_emb(img_tensor.unsqueeze(0))
print(f"Patches shape: {patches.shape}")

# Reshape to sequence
B, C, H, W = patches.shape
patches = patches.reshape(B, C, H * W).transpose(1, 2)
print(f"Patches as sequence: {patches.shape}")
```

### Questions
1. How many patches are there? (224×224 image, 16×16 patches)
2. What does each patch represent?

## Exercise 5: Simple Generation

### Task
Implement a basic text generation loop.

### Code
```python
import torch
import torch.nn.functional as F

# Simplified model (just for demo)
class SimpleModel:
    def __init__(self):
        self.vocab_size = 1000
    
    def __call__(self, tokens):
        # Random logits (in real model, this would be actual predictions)
        return torch.randn(len(tokens), self.vocab_size)

model = SimpleModel()

# Generation function
def generate(prompt_tokens, max_new=10):
    tokens = prompt_tokens.copy()
    
    for _ in range(max_new):
        # Get predictions
        logits = model(tokens)
        
        # Get next token (greedy)
        next_token = torch.argmax(logits[-1]).item()
        tokens.append(next_token)
        
        # Stop if EOS (token 2)
        if next_token == 2:
            break
    
    return tokens

# Test
prompt = [1, 100, 200]  # [BOS, token1, token2]
generated = generate(prompt, max_new=5)
print(f"Prompt: {prompt}")
print(f"Generated: {generated}")
```

### Questions
1. What is "greedy" decoding?
2. How could you make it less deterministic?

## Exercise 6: Training Loop

### Task
Write a minimal training loop.

### Code
```python
import torch
import torch.nn as nn
import torch.optim as optim

# Simple model
model = nn.Linear(10, 1)
optimizer = optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.MSELoss()

# Dummy data
X = torch.randn(32, 10)  # Batch of 32, 10 features
y = torch.randn(32, 1)   # Targets

# Training loop
for epoch in range(10):
    # Forward
    pred = model(X)
    loss = loss_fn(pred, y)
    
    # Backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
```

### Questions
1. Why do we call `optimizer.zero_grad()`?
2. What happens if we don't call `loss.backward()`?

## Exercise 7: Multimodal Fusion

### Task
Combine embeddings from different modalities.

### Code
```python
import torch

# Simulate embeddings
img_emb = torch.randn(1, 1, 256)    # Image: 1 token
audio_emb = torch.randn(1, 10, 256)  # Audio: 10 tokens
text_emb = torch.randn(1, 5, 256)    # Text: 5 tokens

# Fuse (concatenate)
fused = torch.cat([img_emb, audio_emb, text_emb], dim=1)
print(f"Fused shape: {fused.shape}")  # Should be (1, 16, 256)

# Check total tokens
total_tokens = img_emb.shape[1] + audio_emb.shape[1] + text_emb.shape[1]
print(f"Total tokens: {total_tokens}")
```

### Questions
1. What is the total context length used?
2. How would you handle if total exceeds context limit?

## Exercise 8: Checkpoint Loading

### Task
Load and inspect a trained model checkpoint (with full state: model, optimizer, scheduler, etc.).

### Code
```python
import torch

# Load checkpoint (new format includes full training state)
checkpoint = torch.load("checkpoints/thinker_tiny/thinker_best.pt", map_location="cpu")

# Inspect checkpoint structure
print(f"Checkpoint keys: {list(checkpoint.keys())}")
# Expected: ['model', 'optimizer', 'scheduler', 'scaler', 'step', 'best_val_loss']

# Check if it's new format (dict) or legacy (just model weights)
if isinstance(checkpoint, dict) and "model" in checkpoint:
    print("New checkpoint format detected!")
    model_state = checkpoint["model"]
    print(f"Step: {checkpoint.get('step', 'N/A')}")
    print(f"Best validation loss: {checkpoint.get('best_val_loss', 'N/A')}")
    print(f"Has optimizer state: {'optimizer' in checkpoint}")
    print(f"Has scheduler state: {'scheduler' in checkpoint}")
    print(f"Has scaler state: {'scaler' in checkpoint}")
else:
    print("Legacy checkpoint format (model weights only)")
    model_state = checkpoint

# Load into model
from omni.thinker import ThinkerLM
model = ThinkerLM(
    vocab_size=5000,
    n_layers=4,
    d_model=256,
    n_heads=4,
    d_ff=1024,
    dropout=0.1,
    rope_theta=10000,
    ctx_len=512
)
model.load_state_dict(model_state)
print("Model loaded successfully!")
```

### Questions
1. How many parameters does the model have?
2. What happens if model architecture doesn't match checkpoint?
3. What information is stored in the new checkpoint format?

## Exercise 9: Inference Script

### Task
Write a simple inference function.

### Code
```python
import torch
from omni.thinker import ThinkerLM
from omni.tokenizer import BPETokenizer

def simple_inference(model_path, tokenizer_path, prompt):
    # Load model
    model = ThinkerLM(...)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    # Load tokenizer
    tokenizer = BPETokenizer(tokenizer_path)
    
    # Tokenize
    tokens = [1] + tokenizer.encode(prompt)  # Add BOS
    tokens = torch.tensor([tokens])
    
    # Generate
    with torch.no_grad():
        logits = model(tokens)
        next_token = torch.argmax(logits[0, -1, :]).item()
    
    # Decode
    output = tokenizer.decode([next_token])
    return output

# Test
result = simple_inference(
    "checkpoints/thinker_tiny/thinker.pt",
    "checkpoints/thinker_tiny/tokenizer.model",
    "Hello"
)
print(f"Output: {result}")
```

## Exercise 10: Debug Training

### Task
Add debugging to a training loop.

### Code
```python
import torch

def debug_training(model, dataloader, loss_fn, optimizer):
    for batch_idx, (x, y) in enumerate(dataloader):
        # Check input
        print(f"Batch {batch_idx}:")
        print(f"  Input shape: {x.shape}, dtype: {x.dtype}")
        print(f"  Input range: [{x.min():.2f}, {x.max():.2f}]")
        
        # Forward
        pred = model(x)
        loss = loss_fn(pred, y)
        
        # Check output
        print(f"  Output shape: {pred.shape}")
        print(f"  Loss: {loss.item():.4f}")
        
        # Check for NaN (now automatic in training scripts)
        if torch.isnan(loss):
            print("  WARNING: NaN loss!")
            break
        
        # Better: Use validate_loss utility (used in actual training)
        from omni.training_utils import validate_loss
        try:
            validate_loss(loss, min_loss=-1e6, max_loss=1e6)
        except RuntimeError as e:
            print(f"  ERROR: {e}")
            break
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        
        # Check gradients (now automatic in training scripts)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float('inf'))
        print(f"  Gradient norm: {grad_norm:.4f}")
        
        # Better: Use check_gradient_explosion utility (used in actual training)
        from omni.training_utils import check_gradient_explosion
        grad_norm, is_exploded = check_gradient_explosion(model, max_grad_norm=100.0, raise_on_error=False)
        if is_exploded:
            print(f"  WARNING: Gradient explosion detected! grad_norm={grad_norm:.2f}")
            break
        
        optimizer.step()
        
        if batch_idx >= 2:  # Only do first 3 batches
            break
```

## Solutions

### Exercise 1
1. Shape: `(1, 128)` - one token, 128 dimensions
2. Error: index out of range (vocab_size is 1000, can't use 2000)

### Exercise 2
1. Prevents attention scores from becoming too large
2. How much each position attends to each other position

### Exercise 3
1. `(1, 128, T)` - 1 channel, 128 mel bins, T time frames
2. 100 frames/sec (16000 / 160 = 100)

### Exercise 4
1. 14×14 = 196 patches
2. A 16×16 pixel region of the image

### Exercise 5
1. Always picks highest probability token (deterministic)
2. Use sampling (temperature, top-k, top-p)

### Exercise 6
1. Clears previous gradients (they accumulate otherwise)
2. No gradients computed, optimizer.step() does nothing

### Exercise 7
1. 16 tokens total
2. Truncate longest modality or chunk inputs

### Exercise 8
1. Check `sum(p.numel() for p in checkpoint.values())`
2. Error: size mismatch

---

**Next Steps:**
- Try modifying the code
- Experiment with different parameters
- Read the actual implementation in `omni/` folder
- Check [Inference Guide](08_Inference_Guide.md) for more examples

**See Also:**
- [Neural Networks Basics](01_Neural_Networks_Basics.md)
- [Training Workflow](07_Training_Workflow.md)

