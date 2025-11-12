# Inference Guide: Using Trained Models

## What is Inference?

**Inference** is using a trained model to make predictions on new data.

Training = Learning from examples
Inference = Using what was learned

## Basic Usage

### Text-Only Chat

```bash
python infer_chat.py --ckpt_dir checkpoints/thinker_tiny --text "Hello, how are you?"
```

**What happens:**
1. Load Thinker model
2. Tokenize input text
3. Generate response
4. Decode tokens to text
5. Print output

### Image + Text

```bash
python infer_chat.py --ckpt_dir checkpoints/omni_sft_tiny \
  --image examples/sample_image.png \
  --text "What do you see?"
```

**What happens:**
1. Load all models (Thinker, Vision, projectors)
2. Encode image → embeddings
3. Tokenize text → embeddings
4. Fuse embeddings
5. Thinker processes
6. Generate text response

### Audio Input

```bash
python infer_chat.py --ckpt_dir checkpoints/omni_sft_tiny \
  --audio_in examples/sample_audio.wav \
  --text "What did you hear?"
```

### Text-to-Speech

```bash
python infer_chat.py --ckpt_dir checkpoints/omni_sft_tiny \
  --text "Hello world" \
  --audio_out output.wav
```

## Code Walkthrough

### Loading Models

```python
# From infer_chat.py

# Load Thinker
thinker = ThinkerLM(...)
thinker.load_state_dict(torch.load("thinker.pt"))
thinker.eval()  # Set to evaluation mode

# Load encoders
vision_encoder = ViTTiny(...)
audio_encoder = AudioEncoderTiny(...)

# Load projectors (from omni checkpoint)
projectors = torch.load("omni.pt")
vision_projector.load_state_dict(projectors["proj_v"])
audio_projector.load_state_dict(projectors["proj_a"])
```

### Processing Inputs

```python
# Image
if image_path:
    image = Image.open(image_path)
    image_tensor = transform(image)  # (3, 224, 224)
    cls_token, _ = vision_encoder(image_tensor)
    img_emb = vision_projector(cls_token)  # (1, 256)

# Audio
if audio_path:
    audio, sr = torchaudio.load(audio_path)
    mel = mel_spec(audio)
    audio_emb = audio_encoder(mel)
    audio_emb = audio_projector(audio_emb)  # (T, 256)

# Text
text_ids = tokenizer.encode(text)
text_emb = thinker.tok_emb(text_ids)  # (T, 256)
```

### Generation

```python
# Combine embeddings
combined = torch.cat([img_emb, audio_emb, text_emb], dim=1)

# Generate
output = generate(thinker, tokenizer, prompt, 
                  multimodal_emb=combined)

# Output is text string
print(output)
```

## Generation Process

### Autoregressive Generation

```python
def generate(model, tokenizer, prompt, max_new=64):
    # Tokenize prompt
    ids = tokenizer.encode(prompt)
    
    # First forward pass
    logits = model(ids)
    next_id = argmax(logits[:, -1, :])
    generated = ids + [next_id]
    
    # Continue generating
    for _ in range(max_new - 1):
        # Only process new token (KV cache handles rest)
        logits = model([[next_id]])
        next_id = argmax(logits[:, -1, :])
        generated.append(next_id)
        
        if next_id == EOS:
            break
    
    return tokenizer.decode(generated)
```

### KV Caching

Speed up generation by caching attention states:

```python
# Enable caching
model.enable_kv_cache(True)
model.reset_kv_cache()

# First pass: process full sequence
logits, kv_cache = model(prompt, use_cache=True)

# Subsequent passes: only new token
logits, kv_cache = model(new_token, kv_cache=kv_cache)
```

## TTS Generation

### Complete Pipeline

```python
# 1. Generate text
text = "Hello world"

# 2. Generate audio codes
codes = generate_audio_codes(talker, text, max_frames=200)

# 3. Decode codes to mel
mel = rvq.decode(codes)

# 4. Convert mel to audio
audio = vocoder.mel_to_audio(mel)

# 5. Save
soundfile.write("output.wav", audio, 16000)
```

### Audio Code Generation

```python
def generate_audio_codes(talker, max_frames=200):
    codes = torch.zeros(1, 1, 2)  # Start
    
    for _ in range(max_frames):
        base_logits, res_logits = talker(codes)
        base_code = argmax(base_logits[:, -1, :])
        res_code = argmax(res_logits[:, -1, :])
        codes = torch.cat([codes, [[[base_code, res_code]]]], dim=1)
    
    return codes
```

## Interactive Mode

```bash
python infer_chat.py --ckpt_dir checkpoints/thinker_tiny
```

Enters interactive chat:

```
Entering interactive chat mode. Type 'exit' to quit.
You: Hello
μOmni: Hello! How can I help you?
You: What is AI?
μOmni: AI is artificial intelligence...
You: exit
```

## Common Parameters

- `--ckpt_dir`: Checkpoint directory
- `--text`: Text prompt
- `--image`: Image file path
- `--audio_in`: Input audio file
- `--audio_out`: Output audio file (TTS)
- `--video`: Video file path
- `--prompt`: Override default prompt

## Performance Tips

1. **Use GPU**: Much faster than CPU
2. **KV Caching**: Enabled by default
3. **Batch Processing**: Process multiple inputs together
4. **Context Length**: Truncate if too long

## Troubleshooting

### Model Not Found

```
Error: Checkpoint not found
```

**Solution**: Check checkpoint path, ensure training completed

### Out of Memory

```
Error: CUDA out of memory
```

**Solution**: 
- Reduce batch size
- Use CPU (slower)
- Reduce context length

### Poor Quality Output

**Causes**:
- Insufficient training
- Wrong checkpoint
- Poor data quality

**Solutions**:
- Train longer
- Use best checkpoint
- Check data quality

---

**Next:** [09_Hands_On_Exercises.md](09_Hands_On_Exercises.md) - Practice exercises

**See Also:**
- [Training Workflow](07_Training_Workflow.md)
- [Architecture Overview](02_Architecture_Overview.md)

