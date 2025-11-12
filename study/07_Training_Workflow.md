# Training Workflow

## Overview

μOmni uses a **staged training approach** - train components separately, then combine them.

**Why staged?**
- Fits in 12GB VRAM
- Each component learns its task well
- Easier to debug
- Can reuse pretrained components

## Training Stages

```
Stage A: Thinker (Text)
    ↓
Stage B: Audio Encoder (ASR)
    ↓
Stage C: Vision Encoder
    ↓
Stage D: Talker + RVQ
    ↓
Stage E: Multimodal SFT (All Together)
```

## Stage A: Thinker Pretraining

### Goal
Train the core language model on text data.

### Process

```bash
python train_text.py --config configs/thinker_tiny.json
```

### What Happens

1. **Load Data**: Text corpus (`data/text/tiny_corpus.txt`)
2. **Tokenize**: Convert text to token IDs
3. **Create Batches**: Group sequences together
4. **Train**: Predict next token
5. **Save**: Checkpoint to `checkpoints/thinker_tiny/`

### Training Loop

```python
for batch in dataloader:
    # Input: [BOS] + tokens[:-1]
    # Target: tokens[1:]
    
    # Forward
    logits = model(input_ids)
    
    # Loss
    loss = cross_entropy(logits, target_ids)
    
    # Backward
    loss.backward()
    optimizer.step()
```

### Key Metrics

- **Loss**: Should decrease over time
- **Perplexity**: Lower is better (measure of uncertainty)
- **Validation Loss**: Check generalization

### Configuration

```json
{
  "max_steps": 1000,
  "batch_size": 8,
  "lr": 0.0003,
  "warmup_steps": 10
}
```

### Output

- `thinker.pt` - Model weights
- `tokenizer.model` - Trained tokenizer
- `thinker_best.pt` - Best validation model

## Stage B: Audio Encoder (ASR)

### Goal
Train audio encoder to transcribe speech.

### Process

```bash
python train_audio_enc.py --config configs/audio_enc_tiny.json
```

### What Happens

1. **Load Data**: Audio files + transcriptions (`data/audio/asr.csv`)
2. **Convert to Mel**: Audio → mel spectrogram
3. **Encode**: Audio encoder processes mel
4. **CTC Head**: Predicts text characters
5. **CTC Loss**: Handles alignment

### Training Loop

```python
for audio, text in dataloader:
    # Convert to mel
    mel = mel_spec(audio)
    
    # Encode
    embeddings = audio_encoder(mel)
    
    # Predict text
    logits = ctc_head(embeddings)
    
    # CTC loss (handles alignment)
    loss = ctc_loss(logits, text)
    
    loss.backward()
    optimizer.step()
```

### CTC (Connectionist Temporal Classification)

**Problem**: Audio has many frames, text has few tokens

**Solution**: CTC allows:
- Multiple frames per character
- Blank tokens
- Automatic alignment

### Output

- `audio_enc.pt` - Encoder weights + CTC head

## Stage C: Vision Encoder

### Goal
Train vision encoder on image classification.

### Process

```bash
python train_vision.py --config configs/vision_tiny.json
```

### What Happens

1. **Load Data**: Images + captions (`data/images/annotations.json`)
2. **Encode**: Vision encoder processes image
3. **Classify**: Predict label (e.g., "red" in caption?)
4. **Loss**: Cross-entropy classification

### Training Loop

```python
for image, caption in dataloader:
    # Encode
    cls_token, _ = vision_encoder(image)
    
    # Classify
    logits = classifier(cls_token)
    
    # Label: 0 if "red" in caption, else 1
    label = 0 if "red" in caption else 1
    loss = cross_entropy(logits, label)
    
    loss.backward()
    optimizer.step()
```

### Note

This is a simplified task. Real models use:
- Contrastive learning (CLIP)
- Image-text pairs
- Larger datasets

### Output

- `vision.pt` - Encoder weights + classifier head

## Stage D: Talker + RVQ Codec

### Goal
Train Talker to generate speech codes, and RVQ to quantize audio.

### Process

```bash
python train_talker.py --config configs/talker_tiny.json
```

### What Happens

1. **Load Data**: Audio files (`data/audio/tts.csv`)
2. **Convert to Mel**: Audio → mel spectrogram
3. **Encode**: RVQ encodes mel to codes
4. **Train Talker**: Predict codes autoregressively
5. **Train RVQ**: Codebooks learn to represent audio

### Training Loop

```python
for audio in dataloader:
    # Convert to mel
    mel = mel_spec(audio)
    
    # RVQ encode (learns codebooks)
    codes = rvq.encode(mel)  # (B, T, 2)
    
    # Shift codes (teacher forcing)
    prev_codes = torch.roll(codes, 1, dims=1)
    prev_codes[:, 0, :] = 0
    
    # Predict
    base_logits, res_logits = talker(prev_codes)
    
    # Loss on both codebooks
    loss = cross_entropy(base_logits, codes[:, :, 0]) + \
           cross_entropy(res_logits, codes[:, :, 1])
    
    loss.backward()
    optimizer.step()
```

### Key Points

- **RVQ and Talker trained together** - end-to-end
- **Teacher forcing** - use ground truth previous codes
- **Two codebooks** - base and residual

### Output

- `talker.pt` - Talker weights + RVQ codebooks

## Stage E: Multimodal SFT

### Goal
Fine-tune Thinker with multimodal data, train projectors.

### Process

```bash
python sft_omni.py --config configs/omni_sft_tiny.json
```

### What Happens

1. **Load Components**: Thinker, Audio Encoder, Vision Encoder (frozen)
2. **Load Data**: Mixed text, images, audio
3. **Encode Modalities**: Each encoder processes its modality
4. **Project**: Align to Thinker dimension
5. **Fuse**: Concatenate embeddings
6. **Train**: Thinker + projectors (encoders frozen)

### Training Loop

```python
for batch in dataloader:
    # Process each modality
    if has_image:
        img_emb = vision_encoder(image)  # Frozen
        img_emb = vision_projector(img_emb)  # Trainable
    
    if has_audio:
        audio_emb = audio_encoder(audio)  # Frozen
        audio_emb = audio_projector(audio_emb)  # Trainable
    
    # Text
    text_emb = thinker.tok_emb(text_ids)
    
    # Fuse
    combined = torch.cat([img_emb, audio_emb, text_emb], dim=1)
    
    # Thinker forward
    logits = thinker(embeddings=combined)
    
    # Loss (only on text tokens)
    loss = cross_entropy(logits, target_ids)
    
    loss.backward()
    optimizer.step()
```

### Key Points

- **Encoders frozen** - already trained
- **Projectors trainable** - learn to align modalities
- **Thinker fine-tuned** - learns multimodal understanding
- **Loss only on text** - multimodal tokens ignored

### Output

- `omni.pt` - Thinker + projectors

## Training Best Practices

### 1. Learning Rate Scheduling

```python
# Warmup + Cosine decay
warmup_steps = 10
max_steps = 1000

if step < warmup_steps:
    lr = base_lr * (step / warmup_steps)
else:
    lr = base_lr * 0.5 * (1 + cos(π * (step - warmup) / (max - warmup)))
```

### 2. Gradient Clipping

Prevent exploding gradients:

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

### 3. Validation

Check model performance:

```python
if step % val_freq == 0:
    model.eval()
    val_loss = validate(model, val_dataloader)
    if val_loss < best_loss:
        save_checkpoint(model, "best.pt")
    model.train()
```

### 4. Checkpointing

Save periodically:

```python
if step % checkpoint_freq == 0:
    save_checkpoint(model, f"step_{step}.pt")
```

### 5. Mixed Precision

Faster training with float16:

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

with autocast():
    output = model(input)
    loss = loss_fn(output, target)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

## Monitoring Training

### Key Metrics

1. **Loss**: Should decrease smoothly
2. **Learning Rate**: Check scheduler
3. **Gradient Norm**: Should be stable
4. **Validation Loss**: Should track training loss

### Common Issues

1. **Loss not decreasing**:
   - Check learning rate
   - Verify data loading
   - Check model initialization

2. **Loss exploding**:
   - Reduce learning rate
   - Add gradient clipping
   - Check for NaN values

3. **Overfitting**:
   - More data augmentation
   - Increase dropout
   - Early stopping

## Training Time Estimates

For tiny models on 12GB GPU:

| Stage | Steps | Time (approx) |
|-------|-------|---------------|
| Thinker | 1000 | 10-20 min |
| Audio Encoder | 500 | 5-10 min |
| Vision Encoder | 500 | 5-10 min |
| Talker | 500 | 5-10 min |
| Omni SFT | 2000 | 20-40 min |

**Total**: ~1-2 hours for full training

## Next Steps

After training:

1. **Evaluate**: Test on validation set
2. **Inference**: Try generating outputs
3. **Fine-tune**: Adjust hyperparameters
4. **Deploy**: Use in applications

---

**Next:**
- [08_Inference_Guide.md](08_Inference_Guide.md) - Using trained models
- [09_Hands_On_Exercises.md](09_Hands_On_Exercises.md) - Practice exercises

**See Also:**
- [Architecture Overview](02_Architecture_Overview.md)
- [Thinker Deep Dive](03_Thinker_Deep_Dive.md)

