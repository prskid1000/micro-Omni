# Training Workflow

## Overview

μOmni uses a **staged training approach** - train components separately, then combine them. This document explains the **theoretical foundations** behind our specific training implementation, based on the actual code in `train_*.py` files.

## Theoretical Foundation: Why Staged Training?

### The Memory Constraint Problem

**End-to-end training challenge**:
- All components active simultaneously
- Gradients flow through all components
- Memory: O(batch_size × sequence_length × model_size)
- For μOmni: Would exceed 12GB VRAM

**Our staged solution**:
- Train one component at a time
- Memory: O(batch_size × sequence_length × single_component_size)
- Fits comfortably in 12GB VRAM

### The Learning Efficiency Theory

**Why staged works better**:

1. **Specialization**: Each component focuses on one task
   - Thinker: Language modeling only
   - Audio Encoder: Speech recognition only
   - Vision Encoder: Image understanding only
   - Each becomes expert in its domain

2. **Stability**: Easier to train and debug
   - Can monitor each component separately
   - Identify issues early
   - Adjust hyperparameters per component

3. **Reusability**: Pretrained components are valuable
   - Can reuse in other projects
   - Can improve individually
   - Can swap with better pretrained models

4. **Modularity**: Clear separation of concerns
   - Each stage has clear objective
   - Can understand what each component learns
   - Easier to modify and improve

### The Curriculum Learning Perspective

Staged training is a form of **curriculum learning**:
1. **Stage A**: Learn basic language (Thinker)
2. **Stage B**: Learn speech understanding (Audio)
3. **Stage C**: Learn vision understanding (Vision)
4. **Stage D**: Learn speech generation (Talker)
5. **Stage E**: Learn multimodal integration (SFT)

Each stage builds on previous knowledge, creating a **learning curriculum**.

**Why this works**:
- Foundation first (language)
- Then specialized skills (modalities)
- Finally integration (multimodal)
- Natural progression of complexity

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

### Diagram 1: Staged Training Pipeline

```mermaid
graph TD
    Start[Start] --> StageA[Stage A:<br/>Thinker<br/>Text Only]
    StageA --> StageB[Stage B:<br/>Audio Encoder<br/>ASR]
    StageB --> StageC[Stage C:<br/>Vision Encoder<br/>Image Classification]
    StageC --> StageD[Stage D:<br/>Talker + RVQ<br/>TTS]
    StageD --> StageE[Stage E:<br/>Multimodal SFT<br/>All Together]
    StageE --> Done[Training Complete]
    
    style StageA fill:#3498db
    style StageB fill:#9b59b6
    style StageC fill:#9b59b6
    style StageD fill:#9b59b6
    style StageE fill:#27ae60
```

**Explanation**: Training proceeds in stages - first foundation (Thinker), then specialized encoders (Audio, Vision), then generation (Talker), finally multimodal integration (SFT). Each stage builds on previous knowledge.

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

### Training Loop (From `train_text.py`)

```python
# Actual code from train_text.py
for x, y in tqdm(train_dl, desc=f"epoch{epoch}"):
    x, y = x.to(device), y.to(device)
    logits = model(x)  # (B, T, V)
    loss = loss_fn(logits.view(-1, logits.size(-1)), y.view(-1))
    opt.zero_grad()
    loss.backward()
    
    # Gradient clipping (our implementation)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    
    opt.step()
    scheduler.step()
```

### Deep Theoretical Analysis: Next-Token Prediction

#### Why This Objective Works

**The language modeling objective**:
- **Input**: `[BOS] + tokens[:-1]` - context up to position t-1
- **Target**: `tokens[1:]` - token at position t
- **Goal**: Predict next token given all previous tokens

**Why this is powerful**:
- **Self-supervised**: No labels needed (text is its own label)
- **Scalable**: Can use massive text corpora
- **General**: Learns general language understanding
- **Proven**: Foundation of GPT, LLaMA, etc.

#### The Shifted Sequence Trick

**Key insight**: Use the same sequence for input and target, just shifted:

```
Input:  [BOS, "The", "cat", "sat"]
Target: ["The", "cat", "sat", "on"]
```

**Why this works**:
- Efficient: No need to create separate input/target pairs
- Natural: Each position learns to predict from its context
- Parallel: Can process all positions simultaneously

#### Loss Computation (Our Implementation)

**From `train_text.py`**:
```python
loss = loss_fn(logits.view(-1, logits.size(-1)), y.view(-1))
```

**What happens**:
- `logits`: (B, T, vocab_size) → (B×T, vocab_size)
- `y`: (B, T) → (B×T,)
- Loss computed for all positions simultaneously

**Why flatten?**
- Cross-entropy expects 2D logits and 1D targets
- Efficient batch processing
- Standard PyTorch pattern

#### Padding and Ignore Index

**Our implementation**:
```python
loss_fn = nn.CrossEntropyLoss(ignore_index=0)
```

**Why ignore padding (0)?**
- Sequences have different lengths
- Must pad to same length for batching
- Don't want to learn from padding tokens
- `ignore_index=0` excludes padding from loss

**What value do we get?**
- Efficient batching (variable lengths)
- No learning from padding
- Cleaner training signal

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

### Training Loop (From `train_talker.py`)

```python
# Actual code from train_talker.py
for mel in tqdm(train_dl, desc=f"epoch{epoch}"):
    mel = mel.to(device)  # (B, T, 128)
    
    # Batch encode all frames at once (optimized)
    idxs = rvq.encode(mel)  # (B, T, 2) - encodes all frames in batch
    
    # AR training: predict current codes from previous codes
    prev = torch.roll(idxs, 1, dims=1)  # Shift by one
    prev[:, 0, :] = 0  # First frame is zero
    
    base_logit, res_logit = talker(prev)
    
    # Loss on both codebooks
    loss = loss_fn(base_logit.reshape(-1, base_logit.size(-1)), idxs[:, :, 0].reshape(-1)) + \
           loss_fn(res_logit.reshape(-1, res_logit.size(-1)), idxs[:, :, 1].reshape(-1))
    
    opt.zero_grad()
    loss.backward()
    clip_gradients(rvq, max_grad_norm)
    clip_gradients(talker, max_grad_norm)
    opt.step()
    scheduler.step()
```

### Deep Theoretical Analysis: Teacher Forcing

#### Why `torch.roll` for Teacher Forcing?

**The teacher forcing trick**:
```python
prev = torch.roll(idxs, 1, dims=1)  # Shift right by one
prev[:, 0, :] = 0  # First position is zero
```

**What this does**:
- **Before shift**: `[code0, code1, code2, code3]`
- **After shift**: `[0, code0, code1, code2]`
- **Target**: `[code0, code1, code2, code3]`

**Result**: Model learns to predict `code_t` from `code_{t-1}`

#### Why Teacher Forcing Works

**Training with teacher forcing**:
- Use **ground truth** previous codes
- Fast convergence
- Stable gradients
- Standard practice

**Inference without teacher forcing**:
- Use **predicted** previous codes
- Realistic generation
- Can accumulate errors
- Requires careful generation

**Why both?**
- **Training**: Efficiency and stability (teacher forcing)
- **Inference**: Realism (autoregressive)

#### The Two-Codebook Loss

**Our implementation**:
```python
loss = loss_base + loss_residual
```

**Why sum both losses?**
- **Base codebook**: Captures coarse features
- **Residual codebook**: Captures fine details
- **Both matter**: Need both for good reconstruction
- **Equal weight**: Both losses contribute equally

**Alternative**: Weighted sum (e.g., 0.7 base + 0.3 residual)
- Could emphasize one codebook
- But equal weights work well in practice

#### RVQ and Talker Joint Training

**Why train together?**
- **End-to-end**: Codebooks learn optimal representations
- **Adaptive**: Codebooks adapt to Talker's needs
- **Better quality**: Joint optimization improves reconstruction

**What gets learned**:
- **RVQ codebooks**: Learn to represent audio efficiently
- **Talker**: Learns to predict codes autoregressively
- **Together**: Optimal codebook-Talker pairing

#### What Value Do We Get?

1. **Efficient Training**: Teacher forcing speeds up training
2. **Joint Optimization**: RVQ and Talker learn together
3. **Quality**: Better than training separately
4. **Stability**: Stable training with gradient clipping
5. **Flexibility**: Can generate variable-length audio

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

### Training Loop (From `sft_omni.py`)

```python
# Actual code from sft_omni.py
for batch_idx, data in enumerate(train_dl):
    # Process batch (handles variable modalities)
    batch_emb, batch_targets, batch_mask = process_batch(data, is_training=True)
    
    # Forward pass
    logits = think(embeddings=batch_emb)  # (B, T, vocab)
    
    # Calculate loss (mask out padding)
    loss = loss_fn(logits.view(-1, logits.size(-1)), batch_targets.view(-1))
    
    opt.zero_grad()
    loss.backward()
    
    # Gradient clipping (our implementation)
    clip_gradients(think, max_grad_norm)
    clip_gradients(proj_a, max_grad_norm)
    clip_gradients(proj_v, max_grad_norm)
    
    opt.step()
    scheduler.step()
```

### Deep Theoretical Analysis: Multimodal SFT

#### Why Freeze Encoders?

**Our implementation** (from `sft_omni.py`):
```python
# Encoders are loaded but NOT in optimizer
# Only Thinker and projectors are trainable
opt = torch.optim.AdamW(
    list(think.parameters()) + 
    list(proj_a.parameters()) + 
    list(proj_v.parameters()),
    lr=cfg["lr"]
)
# Note: vis and aud are NOT in optimizer (frozen)
```

**Why freeze?**
- **Already trained**: Encoders learned good representations
- **Focus SFT**: Focus on alignment, not encoding
- **Stability**: Frozen encoders = stable training
- **Efficiency**: Fewer parameters to update

**What gets learned in SFT**:
- **Projectors**: How to align modalities to Thinker space
- **Thinker**: How to process multimodal sequences
- **Cross-modal relationships**: Connections between modalities

#### The Batch Processing Strategy

**Our `process_batch` function** (from `sft_omni.py`):
1. **Batch images**: Process all images together
2. **Batch audio**: Process all audio together (with padding)
3. **Process text**: Per sample (then combine)
4. **Fuse**: Concatenate embeddings per sample
5. **Pad**: Pad to same length for final batch

**Why this strategy?**
- **Efficiency**: Batch processing is faster
- **Flexibility**: Handles variable modalities per sample
- **Memory**: Efficient use of GPU memory

#### Context Allocation Theory

**Our implementation**:
```python
# Limit audio length
max_audio_tokens = cfg["ctx_len"] // 4  # 512 // 4 = 128 tokens

# Calculate remaining context for text
multimodal_len = sum(emb.shape[1] for emb in multimodal_emb_list)
max_text_len = cfg["ctx_len"] - multimodal_len - 1
```

**Why ctx_len // 4 for audio?**
- **Balance**: Leaves room for text
- **Practical**: Audio can be long, need to limit
- **Empirical**: Works well in practice

**Why dynamic text allocation?**
- **Flexible**: More multimodal → less text (and vice versa)
- **Efficient**: Uses full context
- **Adaptive**: Adjusts to input

#### Loss Only on Text Tokens

**Our implementation**:
```python
# Loss computed on all tokens, but...
# Multimodal tokens have padding in targets (ignored by loss)
multimodal_padding = torch.zeros(multimodal_len, dtype=y_ids.dtype, device=device)
y_ids = torch.cat([multimodal_padding, y_ids], dim=0)
```

**Why ignore multimodal tokens in loss?**
- **Objective**: Learn to generate text from multimodal input
- **Not generating**: Not generating images/audio, only text
- **Focus**: Focus learning on text generation

**What value do we get?**
- **Clear objective**: Generate text, not modalities
- **Efficient**: Don't waste loss on non-text tokens
- **Standard**: Matches multimodal LLM training

#### What Value Do We Get from SFT?

1. **Multimodal Understanding**: Learns cross-modal relationships
2. **Alignment**: Projectors align modalities to Thinker space
3. **Integration**: Thinker learns to process multimodal sequences
4. **Efficiency**: Frozen encoders = faster training
5. **Flexibility**: Handles any combination of modalities

### Key Points

- **Encoders frozen** - already trained
- **Projectors trainable** - learn to align modalities
- **Thinker fine-tuned** - learns multimodal understanding
- **Loss only on text** - multimodal tokens ignored

### Output

- `omni.pt` - Thinker + projectors

## Training Best Practices (Our Implementation)

All training scripts include specific techniques from our codebase:

### Learning Rate Scheduling (From `omni/training_utils.py`)

**Our implementation**:
```python
def get_lr_scheduler(optimizer, warmup_steps, max_steps, min_lr_ratio=0.1):
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)  # Linear warmup
        else:
            # Cosine decay
            progress = (step - warmup_steps) / max((max_steps - warmup_steps), 1)
            return min_lr_ratio + (1 - min_lr_ratio) * 0.5 * (1 + torch.cos(torch.tensor(progress * math.pi)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
```

#### Why Warmup?

**The problem**: Large learning rates at start can cause:
- Gradient explosion
- Training instability
- Poor convergence

**Our solution**: Linear warmup
- Start: `lr = 0`
- End of warmup: `lr = base_lr`
- Gradual increase prevents instability

**Why this works**:
- **Stable start**: Small gradients initially
- **Gradual increase**: Model adapts gradually
- **Proven**: Standard in transformer training

#### Why Cosine Decay?

**Our implementation**: Cosine decay after warmup
- **Start**: `lr = base_lr` (after warmup)
- **End**: `lr ≈ min_lr_ratio * base_lr` (at max_steps)
- **Shape**: Cosine curve (smooth decrease)

**Why cosine?**
- **Smooth**: Gradual decrease (no sudden drops)
- **Exploration**: High LR early (explore)
- **Refinement**: Low LR late (refine)
- **Proven**: Used in many successful models

### Gradient Clipping (Our Implementation)

**From `omni/training_utils.py`**:
```python
def clip_gradients(model, max_norm=1.0):
    return torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
```

**Why gradient clipping?**
- **Prevents explosion**: Large gradients → weight explosion
- **Stability**: Keeps training stable
- **Convergence**: Helps model converge

**Why max_norm=1.0?**
- **Empirical**: Works well in practice
- **Conservative**: Prevents extreme gradients
- **Standard**: Common value in transformer training

**What value do we get?**
- **Stable training**: No gradient explosions
- **Better convergence**: Smoother optimization
- **Robustness**: Handles difficult batches

### Validation Strategy (Our Implementation)

**From `train_text.py`**:
```python
# Validation every N steps
if step % val_freq == 0 and step > 0:
    model.eval()
    val_loss_sum = 0.0
    val_count = 0
    with torch.no_grad():
        for val_x, val_y in val_dl:
            # ... compute loss ...
            if val_count >= 20:  # Limit validation batches
                break
    
    avg_val_loss = val_loss_sum / val_count
    
    # Save best model
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), "thinker_best.pt")
```

#### Why Periodic Validation?

**The problem**: Training loss can be misleading
- May overfit to training data
- Need to check generalization

**Our solution**: Validate every N steps
- **Frequency**: `val_freq` (e.g., 200 steps)
- **Limited batches**: Only 20 batches (efficiency)
- **Best model**: Save when validation improves

**Why limit to 20 batches?**
- **Efficiency**: Full validation is slow
- **Representative**: 20 batches is enough
- **Practical**: Balance between accuracy and speed

#### What Value Do We Get?

1. **Generalization Check**: Monitors overfitting
2. **Best Model**: Saves best validation model
3. **Early Stopping**: Can stop if validation plateaus
4. **Efficiency**: Limited batches = fast validation
5. **Practical**: Works well in practice

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

