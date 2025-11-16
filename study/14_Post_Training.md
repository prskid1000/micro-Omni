# Post-Training Guide

## Overview

Post-training allows you to continue training a model from a checkpoint using a **different dataset**. This is useful for:

> **📖 Related Guides**: 
> - [Quick Reference Guide](QUICK_REFERENCE.md) - Dataset setup and all commands
> - [Training Workflow](07_Training_Workflow.md) - Regular training process

- **Domain adaptation**: Fine-tune a general model on domain-specific data
- **Task-specific fine-tuning**: Adapt a pretrained model for specific tasks
- **Continued training**: Resume training on new data after initial training completes
- **Transfer learning**: Use a checkpoint as initialization for a new task

## Key Concepts

### Post-Training vs. Regular Training

| Aspect | Regular Training | Post-Training |
|--------|-----------------|---------------|
| **Starting Point** | Random initialization | Pre-trained checkpoint |
| **Dataset** | Same dataset | Different/new dataset |
| **Optimizer State** | Fresh | Optional (can reset for fine-tuning) |
| **Learning Rate** | Initial LR | Often lower (fine-tuning) |
| **Purpose** | Initial training | Adaptation/fine-tuning |

### When to Use Post-Training

✅ **Use post-training when:**
- You have a pretrained model and want to adapt it to new data
- You want to fine-tune on a smaller, domain-specific dataset
- You need to continue training after running out of initial data
- You want to transfer knowledge from one domain to another

❌ **Don't use post-training when:**
- You want to resume training on the same dataset (use regular training with checkpoint resume)
- You're starting from scratch (use regular training)

## Workflow

### Step 1: Prepare Post-Training Data

Prepare your new dataset in the same format as the original training data using the data preparation script:

**Text Data (for Thinker):**
```bash
python scripts/prep_post_training_data.py \
    --input data/your_new_text.txt \
    --output data/post_training/text.txt \
    --format text
```

**Audio ASR Data (for Audio Encoder):**
```bash
python scripts/prep_post_training_data.py \
    --input data/your_new_audio/ \
    --output data/post_training/asr.csv \
    --format audio_asr
```

**Audio TTS Data (for Talker):**
```bash
python scripts/prep_post_training_data.py \
    --input data/your_new_audio/ \
    --output data/post_training/tts.csv \
    --format audio_tts
```

**Image Data (for Vision Encoder):**
```bash
python scripts/prep_post_training_data.py \
    --input data/your_new_images/ \
    --output data/post_training/images.json \
    --format images \
    --caption_file data/raw/captions.txt  # Optional
```

**Script Features:**
- ✅ Automatic format conversion
- ✅ Automatic NaN recovery (Thinker models): Automatically reloads from checkpoint if NaN detected in attention
- ✅ Recursive file discovery (finds all files in subdirectories)
- ✅ Duplicate removal (for text data)
- ✅ Transcript/caption file support
- ✅ Progress reporting and error handling

**For more details on data formats, see [Quick Reference Guide](QUICK_REFERENCE.md)**

### Step 2: Choose a Checkpoint

Select the checkpoint to use as starting point:

- **Best model**: `checkpoints/thinker_tiny/thinker_best.pt` (lowest validation loss)
- **Latest checkpoint**: `checkpoints/thinker_tiny/thinker_step_5000.pt` (specific step)
- **Final model**: `checkpoints/thinker_tiny/thinker.pt` (end of training)

### Step 3: Configure Post-Training

Create or modify a config file for post-training. Key considerations:

```json
{
  "vocab_size": 15000,
  "n_layers": 4,
  "d_model": 256,
  "n_heads": 4,
  "d_ff": 1024,
  "dropout": 0.1,
  "rope_theta": 10000,
  "ctx_len": 512,
  "lr": 0.0001,          // Lower LR for fine-tuning (was 0.0003)
  "wd": 0.1,
  "warmup_steps": 100,   // Shorter warmup for fine-tuning
  "max_steps": 10000,    // Fewer steps needed
  "batch_size": 8,
  "max_epochs": 3,
  "save_dir": "checkpoints/thinker_post_trained"
}
```

**Important differences:**
- **Lower learning rate**: Typically 1/3 to 1/10 of original (e.g., 0.0001 instead of 0.0003)
- **Shorter warmup**: Fewer warmup steps since model is already trained
- **Fewer steps**: Less training needed since starting from pretrained weights

### Step 4: Run Post-Training

The post-training script supports all model types:
- **Thinker** (text LLM)
- **Audio Encoder** (ASR)
- **Vision Encoder** (image-caption)
- **Talker** (TTS)

#### Basic Post-Training (Continue Training)

**Thinker (Text):**
```bash
python post_train.py \
    --config configs/thinker_tiny.json \
    --checkpoint checkpoints/thinker_tiny/thinker_best.pt \
    --new_dataset data/post_training/text.txt
```

**Audio Encoder (ASR):**
```bash
python post_train.py \
    --config configs/audio_enc_tiny.json \
    --checkpoint checkpoints/audio_enc_tiny/audio_enc.pt \
    --new_dataset data/post_training/asr.csv
```

**Vision Encoder:**
```bash
python post_train.py \
    --config configs/vision_tiny.json \
    --checkpoint checkpoints/vision_tiny/vision.pt \
    --new_dataset data/post_training/images.json
```

**Talker (TTS):**
```bash
python post_train.py \
    --config configs/talker_tiny.json \
    --checkpoint checkpoints/talker_tiny/talker.pt \
    --new_dataset data/post_training/tts.csv
```

This will:
- Auto-detect model type from config
- Load the checkpoint
- **Automatic resumption**: If interrupted, automatically resumes from latest `{model}_post_step_*.pt`
- Continue optimizer/scheduler state
- Continue step counter
- Train on the new dataset

#### Fine-Tuning (Reset Optimizer)

```bash
python post_train.py \
    --config configs/thinker_tiny.json \
    --checkpoint checkpoints/thinker_tiny/thinker_best.pt \
    --new_dataset data/post_training/text.txt \
    --reset_optimizer \
    --reset_scheduler \
    --reset_step \
    --lr 0.0001
```

This will:
- Load model weights only
- Reset optimizer state (fresh optimizer)
- Reset scheduler state (fresh scheduler)
- Reset step counter to 0
- Use lower learning rate for fine-tuning

#### Custom Hyperparameters

```bash
python post_train.py \
    --config configs/thinker_tiny.json \
    --checkpoint checkpoints/thinker_tiny/thinker_best.pt \
    --new_dataset data/post_training/text.txt \
    --reset_optimizer \
    --lr 0.00005 \
    --wd 0.01 \
    --warmup_steps 50 \
    --max_steps 5000 \
    --save_dir checkpoints/thinker_finetuned
```

## Command-Line Options

### Required Arguments

- `--config`: Path to config JSON file (defines model architecture)
- `--checkpoint`: Path to checkpoint file (.pt) to load
- `--new_dataset`: Path to new dataset file

### Optional Arguments

- `--reset_optimizer`: Reset optimizer state (recommended for fine-tuning)
- `--reset_scheduler`: Reset scheduler state (recommended for fine-tuning)
- `--reset_step`: Reset step counter to 0 (recommended for fine-tuning)
- `--lr`: Override learning rate (recommended: 1/3 to 1/10 of original)
- `--wd`: Override weight decay
- `--warmup_steps`: Override warmup steps
- `--max_steps`: Override max training steps
- `--save_dir`: Override save directory

## Post-Training Strategies

### Strategy 1: Domain Adaptation

**Goal**: Adapt a general model to a specific domain (e.g., medical, legal, technical)

```bash
# Lower LR, reset optimizer, moderate training
python post_train.py \
    --config configs/thinker_tiny.json \
    --checkpoint checkpoints/thinker_tiny/thinker_best.pt \
    --new_dataset data/domain_specific/medical_text.txt \
    --reset_optimizer \
    --lr 0.0001 \
    --max_steps 20000
```

**Settings:**
- Learning rate: 0.0001 (1/3 of original)
- Reset optimizer: Yes
- Training steps: 20,000-50,000
- Purpose: Adapt general knowledge to domain

### Strategy 2: Task-Specific Fine-Tuning

**Goal**: Fine-tune for a specific task (e.g., question answering, summarization)

```bash
# Very low LR, reset everything, short training
python post_train.py \
    --config configs/thinker_tiny.json \
    --checkpoint checkpoints/thinker_tiny/thinker_best.pt \
    --new_dataset data/task_specific/qa_pairs.txt \
    --reset_optimizer \
    --reset_scheduler \
    --reset_step \
    --lr 0.00005 \
    --warmup_steps 50 \
    --max_steps 5000
```

**Settings:**
- Learning rate: 0.00005 (1/6 of original)
- Reset everything: Yes
- Training steps: 5,000-10,000
- Purpose: Minimal adaptation for specific task

### Strategy 3: Continued Training

**Goal**: Continue training on new data (same domain, more data)

```bash
# Keep optimizer state, similar LR, longer training
python post_train.py \
    --config configs/thinker_tiny.json \
    --checkpoint checkpoints/thinker_tiny/thinker.pt \
    --new_dataset data/continued/new_batch.txt \
    --lr 0.0002 \
    --max_steps 100000
```

**Settings:**
- Learning rate: 0.0002 (slightly lower than original)
- Reset optimizer: No (continue training)
- Training steps: 100,000+ (full training)
- Purpose: Extend training with more data

## Checkpoint Structure

Post-training checkpoints include metadata about the source:

```python
{
    "model": {...},              # Model weights
    "optimizer": {...},          # Optimizer state
    "scheduler": {...},          # Scheduler state
    "scaler": {...},             # AMP scaler state
    "step": 5000,                # Current step
    "best_val_loss": 2.345,      # Best validation loss
    "source_checkpoint": "checkpoints/thinker_tiny/thinker_best.pt",
    "post_training": True        # Flag indicating post-training
}
```

## Best Practices

### 1. Learning Rate Selection

**Rule of thumb**: Use 1/3 to 1/10 of original learning rate

| Original LR | Fine-Tuning LR | Use Case |
|------------|----------------|----------|
| 0.0003 | 0.0001 | Domain adaptation |
| 0.0003 | 0.00005 | Task-specific fine-tuning |
| 0.0003 | 0.0002 | Continued training |

### 2. When to Reset Optimizer

**Reset optimizer when:**
- Fine-tuning on a different domain
- Task-specific adaptation
- Dataset is significantly different

**Keep optimizer when:**
- Continuing training on similar data
- Just adding more data to same domain
- Want to maintain training momentum

### 3. Training Duration

| Strategy | Steps | Epochs |
|----------|-------|--------|
| Fine-tuning | 5,000-10,000 | 1-2 |
| Domain adaptation | 20,000-50,000 | 2-5 |
| Continued training | 100,000+ | 3+ |

### 4. Monitoring

Watch for:
- **Overfitting**: Validation loss increases while training loss decreases
- **Catastrophic forgetting**: Model forgets original knowledge
- **Learning rate too high**: Loss spikes or becomes unstable
- **Learning rate too low**: Very slow improvement

### 5. Validation

Always validate on:
- **Original validation set**: Check if model still performs on original task
- **New validation set**: Check performance on new domain/task
- **Both**: Ensure balanced performance

## Examples

### Example 1: Medical Domain Adaptation

```bash
# Prepare medical text data
python scripts/prep_post_training_data.py \
    --input data/medical/raw_text.txt \
    --output data/post_training/medical.txt \
    --format text

# Fine-tune on medical data
python post_train.py \
    --config configs/thinker_tiny.json \
    --checkpoint checkpoints/thinker_tiny/thinker_best.pt \
    --new_dataset data/post_training/medical.txt \
    --reset_optimizer \
    --lr 0.0001 \
    --warmup_steps 100 \
    --max_steps 20000 \
    --save_dir checkpoints/thinker_medical
```

### Example 2: Code Generation Fine-Tuning

```bash
# Prepare code dataset
python scripts/prep_post_training_data.py \
    --input data/code/python_files/ \
    --output data/post_training/code.txt \
    --format text

# Fine-tune for code generation
python post_train.py \
    --config configs/thinker_tiny.json \
    --checkpoint checkpoints/thinker_tiny/thinker_best.pt \
    --new_dataset data/post_training/code.txt \
    --reset_optimizer \
    --reset_scheduler \
    --reset_step \
    --lr 0.00005 \
    --warmup_steps 50 \
    --max_steps 10000 \
    --save_dir checkpoints/thinker_code
```

### Example 3: Multilingual Adaptation

```bash
# Prepare multilingual data
python scripts/prep_post_training_data.py \
    --input data/multilingual/combined.txt \
    --output data/post_training/multilingual.txt \
    --format text

# Adapt to multilingual
python post_train.py \
    --config configs/thinker_tiny.json \
    --checkpoint checkpoints/thinker_tiny/thinker_best.pt \
    --new_dataset data/post_training/multilingual.txt \
    --reset_optimizer \
    --lr 0.00008 \
    --max_steps 30000 \
    --save_dir checkpoints/thinker_multilingual
```

## Troubleshooting

### Issue: Model Performance Degrades

**Symptoms**: Validation loss increases, model forgets original knowledge

**Solutions:**
- Lower learning rate further (try 0.00003)
- Reduce training steps
- Use smaller dataset
- Add regularization (increase dropout)

### Issue: Model Doesn't Adapt

**Symptoms**: Loss doesn't decrease, model doesn't learn new patterns

**Solutions:**
- Increase learning rate (try 0.00015)
- Reset optimizer state
- Train for more steps
- Check dataset quality

### Issue: Training Unstable

**Symptoms**: Loss spikes, NaN values, gradient explosion

**Solutions:**
- Lower learning rate
- Increase gradient clipping (`max_grad_norm`)
- Reduce batch size
- Disable mixed precision

**Automatic Recovery (Thinker Models Only):**
- When `RuntimeError: NaN detected in attention probabilities after softmax` occurs:
  - **Automatic**: The script automatically reloads from the last saved checkpoint
  - **No manual intervention**: Training continues from the recovered checkpoint
  - **Checkpoint prefix**: Uses `{model_type}_post_step_` (e.g., `thinker_post_step_500.pt`)
  - **Full state recovery**: Model, optimizer, scheduler, and scaler states are restored
  - **Logging**: Recovery actions are logged with checkpoint paths and step numbers
  - **Resume behavior**: Training resumes from the exact step of the recovered checkpoint

### Feature: Automatic Resumption After Interruption

**All model types** now support automatic resumption if training is interrupted (Ctrl+C, crash, power loss, etc.):

**How it works:**
1. On startup, `post_train.py` scans `save_dir` for existing `{model}_post_step_*.pt` files
2. If found, automatically loads the **latest checkpoint** (highest step number)
3. Resumes training from that exact step (skips already-processed batches)
4. Preserves all training state: model, optimizer, scheduler, scaler, best_val_loss

**Example scenario:**
```bash
# Start post-training
python post_train.py \
    --config configs/thinker_tiny.json \
    --checkpoint checkpoints/thinker_tiny/thinker.pt \
    --new_dataset data/post_training/text.txt

# Training runs... saves at step 500, 1000, 1500...
# At step 1823: Training interrupted (Ctrl+C / crash)

# Simply re-run the same command:
python post_train.py \
    --config configs/thinker_tiny.json \
    --checkpoint checkpoints/thinker_tiny/thinker.pt \
    --new_dataset data/post_training/text.txt

# Output:
# 🔄 Found existing post-training checkpoint: thinker_post_step_1500.pt
# Resuming from step 1500...
# ✓ Model state resumed
# ✓ Optimizer state resumed
# ✓ Scheduler state resumed
# ✓ Successfully resumed from step 1500
# Training continues from step 1500...
```

**Benefits:**
- ✅ **Zero configuration**: No special flags needed, just re-run the same command
- ✅ **Safe**: Never loses progress beyond last checkpoint interval
- ✅ **Efficient**: Skips already-processed batches automatically
- ✅ **Smart**: Only resumes if post-training checkpoints exist (not initial checkpoint)
- ✅ **Overridable**: Use `--reset_step` to force restart from initial checkpoint

**To force restart** (ignore existing post-training checkpoints):
```bash
python post_train.py \
    --config configs/thinker_tiny.json \
    --checkpoint checkpoints/thinker_tiny/thinker.pt \
    --new_dataset data/post_training/text.txt \
    --reset_step  # ← Forces restart from step 0
```

**Checkpoint cleanup:** Only the **last** periodic checkpoint is kept (plus best and final), saving 99%+ disk space!

### Issue: Checkpoint Loading Fails

**Symptoms**: Key mismatch errors, shape mismatches

**Solutions:**
- Check model architecture matches config
- Use `strict=False` (handled automatically)
- Verify checkpoint format
- Ensure vocab size matches

## Advanced Topics

### Layer Freezing

To freeze certain layers during post-training, modify the script to:

```python
# Freeze embedding layer
for param in model.tok_emb.parameters():
    param.requires_grad = False

# Freeze first N layers
for i in range(2):  # Freeze first 2 layers
    for param in model.blocks[i].parameters():
        param.requires_grad = False
```

### Differential Learning Rates

Apply different learning rates to different layers:

```python
# Higher LR for new layers, lower for pretrained
optimizer = torch.optim.AdamW([
    {'params': model.blocks[:2].parameters(), 'lr': 0.00005},  # Pretrained layers
    {'params': model.blocks[2:].parameters(), 'lr': 0.0001},   # Later layers
    {'params': model.lm_head.parameters(), 'lr': 0.0001}        # Output layer
], weight_decay=0.1)
```

### Progressive Unfreezing

Gradually unfreeze layers during training:

```python
# Start with only output layer trainable
# After N steps, unfreeze last block
# After M steps, unfreeze all layers
```

## Summary

Post-training is a powerful technique for adapting pretrained models to new domains or tasks. Key points:

1. **Use lower learning rates** (1/3 to 1/10 of original)
2. **Reset optimizer for fine-tuning**, keep it for continued training
3. **Monitor both old and new validation sets** to prevent catastrophic forgetting
4. **Start with shorter training** and extend if needed
5. **Prepare data properly** using the data preparation script

For more information, see:
- [Training Workflow](07_Training_Workflow.md) - Regular training process
- [Checkpoint Structure](13_CHECKPOINT_STRUCTURE.md) - Checkpoint format details
- [Inference Guide](08_Inference_Guide.md) - Using post-trained models

