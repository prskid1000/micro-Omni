# Chapter 26: Training Workflow Overview

[← Previous: Multimodal Fusion](25-multimodal-fusion.md) | [Back to Index](00-INDEX.md) | [Next: Stage A →](27-stage-a-thinker.md)

---

## 🎯 Learning Objectives

By the end of this chapter, you will understand:
- Why μOmni uses a 5-stage training pipeline
- The purpose and goal of each training stage
- How modular training works and its benefits
- The dependencies between stages
- Resource requirements and time estimates
- Training strategy and design philosophy

---

## 💡 Why 5 Stages? The Training Philosophy

### The Challenge of Multimodal Training

**Analogy: Building a Symphony Orchestra**

```
Think of training μOmni like forming an orchestra:

NAIVE APPROACH (train everything together):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Gather musicians who've never played before
Give them symphony sheet music
Tell them: "Play Beethoven's 9th!"

Problems:
❌ Too many things to learn at once
❌ Can't tell which section is struggling
❌ Everyone gets confused
❌ Results: Terrible noise!

STAGED APPROACH (train progressively):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Stage 1: String section practices alone
Stage 2: Wind section practices alone
Stage 3: Brass section practices alone
Stage 4: Percussion section practices alone
Stage 5: All sections play together!

Benefits:
✅ Each section masters their part
✅ Can identify and fix issues per section
✅ Gradual integration
✅ Results: Beautiful symphony! ✓

μOmni TRAINING (same idea!):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Stage A: Thinker learns text (the foundation)
Stage B: Audio Encoder learns sound (audio section)
Stage C: Vision Encoder learns images (vision section)
Stage D: Talker learns speech generation (speech section)
Stage E: All components work together! (full orchestra)

Progressive, modular, effective! ✓
```

**Why Not Train Everything Together?**

```
Problems with joint training from scratch:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. GRADIENT CONFLICTS:
   - Vision gradient pulls one way
   - Audio gradient pulls another way
   - Text gradient pulls a third way
   - Thinker gets confused: "Which to optimize?"
   - Result: Poor convergence ❌

2. DEBUGGING NIGHTMARE:
   - Model doesn't work well
   - Is it the Thinker? Vision? Audio? Talker?
   - Can't isolate the problem!
   - Waste days debugging ❌

3. RESOURCE INTENSIVE:
   - Need ALL data types simultaneously
   - Huge memory footprint
   - Long training time with no checkpoints ❌

4. UNSTABLE TRAINING:
   - Some components learn faster than others
   - Imbalanced learning
   - Hard to tune learning rates ❌

Benefits of staged training:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. FOCUSED LEARNING:
   - Each component has clear objective
   - No conflicting gradients
   - Stable, predictable convergence ✓

2. EASY DEBUGGING:
   - Stage B fails? Problem is in Audio Encoder
   - Can fix and retrain just that stage
   - Save tons of development time ✓

3. RESOURCE EFFICIENT:
   - Train on one modality at a time
   - Smaller memory footprint
   - Can parallelize (train stages simultaneously) ✓

4. MODULAR DEVELOPMENT:
   - Different people can work on different stages
   - Can reuse components (e.g., swap better Vision Encoder)
   - Flexible experimentation ✓

This is why μOmni uses 5 stages!
```

---

## 🏗️ The 5-Stage Training Pipeline

### Complete Overview

```
┌─────────────────────────────────────────────┐
│ STAGE A: Thinker Pretraining               │
│ ═════════════════════════════════════════   │
│ Purpose: Learn language understanding      │
│ Model: Thinker (decoder-only LLM)          │
│ Task: Predict next word                    │
│ Data: Text corpus (books, articles)        │
│ Loss: Cross-entropy (next token)           │
│ Metric: Perplexity (lower = better)        │
│ Time: ~8-12 hours on 12GB GPU              │
│ Output: thinker_checkpoints/               │
└──────────────────┬──────────────────────────┘
                   ↓
         Foundation is ready!
                   ↓
┌─────────────────────────────────────────────┐
│ STAGE B: Audio Encoder Pretraining         │
│ ═════════════════════════════════════════   │
│ Purpose: Learn audio understanding         │
│ Model: Audio Encoder (AuT-Tiny)            │
│ Task: Speech recognition (ASR)             │
│ Data: Audio + transcriptions               │
│ Loss: CTC (alignment-free)                 │
│ Metric: WER (Word Error Rate)              │
│ Time: ~6-10 hours                          │
│ Output: audio_encoder_checkpoints/         │
└──────────────────┬──────────────────────────┘
                   ↓
                   │
┌─────────────────────────────────────────────┐
│ STAGE C: Vision Encoder Training           │
│ ═════════════════════════════════════════   │
│ Purpose: Learn visual understanding        │
│ Model: Vision Encoder (ViT-Tiny)           │
│ Task: Image classification                 │
│ Data: Images + labels                      │
│ Loss: Cross-entropy (classification)       │
│ Metric: Accuracy                           │
│ Time: ~4-8 hours                           │
│ Output: vision_encoder_checkpoints/        │
└──────────────────┬──────────────────────────┘
                   ↓
                   │
┌─────────────────────────────────────────────┐
│ STAGE D: Talker + RVQ Codec Training       │
│ ═════════════════════════════════════════   │
│ Purpose: Learn speech generation           │
│ Models: RVQ Codec + Talker                 │
│ Task: Predict speech codes                 │
│ Data: Speech audio files                   │
│ Loss: MSE (RVQ) + Cross-entropy (Talker)   │
│ Metric: Reconstruction quality            │
│ Time: ~10-15 hours                         │
│ Output: rvq_codec/ + talker_checkpoints/   │
└──────────────────┬──────────────────────────┘
                   ↓
      All components ready!
                   ↓
┌─────────────────────────────────────────────┐
│ STAGE E: Multimodal SFT                    │
│ ═════════════════════════════════════════   │
│ Purpose: Teach multimodal understanding    │
│ Models: ALL (Thinker + Encoders)           │
│ Task: Answer multimodal queries            │
│ Data: Image+text, audio+text pairs         │
│ Loss: Cross-entropy (response generation)  │
│ Metric: Task accuracy                      │
│ Time: ~6-12 hours                          │
│ Output: omni_final/                        │
└─────────────────────────────────────────────┘
                   ↓
      μOmni is ready! 🎉
```

### Detailed Stage Breakdown

```
STAGE A: The Foundation (Text-Only)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Goal: Thinker must understand language before multimodal

What it learns:
- Grammar and syntax
- Common sense reasoning
- World knowledge
- Next token prediction

Think: Teaching reading before showing pictures

STAGE B: Understanding Sound (Audio)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Goal: Audio Encoder converts speech → meaningful embeddings

What it learns:
- Phonemes and words from audio
- Temporal patterns in speech
- Acoustic features
- Alignment between audio and text (via CTC)

Think: Teaching listening comprehension

STAGE C: Understanding Sight (Vision)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Goal: Vision Encoder converts images → meaningful embeddings

What it learns:
- Objects and their features
- Spatial relationships
- Visual patterns
- Semantic understanding of images

Think: Teaching visual recognition

STAGE D: Learning to Speak (Speech Generation)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Goal: System can generate speech (text-to-speech)

What it learns:
Part 1 (RVQ Codec):
- How to discretize mel spectrograms
- Codebook patterns for speech

Part 2 (Talker):
- How to predict speech codes autoregressively
- Prosody and rhythm

Think: Teaching speaking/pronunciation

STAGE E: Bringing It All Together (Multimodal)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Goal: All components work together for cross-modal understanding

What it learns:
- How image relates to text description
- How audio relates to transcription
- Cross-modal reasoning
- Answering questions about images/audio

Think: Teaching to understand and respond to any input

This progressive approach ensures stable, effective learning!
```

---

## 📊 Training Summary Table

### Complete Specifications

| Stage | Component | Primary Task | Data Type | Loss Function | Metric | Est. Time | Dependencies |
|-------|-----------|--------------|-----------|---------------|---------|-----------|--------------|
| **A** | Thinker | Language Modeling | Text | Cross-Entropy | Perplexity | 8-12h | None |
| **B** | Audio Encoder | ASR | Audio + Text | CTC | WER | 6-10h | None |
| **C** | Vision Encoder | Classification | Images + Labels | Cross-Entropy | Accuracy | 4-8h | None |
| **D** | RVQ + Talker | Speech Gen | Audio (TTS) | MSE + CE | Recon Error | 10-15h | None (RVQ), Then Talker needs RVQ |
| **E** | All (Joint) | Multimodal QA | Mixed Modalities | Cross-Entropy | Task Acc | 6-12h | A, B, C, D |

**Total Estimated Time: 40-60 hours** on single 12GB GPU (tiny model, 25.65M params)

**Note:** Training time scales with model size. See "Model Scaling" section below for larger models.

---

## 🎯 Training Strategy & Design Philosophy

### 1. Modularity

**Principle: Each stage is independent**

```
Why modularity matters:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Debugging:
- Stage C fails? Only rerun Stage C
- Save 30+ hours of retraining!
- Pinpoint issues quickly

Development:
- Multiple people work on different stages
- Parallel development possible
- Faster iteration

Experimentation:
- Want better Vision Encoder?
- Just retrain Stage C and E
- No need to retrain A, B, D

Flexibility:
- Can swap components easily
- Modular design = future-proof
```

### 2. Efficiency

**Principle: Maximize learning with minimal resources**

```
Resource Optimizations:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Small Datasets:
- < 5GB per modality
- Synthetic data where needed
- Enough for proof-of-concept

Single GPU:
- 12GB VRAM sufficient
- Gradient accumulation for larger batches
- Mixed precision (FP16) saves memory

Memory Tricks:
- Gradient checkpointing
- Small batch sizes (2-4)
- Frozen components when appropriate

Time Management:
- Each stage < 15 hours
- Total project: 2-3 days on single GPU
- Feasible for research/prototyping
```

### 3. Progressive Learning

**Principle: Simple to complex**

```
Learning Progression:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Stage A: Text-only (simplest)
  ↓
Stages B, C, D: Individual modalities
  ↓
Stage E: Multimodal (most complex)

Why this works:
✅ Strong foundation first (text)
✅ Specialized skills next (vision/audio/speech)
✅ Integration last (multimodal)

Like learning to walk before you run!
```

---

## ⚙️ Before Training: Update Configs Based on Dataset Size

**Important:** After downloading your datasets, update training parameters based on actual data size:

```bash
# Analyze datasets and automatically update all config files
python scripts/update_configs_from_data.py

# Dry run (see what would change without modifying files)
python scripts/update_configs_from_data.py --dry-run
```

**What it does:**
- ✅ Counts samples in production/synthetic datasets
- ✅ Calculates appropriate `max_steps`, `max_epochs`, `warmup_steps`
- ✅ Adjusts `val_freq` and `checkpoint_freq` based on dataset size
- ✅ Updates data paths to production files if they exist
- ✅ Uses best practices: 1-3 epochs for large datasets, 5-10 for small

**When to run:**
- After downloading production datasets
- After generating synthetic test data
- When switching between different dataset sizes

See [Chapter 34: Configuration Files](34-configuration-files.md) for details.

---

## 📈 Model Scaling

### Current Configuration (Tiny)

**Total: 25.65M parameters** - Fits on 12GB GPU

| Component | Config | Parameters |
|-----------|--------|------------|
| **Thinker** | d_model=256, n_layers=4, n_heads=4, d_ff=1024 | 20.32M |
| **Audio Encoder** | d_model=192, n_layers=4, n_heads=3, d_ff=768 | 2.05M |
| **Vision Encoder** | d_model=128, n_layers=4, n_heads=2, d_ff=512 | 914K |
| **Talker** | d_model=192, n_layers=4, n_heads=3, d_ff=768 | 2.24M |

### Scaling to Larger Models

**Moderate Scale (100-200M params):**
- **GPU:** 24GB VRAM
- **Changes:** 2x dimensions, 2x layers
- **Example Thinker:** d_model=512, n_layers=8, n_heads=8, d_ff=2048
- **Training Time:** ~80-120 hours
- **Use Case:** Better quality while staying accessible

**Large Scale (500M-1B params):**
- **GPU:** 40GB+ VRAM (A100) or Multi-GPU
- **Changes:** 3-4x dimensions, 4x layers
- **Example Thinker:** d_model=768, n_layers=16, n_heads=12, d_ff=3072
- **Training Time:** ~200-400 hours
- **Use Case:** Production-quality performance

**Very Large Scale (1B-7B params):**
- **GPU:** Multi-GPU (4-8x A100) or TPU
- **Changes:** 4x dimensions, 8x layers, enable MoE
- **Example Thinker:** d_model=1024, n_layers=32, n_heads=16, d_ff=4096, use_moe=true
- **Training Time:** ~1000+ hours
- **Use Case:** Research, SOTA performance

### Key Parameters to Scale

| Parameter | Impact | Scaling Rule |
|-----------|--------|--------------|
| **d_model** | Quadratic on attention, linear on FFN | 2x d_model ≈ 4x params |
| **n_layers** | Linear increase | 2x layers = 2x params |
| **d_ff** | Linear on FFN | Usually 4x d_model |
| **n_heads** | Minimal | Usually d_model / 64 |
| **vocab_size** | Only embedding layer | Linear increase |

### Memory Requirements

**Training Memory Formula:**
```
Memory ≈ 4 × (model_params × 4 bytes) + (batch_size × ctx_len × d_model × 4 bytes)
```

**Examples:**
- **Tiny (25.65M):** ~12GB VRAM ✓
- **Moderate (150M):** ~24GB VRAM ✓
- **Large (700M):** ~40GB+ VRAM (A100)
- **Very Large (3B):** Multi-GPU required

### Scaling Process

1. **Create new config files:**
   ```bash
   cp configs/thinker_tiny.json configs/thinker_medium.json
   # Edit parameters in new config
   ```

2. **Adjust training parameters:**
   - Reduce `batch_size` if OOM
   - Increase `gradient_accumulation_steps`
   - Increase `max_steps` and `warmup_steps`
   - Always use `use_amp: true`, `use_flash: true`

3. **Update projector dimensions** (for Stage E):
   - When scaling Thinker's d_model, update projectors in `sft_omni.py`
   - Vision: `Linear(128 → new_d_model)`
   - Audio: `Linear(192 → new_d_model)`

4. **Train with new configs:**
   ```bash
   python train_text.py --config configs/thinker_medium.json
   # ... repeat for all stages
   ```

### Important Considerations

- **Memory Management:** Use gradient checkpointing, reduce batch size, use gradient accumulation
- **Training Time:** Larger models need 10-100x more training time
- **Data Requirements:** Larger models may need millions more samples per modality
- **Learning Rate:** Consider scaling: `lr = base_lr * sqrt(d_model / 256)`

### Recommended Scaling Path

1. **Start Small:** Get tiny model (25.65M) working perfectly
2. **Scale to Medium (100-200M):** Test quality improvements with 24GB GPU
3. **Evaluate:** Is quality good enough?
4. **If Not:** Scale to Large (500M-1B) with multi-GPU
5. **Production:** Consider 1B-7B for real applications

**Quick Reference:**

| Scale | Total Params | VRAM | Training Time |
|-------|--------------|------|---------------|
| **Tiny** | 25.65M | 12GB | 40-60 hours |
| **Medium** | ~150M | 24GB | 80-120 hours |
| **Large** | ~700M | 40GB+ | 200-400 hours |
| **XL** | ~3B | Multi-GPU | 1000+ hours |

---

## 📊 Scale vs Performance Analysis

### Performance Scaling with Model Size

**Model Size vs Performance (Quality):**
```
Performance Score (Normalized)
100% │                                    ╱───── Plateau
     │                                 ╱─
 90% │                              ╱─
     │                           ╱─
 80% │                        ╱─
     │                     ╱─
 70% │                  ╱─
     │               ╱─
 60% │            ╱─
     │         ╱─
 50% │      ╱─
     │   ╱─
 40% │╱─
     └───────────────────────────────────────────────
       25M   100M   500M   1B    3B    7B
              Model Size (Parameters)

Key Findings:
- 25M (Tiny): ~40-50% of max performance
- 100M (Medium): ~70-80% of max performance  
- 500M (Large): ~85-90% of max performance
- 1B+: ~90-95% of max performance (diminishing returns)

Research: Models under 15B params can achieve 90% of 
larger model performance on many tasks.
```

**Model Size vs Training Time:**
```
Training Time (Hours)
1000+ │                                    ╱─────
      │                                 ╱─
  500 │                              ╱─
      │                           ╱─
  200 │                        ╱─
      │                     ╱─
  100 │                  ╱─
      │               ╱─
   50 │            ╱─
      │         ╱─
   20 │      ╱─
      │   ╱─
   10 │╱─
      └───────────────────────────────────────────────
        25M   100M   500M   1B    3B    7B
              Model Size (Parameters)

Note: Training time scales roughly linearly with parameters,
but larger models need more data and steps.
```

**Model Size vs Inference Speed:**
```
Tokens per Second (TPS)
 100 │╱─────
     │   ╲─
   50 │      ╲─
     │         ╲─
   20 │            ╲─
     │               ╲─
   10 │                  ╲─
     │                     ╲─
    5 │                        ╲─
     │                           ╲─
    2 │                              ╲─
     │                                 ╲─────
     └───────────────────────────────────────────────
       25M   100M   500M   1B    3B    7B
              Model Size (Parameters)

Inference Speed (12GB GPU, batch_size=1):
- 25M: ~50-100 TPS
- 100M: ~20-40 TPS
- 500M: ~5-10 TPS
- 1B+: <5 TPS (needs larger GPU or quantization)
```

### Expected Performance Benchmarks

**Text Understanding (Perplexity):**
| Model Size | Perplexity | Quality |
|------------|------------|---------|
| 25M (Tiny) | ~30-40 | Basic |
| 100M (Medium) | ~20-25 | Good |
| 500M (Large) | ~15-20 | Very Good |
| 1B+ | ~10-15 | Excellent |

*Lower perplexity = Better*

**Multimodal Understanding (Task Accuracy):**
| Model Size | Image QA | Audio ASR | VQA Score |
|------------|----------|-----------|-----------|
| 25M (Tiny) | ~60% | ~70% | ~55% |
| 100M (Medium) | ~75% | ~85% | ~70% |
| 500M (Large) | ~85% | ~92% | ~80% |
| 1B+ | ~90%+ | ~95%+ | ~85%+ |

*Note: Actual performance depends on training data quality and duration*

### Diminishing Returns Analysis

```
Performance Gain per 2x Parameters
 100% │╱─────
      │   ╲─
  50% │      ╲─
      │         ╲─
  25% │            ╲─
      │               ╲─
  10% │                  ╲─
      │                     ╲─
   5% │                        ╲─────
      └───────────────────────────────────────────────
        25M   100M   500M   1B    3B    7B
              Model Size (Parameters)

Key Insight: Each 2x increase in parameters gives:
- 25M→50M: ~30% performance gain
- 100M→200M: ~15% performance gain
- 500M→1B: ~8% performance gain
- 1B→2B: ~4% performance gain

Diminishing returns become significant after ~500M parameters.
```

**Takeaway:** Performance scales sublinearly - doubling parameters doesn't double performance. The sweet spot for quality/efficiency balance is around 100-500M parameters.

---

## 💻 Quick Start Commands

### Running the Training Pipeline

```bash
# Stage A: Thinker Pretraining
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
python train_text.py --config configs/thinker_tiny.json

# Trains Thinker on text corpus
# Output: checkpoints/thinker_tiny/
# Expected: Perplexity < 30

# Stage B: Audio Encoder (ASR)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
python train_audio_enc.py --config configs/audio_enc_tiny.json

# Trains Audio Encoder for speech recognition
# Output: checkpoints/audio_encoder_tiny/
# Expected: WER < 30%

# Stage C: Vision Encoder
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
python train_vision.py --config configs/vision_tiny.json

# Trains Vision Encoder for image understanding
# Output: checkpoints/vision_encoder_tiny/
# Expected: Accuracy > 70%

# Stage D: Talker + RVQ Codec
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
python train_talker.py --config configs/talker_tiny.json

# Trains RVQ codec and Talker for speech generation
# Output: checkpoints/rvq_codec/ + checkpoints/talker_tiny/
# Expected: Intelligible speech output

# Stage E: Multimodal SFT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
python sft_omni.py --config configs/omni_sft_tiny.json \
  --thinker checkpoints/thinker_tiny/final.pt \
  --audio_encoder checkpoints/audio_encoder_tiny/final.pt \
  --vision_encoder checkpoints/vision_encoder_tiny/final.pt \
  --talker checkpoints/talker_tiny/final.pt

# Trains all components jointly for multimodal understanding
# Output: checkpoints/omni_final/
# Expected: Successful multimodal Q&A

Complete! μOmni is ready for inference! 🎉
```

---

## 🔄 Training Dependencies

### Stage Dependency Graph

```
Independent (Can run in parallel):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

┌─────────┐     ┌──────────────┐     ┌────────────────┐
│ Stage A │     │  Stage B     │     │  Stage C       │
│(Thinker)│     │(Audio Encoder)│     │(Vision Encoder)│
└────┬────┘     └──────┬───────┘     └───────┬────────┘
     │                 │                      │
     │                 │                      │
     └─────────────────┴──────────────────────┘
                       ↓
             All feed into Stage E

┌──────────────────┐
│ Stage D (Part 1) │ ← Independent
│ (RVQ Codec)      │
└────────┬─────────┘
         ↓
┌──────────────────┐
│ Stage D (Part 2) │ ← Depends on Part 1
│ (Talker)         │
└────────┬─────────┘
         │
         └─────────→ Feeds into Stage E

Sequential (Must run in order):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Stage E depends on ALL previous stages:
- Needs trained Thinker (from A)
- Needs trained Audio Encoder (from B)
- Needs trained Vision Encoder (from C)
- Needs trained Talker (from D)

Optimization:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Parallel strategy (if you have multiple GPUs):
- GPU 1: Stage A (8-12h)
- GPU 2: Stage B (6-10h)
- GPU 3: Stage C (4-8h)
- GPU 4: Stage D (10-15h)

Then GPU 1: Stage E (6-12h)

Total wall-clock time: ~25 hours instead of 50!
```

---

## 💡 Key Takeaways

✅ **5-stage pipeline** ensures stable, modular training  
✅ **Independent stages** (A, B, C, D-part1) can run in parallel  
✅ **Stage E** integrates all components for multimodal understanding  
✅ **~40-60 hours total** on single 12GB GPU  
✅ **Small datasets** (<5GB each) make it accessible  
✅ **Modular design** enables easy debugging and experimentation  
✅ **Progressive learning** from simple (text) to complex (multimodal)  
✅ **Efficient** through gradient accumulation, FP16, and checkpointing

---

## 🎓 Self-Check Questions

1. Why does μOmni use 5 separate training stages instead of training everything together?
2. Which stages can be run in parallel and why?
3. What is the purpose of Stage A and why must it come first conceptually?
4. How does modular training help with debugging?
5. What is the total estimated training time on a single 12GB GPU?

<details>
<summary>📝 Click to see answers</summary>

1. Separate stages avoid gradient conflicts, enable focused learning, simplify debugging, and allow modular development. Each component learns its specialized task before multimodal integration
2. Stages A, B, C, and D-part1 can run in parallel because they train independent components with no dependencies on each other. Only Stage E requires all previous stages to be complete
3. Stage A trains the Thinker (core LLM) on text. It must conceptually come first because language understanding is the foundation - the Thinker needs to understand text before it can process multimodal inputs
4. Modular training means if a stage fails, you can identify and fix only that component, then retrain just that stage and subsequent dependent stages. No need to retrain the entire pipeline
5. Approximately 40-60 hours total: Stage A (8-12h) + Stage B (6-10h) + Stage C (4-8h) + Stage D (10-15h) + Stage E (6-12h)
</details>

---

[Continue to Chapter 27: Stage A - Thinker Pretraining →](27-stage-a-thinker.md)

**Chapter Progress:** Training Pipeline ●○○○○○ (1/6 complete)

---

## 📊 Training Summary

| Stage | Model | Task | Loss Function | Key Metric |
|-------|-------|------|---------------|------------|
| **A** | Thinker | Language Modeling | Cross-Entropy | Perplexity |
| **B** | Audio Encoder | ASR | CTC | WER |
| **C** | Vision Encoder | Image Understanding | Cross-Entropy | Accuracy |
| **D** | Talker + RVQ | Speech Generation | Cross-Entropy + MSE | Reconstruction |
| **E** | All (Joint) | Multimodal | Cross-Entropy | Mixed Accuracy |

## 🎯 Training Strategy

### Modularity
- Each stage trains independently
- Debug issues in isolation
- Parallel development possible

### Efficiency
- Small datasets (<5GB per modality)
- Fits 12GB GPU with gradient accumulation
- Uses mixed precision (FP16)
- Gradient checkpointing for memory

### Progressive Learning
- Start with individual modalities
- End with joint understanding
- Specialized encoders preserved

## 💻 Quick Start

```bash
# Stage A
python train_text.py --config configs/thinker_tiny.json

# Stage B
python train_audio_enc.py --config configs/audio_enc_tiny.json

# Stage C
python train_vision.py --config configs/vision_tiny.json

# Stage D
python train_talker.py --config configs/talker_tiny.json

# Stage E
python sft_omni.py --config configs/omni_sft_tiny.json
```

## 💡 Key Takeaways

✅ **5 independent stages** (modular design)  
✅ **~40-60 hours total** training time (12GB GPU)  
✅ **Small datasets** (<5GB each)  
✅ **Progressive learning** (specialized → joint)

---

[Back to Index](00-INDEX.md)

