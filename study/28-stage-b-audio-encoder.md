# Chapter 28: Stage B - Audio Encoder ASR Training

[← Previous: Stage A Thinker](27-stage-a-thinker.md) | [Back to Index](00-INDEX.md) | [Next: Stage C Vision →](29-stage-c-vision-encoder.md)

---

## 🎯 Learning Objectives

By the end of this chapter, you will understand:

- What Stage B trains and its purpose
- How speech recognition (ASR) training works
- What CTC loss is and why it's used
- How to interpret WER (Word Error Rate)
- Configuration and data requirements
- Expected progress and outputs

---

## 💡 What is Stage B?

### Teaching the Audio Encoder to Understand Speech

**Analogy: Learning to Listen**

```
Think of Stage B like learning to understand spoken language:

WITHOUT TRAINING:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Hear sound waves: [wavy patterns]
No understanding: "just noise"
Can't transcribe: ❌

WITH ASR TRAINING (Stage B):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Hear: "hello world" (audio)
Understand: Phonemes → Words
Transcribe: "hello world" ✓

STAGE B TRAINS THE AUDIO ENCODER:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Purpose: Learn speech → meaningful embeddings
Task: Automatic Speech Recognition (ASR)
Input: Audio waveform → Mel spectrogram
Output: Text transcription

This teaches the encoder:
✅ Phonetic patterns (sounds → letters)
✅ Temporal structure (speech timing)
✅ Semantic features (meaning in audio)

Later in Stage E:
These embeddings help Thinker understand multimodal inputs!
```

---

## 📝 The Task: Automatic Speech Recognition (ASR)

### Converting Speech to Text

**How ASR Works:**

```
COMPLETE ASR PIPELINE:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Step 1: Audio Preprocessing
Raw audio (16kHz, 1 second) → 16000 samples
↓
Mel spectrogram: (100, 128)
- 100 frames (10ms per frame)
- 128 mel frequency bins

Step 2: Audio Encoder
Mel (100, 128) → Embeddings (12, 192)
- Convolutional downsampling: 8x
- 100 frames → 12 frames (more efficient!)
- Each frame: 192-dim embedding

Step 3: CTC Decoding
Embeddings (12, 192) → Character probabilities
↓
CTC head: Linear(192 → vocab_size)
↓
Per-frame predictions: [[h], [e], [l], [l], [o], ...]
↓
CTC collapse repeated chars: "hello"

Step 4: Compare to Ground Truth
Predicted: "hello world"
Target: "hello world"
Match! ✓

Training teaches encoder:
✅ Which audio patterns → which letters
✅ Temporal alignment (when sounds occur)
✅ Robust to speed variations
```

**Training Example:**

```
Example: Training on "hello"
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Input Audio:
"hello" spoken → [audio waveform]
↓
Mel spectrogram: (38, 128)  # ~0.3 seconds
↓
Audio Encoder: (38, 128) → (5, 192)  # 8x downsample
↓
CTC Head: (5, 192) → (5, 29)  # 26 letters + blank + space + EOS

Per-frame predictions (simplified):
Frame 0: [h: 0.8, e: 0.05, blank: 0.10]  → "h"
Frame 1: [e: 0.7, h: 0.1, blank: 0.15]   → "e"
Frame 2: [l: 0.85, e: 0.05, blank: 0.05] → "l"
Frame 3: [l: 0.80, o: 0.1, blank: 0.05]  → "l"
Frame 4: [o: 0.75, l: 0.1, blank: 0.10]  → "o"

CTC collapse: "hello" ✓

Loss: CTC computes alignment automatically!
(No need to manually align audio frames to letters)
```

---

## 🔍 CTC Loss Explained

### Connectionist Temporal Classification

**Why CTC?**

```
Problem: Alignment is Hard!
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Audio: "hello" = 0.5 seconds = 50 mel frames
Text: "hello" = 5 characters

Which frame corresponds to which letter?
❌ Different speakers = different timings
❌ Some sounds longer than others ("helllllo")
❌ Manual alignment = impossible at scale!

Solution: CTC Loss!
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

CTC allows:
✅ Variable-length sequences (50 frames → 5 chars)
✅ Automatic alignment (no manual labels needed)
✅ Handles repetitions and silence

How CTC works:
1. Predicts character at EVERY frame
2. Allows "blank" token (silence)
3. Collapses repeated characters
4. Computes all possible alignments
5. Optimizes best alignment automatically
6. **Padding handling:** CTC loss uses `input_lengths` to exclude padding frames from loss calculation, ensuring only valid audio frames contribute to training

Example alignments that CTC considers:
"hheeellllllooo" → "hello" ✓
"h_ee_lll_oo___" → "hello" ✓  (_ = blank)
"_h_e_l_l_o____" → "hello" ✓

All valid! CTC learns which is best!
```

**CTC Rules:**

```
CTC DECODING RULES:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Rule 1: Remove repeated characters
"hheelloo" → "helo"

Rule 2: Remove blanks (_)
"h_e_l_l_o" → "hello"

Rule 3: Repeated chars separated by blank are kept
"hel_lo" → "hello" ✓
"hell_lo" → "helllo" (double l!)

This handles fast/slow speech naturally!
```

---

## 📊 Metrics: CTC Loss and WER

### Understanding Performance

**CTC Loss:**

```
WHAT IS CTC LOSS?
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Measures: Alignment quality and prediction confidence

Typical values:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Random init: 50-100 (terrible)
After epoch 1: 30-45 (learning)
After epoch 10: 10-15 (decent)
Well-trained: <10 (good!)

Lower = better alignment found!
```

**WER (Word Error Rate):**

```
WHAT IS WER?
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

WER = (Substitutions + Insertions + Deletions) / Total Words

Example:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Reference: "the cat sat on the mat"  (6 words)
Hypothesis: "the cat sit on a mat"   (6 words)

Errors:
- "sat" → "sit": 1 substitution
- "the" → "a": 1 substitution

WER = 2 errors / 6 words = 33.3%

Quality thresholds:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

WER < 5%: Excellent (human-level)
WER 5-15%: Good (usable)
WER 15-30%: Okay (needs improvement)
WER > 50%: Poor (barely understanding)

TARGET FOR μOmni:
WER 10-20% = Good for proof-of-concept! ✓

**Expected Validation Loss:**
- Target CTC Loss: < 2.0
- Good: 1.5-2.0
- Excellent: < 1.5
- Initial: 3.0-4.0 (random guessing)
- Mid-training: 2.0-2.5
```

---

## 📊 Configuration Explained

```json
{
  // MODEL ARCHITECTURE
  "d_model": 192, // Encoder dimension (smaller than Thinker's 256)
  "n_layers": 4, // Transformer encoder layers
  "n_heads": 3, // Attention heads (3 for 192-dim)
  "d_ff": 768, // Feedforward size (4x d_model)
  "dropout": 0.1, // Regularization
  "downsample_time": 8, // Temporal compression (8x)
  // 100 frames → 12 frames
  // Reduces computation!

  // DATA
  "data_path": "data/audio/asr.csv", // Audio files + transcriptions
  "batch_size": 8, // Smaller than text (audio = memory-intensive)
  "num_epochs": 20, // More epochs than Stage A
  // ASR is harder to learn!

  // OPTIMIZATION
  "learning_rate": 1e-4, // 0.0001 (lower than Stage A)
  // Audio training needs stability

  "checkpoint_freq": 5000 // Checkpoint frequency (every 1000 steps)
}
```

---

## 📈 Expected Training Progress

```
TRAINING TIMELINE (20 epochs, ~8 hours):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Hour 0 (Random Init):
Step 1: ctc_loss=89.3 wer=100%
→ Complete gibberish
→ Output: "xjkdf wkejf"

Hour 1 (Early Learning):
Step 100: ctc_loss=45.23 wer=78.5%
→ Some phonemes recognized
→ Output: "hllo wrld" (missing vowels)

Hour 2 (Phoneme Learning):
Step 500: ctc_loss=18.67 wer=45.2%
→ Most sounds recognized
→ Output: "hello wrold" (close!)

Hour 5 (Good Recognition):
Epoch 10/20:
ctc_loss=12.34 wer=25.8%
→ Mostly correct transcriptions
→ Output: "hello world" (occasional errors)

Hour 8 (Well-Trained):
Epoch 20/20:
Final: ctc_loss=8.45 wer=12.3%
→ Reliable speech recognition!
→ Output: "hello world" ✓

READY FOR STAGE E! ✓
```

---

## 📁 Data Format

```csv
# data/audio/asr.csv
audio_path,transcription
data/audio/wav/sample1.wav,"hello world"
data/audio/wav/sample2.wav,"how are you"
data/audio/wav/sample3.wav,"the cat sat on the mat"

# Requirements:
- Audio: .wav files, 16kHz sample rate
- Transcriptions: Lowercase, no punctuation
- Duration: 1-10 seconds per sample
- Total: ~1000 samples for proof-of-concept
```

---

## 🎓 Output Files

````
checkpoints/audio_enc_tiny/
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

├── model.pt                 # Latest model weights (overwritten)
├── model_metadata.json      # Training state (step, epoch, config)
│
└── training_log.json        # Metrics history

Load for Stage E:
```python
# Load the latest checkpoint
checkpoint = torch.load('checkpoints/audio_enc_tiny/model.pt')
audio_encoder.load_state_dict(checkpoint['enc'])
````

````

---

## ⚙️ Configuration

**Key Parameters for CUDA Graphs Compatibility:**

```json
{
  "use_compile": true,
  "max_mel_length_percentile": 95.0  // Optional: Percentile for auto-calculation (default: 95.0)
  // max_mel_length is auto-calculated from dataset - no need to set manually
}
````

**Auto-Calculation:**

- `max_mel_length` is **automatically calculated** from your dataset during training
- Uses **95th percentile** by default to minimize padding while covering 95% of data
- Automatically rounds up to nearest 256 for better memory alignment
- ~5% of samples will be skipped if longer (outliers filtered during dataset iteration)

**Frame Rate Reference:**

- Frame rate = sample_rate / hop_length = 16000 / 160 = 100 frames/second
- For 60 seconds: 60 × 100 = 6000 frames
- For 20 seconds: 20 × 100 = 2000 frames

**Why Fixed Length?**

- CUDA graphs require uniform batch shapes
- Prevents "tensor size mismatch" errors
- Enables 10-20% speedup with compilation

**Check Your Dataset:**

```bash
# Analyze actual mel lengths
# (script removed) Use `omni.utils` dataset analysis helpers instead,
# e.g. `omni.utils.calculate_max_mel_length_from_asr_csv()` or
# `omni.utils.analyze_asr_dataset()` to compute recommended max lengths.
```

---

## 💻 Running Stage B

```bash
# Stage B: Audio Encoder ASR Training
python train_audio_enc.py --config configs/audio_enc_tiny.json

# Expected output:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Loading audio data from data/audio/asr.csv...
Found 1000 audio samples

Initializing Audio Encoder (10.5M parameters)...
Using device: cuda:0

Starting ASR training with CTC loss...

Epoch 1/20:
[Step 100/2500] ctc_loss=45.23 wer=78.5% | 3.5s/step
[Step 1000/2500] ctc_loss=18.67 wer=45.2% | 3.2s/step
✓ Saved checkpoint: model.pt + metadata

...

Epoch 20/20:
[Step 2500/2500] ctc_loss=8.45 wer=12.3% | 3.0s/step
✓ Saved checkpoint: model.pt + metadata

Training complete! Time: 8h 15m
Final WER: 12.3%

Ready for Stage E! 🎉
```

---

## 💡 Key Takeaways

✅ **Stage B** trains Audio Encoder on speech recognition  
✅ **CTC loss** enables automatic alignment (no manual labels)  
✅ **WER < 20%** indicates good speech understanding  
✅ **8x downsampling** reduces computation efficiently  
✅ **~8 hours** training time on 12GB GPU  
✅ **Learns phonetic and temporal patterns** in speech  
✅ **Embeddings** will be used in multimodal fusion (Stage E)

---

## 🎓 Self-Check Questions

1. What is the purpose of Stage B and what does it train?
2. Why do we use CTC loss instead of standard cross-entropy?
3. What does WER of 15% mean?
4. Why is 8x temporal downsampling used?
5. How will the Audio Encoder be used in Stage E?

<details>
<summary>📝 Click to see answers</summary>

1. Stage B trains the Audio Encoder to understand speech through ASR (speech recognition). This teaches it to convert audio into meaningful embeddings that capture phonetic and semantic information
2. CTC loss handles variable-length alignment automatically. Audio has many frames (e.g., 100) but text has few characters (e.g., 5). CTC finds the best alignment without manual frame-to-character labels
3. WER (Word Error Rate) of 15% means 15 out of 100 words are incorrectly transcribed (substituted, inserted, or deleted). This is considered "good" for small models
4. 8x downsampling reduces 100 audio frames to 12 frames, making computation much more efficient while preserving temporal information. Reduces memory and speeds up training/inference
5. In Stage E, the trained Audio Encoder will convert audio inputs into 256-dim embeddings that are concatenated with text and image embeddings, enabling the Thinker to process multimodal inputs
</details>

---

[Continue to Chapter 29: Stage C - Vision Encoder →](29-stage-c-vision-encoder.md)

**Chapter Progress:** Training Pipeline ●●●○○○ (3/6 complete)

---
