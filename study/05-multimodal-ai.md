# Chapter 05: What is Multimodal AI?

[← Previous: Transformers Intro](04-transformers-intro.md) | [Back to Index](00-INDEX.md) | [Next: Understanding Embeddings →](06-embeddings-explained.md)

---

## 🎯 Learning Objectives

By the end of this chapter, you will understand:
- What multimodal AI means and why it matters
- How different modalities (text, image, audio, video) are processed
- Challenges in multimodal learning
- Fusion strategies for combining modalities
- How μOmni implements multimodal understanding

---

## 🌈 What is Multimodal AI?

### Definition

**Multimodal AI** systems can understand and generate multiple types of data (modalities) simultaneously.

But what does "modality" really mean? Let's start simple:

**Modality = A way of experiencing or expressing information**

Just like YOU use multiple senses to understand the world!

```
When you meet a friend:
👁️ VISION:  You SEE their face
👂 HEARING: You HEAR their voice
🤝 TOUCH:   You FEEL their handshake

Your brain combines all these → "Ah, it's my friend!"
```

**In AI, modalities are types of data:**
```
Modalities:
📝 Text     - words, sentences, documents ("Hello world")
🖼️ Images   - photos, illustrations, diagrams (pixels: 224×224×3)  
🎤 Audio    - speech, music, sounds (waveforms: 16000 samples/sec)
🎬 Video    - moving images with audio (30 frames/sec + audio)
🎮 Other    - sensor data, 3D models, temperature, etc.
```

### Why Multimodal? (The Power of Multiple Senses)

Think about how YOU understand the world:

**Scenario 1: Reading about a sunset**
```
Text: "The sunset was beautiful with orange and pink colors"
↓
Your imagination: You TRY to picture it in your mind
But you've never seen THIS specific sunset!
```

**Scenario 2: Seeing a photo of the sunset**
```
Image: 🌅 [Beautiful sunset photo]
↓
You see the colors, but no context
Is it morning or evening? Where is this?
```

**Scenario 3: Photo + Description (MULTIMODAL!)**
```
Image: 🌅 [Sunset photo]
Text: "Sunset over the Pacific Ocean in California"
↓
COMPLETE understanding!
- What: Sunset (from image)
- Where: Pacific Ocean, California (from text)
- When: Evening (inferred from both)
```

**This is why multimodal is powerful!**

Humans naturally use multiple senses:

```
Real-world scenario: Watching a cooking video

Visual:  👁️ See ingredients, cutting technique, color changes
Audio:   👂 Hear instructions, sizzling sounds, timer beep
Text:    📝 Read recipe on screen, measurements

Your brain integrates all three seamlessly!
Result: You can cook the dish perfectly!

If you only had ONE modality:
- Only text? You might not know the right consistency
- Only video? You might miss exact measurements
- Only audio? You can't see what "golden brown" looks like

ALL THREE together → Perfect understanding!

Multimodal AI aims to do the same.
```

**Why This Matters:**
```
Single-modal AI (text-only):
Question: "Is this safe to eat?"
Answer: "I don't know what 'this' is" ❌

Multimodal AI:
Input: 🖼️ [Photo of moldy bread] + "Is this safe to eat?"
Answer: "No, this bread has mold and should not be eaten" ✅
```

---

## 🆚 Unimodal vs Multimodal

### Unimodal Systems

```
Text-only model (GPT-3):
Input: "Describe a sunset"
Output: "A sunset features warm colors..."
❌ Has never "seen" a sunset!

Image-only model (ResNet):
Input: 🖼️ [Photo of sunset]
Output: "Sky, clouds, orange"
❌ Can't explain why it's beautiful

Audio-only model (Whisper):
Input: 🎤 [Recording: "Look at that sunset!"]
Output: "look at that sunset"
❌ Doesn't know what "that" refers to
```

### Multimodal Systems

```
Multimodal model (μOmni, GPT-4V):
Input: 🖼️ [Photo of sunset] + "What makes this beautiful?"
Output: "The sunset is beautiful due to the vibrant 
         orange and pink hues created by light 
         scattering through the atmosphere..."

✅ Understands visual content
✅ Connects to linguistic concepts
✅ Provides contextual reasoning
```

---

## 🧩 The Four Main Modalities

### 1. **Text** 📝

**Representation:**
```
Raw: "Hello world"
Tokenized: [15, 24, 89, 42]
Embedded: [[0.23, -0.15, ...], [0.12, 0.34, ...], ...]
```

**Challenges:**
- Ambiguity (bank = financial institution or river side?)
- Context dependence
- Different languages

---

### 2. **Images** 🖼️

**Understanding Images for AI:**

Think about what an image IS to a computer:

```
What YOU see:
🐱 "A cute cat!"

What the COMPUTER sees:
A grid of numbers!

Example 3×3 pixel image:
[255, 200, 180]  [250, 195, 175]  [245, 190, 170]  ← Row 1
[200, 150, 120]  [195, 145, 115]  [190, 140, 110]  ← Row 2
[180, 130, 100]  [175, 125, 95]   [170, 120, 90]   ← Row 3

Each pixel: (Red, Green, Blue) values from 0-255
- [255, 0, 0] = Bright red
- [0, 255, 0] = Bright green
- [255, 255, 255] = White
- [0, 0, 0] = Black
```

**Representation:**
```
Raw image: 224×224×3 RGB image
           224 pixels wide × 224 pixels tall × 3 color channels
           = 150,528 numbers!

Think: A 224×224 photo is like a book with 150,528 numbers!
How to make sense of all this data?

Preprocessed (make it easier for AI):
- Normalize: [0, 255] → [0, 1] (scale down for stability)
  Example: 255 → 1.0, 128 → 0.5, 0 → 0.0
  Why? Smaller numbers are easier for neural networks to process!

- Resize to standard size (all images same size)
  Why? Just like standardized test forms - easier to process!

- Convert to tensor: (3, 224, 224)
  (3 color channels, 224 height, 224 width)

Embedded (convert to tokens like text!):
- Patch-based (ViT): Divide into 16×16 patches
  224÷16 = 14 patches wide × 14 patches tall = 196 patches total
  Each patch becomes ONE token! (just like one word in text)
  
  Visual:
  ┌─────┬─────┬─────┬─────┐
  │  1  │  2  │  3  │  4  │  ← Each square is one 16×16 patch
  ├─────┼─────┼─────┼─────┤    = one token
  │  5  │  6  │  7  │  8  │    = one embedding
  ├─────┼─────┼─────┼─────┤
  │  9  │ 10  │ 11  │ 12  │
  └─────┴─────┴─────┴─────┘
  
  Now the image is like a "sentence" with 196 "words" (patches)!

- Convolutional: Extract features at multiple scales
  (Alternative approach: look for edges, shapes, objects)
```

**Challenges:**
- High dimensionality (millions of pixels) - TOO MUCH data!
  Solution: Reduce to patches (196 tokens instead of 150K pixels)
  
- Spatial relationships - nearby pixels are related
  Example: All pixels of the cat's eye should be understood together
  Solution: Process patches with attention (capture relationships)
  
- Scale and rotation variance
  A cat facing left vs right looks different to the computer!
  A close-up vs far-away cat has different pixel patterns!
  Solution: Data augmentation (train on rotated/scaled images)
  
- Lighting conditions
  Same cat in bright sun vs dark room = very different pixels!
  Solution: Normalization and robust training data

---

### 3. **Audio** 🎤

**Understanding Audio for AI:**

Audio is even more abstract than images! Let's break it down:

```
What YOU hear:
🎤 "Hello!" (a voice saying hello)

What the COMPUTER sees:
A sequence of air pressure measurements!

Raw waveform (simplified):
Time:  0.00s   0.01s   0.02s   0.03s   0.04s
Value: 0.5  → -0.3  →  0.8  → -0.2  →  0.1  → ...

Think of it like: A heart rate monitor showing ups and downs!
```

**Representation:**
```
Raw waveform (time-series):
- Sampled at 16000 Hz (16000 measurements per second)
  Why 16000? Human speech is ~8000 Hz, so 16000 captures it well
  (Nyquist theorem: need 2× the highest frequency)
  
- 3 seconds of audio = 16000 × 3 = 48,000 numbers!

Example: "Hello" (0.5 seconds)
Time: |----0.1s----|-0.2s-|-0.3s-|-0.4s-|-0.5s-|
Wave: ↗↘↗↘ ↗↘ ↗↘ ↗↘ ↗↘ ↗↘
      H   e   l   l   o

Problem: 48,000 numbers is TOO MUCH!
         And waveform doesn't show WHAT sounds are present

Solution: Convert to Mel Spectrogram!

Preprocessed: Mel Spectrogram
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Think of it as: Musical sheet music for AI!

Waveform shows: Amplitude over TIME
Spectrogram shows: FREQUENCIES over TIME

Visual analogy:
Waveform:      ↗↘↗↘↗↘  (hard to interpret)
Spectrogram:   
  High Freq █░░█  ← "S" sounds
  Mid Freq  ░██░  ← "E" vowel sound
  Low Freq  █░█░  ← "O" vowel sound
           ├─┼─┤
         Time →

Now we can SEE the different sounds!

Mel Spectrogram dimensions:
- Time axis: ~100 frames per second → 300 frames for 3 seconds
- Frequency axis: 128 mel bins (frequency buckets)
- Result: (300, 128) = 38,400 values

Still large, but now we can SEE patterns!

Why "Mel"?
- Mel scale = how HUMANS perceive pitch
- Low frequencies: finely separated (we're sensitive)
- High frequencies: coarsely separated (we're less sensitive)
- Example: 100Hz → 200Hz sounds big, but 10,000Hz → 10,100Hz barely noticeable

Embedded (make it like tokens):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- Convolutional encoding (find sound patterns)
- Temporal downsampling: 100 frames/sec → 12.5 frames/sec
  (Compress 8× to reduce computation)
  300 frames → 37.5 frames (about 38 tokens!)
  
- Frame embeddings: Each frame → one embedding vector
  Just like: Each word → one embedding!
```

**Challenges:**
- Temporal dynamics (sounds change over time)
  "Hello" has 5 sounds in sequence: H-E-L-L-O
  Order matters! "olleH" is different!
  Solution: Transformer captures temporal patterns
  
- Speaker variation (everyone sounds different!)
  Same word "hello":
  - Man's voice: deep, low frequencies
  - Woman's voice: higher frequencies
  - Child's voice: even higher!
  Solution: Training on diverse speakers
  
- Background noise
  "Hello" said in: quiet room vs noisy street = very different!
  Solution: Data augmentation (add noise during training)
  
- Different languages and accents
  "Hello" in English vs "Bonjour" in French = totally different!
  Even "Hello" in British vs American accent differs!
  Solution: Large multilingual training data

---

### 4. **Video** 🎬

**Representation:**
```
Raw: Sequence of images + audio
     30 fps × 10 seconds = 300 frames
     + audio stream

Preprocessed:
- Sample key frames (e.g., 1 per second)
- Process images separately
- Process audio separately
- Align temporal information
```

**Challenges:**
- Massive data (combines image + audio challenges)
- Temporal coherence across frames
- Synchronization between visual and audio
- Action understanding

---

## 🔗 Multimodal Fusion Strategies

### How to Combine Different Modalities?

This is the MILLION DOLLAR QUESTION in multimodal AI!

**The Problem:**
```
We have:
- Text embeddings: [0.2, -0.5, 0.3, ...]
- Image embeddings: [0.8, 0.1, -0.2, ...]
- Audio embeddings: [-0.1, 0.6, 0.4, ...]

How do we combine them into ONE understanding?
```

**Analogy: Making a Smoothie**
```
You have:
- Bananas (text)
- Strawberries (image)
- Yogurt (audio)

How to combine them?

Option 1: Throw everything in blender at once (Early Fusion)
Option 2: Blend each separately, then mix (Late Fusion)
Option 3: Process each, then blend together (Hybrid Fusion)
```

Let's explore each approach:

#### 1. **Early Fusion** (Blend Everything at Once)

Combine raw inputs before processing.

**Analogy:** Throw all ingredients in the blender at once.

```
        Text          Image         Audio
         ↓              ↓            ↓
    ┌────────────────────────────────┐
    │  Concatenate raw inputs        │
    │  [text_data][image_pixels][waveform] │
    └────────────────┬───────────────┘
                     ↓
           Unified Neural Network
           (processes everything together)
                     ↓
                  Output

Example:
Text: "cat" = [1, 2, 3]
Image: 🐱  = [150K pixel values]
Audio: "meow" = [48K waveform samples]
↓
Concatenate: [1, 2, 3, ...150K pixels..., ...48K samples...]
↓
One big neural network processes this MASSIVE input

Pros: 
✅ Simple, just concatenate
✅ Learns joint features early (can find patterns across modalities)

Cons: 
❌ High dimensionality (millions of inputs!)
❌ Modality-specific patterns lost
   (The network treats pixels and text the same - but they're different!)
❌ Can't handle missing modalities
   (What if you only have text, no image?)
```

---

#### 2. **Late Fusion** (Process Each Separately, Then Mix)

Process each modality separately, combine results at the end.

**Analogy:** Make banana smoothie, strawberry smoothie, and yogurt separately, then mix.

```
Text → Text Model → Text Features ─┐
       (specialized for text)      │
                                   │
Image → Image Model → Image Features ┬→ Combine → Output
        (specialized for images)  │   (voting or averaging)
                                   │
Audio → Audio Model → Audio Features ─┘
        (specialized for audio)

Example:
Input: 🖼️ [Cat image] + 🎤 [Meow sound] + 📝 "What animal is this?"

Text Model:  "animal" + "this" → Feature vector [0.2, 0.8, ...]
             (Understanding: Question about animal identification)

Image Model: [Cat pixels] → Feature vector [0.9, 0.1, ...]
             (Understanding: This looks like a cat - 90% confidence)

Audio Model: [Meow waveform] → Feature vector [0.85, 0.15, ...]
             (Understanding: This sounds like a cat - 85% confidence)

Combine (e.g., averaging):
Result: 0.9 (image) + 0.85 (audio) + 0.2 (text is neutral) → "Cat!" (91% confidence)

Pros: 
✅ Specialized processing per modality
   Each model is an EXPERT in its domain!
✅ Can handle missing modalities
   (No image? Just use text + audio!)
✅ Easier to train (train each model separately)

Cons: 
❌ Limited cross-modal interaction
   Models don't "talk" to each other until the very end
   Example: Image model can't use audio clues while processing
❌ Late integration may miss subtle interactions
   (Can't learn "when it looks like X and sounds like Y, it means Z")
```

---

#### 3. **Hybrid Fusion** (μOmni uses this!) ⭐

**Best of both worlds!** Process each modality with specialized encoder, THEN let them interact.

**Analogy:** Process banana with banana blender, strawberries with fruit processor, yogurt with mixer, THEN combine and blend together to let flavors meld!

```
   Text          Image           Audio
    ↓              ↓              ↓
Text Encoder  Image Encoder  Audio Encoder
(Expert in    (Expert in     (Expert in
 text)         images)         audio)
    ↓              ↓              ↓
  Embed          Embed          Embed
 (tokens)      (patches)       (frames)
    ↓              ↓              ↓
   Project       Project        Project
 (→256 dim)    (→256 dim)     (→256 dim)
    ↓              ↓              ↓
    └──────────┬───┴──────────────┘
               ↓
     [IMG tokens][AUDIO tokens][TEXT tokens]
     All in the SAME 256-dimensional space!
     Now they can "talk" to each other!
               ↓
      Unified Transformer (Thinker)
      ┌─────────────────────────┐
      │ Text tokens can attend  │
      │ to Image tokens!        │
      │                         │
      │ Image tokens can attend │
      │ to Audio tokens!        │
      │                         │
      │ Everything interacts!   │
      └─────────────────────────┘
               ↓
            Output

Real Example in μOmni:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Input: 🖼️ [Dog image] + 📝 "Describe this animal"

Step 1: Specialized Encoding
Vision Encoder: [Dog pixels] → (1, 196, 128)  (196 patch embeddings)
                Extract CLS token → (1, 1, 128) (summarize whole image)

Text Encoder: "Describe this animal"
              → tokens: [15, 42, 89, 234]
              → embeddings: (1, 4, 256)

Step 2: Project to Same Dimension
Vision Projector: (1, 1, 128) → (1, 1, 256)  ✓ Now 256-dim!
(Text already 256-dim, no projection needed)

Step 3: Concatenate
Combined: [CLS_token (1, 256)] + [TEXT_tokens (4, 256)]
        = (1, 5, 256)  ← 5 tokens total, all 256-dimensional

Step 4: Unified Processing
Thinker Transformer:
- Token 1 (image) attends to Tokens 2-5 (text)
  "Ah, they're asking me to DESCRIBE this"
- Tokens 2-5 (text) attend to Token 1 (image)
  "Ah, THIS is a dog with brown fur"
- All tokens interact and build understanding!

Output: "This is a brown dog sitting on grass"

Pros: 
✅ Specialized encoders per modality (best feature extraction!)
✅ Cross-modal attention in unified space (tokens interact!)
✅ Flexible (can handle any combination)
   Input: Image only? Just use image token!
   Input: Text only? Just use text tokens!
   Input: Both? All tokens work together!
✅ Scalable (can add more modalities easily)
   Want to add video? Just add video encoder + projector!

Cons:
❌ More complex architecture (multiple components)
❌ Need to align modalities (projectors must map to same space)

Why μOmni uses this:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✓ Maximum flexibility
✓ Each modality gets optimal processing
✓ Rich cross-modal interactions (attention connects everything)
✓ Can handle text, image, audio, or any combination!
```

---

## 🏗️ μOmni's Multimodal Architecture

### High-Level Overview

```
┌─────────────────────────────────────────────┐
│              INPUT STAGE                    │
│                                             │
│  🖼️ Image  →  Vision Encoder (ViT)         │
│                ↓                            │
│               CLS token                     │
│                ↓                            │
│           Vision Projector                  │
│                ↓                            │
│          (1, 1, 256) embedding              │
│─────────────────────────────────────────────│
│  🎤 Audio  →  Audio Encoder (AuT)           │
│                ↓                            │
│           Frame embeddings                  │
│                ↓                            │
│           Audio Projector                   │
│                ↓                            │
│          (1, T_audio, 256) embeddings       │
│─────────────────────────────────────────────│
│  📝 Text   →  Tokenizer                     │
│                ↓                            │
│            Token IDs                        │
│                ↓                            │
│         Token Embeddings                    │
│                ↓                            │
│          (1, T_text, 256) embeddings        │
└─────────────────────────────────────────────┘
                 ↓
┌─────────────────────────────────────────────┐
│           FUSION STAGE                      │
│                                             │
│  Concatenate all embeddings:                │
│  [IMG] + [AUDIO] + [TEXT]                   │
│         ↓                                   │
│  (1, 1+T_audio+T_text, 256)                 │
└─────────────────────────────────────────────┘
                 ↓
┌─────────────────────────────────────────────┐
│        PROCESSING STAGE                     │
│                                             │
│     Thinker (Decoder-Only Transformer)      │
│                                             │
│  - Multi-head self-attention                │
│  - All tokens attend to each other          │
│  - Cross-modal interactions emerge          │
└─────────────────────────────────────────────┘
                 ↓
┌─────────────────────────────────────────────┐
│            OUTPUT STAGE                     │
│                                             │
│  📝 Text: Next-token prediction             │
│  🔊 Speech: Talker → RVQ codes → Audio      │
└─────────────────────────────────────────────┘
```

---

## 🎯 Key Multimodal Challenges

### 1. **Alignment Problem**

Different modalities have different scales and representations.

```
Problem:
- Text: 1 word ≈ 1 token
- Image: 1 image ≈ 196 patch tokens (ViT)
- Audio: 1 second ≈ 12.5 frames

How to align them in a unified space?

Solution: Projectors!
Each encoder outputs to same dimension (d_model=256)
```

---

### 2. **Modality Gap**

Different modalities have different statistical properties.

```
Text embeddings cluster:
     *  *
   *      *
  *        *
   *      *
     *  *

Image embeddings cluster:
        +  +
      +      +
    +          +
      +      +
        +  +

Gap between clusters!

Solution: 
- Joint training with contrastive losses (CLIP-style)
- Projectors that learn to align distributions
- Supervised fine-tuning (SFT)
```

---

### 3. **Computational Complexity**

```
Memory usage comparison:

Text only:   512 tokens × 256 dim = 131K values
Image added: 196 tokens × 256 dim = 50K values (38% increase)
Audio added: ~100 tokens × 256 dim = 25K values (19% increase)

Total: ~206K values (57% increase from text-only!)

Plus: Cross-attention between all tokens = O(N²) complexity
```

**μOmni's solution:** Small context (512-2048), efficient architecture

---

### 4. **Data Requirements**

Need paired multimodal data:

```
✅ Good: Image + Caption
   🖼️ [Cat photo] + "A cat sitting on a couch"

✅ Good: Audio + Transcription
   🎤 [Speech audio] + "Hello world"

❌ Hard: Image + Audio + Text + aligned actions
   (Expensive to collect and annotate!)
```

---

## 💡 Cross-Modal Learning

### What Can Multimodal Models Do?

#### 1. **Cross-Modal Retrieval**

```
Query: "sunset over ocean" (text)
Retrieve: 🖼️ [Relevant sunset images]

Query: 🖼️ [Image of guitar]
Retrieve: "acoustic guitar, musical instrument, wooden" (text)
```

#### 2. **Cross-Modal Generation**

```
Input: 🖼️ [Image of food]
Output: "A delicious pizza with mushrooms and peppers" (text)

Input: "A futuristic city at night" (text)  
Output: 🖼️ [Generated image] (not in μOmni, requires diffusion model)
```

#### 3. **Cross-Modal Reasoning**

```
Input: 🖼️ [Image showing a person with umbrella] + 
       "Why is the person carrying an umbrella?"
Output: "It appears to be raining based on the wet ground 
         and the person's protective posture."

Requires:
- Visual understanding (see umbrella, wet ground)
- World knowledge (umbrellas used in rain)
- Reasoning (connect observations)
```

---

## 🚀 μOmni's Multimodal Capabilities

### What μOmni Can Do

```
✅ Image Understanding
   Input: 🖼️ [Photo] + "Describe this image"
   Output: Text description

✅ Visual Question Answering (VQA)
   Input: 🖼️ [Photo] + "What color is the car?"
   Output: "Red"

✅ Audio Understanding (ASR)
   Input: 🎤 [Speech] + "What did you hear?"
   Output: Transcription

✅ Multimodal Reasoning
   Input: 🖼️ [Image] + 🎤 [Audio] + "Explain what's happening"
   Output: Combined understanding

✅ Text-to-Speech
   Input: "Hello world"
   Output: 🔊 [Audio waveform]

✅ OCR (Text Extraction from Images)
   Input: 🖼️ [Image with text] + --ocr flag
   Output: Extracted text (can be integrated with multimodal understanding)
```

### What μOmni Cannot Do (Yet)

```
❌ Image Generation (would need diffusion model)
❌ Video understanding (limited to frame sampling)
❌ Real-time streaming (batch processing only)
❌ Multi-turn audio conversations (no speaker diarization)
```

---

## 📊 Comparison with Other Multimodal Models

| Model | Text | Image | Audio | Video | Generation |
|-------|------|-------|-------|-------|-----------|
| **GPT-4** | ✅ | ✅ | ❌ | ❌ | Text only |
| **GPT-4 Vision** | ✅ | ✅ | ❌ | ✅ | Text only |
| **Gemini** | ✅ | ✅ | ✅ | ✅ | Text, some image |
| **Qwen-Audio** | ✅ | ❌ | ✅ | ❌ | Text + audio |
| **Qwen3 Omni** | ✅ | ✅ | ✅ | ❌ | Text + audio |
| **μOmni** | ✅ | ✅ | ✅ | 🟡 | Text + audio |

🟡 = Limited support (frame sampling)

---

## 🎨 Visualization: Embeddings Space

### How Modalities Align

```
Unified Embedding Space (d=256):

Text "cat":           ●───────┐
                              ├─→ Close in space!
Image [cat photo]:    ●───────┤   (aligned representations)
                              │
Audio "meow":         ●───────┘

Text "dog":           ▲───────┐
                              ├─→ Close to each other
Image [dog photo]:    ▲───────┤   but far from cat
                              │
Audio "bark":         ▲───────┘

Training aligns semantically similar concepts!
```

---

## 💻 Code Example: Multimodal Forward Pass

```python
# Simplified μOmni multimodal processing

def multimodal_forward(image, audio, text):
    embeddings = []
    
    # 1. Process image (if provided)
    if image is not None:
        img_features = vision_encoder(image)  # → (1, 196, 128)
        cls_token = img_features[:, 0:1, :]   # → (1, 1, 128)
        img_emb = vision_projector(cls_token) # → (1, 1, 256)
        embeddings.append(img_emb)
    
    # 2. Process audio (if provided)
    if audio is not None:
        mel = audio_to_mel(audio)             # → (1, T, 128)
        aud_features = audio_encoder(mel)     # → (1, T', 192)
        aud_emb = audio_projector(aud_features) # → (1, T', 256)
        embeddings.append(aud_emb)
    
    # 3. Process text
    token_ids = tokenizer.encode(text)        # → [15, 24, ...]
    text_emb = token_embedding(token_ids)     # → (1, T_text, 256)
    embeddings.append(text_emb)
    
    # 4. Concatenate all modalities
    combined = torch.cat(embeddings, dim=1)   # → (1, T_total, 256)
    
    # 5. Process through Thinker
    output = thinker(embeddings=combined)     # → (1, T_total, vocab_size)
    
    return output
```

---

## 💡 Key Takeaways

✅ **Multimodal AI** processes multiple data types (text, image, audio, video)  
✅ **Hybrid fusion** combines specialized encoders with unified processing  
✅ **Projectors** align different modalities in a common embedding space  
✅ **Transformers** naturally handle multimodal tokens via attention  
✅ **μOmni** implements text + image + audio understanding and generation  
✅ **Challenges**: Alignment, modality gap, computational cost, data requirements

---

## 🎓 Self-Check Questions

1. What does "multimodal" mean in AI?
2. What are the three fusion strategies for multimodal learning?
3. Why do we need projectors in μOmni's architecture?
4. Name three things μOmni can do with multimodal inputs (including OCR).
5. What is the "modality gap" problem?

<details>
<summary>📝 Click to see answers</summary>

1. Multimodal AI systems can understand and generate multiple types of data (text, images, audio, video) simultaneously
2. Early fusion (combine inputs first), Late fusion (process separately, combine results), Hybrid fusion (specialized encoders + unified processing)
3. Projectors map different modality embeddings (different dimensions) to the same dimension (d_model) so they can be processed together
4. Any three: image description, VQA, audio transcription, multimodal reasoning, text-to-speech, OCR (text extraction from images)
5. Different modalities have different statistical properties and tend to cluster separately in embedding space, requiring alignment
</details>

---

## ➡️ Next Steps

Now you understand multimodal AI! Let's dive deeper into how embeddings work.

[Continue to Chapter 06: Understanding Embeddings →](06-embeddings-explained.md)

Or return to the [Index](00-INDEX.md) to choose a different chapter.

---

**Chapter Progress:** Foundation ●●●●● (5/5 complete) 
**Next Section:** Core Concepts →

