# Glossary: Technical Terms Explained

This glossary explains technical terms used throughout the μOmni documentation in simple, beginner-friendly language.

## A

### **AMP (Automatic Mixed Precision)**
**What it is**: A technique that uses both 16-bit (FP16) and 32-bit (FP32) numbers during training.

**Why it matters**: Makes training 1.5-2x faster and uses ~50% less memory, with minimal quality loss.

**Simple analogy**: Like using a smaller, faster car for most of the journey, but switching to a larger truck only when you need more precision.

### **ASR (Automatic Speech Recognition)**
**What it is**: Converting spoken audio into text.

**Example**: You say "Hello" → ASR outputs the text "Hello"

**In μOmni**: The Audio Encoder learns to recognize speech patterns and convert them to text.

### **Attention Mechanism**
**What it is**: A way for the model to "focus" on important parts of the input when making predictions.

**Simple analogy**: Like reading a sentence and paying more attention to key words that help you understand the meaning.

**Example**: When processing "The cat sat on the mat", attention might focus more on "cat" and "mat" to understand the relationship.

### **Autoregressive Generation**
**What it is**: Generating output one piece at a time, where each new piece depends on all previous pieces.

**Example**: Generating text word-by-word: "The" → "The cat" → "The cat sat" → "The cat sat on" → ...

**Why it matters**: Allows the model to generate coherent, context-aware sequences.

## B

### **Backward Pass (Backpropagation)**
**What it is**: The process of calculating how much each weight should change to reduce the error.

**Simple analogy**: Like figuring out which adjustments to make after seeing your test results - you work backwards from the mistakes.

**When it happens**: After the forward pass, during training.

### **Batch**
**What it is**: A group of examples processed together at the same time.

**Example**: Instead of processing one image at a time, process 8 images together.

**Why it matters**: More efficient use of GPU memory and faster training.

### **BOS Token (Beginning of Sequence)**
**What it is**: A special token that marks the start of a text sequence.

**Example**: `[BOS] "Hello world"` instead of just `"Hello world"`

**Why it matters**: Helps the model know where sequences begin.

### **BPE (Byte Pair Encoding)**
**What it is**: A method for breaking text into smaller pieces (tokens) that the model can understand.

**Example**: "Hello world" might become `["Hel", "lo", " wor", "ld"]` instead of individual characters.

**Why it matters**: More efficient than character-level, more flexible than word-level tokenization.

## C

### **Checkpoint**
**What it is**: A saved snapshot of the model's current state, including weights, training progress, and optimizer state.

**Simple analogy**: Like a save file in a video game - you can resume exactly where you left off.

**Contains**: Model weights, optimizer state, scheduler state, current step number, best validation loss.

### **CLIP (Contrastive Language-Image Pre-training)**
**What it is**: A training method that learns to align images and text in the same space.

**Simple analogy**: Like learning that a picture of a cat and the word "cat" should be close together in meaning.

**In μOmni**: Used for training the Vision Encoder to understand images and their captions.

### **CLS Token (Classification Token)**
**What it is**: A special token added to sequences that represents the entire input.

**Example**: In an image, the CLS token learns to represent the whole image after processing all patches.

**Why it matters**: Provides a single vector representing the entire input, useful for downstream tasks.

### **Codebook**
**What it is**: A collection of learned vectors used for quantization (converting continuous values to discrete codes).

**Simple analogy**: Like a dictionary where each word (code) has a specific meaning (vector).

**In μOmni**: RVQ uses 2 codebooks, each with 128 codes, to represent audio.

### **Contrastive Learning**
**What it is**: A training method that learns by comparing similar and different examples.

**Simple analogy**: Learning to recognize faces by seeing many examples of the same person (similar) and different people (different).

**In μOmni**: Vision Encoder learns that matching image-caption pairs should be similar, non-matching pairs should be different.

### **CTC (Connectionist Temporal Classification)**
**What it is**: A method for aligning sequences of different lengths (like audio frames and text characters).

**Problem it solves**: Audio has many frames, text has fewer characters - CTC handles the alignment automatically.

**Example**: 100 audio frames might correspond to 10 text characters - CTC figures out the alignment.

### **CTX_LEN (Context Length)**
**What it is**: The maximum number of tokens the model can process at once.

**Example**: If ctx_len = 512, the model can handle up to 512 tokens in a single input.

**Why it matters**: Limits how much information can be processed in one go.

## D

### **Downsampling**
**What it is**: Reducing the size or resolution of data.

**Example**: Converting 100 audio frames per second to 25 frames per second (4x downsampling).

**Why it matters**: Reduces computational cost while preserving important information.

## E

### **Embedding**
**What it is**: Converting discrete items (like words or tokens) into continuous vectors (lists of numbers).

**Simple analogy**: Like translating words into a universal language of numbers that the computer understands.

**Example**: The word "cat" becomes a vector like `[0.2, -0.5, 0.8, ...]` (256 numbers).

### **EOS Token (End of Sequence)**
**What it is**: A special token that marks the end of a text sequence.

**Example**: `"Hello world" [EOS]` instead of just `"Hello world"`

**Why it matters**: Helps the model know where sequences end.

### **Epoch**
**What it is**: One complete pass through the entire training dataset.

**Example**: If you have 1000 examples and process them all once, that's 1 epoch.

**Why it matters**: Measures training progress - "after 10 epochs" means the model has seen all data 10 times.

## F

### **FP16 (Half Precision)**
**What it is**: Using 16-bit floating point numbers instead of 32-bit.

**Trade-off**: Less precision but faster and uses less memory.

**Simple analogy**: Like using a smaller measuring cup - less precise but faster to fill.

### **FP32 (Full Precision)**
**What it is**: Using 32-bit floating point numbers (standard precision).

**When used**: Critical calculations that need high precision, like gradient updates.

### **Forward Pass**
**What it is**: The process of data flowing through the network to produce an output.

**Simple analogy**: Like taking a test - you read the question (input) and write an answer (output).

**When it happens**: Both during training and inference.

## G

### **Gradient**
**What it is**: A measure of how much the loss changes when you change a weight slightly.

**Simple analogy**: Like the slope of a hill - tells you which direction to go to reduce error.

**Why it matters**: Used to update weights during training.

### **Gradient Accumulation**
**What it is**: Collecting gradients over multiple batches before updating weights.

**Simple analogy**: Like saving up money from several paychecks before making a big purchase.

**Why it matters**: Allows training with larger effective batch sizes when memory is limited.

### **Gradient Clipping**
**What it is**: Limiting the size of gradients to prevent them from becoming too large.

**Why it matters**: Prevents training from becoming unstable when gradients explode.

**Simple analogy**: Like putting a speed limit on how fast you can change direction.

### **GQA (Grouped Query Attention)**
**What it is**: A memory-efficient variant of attention that groups queries together.

**Why it matters**: Reduces memory usage while maintaining performance.

**Trade-off**: Slightly less flexible than full attention, but much more memory efficient.

## I

### **InfoNCE Loss**
**What it is**: A loss function used in contrastive learning that encourages similar items to be close and different items to be far apart.

**Simple analogy**: Like a magnet that pulls matching pairs together and pushes non-matching pairs apart.

**In μOmni**: Used for training the Vision Encoder with image-caption pairs.

### **Inference**
**What it is**: Using a trained model to make predictions on new data.

**Simple analogy**: Like taking a test after studying - you're using what you learned.

**Difference from training**: No learning happens, just prediction.

## K

### **KV Cache (Key-Value Cache)**
**What it is**: Storing previously computed attention states to avoid recomputing them.

**Why it matters**: Makes autoregressive generation much faster.

**Example**: When generating "The cat sat on the mat", you cache the attention for "The cat sat on" and only compute attention for "the mat".

## L

### **Learning Rate**
**What it is**: How big of steps the model takes when learning.

**Simple analogy**: Like the size of steps when walking - too big and you overshoot, too small and you never get there.

**Why it matters**: Critical hyperparameter that affects training speed and stability.

### **Loss Function**
**What it is**: A measure of how wrong the model's predictions are.

**Simple analogy**: Like a score on a test - lower is better.

**Example**: Cross-entropy loss measures how different predicted probabilities are from true labels.

## M

### **Mel Spectrogram**
**What it is**: A representation of audio that shows frequency content over time, using a scale that matches human hearing.

**Simple analogy**: Like a musical score that shows which notes (frequencies) play at which times.

**Why it matters**: More suitable for speech processing than raw audio waveforms.

### **Mixed Precision (see AMP)**
**What it is**: Using both FP16 and FP32 during training.

### **MLP (Multi-Layer Perceptron)**
**What it is**: A type of neural network layer with multiple fully-connected layers.

**Simple analogy**: Like a series of filters that progressively refine the information.

**In transformers**: Used in the feedforward part of each transformer block.

### **MoE (Mixture of Experts)**
**What it is**: A technique where different parts of the model specialize in different tasks.

**Simple analogy**: Like having specialists in a team - each expert handles their area of expertise.

**In μOmni**: Optional feature in Thinker that can improve efficiency.

### **Modality**
**What it is**: A type of data (text, image, audio, etc.).

**Example**: Text is one modality, images are another modality.

**In μOmni**: Handles three modalities - text, images, and audio.

## N

### **NaN (Not a Number)**
**What it is**: An invalid number that can occur when calculations go wrong.

**Why it matters**: Indicates numerical instability - the model needs to detect and handle this.

**Example**: Dividing zero by zero results in NaN.

## O

### **Optimizer**
**What it is**: An algorithm that updates model weights based on gradients.

**Simple analogy**: Like a coach that tells you how to adjust your technique based on your mistakes.

**Examples**: Adam, SGD (Stochastic Gradient Descent).

## P

### **Patch**
**What it is**: A small piece of an image.

**Example**: A 224×224 image might be split into 196 patches of 16×16 pixels each.

**Why it matters**: Allows transformers to process images by treating patches like tokens.

### **Perplexity**
**What it is**: A measure of how "surprised" the model is by the data - lower is better.

**Simple analogy**: Like measuring how confused someone is - lower confusion means better understanding.

**Formula**: Exponential of cross-entropy loss.

### **Projector**
**What it is**: A layer that converts embeddings from one dimension to another.

**Simple analogy**: Like a translator that converts between different languages (dimensions).

**In μOmni**: Converts vision (128-dim) and audio (192-dim) embeddings to Thinker's dimension (256-dim).

## Q

### **Quantization**
**What it is**: Converting continuous values to discrete codes.

**Simple analogy**: Like rounding numbers to the nearest integer.

**In μOmni**: RVQ quantizes continuous audio (mel spectrograms) into discrete codes.

## R

### **Residual Connection**
**What it is**: Adding the input of a layer directly to its output.

**Simple analogy**: Like keeping the original message while also adding new information.

**Why it matters**: Helps gradients flow better during training and allows learning identity mappings.

### **RMSNorm (Root Mean Square Normalization)**
**What it is**: A normalization technique that scales values based on their root mean square.

**Why it matters**: Stabilizes training and allows faster convergence.

**Alternative to**: Layer Normalization.

### **RoPE (Rotary Position Embedding)**
**What it is**: A way of encoding position information using rotations in the embedding space.

**Why it matters**: More efficient than learned position embeddings and handles variable sequence lengths better.

**In μOmni**: Used in Thinker for position encoding.

### **RVQ (Residual Vector Quantization)**
**What it is**: A multi-stage quantization method that quantizes the input, then quantizes the error (residual).

**Simple analogy**: Like taking a photo, then taking another photo of what the first one missed, combining both.

**In μOmni**: Uses 2 codebooks to quantize audio into discrete codes.

## S

### **Scheduler (Learning Rate Scheduler)**
**What it is**: An algorithm that adjusts the learning rate during training.

**Simple analogy**: Like starting with big steps, then gradually taking smaller steps as you get closer to the goal.

**In μOmni**: Uses warmup (gradual increase) then cosine decay (gradual decrease).

### **Self-Attention**
**What it is**: Attention mechanism where each position attends to all positions in the same sequence.

**Simple analogy**: Like reading a sentence and each word "looks at" all other words to understand context.

**Why it matters**: Allows the model to understand relationships between all parts of the input.

### **SFT (Supervised Fine-Tuning)**
**What it is**: Training a model on specific tasks with labeled examples.

**Simple analogy**: Like specialized training after general education.

**In μOmni**: Final training stage where all components work together on multimodal tasks.

### **SwiGLU**
**What it is**: An activation function used in transformer MLPs.

**Why it matters**: Better performance than standard ReLU activation.

**Technical**: Swish-Gated Linear Unit.

## T

### **Temperature**
**What it is**: A parameter that controls randomness in sampling - lower = more deterministic, higher = more random.

**Simple analogy**: Like a thermostat - lower temperature = more focused, higher = more exploratory.

**In μOmni**: Used in contrastive learning (default: 0.07) and text generation.

### **Token**
**What it is**: A piece of text after tokenization.

**Example**: "Hello world" might become tokens `["Hello", " world"]` or `["Hel", "lo", " wor", "ld"]` depending on tokenization method.

**Why it matters**: Models process tokens, not raw text.

### **Tokenization**
**What it is**: Breaking text into smaller pieces (tokens) that the model can process.

**Example**: "Hello world" → `["Hello", " world"]` or `[1234, 5678]` (token IDs).

**Methods**: BPE, word-level, character-level.

### **Transformer**
**What it is**: A neural network architecture based on attention mechanisms.

**Why it matters**: Powers most modern AI models (GPT, BERT, etc.).

**Key components**: Attention, MLP, normalization, residual connections.

### **TTS (Text-to-Speech)**
**What it is**: Converting text into spoken audio.

**Example**: Input "Hello" → Output: audio waveform saying "Hello"

**In μOmni**: Handled by Talker + RVQ + Vocoder.

## V

### **Validation**
**What it is**: Testing the model on data it hasn't seen during training.

**Simple analogy**: Like a practice test before the real exam.

**Why it matters**: Measures how well the model generalizes to new data.

### **Vocoder**
**What it is**: A component that converts mel spectrograms into audio waveforms.

**Simple analogy**: Like a synthesizer that turns musical notes (mel) into sound (waveform).

**In μOmni**: Uses improved Griffin-Lim algorithm.

## W

### **WER (Word Error Rate)**
**What it is**: A metric for measuring ASR accuracy - percentage of words that are incorrect.

**Example**: If the model outputs "The cat sat" but should be "The cat sat on", that's 1 error out of 4 words = 25% WER.

**Why it matters**: Standard metric for evaluating speech recognition systems.

### **Weight**
**What it is**: A learnable parameter in a neural network that determines how inputs are transformed.

**Simple analogy**: Like a dial that you adjust to make the model work better.

**Training**: Weights are updated during training to minimize loss.

---

## Quick Reference: Common Abbreviations

| Abbreviation | Full Name | What It Is |
|-------------|-----------|------------|
| **AMP** | Automatic Mixed Precision | Using FP16 for speed, FP32 for precision |
| **ASR** | Automatic Speech Recognition | Converting speech to text |
| **BOS** | Beginning of Sequence | Start token |
| **BPE** | Byte Pair Encoding | Tokenization method |
| **CLIP** | Contrastive Language-Image Pre-training | Vision training method |
| **CLS** | Classification Token | Special token representing entire input |
| **CTC** | Connectionist Temporal Classification | Sequence alignment method |
| **EOS** | End of Sequence | End token |
| **FP16** | 16-bit Floating Point | Half precision numbers |
| **FP32** | 32-bit Floating Point | Full precision numbers |
| **GQA** | Grouped Query Attention | Memory-efficient attention |
| **KV** | Key-Value | Attention cache |
| **MLP** | Multi-Layer Perceptron | Feedforward network |
| **MoE** | Mixture of Experts | Specialized model components |
| **NaN** | Not a Number | Invalid number |
| **RMSNorm** | Root Mean Square Normalization | Normalization technique |
| **RoPE** | Rotary Position Embedding | Position encoding method |
| **RVQ** | Residual Vector Quantization | Audio quantization method |
| **SFT** | Supervised Fine-Tuning | Final training stage |
| **SwiGLU** | Swish-Gated Linear Unit | Activation function |
| **TTS** | Text-to-Speech | Converting text to audio |
| **WER** | Word Error Rate | ASR accuracy metric |

---

**See Also:**
- [Introduction](00_Introduction.md) - High-level overview
- [Neural Networks Basics](01_Neural_Networks_Basics.md) - Fundamental concepts
- [Quick Reference](QUICK_REFERENCE.md) - Common commands and parameters

