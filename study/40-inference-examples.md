# Chapter 40: Inference Examples

[← Previous: Running Training](39-running-training.md) | [Back to Index](00-INDEX.md) | [Next: Customization →](41-customization-guide.md)

---

## 🎯 Using Trained μOmni

Practical examples of using μOmni for different tasks via the `infer_chat.py` command-line interface.

---

## 💬 Example 1: Text-Only Chat

### Interactive Chat Mode

```bash
# Start interactive chat (text-only)
python infer_chat.py --ckpt_dir checkpoints/thinker_tiny

# You: What is machine learning?
# μOmni: Machine learning is a field of AI that...
```

### Single Query Mode

```bash
# Single text query
python infer_chat.py \
    --ckpt_dir checkpoints/thinker_tiny \
    --text "What is machine learning?"

# Output: Machine learning is a field of AI that enables computers...
```

**Note:** For text-only inference, use `checkpoints/thinker_tiny`. For multimodal, use `checkpoints/omni_sft_tiny`.

---

## 🖼️ Example 2: Image Understanding

### Image Question Answering

```bash
# Ask about an image
python infer_chat.py \
    --ckpt_dir checkpoints/omni_sft_tiny \
    --image data/images/images/n01440764_18.JPEG \
    --text "What animal is in this image?"

# Output: This is a cat sitting on a surface...
```

### Image Captioning (Default Prompt)

```bash
# Automatic captioning with default prompt
python infer_chat.py \
    --ckpt_dir checkpoints/omni_sft_tiny \
    --image data/images/images/n01440764_18.JPEG

# Uses default prompt: "Describe what you see concisely."
```

### Custom Image Prompt

```bash
# Custom prompt for detailed description
python infer_chat.py \
    --ckpt_dir checkpoints/omni_sft_tiny \
    --image data/images/images/n01440764_18.JPEG \
    --text "Describe this image in detail, including colors, objects, and scene composition."
```

---

## 🔍 Example 3: OCR (Text Extraction from Images)

### Extract Text from Image

```bash
# Extract text from image using OCR
python infer_chat.py \
    --ckpt_dir checkpoints/ocr_tiny \
    --image data/ocr/images/000001.png \
    --ocr

# Output: OCR extracted text: "HELLO"
#         μOmni (text): Extracted text from image: HELLO. Describe what you see.
```

### OCR with Text Prompt

```bash
# Extract text and describe the image
python infer_chat.py \
    --ckpt_dir checkpoints/omni_sft_tiny \
    --image data/ocr/images/000001.png \
    --text "What text do you see in this image?" \
    --ocr

# Output: OCR extracted text: "HELLO"
#         μOmni (text): The image contains the text "HELLO"...
```

**Note:** OCR model extracts text from images and can be combined with multimodal understanding for richer descriptions.

---

## 🎤 Example 4: Audio Transcription

### Speech-to-Text

```bash
# Transcribe audio with default prompt
python infer_chat.py \
    --ckpt_dir checkpoints/omni_sft_tiny \
    --audio_in data/audio/wav/000000.wav

# Uses default prompt: "What did you hear?"
```

### Audio with Custom Prompt

```bash
# Transcribe with specific instruction
python infer_chat.py \
    --ckpt_dir checkpoints/omni_sft_tiny \
    --audio_in data/audio/wav/000000.wav \
    --text "Transcribe this audio accurately."
```

### Multimodal: Image + Audio

```bash
# Combine image and audio input
python infer_chat.py \
    --ckpt_dir checkpoints/omni_sft_tiny \
    --image data/images/images/n01440764_18.JPEG \
    --audio_in data/audio/wav/000000.wav \
    --text "What do you see and hear?"

# Output: I see a cat and hear [transcription]...
```

---

## 🔊 Example 4: Text-to-Speech Output

### Generate Speech from Text Response

```bash
# Text input with audio output (TTS)
python infer_chat.py \
    --ckpt_dir checkpoints/omni_sft_tiny \
    --text "Hello world, how are you today?" \
    --audio_out output.wav

# Generates both text response and audio file
```

### Image Description with Audio Output

```bash
# Describe image and generate speech
python infer_chat.py \
    --ckpt_dir checkpoints/omni_sft_tiny \
    --image data/images/images/n01440764_18.JPEG \
    --text "Describe this image." \
    --audio_out description.wav

# Saves both text response and audio narration
```

### Interactive Chat with TTS

```bash
# Interactive mode with audio output
python infer_chat.py \
    --ckpt_dir checkpoints/omni_sft_tiny \
    --audio_out output.wav

# Each response is also saved as audio
```

---

## 🎬 Example 5: Video Processing

### Video Understanding

```bash
# Process video (extracts frames automatically)
python infer_chat.py \
    --ckpt_dir checkpoints/omni_sft_tiny \
    --video examples/sample_video.mp4 \
    --text "Describe what happens in this video."

# Extracts 4 evenly-spaced frames and processes first frame
```

**Note:** Video processing extracts frames automatically. For better results, consider extracting frames manually and processing multiple frames.

---

## 🔄 Example 6: Using Random Data Samples

### Test with Random Data

```bash
# Use the test script to pick random samples from data/
python test_all_media.py

# Automatically finds random images, audio, and text from data folders
# Runs comprehensive tests across all modalities
```

---

## 🎛️ Command-Line Arguments

### Available Options

```bash
python infer_chat.py --help

# Required:
--ckpt_dir <path>          # Checkpoint directory (e.g., checkpoints/omni_sft_tiny)

# Optional inputs:
--image <path>             # Path to image file (.jpg, .png, etc.)
--video <path>             # Path to video file (.mp4, .avi, etc.)
--audio_in <path>          # Path to audio input file (.wav, .flac)
--text <string>            # Text prompt (optional for multimodal)
--prompt <string>          # Override default prompt

# Output:
--audio_out <path>         # Path to save audio output (TTS) (.wav)
```

### Checkpoint Selection

- **Text-only:** `--ckpt_dir checkpoints/thinker_tiny`
- **Multimodal:** `--ckpt_dir checkpoints/omni_sft_tiny`

The script automatically:
- Loads appropriate configs from checkpoint directory
- Falls back to default configs if not found
- Loads tokenizer, encoders, and projectors as needed

---

## 💡 Tips & Best Practices

### Image Processing
✅ **Supported formats:** `.jpg`, `.jpeg`, `.png`  
✅ **Auto-resized** to 224×224  
✅ **RGB conversion** handled automatically

### Audio Processing
✅ **Sample rate:** Automatically resampled to 16kHz  
✅ **Supported formats:** `.wav`, `.flac`  
✅ **Mel spectrogram** extraction handled automatically

### Performance
✅ **KV caching** enabled by default for faster generation  
✅ **Mixed precision (FP16)** used on CUDA devices  
✅ **Autoregressive generation** with greedy decoding

### Data Sources
✅ **Use random samples** from `data/` folders for testing  
✅ **Test script** (`test_all_media.py`) picks random samples automatically  
✅ **Production data** from download scripts is ready to use

### Troubleshooting
⚠️ **Missing checkpoints:** Script will warn but continue with untrained models  
⚠️ **Missing projectors:** Multimodal features won't be used if `omni.pt` not found  
⚠️ **Audio output:** Requires vocoder (Griffin-Lim) - will warn if unavailable

---

## 📝 Example Workflows

### Workflow 1: Image Analysis Pipeline

```bash
# 1. Find random image
find data/images -name "*.jpg" | shuf -n 1

# 2. Analyze it
python infer_chat.py \
    --ckpt_dir checkpoints/omni_sft_tiny \
    --image $(find data/images -name "*.jpg" | shuf -n 1) \
    --text "Describe this image in detail."

# 3. Generate audio narration
python infer_chat.py \
    --ckpt_dir checkpoints/omni_sft_tiny \
    --image $(find data/images -name "*.jpg" | shuf -n 1) \
    --text "Describe this image." \
    --audio_out narration.wav
```

### Workflow 2: Audio Transcription Batch

```bash
# Process multiple audio files
for audio in data/audio/wav/*.wav; do
    python infer_chat.py \
        --ckpt_dir checkpoints/omni_sft_tiny \
        --audio_in "$audio" \
        --text "Transcribe this audio."
done
```

### Workflow 3: Interactive Multimodal Chat

```bash
# Start interactive mode (supports text, image, audio inputs)
python infer_chat.py --ckpt_dir checkpoints/omni_sft_tiny

# In interactive mode, you can:
# - Type text questions
# - The model responds with text
# - Use --audio_out to also generate speech
```

---

## 🔗 Related: Model Export

After training and testing, you may want to export your model for deployment:

- **Export to safetensors**: See [Chapter 46: Model Export and Deployment](46-model-export-deployment.md)
- **Quick export guide**: See [Chapter 47: Quick Start Export](47-quick-start-export.md)

The exported model can be used with `infer_standalone.py` (included in export folder) which uses Hugging Face transformers library and loads from a single safetensors file. No codebase required!

---

[Continue to Chapter 41: Customization Guide →](41-customization-guide.md)

---
