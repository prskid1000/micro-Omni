# Quick Reference Guide

## File Structure

```
study/
├── README.md                    # Start here!
├── 00_Introduction.md           # What is μOmni?
├── 01_Neural_Networks_Basics.md # Fundamentals
├── 02_Architecture_Overview.md  # System design
├── 03_Thinker_Deep_Dive.md     # Core LLM
├── 04_Audio_Encoder.md         # Speech input
├── 05_Vision_Encoder.md         # Image input
├── 06_Talker_Codec.md          # Speech output
├── 07_Training_Workflow.md     # How to train
├── 08_Inference_Guide.md       # How to use
├── 09_Hands_On_Exercises.md    # Practice
├── QUICK_REFERENCE.md          # This file
└── diagrams/                    # Visual aids
```

## Key Concepts

### Neural Networks
- **Neuron**: Basic processing unit
- **Layer**: Group of neurons
- **Forward Pass**: Data flows through network
- **Backward Pass**: Gradients flow back (training)

### Transformers
- **Attention**: Focus on relevant parts
- **Self-Attention**: Look at all positions
- **Embeddings**: Convert tokens to vectors
- **Position Encoding**: Handle sequence order

### μOmni Components
- **Thinker**: Core language model
- **Audio Encoder**: Processes speech
- **Vision Encoder**: Processes images
- **Talker**: Generates speech
- **RVQ Codec**: Audio quantization
- **Projectors**: Align modalities

## Training Commands

```bash
# Stage A: Thinker
python train_text.py --config configs/thinker_tiny.json

# Stage B: Audio Encoder
python train_audio_enc.py --config configs/audio_enc_tiny.json

# Stage C: Vision Encoder
python train_vision.py --config configs/vision_tiny.json

# Stage D: Talker
python train_talker.py --config configs/talker_tiny.json

# Stage E: Multimodal SFT
python sft_omni.py --config configs/omni_sft_tiny.json
```

## Inference Commands

```bash
# Text only
python infer_chat.py --ckpt_dir checkpoints/thinker_tiny --text "Hello"

# Image + Text
python infer_chat.py --ckpt_dir checkpoints/omni_sft_tiny \
  --image path/to/image.png --text "Describe this"

# Audio Input
python infer_chat.py --ckpt_dir checkpoints/omni_sft_tiny \
  --audio_in path/to/audio.wav --text "What did you hear?"

# Text-to-Speech
python infer_chat.py --ckpt_dir checkpoints/omni_sft_tiny \
  --text "Hello world" --audio_out output.wav
```

## Configuration Files

- `configs/thinker_tiny.json` - Thinker settings
- `configs/audio_enc_tiny.json` - Audio encoder
- `configs/vision_tiny.json` - Vision encoder
- `configs/talker_tiny.json` - Talker + RVQ
- `configs/omni_sft_tiny.json` - Multimodal SFT

## Key Parameters

### Model Size
- `vocab_size`: Number of tokens (5000)
- `d_model`: Embedding dimension (256)
- `n_layers`: Number of transformer blocks (4)
- `n_heads`: Attention heads (4)
- `d_ff`: Feedforward dimension (1024)

### Training
- `max_steps`: Training steps (1000)
- `batch_size`: Examples per batch (8)
- `lr`: Learning rate (0.0003)
- `warmup_steps`: Warmup period (10)

### Audio
- `sample_rate`: Audio sample rate (16000)
- `mel_bins`: Mel spectrogram bins (128)
- `downsample_time`: Temporal reduction (8x)
- `frame_rate`: Target frame rate (12.5 Hz)

### Vision
- `img_size`: Image size (224)
- `patch`: Patch size (16)
- `n_patches`: Patches per image (196)

## Data Formats

### Text
- Format: Plain text files
- Location: `data/text/tiny_corpus.txt`
- Processing: BPE tokenization

### Images
- Format: PNG/JPG
- Size: 224×224 (resized)
- Location: `data/images/images/`
- Annotations: `data/images/annotations.json`

### Audio
- Format: WAV
- Sample Rate: 16 kHz
- Location: `data/audio/wav/`
- Metadata: `data/audio/asr.csv`, `data/audio/tts.csv`

## Common Issues

### Training
- **Loss not decreasing**: Check learning rate, data loading
- **Out of memory**: Reduce batch size
- **NaN values**: Check gradients, reduce LR

### Inference
- **Model not found**: Check checkpoint path
- **Poor quality**: Model needs more training
- **Slow generation**: Use GPU, enable KV cache

## Code Locations

- `omni/thinker.py` - Core LLM
- `omni/audio_encoder.py` - Audio processing
- `omni/vision_encoder.py` - Image processing
- `omni/talker.py` - Speech generation
- `omni/codec.py` - RVQ codec
- `omni/tokenizer.py` - Text tokenization

## Learning Path

1. **Beginner**: Read 00-02, try exercises
2. **Intermediate**: Read 03-06, understand components
3. **Advanced**: Read 07-08, train and deploy
4. **Expert**: Modify code, experiment

## Resources

- Main README: `../README.md`
- PyTorch Docs: https://pytorch.org/docs/
- Transformer Paper: "Attention Is All You Need"
- ViT Paper: "An Image is Worth 16x16 Words"

## Tips

1. **Start Small**: Use tiny configs first
2. **Read Code**: Look at actual implementations
3. **Experiment**: Try different parameters
4. **Debug**: Add print statements, check shapes
5. **Visualize**: Plot losses, attention weights

---

**Need Help?**
- Check the main [README.md](../README.md)
- Review specific component guides
- Look at code examples in `examples/`
- Run `test_all_media.py` to verify setup

