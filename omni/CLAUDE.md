# omni/ — Core Model Library

All neural network modules for μOmni. Imported by training scripts, inference, and tests.

## Files

| File | Classes | Purpose |
|------|---------|---------|
| `thinker.py` | ThinkerLM, Block, Attention, MLP, MoE, SwiGLU, SpikingNeuron, LiquidTimeConstant | Core LLM — decoder-only transformer with RoPE, optional GQA/MoE/Arthemis |
| `audio_encoder.py` | AudioEncoderTiny, ConvDown, EncoderBlock, AttentionPooling | Mel → 8x Conv downsample → Transformer encoder (CTC or CLAP mode) |
| `vision_encoder.py` | ViTTiny, TransformerTextEncoder, AttentionPooling | ViT image encoder + CLIP text encoder for contrastive training |
| `talker.py` | TalkerTiny | AR speech code predictor — predicts RVQ base+residual codes per frame |
| `codec.py` | RVQ, HiFiGANVocoder, GriffinLimVocoder, MultiPeriodDiscriminator, MultiScaleDiscriminator, ResBlock, NeuralVocoder | Speech codec (2 codebooks × 128) + vocoders (neural + classical) |
| `ocr_model.py` | OCRModel, OCRDecoder, OCRDecoderBlock | ViT encoder + cross-attention decoder for text extraction from images |
| `tokenizer.py` | BPETokenizer | SentencePiece BPE wrapper (train_new, encode, decode) |
| `utils.py` | RoPE, RMSNorm, EMA, LRFinder, LRSpike, ProjectionHead, LearnableTemperature, TextDataset, ASRDataset, TTSDataset, VocoderDataset, ImgCapDataset, MixDataset, OCRDataset | Shared utilities — positional encoding, normalization, training helpers, all streaming datasets, checkpoint management, collate functions |

## Performance Rules
- **RoPE** (`utils.py`): `inv_freq` is a registered buffer; cos/sin tables cached lazily in `_build_cache()`. Never recompute per forward.
- **Causal masks** (`thinker.py`, `talker.py`, `ocr_model.py`): Pre-allocated via `register_buffer("_causal_mask", ...)`. Slice with `self._causal_mask[:, :, :T, :T]`.
- **GQA** (`thinker.py` Attention): Expand KV heads with `unsqueeze(2).expand().reshape()` — zero-copy. Never use `repeat_interleave`.
- **MoE** (`thinker.py` MoE): Sort tokens by expert ID, process in single loop with `index_add_`. Never nested for-loops.
- **RVQ** (`codec.py`): Use `torch.cdist` for nearest-neighbor search. Never broadcast `(residual[:,None,:] - code[None,:,:])`.
- **No NaN/Inf checks** in any forward pass — they sync the GPU and kill throughput.
- **Flash Attention**: All attention modules use `scaled_dot_product_attention` when available (PyTorch 2.0+).

## Config vs Code Defaults
Class `__init__` defaults (e.g., `ThinkerLM(d=512)`) are for standalone use. **JSON configs are the source of truth** for training. The configs use different values (e.g., `d_model=384`).
