# Chapter 36: Optimization Techniques

[← Previous: Data Preparation](35-data-preparation.md) | [Back to Index](00-INDEX.md) | [Next: Debugging →](37-debugging-troubleshooting.md)

---

## ⚡ Performance Optimizations

Key techniques to speed up training and inference in μOmni.

---

## 🚀 Training Optimizations

### 1. Mixed Precision (FP16)

**What:** Use 16-bit floats instead of 32-bit  
**Benefit:** 2x faster, 50% less memory  
**Enabled by default** in μOmni

```python
# Automatically uses torch.cuda.amp.autocast
with torch.cuda.amp.autocast():
    output = model(input)
```

### 2. Gradient Accumulation

**What:** Accumulate gradients over multiple batches  
**Benefit:** Simulate larger batch sizes without OOM

```json
{
  "batch_size": 4,
  "gradient_accumulation_steps": 4
  // Effective batch size = 16
}
```

### 3. Gradient Checkpointing

**What:** Trade compute for memory  
**Benefit:** Train larger models on same GPU

```python
model.gradient_checkpointing_enable()
```

### 4. Flash Attention

**What:** Memory-efficient attention implementation  
**Benefit:** 2-4x faster, less memory

```python
# Automatically used if available
pip install flash-attn
```

### 5. CUDA Graphs Compilation

**What:** Compile models with `torch.compile()` using CUDA graphs backend  
**Benefit:** 10-20% speedup, reduced overhead  
**Requirement:** Fixed-length padding for variable-length sequences

```json
{
  "use_compile": true,
  "max_mel_length": 6000,    // For audio training (frames)
  "max_text_length": 256     // For OCR training (characters)
}
```

**Why Fixed Length?**
- CUDA graphs require uniform tensor shapes across batches
- Variable-length sequences cause "tensor size mismatch" errors
- Fixed padding ensures all batches have identical shapes

**Implementation:**
- Audio training: All mel spectrograms padded/truncated to `max_mel_length`
- OCR training: All text sequences padded/truncated to `max_text_length`
- Collate functions in `omni/utils.py` handle padding automatically:
  - `collate_mel_fn()` - For mel-only batches (talker training)
  - `collate_mel_text_fn()` - For mel+text batches (audio encoder training)
  - `collate_mel_audio_fn()` - For mel+audio batches (vocoder training)
- All collate functions support `max_mel_length` parameter for fixed-length padding

**Memory Trade-off:**
- Slightly more memory due to padding
- But enables CUDA graphs optimization (10-20% faster)
- Worth it for most use cases

**Determining Optimal Values:**
```bash
# Check actual lengths in your dataset
python scripts/check_mel_lengths.py --csv data/audio/production_asr.csv

# Recommendations:
# - Set to 99.5th percentile to cover most data
# - Round up to nearest 256 for memory alignment
# - Default: 2048 frames (~20s) for audio, 256 chars for text
```

---

## 🎯 Inference Optimizations

### 1. KV Caching

**Essential for generation!** Reuses computed key-value pairs.

**Without KV cache:**
- Generate 100 tokens: ~30 seconds
- Recomputes attention for all previous tokens every step

**With KV cache:**
- Generate 100 tokens: ~3 seconds
- 10x speedup!

### 2. Batch Inference

Process multiple inputs simultaneously:
```python
responses = model.chat_batch([
    "Question 1?",
    "Question 2?",
    "Question 3?"
])
```

### 3. Quantization (INT8)

**What:** Convert weights to 8-bit integers  
**Benefit:** 4x smaller model, faster inference  
**Trade-off:** Slight quality degradation

```python
model = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)
```

---

## 💾 Memory Optimizations

### 1. Streaming Dataset Loading

**What:** Stream data directly from files without pre-loading into RAM  
**Benefit:** Reduces RAM usage by 90%+ for large datasets  
**Status:** ✅ Implemented in all μOmni training scripts (2024 optimization)

**Technical Approach:**
- All datasets use `IterableDataset` for true streaming
- Sequential I/O with large buffers (8MB) for efficiency
- Direct file iteration - no offset tracking or caching
- Worker sharding for multi-process data loading
- Buffer-based shuffling for randomization

**Optimized Datasets (All Training Scripts):**

| Dataset | Files | Optimization | Memory Savings |
|---------|-------|--------------|----------------|
| **TextDataset** | `train_text.py` | Direct line streaming | No pre-loading |
| **ASRDataset** | `train_audio_enc.py` | CSV row streaming | No pre-loading |
| **TTSDataset** | `train_talker.py` | CSV row streaming | No pre-loading |
| **OCRDataset** | `train_ocr.py` | CSV row streaming | No pre-loading |
| **ImgCapDataset** | `train_vision.py` | JSON item streaming | No pre-loading |
| **VocoderDataset** | `train_vocoder.py` | CSV row streaming | No pre-loading |
| **MixDataset** | `sft_omni.py` | All three types streaming | Combined savings |

**Implementation Details:**

**Text Files:**
```python
# Streaming: Reads line-by-line directly
def __iter__(self):
    with open(self.path, 'r', encoding='utf-8', errors='ignore', buffering=8192*1024) as f:
        for idx, line in enumerate(f):
            # Worker sharding, train/val split, skip_samples support
            text = line.strip()
            if text:
                # Process and yield immediately
                yield processed_text
```

**CSV Files:**
```python
# Streaming: Uses csv.DictReader directly
def __iter__(self):
    with open(self.csv_path, 'r', encoding='utf-8', errors='ignore', buffering=8192*1024) as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            # Worker sharding, train/val split, skip_samples support
            # Process and yield immediately
            yield processed_data
```

**JSON Files:**
```python
# Streaming: Loads JSON once, then iterates
def __iter__(self):
    with open(self.manifest_path, 'r', encoding='utf-8') as f:
        items = json.load(f)
    for idx, item in enumerate(items):
        # Worker sharding, train/val split, skip_samples support
        # Process and yield immediately
        yield processed_item
```

**Memory Savings Examples:**
- **10M line text file**: ~500MB → ~0MB (only current line in memory)
- **1M row CSV**: ~200MB → ~0MB (only current row in memory)
- **100K image JSON**: ~50MB → ~50MB (JSON loaded once, then streamed)
- **Combined (MixDataset)**: All three optimizations applied simultaneously

**Key Benefits:**
- ✅ RAM usage now typically **lower than VRAM** (was opposite before)
- ✅ Can train on systems with limited RAM (8GB+)
- ✅ No cache files needed - simpler and cleaner
- ✅ True streaming - data processed on-demand
- ✅ Efficient resuming via `skip_samples` parameter (automatically handled by `setup_resume_data_loading()`)
- ✅ Worker sharding for multi-process data loading
- ✅ Buffer-based shuffling for randomization
- ✅ Automatic checkpoint detection and resuming (via `load_checkpoint()`)
- ✅ Proper validation on full dataset (via `ValidationSkipSamplesContext`)

### 2. Efficient Tokenizer Training

**What:** Train tokenizers on entire dataset efficiently  
**Benefit:** Train on full dataset with optimized settings

**Implementation:**
- **Plain text files:** Passed directly to SentencePiece (no streaming, no temp files)
- **CSV/JSON files:** Stream extraction to temp file in `data/.temp/` (streams row-by-row/item-by-item to extract text)
- Processes entire dataset (not just samples) efficiently
- Temporary files auto-cleaned after training
- **Always enables `train_extremely_large_corpus=True`:** Uses 64-bit indexing for maximum file size compatibility
- **BPE model type:** Faster than Unigram, good balance of speed and quality
- **Default speed optimization:** `input_sentence_size=10000000` (10M sentences) limits training data for faster training by default
- **Use all data:** Set `input_sentence_size=0` to use entire corpus (slower but uses more data)

**Memory Behavior:**
- **Plain text:** SentencePiece loads entire file into memory during training (no streaming)
- **CSV/JSON extraction:** Streams data extraction (avoids loading structured data into memory), but SentencePiece still loads the extracted temp file into memory
- **Temp files:** Only used for CSV/JSON text extraction, stored in `data/.temp/` and auto-cleaned

**Note:** SentencePiece loads the entire file into memory during training (whether original or extracted temp file). The `train_extremely_large_corpus` flag enables 64-bit indexing (instead of 32-bit) to handle files > 2GB, but doesn't reduce memory usage. Streaming is only used for extracting text from CSV/JSON structured data.

### 3. Resumable Preprocessing

**What:** All preprocessing operations can resume if interrupted  
**Benefit:** No need to restart from beginning if process is stopped

**Resumable Operations:**
- ✅ **Tokenizer training:** SentencePiece handles large files directly (no streaming needed for plain text)
- ✅ **CSV/JSON extraction:** Temp files created in `data/.temp/` (only when needed)
- ✅ **Vocabulary building:** Saves progress every 10K items (vision, OCR)
- ✅ **Token counting:** Saves progress every 10K samples (text, CSV, images)
- ✅ **Training loops:** Already resumable via checkpoints

**Checkpoint Locations:**
- Vocabulary building: `{save_dir}/vocab_build_checkpoint.json`
- Token counting: `{file_path}.token_count_checkpoint.json`
- OCR vocabulary: `{csv_path}.vocab_checkpoint.json`
- All checkpoints auto-cleaned after successful completion

**Related Optimizations:**
- Removed `gc.collect()` calls (Python's GC handles this automatically)
- Removed `torch.cuda.empty_cache()` calls (PyTorch manages CUDA memory efficiently)
- These manual calls were unnecessary and could actually hurt performance

### 4. DataLoader Workers

**What:** Parallel data loading processes  
**Trade-off:** More workers = faster loading but more RAM

```json
{
  "num_workers": 2,  // Default: 2 workers
  // Reduce to 0 or 1 if RAM is limited
}
```

**Recommendation:**
- **High RAM (32GB+)**: `num_workers: 2-4`
- **Medium RAM (16GB)**: `num_workers: 1-2`
- **Low RAM (8GB)**: `num_workers: 0-1`

### 5. Batch Size Tuning

**What:** Adjust batch size based on available memory  
**Benefit:** Maximize GPU utilization without OOM

```json
{
  "batch_size": 4,  // Start conservatively
  "gradient_accumulation_steps": 4  // Simulate batch_size=16
}
```

**Strategy:**
1. Start with `batch_size: 4` (conservative, works on most GPUs)
2. Increase gradually (8, 16, 32) until you hit OOM
3. Use gradient accumulation to simulate larger batches without OOM

---

## 💡 Best Practices

✅ **Always use FP16** for training  
✅ **Enable KV caching** for generation  
✅ **Use Flash Attention** if available  
✅ **Gradient accumulation** for large batches  
✅ **Streaming datasets enabled** by default (no action needed)  
✅ **Direct file iteration** - no cache files needed  
✅ **Efficient tokenizer training** - plain text passed directly to SentencePiece  
✅ **Resumable preprocessing** - safe to interrupt and resume  
✅ **Monitor GPU memory** with `nvidia-smi`  
✅ **Monitor RAM usage** - should be much lower than VRAM now  
✅ **Reduce num_workers** if RAM is limited

---

[Continue to Chapter 37: Debugging →](37-debugging-troubleshooting.md)

---
