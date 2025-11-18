# Tokenizer Files Explanation

## Our Tokenizer: SentencePiece

μOmni uses **SentencePiece** tokenizer, which uses a single binary model file.

### Required Files

✅ **`tokenizer.model`** - The SentencePiece model file (contains everything)

✅ **`tokenizer_config.json`** - Configuration telling Hugging Face to use SentencePiece

### NOT Needed (for our tokenizer)

❌ **`vocab.json`** - Used by GPT-2 style BPE tokenizers (not SentencePiece)

❌ **`merges.txt`** - Used by GPT-2 style BPE tokenizers (not SentencePiece)

## Why the Difference?

### GPT-2 Style BPE Tokenizers
- Use two files: `vocab.json` (vocabulary mapping) + `merges.txt` (merge rules)
- Example: GPT-2, GPT-3, some Hugging Face models

### SentencePiece Tokenizers
- Use one file: `tokenizer.model` (contains everything)
- Example: T5, mT5, LLaMA, μOmni

## Our Implementation

```python
# From omni/tokenizer.py
class BPETokenizer:
    def __init__(self, model_path):
        self.sp = spm.SentencePieceProcessor(model_file=model_path)
        # That's it! Single file contains everything
```

The `tokenizer.model` file is self-contained and includes:
- Vocabulary
- Merge rules
- Special tokens
- All tokenizer parameters

## Hugging Face Compatibility

Our `tokenizer_config.json` correctly specifies:

```json
{
  "tokenizer_class": "SentencePieceTokenizer",
  "model_file": "tokenizer.model"
}
```

This tells Hugging Face to use SentencePiece with our single model file, so no additional files are needed.

