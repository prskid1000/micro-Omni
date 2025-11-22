# Auto-Calculation of `max_audio_length` for Vocoder Training

## Overview

The `max_audio_length` parameter is now **automatically calculated** from your dataset using percentile-based analysis, similar to how `max_mel_length` is calculated for other models (audio encoder, talker).

## Why This Matters

### Before (Manual Configuration)

```json
{
  "max_audio_length": 8192 // Fixed value, might not be optimal for your dataset
}
```

- ❌ Hard-coded value (8192 samples ≈ 0.51s at 16kHz)
- ❌ May waste memory with excessive padding
- ❌ May truncate too much if your audio is longer
- ❌ Not optimized for your specific dataset

### After (Auto-Calculation)

```json
{
  "max_audio_length_percentile": 95.0 // Optional, default is 95.0
  // max_audio_length is automatically calculated!
}
```

- ✅ Automatically calculated from your dataset
- ✅ Uses 95th percentile by default (configurable)
- ✅ Minimal padding while covering 95% of data
- ✅ Optimized for your specific dataset
- ✅ Rounded to nearest 256 for memory alignment

## How It Works

### 1. **Dataset Analysis**

When you start training, the script:

1. Scans your audio CSV file
2. Loads audio files and measures their lengths
3. Calculates statistics (min, max, mean, median, percentiles)
4. Recommends optimal `max_audio_length` based on percentile

### 2. **Percentile-Based Approach**

```
Example dataset:
- 90% of audios: < 7500 samples
- 95% of audios: < 8100 samples  ← 95th percentile (recommended)
- 99% of audios: < 12000 samples
- 100% of audios: < 50000 samples (outliers)

Recommended max_audio_length: 8192 (rounded up from 8100)
Result: 95% of data fits perfectly, 5% is truncated
```

### 3. **Memory Alignment**

The calculated value is rounded up to the nearest 256 for better memory alignment:

```python
max_audio_len = int(np.ceil(max_audio_len / 256) * 256)
```

## Usage

### Method 1: Automatic (Recommended)

Simply run training without specifying `max_audio_length`:

```bash
python train_vocoder.py --config configs/vocoder_tiny.json
```

Output:

```
🔍 Analyzing vocoder dataset...
  Analyzing vocoder dataset: 1000/1000 files...
  Analyzed vocoder dataset: 1000 files processed

📊 Audio Length Statistics:
  • Total files analyzed: 1000
  • Min length: 1024 samples (0.06s)
  • Max length: 48000 samples (3.00s)
  • Mean length: 7234 samples (0.45s)
  • Median length: 7168 samples (0.45s)
  • 95th percentile: 8076 samples (0.50s)
  • Recommended max_audio_length: 8192 samples (~0.51s)
  • Coverage: ~95.0% of data will fit without truncation

✓ Using auto-calculated max_audio_length: 8192
```

### Method 2: Standalone Analysis

Use the analysis script to explore different percentiles:

```bash
# Analyze with default 95th percentile
python scripts/analyze_audio_lengths.py --csv data/audio/production_tts.csv

# Try 99th percentile for less truncation
python scripts/analyze_audio_lengths.py --csv data/audio/production_tts.csv --percentile 99

# Sample only 1000 files for faster analysis
python scripts/analyze_audio_lengths.py --csv data/audio/production_tts.csv --samples 1000
```

### Method 3: Manual Override

If you want to use a specific value:

```json
{
  "max_audio_length": 16384, // Manually override
  "max_audio_length_percentile": 95.0
}
```

You'll see a warning:

```
⚠ Warning: Config max_audio_length=16384 differs from dataset audio length=8192
  Using config value: 16384
```

## Configuration Parameters

### `max_audio_length` (Optional)

- **Type**: Integer
- **Default**: Auto-calculated from dataset
- **Description**: Maximum audio length in samples. If not set, automatically calculated using percentile.
- **Example**: `8192` (512ms at 16kHz)

### `max_audio_length_percentile` (Optional)

- **Type**: Float
- **Default**: `95.0`
- **Description**: Percentile to use for auto-calculation
- **Range**: `0.0` to `100.0`
- **Example**: `95.0` means 95% of data will fit without truncation

### `dataset_sample_size` (Optional)

- **Type**: Integer
- **Default**: `None` (analyze all files)
- **Description**: Number of files to sample for analysis (for large datasets)
- **Example**: `1000` (analyze random 1000 files)

## Percentile Comparison

| Percentile | Coverage | Padding   | Truncation | Use Case                             |
| ---------- | -------- | --------- | ---------- | ------------------------------------ |
| **90%**    | 90%      | Low       | 10%        | Minimize memory, tolerate truncation |
| **95%** ⭐ | 95%      | Balanced  | 5%         | **RECOMMENDED** - Best balance       |
| **99%**    | 99%      | High      | 1%         | Preserve most data, more memory      |
| **100%**   | 100%     | Very High | 0%         | No truncation, wasteful padding      |

## Memory Impact

### Example Calculation (16kHz sample rate)

```
max_audio_length = 8192 samples
Bytes per sample = 4 (float32)
Memory per audio = 8192 * 4 = 32,768 bytes ≈ 32 KB

Batch size 16:
Total audio memory = 16 * 32 KB = 512 KB

Compare to:
- max_audio_length = 16384: 1 MB per batch
- max_audio_length = 32768: 2 MB per batch
```

**Note**: Vocoder also stores mel spectrograms, so total memory is higher.

## New Function: `analyze_vocoder_dataset()`

Added to `omni/utils.py`:

```python
def analyze_vocoder_dataset(
  csv_path,
  sr=16000,
  n_fft=1024,
  hop_length=256,
  n_mels=128,
  sample_size=None,
  audio_percentile=95.0,
):
  """
  Analyze dataset to calculate percentile-based padding limits for audio and mel frames.

  Args:
    csv_path: Path to vocoder CSV file with 'wav' column
    sr: Sample rate (default: 16000)
    n_fft: FFT size for mel spectrograms (default: 1024)
    hop_length: Hop length used for mel spectrograms (default: 256)
    n_mels: Number of mel bins (default: 128)
    sample_size: Number of samples to check (None = all)
    audio_percentile: Percentile to use (default: 95.0)

  Returns:
    tuple[int, int]: Recommended (max_audio_length, max_mel_length)
  """
```

## Integration with Training

The `train_vocoder.py` script now:

1. Automatically calls `analyze_vocoder_dataset()` before training
2. Calculates optimal `max_audio_length` **and** matching `max_mel_length` using the same mel transform as training
3. Updates the config with the calculated values
4. Passes them to `VocoderDataset`/collate for cropping + padding alignment

## Best Practices

1. **First Time**: Let it auto-calculate (remove `max_audio_length` from config)
2. **Review Statistics**: Check the analysis output to understand your dataset
3. **Adjust if Needed**: Try different percentiles if default doesn't work well
4. **Large Datasets**: Use `dataset_sample_size` to speed up analysis
5. **Production**: Once you find optimal value, you can hard-code it in config

## Benefits

✅ **Automatic Optimization**: No manual tuning required
✅ **Data-Driven**: Based on actual dataset characteristics
✅ **Memory Efficient**: Minimizes padding overhead
✅ **Flexible**: Easy to adjust percentile for different requirements
✅ **Consistent**: Same approach used for mel/text length calculations
✅ **Transparent**: Shows statistics and recommendations

## Example Workflow

```bash
# 1. Analyze dataset (optional, just to see statistics)
python scripts/analyze_audio_lengths.py --csv data/audio/production_tts.csv

# 2. Train with auto-calculation (recommended)
python train_vocoder.py --config configs/vocoder_tiny.json

# 3. If you want to try different percentile, edit config:
# "max_audio_length_percentile": 99.0  # Less truncation, more memory

# 4. Re-run training to see new calculation
python train_vocoder.py --config configs/vocoder_tiny.json
```

## Troubleshooting

### "Most of my audio is getting truncated"

- Increase percentile: `"max_audio_length_percentile": 99.0`
- Or manually set higher value: `"max_audio_length": 16384`

### "Too much padding, wasting memory"

- Decrease percentile: `"max_audio_length_percentile": 90.0`
- Or manually set lower value: `"max_audio_length": 4096`

### "Analysis takes too long"

- Use sampling: `"dataset_sample_size": 1000`
- This analyzes random 1000 files instead of all

### "I want exact control"

- Simply set `max_audio_length` in config manually
- Auto-calculation will be skipped (with a warning)

## Summary

**Before**: Manual guesswork for `max_audio_length`
**Now**: Automatic calculation optimized for your specific dataset! 🚀

Just run training and it will figure out the best value automatically.
