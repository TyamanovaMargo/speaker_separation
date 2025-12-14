# MossFormer2 16kHz + TensorRT Speaker Separation

Simple, fast speaker separation using MossFormer2 16kHz model with optional TensorRT optimization for 2-3x speedup.

## Quick Start

### 1. Install

```bash
chmod +x install.sh
bash install.sh
```

### 2. Activate Environment

```bash
source venv/bin/activate
```

### 3. Run Separation

```bash
# Basic usage (with TensorRT)
python separate.py --input your_audio.wav --output results/

# Without TensorRT (slower but compatible)
python separate.py --input your_audio.wav --output results/ --no-tensorrt

# Adjust chunk size for memory
python separate.py --input your_audio.wav --output results/ --chunk-size 20
```

### 4. Check Results

```bash
ls results/
# speaker1.wav  speaker2.wav
```

## What You Get

- ✅ **MossFormer2 16kHz** - Better quality than 8kHz version
- ✅ **TensorRT** - 2-3x faster processing (optional)
- ✅ **Simple API** - One command to separate
- ✅ **Smart Chunking** - Handles long audio files
- ✅ **Auto-resampling** - Converts any sample rate to 16kHz

## Performance

| Configuration | Speed | Quality |
|--------------|-------|---------|
| 16kHz PyTorch | 1.0x | Good |
| 16kHz + TensorRT | 2.5x | Good |

*Note: First run with TensorRT takes 2-3 minutes for compilation, then it's cached*

## Requirements

- Python 3.10-3.12
- CUDA GPU (recommended, CPU works but slower)
- 4GB+ VRAM for GPU

## Troubleshooting

**Out of memory?**
```bash
python separate.py --input audio.wav --output results/ --chunk-size 15
```

**No GPU?**
```bash
# CPU mode (slower)
python separate.py --input audio.wav --output results/ --no-tensorrt
```

**TensorRT issues?**
```bash
# Disable TensorRT, use standard PyTorch
python separate.py --input audio.wav --output results/ --no-tensorrt
```

## Parameters

```
--input PATH          Input audio file (required)
--output PATH         Output directory (default: output/)
--no-tensorrt         Disable TensorRT optimization
--chunk-size N        Chunk size in seconds (default: 30)
```

## Output Quality

After processing, you'll see quality metrics:
- **Excellent**: correlation < 0.3
- **Good**: correlation 0.3-0.6
- **Moderate**: correlation > 0.6

Lower correlation = better separation.

## Examples

**Process podcast:**
```bash
python separate.py --input podcast.mp3 --output podcast_separated/
```

**Process long interview:**
```bash
python separate.py --input interview.wav --output interview_output/ --chunk-size 45
```

**Batch processing:**
```bash
for file in audio/*.wav; do
    python separate.py --input "$file" --output "results/$(basename $file .wav)/"
done
```

That's it! Simple and effective speaker separation.
