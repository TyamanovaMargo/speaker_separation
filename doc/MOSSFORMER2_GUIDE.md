# MossFormer2 Separation - Complete Guide

## 🎯 What is MossFormer2?

**MossFormer2** is a state-of-the-art speech separation model from Alibaba DAMO Academy. It's specifically designed to separate overlapping speech into individual speakers.

**Key Features:**
- ✅ Separates 2 speakers
- ✅ Works on 16kHz audio
- ✅ ~100MB model (downloaded once)
- ✅ Very good quality separation
- ✅ Handles overlapping speech well

---

## 📦 Installation

### 1. Install ModelScope

```bash
pip install modelscope
```

**If you have network issues:**
```bash
pip install modelscope -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 2. Install Other Dependencies

```bash
pip install numpy librosa soundfile
```

---

## 🚀 Usage

### Option 1: MossFormer2 Only (Quick)

```bash
python scripts/separation/mossformer2_separate.py \
    --input /home/margo/Desktop/separation_voice_model/output/tafdenok.wav \
    --output_dir output/separated/
```

**Output:**
```
output/separated/
├── speaker1.wav
└── speaker2.wav
```

---

### Option 2: Complete Pipeline (Recommended)

Preprocessing + MossFormer2 for best results:

```bash
python complete_pipeline.py \
    --input /home/margo/Desktop/separation_voice_model/output/tafdenok.wav \
    --output_dir results/tafdenok/
```

**Output:**
```
results/tafdenok/
├── preprocessed/              (cleaned audio)
│   ├── diagnostics.json
│   ├── step2_resampled.wav
│   ├── step5_denoised.wav
│   └── preprocessed_final.wav
└── separated/                 (final result)
    ├── speaker1.wav
    └── speaker2.wav
```

---

## 📋 Step-by-Step Workflow

### Step 1: Preprocess Your Audio

```bash
# First, clean the audio
python scripts/preprocess/run_all.py \
    --input /home/margo/Desktop/separation_voice_model/output/tafdenok.wav \
    --output_dir preprocessed/
```

### Step 2: Separate with MossFormer2

```bash
# Then separate speakers
python scripts/separation/mossformer2_separate.py \
    --input preprocessed/preprocessed_final.wav \
    --output_dir separated/
```

### Step 3: Check Results

```bash
# List files
ls -lh separated/

# Listen (Linux)
aplay separated/speaker1.wav
aplay separated/speaker2.wav
```

---

## 🎓 Understanding MossFormer2 Output

### What You Get:

**speaker1.wav** - First separated speaker
**speaker2.wav** - Second separated speaker

### Quality Metrics:

The script shows:
- **Cross-correlation:** Lower is better (< 0.3 = good separation)
- **Energy ratio:** Should be > 0.3 (both speakers have content)

Example output:
```
SEPARATION QUALITY METRICS
================================================================================

Cross-correlation: 0.2145
  ✓ Good separation (low correlation)

Energy ratio: 0.7823
  ✓ Both speakers have significant presence
```

---

## 💡 Tips for Best Results

### 1. Preprocess First! (Important)

```bash
# Bad: Separate noisy audio directly
python scripts/separation/mossformer2_separate.py --input noisy.wav --output_dir out/

# Good: Preprocess first, then separate
python scripts/preprocess/run_all.py --input noisy.wav --output_dir clean/
python scripts/separation/mossformer2_separate.py --input clean/preprocessed_final.wav --output_dir out/
```

**Why?** MossFormer2 works ~30% better on cleaned audio!

### 2. Use 16kHz Audio

MossFormer2 requires 16kHz. The script auto-resamples, but it's better to preprocess first:

```bash
python scripts/preprocess/02_resample.py --input audio.wav --output audio_16k.wav --sr 16000
```

### 3. Remove Noise First

Heavy background noise confuses the model:

```bash
python scripts/preprocess/05_denoise.py --input audio.wav --output denoised.wav --strength 0.8
```

### 4. Check Audio Quality First

```bash
python scripts/preprocess/01_audio_diagnostics.py --input audio.wav
```

If SNR < 10 dB, definitely preprocess before separating!

---

## 🔧 Common Issues & Solutions

### Issue 1: "ModelScope not installed"

**Solution:**
```bash
pip install modelscope
```

### Issue 2: Model download fails

**Solution 1** - Use Chinese mirror:
```bash
pip install modelscope -i https://pypi.tuna.tsinghua.edu.cn/simple
```

**Solution 2** - Set cache directory:
```bash
export MODELSCOPE_CACHE=./modelscope_cache
python scripts/separation/mossformer2_separate.py --input audio.wav --output_dir out/
```

### Issue 3: Poor separation quality

**Solutions:**
1. Preprocess the audio first
2. Check if audio has > 2 speakers (MossFormer2 only handles 2)
3. Try increasing denoising strength: `--denoise_strength 0.9`

### Issue 4: Out of memory

**Solution:**
```bash
# Process in chunks (for very long audio)
# Split audio into 5-minute segments first
ffmpeg -i long_audio.wav -f segment -segment_time 300 -c copy chunk%03d.wav

# Process each chunk
for f in chunk*.wav; do
    python scripts/separation/mossformer2_separate.py --input "$f" --output_dir "separated_$f/"
done
```

### Issue 5: "CUDA out of memory"

**Solution:** MossFormer2 runs on CPU by default (no GPU needed), but if you get this error:
```bash
export CUDA_VISIBLE_DEVICES=""  # Force CPU mode
```

---

## 📊 Expected Performance

### Processing Time:
- **1 minute audio:** ~30 seconds - 2 minutes
- **5 minute audio:** ~3-10 minutes
- **First run:** +2 minutes (model download)

### Quality:
- **Good audio (SNR > 20 dB):** Excellent separation
- **Moderate audio (SNR 10-20 dB):** Good separation
- **Poor audio (SNR < 10 dB):** Fair (preprocess first!)

---

## 🎯 Complete Example for Your File

```bash
#!/bin/bash

# Your audio file
INPUT="/home/margo/Desktop/separation_voice_model/output/tafdenok.wav"
OUTPUT="results/tafdenok"

# Activate virtual environment
source venv/bin/activate

# Option 1: Complete pipeline (recommended)
python complete_pipeline.py --input "$INPUT" --output_dir "$OUTPUT"

# Option 2: Manual steps
# Step 1: Analyze
python scripts/preprocess/01_audio_diagnostics.py --input "$INPUT"

# Step 2: Preprocess
python scripts/preprocess/run_all.py --input "$INPUT" --output_dir "$OUTPUT/preprocessed"

# Step 3: Separate
python scripts/separation/mossformer2_separate.py \
    --input "$OUTPUT/preprocessed/preprocessed_final.wav" \
    --output_dir "$OUTPUT/separated"

# Done!
echo "✓ Separated speakers:"
ls -lh "$OUTPUT/separated/"
```

---

## 📚 Advanced Usage

### Compare Different Preprocessing

```bash
# Test different denoising strengths
python complete_pipeline.py --input audio.wav --output_dir results/light/ --denoise_strength 0.3
python complete_pipeline.py --input audio.wav --output_dir results/medium/ --denoise_strength 0.6
python complete_pipeline.py --input audio.wav --output_dir results/heavy/ --denoise_strength 0.9

# Compare results
aplay results/light/separated/speaker1.wav
aplay results/medium/separated/speaker1.wav
aplay results/heavy/separated/speaker1.wav
```

### Batch Processing

```bash
#!/bin/bash
# Process multiple files

for audio in *.wav; do
    echo "Processing: $audio"
    python complete_pipeline.py \
        --input "$audio" \
        --output_dir "results/${audio%.wav}/"
done
```

---

## 🎓 Understanding the Output

### Speaker Assignment:

MossFormer2 assigns speakers based on energy:
- **speaker1.wav** = Louder/more active speaker
- **speaker2.wav** = Quieter/less active speaker

### If Speakers Are Swapped:

Just rename the files:
```bash
mv speaker1.wav temp.wav
mv speaker2.wav speaker1.wav
mv temp.wav speaker2.wav
```

---

## 📖 Summary Commands

```bash
# Complete pipeline (easiest)
python complete_pipeline.py \
    --input /home/margo/Desktop/separation_voice_model/output/tafdenok.wav \
    --output_dir results/tafdenok/

# MossFormer2 only (if already preprocessed)
python scripts/separation/mossformer2_separate.py \
    --input preprocessed.wav \
    --output_dir separated/

# Manual control (all steps)
python scripts/preprocess/run_all.py --input audio.wav --output_dir preprocessed/
python scripts/separation/mossformer2_separate.py --input preprocessed/preprocessed_final.wav --output_dir separated/
```

---

**That's it! MossFormer2 will separate your audio into 2 clean speaker tracks.** 🎉

For best results, always preprocess your audio first!
