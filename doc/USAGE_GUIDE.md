# How to Use - Step by Step Guide

## 📥 Setup (One Time)

### 1. Navigate to project
```bash
cd speaker_separation_pipeline_modular
```

### 2. Create virtual environment
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r config/requirements.txt
```

---

## 🎯 Usage - Your Audio File

Your file: `/home/margo/Desktop/separation_voice_model/output/tafdenok.wav`

### Option 1: Quick Start (Recommended)

```bash
# Activate virtual environment
source venv/bin/activate

# Run quick start script
bash quick_start.sh
```

This will:
1. Analyze audio quality
2. Run all preprocessing steps
3. Show you the results

---

### Option 2: Run Steps Individually

#### Step 1: Analyze Audio Quality First

```bash
python scripts/preprocess/01_audio_diagnostics.py \
    --input /home/margo/Desktop/separation_voice_model/output/tafdenok.wav
```

This shows you:
- Sample rate
- SNR (signal quality)
- If audio is clipped
- If there's power line hum
- Overall quality assessment

**Look at the output and decide which steps you need!**

---

#### Step 2: Run All Steps

```bash
mkdir -p output/tafdenok

python scripts/preprocess/run_all.py \
    --input /home/margo/Desktop/separation_voice_model/output/tafdenok.wav \
    --output_dir output/tafdenok/
```

---

#### Or: Run Steps One by One

```bash
INPUT="/home/margo/Desktop/separation_voice_model/output/tafdenok.wav"
mkdir -p output

# Step 2: Resample to 16kHz
python scripts/preprocess/02_resample.py \
    --input "$INPUT" \
    --output output/step2_resampled.wav

# Step 3: Fix clipping (if detected in step 1)
python scripts/preprocess/03_declip.py \
    --input output/step2_resampled.wav \
    --output output/step3_declipped.wav

# Step 4: Remove hum (if detected in step 1)
python scripts/preprocess/04_remove_hum.py \
    --input output/step3_declipped.wav \
    --output output/step4_dehum.wav

# Step 5: Reduce noise
python scripts/preprocess/05_denoise.py \
    --input output/step4_dehum.wav \
    --output output/step5_denoised.wav \
    --strength 0.8

# Step 6: Normalize
python scripts/preprocess/06_normalize.py \
    --input output/step5_denoised.wav \
    --output output/step6_normalized.wav
```

---

### Option 3: Run Only Specific Steps

```bash
# Only run steps 1, 2, 5, and 6
python scripts/preprocess/run_all.py \
    --input /home/margo/Desktop/separation_voice_model/output/tafdenok.wav \
    --output_dir output/tafdenok/ \
    --steps 1 2 5 6
```

```bash
# Skip steps 3 and 4
python scripts/preprocess/run_all.py \
    --input /home/margo/Desktop/separation_voice_model/output/tafdenok.wav \
    --output_dir output/tafdenok/ \
    --skip 3 4
```

---

## 📊 Check Results

```bash
# List all output files
ls -lh output/tafdenok/

# View diagnostics
cat output/tafdenok/diagnostics.json

# Listen to files (Linux)
aplay output/tafdenok/step2_resampled.wav
aplay output/tafdenok/step5_denoised.wav
aplay output/tafdenok/preprocessed_final.wav
```

---

## 🔧 Adjust Parameters

### More/Less Denoising

```bash
# Light denoising (strength 0.3)
python scripts/preprocess/05_denoise.py \
    --input audio.wav \
    --output denoised_light.wav \
    --strength 0.3

# Heavy denoising (strength 0.9)
python scripts/preprocess/05_denoise.py \
    --input audio.wav \
    --output denoised_heavy.wav \
    --strength 0.9
```

### Different Normalization Level

```bash
# Normalize to -6dB instead of -3dB
python scripts/preprocess/06_normalize.py \
    --input audio.wav \
    --output normalized.wav \
    --level -6
```

---

## 💡 Common Workflows

### Workflow 1: Analyze First, Then Process

```bash
# 1. Analyze
python scripts/preprocess/01_audio_diagnostics.py --input audio.wav

# 2. Based on results, run appropriate steps
# If quality is good: skip declipping and hum removal
python scripts/preprocess/run_all.py --input audio.wav --output_dir output/ --skip 3 4

# If quality is bad: run all steps
python scripts/preprocess/run_all.py --input audio.wav --output_dir output/
```

---

### Workflow 2: Compare Different Settings

```bash
mkdir -p output/comparison

# Test different denoising strengths
python scripts/preprocess/05_denoise.py --input audio.wav --output output/comparison/denoise_0.3.wav --strength 0.3
python scripts/preprocess/05_denoise.py --input audio.wav --output output/comparison/denoise_0.5.wav --strength 0.5
python scripts/preprocess/05_denoise.py --input audio.wav --output output/comparison/denoise_0.8.wav --strength 0.8

# Listen and compare
aplay output/comparison/denoise_0.3.wav
aplay output/comparison/denoise_0.5.wav
aplay output/comparison/denoise_0.8.wav
```

---

### Workflow 3: Educational - See Each Step's Effect

```bash
mkdir -p output/steps

# Copy original
cp audio.wav output/steps/00_original.wav

# Apply each step
python scripts/preprocess/02_resample.py --input output/steps/00_original.wav --output output/steps/01_resampled.wav
python scripts/preprocess/03_declip.py --input output/steps/01_resampled.wav --output output/steps/02_declipped.wav
python scripts/preprocess/04_remove_hum.py --input output/steps/02_declipped.wav --output output/steps/03_dehum.wav
python scripts/preprocess/05_denoise.py --input output/steps/03_dehum.wav --output output/steps/04_denoised.wav
python scripts/preprocess/06_normalize.py --input output/steps/04_denoised.wav --output output/steps/05_final.wav

# Listen to each step
for file in output/steps/*.wav; do
    echo "Playing: $file"
    aplay "$file"
done
```

---

## 📖 Summary Commands

### Your Audio File - Complete Processing

```bash
# Setup (one time)
cd speaker_separation_pipeline_modular
python3 -m venv venv
source venv/bin/activate
pip install -r config/requirements.txt

# Every time you use it
source venv/bin/activate

# Quick start
bash quick_start.sh

# Or manual
python scripts/preprocess/run_all.py \
    --input /home/margo/Desktop/separation_voice_model/output/tafdenok.wav \
    --output_dir output/tafdenok/

# Check results
ls -lh output/tafdenok/
aplay output/tafdenok/preprocessed_final.wav
```

---

## ❓ Help

Each script has built-in help:

```bash
python scripts/preprocess/01_audio_diagnostics.py --help
python scripts/preprocess/05_denoise.py --help
python scripts/preprocess/run_all.py --help
```

---

**That's it! Start with Step 1 (diagnostics) to see what your audio needs, then run the appropriate steps!** 🚀
