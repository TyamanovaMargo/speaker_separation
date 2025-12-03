# Modular Speaker Separation Pipeline

**Complete modular pipeline with MossFormer2 separation**

Run each preprocessing step independently + MossFormer2 speaker separation.

Perfect for experimenting, debugging, and getting high-quality 2-speaker separation.

---

## 🎯 What You Get

✅ **6 independent preprocessing steps** (run each separately)
✅ **MossFormer2 separation** (state-of-the-art 2-speaker separation)
✅ **Complete pipeline** (preprocess + separate in one command)
✅ **Full control** (customize every step)

---

## 📁 Project Structure

```
speaker_separation_pipeline_modular/
│
├── complete_pipeline.py           ⚡ FULL PIPELINE (preprocess + separate)
├── install.sh                     Installation script
│
├── 📖 Documentation
│   ├── README.md                  This file
│   ├── INSTALL.md                 Installation guide
│   ├── USAGE_GUIDE.md             Preprocessing guide
│   └── MOSSFORMER2_GUIDE.md       MossFormer2 guide
│
├── config/
│   ├── config.yaml                Settings
│   └── requirements.txt           Dependencies
│
└── scripts/
    ├── preprocess/                6 preprocessing steps
    │   ├── 01_audio_diagnostics.py
    │   ├── 02_resample.py
    │   ├── 03_declip.py
    │   ├── 04_remove_hum.py
    │   ├── 05_denoise.py
    │   ├── 06_normalize.py
    │   └── run_all.py
    │
    └── separation/
        └── mossformer2_separate.py  MossFormer2 separation
```

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
cd speaker_separation_pipeline_modular
bash install.sh
source venv/bin/activate
```

### 2. Complete Pipeline (Preprocess + Separate)

```bash
python complete_pipeline.py \
    --input /home/margo/Desktop/separation_voice_model/output/tafdenok.wav \
    --output_dir results/tafdenok/
```

**Output:**
```
results/tafdenok/
├── preprocessed/              Cleaned audio
│   └── preprocessed_final.wav
└── separated/                 ✨ Final result
    ├── speaker1.wav
    └── speaker2.wav
```

### 3. Or Run Steps Separately

```bash
# Step 1: Analyze audio quality
python scripts/preprocess/01_audio_diagnostics.py --input audio.wav

# Step 2: Preprocess
python scripts/preprocess/run_all.py --input audio.wav --output_dir preprocessed/

# Step 3: Separate with MossFormer2
python scripts/separation/mossformer2_separate.py \
    --input preprocessed/preprocessed_final.wav \
    --output_dir separated/
```

---

## 📋 Preprocessing Steps Explained

### **Step 1: Audio Diagnostics** 
**What it does:** Analyzes audio quality and detects issues
**Output:** JSON file with metrics

```bash
python scripts/preprocess/01_audio_diagnostics.py --input audio.wav
```

**Metrics analyzed:**
- Sample rate
- SNR (Signal-to-Noise Ratio)
- Clipping detection
- Power line hum (50/60/100/120 Hz)
- Dynamic range
- Overall quality assessment

**Output example:**
```
AUDIO DIAGNOSTICS
================================================================================

Loading: audio.wav
  Sample rate: 44100 Hz
  Duration: 45.23 seconds

1. Peak Amplitude: 0.9823
   ✓ No clipping

2. Signal-to-Noise Ratio: 12.34 dB
   ⚙️  MODERATE SNR - Acceptable quality

3. Hum/Buzz Detection:
   ⚠️  HUM DETECTED at: [60, 120] Hz

4. Dynamic Range: 18.45 dB

OVERALL ASSESSMENT
Quality: MODERATE ⚙️
Recommendation: Use 'bad' quality mode (default)
Issues found: Power line hum
```

---

### **Step 2: Resample**
**What it does:** Converts audio to target sample rate (default 16kHz)

```bash
python scripts/preprocess/02_resample.py \
    --input audio.wav \
    --output resampled.wav \
    --sr 16000
```

**Why needed:** Most speech models work best at 16kHz

---

### **Step 3: Declipping**
**What it does:** Repairs clipped/distorted audio using interpolation

```bash
python scripts/preprocess/03_declip.py \
    --input audio.wav \
    --output declipped.wav \
    --threshold 0.95
```

**When to use:** If Step 1 detects clipping

---

### **Step 4: Hum Removal**
**What it does:** Removes power line interference (50Hz, 60Hz, harmonics)

```bash
python scripts/preprocess/04_remove_hum.py \
    --input audio.wav \
    --output dehum.wav \
    --freq 50 60 100 120
```

**When to use:** If Step 1 detects hum

---

### **Step 5: Noise Reduction**
**What it does:** Reduces background noise using spectral subtraction

```bash
python scripts/preprocess/05_denoise.py \
    --input audio.wav \
    --output denoised.wav \
    --strength 0.8
```

**Parameters:**
- `--strength`: 0.0 to 1.0 (higher = more aggressive)

---

### **Step 6: Normalize**
**What it does:** Normalizes audio amplitude to prevent clipping

```bash
python scripts/preprocess/06_normalize.py \
    --input audio.wav \
    --output normalized.wav \
    --level -3
```

**Why needed:** Ensures consistent levels for downstream processing

---

## 🎯 Usage Scenarios

### **Scenario 1: Test Each Step**

Run each step individually to see the effect:

```bash
mkdir -p output

# Original audio
cp input.wav output/00_original.wav

# Step by step
python scripts/preprocess/01_audio_diagnostics.py --input output/00_original.wav
python scripts/preprocess/02_resample.py --input output/00_original.wav --output output/01_resampled.wav
python scripts/preprocess/03_declip.py --input output/01_resampled.wav --output output/02_declipped.wav
python scripts/preprocess/04_remove_hum.py --input output/02_declipped.wav --output output/03_dehum.wav
python scripts/preprocess/05_denoise.py --input output/03_dehum.wav --output output/04_denoised.wav
python scripts/preprocess/06_normalize.py --input output/04_denoised.wav --output output/05_normalized.wav

# Compare results by listening to each file
```

---

### **Scenario 2: Run Only Specific Steps**

```bash
# Only run diagnostics and denoising
python scripts/preprocess/run_all.py \
    --input audio.wav \
    --output_dir output/ \
    --steps 1 5
```

---

### **Scenario 3: Skip Certain Steps**

```bash
# Run all except declipping and hum removal
python scripts/preprocess/run_all.py \
    --input audio.wav \
    --output_dir output/ \
    --skip 3 4
```

---

### **Scenario 4: Experiment with Parameters**

```bash
# Try different denoising strengths
python scripts/preprocess/05_denoise.py --input audio.wav --output denoised_light.wav --strength 0.3
python scripts/preprocess/05_denoise.py --input audio.wav --output denoised_medium.wav --strength 0.6
python scripts/preprocess/05_denoise.py --input audio.wav --output denoised_heavy.wav --strength 0.9

# Compare results
```

---

## 💡 For Your File

```bash
INPUT_FILE="/home/margo/Desktop/separation_voice_model/output/tafdenok.wav"

# Step 1: Analyze quality
python scripts/preprocess/01_audio_diagnostics.py --input "$INPUT_FILE"

# Step 2: Run all preprocessing
python scripts/preprocess/run_all.py \
    --input "$INPUT_FILE" \
    --output_dir output/tafdenok_preprocessed/

# Check results
ls -lh output/tafdenok_preprocessed/
```

---

## 🔧 Customization

### Edit Parameters

All default parameters are in `config/config.yaml`:

```yaml
preprocessing:
  clipping_threshold: 0.95
  snr_poor_threshold: 10.0
  
  denoise:
    strength: 0.8
  
  hum_removal:
    frequencies: [50, 60, 100, 120]
```

### Create Custom Workflow

```bash
# Example: Only denoise and normalize
python scripts/preprocess/05_denoise.py --input audio.wav --output temp.wav --strength 0.7
python scripts/preprocess/06_normalize.py --input temp.wav --output final.wav --level -3
```

---

## 📊 Output Files

When running `run_all.py`:

```
output_dir/
├── diagnostics.json           Step 1 output (metrics)
├── step2_resampled.wav       Step 2 output
├── step3_declipped.wav       Step 3 output
├── step4_dehum.wav           Step 4 output
├── step5_denoised.wav        Step 5 output
├── step6_normalized.wav      Step 6 output
└── preprocessed_final.wav    Final result (copy of last step)
```

---

## 🎓 Benefits of Modular Approach

✅ **Test each step independently**
✅ **See the effect of each process**
✅ **Skip unnecessary steps**
✅ **Experiment with parameters**
✅ **Debug issues easily**
✅ **Customize your pipeline**
✅ **Educational - understand what each step does**

---

## 🔍 Troubleshooting

### "noisereduce not installed"
```bash
pip install noisereduce
```

### "Module not found"
```bash
# Make sure virtual environment is activated
source venv/bin/activate

# Reinstall dependencies
pip install -r config/requirements.txt
```

### Want to see intermediate results
```bash
# Each step saves its output, just listen to the files!
aplay output/step3_declipped.wav
aplay output/step5_denoised.wav
```

---

## 🚀 Next Steps

After preprocessing, you can:
1. Use the final audio for separation (coming soon)
2. Analyze the improvements using Step 1 diagnostics
3. Experiment with different parameter combinations

---

## 📖 Complete Example

```bash
#!/bin/bash

# Your audio file
INPUT="/home/margo/Desktop/separation_voice_model/output/tafdenok.wav"
OUTPUT_DIR="output/tafdenok"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Step 1: Check quality
echo "Step 1: Analyzing audio quality..."
python scripts/preprocess/01_audio_diagnostics.py --input "$INPUT"

# Step 2-6: Process
echo -e "\nStep 2-6: Running preprocessing..."
python scripts/preprocess/run_all.py \
    --input "$INPUT" \
    --output_dir "$OUTPUT_DIR"

# Done
echo -e "\n✓ Complete! Check results in: $OUTPUT_DIR/"
ls -lh "$OUTPUT_DIR"
```

---

**That's it! Each preprocessing step is now independent and can be run separately.**

Start with Step 1 to analyze your audio, then run the steps you need! 🎉
