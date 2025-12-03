# Installation Guide

## Quick Install (Recommended)

```bash
cd speaker_separation_pipeline_modular
bash install.sh
```

This will:
1. Create virtual environment
2. Install all required packages
3. Test the installation

---

## Manual Install

If the automatic installer doesn't work, follow these steps:

### 1. Create Virtual Environment

```bash
cd speaker_separation_pipeline_modular
python3 -m venv venv
source venv/bin/activate
```

### 2. Upgrade pip

```bash
pip install --upgrade pip
```

### 3. Install Core Dependencies

```bash
pip install numpy scipy librosa soundfile pyyaml tqdm noisereduce
```

### 4. Test Installation

```bash
python3 -c "import numpy, scipy, librosa, soundfile, noisereduce; print('✓ All packages installed!')"
```

---

## Troubleshooting

### Issue: "df>=0.5.0 not found"

**Solution:** This is expected! The `df` package is optional and not needed for basic preprocessing.

### Issue: Python version incompatibility

**Your Python version:** Check with `python3 --version`

**Required:** Python 3.8 or higher (but not 3.13+)

**Solution:** Use Python 3.9, 3.10, 3.11, or 3.12

### Issue: "numpy.distutils is deprecated"

**Solution:** This is just a warning, ignore it. Everything still works.

### Issue: "soundfile needs libsndfile"

**Solution on Ubuntu/Debian:**
```bash
sudo apt-get install libsndfile1
```

**Solution on macOS:**
```bash
brew install libsndfile
```

### Issue: Virtual environment activation fails

**On Linux/Mac:**
```bash
source venv/bin/activate
```

**On Windows (PowerShell):**
```powershell
venv\Scripts\Activate.ps1
```

**On Windows (CMD):**
```cmd
venv\Scripts\activate.bat
```

---

## What Gets Installed

### Required (Always):
- `numpy` - Numerical computing
- `scipy` - Scientific computing
- `librosa` - Audio analysis
- `soundfile` - Audio I/O
- `pyyaml` - Config file reading
- `tqdm` - Progress bars
- `noisereduce` - Noise reduction

### Optional (Not needed for basic preprocessing):
- `torch` - PyTorch (for future ML features)
- `speechbrain` - Speech processing
- `transformers` - For transcription
- `pyannote` - For diarization
- `whisper` - For transcription

---

## Verify Installation

After installation, test it:

```bash
source venv/bin/activate

# Test with diagnostics
python scripts/preprocess/01_audio_diagnostics.py --input test.wav
```

If you see the diagnostics output, everything is working!

---

## Quick Start After Installation

```bash
# 1. Activate environment
source venv/bin/activate

# 2. Test your audio
python scripts/preprocess/01_audio_diagnostics.py \
    --input /home/margo/Desktop/separation_voice_model/output/tafdenok.wav

# 3. Process it
python scripts/preprocess/run_all.py \
    --input /home/margo/Desktop/separation_voice_model/output/tafdenok.wav \
    --output_dir output/tafdenok/
```

---

## Minimal Install (Just What You Need)

If you only want to test specific steps:

```bash
# For diagnostics, resample, normalize only:
pip install numpy scipy librosa soundfile

# Add this for noise reduction:
pip install noisereduce

# That's it!
```

---

## Uninstall

To remove everything:

```bash
# Deactivate virtual environment
deactivate

# Remove it
rm -rf venv/
```

Then reinstall from scratch if needed.

---

## System Requirements

- **OS:** Linux, macOS, or Windows
- **Python:** 3.8, 3.9, 3.10, 3.11, or 3.12
- **Disk:** ~500 MB for dependencies
- **RAM:** 2 GB minimum
- **CPU:** Any modern CPU (GPU not required)

---

## After Installation

Read these files:
- `README.md` - Complete documentation
- `USAGE_GUIDE.md` - How to use each step
- `quick_start.sh` - Ready-to-run example

---

**Installation should take 2-5 minutes depending on your internet speed.**
