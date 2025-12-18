# Modular Speaker Separation Pipeline

**Complete modular pipeline with MossFormer2 separation and ClearerVoice-Studio integration**

Run each preprocessing step independently, use MossFormer2 for separation, or leverage the new ClearerVoice-Studio models for even clearer results.

Perfect for experimenting, debugging, and getting high-quality 2-speaker separation.

---

## 🎯 What You Get

✅ **6 independent preprocessing steps** (run each separately)  
✅ **MossFormer2 separation** (state-of-the-art 2-speaker separation)  
✅ **ClearerVoice-Studio integration** (advanced enhancement and separation, 16kHz/48kHz)  
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
├── ClearerVoice-Studio/           ClearerVoice models and scripts
│   └── clearvoice/
│       ├── separate_clearvoice.py     # 16kHz separation/enhancement
│       └── separate_clearvoice_48k.py # 48kHz full pipeline
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

---

### 2. Run the Complete Modular Pipeline (Preprocess + Separate)

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

---

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

## 🗣️ Using ClearerVoice-Studio for Even Clearer Results

### 1. 16kHz Separation/Enhancement

```bash
cd ClearerVoice-Studio/clearvoice
# (optional) pip install -r ../requirements.txt

# Basic separation (2 speakers, 16kHz)
python separate_clearvoice.py --input /path/to/audio.wav --output results/

# Enhance first, then separate (for noisy audio)
python separate_clearvoice.py --input /path/to/audio.wav --output results/ --enhance-first

# Batch process a folder
python separate_clearvoice.py --input-dir /path/to/folder --output results/
```
cd ~/Desktop/speaker_separation/ClearerVoice-Studio/clearvoice

# Basic (TensorRT, fastest)
python separate_tensorrt_full.py -i audio.wav -o results/

# Batch process folder
python separate_tensorrt_full.py --input-dir "../../input/samples_of_low_quality_after_light_diarization/" -o /home/margo/Desktop/speaker_separation/results/marlibs_trt

# With enhancement for noisy audio
python separate_tensorrt_full.py --input-dir folder/ -o results/ --enhance-first

# Custom chunk size for very long audio
python separate_tensorrt_full.py -i long_audio.wav -o results/ --chunk-sec 60 --overlap-sec 3



## New Features

### 1. Chunked Processing
```
Audio: [====|====|====|====]
        chunk1  chunk2  chunk3
              ↘↙      ↘↙
           overlap  overlap
```
- Processes long audio in 30-second chunks
- 2-second overlap prevents boundary artifacts

### 2. Crossfade Merging
```
Chunk 1: ━━━━━━━━━╲
Chunk 2:          ╱━━━━━━━━━
Result:  ━━━━━━━━━━━━━━━━━━━
              ↑
         smooth blend
```

### 3. Quality Metrics
```
📊 Quality:
   • Speaker correlation: 0.142 (lower = better)
   • Energy ratio: 0.87
   • Rating: Excellent
```

### 4. Proper Output Naming
```
input: meeting_audio.mp3
output:
  └── meeting_audio/
      ├── meeting_audio_speaker1.wav
      └── meeting_audio_speaker2.wav



**Other modes:**
- Enhance only: `--mode enhance`
- Separate only: `--mode separate`
- Super-res only: `--mode super-res`

---

## 📝 Tips

- All scripts support `--help` for usage details.
- For ClearerVoice-Studio, see the README in `ClearerVoice-Studio/clearvoice/`.
- You can mix and match: preprocess with modular pipeline, then use ClearerVoice for separation.

---

## 🔧 Customization

- Edit parameters in `config/config.yaml`
- See each script's `--help` for more options

---

## 📖 Documentation

- [INSTALL.md](INSTALL.md): Installation instructions
- [USAGE_GUIDE.md](USAGE_GUIDE.md): Step-by-step usage and examples
- [MOSSFORMER2_GUIDE.md](MOSSFORMER2_GUIDE.md): MossFormer2 details
- `ClearerVoice-Studio/clearvoice/README.md`: ClearerVoice usage

---

**For questions, see the documentation or