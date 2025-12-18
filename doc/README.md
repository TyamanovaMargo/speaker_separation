# Speaker Separation Pipeline with ClearerVoice-Studio

**One-click batch processing for 2-speaker separation using ClearerVoice-Studio's TensorRT-improved model**

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
bash install.sh

input/
├── audio1.wav
├── audio2.wav
└── ...


speaker_separation/
│
├── run.sh                          ⚡ ONE-CLICK BATCH RUNNER
├── install.sh                      Installation script
│
├── input/                          📁 Place your .wav/.WAV files here
│
├── results/                        📊 Output folder (auto-created)
│   └── <audio_name>/
│       ├── speaker1.wav
│       └── speaker2.wav
│
├── ClearerVoice-Studio/            🎙️ Main separation engine
│   └── clearvoice/
│       ├── separate_tensorrt_improved.py  ⭐ Core separation script
│       ├── requirements.txt
│       └── requirements_tensorrt.txt
│
├── config/
│   ├── config.yaml
│   └── requirements.txt
│
├── venv_moss/                      Python virtual environment
│
└── doc/
    └── README.md

bash run.sh
or
bash run.sh custom_output/


results/
├── call_center_audio_1/
│   ├── speaker1.wav     ← Speaker 1
│   └── speaker2.wav     ← Speaker 2
├── call_center_audio_2/
│   ├── speaker1.wav
│   └── speaker2.wav
└── ...


# Play all results
play results/*/*.wav

# Play specific audio
play results/audio_name/speaker1.wav

🛠️ Advanced Options
Virtual Environment
The script automatically uses:

venv_moss (if exists) — preferred
venv (if exists)
Creates new venv via install.sh (if neither exists)


Requirements
All dependencies are installed by install.sh:

Core: numpy, scipy, soundfile
ClearerVoice-Studio: From requirements.txt
TensorRT: From ClearerVoice-Studio/clearvoice/requirements_tensorrt.txt


🎙️ About ClearerVoice-Studio
This project uses ClearerVoice-Studio — an AI-powered speech processing toolkit by Alibaba that provides capabilities for:

Speech Enhancement (denoising)
Speech Separation (2-speaker, via MossFormer & TensorRT optimization)
Speech Super-Resolution (16kHz → 48kHz bandwidth extension)
Target Speaker Extraction (audio-visual and EEG-based)
Training & Fine-tuning support for all tasks
Latest Updates:

[2025.6] NumPy array interface for flexible model integration
[2025.5] Enhanced SpeechScore with NISQA and DISTILL_MOS metrics
[2025.4] pip install support: pip install clearvoice
[2025.4] Speech super-resolution training scripts
[2025.1] Multi-format audio support (WAV, MP3, AAC, FLAC, etc.)
[2024.11] 3M+ uses of FRCRN denoiser, 2.5M+ uses of MossFormer separator

