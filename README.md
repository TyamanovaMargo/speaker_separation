# Speaker Separation Pipeline with ClearerVoice-Studio

**One-click batch processing for 2-speaker separation using ClearerVoice-Studio's TensorRT-improved model**

---

## 🚀 Quick Start

### 1. Place Audio Files

```
input/
├── audio1.wav
├── audio2.wav
└── ...
```

### 2. Build and Run with Docker (Recommended)

#### Requirements
- **NVIDIA GPU** with drivers and [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
- Docker + docker-compose
- (Optional) Docker Hub login for pulling NVIDIA base images

#### Build the container:
```bash
docker-compose build
```

#### Batch process all files:
```bash
docker-compose run separation
```

#### Process a single file:
```bash
docker-compose run separation python separate_tensorrt_improved.py \
    --input /app/input/audio1.wav \
    --output /app/results/
```

---

## 📁 Project Structure

```
speaker_separation/
│
├── run.sh                              # ⚡ One-click batch runner
├── install.sh                          # Installation script
├── Dockerfile                          # Docker containerization
├── docker-compose.yml                  # Docker Compose config
│
├── input/                              # 📁 Place your .wav/.WAV files here
│
├── output_batch/                       # 📊 Output folder (auto-created)
│   └── <audio_name>/
│       ├── speaker1.wav
│       └── speaker2.wav
│
├── ClearerVoice-Studio/                # 🎙️ Main separation engine
│   └── clearvoice/
│       ├── separate_tensorrt_improved.py   # ⭐ Core separation script
│       ├── requirements.txt
│       └── requirements_tensorrt.txt
│
├── config/
│   └── config.yaml
│
├── venv_moss/                          # Python virtual environment
│
└── doc/
    └── README.md
```

---

## 📊 Output Structure

```
output_batch/
├── call_center_audio_1/
│   ├── speaker1.wav                    # ← Speaker 1
│   └── speaker2.wav                    # ← Speaker 2
├── call_center_audio_2/
│   ├── speaker1.wav
│   └── speaker2.wav
└── ...
```

### Playing Results

```bash
# Play all results
play output_batch/*/*.wav

# Play specific audio
play output_batch/audio_name/speaker1.wav
```

---

## 🐳 Docker Usage

### Build and Run

```bash
# Build the image
docker-compose build

# Run batch processing (all files in input/)
docker-compose run separation

# Run on a single file
docker-compose run separation python separate_tensorrt_improved.py \
    --input /app/input/audio1.wav \
    --output /app/results/
```

### Volume Mounts

| Host Path         | Container Path         | Purpose                |
|-------------------|-----------------------|------------------------|
| `./input`         | `/app/input`          | Input audio files      |
| `./output_batch`  | `/app/results`        | Output separated files |
| `./checkpoints`   | `/app/checkpoints`    | Model checkpoints      |
| `./config`        | `/app/config`         | Config files           |
| `./models`        | `/app/.cache`         | Cached models          |

### GPU Requirements
- NVIDIA GPU and drivers
- NVIDIA Container Toolkit (for `runtime: nvidia`)

### Troubleshooting
- If you get errors about missing `nvidia-smi` or CUDA, check your driver and toolkit installation.
- The container must be run with GPU access enabled (see docker-compose.yml).

---

## 🛠️ Advanced Options

### Virtual Environment

The script automatically uses:

1. `venv_moss` (if exists) — preferred
2. `venv` (if exists)
3. Creates new venv via `install.sh` (if neither exists)

### Command Line Options

```
--input, -i       Input audio file
--input-dir       Input directory for batch processing
--output, -o      Output directory (default: output/)
--opt             Optimization level:
                    0 = Base PyTorch
                    1 = FP16 (half precision)
                    2 = torch.compile
                    3 = TensorRT (default, fastest)
--chunk-sec       Chunk size in seconds (default: 30)
--overlap-sec     Overlap between chunks (default: 2)
--enhance-first   Apply enhancement before separation
--output-sr       Output sample rate (default: 16000)
```

### Requirements

All dependencies are installed by `install.sh`:

- **Core:** numpy, scipy, soundfile, librosa
- **ClearerVoice-Studio:** From requirements.txt
- **TensorRT:** From requirements_tensorrt.txt

---

## ⚡ Performance

Approximate speeds on RTX A5000 (24GB):

| Optimization | Speed vs Real-time |
|--------------|-------------------|
| Base PyTorch | ~0.5x |
| FP16 | ~1-1.5x |
| torch.compile | ~2x |
| TensorRT | ~3-5x |

---

## 🎙️ About ClearerVoice-Studio

This project uses [ClearerVoice-Studio](https://github.com/modelscope/ClearerVoice-Studio) — an AI-powered speech processing toolkit by Alibaba that provides:

- **Speech Enhancement** — Denoising
- **Speech Separation** — 2-speaker separation via MossFormer & TensorRT
- **Speech Super-Resolution** — 16kHz → 48kHz bandwidth extension
- **Target Speaker Extraction** — Audio-visual and EEG-based
- **Training & Fine-tuning** — Support for all tasks

### Latest Updates

- **[2025.6]** NumPy array interface for flexible model integration
- **[2025.5]** Enhanced SpeechScore with NISQA and DISTILL_MOS metrics
- **[2025.4]** pip install support: `pip install clearvoice`
- **[2025.4]** Speech super-resolution training scripts
- **[2025.1]** Multi-format audio support (WAV, MP3, AAC, FLAC, etc.)
- **[2024.11]** 3M+ uses of FRCRN denoiser, 2.5M+ uses of MossFormer separator

---

## 📝 License

See [ClearerVoice-Studio](https://github.com/modelscope/ClearerVoice-Studio) for license information.
