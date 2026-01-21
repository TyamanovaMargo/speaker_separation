# Running ClearerVoice Speaker Separation with Docker

## Prerequisites
- Docker installed
- NVIDIA GPU with drivers and nvidia-docker support
- Audio file(s) to process

## Quick Start

### 1. Prepare Directories
Create the following directories in the project root:
```
mkdir -p input output_batch checkpoints config models
```

### 2. Place Audio Files
Put your input audio files in the `input/` directory.

### 3. Build and Run with Docker Compose (Recommended)

#### Build the image
```bash
docker-compose up --build
```

#### Run separation on a single file
```bash
docker-compose run separation \
  -i /input/your_audio.wav \
  -o /results/
```

#### Run separation with silence removal only
```bash
docker-compose run separation \
  -i /input/your_audio.wav \
  -o /results/ \
  --remove-silence
```

#### Run batch processing on all files in input/
```bash
docker-compose run batch
```

### 4. Run with Docker (Alternative)

#### Build image manually
```bash
docker build -t clearvoice-separation .
```

#### Run separation
```bash
docker run --gpus all \
  -v $(pwd)/input:/app/input:ro \
  -v $(pwd)/output_batch:/app/results \
  -v $(pwd)/checkpoints:/app/checkpoints \
  -v $(pwd)/config:/app/config \
  -v $(pwd)/models:/app/.cache \
  clearvoice-separation \
  python separate_tensorrt_v2.py -i /input/your_audio.wav -o /results/
```

#### Run silence removal only
```bash
docker run --gpus all \
  -v $(pwd)/input:/app/input:ro \
  -v $(pwd)/output_batch:/app/results \
  -v $(pwd)/checkpoints:/app/checkpoints \
  -v $(pwd)/config:/app/config \
  -v $(pwd)/models:/app/.cache \
  clearvoice-separation \
  python separate_tensorrt_v2.py -i /input/your_audio.wav -o /results/ --remove-silence
```

## Options

- `--opt 3`: Use TensorRT optimization (default, fastest)
- `--opt 2`: Use torch.compile
- `--opt 1`: Use FP16
- `--opt 0`: Base PyTorch (slowest)
- `--no-post-process`: Disable post-processing for faster processing
- `--enhance-first`: Apply enhancement before separation
- `--chunk-sec 30`: Chunk size in seconds
- `--overlap-sec 5`: Overlap between chunks

## Output

Results will be saved in:
- `output_batch/` when using docker-compose
- Your specified output directory when using docker run

Each input file produces:
- `filename_speaker1.wav`: First speaker
- `filename_speaker2.wav`: Second speaker
- `filename_metrics.json`: Quality metrics

When using `--remove-silence`:
- `filename_nosilence.wav`: Input audio with all silent gaps removed

## Troubleshooting

- Ensure the `models/` directory exists (can be empty)
- Make sure you have GPU support: `docker run --gpus all nvidia/cuda:11.8-base-ubuntu20.04 nvidia-smi`
- Check logs with `docker-compose logs separation`
