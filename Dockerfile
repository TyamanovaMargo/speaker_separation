# =============================================================================
# ClearerVoice-Studio Speaker Separation - Docker Image
# =============================================================================

FROM nvcr.io/nvidia/tensorrt:23.10-py3

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Fix GPG signature issues and install system dependencies
RUN rm -rf /var/lib/apt/lists/* && \
    apt-get clean && \
    apt-get update --fix-missing && \
    apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    libsndfile1-dev \
    sox \
    git \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Install PyTorch 2.0.1 + matching torchaudio (compatible with torch-tensorrt 1.4.0)
RUN pip install --no-cache-dir \
    torch==2.0.1 \
    torchaudio==2.0.2 \
    --index-url https://download.pytorch.org/whl/cu118

# Install torch-tensorrt (compatible with torch 2.0.1)
RUN pip install --no-cache-dir torch-tensorrt==1.4.0

# Install audio/ML dependencies
RUN pip install --no-cache-dir \
    numpy scipy soundfile librosa tqdm pyyaml \
    transformers huggingface_hub modelscope

# Copy ClearerVoice-Studio
COPY ClearerVoice-Studio /app/ClearerVoice-Studio

# Install ClearerVoice requirements
WORKDIR /app/ClearerVoice-Studio
RUN if [ -f requirements.txt ]; then pip install --no-cache-dir -r requirements.txt; fi

WORKDIR /app

# Copy config
COPY config /app/config

# Create directories
RUN mkdir -p /app/input /app/output

# Environment
ENV HF_HOME=/app/.cache/huggingface
ENV TORCH_HOME=/app/.cache/torch
ENV PYTHONPATH="/app/ClearerVoice-Studio:/app/ClearerVoice-Studio/clearvoice:${PYTHONPATH}"

RUN mkdir -p $HF_HOME $TORCH_HOME

WORKDIR /app/ClearerVoice-Studio/clearvoice

CMD ["python", "separate_tensorrt_improved.py", "--help"]
