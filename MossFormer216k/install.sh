#!/bin/bash
# Quick install script for MossFormer2 16kHz + TensorRT

set -e

echo "============================================"
echo "MossFormer2 16kHz + TensorRT Installation"
echo "============================================"
echo

# Check Python
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python: $PYTHON_VERSION"

# Detect CUDA
if command -v nvcc &> /dev/null; then
    CUDA_VERSION=$(nvcc --version | grep "release" | sed -n 's/.*release \([0-9]\+\.[0-9]\+\).*/\1/p')
    echo "CUDA: $CUDA_VERSION"
    HAS_CUDA=true
else
    echo "CUDA: Not found (will install CPU version)"
    HAS_CUDA=false
fi
echo

# Create venv
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

source venv/bin/activate

# Upgrade pip
pip install --upgrade pip setuptools wheel

# Install PyTorch
echo
echo "Installing PyTorch..."
if [ "$HAS_CUDA" = true ]; then
    CUDA_MAJOR=$(echo $CUDA_VERSION | cut -d. -f1)
    
    if [ "$CUDA_MAJOR" = "12" ]; then
        echo "Installing PyTorch with CUDA 12.1..."
        pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
    elif [ "$CUDA_MAJOR" = "11" ]; then
        echo "Installing PyTorch with CUDA 11.8..."
        pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118
    else
        echo "Installing PyTorch with default CUDA..."
        pip install torch torchaudio
    fi
else
    echo "Installing PyTorch (CPU only)..."
    pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
fi

# Install core dependencies
echo
echo "Installing dependencies..."
pip install numpy scipy librosa soundfile tqdm modelscope

# Install TensorRT (optional)
if [ "$HAS_CUDA" = true ]; then
    echo
    read -p "Install TensorRT for 2-3x speedup? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Installing TensorRT..."
        if [ "$CUDA_MAJOR" = "12" ]; then
            pip install torch-tensorrt --extra-index-url https://download.pytorch.org/whl/cu121
        elif [ "$CUDA_MAJOR" = "11" ]; then
            pip install torch-tensorrt --extra-index-url https://download.pytorch.org/whl/cu118
        else
            pip install torch-tensorrt
        fi
        echo "✓ TensorRT installed"
    fi
fi

# Test
echo
echo "Testing installation..."
python3 << 'EOF'
import torch
print(f"✓ PyTorch {torch.__version__}")

if torch.cuda.is_available():
    print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
else:
    print("⚪ CUDA not available (CPU mode)")

try:
    import torch_tensorrt
    print("✓ TensorRT available")
except:
    print("⚪ TensorRT not installed (optional)")

import numpy, librosa, soundfile
print("✓ Audio libraries ready")

from modelscope.pipelines import pipeline
print("✓ ModelScope ready")
EOF

if [ $? -eq 0 ]; then
    echo
    echo "============================================"
    echo "✓ Installation complete!"
    echo "============================================"
    echo
    echo "Usage:"
    echo "  source venv/bin/activate"
    echo "  python separate.py --input audio.wav --output results/"
    echo
else
    echo
    echo "⚠ Installation had errors. Check messages above."
fi
