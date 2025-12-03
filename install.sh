#!/bin/bash
################################################################################
# Installation Script for Modular Audio Preprocessing Pipeline
################################################################################

set -e  # Exit on error

echo "================================================================================"
echo "MODULAR AUDIO PREPROCESSING - INSTALLATION"
echo "================================================================================"
echo ""

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $PYTHON_VERSION"
echo ""

# Check if virtual environment exists
if [ -d "venv" ]; then
    echo "⚠️  Virtual environment already exists."
    read -p "Remove and recreate? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf venv
        echo "✓ Removed old virtual environment"
    else
        echo "Using existing virtual environment"
    fi
fi

# Create virtual environment
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    echo "✓ Virtual environment created"
fi

# Activate virtual environment
echo ""
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip

# Install core dependencies
echo ""
echo "================================================================================"
echo "Installing Core Dependencies"
echo "================================================================================"
echo ""

pip install numpy scipy librosa soundfile pyyaml tqdm noisereduce

echo ""
echo "✓ Core dependencies installed"

# Ask about PyTorch
echo ""
echo "================================================================================"
echo "PyTorch Installation (Optional - for future ML features)"
echo "================================================================================"
echo ""
echo "Do you want to install PyTorch?"
echo "  (y) Yes, with CUDA GPU support"
echo "  (c) Yes, CPU only"
echo "  (n) No, skip for now"
echo ""
read -p "Choice [y/c/n]: " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Installing PyTorch with CUDA support..."
    pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118
    echo "✓ PyTorch with CUDA installed"
elif [[ $REPLY =~ ^[Cc]$ ]]; then
    echo "Installing PyTorch (CPU only)..."
    pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
    echo "✓ PyTorch CPU installed"
else
    echo "Skipping PyTorch installation"
fi

# Test imports
echo ""
echo "================================================================================"
echo "Testing Installation"
echo "================================================================================"
echo ""

python3 << 'PYTHON_TEST'
import sys

print("Testing imports...")
failed = []

try:
    import numpy
    print("✓ numpy")
except ImportError as e:
    print("✗ numpy")
    failed.append("numpy")

try:
    import scipy
    print("✓ scipy")
except ImportError:
    print("✗ scipy")
    failed.append("scipy")

try:
    import librosa
    print("✓ librosa")
except ImportError:
    print("✗ librosa")
    failed.append("librosa")

try:
    import soundfile
    print("✓ soundfile")
except ImportError:
    print("✗ soundfile")
    failed.append("soundfile")

try:
    import noisereduce
    print("✓ noisereduce")
except ImportError:
    print("✗ noisereduce (optional)")

try:
    import torch
    print("✓ torch")
except ImportError:
    print("⚪ torch (optional, not installed)")

if failed:
    print(f"\n✗ Some required packages failed: {', '.join(failed)}")
    sys.exit(1)
else:
    print("\n✓ All required packages imported successfully!")
PYTHON_TEST

# Final instructions
echo ""
echo "================================================================================"
echo "INSTALLATION COMPLETE!"
echo "================================================================================"
echo ""
echo "Next steps:"
echo ""
echo "1. Activate virtual environment:"
echo "   source venv/bin/activate"
echo ""
echo "2. Test with your audio file:"
echo "   python scripts/preprocess/01_audio_diagnostics.py --input your_audio.wav"
echo ""
echo "3. Or use quick start:"
echo "   bash quick_start.sh"
echo ""
echo "4. Read documentation:"
echo "   cat README.md"
echo ""
