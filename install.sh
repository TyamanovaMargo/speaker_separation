#!/bin/bash
################################################################################
# Installation Script for ClearerVoice-Studio (TensorRT Improved)
################################################################################

set -e  # Exit on error

echo "================================================================================"
echo "CLEARERVOICE-STUDIO (TensorRT Improved) - INSTALLATION"
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

pip install numpy scipy soundfile tqdm

echo ""
echo "✓ Core dependencies installed"

# Install ClearerVoice-Studio requirements if present
if [ -f "ClearerVoice-Studio/requirements.txt" ]; then
    echo ""
    echo "================================================================================"
    echo "Installing ClearerVoice-Studio Requirements"
    echo "================================================================================"
    echo ""
    pip install -r ClearerVoice-Studio/requirements.txt
    echo "✓ ClearerVoice-Studio requirements installed"
fi

# Install TensorRT dependencies if present
if [ -f "ClearerVoice-Studio/clearvoice/requirements_tensorrt.txt" ]; then
    echo ""
    echo "================================================================================"
    echo "Installing TensorRT-specific Requirements"
    echo "================================================================================"
    echo ""
    pip install -r ClearerVoice-Studio/clearvoice/requirements_tensorrt.txt
    echo "✓ TensorRT requirements installed"
fi

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
echo "2. Run separation:"
echo "   cd ClearerVoice-Studio/clearvoice"
echo "   python separate_tensorrt_improved.py --input /path/to/input.wav --output /path/to/output_dir/"
echo ""