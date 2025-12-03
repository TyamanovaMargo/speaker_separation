#!/bin/bash
################################################################################
# Test Script - Verify Installation and Setup
################################################################################

echo "================================================================================"
echo "SPEAKER SEPARATION PIPELINE - SYSTEM TEST"
echo "================================================================================"
echo ""

# Test 1: Check Python version
echo "Test 1: Python version"
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo "  Found: Python $PYTHON_VERSION"

MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)

if [ "$MAJOR" -eq 3 ] && [ "$MINOR" -ge 10 ] && [ "$MINOR" -le 12 ]; then
    echo "  ✓ Compatible (requires 3.10-3.12)"
else
    echo "  ⚠ Warning: Python 3.10-3.12 recommended (MossFormer2 compatibility)"
fi
echo ""

# Test 2: Check Docker
echo "Test 2: Docker availability"
if command -v docker &> /dev/null; then
    DOCKER_VERSION=$(docker --version | awk '{print $3}' | tr -d ',')
    echo "  ✓ Docker installed: $DOCKER_VERSION"
    
    if docker images | grep -q speaker-separation; then
        echo "  ✓ Docker image 'speaker-separation' found"
    else
        echo "  ⚠ Docker image not built yet (run: docker build -t speaker-separation .)"
    fi
else
    echo "  ⚠ Docker not found (optional, but recommended)"
fi
echo ""

# Test 3: Check project structure
echo "Test 3: Project structure"
DIRS=("scripts/preprocess" "scripts/separation" "output")
for dir in "${DIRS[@]}"; do
    if [ -d "$dir" ]; then
        echo "  ✓ $dir/"
    else
        echo "  ✗ $dir/ (missing)"
    fi
done
echo ""

# Test 4: Check key files
echo "Test 4: Key files"
FILES=(
    "Dockerfile"
    "requirements.txt"
    "complete_pipeline.py"
    "scripts/preprocess/run_all.py"
    "scripts/separation/mossformer2_separate.py"
)

for file in "${FILES[@]}"; do
    if [ -f "$file" ]; then
        echo "  ✓ $file"
    else
        echo "  ✗ $file (missing)"
    fi
done
echo ""

# Test 5: Check virtual environment (if exists)
echo "Test 5: Python virtual environment"
if [ -d "venv" ]; then
    echo "  ✓ venv/ exists"
    
    if [ -f "venv/bin/activate" ]; then
        echo "  ✓ Activation script found"
        
        # Try to activate and check packages
        source venv/bin/activate 2>/dev/null
        
        PACKAGES=("numpy" "scipy" "librosa" "soundfile")
        for pkg in "${PACKAGES[@]}"; do
            if python -c "import $pkg" 2>/dev/null; then
                echo "  ✓ $pkg installed"
            else
                echo "  ✗ $pkg not installed"
            fi
        done
        
        # Check ModelScope (optional)
        if python -c "from modelscope.pipelines import pipeline" 2>/dev/null; then
            echo "  ✓ ModelScope installed"
        else
            echo "  ⚠ ModelScope not installed (required for separation)"
        fi
        
        deactivate 2>/dev/null
    fi
else
    echo "  ⚠ Virtual environment not created yet"
    echo "    Run: bash install.sh"
fi
echo ""

# Test 6: Check if we can import key modules (Docker test)
if command -v docker &> /dev/null && docker images | grep -q speaker-separation; then
    echo "Test 6: Docker container test"
    
    docker run --rm speaker-separation python -c "
import sys
print('  ✓ Python', sys.version.split()[0])

try:
    import numpy
    print('  ✓ numpy', numpy.__version__)
except:
    print('  ✗ numpy')

try:
    import torch
    print('  ✓ torch', torch.__version__)
except:
    print('  ✗ torch')

try:
    from modelscope.pipelines import pipeline
    print('  ✓ ModelScope ready')
except Exception as e:
    print('  ✗ ModelScope:', str(e))
    " 2>&1
fi
echo ""

# Summary
echo "================================================================================"
echo "SUMMARY"
echo "================================================================================"
echo ""
echo "Ready to run?"
echo ""
echo "Method 1: Docker (recommended)"
echo "  bash run.sh your_audio.wav --docker"
echo ""
echo "Method 2: Python virtual environment"
echo "  bash run.sh your_audio.wav"
echo ""
echo "Method 3: Complete pipeline directly"
echo "  python complete_pipeline.py --input your_audio.wav --output_dir output/"
echo ""
