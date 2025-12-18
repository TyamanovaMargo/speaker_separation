#!/bin/bash
################################################################################
# SPEAKER SEPARATION - EASY RUN SCRIPT
################################################################################

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo "================================================================================"
echo -e "${BLUE}SPEAKER SEPARATION PIPELINE - EASY RUNNER${NC}"
echo "================================================================================"
echo ""

# Check if input file provided
if [ -z "$1" ]; then
    echo -e "${YELLOW}Usage:${NC}"
    echo "  bash run.sh <input_audio_file> [output_directory]"
    echo ""
    echo "Examples:"
    echo "  bash run.sh my_audio.wav"
    echo "  bash run.sh my_audio.wav results/my_audio/"
    echo ""
    echo "Options:"
    echo "  --docker              Use Docker (recommended)"
    echo "  --skip-preprocess     Skip preprocessing step"
    echo "  --denoise 0.9         Set denoising strength (0.0-1.0)"
    echo "  --clearervoice        Use ClearerVoice-Studio (16kHz separation)"
    echo "  --clearervoice-48k    Use ClearerVoice-Studio full 48kHz pipeline"
    echo ""
    exit 1
fi

INPUT_FILE="$1"
OUTPUT_DIR="${2:-output/$(basename $INPUT_FILE .wav)}"

# Check if input exists
if [ ! -f "$INPUT_FILE" ]; then
    echo -e "${RED}✗ Error: Input file not found: $INPUT_FILE${NC}"
    exit 1
fi

echo -e "${GREEN}Input file:${NC} $INPUT_FILE"
echo -e "${GREEN}Output directory:${NC} $OUTPUT_DIR"
echo ""

# Parse flags
USE_DOCKER=false
SKIP_PREPROCESS=""
DENOISE_STRENGTH="0.8"
USE_CLEARERVOICE=false
USE_CLEARERVOICE_48K=false

for arg in "$@"; do
    if [ "$arg" = "--docker" ]; then
        USE_DOCKER=true
    elif [ "$arg" = "--skip-preprocess" ]; then
        SKIP_PREPROCESS="--skip_preprocess"
    elif [[ "$arg" =~ ^--denoise ]]; then
        DENOISE_STRENGTH="${!#}"
    elif [ "$arg" = "--clearervoice" ]; then
        USE_CLEARERVOICE=true
    elif [ "$arg" = "--clearervoice-48k" ]; then
        USE_CLEARERVOICE_48K=true
    fi
done

# Function to run with Docker (modular pipeline only)
run_with_docker() {
    echo -e "${BLUE}Running with Docker...${NC}"
    echo ""
    
    # Check if Docker image exists
    if ! docker images | grep -q speaker-separation; then
        echo -e "${YELLOW}Docker image not found. Building...${NC}"
        docker build -t speaker-separation .
    fi
    
    # Get absolute paths
    ABS_INPUT=$(readlink -f "$INPUT_FILE")
    ABS_OUTPUT=$(mkdir -p "$OUTPUT_DIR" && readlink -f "$OUTPUT_DIR")
    
    docker run --rm \
        -v "$ABS_INPUT:/workspace/input.wav" \
        -v "$ABS_OUTPUT:/workspace/output" \
        speaker-separation \
        python complete_pipeline.py \
            --input /workspace/input.wav \
            --output_dir /workspace/output \
            $SKIP_PREPROCESS \
            --denoise_strength $DENOISE_STRENGTH
}

# Function to run modular pipeline with Python
run_with_python() {
    echo -e "${BLUE}Running with Python (modular pipeline)...${NC}"
    echo ""
    
    # Check if virtual environment exists
    if [ ! -d "venv" ]; then
        echo -e "${YELLOW}Virtual environment not found. Running install.sh...${NC}"
        bash install.sh
    fi
    
    # Activate virtual environment
    source venv/bin/activate
    
    # Run pipeline
    python complete_pipeline.py \
        --input "$INPUT_FILE" \
        --output_dir "$OUTPUT_DIR" \
        $SKIP_PREPROCESS \
        --denoise_strength $DENOISE_STRENGTH
}

# Function to run ClearerVoice-Studio 16kHz
run_clearervoice() {
    echo -e "${BLUE}Running ClearerVoice-Studio (16kHz separation)...${NC}"
    echo ""
    cd ClearerVoice-Studio/clearvoice
    if [ ! -f "separate_clearvoice.py" ]; then
        echo -e "${RED}✗ Error: separate_clearvoice.py not found in ClearerVoice-Studio/clearvoice${NC}"
        exit 1
    fi
    # Activate venv if not already
    if [ -f "../../../venv/bin/activate" ]; then
        source ../../../venv/bin/activate
    fi
    python separate_clearvoice.py --input "$INPUT_FILE" --output "$OUTPUT_DIR"
    cd - > /dev/null
}

# Function to run ClearerVoice-Studio 48kHz pipeline
run_clearervoice_48k() {
    echo -e "${BLUE}Running ClearerVoice-Studio (48kHz full pipeline)...${NC}"
    echo ""
    cd ClearerVoice-Studio/clearvoice
    if [ ! -f "separate_clearvoice_48k.py" ]; then
        echo -e "${RED}✗ Error: separate_clearvoice_48k.py not found in ClearerVoice-Studio/clearvoice${NC}"
        exit 1
    fi
    # Activate venv if not already
    if [ -f "../../../venv/bin/activate" ]; then
        source ../../../venv/bin/activate
    fi
    python separate_clearvoice_48k.py --mode full --input "$INPUT_FILE" --output "$OUTPUT_DIR"
    cd - > /dev/null
}

# Run based on method/flag
if [ "$USE_CLEARERVOICE_48K" = true ]; then
    run_clearervoice_48k
elif [ "$USE_CLEARERVOICE" = true ]; then
    run_clearervoice
elif [ "$USE_DOCKER" = true ]; then
    run_with_docker
else
    run_with_python
fi

# Check results
echo ""
echo "================================================================================"
echo -e "${GREEN}✓ PIPELINE COMPLETE!${NC}"
echo "================================================================================"
echo ""
echo "Results:"
if [ "$USE_CLEARERVOICE" = true ] || [ "$USE_CLEARERVOICE_48K" = true ]; then
    echo "  📁 $OUTPUT_DIR"
    echo "  (See ClearerVoice-Studio output for separated/enhanced files)"
else
    echo "  📁 $OUTPUT_DIR/separated/speaker1.wav"
    echo "  📁 $OUTPUT_DIR/separated/speaker2.wav"
    if [ -d "$OUTPUT_DIR/preprocessed" ]; then
        echo "Preprocessed files:"
        echo "  📊 $OUTPUT_DIR/preprocessed/diagnostics.json"
        echo "  📁 $OUTPUT_DIR/preprocessed/preprocessed_final.wav"
    fi
fi
echo ""

echo "To play the results:"
if [ "$USE_CLEARERVOICE" = true ] || [ "$USE_CLEARERVOICE_48K" = true ]; then
    echo "  play $OUTPUT_DIR/*.wav"
else
    echo "  play $OUTPUT_DIR/separated/speaker1.wav"
    echo "  play $OUTPUT_DIR/separated/speaker2.wav"
