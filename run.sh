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
    echo "  --docker          Use Docker (recommended)"
    echo "  --skip-preprocess Skip preprocessing step"
    echo "  --denoise 0.9     Set denoising strength (0.0-1.0)"
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

# Check for Docker flag
USE_DOCKER=false
SKIP_PREPROCESS=""
DENOISE_STRENGTH="0.8"

for arg in "$@"; do
    if [ "$arg" = "--docker" ]; then
        USE_DOCKER=true
    elif [ "$arg" = "--skip-preprocess" ]; then
        SKIP_PREPROCESS="--skip_preprocess"
    elif [[ "$arg" =~ ^--denoise ]]; then
        DENOISE_STRENGTH="${!#}"
    fi
done

# Function to run with Docker
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

# Function to run with Python
run_with_python() {
    echo -e "${BLUE}Running with Python...${NC}"
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

# Run based on method
if [ "$USE_DOCKER" = true ]; then
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
echo "  📁 $OUTPUT_DIR/separated/speaker1.wav"
echo "  📁 $OUTPUT_DIR/separated/speaker2.wav"
echo ""

if [ -d "$OUTPUT_DIR/preprocessed" ]; then
    echo "Preprocessed files:"
    echo "  📊 $OUTPUT_DIR/preprocessed/diagnostics.json"
    echo "  📁 $OUTPUT_DIR/preprocessed/preprocessed_final.wav"
    echo ""
fi

echo "To play the results:"
echo "  play $OUTPUT_DIR/separated/speaker1.wav"
echo "  play $OUTPUT_DIR/separated/speaker2.wav"
echo ""
