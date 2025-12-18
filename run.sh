#!/bin/bash
################################################################################
# ONE-CLICK BATCH RUN SCRIPT for ClearerVoice-Studio (TensorRT Improved)
# Process all .wav/.WAV files in input folder with separate_tensorrt_improved.py
################################################################################

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo "================================================================================"
echo -e "${BLUE}CLEARERVOICE-STUDIO (TensorRT Improved) - ONE-CLICK BATCH RUNNER${NC}"
echo "================================================================================"
echo ""

# Default input/output folders
INPUT_DIR="${1:-input/data_call_center}"
OUTPUT_DIR="${2:-results}"

if [ ! -d "$INPUT_DIR" ]; then
    echo -e "${RED}✗ Error: Input folder not found: $INPUT_DIR${NC}"
    echo ""
    echo "Usage:"
    echo "  bash run.sh [input_folder] [output_folder]"
    echo ""
    echo "Example:"
    echo "  bash run.sh input/data_call_center results/"
    echo ""
    exit 1
fi

echo -e "${GREEN}Input folder:${NC} $INPUT_DIR"
echo -e "${GREEN}Output folder:${NC} $OUTPUT_DIR"
echo ""

# Check and activate venv (prefer venv_moss if it exists)
if [ -d "venv_moss" ]; then
    echo -e "${GREEN}✓ Found venv_moss${NC}"
    source venv_moss/bin/activate
elif [ -d "venv" ]; then
    echo -e "${GREEN}✓ Found venv${NC}"
    source venv/bin/activate
else
    echo -e "${YELLOW}⚠️  No virtual environment found. Running install.sh...${NC}"
    bash install.sh
    source venv/bin/activate
fi

echo ""

# Navigate to separation script directory
cd ClearerVoice-Studio/clearvoice

if [ ! -f "separate_tensorrt_improved.py" ]; then
    echo -e "${RED}✗ Error: separate_tensorrt_improved.py not found${NC}"
    exit 1
fi

# Prepare output directory
mkdir -p "../../$OUTPUT_DIR"

# Find all .wav and .WAV files
shopt -s nullglob
AUDIO_FILES=("$OLDPWD/$INPUT_DIR"/*.wav "$OLDPWD/$INPUT_DIR"/*.WAV)

if [ ${#AUDIO_FILES[@]} -eq 0 ]; then
    echo -e "${RED}✗ No .wav or .WAV files found in $INPUT_DIR${NC}"
    exit 1
fi

echo -e "${BLUE}Found ${#AUDIO_FILES[@]} audio file(s)${NC}"
echo ""

# Process each audio file
COUNT=0
for AUDIO in "${AUDIO_FILES[@]}"; do
    COUNT=$((COUNT + 1))
    BASENAME=$(basename "$AUDIO")
    BASENAME_NO_EXT="${BASENAME%.*}"
    OUT_SUBDIR="../../$OUTPUT_DIR/$BASENAME_NO_EXT"
    
    mkdir -p "$OUT_SUBDIR"
    
    echo -e "${BLUE}[${COUNT}/${#AUDIO_FILES[@]}] Processing:${NC} $BASENAME_NO_EXT"
    python separate_tensorrt_improved.py --input "$AUDIO" --output "$OUT_SUBDIR"
    echo -e "${GREEN}✓ Done${NC}"
    echo ""
done

cd - > /dev/null

echo "================================================================================"
echo -e "${GREEN}✓ BATCH SEPARATION COMPLETE!${NC}"
echo "================================================================================"
echo ""
echo "📁 Results in: $OUTPUT_DIR/"
echo ""
echo "To play separated audio files:"
echo "  play $OUTPUT_DIR/<audio_name>/*.wav"
echo ""