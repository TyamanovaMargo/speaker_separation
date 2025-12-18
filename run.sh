#!/bin/bash
################################################################################
# ONE-CLICK BATCH RUN SCRIPT for ClearerVoice-Studio (TensorRT Improved)
################################################################################

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo "================================================================================"
echo -e "${BLUE}CLEARERVOICE-STUDIO (TensorRT Improved) - BATCH RUNNER${NC}"
echo "================================================================================"
echo ""

if [ -z "$1" ]; then
    echo -e "${YELLOW}Usage:${NC}"
    echo "  bash run.sh <input_folder_with_wavs> [output_directory]"
    echo ""
    echo "Example:"
    echo "  bash run.sh input_audios/ results/"
    echo ""
    exit 1
fi

INPUT_DIR="$1"
OUTPUT_DIR="${2:-output_batch}"

if [ ! -d "$INPUT_DIR" ]; then
    echo -e "${RED}✗ Error: Input folder not found: $INPUT_DIR${NC}"
    exit 1
fi

echo -e "${GREEN}Input folder:${NC} $INPUT_DIR"
echo -e "${GREEN}Output directory:${NC} $OUTPUT_DIR"
echo ""

# Check and activate venv
if [ -d "venv_moss" ]; then
    echo -e "${GREEN}Found existing virtual environment: venv_moss${NC}"
    source venv_moss/bin/activate
elif [ -d "venv" ]; then
    echo -e "${GREEN}Found existing virtual environment: venv${NC}"
    source venv/bin/activate
else
    echo -e "${YELLOW}No virtual environment found. Running install.sh to create venv...${NC}"
    bash install.sh
    source venv/bin/activate
fi

mkdir -p "../../$OUTPUT_DIR"

shopt -s nullglob
AUDIO_FILES=("$OLDPWD/$INPUT_DIR"/*.wav)
if [ ${#AUDIO_FILES[@]} -eq 0 ]; then
    echo -e "${RED}✗ No .wav files found in $INPUT_DIR${NC}"
    exit 1
fi

for AUDIO in "${AUDIO_FILES[@]}"; do
    BASENAME=$(basename "$AUDIO" .wav)
    OUT_SUBDIR="../../$OUTPUT_DIR/$BASENAME"
    mkdir -p "$OUT_SUBDIR"
    echo -e "${BLUE}Processing:${NC} $AUDIO -> $OUT_SUBDIR"
    python separate_tensorrt_improved.py --input "$AUDIO" --output "$OUT_SUBDIR"
done

cd - > /dev/null

echo ""
echo "================================================================================"
echo -e "${GREEN}✓ BATCH SEPARATION COMPLETE!${NC}"
echo "================================================================================"
echo ""
echo "Results in:"
echo "  📁 $OUTPUT_DIR/<audio_name>/*.wav"
echo ""
echo "To play the results for a file:"
echo "  play $OUTPUT_DIR/<audio_name>/*.wav"
echo ""