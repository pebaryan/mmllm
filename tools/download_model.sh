#!/bin/bash
# Download and export a TinyStories model to .mlm format
# Usage: bash tools/download_model.sh [model_name] [output_path]

set -e

MODEL="${1:-roneneldan/TinyStories-1M}"
OUTPUT="${2:-models/tinystories-1m.mlm}"

echo "== mmllm Model Downloader =="
echo "Model: $MODEL"
echo "Output: $OUTPUT"

# Check for Python dependencies
python3 -c "import torch, transformers" 2>/dev/null || {
    echo "Installing dependencies..."
    pip install torch transformers --quiet
}

# Create output directory
mkdir -p "$(dirname "$OUTPUT")"

# Export model
python3 tools/export_model.py --model "$MODEL" --output "$OUTPUT"

echo ""
echo "Done! To run inference:"
echo "  ./mmllm --model $OUTPUT --tokens 100"
