#!/bin/sh
# mmllm benchmark script (headless, no X server needed).
# Output: ~/mmllm/benchmark_*.out (plus prints to terminal)
#
# Usage:
#   bash tools/benchmark.sh          # GPU (nouveau), the default
#   bash tools/benchmark.sh --gpu    # same
#   bash tools/benchmark.sh --cpu    # Mesa software renderer (llvmpipe)

set -e

MMLLM_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$MMLLM_DIR"

MODEL="models/tinystories-3m.mlm"
PROMPT_IDS="7454,2402,257,640"
PROMPT_TEXT="Once upon a time"
TOKENS=100
TEMP=0.8
TOP_K=40
OUTFILE="$MMLLM_DIR/benchmark_$(date +%Y%m%d-%H%M%S).out"

MODE="gpu"
for arg in "$@"; do
    case "$arg" in
        --gpu) MODE="gpu" ;;
        --cpu) MODE="cpu" ;;
        --help|-h)
            echo "Usage: bash tools/benchmark.sh [--gpu|--cpu]"
            echo "  --gpu   Real GPU through headless EGL (default)"
            echo "  --cpu   Software rendering (LIBGL_ALWAYS_SOFTWARE=1)"
            exit 0
            ;;
    esac
done

if [ "$MODE" = "cpu" ]; then
    export LIBGL_ALWAYS_SOFTWARE=1
    LABEL="CPU (Mesa software)"
else
    LABEL="GPU (nouveau, headless EGL)"
fi

{
    echo "============================================"
    echo "  mmllm Benchmark"
    echo "  Model:      TinyStories-3M"
    echo "  Prompt:     \"$PROMPT_TEXT\""
    echo "  Tokens:     $TOKENS"
    echo "  Temp/TopK:  $TEMP / $TOP_K"
    echo "  Mode:       $LABEL"
    echo "============================================"
    echo ""
} | tee "$OUTFILE"

./build/mmllm \
    --model "$MODEL" \
    --tokens "$TOKENS" \
    --temperature "$TEMP" \
    --top-k "$TOP_K" \
    --prompt "$PROMPT_IDS" 2>&1 | tee -a "$OUTFILE"

echo ""
echo "=== Full output saved to: $OUTFILE ==="
