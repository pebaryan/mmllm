#!/bin/sh
# mmllm scaling benchmark: test all model sizes (headless, no X server needed).
# Usage: bash tools/scale_benchmark.sh [--gpu|--cpu]
#   --gpu   Real GPU through headless EGL (default)
#   --cpu   Mesa software renderer (LIBGL_ALWAYS_SOFTWARE=1)

set -e

MMLLM_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$MMLLM_DIR"

RESULTS_DIR="$MMLLM_DIR/scale_results"
mkdir -p "$RESULTS_DIR"

TIMESTAMP=$(date +%Y%m%d-%H%M%S)
TOKENS=50
TEMP=0.8
TOP_K=40

# Models to test: model_path|display_name
MODELS="
models/test_d128_l4.mlm|d128_L4_(~3M)
models/test_d256_l4.mlm|d256_L4_(~13M)
models/tinystories-3m.mlm|tinystories-3m_(32M)
models/test_d512_l4.mlm|d512_L4_(~51M)
models/test_d1024_l2.mlm|d1024_L2_(~101M)
"

MODE="gpu"
case "${1:-}" in
    --cpu) MODE="cpu" ;;
    --gpu|"") MODE="gpu" ;;
    *) echo "Usage: bash tools/scale_benchmark.sh [--gpu|--cpu]"; exit 1 ;;
esac

if [ "$MODE" = "cpu" ]; then
    export LIBGL_ALWAYS_SOFTWARE=1
    RENDERER="llvmpipe (Mesa software)"
else
    RENDERER="GeForce 9400M (nouveau, headless EGL)"
fi

SUMMARY="$RESULTS_DIR/${MODE}_summary_${TIMESTAMP}.txt"

echo "=== ${MODE} benchmark ===" > "$SUMMARY"
echo "Date: $(date)" >> "$SUMMARY"
echo "Renderer: $RENDERER" >> "$SUMMARY"
echo "" >> "$SUMMARY"
printf "%-35s %12s %12s %12s\n" "Model" "Time(s)" "Tokens" "tok/s" >> "$SUMMARY"
printf "%s\n" "----------------------------------------" >> "$SUMMARY"

IFS="
"
for line in $MODELS; do
    [ -z "$line" ] && continue
    MODEL_PATH=$(echo "$line" | cut -d'|' -f1)
    DISPLAY_NAME=$(echo "$line" | cut -d'|' -f2)
    FULL_PATH="$MMLLM_DIR/$MODEL_PATH"

    if [ ! -f "$FULL_PATH" ]; then
        echo "SKIP: $DISPLAY_NAME (not found)"
        continue
    fi

    echo "Running $DISPLAY_NAME ($(ls -lh "$FULL_PATH" | awk '{print $5}'))..."
    MODEL_OUT="$RESULTS_DIR/${MODE}_${DISPLAY_NAME%%_*}_${TIMESTAMP}.out"

    ./build/mmllm \
        --model "$FULL_PATH" \
        --tokens "$TOKENS" \
        --temperature "$TEMP" \
        --top-k "$TOP_K" \
        --prompt "42" > "$MODEL_OUT" 2>&1 || true

    TOK_PER_SEC=$(grep -oP '[\d.]+(?= tok/s)' "$MODEL_OUT" || echo "N/A")
    TIME_SEC=$(grep -oP 'in [\d.]+ seconds' "$MODEL_OUT" | grep -oP '[\d.]+' || echo "N/A")

    printf "  %-31s %6ss %8s %8s\n" "$DISPLAY_NAME" "$TIME_SEC" "$TOKENS" "$TOK_PER_SEC"
    printf "%-35s %12s %12s %12s\n" "$DISPLAY_NAME" "$TIME_SEC" "$TOKENS" "$TOK_PER_SEC" >> "$SUMMARY"
done

echo "" >> "$SUMMARY"
echo "=== ${MODE} benchmark complete ===" >> "$SUMMARY"
echo ""
echo "Results saved to: $SUMMARY"
echo ""
cat "$SUMMARY"
