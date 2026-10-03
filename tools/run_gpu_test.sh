#!/bin/sh
# Run the mmllm self-test on the physical GPU (nouveau), headless via EGL.
# Saves output to ~/mmllm/gpu_test.out
MMLLM_DIR="$(cd "$(dirname "$0")/.." && pwd)"
OUTFILE="$MMLLM_DIR/gpu_test.out"

cd "$MMLLM_DIR" || exit 1
./build/mmllm --self-test > "$OUTFILE" 2>&1
echo "Exit code: $?" >> "$OUTFILE"
cat "$OUTFILE"
