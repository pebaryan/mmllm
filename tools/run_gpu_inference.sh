#!/bin/sh
# Run mmllm inference on the physical GPU (nouveau), headless via EGL.
# Runs TinyStories-3M with multiple sampling settings.
# Saves output to ~/mmllm/gpu_inference.out
MMLLM_DIR="$(cd "$(dirname "$0")/.." && pwd)"
OUTFILE="$MMLLM_DIR/gpu_inference.out"

cd "$MMLLM_DIR" || exit 1

run() {
    echo "========================================" >> "$OUTFILE"
    echo " $1" >> "$OUTFILE"
    echo "========================================" >> "$OUTFILE"
    shift
    ./build/mmllm --model models/tinystories-3m.mlm --tokens 10 "$@" >> "$OUTFILE" 2>&1
    echo "" >> "$OUTFILE"
}

echo "=== TinyStories-3M on GeForce 9400M (nouveau, headless EGL) ===" > "$OUTFILE"
echo "" >> "$OUTFILE"

run "1) temp=0 (greedy)" --temperature 0
run "2) temp=0.7" --temperature 0.7
run "3) temp=1.0" --temperature 1.0
run "4) temp=0.8 top-k=40" --temperature 0.8 --top-k 40

echo "=== All runs complete ===" >> "$OUTFILE"
echo "Output saved to: $OUTFILE"
