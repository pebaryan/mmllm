#!/bin/sh
# Run the mmllm self-test on the physical GPU (nouveau), headless via EGL.
# No X server, sudo or DISPLAY needed.
cd "$(dirname "$0")" || exit 1
exec ./build/mmllm --self-test
