# mmllm — Minimal GLSL LLM Inference Engine

**mmllm** runs small transformer language models on old GPUs using nothing but **OpenGL 3.3 fragment
shaders (GLSL 3.30)**: no CUDA, no OpenCL, and **no X server** (it creates a surfaceless EGL context on
the DRM render node, so it works over SSH).

It also contains a **CPU port of Cactus Needle 3**, a 121M-parameter tool-calling model — see
[NEEDLE.md](NEEDLE.md).

The classic "render-to-texture" GPGPU approach:
- Weights, activations and the KV cache live in **textures** (RGBA16F for weights/KV, RGBA32F for activations)
- Matrix multiplication is a **fragment shader** drawn over a full-screen triangle
- Each fragment computes 4 output elements with 4-wide dot products
- Stages chain through **FBOs**; LayerNorm, attention and the LM head also run on the GPU

## Target hardware

- **Macmini3,1** (early 2009), **NVIDIA GeForce 9400M** (shared memory, ~200 MB usable), Core 2 Duo P7350
- Zorin OS 17 / Ubuntu 22.04, kernel 6.8, **nouveau** + Mesa → OpenGL 3.3

## Measured results (Mac mini above)

Greedy output of every model below was checked against the HuggingFace reference.

| Model | Params | tok/s (GPU) |
|---|---|---|
| TinyStories-3M | 3M | ~74 |
| TinyStories-8M | 8M | ~45 |
| TinyStories-28M | 28M | ~23 |
| TinyStories-33M | 33M | ~15.5 |
| GPT-2 small | 124M | ~9 (LM head on the CPU) |

Hardware limits that shaped the code (details in the source comments):

- Never sample a texture wider than 4096 texels or taller than 2048 rows (the LM-head table is split
  across several textures).
- Do not unroll the matmul K loop: it hung the whole machine.
- ~200 MB GPU memory budget; going over it is a cliff (≈5× slower, then `ENOMEM`). Weights and KV are
  fp16, and the LM head falls back to the CPU automatically when the footprint is too large
  (`MMLLM_LM=cpu|gpu`, `MMLLM_GPU_BUDGET_MB`).
- Texture uploads race with queued draws; `glFinish()` before uploading into a texture the GPU may still read.
- The GPU is ~7–8× slower with no display server running at all, so keep GDM (or any X server) up.

## Building

```bash
sudo apt install build-essential cmake libegl1-mesa-dev libgl1-mesa-dev libglew-dev python3-pip
mkdir -p build && cd build && cmake .. && make -j$(nproc)
./mmllm --self-test          # checks the GL context and the matmul shader
```

Run from the repository root or from `build/`: the executable finds `src/shaders` in either place.

## Getting a model

```bash
pip install torch transformers numpy
python3 tools/export_model.py --model roneneldan/TinyStories-33M --output models/tinystories-33m.mlm
python3 tools/export_model.py --dummy --output models/test.mlm        # random weights for smoke tests
```

## Running

```bash
./build/mmllm --model models/tinystories-33m.mlm --tokens 100
./build/mmllm --model models/tinystories-33m.mlm --prompt "42,128,256" --tokens 50   # token ids
python3 tools/decode.py --model roneneldan/TinyStories-33M "42 128 256 512 1"        # ids → text
```

Useful environment variables: `MMLLM_DEBUG`, `MMLLM_PROFILE=1` (per-stage timing), `MMLLM_CPU_ATTN`,
`MMLLM_CPU_LN`, `MMLLM_FUSE`, `MMLLM_W_FP32`, `MMLLM_KV_FP32`, `MMLLM_LM_FP32`, `MMLLM_MAX_SEQ`.

## Needle 3 (tool calling) mode

```bash
./build/mmllm --needle models/needle3.cact --tools tools.json --prompt "dim the living room to 30"
./build/mmllm --needle models/needle3.cact --tools tools.json --serve --port 8080   # localhost HTTP
```

A from-scratch C++ port of the reference network: packed 2-bit weights multiplied directly, int8
activation simulation, prefix caching, grammar-constrained decoding, argument repair and gates, and the
confidence head. On the Mac mini's CPU it decodes at ~27 tok/s and uses ~105 MB, against 6.8 tok/s and
79 MB for the official runner on the same machine; calls match the official runner on 45 of 47 test
queries. An opt-in `--gpu` path exists but is ~3× slower than the CPU on this hardware. Everything is
documented in [NEEDLE.md](NEEDLE.md), and the test tooling is in `tools/needle_ref`, `tools/needle_battery`
and `tools/gpu_bench`.

## License

MIT — do whatever you want with this. Needle 3 weights are Apache-2.0 (Cactus Compute) and are not
included in this repository.
