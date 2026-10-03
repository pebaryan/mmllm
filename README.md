# mmllm — Minimal GLSL LLM Inference Engine

**mmllm** is a minimal LLM inference engine that uses **OpenGL 3.3 fragment shaders (GLSL 3.30)** to accelerate small transformer language models on old GPUs.

Instead of CUDA or OpenCL, it uses the classic "render-to-texture" GPGPU approach:
- Matrices are stored as **RGBA32F textures** (4 floats per texel)
- Matrix multiplication is a **fragment shader** running on a full-screen quad
- Each pass computes 4 output elements, accumulated via 4-wide dot products
- Pipelines chain through **FBOs** (Framebuffer Objects)

## Target Hardware

- **Macmini3,1** (early 2009) with **NVIDIA GeForce 9400M**
- **ZorinOS** (Ubuntu Jammy) with **nouveau** driver → OpenGL 3.3
- Target model: **TinyStories-1M** (~4 MB FP32, fits in GPU memory)

## Architecture

```
                    ┌─────────────────────┐
 Token IDs ─────────▶  CPU Embedding      │
                    └────────┬────────────┘
                             ▼
                    ┌─────────────────────┐
                    │  Block 0..N         │
                    │  ┌───────────────┐  │
                    │  │ LayerNorm     │  │  (fragment shader)
                    │  │ Self-Attention│  │  (MatMul + CPU softmax)
                    │  │ Residual Add  │  │  (fragment shader)
                    │  │ LayerNorm     │  │  (fragment shader)
                    │  │ FFN Gate      │  │  (MatMul shader)
                    │  │ GELU          │  │  (fragment shader)
                    │  │ FFN Down      │  │  (MatMul shader)
                    │  │ Residual Add  │  │  (fragment shader)
                    │  └───────────────┘  │
                    └────────┬────────────┘
                             ▼
                    ┌─────────────────────┐
                    │  Final LayerNorm    │
                    │  LM Head (MatMul)   │
                    │  Softmax + Sample   │
                    └────────┬────────────┘
                             ▼
                      Next Token ID
```

## Data Layout

| Concept | Implementation |
|---|---|
| Matrix packing | 4 elements per RGBA32F texel (column-major within texel) |
| Texture A (M×K) | Size: [⌈K/4⌉, M], element A[m][k] = texel(k/4, m).channel(k%4) |
| Texture B (K×N) | Size: [⌈N/4⌉, K], element B[k][n] = texel(n/4, k).channel(n%4) |
| MatMul C = A·B | Render to [⌈N/4⌉, M]. Each fragment computes 4 output elements |
| Inner loop | 4-wide dot product: sum_q A[m][k·4+q] · B[k·4+q][n·4+c] |

## Building

```bash
# Install dependencies (Ubuntu/ZorinOS)
sudo apt install build-essential cmake libgl1-mesa-dev libglew-dev python3-pip

# Build
mkdir -p build && cd build
cmake ..
make -j$(nproc)

# Run self-test (verifies GPU functionality)
./mmllm --self-test
```

## Getting a Model

### Option A: Download from HuggingFace (if available)

```bash
# Install Python deps
pip install torch transformers numpy

# Download and export
python3 tools/download_model.sh
```

### Option B: Generate a dummy model for testing

```bash
# Create a random 2-layer test model
python3 tools/export_model.py --dummy --output models/test.mlm

# Or configure dimensions
python3 tools/export_model.py --dummy --output models/test.mlm \
    --config '{"d_model": 64, "n_layers": 2, "ffn_hidden": 256}'
```

### Option C: Export your own model

```bash
# From HuggingFace
python3 tools/export_model.py --model roneneldan/TinyStories-1M --output models/tinystories-1m.mlm

# From a local checkpoint (coming soon)
python3 tools/export_model.py --checkpoint model.pt --config config.json --output models/model.mlm
```

## Running

```bash
# Run self-test (no model required)
cd build && ./mmllm --self-test

# Generate text
cd build && ./mmllm --model ../models/tinystories-1m.mlm --tokens 100

# Custom prompt (comma-separated token IDs)
cd build && ./mmllm --model ../models/tinystories-1m.mlm \
    --prompt "42,128,256" --tokens 50
```

## Decoding Output

The engine outputs token IDs. Use the Python decoder to convert to text:

```bash
python3 tools/decode.py --model roneneldan/TinyStories-1M "42 128 256 512 1"
```

## Performance

On the GeForce 9400M (16 CUDA cores, 450 MHz):

| Model | Params | Expected tok/s |
|---|---|---|
| TinyStories-1M | ~1M | 1-5 tok/s (GPU matmul) |
| TinyStories-8M | ~8M | <1 tok/s |

*Note: The CPU fallback (Core 2 Duo) should match or exceed GPU performance
for these tiny models since data transfer overhead dominates.*

## License

MIT — Do whatever you want with this.

## Needle 3 (tool calling) mode

Cactus Compute Needle 3 lives in src/needle. It runs on the CPU by default, with an optional GPU
backend for its 2-bit matrix-vector products (`--gpu`): see [NEEDLE.md](NEEDLE.md) — on the GeForce
9400M the CPU is ~4x faster, so the GPU path is opt-in.
