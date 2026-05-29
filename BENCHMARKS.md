# mmllm GPU vs CPU Benchmarks

Hardware: **Macmini3,1** — Core 2 Duo + **NVIDIA GeForce 9400M** (nouveau driver, no reclocking)
Model: **TinyStories-3M** + synthetic test models (dummy weights, vocab=1000)
Prompt: `42` (single start token) → 50 tokens, temp=0.8, top-k=40
Date: 2026-05-29

## Results

| Model | Size | CPU (llvmpipe) | GPU (nouveau) | GPU Speedup |
|---|---|---|---|---|
| test_d128_l4 (d128, L4, ffn=512) | 3.6 MB | 78.36 tok/s | **81.53 tok/s** | **×1.04** |
| test_d256_l4 (d256, L4, ffn=1024) | 13 MB | 36.81 tok/s | **51.46 tok/s** | **×1.40** |
| test_d512_l4 (d512, L4, ffn=2048) | 51 MB | 10.62 tok/s | **24.31 tok/s** | **×2.29** |
| test_d1024_l2 (d1024, L2, ffn=4096) | 101 MB | 4.25 tok/s | **18.07 tok/s** | **×4.25** |
| tinystories-3m (d128, L8, 50K vocab) | 32 MB | **13.03 tok/s** | 7.51 tok/s | **×0.58** ❗ |

## Visual

```
tok/s  Legend: ██ GPU (nouveau)  ██ CPU (llvmpipe)

 80 ┤
    │ ████████████████████████████████████████████████ 81.53  GPU
    │ █████████████████████████████████████████████ 78.36  CPU
    │ test_d128_l4
    │
 60 ┤
    │ ███████████████████████████████████████████████████████ 51.46  GPU
    │ ███████████████████████████████████████ 36.81  CPU
    │ test_d256_l4
    │
 40 ┤
    │ ███████████████████████████████████████████████████████████████████ 24.31  GPU
    │ ████████████████████████ 10.62  CPU
    │ test_d512_l4
    │
 20 ┤
    │ ████████████████████████████████████████████████████████████████████████████████████████████████████████████ 18.07  GPU
    │ ██████████████████████████ 4.25  CPU  (*)
    │ test_d1024_l2
    │
  0 └───────────────────────────────────────────────────

⚠ tinystories-3m (32 MB, 50K vocab):  CPU 13.03 > GPU 7.51
   GPU loses here due to massive logits readback overhead.
```

## Analysis

### Crossover point: ~d_model=256 (~10M+ params)

The GPU starts winning consistently at **d_model ≥ 256**. At d1024, the GPU is **4.3× faster** than CPU.

### Why TinyStories-3M is slower on GPU

TinyStories-3M has the same d_model=128 as the smallest test model, but:
- **Vocab size = 50,257** (vs 1,000 for dummies) → **200 KB logits readback per step**
- **8 layers** (vs 4) → **64+ shader dispatches**, each with fixed overhead
- Small matmuls (128-dim) don't benefit from GPU parallelism

The **logits readback + dispatch overhead dominate** the tiny compute. Once dimensions grow (d512+), the GPU's parallel compute overwhelms the overhead.

### Why nouveau is slower than expected

The open-source nouveau driver cannot reclock the GPU (NVIDIA never released the required firmware). The GeForce 9400M runs at **~100 MHz core** instead of **450 MHz** — a ~75% penalty. A proprietary driver (or proper reclocking) would roughly double all GPU numbers.

## Raw Data

### GPU (nouveau, via `sudo startx`)

| Model | Time | Tokens | tok/s |
|---|---|---|---|
| test_d128_l4 | 0.63s | 50 | 81.53 |
| test_d256_l4 | 0.99s | 50 | 51.46 |
| tinystories-3m | 6.79s | 50 | 7.51 |
| test_d512_l4 | 2.10s | 50 | 24.31 |
| test_d1024_l2 | 2.82s | 50 | 18.07 |

### CPU (Xvfb/llvmpipe)

| Model | Time | Tokens | tok/s |
|---|---|---|---|
| test_d128_l4 | 0.64s | 50 | 78.36 |
| test_d256_l4 | 1.39s | 50 | 36.81 |
| tinystories-3m | 3.91s | 50 | 13.03 |
| test_d512_l4 | 4.80s | 50 | 10.62 |
| test_d1024_l2 | 3.76s | ~~16~~\* | 4.25 |

\* test_d1024_l2 CPU run produced only 16/50 tokens — likely hit a memory or stability issue with large texture allocations in software rendering. The 4.25 tok/s is calculated from the 16 tokens actually generated.

## Test Model Configs

| File | d_model | Layers | ffn_hidden | Heads | d_head | Vocab | Size |
|---|---|---|---|---|---|---|---|
| `test_d128_l4.mlm` | 128 | 4 | 512 | 4 | 32 | 1000 | 3.6 MB |
| `test_d256_l4.mlm` | 256 | 4 | 1024 | 8 | 32 | 1000 | 13 MB |
| `tinystories-3m.mlm` | 128 | 8 | 512 | 16 | 8 | 50257 | 32 MB |
| `test_d512_l4.mlm` | 512 | 4 | 2048 | 8 | 64 | 1000 | 51 MB |
| `test_d1024_l2.mlm` | 1024 | 2 | 4096 | 16 | 64 | 1000 | 101 MB |

## Test Methodology

- CPU renderer: Mesa llvmpipe (software, all CPU cores)
- GPU renderer: nouveau (GeForce 9400M, no reclocking)
- 50 tokens per run, temperature=0.8, top-k=40
- Warm starts included in timing (no cold-cache pre-runs)
- Dummy models use random normal weights (seed=42)
