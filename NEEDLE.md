# Needle 3 mode

`mmllm --needle` runs Cactus Compute's **Needle 3** (a 121M-parameter tool-calling / extraction model,
Apache-2.0) from its `.cact` archive. It is a from-scratch C++ port of the reference
`SimpleAttentionNetwork`; it uses no code from the Cactus engine.

It runs on the **CPU** by default; an optional **GPU backend** for the 2-bit matrix-vector products is
described at the bottom of this file (`--gpu`).

```sh
mmllm --needle models/needle3.cact --tools tools.json --prompt "dim the living room to 30"
```

Output is one JSON object: `function_calls`, `reasoning`, token counts and tokens/s.
Options: `--system <facts>`, `--max <n>`, `--threads <n>`, `--no-quant`, `--strip-constraints`,
`--raw` (also print the raw generation), `--needle-test <dir>`.
Env: `NEEDLE_PROFILE=1` prints ms/token per section.

The weights are `Cactus-Compute/needle3` on Hugging Face (`needle3.cact`, 35 MB).

## What is implemented

* `.cact` reader (`src/needle/cact.*`): header, directory, CQ dequantisation (LSB-first indices,
  Lloyd-Max codebooks from the header, fp16 group norms, normalised 128-point Walsh-Hadamard).
* SentencePiece BPE tokenizer read from the archive (`tokenizer.*`), byte-identical to the reference.
* The full model (`nmodel.*`): 4-lane hyper-connections with Sinkhorn mixing, GQA attention with 3-tap
  convolutions / per-head RMSNorm / RoPE / local window, the Hadamard MLP, hashed n-gram "engram"
  memory at layers 3/7/11/15/19, tied output head. One token at a time with KV caches.
* 2-bit weight matrices stay packed and are multiplied directly (the activation is rotated once per
  128-group); this CPU is memory-bandwidth bound, so this is ~12x less DRAM traffic than float32.
* `--quant` (default) simulates the engine's int8 activations / int8 KV cache, which is what the archive
  was tuned for. `--no-quant` uses float activations.

## What is NOT implemented (the shipped engine does these)

* Grammar-constrained decoding (the model is run greedily; output is parsed afterwards).
* The deterministic argument repair step (e.g. it drops a `brightness: 0` the user never stated).
* The confidence head and the suppression gates.
* Multi-turn conversations, tool-name snake_casing / alias table, tool retrieval (> 5 tools).
* The engine's 256-token KV window (prompts longer than ~256 tokens may differ).

## Verification

* NumPy reference (`tools/needle_ref/`) vs the **JAX reference**: cosine 1.0, max logit diff 8e-5.
* C++ vs NumPy golden data (`--needle-test tools/needle_ref/golden`): tokenizer ids identical; float
  logits max diff 5e-5; with the int8 simulation cosine 0.99999 and the same top choice.
* End to end vs the official x86-64 runner on 10 queries: reasoning text identical 10/10, function
  calls identical 9/10 (the exception is the repair step above).

## Speed (Mac mini 2009, Core 2 Duo P7350, 2 threads, 113-token prompt)

prefill ~34 tok/s, decode ~27 tok/s, ~510 MB RAM; the official runner: 9.7 / 6.9 tok/s.

## Confidence
The probe head is implemented on the CPU (nmodel.cpp). `confidence = min(sigmoid(head), lowest probability of the call tokens the model chose itself)`; calls with confidence < 0.1 are moved to `suppressed_calls`. On a 47-query battery the calls match the official engine in 45/47 cases and confidence differs by 0.03 on average. Set NEEDLE_CONF_DEBUG=1 to print the components.

## Activation quantisation and speed
Default (NEEDLE_QMODE=1): the int8 activation simulation is applied per 128-group after the Walsh-Hadamard rotation, which is where the weights are consumed. This matches the official engine better than whole-row quantisation before the rotation (battery: reasoning text identical 38/47 vs 36/47, confidence error 0.019 vs 0.029; calls 45/47 either way). NEEDLE_QMODE=0 gives the whole-row variant used by the JAX oracle (the golden test pins it). Mode 2 (both) is worse (28/47).
Decode is ~27 tok/s on the Core 2 Duo: kron product vectorised, packed matvec does two rows per pass.

## Tool-result turns
Turns whose input is JSON are rendered as `\n<|im_start|>tool\n<tool_result>…</tool_result><|im_end|>\n<|im_start|>assistant\n`, as in the porting notes, and a result turn that ends without a call is reported as `"type":"respond"` like the official engine. The official engine answers `respond` (empty call list) to most plain results; this port often does not, because the next-token margins in the reasoning text are only 0.02-0.4 logits and the int8 arithmetic differs slightly, so the turn can flip into a call. Variants of the history (no closing <|im_end|> via NEEDLE_HIST=1, call-only history, forced structure) did not improve it. NEEDLE_TOP2_DEBUG=1 prints the margins.

## Engine arithmetic (investigated, not matched)
The official binary rounds activations with lroundf (half away from zero) and accumulates with integer dot products (AVX-VNNI / vpmaddwd). NEEDLE_QROUND=1 switches this port to half-away rounding; on the battery it scored 35/47 reasoning (default 38/47), calls 45/47 either way. Exact ties are rare, so the switch only perturbs near-tied tokens: differences of a few queries are noise, not signal. Closing the gap for real needs the integer weight and activation format of the engine (codebook scaling for the int8 dot), which is not documented.

## Memory
The engram tables (about 280 MB as float32) stay in their packed CQ form and a 128-wide row is expanded on lookup (same arithmetic as the loader, 30 rows per token). The raw archive image is freed after loading. Peak RSS 380 MB -> 105 MB (official engine: 79 MB) with no change in speed or output.

## GPU backend (optional)

`src/needle/ngpu.*` offloads the **2-bit CQ matrix-vector products** to the GPU: the q/k/v/gate/out
projections and the engram key/value projections — 110 matrices, ~42M multiply-accumulates, i.e. all
of the model's quantised weight traffic. Everything else (zcrms/rmsnorm, RoPE, the conv taps,
attention + softmax, the Hadamard MLP, the engram gather, the confidence head, the tied float head)
stays on the CPU.

```sh
mmllm --needle models/needle3.cact --tools tools.json --prompt "dim the living room to 30" --gpu
mmllm --needle models/needle3.cact --gpu-check     # verify GPU against CPU on every projection
```

How it works:

* The weights stay **packed** exactly as the archive stores them. The `in/4` bytes of a row go into an
  `R8UI` texture (one byte = four 2-bit indices per texel, decoded with bit ops in the shader) and the
  per-128-group norms into an `R32F` texture. No weight is ever expanded to float32 — that matters,
  because dequantised the model would be ~484 MB and would not fit the 9400M's 256 MB.
* The weights were rotated with the normalised Walsh-Hadamard matrix `H` (symmetric, orthogonal), so
  `w·x = norm · Σ_k codebook[idx_k]·(Hx)_k`: the *same* rotated activation the CPU path feeds its
  integer kernel can be handed straight to the shader. The CPU keeps doing the rotation and the int8
  simulation, so both paths consume identical inputs.
* Two passes: `needle_gemv2_partial.frag` writes one partial per (128-group, output row) — `in/128`
  times more fragments than one-row-per-fragment, which the weak GPU needs — and
  `needle_gemv2_reduce.frag` sums them.

### Verification

`--gpu-check` recomputes all 110 matrices on both paths and compares:

```
110 matrices checked: worst |CPU-GPU| = 3.43e-05, largest |value| = 1.81e+02, worst relative = 3.42e-07
```

End to end the golden test passes with `--gpu` too (float mode max |logit diff| 6e-05 vs the JAX
reference, cosine 1.00000000, top-1 7/7), and a prompt returns a byte-identical call whose confidence
moves only in the 4th decimal (0.7397 CPU vs 0.7395 GPU).

### Performance — why `--gpu` is not the default

On the target hardware (GeForce 9400M, 16 shader cores at 450 MHz, nouveau) the GPU is **slower**:

| 768x768 2-bit matvec | time | throughput |
|---|---:|---:|
| CPU (Core 2 Duo P7350, 2 threads) | 0.15 ms | 3.9 GMAC/s |
| GPU (split-K, this implementation) | 0.61 ms | 1.0 GMAC/s |

Measured end to end (113-token prefix, 38 generated tokens):

| | prefill | decode |
|---|---:|---:|
| CPU (default) | 36.3 tok/s | 29.6 tok/s |
| `--gpu` | 9.0 tok/s | 8.4 tok/s |

The GPU is ~4x slower, and the per-token profile shows why: qkv+taps+rope goes 5.3 ms -> 42.9 ms and
gate+out 7.8 ms -> 43.4 ms. A matvec this small is memory/fetch-bound, and the 9400M's texture
throughput is low, while each product also costs a download sync; the Core 2 Duo's SSE integer-LUT
kernel already sustains 3.9 GMAC/s. The backend is kept for correctness and portability — it is the
faster choice on any modern GPU — and the CPU remains the default.

## Could the GPU help? (measured, GeForce 9400M via the engine MatMul, fp16 weights, validated against CPU)
Build/run: tools/gpu_bench/enginebench.cpp (see header; run from ~/mmllm).
| shape (M tokens x K x N) | GPU ms/call | GPU GMAC/s |
| 1 x 768 x 96 | 0.30 | 0.25 (latency floor) |
| 1 x 768 x 768 | 0.44 | 1.35 |
| 1 x 768 x 2048 | 0.79 | 2.0 |
| 8 x 768 x 768 | 1.22 | 3.9 |
| 32 x 768 x 768 | 4.7 | 4.0 |
CPU (packed 2-bit, 2 cores): about 0.19 ms for 768x768 (3 GMAC/s). Per token Needle does ~110 matmul calls (~48M MACs) interleaved with CPU-only work (Kronecker MLP, mHC, attention, engram gather), so a GPU decode would pay a draw floor of ~0.3 ms plus a readback sync per dependent op: estimated 80+ ms/token versus 31 ms on the CPU. Only batched matmuls (M >= 8, i.e. prefill) beat the CPU, by about 1.3-2x on the matmul share of prefill; not worth the rewrite and the hang risk. Decision: keep Needle on the CPU.

## Batched prefill (`--batch-prefill`)

`Model::prefill()` (`src/needle/nmodel.cpp`) runs the prompt through the network with the **layer loop
outermost**, so each 2-bit projection becomes one GEMM over the whole batch (`Y[m] = W x[m]`) instead of
one GEMV per token. It reproduces the per-token path **bit-identically**: checked over a 107-token
prefix against 20 layers x (k, v, prevQ, prevK, prevV), the 5 engram value histories, the probe pool
(M/S/R), the logits of one further step, and the confidence logit — `max|diff| = 0` on every tensor.

Layer-outer order is equivalent because at layer *i* token *m* sees exactly tokens 0..m in the KV cache
(sequentially, every earlier token has already finished every layer), the q/k/v conv taps read the
previous tokens' raw projections *at the same layer*, the engram value history is appended for the whole
batch in order before any tap is read, and the probe pool is fed in strict token order from buffered
cell vectors. Matrices left on the GPU are not batched (there is no batched GPU kernel), so
`--gpu --batch-prefill` falls back to per-token matvecs for those.

Measured on the target Core 2 Duo, 2 threads, 3 runs each, best of:

| prefix tokens | per-token `step()` | batched prefill | speedup |
|---:|---:|---:|---:|
| 32 | 786 ms | 882 ms | 0.89x |
| 64 | 1610 ms | 1806 ms | 0.89x |
| 101 | 2619 ms | 2917 ms | 0.90x |
| 256 | 7392 ms | 8087 ms | 0.91x |
| 512 | 17211 ms | 18493 ms | 0.93x |

So it is **7–11% slower** here and only converges toward parity as the batch grows. The cause is not the
loop order — every batch blocking factor (1, 2, 4, 8, 16, 32, 101) was measured and all land at
0.64–0.78x of the per-token order — but that the per-token order is already optimal on this chip: the
147 KB packed weight matrix is L2-resident and the 3 KB activation is L1-resident, so reusing a decoded
weight row across the batch buys nothing while it costs re-reading activations out of L2, and the 2-bit
decode is ALU-bound, so batching cannot reduce the work per multiply-accumulate.

Off by default, for the same reason as `--gpu`: on this machine it is the slower choice. It is kept
because this is the ordering a wider machine (more cores, AVX-512, or a real GPU) actually wants.

