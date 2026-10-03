Reference implementations used to validate the C++ Needle 3 port (src/needle)
==========================================================================

needle_np.py     pure-NumPy forward pass (full sequence), written against the authors' architecture.py
needle_inc.py    the same model, token-at-a-time with KV caches: the structure the C++ code follows
loaders.py       maps (a) the master safetensors parameter tree or (b) the dequantised .cact tensor list
                 into the arrays needle_np expects
check_jax.py     cross-check vs the JAX reference (cosine 1.0, max logit diff 8e-5)
check_inc.py     incremental vs full forward
gen_np.py        greedy tool-call generation + comparison with the official engine's answers
golden.py        writes golden.bin / golden_prompt.txt used by `mmllm --needle-test`
cact_cache.py    dequantises needle3.cact once and caches float32 tensors (npz)
golden/          golden.bin, golden_prompt.txt (prompt ids + full logits, quant off / on)

These scripts import the authors' package modules (needle.model.architecture / export / quantize /
checkpoints / tokenizer, Apache-2.0, https://github.com/cactus-compute/needle). They are NOT copied here:
get them with  pip install "cactus-needle[train]"  (note: the package sends anonymous usage counts unless
NEEDLE_TELEMETRY=0), or place the files from the repository under  needle/model/  next to these scripts.
Running JAX needs a CPU with AVX; it does not run on the Mac mini - the reference was run on a PC.
