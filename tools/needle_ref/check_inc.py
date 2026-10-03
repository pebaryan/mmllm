"""Incremental KV-cached forward must reproduce the full forward exactly (same weights)."""
import dataclasses, sys, time
import numpy as np
sys.path.insert(0, ".")
from needle.model.checkpoints import read_checkpoint
from needle.model.architecture import TransformerConfig, _hada_perms
import needle_np as N, needle_inc as I
from loaders import load_from_params

ckpt = read_checkpoint("../weights/needle3.safetensors")
saved = ckpt["config"] if isinstance(ckpt["config"], dict) else dict(vars(ckpt["config"]))
config = TransformerConfig.from_saved(saved)
params = {k: v for k, v in ckpt["params"].items() if not k.startswith("mtp_")}
cfg = N.Cfg(); cfg.seed_heads = int(config.engram_seed_heads or 0)
p1, p2 = _hada_perms(1024, False)
W = load_from_params(params, cfg, (np.asarray(p1), np.asarray(p2)))

rng = np.random.RandomState(1)
tokens = np.concatenate([[2], rng.randint(16, 8192, size=39)]).astype(np.int64)
t0 = time.time(); full = N.forward(W, cfg, tokens); t_full = time.time() - t0
t0 = time.time(); inc, _ = I.run_tokens(W, cfg, tokens); t_inc = time.time() - t0
print("full %.1fs  incremental %.1fs  (%d tokens)" % (t_full, t_inc, len(tokens)))
d = np.abs(full - inc)
print("max |diff| = %.6f   mean = %.8f   top-1 agreement %d/%d" % (d.max(), d.mean(), (full.argmax(-1) == inc.argmax(-1)).sum(), len(tokens)))
