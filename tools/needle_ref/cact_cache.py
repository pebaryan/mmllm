"""Dequantise needle3.cact once and cache the float32 tensors (npz) + tokenizer blob + header."""
import sys, time, pickle
import numpy as np
sys.path.insert(0, ".")
from needle.model.export import read_export
import needle_np as N
from loaders import load_from_cact

t0 = time.time()
meta, tensors = read_export("../weights/needle3.cact")
print("read_export %.1fs; header:" % (time.time() - t0))
for k, v in meta.items():
    if k != "codebook": print("  %-22s %s" % (k, v))
print("  codebook (cb2|cb3|cb4):", np.round(meta["codebook"][:4], 5), "...", len(meta["codebook"]), "floats")
cfg = N.Cfg()
W = load_from_cact(tensors, cfg)
print("tensors:", len(tensors), "| W keys:", len(W), "| head tensors:", [getattr(t, "shape", len(t)) for t in W["_heads"]][:8])
blob = W.pop("_tokenizer_blob"); W.pop("_heads")
np.savez("../weights/needle3_deq.npz", **W)
open("../weights/needle3_tok.bin", "wb").write(blob)
pickle.dump({k: v for k, v in meta.items() if k != "codebook"}, open("../weights/needle3_meta.pkl", "wb"))
print("cached to weights/needle3_deq.npz (%.0f MB)" % (sum(a.nbytes for a in W.values()) / 1e6), "| tokenizer blob", len(blob), "bytes")
