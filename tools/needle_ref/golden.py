"""Write weights/golden.bin: prompt ids + full logits at selected positions (quant off/on)."""
import json, struct, sys
import numpy as np
sys.path.insert(0, ".")
from needle.model.export import parse_tokenizer_blob, RefTokenizer
import needle_np as N, needle_inc as I
import gen_np as G

d = np.load("../weights/needle3_deq.npz"); W = {k: d[k] for k in d.files}
tok = RefTokenizer(parse_tokenizer_blob(open("../weights/needle3_tok.bin", "rb").read()))
cfg = N.Cfg()
NOMM = json.loads(json.dumps(G.TOOLS))
for p in NOMM[0]["parameters"]["properties"].values():
    p.pop("minimum", None); p.pop("maximum", None)
text = G.render_prompt("dim the living room to 30", NOMM)
ids = [2] + tok.encode(text)
print("prompt: %d tokens; first 12 ids: %s" % (len(ids), ids[:12]))
positions = [0, 1, 2, 3, 40, 80, len(ids) - 1]
cos, sin = N.rope_tables(cfg.qk, len(ids), cfg.rope_theta)
out = bytearray(struct.pack("<I", len(ids)) + np.asarray(ids, np.int32).tobytes())
out += struct.pack("<I", len(positions))
for quant in (0, 1):
    st = I.State(cfg)
    recs = {}
    for t, i in enumerate(ids):
        lg = I.step(W, cfg, st, i, cos[t], sin[t], bool(quant))
        if t in positions: recs[t] = lg
    for t in positions:
        out += struct.pack("<II", quant, t) + recs[t].astype(np.float32).tobytes()
    print("quant=%d: last-position top-5 ids %s, top logit %.4f" % (quant, np.argsort(-recs[positions[-1]])[:5], recs[positions[-1]].max()))
open("../weights/golden.bin", "wb").write(bytes(out))
open("../weights/golden_prompt.txt", "w", encoding="utf-8").write(text)
print("golden.bin: %d bytes" % len(out))
