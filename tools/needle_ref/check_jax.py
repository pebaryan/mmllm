"""Cross-check: JAX reference (float32, unquantised master weights) vs needle_np on the same weights."""
import dataclasses, sys, time
import numpy as np

sys.path.insert(0, ".")
import jax, jax.numpy as jnp
from needle.model.checkpoints import read_checkpoint
from needle.model.architecture import SimpleAttentionNetwork, TransformerConfig, _hada_perms
import needle_np as N
from loaders import load_from_params

CKPT = "../weights/needle3.safetensors"
t0 = time.time()
ckpt = read_checkpoint(CKPT)
saved = ckpt["config"] if isinstance(ckpt["config"], dict) else dict(vars(ckpt["config"]))
config = TransformerConfig.from_saved(saved)
params = {k: v for k, v in ckpt["params"].items() if not k.startswith("mtp_")}
print("loaded master checkpoint in %.1fs; config: d=%d L=%d heads=%d/%d qk=%d v=%d vocab=%d out_vocab=%d dtype=%s "
      "engram_layers=%s global=%s window=%d taps=%d lanes=%d slots=%d seed_heads=%d" % (
          time.time() - t0, config.d_model, config.num_layers, config.num_heads, config.num_kv_heads,
          config.qk_head_dim, config.v_head_dim, config.vocab_size, config.out_vocab, config.dtype,
          config.engram_layers, config.global_layers, config.sliding_window, config.qkv_conv_taps,
          config.mhc_lanes, config.engram_slots, config.engram_seed_heads))
print("param dtypes sample:", {k: str(np.asarray(v).dtype) for k, v in
                                 list({"emb": params["embedding"]["embedding"]}.items())})

cfg32 = dataclasses.replace(config, dtype="float32")
model = SimpleAttentionNetwork(cfg32)

rng = np.random.RandomState(0)
tokens = np.concatenate([[2], rng.randint(16, 8192, size=47)]).astype(np.int32)   # BOS + 47 random pieces
print("tokens:", tokens[:12], "... len", len(tokens))

t0 = time.time()
jparams = jax.tree_util.tree_map(lambda a: jnp.asarray(a, jnp.float32), params)
ref = np.asarray(model.apply({"params": jparams}, jnp.asarray(tokens)[None], quant=False))[0]
print("JAX forward: %.1fs, logits %s" % (time.time() - t0, ref.shape))

cfg = N.Cfg()
cfg.seed_heads = int(config.engram_seed_heads or 0)
p1, p2 = _hada_perms(1 << (config.d_model - 1).bit_length(), bool(getattr(config, "ladder_widths", ())))
W = load_from_params(params, cfg, (np.asarray(p1), np.asarray(p2)))
t0 = time.time()
mine = N.forward(W, cfg, tokens, quant=False)
print("NumPy forward: %.1fs, logits %s" % (time.time() - t0, mine.shape))


def cos(a, b):
    a, b = a.ravel().astype(np.float64), b.ravel().astype(np.float64)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


print("\n== logits, all positions ==")
print("cosine (all)        : %.8f" % cos(ref, mine))
print("max |diff|          : %.5f   (logit scale: std %.3f, max %.3f)" % (np.abs(ref - mine).max(), ref.std(), np.abs(ref).max()))
print("top-1 agreement     : %d / %d positions" % ((ref.argmax(-1) == mine.argmax(-1)).sum(), len(tokens)))
print("last position top-5 : JAX %s | mine %s" % (np.argsort(-ref[-1])[:5], np.argsort(-mine[-1])[:5]))
