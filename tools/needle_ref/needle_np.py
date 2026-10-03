"""Pure-NumPy forward pass of Needle 3 (SimpleAttentionNetwork), float32, written against
needle/model/architecture.py. Used as (a) the executable specification for the C++ port and
(b) a cross-check against the JAX reference.

Weights come in as a dict of named float32 arrays (see load_from_params / load_from_cact):
  emb [V, D]; per layer L{i}: norm_in, q_proj [out,in], k_proj, v_proj, q_taps, k_taps, v_taps,
  q_norm, k_norm, gate_proj, out_proj, post_norm, attn_gate, pre_hada, d1, d2, b2, d3, d4,
  w1a..w3b, cond_v [D, 8], cond_u [8, n]; mhc_* ; hada_p1/p2; engram{s}.* ; final_norm.
"""
import math
import numpy as np

f32 = np.float32


def zcrms(x, scale, eps=1e-6):
    """ZCRMSNorm: (1 + scale) * x / sqrt(mean(x^2) + eps), reduced over the last axis."""
    x = x.astype(f32)
    rms = np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + f32(eps))
    return ((1 + scale.astype(f32)) * x / rms).astype(f32)


def rms_unit(x, eps=1e-6):
    x = x.astype(f32)
    return (x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + f32(eps))).astype(f32)


def sigmoid(x):
    return (1.0 / (1.0 + np.exp(-x.astype(np.float64)))).astype(f32)


def silu(x):
    return (x * sigmoid(x)).astype(f32)


def softmax(x, axis=-1):
    x = x.astype(np.float64)
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return (e / e.sum(axis=axis, keepdims=True)).astype(f32)


def fake_quant_rows(x, bits=8):
    """Symmetric per-row fake quantisation (quantize.fake_quant with group = last dim)."""
    qmax = 2 ** (bits - 1) - 1
    x = x.astype(f32)
    absmax = np.max(np.abs(x), axis=-1, keepdims=True)
    scale = np.where(absmax > 0, absmax / qmax, 1.0).astype(f32)
    q = np.clip(np.round(x / scale), -qmax - 1, qmax) * scale
    return q.astype(f32)


def shift_right(x, k):
    """Shift along axis 0 (time) by k, zero fill (architecture._shift_right)."""
    if k == 0:
        return x
    out = np.zeros_like(x)
    if k < x.shape[0]:
        out[k:] = x[:-k]
    return out


def rope_tables(head_dim, T, theta):
    freqs = 1.0 / (theta ** (np.arange(0, head_dim, 2).astype(np.float32) / head_dim))
    ang = np.outer(np.arange(T).astype(np.float32), freqs).astype(f32)
    return np.cos(ang).astype(f32), np.sin(ang).astype(f32)


def apply_rope(x, cos, sin):
    """x [T, H, hd]; half-rotation as in architecture.apply_rope."""
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    c, s = cos[:, None, :], sin[:, None, :]
    return np.concatenate([x1 * c - x2 * s, x2 * c + x1 * s], axis=-1).astype(f32)


def sinkhorn(logits, iters=20):
    lk = logits.astype(np.float64)
    for _ in range(iters):
        lk = lk - np.log(np.exp(lk - lk.max(-1, keepdims=True)).sum(-1, keepdims=True)) - lk.max(-1, keepdims=True)
        lk = lk - np.log(np.exp(lk - lk.max(-2, keepdims=True)).sum(-2, keepdims=True)) - lk.max(-2, keepdims=True)
    return np.exp(lk).astype(f32)


def kron_apply(z, a, b):
    """z [T, a0*b0]; out[k,l] = sum_ij z[i,j] a[i,k] b[j,l]."""
    T = z.shape[0]
    zz = z.reshape(T, a.shape[0], b.shape[0])
    out = np.einsum("tij,ik,jl->tkl", zz, a, b, optimize=True)
    return out.reshape(T, a.shape[0] * b.shape[0]).astype(f32)


ENGRAM_SEED = 0x9E3779B9
ENGRAM_PRIME = 0x01000193
ENGRAM_CONV_TAPS = 4


def engram_indices(tokens, orders, heads, slots, seed_heads=0):
    u = tokens.astype(np.uint32)
    stride = seed_heads or heads
    idx = []
    for oi, order in enumerate(orders):
        for h in range(heads):
            seed = (ENGRAM_SEED * (oi * stride + h + 1)) & 0xFFFFFFFF
            acc = np.full_like(u, np.uint32(seed))
            for j in range(order):
                acc = (acc ^ shift_right(u, j)) * np.uint32(ENGRAM_PRIME)
            acc = acc ^ (acc >> np.uint32(15))
            idx.append((acc % np.uint32(slots)).astype(np.int64))
    return np.stack(idx, axis=-1)  # [T, num_tables]


class Cfg:
    d = 768
    layers = 20
    heads = 12
    kv_heads = 2
    qk = 48
    vh = 64
    rope_theta = 100000.0
    window = 1024
    global_layers = (4, 9, 14, 19)
    engram_layers = (3, 7, 11, 15, 19)
    orders = (2, 3)
    engram_heads = 3
    slots = 18432
    sub = 128
    lanes = 4
    taps = 3
    vocab = 8192
    seed_heads = 0


def engram_kv(W, cfg, tokens, quant=False):
    T = len(tokens)
    idx = engram_indices(tokens, cfg.orders, cfg.engram_heads, cfg.slots, cfg.seed_heads)
    num_tables = len(cfg.orders) * cfg.engram_heads
    ok = np.zeros((T, num_tables), f32)
    for t_i, o in enumerate(o for o in cfg.orders for _ in range(cfg.engram_heads)):
        ok[o - 1:, t_i] = 1.0
    ks, vs = [], []
    for s in range(len(cfg.engram_layers)):
        tables = W[f"engram{s}.tables"].reshape(num_tables, cfg.slots, cfg.sub)
        fetched = np.stack([tables[k][idx[:, k]] for k in range(num_tables)], axis=1)  # [T, nt, sub]
        fetched = fetched * ok[..., None]
        e = fetched.reshape(T, num_tables * cfg.sub).astype(f32)
        if quant:
            e = fake_quant_rows(e)
        k = e @ W[f"engram{s}.key_proj"].T
        v = e @ W[f"engram{s}.value_proj"].T
        taps = W[f"engram{s}.taps"]
        dil = max(cfg.orders)
        v = sum(taps[j] * shift_right(v, j * dil) for j in range(ENGRAM_CONV_TAPS)).astype(f32)
        ks.append(k.astype(f32)); vs.append(v)
    return np.stack(ks), np.stack(vs)  # [S, T, D]


def attention(W, i, x, cfg, cos, sin, is_global, quant):
    """x: normalised input [T, D]. Returns [T, D] (output of out_proj)."""
    T = x.shape[0]
    p = f"L{i}."
    if quant:
        x = fake_quant_rows(x)
    q = x @ W[p + "q_proj"].T
    k = x @ W[p + "k_proj"].T
    v = x @ W[p + "v_proj"].T
    nt = cfg.taps
    qt, kt, vt = W[p + "q_taps"], W[p + "k_taps"], W[p + "v_taps"]
    q = sum(qt[j] * shift_right(q, j) for j in range(nt)).astype(f32)
    k = sum(kt[j] * shift_right(k, j) for j in range(nt)).astype(f32)
    v = sum(vt[j] * shift_right(v, j) for j in range(nt)).astype(f32)
    q = q.reshape(T, cfg.heads, cfg.qk)
    k = k.reshape(T, cfg.kv_heads, cfg.qk)
    v = v.reshape(T, cfg.kv_heads, cfg.vh)
    q = zcrms(q, W[p + "q_norm"])
    k = zcrms(k, W[p + "k_norm"])
    q = apply_rope(q, cos, sin)
    k = apply_rope(k, cos, sin)
    if quant:  # per-head int8 (maybe_quant_query / maybe_quant_kv)
        q, k, v = fake_quant_rows(q), fake_quant_rows(k), fake_quant_rows(v)
    group = cfg.heads // cfg.kv_heads
    scale = 1.0 / math.sqrt(cfg.qk)
    pos = np.arange(T)
    mask = pos[None, :] <= pos[:, None]
    if not is_global and cfg.window:
        mask = mask & ((pos[:, None] - pos[None, :]) < cfg.window)
    out = np.zeros((T, cfg.heads, cfg.vh), f32)
    for h in range(cfg.heads):
        kh = h // group
        sc = (q[:, h, :] @ k[:, kh, :].T) * scale
        sc = np.where(mask, sc, -np.inf)
        out[:, h, :] = softmax(sc, axis=-1) @ v[:, kh, :]
    out = out.reshape(T, cfg.heads * cfg.vh)
    out = out * sigmoid(x @ W[p + "gate_proj"].T)
    if quant:
        out = fake_quant_rows(out)
    return (out @ W[p + "out_proj"].T).astype(f32)


def hadamard_mlp(W, i, x, cfg):
    p = f"L{i}."
    n = 1024
    d1, d2, b2, d3, d4 = (W[p + k] for k in ("d1", "d2", "b2", "d3", "d4"))
    cond = 1 + softmax(x @ W[p + "cond_v"], axis=-1) @ W[p + "cond_u"]
    z = np.zeros((x.shape[0], n), f32)
    z[:, :cfg.d] = x
    p1 = W["hada_p1"].astype(np.int64)
    p2 = W["hada_p2"].astype(np.int64)
    z = kron_apply(d1 * z, W[p + "w1a"], W[p + "w1b"])[:, p1]
    z = kron_apply(silu(d2 * cond * z + b2), W[p + "w2a"], W[p + "w2b"])[:, p2]
    z = kron_apply(d3 * z, W[p + "w3a"], W[p + "w3b"])
    return (d4 * z)[:, :cfg.d].astype(f32)


def block(W, i, u, cfg, cos, sin, ek, ev, quant):
    """Block.__call__ on the lane-mixed stream u [T, D]; returns skip + mlp (full residual)."""
    x = u
    if i in cfg.engram_layers:
        s = cfg.engram_layers.index(i)
        a = sigmoid(np.sum(rms_unit(x) * rms_unit(ek[s]), axis=-1) / math.sqrt(cfg.d))
        x = (x + a[:, None] * ev[s]).astype(f32)
    skip = x
    p = f"L{i}."
    xn = zcrms(x, W[p + "norm_in"])
    a_out = attention(W, i, xn, cfg, cos, sin, i in cfg.global_layers, quant)
    a_out = zcrms(a_out, W[p + "post_norm"])
    x = skip + sigmoid(W[p + "attn_gate"].reshape(())) * a_out
    skip = x
    xn = zcrms(x, W[p + "pre_hada"])
    return (skip + hadamard_mlp(W, i, xn, cfg)).astype(f32)


def forward(W, cfg, tokens, quant=False, return_cells=False):
    """Returns logits [T, vocab] (and optionally per-layer lane-mean cells)."""
    tokens = np.asarray(tokens)
    T = len(tokens)
    x0 = (W["emb"][tokens] * f32(math.sqrt(cfg.d))).astype(f32)
    cos, sin = rope_tables(cfg.qk, T, cfg.rope_theta)
    ek, ev = engram_kv(W, cfg, tokens, quant)
    n, L, C = cfg.lanes, cfg.layers, cfg.d
    stream = np.broadcast_to(x0[:, None, :], (T, n, C)).astype(f32).copy()
    cells = [x0]
    for i in range(L):
        lane = np.eye(n, dtype=f32)[i % n]
        pre_off = 8 * lane - 4
        post_off = -4 * (1 - lane)
        nx = rms_unit(stream.reshape(T, n * C))
        if quant:
            nx = fake_quant_rows(nx)
        phi_pre = W["mhc_phi_pre"][i * n:(i + 1) * n].T          # [nC, n]
        phi_post = W["mhc_phi_post"][i * n:(i + 1) * n].T
        phi_res = W["mhc_phi_res"][i * n * n:(i + 1) * n * n].T  # [nC, n*n]
        hpre = sigmoid(W["mhc_a_pre"][i] * (nx @ phi_pre) + W["mhc_b_pre"][i] + pre_off)
        u = np.einsum("tn,tnc->tc", hpre, stream).astype(f32)
        y = block(W, i, u, cfg, cos, sin, ek, ev, quant) - u
        hpost = 2 * sigmoid(W["mhc_a_post"][i] * (nx @ phi_post) + W["mhc_b_post"][i] + post_off)
        res = (nx @ phi_res).reshape(T, n, n)
        hres = sinkhorn(W["mhc_a_res"][i] * res + W["mhc_b_res"][i])
        stream = (np.einsum("tij,tjc->tic", hres, stream) + hpost[..., None] * y[:, None, :]).astype(f32)
        cells.append(stream.mean(axis=1))
    xf = stream.mean(axis=1)
    xf = zcrms(xf, W["final_norm"])
    if quant:
        xf = fake_quant_rows(xf)
    logits = (xf @ W["emb"].T).astype(f32)
    return (logits, np.stack(cells, axis=1)) if return_cells else logits
