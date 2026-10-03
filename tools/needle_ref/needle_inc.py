"""Incremental (token-at-a-time, KV-cached) Needle 3 forward in NumPy.
This is the structure the C++ port follows: per-token state = KV caches, 2-step raw q/k/v history
for the causal conv taps, 9-step engram value history, and the last 2 token ids for n-gram hashes."""
import math
import numpy as np
import needle_np as N

f32 = np.float32


class State:
    def __init__(self, cfg):
        self.cfg = cfg
        self.tokens = []
        L = cfg.layers
        self.k_cache = [[] for _ in range(L)]      # each: [kv_heads, qk]
        self.v_cache = [[] for _ in range(L)]      # each: [kv_heads, vh]
        self.raw_q = [[] for _ in range(L)]        # pre-tap projections (last cfg.taps-1 kept)
        self.raw_k = [[] for _ in range(L)]
        self.raw_v = [[] for _ in range(L)]
        self.eg_v = [[] for _ in cfg.engram_layers]  # engram value_proj outputs (pre-tap)


def _hist(lst, j):
    """Element j steps back from the newest entry (0 = newest), or None."""
    return lst[-1 - j] if j < len(lst) else None


def _taps(lst_new, hist, taps, ntap, dil=1):
    """sum_j taps[j] * x[t - j*dil]; hist excludes the newest value which is lst_new."""
    out = taps[0] * lst_new
    for j in range(1, ntap):
        prev = _hist(hist, j * dil - 1)
        if prev is not None:
            out = out + taps[j] * prev
    return out.astype(f32)


def engram_site_kv(W, cfg, st, s, token_idx_t, quant=False):
    """k, v for engram site s at the newest position."""
    t = len(st.tokens) - 1
    nt = len(cfg.orders) * cfg.engram_heads
    tables = W[f"engram{s}.tables"]
    ok_orders = [o for o in cfg.orders for _ in range(cfg.engram_heads)]
    parts = []
    for k in range(nt):
        row = tables[k * cfg.slots + token_idx_t[k]]
        parts.append(row if t >= ok_orders[k] - 1 else np.zeros_like(row))
    e = np.concatenate(parts).astype(f32)
    if quant:   # architecture.Engram: e = _aq(e) before key_proj / value_proj
        e = N.fake_quant_rows(e[None])[0]
    k = W[f"engram{s}.key_proj"] @ e
    v_raw = W[f"engram{s}.value_proj"] @ e
    return k.astype(f32), v_raw.astype(f32)


def step(W, cfg, st, token, cos_t, sin_t, quant=False):
    """Process one token (appends to the state); returns logits [vocab]."""
    st.tokens.append(int(token))
    t = len(st.tokens) - 1
    D, n, L = cfg.d, cfg.lanes, cfg.layers

    # --- engram hash indices for this position (needs the previous 2 tokens; 0 when absent).
    # Plain Python ints with explicit 32-bit masking: exactly the arithmetic the C++ port uses.
    def tok_back(j):
        return st.tokens[-1 - j] if j < len(st.tokens) else 0
    idx = []
    for oi, order in enumerate(cfg.orders):
        for h in range(cfg.engram_heads):
            acc = (N.ENGRAM_SEED * (oi * (cfg.seed_heads or cfg.engram_heads) + h + 1)) & 0xFFFFFFFF
            for j in range(order):
                acc = ((acc ^ tok_back(j)) * N.ENGRAM_PRIME) & 0xFFFFFFFF
            acc ^= acc >> 15
            idx.append(acc % cfg.slots)

    x0 = (W["emb"][token] * f32(math.sqrt(D))).astype(f32)
    stream = np.tile(x0, (n, 1)).astype(f32)                      # [n, C]
    for i in range(L):
        p = f"L{i}."
        lane = np.eye(n, dtype=f32)[i % n]
        pre_off, post_off = 8 * lane - 4, -4 * (1 - lane)
        nx = N.rms_unit(stream.reshape(1, n * D))[0]
        if quant:
            nx = N.fake_quant_rows(nx[None])[0]
        phi_pre = W["mhc_phi_pre"][i * n:(i + 1) * n]               # [n, nC]
        phi_post = W["mhc_phi_post"][i * n:(i + 1) * n]
        phi_res = W["mhc_phi_res"][i * n * n:(i + 1) * n * n]
        hpre = N.sigmoid(W["mhc_a_pre"][i] * (phi_pre @ nx) + W["mhc_b_pre"][i] + pre_off)
        u = (hpre @ stream).astype(f32)                             # [C]

        # ---- block
        x = u
        if i in cfg.engram_layers:
            s = cfg.engram_layers.index(i)
            k_e, v_raw = engram_site_kv(W, cfg, st, s, idx, quant)
            st.eg_v[s].append(v_raw)
            hist = st.eg_v[s][:-1]
            v_e = _taps(v_raw, hist, W[f"engram{s}.taps"], N.ENGRAM_CONV_TAPS, dil=max(cfg.orders))
            a = N.sigmoid(np.array([np.sum(N.rms_unit(x[None])[0] * N.rms_unit(k_e[None])[0]) / math.sqrt(D)], f32))[0]
            x = (x + a * v_e).astype(f32)
        skip = x
        xn = N.zcrms(x[None], W[p + "norm_in"])[0]
        if quant:
            xn = N.fake_quant_rows(xn[None])[0]
        q_raw = (W[p + "q_proj"] @ xn).astype(f32)
        k_raw = (W[p + "k_proj"] @ xn).astype(f32)
        v_raw2 = (W[p + "v_proj"] @ xn).astype(f32)
        st.raw_q[i].append(q_raw); st.raw_k[i].append(k_raw); st.raw_v[i].append(v_raw2)
        q = _taps(q_raw, st.raw_q[i][:-1], W[p + "q_taps"], cfg.taps)
        k = _taps(k_raw, st.raw_k[i][:-1], W[p + "k_taps"], cfg.taps)
        v = _taps(v_raw2, st.raw_v[i][:-1], W[p + "v_taps"], cfg.taps)
        for lst in (st.raw_q[i], st.raw_k[i], st.raw_v[i]):
            del lst[:-(cfg.taps - 1)]
        q = N.zcrms(q.reshape(cfg.heads, cfg.qk), W[p + "q_norm"])
        k = N.zcrms(k.reshape(cfg.kv_heads, cfg.qk), W[p + "k_norm"])
        v = v.reshape(cfg.kv_heads, cfg.vh)
        half = cfg.qk // 2
        def rope(z):
            z1, z2 = z[:, :half], z[:, half:]
            return np.concatenate([z1 * cos_t - z2 * sin_t, z2 * cos_t + z1 * sin_t], axis=-1).astype(f32)
        q, k = rope(q), rope(k)
        if quant:
            q, k, v = N.fake_quant_rows(q), N.fake_quant_rows(k), N.fake_quant_rows(v)
        st.k_cache[i].append(k); st.v_cache[i].append(v)
        is_global = i in cfg.global_layers
        lo = 0 if (is_global or not cfg.window) else max(0, t - cfg.window + 1)
        K = np.stack(st.k_cache[i][lo:])                            # [S, kvh, qk]
        V = np.stack(st.v_cache[i][lo:])                            # [S, kvh, vh]
        group = cfg.heads // cfg.kv_heads
        out = np.zeros((cfg.heads, cfg.vh), f32)
        for h in range(cfg.heads):
            kh = h // group
            sc = (K[:, kh, :] @ q[h]) / math.sqrt(cfg.qk)
            out[h] = N.softmax(sc[None], axis=-1)[0] @ V[:, kh, :]
        out = out.reshape(-1)
        out = out * N.sigmoid(W[p + "gate_proj"] @ xn)
        if quant:
            out = N.fake_quant_rows(out[None])[0]
        a_out = (W[p + "out_proj"] @ out).astype(f32)
        a_out = N.zcrms(a_out[None], W[p + "post_norm"])[0]
        x = skip + N.sigmoid(W[p + "attn_gate"].reshape(())) * a_out
        skip = x
        xn = N.zcrms(x[None], W[p + "pre_hada"])
        mlp = N.hadamard_mlp(W, i, xn, cfg)[0]
        blk = (skip + mlp).astype(f32)

        y = blk - u
        hpost = 2 * N.sigmoid(W["mhc_a_post"][i] * (phi_post @ nx) + W["mhc_b_post"][i] + post_off)
        res = (phi_res @ nx).reshape(n, n)
        hres = N.sinkhorn(W["mhc_a_res"][i] * res + W["mhc_b_res"][i])
        stream = (hres @ stream + hpost[:, None] * y[None, :]).astype(f32)
    xf = N.zcrms(stream.mean(axis=0)[None], W["final_norm"])[0]
    if quant:
        xf = N.fake_quant_rows(xf[None])[0]
    return (W["emb"] @ xf).astype(f32)


def run_tokens(W, cfg, tokens, quant=False, max_pos=None):
    """Feed tokens one at a time; returns logits for every position [T, vocab]."""
    T = len(tokens)
    cos, sin = N.rope_tables(cfg.qk, max_pos or T, cfg.rope_theta)
    st = State(cfg)
    return np.stack([step(W, cfg, st, tok, cos[t], sin[t], quant) for t, tok in enumerate(tokens)]), st
