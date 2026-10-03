"""Weight loaders for needle_np: (a) master safetensors params tree, (b) .cact positional list."""
import numpy as np

f32 = np.float32
LAYER_TENSORS = ["norm_in", "q_proj", "k_proj", "v_proj", "q_taps", "k_taps", "v_taps",
                 "q_norm", "k_norm", "gate_proj", "out_proj", "post_norm", "attn_gate",
                 "pre_hada", "d1", "d2", "b2", "d3", "d4", "w1a", "w1b", "w2a", "w2b",
                 "w3a", "w3b", "cond_v", "cond_u"]


def load_from_params(params, cfg, hada_perms):
    """params: unflattened safetensors tree (numpy). hada_perms: (p1, p2) int arrays."""
    A = lambda x: np.asarray(x).astype(f32)
    W = {"emb": A(params["embedding"]["embedding"])}
    b = params["stack"]["layers"]["block"]
    sa, ha = b["self_attn"], b["hadamard_mlp"]
    for i in range(cfg.layers):
        p = f"L{i}."
        W[p + "norm_in"] = A(b["ZCRMSNorm_0"]["scale"][i])
        for n in ("q_proj", "k_proj", "v_proj", "gate_proj", "out_proj"):
            W[p + n] = A(sa[n]["kernel"][i]).T.copy()          # [out, in]
        for n in ("q_taps", "k_taps", "v_taps"):
            W[p + n] = A(sa[n][i])
        W[p + "q_norm"] = A(sa["q_norm"]["scale"][i])
        W[p + "k_norm"] = A(sa["k_norm"]["scale"][i])
        W[p + "post_norm"] = A(b["post_attn_norm"]["scale"][i])
        W[p + "attn_gate"] = A(b["attn_gate"][i]).reshape(1)
        W[p + "pre_hada"] = A(b["pre_hada_norm"]["scale"][i])
        for n in ("d1", "d2", "b2", "d3", "d4", "w1a", "w1b", "w2a", "w2b", "w3a", "w3b",
                  "cond_v", "cond_u"):
            W[p + n] = A(ha[n][i])
    st = params["stack"]
    for n in ("mhc_a_pre", "mhc_a_post", "mhc_a_res", "mhc_b_pre", "mhc_b_post", "mhc_b_res"):
        W[n] = A(st[n])
    for n in ("mhc_phi_pre", "mhc_phi_post", "mhc_phi_res"):
        phi = A(st[n])                                          # [L, nC, k]
        L, nC, k = phi.shape
        W[n] = phi.transpose(0, 2, 1).reshape(L * k, nC).copy()
    W["hada_p1"], W["hada_p2"] = (np.asarray(p, np.int64) for p in hada_perms)
    for s in range(len(cfg.engram_layers)):
        eg = params[f"engrams_{s}"]
        nt = eg["embedding"].shape[0]
        W[f"engram{s}.tables"] = A(eg["embedding"]).reshape(nt * cfg.slots, cfg.sub)
        W[f"engram{s}.key_proj"] = A(eg["key_proj"]["kernel"]).T.copy()
        W[f"engram{s}.value_proj"] = A(eg["value_proj"]["kernel"]).T.copy()
        W[f"engram{s}.taps"] = A(eg["taps"])
    W["final_norm"] = A(st["final_norm"]["scale"])
    return W


def load_from_cact(tensors, cfg):
    """tensors: list from export.read_export (dequantised float32 arrays, RAW as bytes)."""
    it = iter(tensors)
    W = {"emb": next(it)}
    for i in range(cfg.layers):
        for n in LAYER_TENSORS:
            W[f"L{i}.{n}"] = next(it)
    for n in ("mhc_a_pre", "mhc_a_post", "mhc_a_res", "mhc_b_pre", "mhc_b_post", "mhc_b_res",
              "mhc_phi_pre", "mhc_phi_post", "mhc_phi_res"):
        W[n] = next(it)
    W["hada_p1"] = next(it).astype(np.int64)
    W["hada_p2"] = next(it).astype(np.int64)
    for s in range(len(cfg.engram_layers)):
        for n in ("tables", "key_proj", "value_proj", "taps"):
            W[f"engram{s}.{n}"] = next(it)
    W["final_norm"] = next(it)
    rest = list(it)  # heads.manifest ..., then RAW tokenizer last
    W["_heads"] = rest[:-1]
    W["_tokenizer_blob"] = rest[-1]
    return W
