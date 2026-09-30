"""
sweep.py - accuracy-vs-compression experiments on a saved trace.

    python sweep.py check first, always: proves the trace is a faithful copy of the model
    python sweep.py quant   quantization sweep      -> results_quant.csv
    python sweep.py evict   eviction sweep          -> results_evict.csv
    python sweep.py evict --cfg int4-tok-g32        eviction + quantization combined

Accuracy metric = cosine similarity / relative-L2 of the attention OUTPUT versus
full-precision full-cache attention (per query, averaged). Perplexity comes later.
"""
import argparse
import csv

import numpy as np

import kvgold as kg

# name, k_mode, k_bits, k_group, v_bits, v_group
#   k_group: channels per group for 'token' mode, tokens per group for 'channel' mode
CONFIGS = [
    ("fp16",          "token",   16, 64, 16, 64),
    ("int8-tok",      "token",    8, 64,  8, 64),
    ("int4-tok-g64",  "token",    4, 64,  4, 64),
    ("int4-tok-g32",  "token",    4, 32,  4, 32),
    ("k4ch-v4tok-g32", "channel", 4, 32,  4, 32),
    ("k2tok-v2tok-g32", "token",  2, 32,  2, 32),
    ("k2ch-v2tok-g32", "channel", 2, 32,  2, 32),   # KIVI-style
    ("k4ch-v2tok-g32", "channel", 4, 32,  2, 32),
]
CFG = {c[0]: c for c in CONFIGS}


def load(path, layers, heads):
    z = np.load(path)
    q, k, v = z["q"], z["k"], z["v"]
    L, Hkv, S, D = k.shape
    Hq = q.shape[1]
    n_rep = Hq // Hkv
    if layers is None:
        layers = sorted(set(np.linspace(0, L - 1, 4).round().astype(int).tolist()))
    hs = list(range(Hkv)) if heads is None else heads
    return z, q, k, v, layers, hs, n_rep, S, D


def qk_of(q, k, v, l, h, n_rep):
    return q[l, h * n_rep:(h + 1) * n_rep], k[l, h], v[l, h]   # [G,S,D], [S,D], [S,D]


def cmd_check(a):
    z, q, k, v, layers, hs, n_rep, S, D = load(a.trace, [0, 1], None)
    o = z["o"]
    for l in layers:
        worst = 0.0
        for h in hs:
            qq, kk, vv = qk_of(q, k, v, l, h, n_rep)
            ref = o[l, h * n_rep:(h + 1) * n_rep]
            worst = max(worst, float(np.abs(kg.attn_causal(qq, kk, vv) - ref).max()))
        print(f"layer {l}: max |recomputed attention - model attention| = {worst:.2e}")
        assert worst < 1e-3, "trace does not reproduce the model - fix before going on"
    print("TRACE OK: q/k/v are a faithful copy of what the model used")


def cmd_quant(a):
    z, q, k, v, layers, hs, n_rep, S, D = load(a.trace, a.layers, a.heads)
    print(f"S={S} D={D} layers={layers} kv-heads={hs}")
    acc = {c[0]: [] for c in CONFIGS}
    for l in layers:
        for h in hs:
            qq, kk, vv = qk_of(q, k, v, l, h, n_rep)
            ref = kg.attn_causal(qq, kk, vv)
            for name, km, kb, kgp, vb, vgp in CONFIGS:
                out = kg.attn_causal(qq, kg.fake_quant_k(kk, km, kb, kgp),
                                     kg.fake_quant_v(vv, vb, vgp))
                acc[name].append(kg.compare(out, ref))
    base = kg.kv_bytes_per_token(D, "token", 16, 64, 16, 64)
    rows = []
    print(f"\n{'config':18s} {'B/token':>8s} {'ratio':>6s} {'cos':>9s} {'relL2':>8s}")
    for name, km, kb, kgp, vb, vgp in CONFIGS:
        bpt = kg.kv_bytes_per_token(D, km, kb, kgp, vb, vgp)
        cos, rel = np.mean(acc[name], axis=0)
        rows.append([name, bpt, round(base / bpt, 2), cos, rel])
        print(f"{name:18s} {bpt:8.1f} {base / bpt:5.2f}x {cos:9.5f} {rel:8.4f}")
    with open("results_quant.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["config", "bytes_per_token_per_head", "compression", "cosine", "rel_l2"])
        w.writerows(rows)
    print("\nwrote results_quant.csv")


def cmd_evict(a):
    z, q, k, v, layers, hs, n_rep, S, D = load(a.trace, a.layers, a.heads)
    name, km, kb, kgp, vb, vgp = CFG[a.cfg]
    skip = S // 2                      # score only steps where every budget is evicting
    fracs = [0.5, 0.25, 0.125]
    print(f"S={S} D={D} layers={layers} kv-heads={hs} cfg={name} (scored on steps >= {skip})")
    acc = {}
    for l in layers:
        for h in hs:
            qq, kk, vv = qk_of(q, k, v, l, h, n_rep)
            ref = kg.attn_causal(qq, kk, vv)
            kq, vq = kg.fake_quant_k(kk, km, kb, kgp), kg.fake_quant_v(vv, vb, vgp)
            for pol in ("streaming", "h2o"):
                for fr in fracs:
                    out = kg.simulate_decode(qq, kq, vq, pol, int(S * fr))
                    acc.setdefault((pol, fr), []).append(kg.compare(out, ref, skip))
    per_tok = kg.kv_bytes_per_token(D, km, kb, kgp, vb, vgp)
    base = kg.kv_bytes_per_token(D, "token", 16, 64, 16, 64)
    rows = []
    print(f"\n{'policy':10s} {'budget':>7s} {'mem vs fp16-full':>17s} {'cos':>9s} {'relL2':>8s}")
    for (pol, fr), vals in acc.items():
        cos, rel = np.mean(vals, axis=0)
        mem = per_tok * fr / base       # final cache size vs fp16 full cache
        rows.append([pol, fr, mem, cos, rel])
        print(f"{pol:10s} {fr:7.3f} {mem:16.3f}x {cos:9.5f} {rel:8.4f}")
    with open("results_evict.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["policy", "budget_frac", "mem_vs_fp16_full", "cosine", "rel_l2"])
        w.writerows(rows)
    print("\nwrote results_evict.csv")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["check", "quant", "evict"])
    ap.add_argument("--trace", default="traces/qwen05b_2048.npz")
    ap.add_argument("--layers", type=int, nargs="*", default=None)
    ap.add_argument("--heads", type=int, nargs="*", default=None, help="KV-head indices")
    ap.add_argument("--cfg", default="fp16", choices=list(CFG))
    a = ap.parse_args()
    {"check": cmd_check, "quant": cmd_quant, "evict": cmd_evict}[a.cmd](a)
