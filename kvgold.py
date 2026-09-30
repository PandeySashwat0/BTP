"""
kvgold.py - algorithm-level golden models for KV-cache compression.

This is the *step-1* golden model: float32 math, but with the conventions the
RTL will inherit (asymmetric min-max, fp16 scale/zero-point, round-half-up,
little-end packing, deterministic tie-breaks). In step 2 you add a fixed-point
"RTL-level" model on top of this one and check the two against each other.

Shapes used everywhere (one KV head at a time):
    k, v : [S, D]        cached keys / values (keys are POST-RoPE)
    q    : [G, S, D]     queries of the G query-heads that share this KV head
"""
import numpy as np

# --------------------------------------------------------------------------
# 1. Core group quantizer  (asymmetric min-max, fp16 metadata, round-half-up)
# --------------------------------------------------------------------------
def _quant_last(x, bits):
    """Quantize along the LAST axis; every row of the last axis is one group."""
    x = x.astype(np.float32)
    qmax = (1 << bits) - 1
    mn = x.min(axis=-1, keepdims=True)
    mx = x.max(axis=-1, keepdims=True)
    zp = mn.astype(np.float16)                              # stored as fp16
    scale = ((mx - mn) / qmax).astype(np.float16)           # stored as fp16
    scale = np.where(scale == 0, np.float16(1), scale).astype(np.float16)
    # round-half-up = floor(x + 0.5): cheap in hardware, mirror it in RTL
    codes = np.floor((x - zp.astype(np.float32)) / scale.astype(np.float32) + 0.5)
    codes = np.clip(codes, 0, qmax).astype(np.uint8)        # fp16 scale can overshoot
    return codes, scale, zp


def _dequant_last(codes, scale, zp):
    return codes.astype(np.float32) * scale.astype(np.float32) + zp.astype(np.float32)


# --------------------------------------------------------------------------
# 2. Two quantization layouts
# --------------------------------------------------------------------------
def quant_per_token(x, bits, group):
    """Groups of `group` channels inside ONE token. Simple: no cross-token state."""
    S, D = x.shape
    assert D % group == 0, "group must divide head_dim"
    codes, scale, zp = _quant_last(x.reshape(S, D // group, group), bits)
    return codes.reshape(S, D), scale[..., 0], zp[..., 0]      # meta: [S, D/group]


def dequant_per_token(codes, scale, zp, group):
    S, D = codes.shape
    g = codes.reshape(S, D // group, group)
    return _dequant_last(g, scale[..., None], zp[..., None]).reshape(S, D)


def quant_per_channel(x, bits, group):
    """KIVI-style keys: each channel quantized over `group` consecutive tokens.
    The trailing S % group tokens stay in fp16 (the 'residual buffer')."""
    S, D = x.shape
    Sq = (S // group) * group
    assert Sq > 0, "sequence shorter than one group"
    blk = x[:Sq].reshape(Sq // group, group, D).transpose(0, 2, 1)   # [nblk, D, group]
    codes, scale, zp = _quant_last(blk, bits)
    codes = codes.transpose(0, 2, 1).reshape(Sq, D)
    return codes, scale[..., 0], zp[..., 0], x[Sq:].astype(np.float16)   # meta: [nblk, D]


def dequant_per_channel(codes, scale, zp, residual, group):
    Sq, D = codes.shape
    blk = codes.reshape(Sq // group, group, D).transpose(0, 2, 1)
    x = _dequant_last(blk, scale[..., None], zp[..., None]).transpose(0, 2, 1).reshape(Sq, D)
    return np.concatenate([x, residual.astype(np.float32)], axis=0)


def fake_quant_k(k, mode, bits, group):
    if bits >= 16:
        return k.astype(np.float16).astype(np.float32)
    if mode == "token":
        return dequant_per_token(*quant_per_token(k, bits, group), group)
    if mode == "channel":
        return dequant_per_channel(*quant_per_channel(k, bits, group), group)
    raise ValueError(mode)


def fake_quant_v(v, bits, group):
    if bits >= 16:
        return v.astype(np.float16).astype(np.float32)
    return dequant_per_token(*quant_per_token(v, bits, group), group)


# --------------------------------------------------------------------------
# 3. Bit packing (the reference for your RTL packer/unpacker)
#    Convention: element 0 of each byte sits in the LOW bits.
# --------------------------------------------------------------------------
def pack_codes(codes, bits):
    assert 8 % bits == 0
    per = 8 // bits
    c = codes.reshape(-1, per).astype(np.uint16)
    shifts = (np.arange(per) * bits).astype(np.uint16)
    return (c << shifts).sum(axis=1).astype(np.uint8)


def unpack_codes(packed, bits, n):
    per = 8 // bits
    mask = (1 << bits) - 1
    shifts = np.arange(per) * bits
    out = (packed[:, None].astype(np.uint16) >> shifts) & mask
    return out.astype(np.uint8).reshape(-1)[:n]


# --------------------------------------------------------------------------
# 4. Storage accounting (bytes per token, per KV head)
# --------------------------------------------------------------------------
def _side_bytes(D, mode, bits, group):
    if bits >= 16:
        return 2.0 * D
    codes = D * bits / 8
    meta = (D / group) * 4 if mode == "token" else D * 4 / group   # fp16 scale + fp16 zp
    return codes + meta


def kv_bytes_per_token(D, k_mode, k_bits, k_group, v_bits, v_group):
    return _side_bytes(D, k_mode, k_bits, k_group) + _side_bytes(D, "token", v_bits, v_group)


# --------------------------------------------------------------------------
# 5. Attention (reference + eviction simulation)
# --------------------------------------------------------------------------
def attn_causal(q, k, v):
    """Full causal attention. q [G,S,D], k,v [S,D] -> [G,S,D]."""
    S, D = k.shape
    s = np.einsum("gsd,td->gst", q, k) / np.sqrt(D)
    s = np.where(np.triu(np.ones((S, S), bool), 1), -np.inf, s)
    s = s - s.max(-1, keepdims=True)
    p = np.exp(s)
    p /= p.sum(-1, keepdims=True)
    return np.einsum("gst,td->gsd", p, v).astype(np.float32)


def simulate_decode(q, k, v, policy, budget, n_sink=4):
    """Generate token by token from scratch with a cache of at most `budget`.

    policy = 'full'      : never evict
             'streaming' : keep first n_sink tokens + most recent window
             'h2o'       : protect budget//2 recent tokens, evict the cached
                           token with the lowest accumulated attention score
    Ties in H2O go to the OLDEST token (np.argmin picks the first) - pick a
    tie-break rule now and mirror it in RTL.
    Returns outputs [G,S,D].
    """
    G, S, D = q.shape
    out = np.zeros((G, S, D), np.float32)
    alive, score = [], np.zeros(S, np.float64)
    inv = 1.0 / np.sqrt(D)
    recent = max(1, budget // 2)
    for t in range(S):
        alive.append(t)
        idx = np.asarray(alive)
        s = (q[:, t] @ k[idx].T) * inv
        s -= s.max(-1, keepdims=True)
        p = np.exp(s)
        p /= p.sum(-1, keepdims=True)
        out[:, t] = p @ v[idx]
        if policy == "h2o":
            score[idx] += p.sum(0)                 # aggregate over the GQA group
        if policy != "full" and len(alive) > budget:
            if policy == "streaming":
                victim = n_sink                    # oldest non-sink token
            elif policy == "h2o":
                cand = idx[: len(idx) - recent]    # everything except recent window
                victim = int(np.argmin(score[cand]))
            else:
                raise ValueError(policy)
            alive.pop(victim)
    return out


# --------------------------------------------------------------------------
# 6. Metrics
# --------------------------------------------------------------------------
def compare(out, ref, skip=0):
    """Mean cosine similarity and mean relative L2 error of attention outputs."""
    o = out[:, skip:].reshape(-1, out.shape[-1]).astype(np.float64)
    r = ref[:, skip:].reshape(-1, ref.shape[-1]).astype(np.float64)
    no, nr = np.linalg.norm(o, axis=-1), np.linalg.norm(r, axis=-1)
    cos = (o * r).sum(-1) / (no * nr + 1e-12)
    rel = np.linalg.norm(o - r, axis=-1) / (nr + 1e-12)
    return float(cos.mean()), float(rel.mean())


# --------------------------------------------------------------------------
# Self-test:  python kvgold.py
# --------------------------------------------------------------------------
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    S, D, G = 200, 64, 4
    k = rng.normal(size=(S, D)).astype(np.float32)
    k[:, [3, 17]] *= 20                              # outlier channels, like real keys
    v = rng.normal(size=(S, D)).astype(np.float32)
    q = rng.normal(size=(G, S, D)).astype(np.float32)

    # pack/unpack round trip
    for b in (2, 4, 8):
        c = rng.integers(0, 1 << b, size=4096).astype(np.uint8)
        assert np.array_equal(unpack_codes(pack_codes(c, b), b, c.size), c)
    print("pack/unpack round-trip: OK")

    # more bits => less error, both layouts
    prev = {"token": 9, "channel": 9}
    for b in (2, 4, 8):
        for mode in ("token", "channel"):
            err = np.abs(fake_quant_k(k, mode, b, 32) - k).mean()
            assert err < prev[mode]
            prev[mode] = err
    print("error decreases with bit-width: OK")

    # per-channel beats per-token on outlier-channel keys at 2 bits
    e_tok = np.abs(fake_quant_k(k, "token", 2, 32) - k).mean()
    e_ch = np.abs(fake_quant_k(k, "channel", 2, 32) - k).mean()
    print(f"2-bit key error  per-token={e_tok:.3f}  per-channel={e_ch:.3f}")
    assert e_ch < e_tok

    # eviction sanity
    ref = attn_causal(q, k, v)
    assert compare(simulate_decode(q, k, v, "full", S), ref)[0] > 0.999999
    assert compare(simulate_decode(q, k, v, "h2o", S), ref)[0] > 0.999999
    assert compare(simulate_decode(q, k, v, "streaming", S), ref)[0] > 0.999999
    print("budget >= S reproduces full attention: OK")

    print("bytes/token fp16      :", kv_bytes_per_token(D, "token", 16, 64, 16, 64))
    print("bytes/token int4 g32  :", kv_bytes_per_token(D, "token", 4, 32, 4, 32))
    print("bytes/token K2ch V2tok:", kv_bytes_per_token(D, "channel", 2, 32, 2, 32))
    print("ALL SELF-TESTS PASSED")
