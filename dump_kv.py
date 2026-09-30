"""
dump_kv.py - run a small HF model once and save real Q/K/V traces.

    pip install torch transformers datasets numpy
    python dump_kv.py --model Qwen/Qwen2.5-0.5B --seq 2048 --out traces/qwen05b_2048.npz

Saved arrays (float32):
    q : [L, Hq,  S, D]   post-RoPE queries
    k : [L, Hkv, S, D]   post-RoPE keys  (exactly what a KV cache stores)
    v : [L, Hkv, S, D]   values
    o : [L, Hq,  S, D]   attention output before o_proj (used to VALIDATE the trace)

NOTE: written against Llama/Qwen2-style models in recent `transformers`.
It monkeypatches `apply_rotary_pos_emb` in the model's modeling file. If a
newer version moves that function, the assert at the end will tell you.
"""
import argparse
import importlib
import os

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--seq", type=int, default=2048)
    ap.add_argument("--out", default="traces/qwen05b_2048.npz")
    a = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(
        a.model, torch_dtype=torch.float32, attn_implementation="eager"
    ).eval()
    cfg = model.config
    Hq, Hkv = cfg.num_attention_heads, cfg.num_key_value_heads
    D = getattr(cfg, "head_dim", None) or cfg.hidden_size // Hq
    print(f"model={a.model}  layers={cfg.num_hidden_layers}  Hq={Hq}  Hkv={Hkv}  D={D}")

    mod = importlib.import_module(
        f"transformers.models.{cfg.model_type}.modeling_{cfg.model_type}"
    )
    qs, ks, vs, os_ = [], [], [], []

    orig = mod.apply_rotary_pos_emb

    def patched(q, k, cos, sin, *args, **kw):
        qe, ke = orig(q, k, cos, sin, *args, **kw)
        qs.append(qe[0].detach().cpu().numpy())      # [Hq,  S, D]
        ks.append(ke[0].detach().cpu().numpy())      # [Hkv, S, D]
        return qe, ke

    mod.apply_rotary_pos_emb = patched
    for layer in model.model.layers:
        attn = layer.self_attn
        attn.v_proj.register_forward_hook(
            lambda m, i, o: vs.append(o[0].detach().cpu().numpy())        # [S, Hkv*D]
        )
        attn.o_proj.register_forward_pre_hook(
            lambda m, i: os_.append(i[0][0].detach().cpu().numpy())       # [S, Hq*D]
        )

    text = "\n\n".join(load_dataset("wikitext", "wikitext-2-raw-v1", split="test")["text"])
    ids = tok(text[: a.seq * 8], return_tensors="pt").input_ids[:, : a.seq]
    S = ids.shape[1]
    assert S == a.seq, f"only {S} tokens available; lower --seq"
    with torch.no_grad():
        model(ids)

    L = cfg.num_hidden_layers
    assert len(qs) == len(ks) == len(vs) == len(os_) == L, (
        f"captured {len(qs)} q / {len(vs)} v for {L} layers - patch point moved?"
    )
    q = np.stack(qs)
    k = np.stack(ks)
    v = np.stack([x.reshape(S, Hkv, D).transpose(1, 0, 2) for x in vs])
    o = np.stack([x.reshape(S, Hq, D).transpose(1, 0, 2) for x in os_])

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    np.savez(a.out, q=q, k=k, v=v, o=o)
    print("saved", a.out, {n: x.shape for n, x in dict(q=q, k=k, v=v, o=o).items()})


if __name__ == "__main__":
    main()
