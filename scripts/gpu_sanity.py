#!/usr/bin/env python3
import json
import os

import torch

from nsa.cache.kv_cache import NSA_KV
from nsa.core.block_index import build_block_meta
from nsa.core.flags import execution_routing_summary
from nsa.model.llama_block_nsa import LlamaBlockNSA


def make_empty_kv(B: int, G: int, d_k: int, d_v: int, meta) -> NSA_KV:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    zeros_k = torch.zeros((B, G, 0, d_k), device=device)
    zeros_v = torch.zeros((B, G, 0, d_v), device=device)
    return NSA_KV(
        K_sel=zeros_k.clone(),
        V_sel=zeros_v.clone(),
        K_win=zeros_k.clone(),
        V_win=zeros_v.clone(),
        K_cmp_raw_seq=zeros_k.clone(),
        V_cmp_raw_seq=zeros_v.clone(),
        K_cmp=zeros_k.clone(),
        V_cmp=zeros_v.clone(),
        win_ptr=torch.zeros((B, G), dtype=torch.int64, device=device),
        cmp_emit_next=torch.zeros((B, G), dtype=torch.int64, device=device),
        reads_pred=torch.zeros((0,), dtype=torch.int64, device=device),
        reads_act_total=torch.zeros((0,), dtype=torch.int64, device=device),
        reads_act_sel=torch.zeros((0,), dtype=torch.int64, device=device),
        reads_act_cmp=torch.zeros((0,), dtype=torch.int64, device=device),
        reads_act_win=torch.zeros((0,), dtype=torch.int64, device=device),
        meta=meta,
    )


def main():
    # Print routing snapshot
    print(json.dumps(execution_routing_summary(), indent=2))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("[warn] CUDA not available; running CPU sanity only")

    # Tiny model config
    B, S, dim = 1, 8, 64
    n_heads, n_kv_groups, d_k, d_v = 4, 2, 16, 16
    l, d, l_sel, n_sel, w = 8, 4, 16, 4, 8

    block = LlamaBlockNSA(
        dim=dim,
        n_heads=n_heads,
        n_kv_groups=n_kv_groups,
        d_k=d_k,
        d_v=d_v,
        l=l,
        d=d,
        l_sel=l_sel,
        n_sel=n_sel,
        w=w,
    ).to(device)
    x = torch.randn(B, S, dim, device=device)
    # Prefill path
    out = block(x)
    assert out.shape == (B, S, dim)

    # Decode step with branch forcing (checks gate broadcasting and paths)
    meta = build_block_meta(seq_len=S + w, l=l, d=d, l_sel=l_sel, n_sel=n_sel, w=w)
    kv = make_empty_kv(B, n_kv_groups, d_k, d_v, meta)
    with torch.no_grad():
        attn = block.attn
        _, kv = attn(x, kv, prefill=True)
        for fb in ("cmp", "sel", "win"):
            prev = os.environ.get("NSA_FORCE_BRANCH")
            os.environ["NSA_FORCE_BRANCH"] = fb
            try:
                y, kv2 = attn(torch.randn(B, 1, dim, device=device), kv, prefill=False)
                assert y.shape == (B, 1, dim)
            finally:
                if prev is None:
                    os.environ.pop("NSA_FORCE_BRANCH", None)
                else:
                    os.environ["NSA_FORCE_BRANCH"] = prev
            print(f"branch {fb}: OK")

    print("sanity: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
