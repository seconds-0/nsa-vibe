#!/usr/bin/env python3
import argparse
import json

import torch

from nsa.core.block_index import build_block_meta
from nsa.core.compress_pool import avg_pool_phi_rope_kv
from nsa.core.nsa_attention import NSAAttention
from nsa.core.rope import apply_rope
from nsa.core.selection_scorer import (
    compute_pcmp_all,
    map_pcmp_to_pslc_batched,
    select_topn_ranges_batched,
)


def main():
    ap = argparse.ArgumentParser(description="Print selection ranges per step for a toy sequence")
    ap.add_argument("--B", type=int, default=1)
    ap.add_argument("--S", type=int, default=32)
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--groups", type=int, default=2)
    ap.add_argument("--dk", type=int, default=16)
    ap.add_argument("--dv", type=int, default=16)
    ap.add_argument("--l", type=int, default=8)
    ap.add_argument("--d", type=int, default=4)
    ap.add_argument("--l_sel", type=int, default=16)
    ap.add_argument("--n_sel", type=int, default=4)
    ap.add_argument("--w", type=int, default=16)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--json", action="store_true", help="Emit JSON lines per step")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cpu")

    # Build attention module (we won't call forward; we'll compute Q/K internally)
    nsa = NSAAttention(
        dim=args.dim,
        n_heads=args.heads,
        n_kv_groups=args.groups,
        d_k=args.dk,
        d_v=args.dv,
        l=args.l,
        d=args.d,
        l_sel=args.l_sel,
        n_sel=args.n_sel,
        w=args.w,
    )
    B, S = args.B, args.S
    x = torch.randn(B, S, args.dim, device=device)

    # Projections (match prefill path)
    Q_lin = nsa._shape_q(nsa.W_Q(x), B, S)
    pos = torch.arange(S, device=device)
    Q = apply_rope(Q_lin.view(B, S, nsa.n_heads, nsa.d_k).reshape(B, S, nsa.n_heads * nsa.d_k), pos)
    Q = Q.view(B, S, nsa.n_heads, nsa.d_k).view(B, S, nsa.n_kv_groups, nsa.h_per_group, nsa.d_k)
    K_cmp_raw = nsa._shape_kv(nsa.W_K_cmp(x), B, S)
    V_cmp_raw = nsa._shape_kv(nsa.W_V_cmp(x), B, S)
    # Build compressed via avg pool ϕ with RoPE on K
    K_cmp, V_cmp = avg_pool_phi_rope_kv(
        K_cmp_raw, V_cmp_raw, nsa.l, nsa.d, pos=torch.arange(S, device=device)
    )
    meta = build_block_meta(seq_len=S, l=nsa.l, d=nsa.d, l_sel=nsa.l_sel, n_sel=nsa.n_sel, w=nsa.w)

    scale = 1.0 / (nsa.d_k**0.5)
    p_cmp_all = compute_pcmp_all(Q, K_cmp, scale)
    p_slc_all = map_pcmp_to_pslc_batched(p_cmp_all, meta)
    p_grp_all = p_slc_all.sum(dim=3)  # [B,S,G,S_sel]
    ranges = select_topn_ranges_batched(p_grp_all, meta, nsa.n_sel, S)  # [B,S,G,n,2]

    # Print
    for t in range(S):
        row = {
            "t": t,
            "ranges": ranges[0, t].cpu().tolist(),  # [G,n,2]
        }
        if args.json:
            print(json.dumps(row))
        else:
            pretty = ", ".join(
                f"g{g}:" + ";".join(f"[{s},{e})" for s, e in ranges[0, t, g].tolist())
                for g in range(args.groups)
            )
            print(f"t={t}: {pretty}")


if __name__ == "__main__":
    main()
