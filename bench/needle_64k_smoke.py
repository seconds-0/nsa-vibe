#!/usr/bin/env python3
"""
Long-context needle-in-haystack smoke harness.

This does not train. It constructs an ideal selection range containing the needle
and verifies the selection attention path can retrieve the value at that position
at very long context (default 65,536).

Usage examples:
  PYTHONPATH=. python bench/needle_64k_smoke.py --S 65536 --G 4 --h 4 --D 64 --Dv 64
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python bench/needle_64k_smoke.py --device cuda --S 65536
"""
import argparse
import math
import time

import torch

from nsa.core.attention_kernels import grouped_selection_attention


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--S", type=int, default=65536, help="sequence length (KV)")
    p.add_argument("--B", type=int, default=1)
    p.add_argument("--G", type=int, default=4)
    p.add_argument("--h", type=int, default=4)
    p.add_argument("--D", type=int, default=64)
    p.add_argument("--Dv", type=int, default=64)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--pos", type=int, default=None, help="needle position; default S//2")
    args = p.parse_args()
    device = torch.device(args.device)

    B, S, G, h, D, Dv = args.B, args.S, args.G, args.h, args.D, args.Dv
    pos = args.pos if args.pos is not None else S // 2
    assert 0 <= pos < S

    torch.manual_seed(0)
    K = torch.randn(B, G, S, D, device=device)
    V = torch.randn(B, G, S, Dv, device=device)

    # Make a distinctive needle value and aligned query for each head
    needle_v = torch.randn(B, G, Dv, device=device)
    V[:, :, pos] = needle_v
    Q = torch.zeros(B, 1, G, h, D, device=device)
    # Set Q per head equal to the K at the needle position for strong match
    for b in range(B):
        for g in range(G):
            for i in range(h):
                Q[b, 0, g, i] = K[b, g, pos]

    # Ideal selection: a single-range bucket exactly at the needle
    ranges = torch.zeros(B, 1, G, 1, 2, dtype=torch.int32, device=device)
    ranges[..., 0, 0] = pos
    ranges[..., 0, 1] = pos + 1

    t0 = time.perf_counter()
    O = grouped_selection_attention(Q, K, V, ranges)  # [B,1,G,h,Dv]
    dt_ms = (time.perf_counter() - t0) * 1e3
    # Average across heads to compare with needle_v
    O_mean = O[:, 0].mean(dim=2)  # [B,G,Dv]
    cos = torch.nn.functional.cosine_similarity(O_mean.flatten(1), needle_v.flatten(1))
    mae = (O_mean - needle_v).abs().mean()
    print({
        "S": S, "pos": pos, "cos": float(cos.mean().item()), "mae": float(mae.item()), "time_ms": dt_ms,
    })


if __name__ == "__main__":
    main()

