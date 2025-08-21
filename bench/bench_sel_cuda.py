#!/usr/bin/env python3
import argparse
import time

import torch

from nsa.core.attention_kernels import grouped_selection_attention_packed
from nsa.kernels.cuda_sel_kernel import selection_attention_cuda


def make_data(N, H, D, Dv, L, device):
    # Build a batch with B=N rows, S=1, G=1, h=H
    Q = torch.randn(N, 1, 1, H, D, device=device)
    K = torch.randn(N, 1, L, D, device=device)
    V = torch.randn(N, 1, L, Dv, device=device)
    ranges = torch.zeros(N, 1, 1, 1, 2, dtype=torch.int32, device=device)
    ranges[:, 0, 0, 0, 0] = 0
    ranges[:, 0, 0, 0, 1] = L
    return Q, K, V, ranges


def bench_once(fn, *args, iters=50, warmup=5):
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    t0 = time.perf_counter()
    for _ in range(iters):
        out = fn(*args)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    dt = (time.perf_counter() - t0) * 1e3 / iters
    return dt, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=1024)
    ap.add_argument("--H", type=int, default=8)
    ap.add_argument("--D", type=int, default=128)
    ap.add_argument("--Dv", type=int, default=128)
    ap.add_argument("--L_list", type=str, default="128,256,512")
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    for L in [int(x) for x in args.L_list.split(",")]:
        Q, K, V, ranges = make_data(args.N, args.H, args.D, args.Dv, L, device)
        dt_cuda, O_cuda = bench_once(
            selection_attention_cuda, Q, K, V, ranges, iters=args.iters, warmup=args.warmup
        )
        dt_ref, O_ref = bench_once(
            grouped_selection_attention_packed,
            Q,
            K,
            V,
            ranges,
            iters=args.iters,
            warmup=args.warmup,
        )
        mae = (O_cuda - O_ref).abs().mean().item()
        speedup = dt_ref / max(1e-9, dt_cuda)
        print(
            f"cuda,N={args.N},H={args.H},D={args.D},Dv={args.Dv},L={L},iters={args.iters},cuda_ms={dt_cuda:.3f},ref_ms={dt_ref:.3f},speedup={speedup:.3f}x,mae={mae:.3e}"
        )


if __name__ == "__main__":
    main()
