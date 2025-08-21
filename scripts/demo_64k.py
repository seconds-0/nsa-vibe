#!/usr/bin/env python3
import os
import time
import argparse
import torch

from nsa.core.nsa_attention import NSAAttention
from nsa.cache.kv_cache import NSA_KV
from nsa.core.block_index import build_block_meta


def create_empty_kv(B: int, G: int, d_k: int, d_v: int, meta) -> NSA_KV:
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
        meta=meta,
        reads_pred=torch.zeros((0,), dtype=torch.int64, device=device),
        reads_act_total=torch.zeros((0,), dtype=torch.int64, device=device),
        reads_act_sel=torch.zeros((0,), dtype=torch.int64, device=device),
        reads_act_cmp=torch.zeros((0,), dtype=torch.int64, device=device),
        reads_act_win=torch.zeros((0,), dtype=torch.int64, device=device),
    )


def main():
    ap = argparse.ArgumentParser(description="NSA 64k demo (chunked prefill + RoPE scaling)")
    ap.add_argument("--S", type=int, default=65536, help="Context length to prefill")
    ap.add_argument("--B", type=int, default=1)
    ap.add_argument("--dim", type=int, default=512)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--groups", type=int, default=2)
    ap.add_argument("--dk", type=int, default=64)
    ap.add_argument("--dv", type=int, default=64)
    ap.add_argument("--l", type=int, default=32)
    ap.add_argument("--d", type=int, default=16)
    ap.add_argument("--l_sel", type=int, default=64)
    ap.add_argument("--n_sel", type=int, default=8)
    ap.add_argument("--w", type=int, default=4096)
    ap.add_argument("--rope_scale", type=float, default=8.0)
    ap.add_argument("--prefill_tile", type=int, default=4096, help="Enable chunked prefill via decode path when >0")
    ap.add_argument("--use_fa2", type=int, default=1)
    args = ap.parse_args()

    # Routing/env knobs
    if args.use_fa2:
        os.environ["NSA_USE_FA2"] = "1"
    os.environ["NSA_ROPE_SCALE"] = str(args.rope_scale)
    os.environ["NSA_PREFILL_TILE"] = str(args.prefill_tile)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)

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
    ).to(device)

    # Synthetic input (replace with tokenized real text for a fuller demo)
    # Memory estimate and warning
    bytes_est = args.B * args.S * args.dim * 4  # float32
    if bytes_est > 3 * (1024**3):  # >3GB for input alone
        print(f"[warn] Large allocation: ~{bytes_est/1024**3:.1f} GB for x_ctx; consider reducing B/S/dim")
    x_ctx = torch.randn(args.B, args.S, args.dim, device=device)
    meta = build_block_meta(args.S + args.w, args.l, args.d, args.l_sel, n_sel=args.n_sel, w=args.w)
    kv = create_empty_kv(args.B, nsa.n_kv_groups, nsa.d_k, nsa.d_v, meta)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        _out, kv = nsa(x_ctx, kv, prefill=True)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) * 1000.0

    reads_total = int(kv.reads_act_total[-1].item()) if kv.reads_act_total.numel() else 0
    print("-- demo_64k summary --")
    print(f"device: {device}")
    print(f"B={args.B} S={args.S} dim={args.dim} heads={args.heads} G={args.groups} dk={args.dk} dv={args.dv}")
    print(f"rope_scale={args.rope_scale} prefill_tile={args.prefill_tile} w={args.w} n_sel={args.n_sel}")
    print(f"prefill_time_ms={ms:.2f} reads_total={reads_total}")


if __name__ == "__main__":
    main()
