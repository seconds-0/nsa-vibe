import os
import time
import argparse
import torch

from nsa.core.selection_scorer import compute_pcmp_all


def bench_once(B, S, G, h, Dk, S_cmp, device, iters=50):
    Q = torch.randn(B, S, G, h, Dk, device=device, dtype=torch.float32)
    K = torch.randn(B, G, S_cmp, Dk, device=device, dtype=torch.float32)
    scale = 1.0 / (Dk ** 0.5)

    # Warmup
    compute_pcmp_all(Q, K, scale)
    if device.type == "cuda":
        torch.cuda.synchronize()

    # Baseline precise
    os.environ["NSA_P_CMP_MIXED"] = "0"
    t0 = time.time()
    for _ in range(iters):
        compute_pcmp_all(Q, K, scale)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t1 = time.time()

    # Mixed
    os.environ["NSA_P_CMP_MIXED"] = "1"
    t2 = time.time()
    for _ in range(iters):
        compute_pcmp_all(Q, K, scale)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t3 = time.time()

    base = (t1 - t0) / iters
    mixed = (t3 - t2) / iters
    speedup = base / max(mixed, 1e-9)
    print(
        f"B={B} S={S} G={G} h={h} Dk={Dk} S_cmp={S_cmp} precise {base*1e3:.2f} ms  mixed {mixed*1e3:.2f} ms  x{speedup:.2f}"
    )


def main():
    ap = argparse.ArgumentParser(description="Benchmark p_cmp mixed precision")
    ap.add_argument("--device", default="auto", help="auto|cuda|cpu")
    ap.add_argument("--iters", type=int, default=50)
    args = ap.parse_args()

    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device)
    if isinstance(device, str):
        device = torch.device(device)

    # Sweep a few shapes
    for S, S_cmp in [(128, 16), (256, 32), (512, 64)]:
        bench_once(B=2, S=S, G=2, h=4, Dk=64, S_cmp=S_cmp, device=device, iters=args.iters)


if __name__ == "__main__":
    main()

