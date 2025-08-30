import os
import time
import argparse
import torch

from nsa.core.nsa_attention import GateMLP


def bench_gate(B, S, G, h, Dk, Dv, iters=200, device=torch.device("cuda")):
    q_gp = torch.randn(B, S, G, Dk, device=device)
    O_cmp = torch.randn(B, S, G, h, Dv, device=device)
    O_sel = torch.randn(B, S, G, h, Dv, device=device)
    O_win = torch.randn(B, S, G, h, Dv, device=device)
    gate = GateMLP(Dk).to(device)
    tau = 1.0

    def plain():
        # Flatten per [B*S*G, Dk]
        g = gate(q_gp.reshape(B * S * G, Dk), tau=tau).view(B, S, G, 3)
        w_cmp = g[..., 0:1].unsqueeze(3)
        w_sel = g[..., 1:2].unsqueeze(3)
        w_win = g[..., 2:3].unsqueeze(3)
        return w_cmp * O_cmp + w_sel * O_sel + w_win * O_win

    # Warmup
    plain()
    if device.type == "cuda":
        torch.cuda.synchronize()

    t0 = time.time()
    for _ in range(iters):
        plain()
    if device.type == "cuda":
        torch.cuda.synchronize()
    t1 = time.time()

    print(f"gate+combine plain: {(t1 - t0)/iters*1e3:.3f} ms per iter")


def main():
    ap = argparse.ArgumentParser(description="Benchmark gate+combine path")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--iters", type=int, default=200)
    args = ap.parse_args()
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device)
    if isinstance(device, str):
        device = torch.device(device)
    bench_gate(B=2, S=128, G=2, h=4, Dk=64, Dv=64, iters=args.iters, device=device)


if __name__ == "__main__":
    main()

