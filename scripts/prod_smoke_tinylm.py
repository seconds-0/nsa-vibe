#!/usr/bin/env python3
import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn


def parse_args():
    p = argparse.ArgumentParser(
        description="Production-like 12L TinyLM smoke (forward+backward) with telemetry"
    )
    p.add_argument("--vocab", type=int, default=50257)
    p.add_argument("--dim", type=int, default=768)
    p.add_argument("--layers", type=int, default=12)
    p.add_argument("--heads", type=int, default=12)
    p.add_argument("--groups", type=int, default=12)
    p.add_argument("--d-k", type=int, default=64)
    p.add_argument("--d-v", type=int, default=64)
    p.add_argument("--l", type=int, default=16)
    p.add_argument("--d", type=int, default=16)
    p.add_argument("--l-sel", type=int, default=64)
    p.add_argument("--n-sel", type=int, default=16)
    p.add_argument("--w", type=int, default=512)
    p.add_argument("--seq-len", type=int, default=2048)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16")
    p.add_argument(
        "--grad-checkpointing", action="store_true", help="Enable gradient checkpointing"
    )
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--out-dir", type=str, default="artifacts/prod_smoke")
    p.add_argument("--tag", type=str, default="")
    p.add_argument("--anomaly", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    assert torch.cuda.is_available(), "CUDA required"
    if args.anomaly:
        torch.autograd.set_detect_anomaly(True)
    device = torch.device("cuda")

    from scripts.train_showcase import TinyLM as _TinyLM

    model = _TinyLM(
        vocab=args.vocab,
        dim=args.dim,
        n_layers=args.layers,
        n_heads=args.heads,
        n_kv_groups=args.groups,
        d_k=args.d_k,
        d_v=args.d_v,
        l=args.l,
        d=args.d,
        l_sel=args.l_sel,
        n_sel=args.n_sel,
        w=args.w,
        grad_checkpointing=args.grad_checkpointing,
    ).to(device)

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    model = model.to(dtype=dtype)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    B, S = args.batch, args.seq_len
    out_root = Path(args.out_dir)
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    tag = f"_{args.tag}" if args.tag else ""
    out_dir = out_root / f"run_{ts}{tag}"
    out_dir.mkdir(parents=True, exist_ok=True)
    hb_path = out_dir / "heartbeat.jsonl"

    def mem():
        torch.cuda.synchronize()
        return {
            "alloc_mib": int(torch.cuda.memory_allocated() // (1024 * 1024)),
            "reserved_mib": int(torch.cuda.memory_reserved() // (1024 * 1024)),
        }

    def write_hb(step: int, msg: str, extra=None):
        payload = {
            "ts": time.time(),
            "iso": datetime.utcnow().isoformat() + "Z",
            "step": step,
            "msg": msg,
        }
        payload.update(mem())
        if extra:
            payload.update(extra)
        with open(hb_path, "a") as f:
            f.write(json.dumps(payload) + "\n")

    # Save env
    with open(out_dir / "env.json", "w") as f:
        json.dump(
            {
                k: v
                for k, v in os.environ.items()
                if k.startswith("NSA_") or k in ("PYTORCH_CUDA_ALLOC_CONF",)
            },
            f,
            indent=2,
        )

    # Synthetic data
    torch.manual_seed(42)
    x = torch.randint(0, args.vocab, (B, S), device=device)
    target = torch.randn(B, S, args.dim, device=device, dtype=dtype)
    loss_fn = nn.MSELoss()

    write_hb(0, "start")
    t0 = time.time()
    for step in range(1, args.steps + 1):
        opt.zero_grad(set_to_none=True)
        y = model(x)
        # Ensure dtype match for loss
        if y.dtype != dtype:
            y = y.to(dtype)
        loss = loss_fn(y, target)
        loss.backward()
        opt.step()
        if step % 5 == 0 or step == 1:
            write_hb(step, "step", {"loss": float(loss.detach().to(torch.float32).item())})
    dt = time.time() - t0
    write_hb(args.steps, "done", {"seconds": dt})
    print(f"Smoke completed: steps={args.steps} time={dt:.2f}s mem={mem()}")


if __name__ == "__main__":
    main()
