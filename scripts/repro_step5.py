#!/usr/bin/env python3
"""
Minimal NSA block repro: repeat forward+backward N times to probe step-5 stall.

Usage:
  python scripts/repro_step5.py --steps 6 --seq-len 1024 --dim 256 --layers 1

Notes:
  - Uses TinyLM with 1 LlamaBlockNSA by default to isolate NSA core.
  - Synthetic byte-token data; optimizer AdamW; fp16/bf16 via --precision.
  - Respects key NSA_* env toggles (e.g., NSA_PREFILL_BATCHED, SDPA flags).
"""

import argparse
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from nsa.model.llama_block_nsa import LlamaBlockNSA, RMSNorm


class TinyOneBlock(nn.Module):
    def __init__(
        self,
        vocab: int,
        dim: int,
        n_layers: int,
        n_heads: int,
        n_kv_groups: int,
        d_k: int,
        d_v: int,
        l: int,
        d: int,
        l_sel: int,
        n_sel: int,
        w: int,
    ):
        super().__init__()
        self.embed = nn.Embedding(vocab, dim)
        self.blocks = nn.ModuleList(
            [
                LlamaBlockNSA(
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
                )
                for _ in range(n_layers)
            ]
        )
        self.norm_f = RMSNorm(dim)
        self.lm_head = nn.Linear(dim, vocab, bias=False)

    def forward(self, x_tok: torch.Tensor) -> torch.Tensor:
        x = self.embed(x_tok)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm_f(x)
        return self.lm_head(x)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=6)
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--dim", type=int, default=256)
    ap.add_argument("--layers", type=int, default=1)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--kv-groups", type=int, default=2)
    ap.add_argument("--d-k", type=int, default=32)
    ap.add_argument("--d-v", type=int, default=32)
    ap.add_argument("--l", type=int, default=16)
    ap.add_argument("--d", type=int, default=8)
    ap.add_argument("--l-sel", type=int, default=32)
    ap.add_argument("--n-sel", type=int, default=8)
    ap.add_argument("--w", type=int, default=256)
    ap.add_argument("--precision", type=str, default=os.getenv("NSA_DTYPE", "fp16"))
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    if args.precision.lower() in ("bf16", "bfloat16"):
        dtype = torch.bfloat16
    elif args.precision.lower() in ("fp16", "float16"):
        dtype = torch.float16

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    vocab = 256
    model = TinyOneBlock(
        vocab,
        args.dim,
        args.layers,
        args.heads,
        args.kv_groups,
        args.d_k,
        args.d_v,
        args.l,
        args.d,
        args.l_sel,
        args.n_sel,
        args.w,
    ).to(device)
    if dtype != torch.float32:
        model = model.to(dtype=dtype)
    model.train()
    loss_fn = nn.CrossEntropyLoss()
    opt = optim.AdamW(model.parameters(), lr=2e-4)

    S = int(args.seq_len)
    for step in range(1, args.steps + 1):
        x = torch.randint(low=0, high=vocab, size=(1, S), device=device)
        y = x[:, 1:].contiguous()
        logits = model(x)
        loss = loss_fn(logits[:, :-1, :].contiguous().view(1 * (S - 1), vocab), y.view(1 * (S - 1)))
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        print(f"[repro] step {step} ok | loss {float(loss):.4f}")
    print("[repro] completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
