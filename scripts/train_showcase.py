#!/usr/bin/env python3
import json
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from omegaconf import OmegaConf

from nsa.model.llama_block_nsa import LlamaBlockNSA


class TinyLM(nn.Module):
    def __init__(
        self,
        vocab: int,
        dim: int,
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
        self.block = LlamaBlockNSA(
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
        self.lm_head = nn.Linear(dim, vocab, bias=False)

    def forward(self, x_tok: torch.Tensor) -> torch.Tensor:
        x = self.embed(x_tok)  # [B,S,dim]
        x = self.block(x)
        return self.lm_head(x)


def set_seed(seed: int):
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    cfg_path = os.environ.get("CONFIG", "configs/train_showcase.yaml")
    cfg = OmegaConf.load(cfg_path)
    os.makedirs(cfg.train.out_dir, exist_ok=True)

    set_seed(int(cfg.train.seed))
    device = (
        torch.device("cuda")
        if (
            cfg.runtime.device == "cuda"
            or (cfg.runtime.device == "auto" and torch.cuda.is_available())
        )
        else torch.device("cpu")
    )
    dtype = torch.float32
    if str(cfg.runtime.precision).lower() in ("bf16", "bfloat16"):
        dtype = torch.bfloat16
    elif str(cfg.runtime.precision).lower() in ("fp16", "float16"):
        dtype = torch.float16

    vocab = 1024
    model = TinyLM(
        vocab=vocab,
        dim=int(cfg.model.dim),
        n_heads=int(cfg.model.n_heads),
        n_kv_groups=int(cfg.model.n_kv_groups),
        d_k=int(cfg.model.d_k),
        d_v=int(cfg.model.d_v),
        l=int(cfg.nsa.l),
        d=int(cfg.nsa.d),
        l_sel=int(cfg.nsa.l_sel),
        n_sel=int(cfg.nsa.n_sel),
        w=int(cfg.nsa.w),
    ).to(device)
    model.train()
    if dtype != torch.float32:
        model = model.to(dtype=dtype)

    steps = int(cfg.train.steps)
    S = int(cfg.train.seq_len)
    B = int(cfg.train.batch_size)
    lr = float(cfg.train.lr)

    opt = optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    losses = []
    for step in range(1, steps + 1):
        # synthetic next-token prediction on random tokens
        x = torch.randint(low=0, high=vocab, size=(B, S), device=device)
        y = x.clone()  # predict identity next-token for sanity
        logits = model(x)  # [B,S,V]
        loss = loss_fn(logits.view(B * S, vocab), y.view(B * S))

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        losses.append(loss.detach().float().item())
        if step % int(cfg.train.log_every) == 0 or step == 1:
            print(f"step {step:04d} | loss {loss.item():.4f}")

    # Save simple artifacts
    meta = {
        "device": str(device),
        "dtype": str(dtype),
        "steps": steps,
        "seq_len": S,
        "batch": B,
        "lr": lr,
        "loss_first": float(losses[0]) if losses else None,
        "loss_last": float(losses[-1]) if losses else None,
    }
    with open(Path(cfg.train.out_dir) / "metrics.json", "w") as f:
        json.dump(meta, f, indent=2)
    with open(Path(cfg.train.out_dir) / "loss.txt", "w") as f:
        for v in losses:
            f.write(f"{v}\n")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
