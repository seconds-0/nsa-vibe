import math
import os
import random
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast

from nsa.model.llama_block_nsa import LlamaBlockNSA


@dataclass
class TrainConfig:
    seed: int = 1337
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dim: int = 256
    n_heads: int = 8
    n_kv_groups: int = 2
    d_k: int = 64
    d_v: int = 64
    l: int = 32
    d: int = 16
    l_sel: int = 64
    n_sel: int = 16
    w: int = 128
    lr: float = 3e-4
    warmup_steps: int = 50
    max_grad_norm: float = 1.0
    batch_size: int = 8
    max_len: int = 128
    steps: int = 200
    use_amp: bool = True


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(False)


def synthetic_batch(cfg: TrainConfig):
    # Produce a batch of variable-length token ids and attention mask
    B = cfg.batch_size
    lengths = torch.randint(low=8, high=cfg.max_len + 1, size=(B,))
    S_max = int(lengths.max().item())
    # Toy embedding space: just float inputs; labels are next-token ids modulo vocab
    vocab = 1024
    x = torch.randn(B, S_max, cfg.dim)
    labels = torch.randint(0, vocab, (B, S_max))
    # Create mask to ignore pads; last token has no next-label
    mask = torch.zeros(B, S_max, dtype=torch.bool)
    for b in range(B):
        mask[b, : lengths[b]] = True
    mask[:, -1] = False
    return x, labels, mask


def masked_ce(logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    # logits: [B,S,V]; labels: [B,S]; mask: [B,S]
    logits = logits.reshape(-1, logits.shape[-1])
    labels = labels.reshape(-1)
    mask = mask.reshape(-1)
    loss = (
        nn.functional.cross_entropy(logits[mask], labels[mask]) if mask.any() else logits.sum() * 0
    )
    return loss


class ToyHead(nn.Module):
    def __init__(self, dim: int, vocab: int) -> None:
        super().__init__()
        self.proj = nn.Linear(dim, vocab)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


def main():
    cfg = TrainConfig()
    # Env overrides for quick tuning
    cfg.seed = int(os.getenv("NSA_TRAIN_SEED", str(cfg.seed)))
    cfg.lr = float(os.getenv("NSA_TRAIN_LR", str(cfg.lr)))
    cfg.warmup_steps = int(os.getenv("NSA_TRAIN_WARMUP", str(cfg.warmup_steps)))
    cfg.max_grad_norm = float(os.getenv("NSA_TRAIN_CLIP", str(cfg.max_grad_norm)))
    cfg.batch_size = int(os.getenv("NSA_TRAIN_BS", str(cfg.batch_size)))
    cfg.max_len = int(os.getenv("NSA_TRAIN_MAX_LEN", str(cfg.max_len)))
    cfg.steps = int(os.getenv("NSA_TRAIN_STEPS", str(cfg.steps)))
    set_seed(cfg.seed)
    device = torch.device(cfg.device)
    model = nn.Sequential(
        LlamaBlockNSA(
            dim=cfg.dim,
            n_heads=cfg.n_heads,
            n_kv_groups=cfg.n_kv_groups,
            d_k=cfg.d_k,
            d_v=cfg.d_v,
            l=cfg.l,
            d=cfg.d,
            l_sel=cfg.l_sel,
            n_sel=cfg.n_sel,
            w=cfg.w,
        ),
        ToyHead(cfg.dim, vocab=1024),
    ).to(device)
    opt = optim.AdamW(model.parameters(), lr=cfg.lr)
    scaler = GradScaler(enabled=cfg.use_amp)
    debug = os.getenv("NSA_DEBUG_TRAIN", "0").lower() in ("1", "true", "yes")

    def lr_at(step: int) -> float:
        # Linear warmup to cosine decay
        if step < cfg.warmup_steps:
            return cfg.lr * (step + 1) / max(1, cfg.warmup_steps)
        # Cosine from warmup to steps
        t = (step - cfg.warmup_steps) / max(1, cfg.steps - cfg.warmup_steps)
        return cfg.lr * 0.5 * (1.0 + math.cos(math.pi * t))

    model.train()
    for step in range(cfg.steps):
        for pg in opt.param_groups:
            pg["lr"] = lr_at(step)
        x, labels, mask = synthetic_batch(cfg)
        x = x.to(device)
        labels = labels.to(device)
        mask = mask.to(device)
        opt.zero_grad(set_to_none=True)
        with autocast(
            device_type="cuda", dtype=torch.float16, enabled=(cfg.use_amp and device.type == "cuda")
        ):
            logits = model(x)
            loss = masked_ce(logits, labels, mask)
        if cfg.use_amp and device.type == "cuda":
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            scaler.step(opt)
            scaler.update()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            opt.step()
        if debug and step % 10 == 0:
            # Print loss, lr, grad-norm
            with torch.no_grad():
                gn = 0.0
                for p in model.parameters():
                    if p.grad is not None:
                        gn += float(p.grad.detach().norm().item())
                print(
                    f"step={step} lr={opt.param_groups[0]['lr']:.2e} loss={loss.item():.4f} grad_norm≈{gn:.2f}"
                )

    print("done")


if __name__ == "__main__":
    main()
