# Training How‑To (M2)

## Quick start

```bash
uv venv -p 3.10 .venv
uv pip sync -r requirements.txt
# CPU smoke (slow, correctness only)
PYTHONPATH=. ./.venv/bin/python scripts/train_toy.py
```

## Recommended (GPU)

```bash
# Enable FA‑2 paths on GPU and timing if desired
NSA_USE_FA2=1 NSA_DEBUG_TRAIN=1 PYTHONPATH=. ./.venv/bin/python scripts/train_toy.py
```

## Useful env flags
- NSA_TRAIN_SEED (default 1337)
- NSA_TRAIN_LR (default 3e‑4), NSA_TRAIN_WARMUP (default 50), NSA_TRAIN_CLIP (default 1.0)
- NSA_TRAIN_BS (default 8), NSA_TRAIN_MAX_LEN (default 128), NSA_TRAIN_STEPS (default 200)
- NSA_DEBUG_TRAIN=1 to print step, lr, loss, grad‑norm periodically
- NSA_USE_FA2=1 to use FA‑2 paths on GPU

## Notes
- AMP: bf16 preferred on capable GPUs; fp16 uses GradScaler.
- Loss masking: pad tokens excluded; last token has no next‑token label.
- Determinism: seeds set; CI remains CPU-only and deterministic.
