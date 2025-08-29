# FlashAttention Enablement Test Runbook (A100/H100)

Purpose: Verify that FlashAttention-2 (FA‑2) is installed, detected, and used by NSA paths. Focus only on enabling FA‑2 and validating parity and basic throughput, before any broader optimizations.

## 1) Environment Prep

- GPU: A100 80GB (preferred) or H100
- OS: Ubuntu 22.04, CUDA 12.x image
- Python: 3.11

```bash
cd nsa-vibe
uv venv -p 3.11 .venv
source .venv/bin/activate
uv pip sync -r requirements-gpu-cu121-torch24.txt
pip install --no-cache-dir --no-build-isolation flash-attn
python - <<'PY'
from flash_attn import flash_attn_func
print('FA2 dense OK')
try:
  from flash_attn import flash_attn_varlen_func
  print('FA2 varlen OK')
except Exception:
  from flash_attn import flash_attn_varlen_kvpacked_func
  print('FA2 kvpacked varlen OK')
PY
```

## 2) Minimal FA‑2 Activation

Enable FA‑2 routing and keep other toggles at defaults.

```bash
export NSA_USE_FA2=1
# Optional: ensure varlen paths engage at small lengths
export NSA_FA2_MIN_LEN_WIN=1
export NSA_FA2_MIN_LEN_CMP=1
```

Sanity probe (dense fallback path):
```bash
python - <<'PY'
import torch
from nsa.kernels.flash_wrappers import is_flash_available, is_flash_varlen_available
print('flash_dense_available:', is_flash_available())
print('flash_varlen_available:', is_flash_varlen_available())
PY
```

Expected: both True on GPU host.

## 3) Unit/Parity Tests (FA‑2)

Run FA‑2 specific tests (GPU-only, opt-in):

```bash
NSA_TEST_FA2=1 uv run -q pytest -k "fa2_parity or backward_varlen"
```

Targets include:
- `test_fa2_parity.py` (sliding/compressed parity, decode)
- `test_fa2_parity_improved.py` (dense parity + shape/causal sanity)
- `test_backward_varlen.py` (gradient parity vs masked reference)

Acceptance:
- All passing; no CUDA illegal memory access; no non‑finite outputs.

## 4) Quick Runtime Smoke (Prefill)

Use the built-in bench or a small forward smoke to touch common code paths.

Option A — Bench FA‑2:
```bash
PYTHONPATH=. NSA_USE_FA2=1 uv run -q python bench/bench_fa2.py
```

Option B — Tiny model forward smoke:
```bash
python - <<'PY'
import torch
from nsa.model.llama_block_nsa import LlamaBlockNSA
torch.manual_seed(0)
blk = LlamaBlockNSA(dim=256, n_heads=8, n_kv_groups=2, d_k=32, d_v=32).cuda()
blk.attn._cache_env_vars()  # refresh flags in case of new env
blk.train(False)
x = torch.randn(1, 128, 256, device='cuda', dtype=torch.float32)
with torch.inference_mode():
    y = blk(x)
print('ok, out shape:', tuple(y.shape), 'finite:', torch.isfinite(y).all().item())
PY
```

Acceptance:
- No crashes; finite outputs; completion < 1s for S=128 on A100.

## 5) Throughput Sanity (Short)

Run a very short smoke to confirm FA‑2 impact without full training.

```bash
PYTHONPATH=. NSA_USE_FA2=1 NSA_FA2_MIN_LEN_WIN=1 NSA_FA2_MIN_LEN_CMP=1 \
uv run -q python bench/bench_decode.py --config configs/base.yaml --steps 200
```

Acceptance:
- Sliding/compressed calls route through FA‑2 without fallback; no non‑finite warnings.
- On A100, observed prefills+decode exceed 300 tok/s for S≈2048 (informal check; full runbook has stricter gate).

## 6) Troubleshooting

- `ImportError: flash_attn`: reinstall with `pip install --no-build-isolation flash-attn` in the CUDA 12.x image.
- `flash_varlen_available: False`: ensure FA‑2 version exposes varlen entrypoints; kvpacked variant is acceptable.
- `CUDA illegal memory access`: confirm the repo includes cuq/cuk initialization fixes; update to latest commit.
- Forced CPU fallback on Ada (SM 8.9): set `NSA_FA2_FORCE=1` to override if testing on RTX 4090 (not recommended for production).

## 7) Exit Criteria

- FA‑2 parity and backward tests (GPU) pass cleanly.
- Runtime smoke completes with FA‑2 active; no fallbacks; finite outputs.
- If all green, proceed to the main Single-A100 runbook for full production smoke and throughput acceptance.

