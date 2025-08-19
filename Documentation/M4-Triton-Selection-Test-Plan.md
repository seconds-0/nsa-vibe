# M4 Triton Selection – Test Plan

Audience: Testing engineer running GPU validation and benches.

## Goals
- Validate correctness parity vs packed SDPA for selection attention.
- Verify robustness across distributions, shapes, and edge cases.
- Measure performance to set \`sel_triton_min_L\` and confirm tile defaults.
- Capture diagnostics for future regressions.

## Environment
- GPU: e.g., RTX 4090 (note model/SM)
- PyTorch ≥ 2.5, CUDA ≥ 12.1
- Triton 3.1.0 (also note if 3.0.0 is tested)

Set base flags as needed:
- \`NSA_USE_TRITON_SEL=1\` to enable Triton selection
- Optional diagnostics:
  - \`NSA_DEBUG_TIMING=1\` per-bucket timing
  - \`NSA_DEBUG_SHAPES=1\` shape/stride logs
  - \`NSA_DEBUG_COMPARE=1\` parity MAE logging (slows runs)

## Quick Sanity
1) Import compile tests (dense + varlen):
```bash
python - <<'PY'
import torch
from nsa.kernels.triton_sel_kernel.sel_fwd import sel_attn_fwd_dense, sel_attn_fwd_varlen
assert torch.cuda.is_available()
N,H,D,Dv,L=4,4,64,64,128
Q=torch.randn(N,H,D,device='cuda',dtype=torch.float16)
K=torch.randn(N,L,D,device='cuda',dtype=torch.float16)
V=torch.randn(N,L,Dv,device='cuda',dtype=torch.float16)
_ = sel_attn_fwd_dense(Q,K,V)
TotalL=N*L
cu=torch.tensor([i*L for i in range(N+1)],device='cuda',dtype=torch.int32)
Kp=torch.randn(TotalL,D,device='cuda',dtype=torch.float16)
Vp=torch.randn(TotalL,Dv,device='cuda',dtype=torch.float16)
_ = sel_attn_fwd_varlen(Q,Kp,Vp,cu)
print('OK')
PY
```

## Parity Tests (GPU)
Opt-in tests (run with CUDA):
- Enable: \`NSA_TEST_TRITON_SEL=1\` then run pytest subset:
```bash
NSA_USE_TRITON_SEL=1 NSA_TEST_TRITON_SEL=1 pytest -q nsa/tests/test_triton_sel_parity_gpu.py::test_triton_wrapper_parity_multirange_small_gpu
```
Expect MAE < 1e-3.

Matrix (manual, sample):
- L ∈ {64,128,256,512}, D ∈ {64,128}, Dv ∈ {64,128}, H ∈ {2,8}
- Dist ∈ {few,many,mixed}
Use bench harness for quick checks (small iters):
```bash
NSA_USE_TRITON_SEL=1 python bench/bench_sel_triton.py --N 256 --H 8 --D 128 --Dv 128 --L_list 64,128,256 --dist mixed --iters 10 --warmup 5 --csv
```

## Edge Cases
- Empty ranges and invalid ranges (clamped) – already covered in CPU tests; spot check GPU if desired by constructing \`ranges\` with zeros/negatives and ensuring fallback parity.
- Single contiguous span triggers dense fast-path.
- Very small shapes: D,Dv,H small (e.g., 16 or 8) – expect fallback if alignment guard rejects.

## Performance & Thresholds
Primary run (collect timing; repeat 3x and average):
```bash
NSA_USE_TRITON_SEL=1 NSA_DEBUG_TIMING=1 python bench/bench_sel_triton.py   --N 1024 --H 8 --D 128 --Dv 128 --L_list 64,128,256,512,1024 --dist many --iters 50 --warmup 10 --csv > results.csv
```
- Compare SDPA vs Triton times; look for ≥1.2x speedup threshold.
- Recommend \`sel_triton_min_L\` as the smallest L where Triton ≥1.2x, rounded up conservatively.

Optional tuning:
- Try env block overrides: \`NSA_SEL_TRITON_BLOCK_D\`, \`NSA_SEL_TRITON_BLOCK_L\`, \`NSA_SEL_TRITON_BLOCK_DV\`.
- Record per-bucket \`sel.triton.bucket_timing\` logs for GB/s; attach to report.

## Diagnostics (if issues)
- Compiler logs: \`TRITON_DEBUG=1\` with a minimal repro.
- Memory summary: \`python -c "import torch; print(torch.cuda.memory_summary())"\`
- Collect device log: ensure \`sel.triton.device\` printed (one-time).

## Acceptance Criteria
- Forward parity: MAE < 1e-3 across the matrix spot checks.
- No kernel compile/runtime errors on supported shapes/dtypes.
- Bench data clearly indicates a minimum \`sel_triton_min_L\`; propose a value (likely ≥1024 if no gain earlier).
- Observability logs present and consistent.

## Deliverables
- CSV timing results and brief summary (threshold recommendation).
- Optional logs: parity MAE, bucket timings, device info.
- Any anomalies noted with repro steps.
