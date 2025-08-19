### Task ID
M4-Triton-Selection-Benchmark-Plan

### Purpose
Benchmark the Triton selection forward path vs the packed SDPA reference to: (1) validate numerical parity on GPU, (2) quantify speedups vs SDPA across realistic NSA shapes and selection patterns, and (3) determine safe defaults for enabling Triton with a minimal-L guard and tuned tile sizes.

### Target Hardware & Software
- GPU: RTX 4090 (primary). Optional later: A100 (80GB), H100.
- Drivers / Tooling:
  - CUDA ≥ 12.x (matching the environment used in FA‑2 benches)
  - PyTorch ≥ 2.2
  - Triton ≥ 3.0
- Repo: `github.com/seconds-0/nsa-vibe`

### Toggles & Config
- Enable Triton selection:
  - Env: `NSA_USE_TRITON_SEL=1`
  - Config: `runtime.use_triton_sel: true` (optional)
- Minimal selected length per row to prefer Triton: `NSA_SEL_TRITON_MIN_L` (default 64)
- Packed SDPA reference (default): `NSA_USE_SEL_PACK=1`
- Force parity reference (for debugging): `NSA_FORCE_PARITY=1` (disables Triton/FA‑2)
- Recommended defaults while benchmarking:
  - Triton: ON (via env)
  - Min‑L: 64 (start), then sweep
  - FA‑2 settings unchanged (we benchmark selection only)

### Parity Validation (GPU)
- Goal: Triton vs SDPA numerical agreement on dense, packed rows.
- Command (opt‑in):
  - `NSA_USE_TRITON_SEL=1 NSA_TEST_TRITON_SEL=1 /usr/bin/python3 -m pytest -q -k triton_sel_parity`
- Expected: MAE ≤ 1e‑3 for fp16/bf16 vs SDPA, on small dense tests.
- Notes:
  - Set `torch.backends.cuda.matmul.allow_tf32=False` to keep comparisons tight (optional).
  - If parity fails, capture seeds, shapes, and print a minimal repro.

### Microbenchmark Matrix (Selection Only)
- Dimensions:
  - Query Heads H ∈ {4, 8}
  - Head dims: D ∈ {64, 128}; Dv ∈ {64, 128}
  - Selected length L ∈ {16, 32, 64, 128, 256, 512, 1024}
  - Row count N ∈ {64, 256, 1024}
- Selection span distributions (per row):
  - Few long: n=1, ranges [(0, L)]
  - Many small: n=8, equally split spans summing to L (e.g., 8 spans of L/8)
  - Mixed: n ∈ {2,4,8}, randomized spans whose lengths sum to L
- Initialization:
  - Random seeds fixed per trial; Q, K, V ~ N(0,1) cast to bf16/fp16 as needed
- Comparisons:
  - Triton (current kernel) vs reference Packed SDPA
- Metrics (per configuration):
  - Median time over 50 iters (5 warmup), 95% CI (optional)
  - Approx bytes read (K + V) per row; effective GB/s
  - Speedup = `SDPA_time / Triton_time`

### Decode-Path Benchmark (S=1)
- Shapes representative of NSA decode:
  - h ∈ {2, 4}, D/Dv ∈ {64, 128}
  - l_sel = 64, n_sel ∈ {8, 16}; total L ∈ {128, 256, 512}
- Build selection ranges from `n_sel` blocks of length `l_sel` (clamped ≤ t), then pack rows
- Compare Triton vs Packed SDPA as above

### Autotune Procedure (Optional)
- For a hot shape (e.g., H=8, D=Dv=128, N=1024, L ∈ {128, 256, 512}), sweep:
  - BLOCK_L ∈ {64, 128, 256}
  - BLOCK_D ∈ {32, 64, 128}
  - BLOCK_DV ∈ {32, 64, 128}
  - num_warps ∈ {2, 4, 8}
  - num_stages ∈ {1, 2, 3}
- Pick the top‑k configs per L bucket and record; consider static defaults per bucket.

### Scripts
- Suggested: `bench/bench_sel_triton.py`
  - Args: `--N`, `--H`, `--D`, `--Dv`, `--L_list`, `--dist {few,many,mixed}`, `--iters`, `--warmup`, `--decode`
  - Behavior: generate packed rows for each (L, dist), run Triton vs Packed SDPA, print CSV/Markdown lines
- Example:
  - `NSA_USE_TRITON_SEL=1 /usr/bin/python3 bench/bench_sel_triton.py --N 1024 --H 8 --D 128 --Dv 128 --L_list 64,128,256,512 --dist many --iters 50 --warmup 5`
  - `NSA_USE_TRITON_SEL=1 /usr/bin/python3 bench/bench_sel_triton.py --decode 1 --H 4 --D 64 --Dv 64 --L_list 128,256,512 --dist few`

### Acceptance Criteria
- Parity: MAE ≤ 1e‑3 on dense parity test (GPU), spot check decode parity
- Performance: Identify L thresholds where Triton ≥ 1.2× SDPA; propose `sel_triton_min_L` accordingly
- Stability: No perf regressions >10% in any configuration below the threshold; otherwise document and fallback

### Outputs & Reporting
- CSV/Markdown table per configuration with: method, time (ms), speedup, GB/s
- Proposed `sel_triton_min_L` and tuned tile config per L bucket
- Patches:
  - Update `configs/base.yaml` with tuned `sel_triton_min_L`
  - `PRD.md` notes on Triton defaults and min‑L rationale
- Store logs under `bench/results/sel_triton/{date}/` (optional)

### Troubleshooting
- If parity fails: confirm 1/√D scaling, ensure ranges ≤ t, verify dtype (bf16/fp16 with FP32 accums)
- If Triton slower: raise min‑L threshold; try larger BLOCK_L; increase num_warps; ensure coalesced loads
- OOM: reduce N or L; benchmark per bucket in smaller chunks

### Status
Ready to run
