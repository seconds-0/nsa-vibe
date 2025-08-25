# NSA Backward Hang: Test Agent + Engineer Plan

Date: 2025-08-25

This doc outlines a concrete, prioritized checklist for the test agent (GPU + online search) and the core engineer (this repo, no core logic changes) to diagnose the NSA backward hang at ≥5 layers.

## Test Agent Tasks (GPU + Online)

Priority A — fastest isolation

1) Branch isolation
- Goal: Identify offending branch. 
- How: Run 5-layer repro forcing a single branch.
- Commands (examples, see scripts below):
  - `NSA_FORCE_BRANCH=win python scripts/nsa_backward_repro.py --layers 5 --seq-len 128 --branch win`
  - `NSA_FORCE_BRANCH=sel ... --branch sel`
  - `NSA_FORCE_BRANCH=cmp ... --branch cmp`
- Record: pass/hang, elapsed time, alloc/reserved MiB, `nvidia-smi` snapshot.

2) Selection backend sweep
- Goal: Determine if masked/packed/gather drives hang.
- How: With branch=sel only, sweep:
  - masked: `--sel masked`
  - packed: `--sel packed`
  - gather: `--sel gather`
- Record: pass/hang, memory, profiler top-10 (if feasible).

3) Compressed backend sweep
- Goal: Compare masked vs parity.
- How:
  - masked: `--cmp masked`
  - parity: `--cmp parity` (note: parity forces both cmp/win parity via env)
- Record as above.

4) Sliding backend sweep
- Goal: Check sliding path behavior.
- How:
  - masked: `--win masked`
  - parity: `--win parity`

Priority B — scaling + introspection

5) Sequence length scaling
- Goal: See if the ≥5 layer threshold shifts with `S`.
- How: Run with `--seq-len {32,64,128,512}` and note the minimum layer count that hangs.

6) Autograd anomaly + blocking
- Goal: Surface hidden errors.
- How: Add flags
  - `--anomaly` to enable `torch.autograd.set_detect_anomaly(True)`
  - `--blocking` to set `CUDA_LAUNCH_BLOCKING=1`
- Record any stack traces.

7) Profiler with memory
- Goal: Identify top allocators.
- How: `--profile --out-dir artifacts/hang_profile/<tag>`
- Save: trace, `memory_summary` before/after backward, and profiler table sorted by `cuda_memory_usage`.

8) Allocator config sensitivity
- Goal: Check allocator fragmentation sensitivity.
- How: Run with `PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64,expandable_segments:True` vs default.
- Record delta in reserved/alloc and pass/hang.

Priority C — deeper isolation

9) Gradient breadcrumbing
- Goal: Find the first layer where backward stops.
- How: `--hooks` to register backward hooks on each `LlamaBlockNSA` and MLP; collect log of hook firings.

10) Minimal single-branch toy
- Goal: Reproduce with a model that repeats only selection (or compressed) subgraph N times within a single layer.
- How: Adapt `nsa_backward_repro.py` with `--toy selN=...` (if needed; can do a quick standalone script).

Artifacts to return per run
- `stdout`/`stderr` logs with env vars echoed
- `nvidia-smi` output around hang
- Profiler trace/table (if enabled)
- `memory_summary.txt` and `memory_stats.json`

## Online Research (Test Agent)

- PyTorch SDPA backward with large/dense masks: known issues, memory spikes, or hangs.
- `index_add_`/scatter backward complexity and memory growth for COO/CSR style updates.
- Advanced indexing gradients (e.g., `K[b,g,idx]`) performance pitfalls in loops; mitigation patterns.
- FA-2 varlen autograd behaviors and any advisories (even though FA-2 can be off).
- CUDA allocator fragmentation/hangs on A100; recommended `PYTORCH_CUDA_ALLOC_CONF` tweaks.
- Autograd dead-code pruning: whether computation used only to produce integer indices can/should be detached safely (for future changes).

## Core Engineer Tasks (This Repo)

Priority A (now)

1) Add turnkey repro/profiling script
- File: `scripts/nsa_backward_repro.py`
- Features: sets env toggles, builds TinyLM, runs forward/backward, optional anomaly/blocking/profiler, dumps memory telemetry; no core code changes.

2) Add matrix runner
- File: `scripts/nsa_backward_matrix.sh`
- Runs a compact set of combinations across branch/backend/seq-len, writing results into timestamped artifact dirs.

3) Provide this plan
- File: `docs/NSA_BACKWARD_HANG_PLAN.md` (this doc).

Priority B (after data comes back)

4) Analyze profiler outputs and identify the pathological backend
- Deliver a short root-cause hypothesis update with next minimal changes to reduce backward memory (still avoiding core logic changes unless explicitly green-lit).

