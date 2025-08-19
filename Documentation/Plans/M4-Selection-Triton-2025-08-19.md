### Task ID
M4-Selection-Triton

### Problem Statement
Selection attention currently uses a parity-first gather/SDPA path (with an optional packed varlen implementation). To reach NSA’s intended decode/prefill performance and enable efficient training, we need a custom Triton selection kernel that executes selection attention efficiently and provides backward, while preserving numerical parity and group-consistency semantics.

### Goals
- Triton selection attention kernels (forward and backward) with group-major layout [B,G,S,D*].
- Support decode (S_q=1) and prefill (batched S_q) with identical numerics to reference gather path.
- Deterministic semantics: ranges identical across heads in a group (Eq. 10), no reads > t.
- Autograd integration: torch.autograd.Function wrapper with safe fallbacks.
- Performance parity targets: match or beat masked SDPA and packed gather baselines on CPU/GPU; strong gains on GPU.

### Non-Goals
- Changing compressed/sliding branches (handled by FA‑2).
- Multi-token decode batching beyond S_q=1.

### Proposed Solution
- Kernel design (forward):
  - Input: Q:[B,G,h,Dk], K/V:[B,G,S_kv,D*], sel_ranges:[B,G,n,2].
  - For each (b,g), materialize per-head contiguous segments by iterating ranges, compute Q·K^T with causal masking, softmax, and multiply by V.
  - Avoid materializing full masks; operate over gathered segments directly.
  - Expose optional workspace for temporary buffers to avoid frequent allocations.
- Backward kernel:
  - Compute dQ, dK, dV only for the selected indices; honor causal masking; aggregate across heads.
  - Return zeros for unselected K/V positions (correct semantics for sparse write-back).
- Autograd wrapper:
  - Python guard that dispatches to Triton when available; otherwise fall back to gather/SDPA path.
  - Check device, dtype, head_dim constraints.
- Numerics and safety:
  - Maintain deterministic block selection (done pre-kernel); kernel only consumes ranges.
  - Clamp all reads ≤ t; unit asserts for bounds.
  - Use FA-style log-sum-exp numerics; FP32 accumulation for stability; dtypes fp16/bf16 supported with FP32 accum.

### Edge Cases & Constraints
- Empty selection or padded zero-length ranges must return exact zeros and skip work.
- Overlapping/adjacent ranges are pre-merged upstream; assert monotonic non-overlapping at entry (debug-only).
- Variable range count: accept fixed n with padding; ignore zero-length ranges. Future: packed list with cu_seqlens.
- Tiny shapes: support Dk∈{8,16}, S_kv small; add early-exit when total selected length L=0.
- Head-dim/device/dtype constraints: document supported head dims (e.g., multiples of 8/16), compute capability, and dtypes; strict guard + SDPA gather fallback.
- Determinism: stable reduction ordering per (B,G,h); no atomics across heads.
- Thresholds: `sel_triton_min_L` optional threshold to skip Triton for tiny L; measure via microbench.

### Automated Test Plan
- Forward parity: compare Triton fwd vs `grouped_selection_attention` across random shapes; MAE ≤ 1e‑6 (FP32).
- Backward parity: gradcheck on tiny shapes; compare gradients vs reference masked/gather path.
- Decode parity: S_q=1 path matches reference across random ranges.
- Group consistency: verify identical selected ranges across heads (pre-kernel), and outputs equal across heads when Q identical.
- Perf microbenches: per-range distribution sweeps; bucketed vs naive gather comparisons.
- Edge tests: empty selection → zeros; padded ranges; tiny shapes (Dk=8), multi-group (G>1), multi-head per group.
- Dtype tests: bf16/fp16 with FP32 accum vs FP32 oracle.
- Determinism: repeated runs equal outputs (fixed seed).

### Components Involved
- `nsa/kernels/triton_sel_kernel/sel_fwd.triton`, `sel_bwd.triton`.
- `nsa/kernels/triton_sel_kernel/__init__.py` autograd wrapper.
- Reference paths in `nsa/core/attention_kernels.py` for fallback.
- Tests under `nsa/tests/` for parity and gradcheck.

### Dependencies
- Triton availability on target GPU; CI job with GPU.

### Implementation Checklist
- [ ] Triton fwd kernel
- [ ] Triton bwd kernel
- [ ] Autograd Function + python wrapper
- [ ] Unit tests (fwd/bwd parity, gradcheck)
- [ ] Perf microbench harness
- [ ] CI gating and docs
- [ ] Guards & fallbacks (dtype, head_dim, capability, min-L)
- [ ] Observability counters (selected tokens/bytes)

### Status
In Progress
