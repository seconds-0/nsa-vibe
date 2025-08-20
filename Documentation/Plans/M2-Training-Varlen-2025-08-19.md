### Task ID
M2-Training-Varlen

### Problem Statement
We have an NSA attention module with M0 correctness and M1 FA‑2 performance for sliding/compressed. The next milestone is to make NSA training-ready and var-length capable for realistic workloads. We must integrate NSA into a decoder block for toy LM training, support FA‑2 forward/backward on GPU, add var-length packing/bucketing for batches, and provide training‑time observability and acceptance tests. CPU remains a fallback (slow but correct).

### Goals (What M2 Delivers)
- End‑to‑end training readiness:
  - NSA integrated into a minimal decoder‑only block; toy language modeling loop reduces loss.
  - GPU training (single GPU) with FA‑2 forward/backward for compressed/sliding branches.
  - CPU fallback training path via SDPA remains available (slow, for CI smoke).
- Var‑length batching for prefill (training):
  - Length bucketing + varlen packing with `cu_seqlens_q` and `cu_seqlens_kv`.
  - Collate helper to build batch tensors and length metadata.
- Mixed precision:
  - AMP (bf16/fp16) support for training; gradients stable; gating MLP zero‑init policy respected.
- Observability & determinism:
  - Train‑time logging of gate distributions, token read stats, per‑branch contribution ratios.
  - Determinism policy documented; CI CPU smoke remains deterministic.
- Acceptance tests:
  - Gradcheck/micro backward tests for cmp/win branches.
  - Tiny training loop: loss decreases on a small corpus; parity/backward checks for varlen path on small grids.

### Non-Goals (M2)
- Triton selection kernel (forward/backward) — deferred to M4/M5.
- Multi‑GPU/distributed training — defer to M3.
- Quantization, 8‑bit optimizers, or ZeRO — later milestones.

### Proposed Solution

1) Minimal model integration
- Add `nsa/model/llama_block_nsa.py` (or similar) with: input proj → NSAAttention → MLP → residuals and RMSNorm.
- Config toggles to swap in NSA for attention; small dimensions defaults for toy runs.

2) Var‑length batching (training prefill)
- Data pipeline produces batches with variable sequence lengths per sample.
- Implement collate that builds:
  - Packed Q rows and KV streams with per‑row lengths.
  - `cu_seqlens_q`, `cu_seqlens_k` for FA‑2 varlen; fallback dense padding per bucket if varlen unavailable.
- Length bucketing to minimize padding and maximize bucket efficiency.
- Loss masking: construct per‑row causal masks and label masks to ignore pad; handle next‑token label shift with varlen.

3) FA‑2 forward/backward
- Ensure wrappers and kernels engage FA‑2 on GPU for both fwd/bwd paths; verify support via capability checks.
- Fallback to masked SDPA on unsupported cases; keep tolerances documented.

4) Mixed precision & stability
- Enable AMP autocast (bf16/fp16) in training loop; keep accumulations in FP32 where needed.
- Maintain gating MLP last‑layer zero‑init; optional temperature; monitor gate distributions.
- Use GradScaler for fp16 where appropriate; prefer bf16 on H100/L40S.

5) Observability
- Add optional logging hooks for:
  - Gate means/std per batch and per branch.
  - Branch contribution ratios (mean |O_cmp|, |O_sel|, |O_win|) and token read counts.
  - Time per branch if `NSA_DEBUG_TIMING=1` during training micro‑runs.
- (Optional) TensorBoard scalar writer in `scripts/train_toy.py`.
- Track FA‑2 path usage counts, bucket histograms, gradient norms, and clipping events.

6) Acceptance & tests
- Gradcheck on small shapes for cmp/win (FA‑2 on GPU; SDPA on CPU) — skip if unsupported.
- Backward parity on tiny grids: FA‑2 vs SDPA grads close within tolerance.
- Training smoke: run N steps on tiny dataset; verify monotonic loss decrease (tolerance‑based). CPU smoke for CI.

### Automated Test Plan
- Unit:
  - Varlen pack correctness (cu_seqlens computed matches lengths; scatter/gather idempotence on small cases).
  - Backward gradcheck for cmp/win on tiny shapes (skip if FA‑2 unsupported).
  - Backward parity (FA‑2 vs SDPA) tolerance tests on GPU for a handful of shapes.
  - Eq. 10 group‑consistency during training: selected ranges identical across heads in a GQA group for varlen batches.
  - Varlen causality: no token read beyond t across rows; unit asserts on gathered indices.
- Integration:
  - Tiny training loop (1 GPU) on toy corpus; assert loss decreases ≥ X% over Y steps.
  - CPU training smoke (very small shapes) to keep CI green without GPU.
- Determinism:
  - Repeat‑run GPU outputs close within tolerance (MAE ≤ 1e‑5) where bitwise determinism not guaranteed.

### Components Involved
- `nsa/model/llama_block_nsa.py` (new): minimal block wiring NSA.
- `scripts/train_toy.py` (new): toy LM training driver.
- `nsa/core/packing.py`: extend for training varlen collate helpers.
- `nsa/kernels/flash_wrappers.py`: ensure varlen paths support backward; tighten probes.
- `nsa/core/attention_kernels.py`: training paths reuse existing FA‑2 varlen packers; ensure contiguous tensors; workspace reuse.
- `nsa/tests/`: add `test_train_smoke.py`, `test_backward_varlen.py`.

### Dependencies
- PyTorch ≥ 2.3; FlashAttention‑2 ≥ 2.x on GPU (optional, guarded).
- CPU CI remains SDPA‑only; GPU tests opt‑in.

### Implementation Checklist
- [ ] Minimal NSA block module and config glue; unit tests for block forward shape.
- [ ] Varlen collate helpers for training batches; length bucketing and `cu_seqlens` creation.
  - [ ] Loss masking for varlen: label shift, pad ignore, causal alignment tests.
- [ ] Ensure FA‑2 varlen wrappers support backward; add contiguity assertions; robust fallbacks.
- [ ] Gradcheck (cmp/win) tiny shapes; skip on unsupported GPUs.
- [ ] Backward parity tests (FA‑2 vs SDPA grads) on tiny grids (GPU opt‑in).
- [ ] Training toy script; AMP enabled; gate stats logging (optional TB).
  - [ ] Optimizer defaults (AdamW + cosine/warmup), gradient clipping, gradient norm logging.
- [ ] Training smoke tests: CPU tiny and GPU opt‑in; assert loss reduction.
- [ ] Docs: PRD training notes; rules updated with train commands and flags.
- [ ] CI: keep default CPU tests green; provide instructions for GPU‑opt‑in suite.

### Verification Steps
- Run default suite (CPU) → green.
- Run GPU opt‑in backward tests and tiny training (`NSA_TEST_TRAIN=1`) → pass tolerances; loss decreases.
- Benchmarks (optional): measure backward timings of cmp/win vs SDPA to confirm FA‑2 benefits in training.

### Decision Authority
- Engineering owns model/block integration and acceptance thresholds; Product signs off that training readiness and observability meet requirements.

### Questions / Uncertainties
- FA‑2 varlen backward coverage on specific GPUs/dtypes — if partial, document and fallback to dense/bucketed or SDPA with clear tolerances.
- AMP behavior on different GPUs (bf16 vs fp16) — prefer bf16 on H100/L40S if available.

### Acceptable Tradeoffs
- Prefer clarity and robust fallbacks to peak performance in training path; maintain SDPA oracle for tests.
- Allow small numeric drift in grads within FP32/AMP tolerances.

### Status
Completed

### Completion Evidence
- Minimal model/block integration present; toy training driver in `scripts/train_toy.py`.
- Varlen collate helpers and packing used across attention paths; tests in `nsa/tests/test_collate_varlen.py`.
- Backward/gradcheck tests: `nsa/tests/test_backward_varlen.py`, `nsa/tests/test_gradcheck_varlen.py` (opt‑in GPU; CPU fallback in CI).
- Training smoke: `nsa/tests/test_train_smoke.py` passes; loss decreases on tiny shapes.
- Learnable ϕ: opt‑in Conv1d path added, parity‑safe at init (`nsa/tests/test_phi_mlp_equiv.py`).

### Notes
- Keep `NSA_FORCE_PARITY=1` in CI by default; training GPU suite opt‑in with `NSA_TEST_TRAIN=1`.
- Consider adding a small corpus in `scripts/data/` or generate synthetic tokens to avoid IP concerns.
