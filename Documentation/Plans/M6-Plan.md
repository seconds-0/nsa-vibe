# M6 — Varlen, Performance Hardening, and Production Readiness

Status: Proposed (execution begins after M5 parity/benches)

Objective
- Finish all non-training items to make NSA robust and production-ready across devices. Focus on var-length support, FA‑2 integration and thresholds, long-context correctness, performance signals, CI/versioning, and observability — without introducing new model-training scope (that’s M7).

Scope & Workstreams

1) Var-length / Padded Batch Support
- Add optional varlen inputs for prefill and selection code paths with safe fallbacks (dense SDPA when varlen not available).
- Unify length bucketing and cu_seqlens builders across sliding/compressed/selection to avoid drift.
- Tests: ragged-batch parity, causal masks with padding, group-consistency (Eq.10) under varlen.
- Files: `nsa/core/packing.py` (extend), `nsa/core/attention_kernels.py` (varlen codepaths), `nsa/tests/test_collate_varlen.py` (expand), new `test_varlen_masks.py`.

2) FA‑2 Integration, Thresholds, and Fallbacks
- Confirm FA‑2 support checks on 4090/A100; prefer varlen FA‑2 when available, else dense batch; always fallback to masked SDPA if unsupported or numerically unstable.
- Autotune thresholds (min length cutoffs) per device: expose in `configs/base.yaml` with device overrides (e.g., `configs/profiles/sm89.yaml`, `a100.yaml`).
- Guards: on Ada (SM 8.9), default FA‑2 disabled for selection path; allow cmp/win opt‑in with thresholds.
- Tests: FA‑2 parity varlen/dense on GPU (opt-in), numerical finiteness guards, deterministic seeds.
- Files: `nsa/core/attention_kernels.py` (guards + thresholds), `configs/profiles/*` (thresholds), `nsa/tests/test_fa2_parity*.py` (expand).

3) Selection Kernel Execution & Gating (Production Defaults)
- Keep Triton selection disabled by default on 4090 per ADR; maintain robust fallback to packed SDPA.
- Enable Triton on A100/H100 only behind flags; provide minimal perf presets and clear logging.
- Add an environment probe and a single source of truth for execution routing (SDPA vs FA‑2 vs Triton) to avoid drift across call sites.
- Tests: parity + smoke on compatible GPUs, forced-fallback tests on 4090.
- Files: `nsa/kernels/triton_sel_kernel/__init__.py`, `nsa/core/flags.py` (routing), `nsa/tests/test_sel_triton_wrapper*.py`.

4) Long-Context Correctness & Efficiency Signals
- 64k needle-in-a-haystack retrieval for decode (functional): ensure strict causality, block mapping (Eq.9), and group consistency (Eq.10) hold at scale.
- Counters: verify decode reads trend: reads(S) = num_cmp(S) + n·l′ + min(w, S); monotonicity across steps.
- Benches: extend decode/prefill benches with CSV outputs and a small summary script; ensure shape trends match PRD (not exact ms).
- Files: `nsa/tests/test_long_context_needle.py` (smoke), `bench/bench_decode.py` (CSV stays legacy), `scripts/summarize_bench.py` (new).

5) CI, Version Matrix, and Skips
- Enforce Torch↔Triton matrix: 2.2↔2.2, 2.3↔2.3, 2.4+↔3.x; fail fast on unsupported combos; document in GPU test plan.
- Split CI lanes: CPU-only parity (always), GPU opt-in lanes (FA‑2, Triton) with explicit skips and clear messages.
- Ensure 4090 lanes use SDPA defaults; Triton runs only with NSA_TRITON_SEL_FORCE=1 in parity suites.
- Files: `.github/workflows/ci.yml` (matrix), `nsa/tests/conftest.py` (skip markers), `Documentation/Test-Plans/GPU-Test-Plan.md` (version table updated).

6) Observability & Docs
- Ensure gate mean/std, per-branch read counters, and parity MAE logs are finite and rate-limited; add one-line sampler for decode step.
- Developer docs: execution routing, thresholds, device-specific guidance (4090 vs A100), known ADRs.
- Files: `nsa/core/debug.py` (rate limits), `Documentation/Guides/Execution-Routing.md` (new), update `README.md` pointers.

Deliverables
- Varlen support: packing utils, masks, and tests.
- FA‑2 hardened integration with thresholds and fallbacks; device profiles updated.
- Selection kernel routing hardened (ADR-compliant), parity tests green; no regressions on CPU.
- Long-context smoke tests; decode/prefill CSV benches and summarizer.
- CI gating for Torch/Triton versions; stable skips with messages.
- Updated docs for operators and flags.

Acceptance Criteria
- All M0–M5 core suites pass on CPU; GPU opt-in suites pass or skip with clear reason.
- Varlen tests pass (ragged batch masks, parity vs dense reference on small shapes).
- FA‑2 parity tests pass on supported GPUs; no NaNs/Infs propagate (fallbacks kick in if detected).
- Long-context 64k needle smoke passes; counters match formula within tolerance at steps.
- Decode/prefill benches run and produce CSV; summarizer shows trend-aligned timing and read counts.
- CI enforces Torch↔Triton matrix and conditional Triton tests; ADR for 4090 respected by default.

Risk & Mitigations
- 4090 instability (FA‑2/Triton): keep SDPA defaults, guard via thresholds and fallbacks.
- Detection drift: centralize probes; unit-test the detection/routing outcome.
- Varlen corner cases: add exhaustive small-shape tests; fall back to dense for unsupported.

Timeline (indicative)
- Week 1: Varlen packing/masks + tests; routing unification; CI matrix + skips.
- Week 2: FA‑2 hardening & thresholds; long-context smoke; bench summarizer.
- Week 3: Device-profile polish; docs; wrap-up and PRD acceptance checks.

