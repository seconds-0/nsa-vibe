# AGENTS.md

Guidance for coding agents (Codex CLI, Cursor, Claude Code) contributing to this repository. This document aligns with CLAUDE.md and the Cursor rules under `.cursor/`.

## Source of Truth

- PRD: `PRD.md`
- Execution rules: `.cursor/rules/20-m0-execution.mdc`
- Roles and routing: `.cursor/rules/10-agents-roles.mdc`
- Claude setup and commands: `CLAUDE.md`

## Project Overview (NSA)

- Native Sparse Attention (NSA): decoder-only attention with three branches — Compressed (cmp), Selected (sel), Sliding window (win) — combined by a learned gate. Targets trainable, hardware-aligned sparsity with strict causality and GQA group consistency.

## Roles (A–F) and Ownership

- NSA Architect: Approves architecture vs PRD; guards invariants.
- Agent A — BlockIndex: `nsa/core/block_index.py`, CSR `M_csl`, Eq.9 tests.
- Agent B — CompressPool: `nsa/core/compress_pool.py`, ϕ + emission schedule, RoPE ordering.
- Agent C — SelectionScorer: `nsa/core/selection_scorer.py`, p_cmp→p_slc→group reduce→top‑n.
- Agent D — NSAAttention: `nsa/core/nsa_attention.py`, caches, gating, masks, counters.
- Agent E — Tests: `nsa/tests/*` for causality, Eq.9/10, equivalence, counters.
- Agent F — TritonKernel (future): `nsa/kernels/triton_sel_kernel/*` (M4/M5).

## Global Rules (apply to all agents)

- Follow PRD defaults and plans; do not weaken invariants without Architect approval.
- Keep edits minimal and scoped; preserve formatting; avoid unrelated refactors.
- Determinism in tests: fixed seeds; no flaky nondeterminism.
- Causality: no branch reads indices > t. Enforce masks defensively.
- GQA consistency: groups share selection decisions (Eq.10).
- CPU fallback must work: selection gathers then SDPA; cmp/win use SDPA.
- Observability: expose/read counters, per‑branch token reads, gate stats; heatmaps for p_cmp/p_slc when applicable.

## Communication Style (for agents)

- Be opinionated and decisive: recommend and execute actions that advance the PRD without asking “if you want” for obvious, low‑risk steps.
- Take initiative on small, reversible changes (bench fixes, docs alignment, tests) and push them with clear commit messages.
- Ask for approval only for destructive operations, costly builds, or scope changes that affect milestones.
- Before tool calls, state succinctly what you will do next (1–2 sentences), then do it.

## M0 Execution Constraints (steel thread)

- SDPA everywhere for cmp/sel/win. No Triton kernels or imports.
- Fixed sequence length batches; no varlen APIs.
- p_cmp tiling in prefill allowed to bound memory; keep deterministic flags.
- Gate MLP: last layer zero‑init; softmax with τ=1.0; log means/std to catch collapse.
- Counters: implement exact decode reads and expose per‑step values.
- torch.compile: disabled in tests; allowed in benches for cmp/win only (selection path stays eager).
- Env ablations: `NSA_FORCE_BRANCH={cmp|sel|win}` for one‑hot gating in tests/debug.

## Selection Semantics

- Mapping: CSR fractional‑overlap Eq.9 from cmp blocks → selection blocks.
- Deterministic top‑k: break ties by lower index; include forced block 0 and two local blocks; de‑duplicate; merge adjacent; clamp ≤ t.
- Separate caches: `K_win,V_win` vs `K_sel,V_sel`.

## Defaults and Constraints (from PRD / paper)

- Blocks: l=32, d=16, l′=64, n=16 (forced initial + 2 local included), w=512.
- GQA: G=4, H=64, d_k=192, d_v=128.
- Divisibility: enforce d|l and d|l′.
- Causality: strictly causal masks across branches.
- Decode memory/reads:
  - num_cmp(S) = 0 if S < l else floor((S − l)/d) + 1
  - reads(S) = num_cmp(S) + n·l′ + min(w, S)

## RoPE and ϕ

- Apply RoPE to Q and to per‑branch K before ϕ; ϕ runs on RoPE‑transformed K/V; Q uses RoPE.

## Testing Priorities

- Correctness: causal masking invariants; Eq.9 mapping; Eq.10 group consistency; small‑S equivalence with full attention.
- Long context: decode counter verification; 64k needle retrieval.
- Trainability (M2+): gradcheck on small shapes; toy convergence.

## Commands (uv workflow)

- Setup: `uv venv -p 3.11 .venv && uv pip sync -r requirements.txt`
- Test (fast): `uv run -q pytest`
- Long tests: `uv run -q pytest -m long`
- Lint: `uv run ruff check .`
- Type check: `uv run mypy -p nsa`
- Prefill bench: `PYTHONPATH=. uv run python bench/bench_prefill.py --config configs/base.yaml`
- Decode bench: `PYTHONPATH=. uv run python bench/bench_decode.py --config configs/base.yaml`

## Test Toggles and Flags

- Fast path toggles (default ON), with global escape hatch:
  - `NSA_USE_WIN_MASK=1` enable sliding masked SDPA
  - `NSA_USE_CMP_MASK=1` enable compressed masked SDPA
  - `NSA_USE_SEL_PACK=1` enable selection packing
  - `NSA_FORCE_PARITY=1` force reference per‑token/gather paths (overrides above)
- Masked SDPA parity and tiny shapes: `NSA_TEST_MASKED=1 uv run -q pytest -k "test_batched_parity or test_masked_tiny"`
- Selection packed parity: `NSA_TEST_SEL_PACK=1 uv run -q pytest -k test_selection_packed`
- FA‑2 (M1 opt‑in):
  - `NSA_USE_FA2=1`, `NSA_USE_FA2_WIN=1`, `NSA_USE_FA2_CMP=1`
  - `NSA_FA2_MIN_LEN_WIN`, `NSA_FA2_MIN_LEN_CMP`
  - GPU parity: `NSA_TEST_FA2=1 uv run -q pytest -k fa2_gpu_varlen`
  - Bench: `PYTHONPATH=. uv run python bench/bench_fa2.py`
- Training (M2, GPU opt‑in):
  - Train toy: `NSA_USE_FA2=1 PYTHONPATH=. uv run python scripts/train_toy.py --config configs/base.yaml`
  - GPU training tests: `NSA_TEST_TRAIN=1 uv run -q pytest -k train`
  - CPU smoke (CI): `PYTHONPATH=. uv run -q pytest -k train_smoke`

## M7 Training Debug Commands

- Unbuffered training with diagnostics (env dump, heartbeat, watchdog):
  - `export CONFIG=configs/m7c_125m_fast_log.yaml; export PYTHONUNBUFFERED=1`
  - `python -u scripts/train_showcase.py --dataset fineweb_edu --ddp 0 --fwe-report-docs 500 --loader-timeout 120 2>&1 | tee training.log`
- Heartbeat and dumps:
  - Tail: `tail -f artifacts/train_showcase/heartbeat_rank0.jsonl`
  - Stack dump: `kill -USR1 <PID>` then inspect `artifacts/train_showcase/stackdump_*.txt`
- Loader-only smoke tool:
  - `python scripts/automation/fwe_smoke.py --seq-len 1024 --batch 1 --timeout 60 --tokenizer byte`
  - `python scripts/automation/fwe_smoke.py --seq-len 1024 --batch 1 --timeout 60 --tokenizer gpt2`
- HF streaming sanity:
  - `python - <<'PY'\nfrom datasets import load_dataset\ns=load_dataset('HuggingFaceFW/fineweb-edu', split='train', streaming=True)\nprint('ok, first text head:', next(iter(s))['text'][:80])\nPY`
- Local/offline fallback:
  - `python -u scripts/train_showcase.py --dataset fineweb_edu_local --local-path /data/local.jsonl --ddp 0`
- Synthetic sanity:
  - `python -u scripts/train_showcase.py --dataset synthetic --ddp 0`

## Agent Workflow (for this repo)

- Plan: outline small, verifiable steps; keep exactly one step in progress.
- Communicate: brief preambles before tool calls; group related actions.
- Edit: make small, focused patches; avoid unrelated formatting; keep imports stable.
- Validate: run targeted tests for the area you changed, then broader suites.
- Safety: ask before destructive actions; keep CPU fallback working; do not introduce Triton in M0.

## Acceptance Checklist (per change)

- Invariants preserved (causality, GQA consistency, Eq.9/10 semantics, counters).
- Tests relevant to the change are green; core M0 suites pass:
  - `test_masks.py`, `test_block_math.py`, `test_group_consistency.py`,
    `test_equiv_small.py`, `test_decode_counters.py` (subset acceptable for scoped PRs).
- Logging/observability intact for gates and counters.
- No Triton usage in M0; optional FA‑2 guarded by flags only.

## GPU Pod (reference)

- See `CLAUDE.md` for Prime Intellect pod access, environment bootstrap, and GPU benchmarking commands.

## PR/Change Hygiene

- Keep edits minimal; reference the workplan milestone (e.g., M0 Steel Thread) in commit messages.
- Commit only core code/tests/docs; avoid extraneous artifacts.

## Branch Map (WIP)

- main: Safe, opt-in improvements with fallbacks
  - Selection: High default gate (`NSA_SEL_TRITON_MIN_L=4096`), hard try/except fallbacks to packed SDPA.
  - Triton varlen fix: Removed pointer-shape broadcast masks; use tile-boolean masks.
  - Group-centric kernels (opt-in via `NSA_SEL_TRITON_GROUP=1`):
    - Dense: `_sel_attn_fwd_dense_group_kernel` + `sel_attn_fwd_dense_group`, FP32 LSE, K/V tile reuse across heads, p reused across Dv tiles.
    - Varlen: `_sel_attn_fwd_varlen_group_kernel` + `sel_attn_fwd_varlen_group`, cu_seqlens, same reuse.
    - Expanded autotune configs (warps/stages) for group kernels.
  - Bench + CI:
    - `bench/bench_sel_triton.py` (dense/varlen, streams, CSV) for Triton vs packed SDPA.
    - Modal integration (`bench/modal_gpu_bench.py`) runs selection benches, parses speedups, proposes `sel_triton_min_L`.
    - Threshold optimizer (`bench/threshold_optimizer.py`) updates `configs/base.yaml` with FA‑2 + selection thresholds; GitHub Action PR includes them.

- Decision (ADR-2025-08-M4-02): Triton selection is non‑viable on RTX 4090 (SM 8.9) — wrapper forces fallback to packed SDPA unless `NSA_TRITON_SEL_FORCE=1`. Prefer SDPA/FA‑2 for production.
  - Experimental alternative: `NSA_SEL_CUDA=1` routes selection through a CUDA wrapper (forward-only), defaulting to packed SDPA until kernel lands.

## Milestones Status
- M0 Steel Thread: Completed — SDPA paths, selection mapping, gating, counters, causality and group consistency tests green.
- M1 FA‑2 Integration: Completed — sliding/compressed FA‑2 with thresholds and robust fallbacks; benches and parity smokes available.
- M2 Trainability: Completed — toy training, gradcheck/backward tests; learnable ϕ opt‑in with init parity.
- M3 Full Decode Caches: Completed — per‑branch caches, emission schedule, decode reads formula, long‑context smoke.
- M4 Triton Selection: Paused on 4090 per ADR; SDPA remains production; revisit on A100/H100 or later compiler improvements.

- feat/m4-group-kernel: Riskier performance refactors and backward (to be created when work starts)
  - Double-buffering for K/V tiles and deeper tile reshapes (behind `NSA_SEL_TRITON_DB=1`).
  - Backward kernel (M5) with group-centric schedule (behind `NSA_SEL_TRITON_ALLOW_GRAD=1`).
  - Additional autotune expansions and schedule experiments.
