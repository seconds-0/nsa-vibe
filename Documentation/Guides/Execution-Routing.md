# Execution Routing Guide

Overview
- NSA chooses among SDPA, FA‑2, and Triton per branch based on device, dtype, sizes, and flags. Defaults are conservative: SDPA baseline; FA‑2 opt‑in for cmp/win after parity; Triton selection off by default.

Key Flags
- `NSA_FORCE_PARITY=1`: force reference SDPA for all branches (bypasses FA‑2/Triton/packed/masked fast paths).
- `NSA_USE_WIN_MASK` / `NSA_USE_CMP_MASK` / `NSA_USE_SEL_PACK`: enable masked SDPA or packed selection (default ON when not forcing parity).
- `NSA_USE_FA2` / `NSA_USE_FA2_{WIN,CMP}`: opt‑in FA‑2; cutoffs via `NSA_FA2_MIN_LEN_{WIN,CMP}`.
- `NSA_USE_TRITON_SEL`: opt‑in Triton selection (A100/H100 only, after parity); `NSA_TRITON_SEL_FORCE=1` for experiments.
- `NSA_FORCE_BRANCH={cmp|sel|win}`: force one‑hot gating (benches/ablations only).

Device Policies
- RTX 4090 (SM 8.9):
  - Triton selection disabled by ADR; selection uses SDPA by default. Use `NSA_TRITON_SEL_FORCE=1` only for parity tests.
  - FA‑2 defaults OFF; set high min length cutoffs (profiles/sm89.yaml). Enable selectively via flags.
- A100/H100:
  - SDPA baseline for all branches. Opt‑in FA‑2 for cmp/win after parity smokes and cutoff tuning.
  - Triton selection remains OFF by default; may be enabled after A100 parity smokes meet thresholds below.

Parity & Acceptance (A100)
- FA‑2 (cmp/win):
  - Forward MAE vs SDPA < 1e‑5 on representative shapes; no NaN/Inf.
  - Backward stable on toy tasks; grad norms finite.
- Triton selection (forward only):
  - Outputs within tolerance vs packed SDPA; edge cases green (empty ranges, tiny shapes).
  - Fall back on kernel errors; maintain throughput ≥ SDPA on accepted lengths.

Fallbacks & Safety
- FA‑2 outputs are checked for finiteness (NaN/Inf). On failure, route to masked SDPA, log once.
- Triton wrapper normalizes selection ranges to `[B,S,G,n,2]` and falls back to packed SDPA on errors.
- Determinism: M8 tie‑break uses float32 + tiny bias; `sorted=True` everywhere; tests cover float32/16/bf16.
- Causality: strict assertions under `NSA_STRICT_ASSERTS=1` catch window/selection/compressed leaks in smokes.

Version Matrix (Torch ↔ Triton)
- Torch 2.2 → Triton 2.2
- Torch 2.3 → Triton 2.3 (pin `triton>=2.3,<3.1`)
- Torch 2.4+ → Triton 3.x

Observability
- Logs are rate‑limited via `NSA_LOG_LIMIT`. Tags: `decode.reads`, `sel.triton.*`, `fa2.*`, gate stats.
- Heartbeat fields: `loss`, `toks_per_s`, `dt_fetch_s`, `gate_entropy_mean`, `gate_collapse_frac`, `gate_branch_shares`.
