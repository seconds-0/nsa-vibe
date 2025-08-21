# M1 — FlashAttention‑2 Integration (Consolidated Plan)

Status: Completed (per repo); consolidates FA‑2 varlen/dense plans and thresholds, aligned to PRD.md.

## Scope & Goals
- Integrate FA‑2 for sliding and compressed branches; keep SDPA reference and robust fallbacks.
- Maintain numerical parity vs SDPA oracle within FP32 ≤ 5e‑5 (BF16/FP16 ≤ 2e‑4 when tested).
- Threshold gating to prefer FA‑2 only where it outperforms masked SDPA.

## Deliverables
- Attention wrappers with FA‑2 varlen and dense paths and fallback ladder to masked SDPA.
- Length bucketing + cu_seqlens utilities; workspace packing to reduce allocations.
- Threshold optimizer and benches to select `fa2_min_len_win` and `fa2_min_len_cmp`.

## Acceptance (from PRD)
- Parity tests pass for cmp/win FA‑2 vs SDPA (tolerances above).
- Head_dim/device constraints guarded; CPU path uses SDPA.
- Config stores safe thresholds; environment flags allow overrides.

## Tests & Benches
- Parity: `test_fa2_parity.py`, `test_fa2_parity_improved.py` (GPU opt‑in).
- GPU varlen smoke: `test_fa2_gpu_varlen.py` (opt‑in).
- Benches: `bench/bench_fa2.py`, with outputs `fa2_win.txt`, `fa2_cmp.txt`, summarized in `fa2_thresholds.md`.

## Flags & Config
- Enable: `NSA_USE_FA2=1` or branch‑specific `NSA_USE_FA2_WIN/CMP`.
- Minimum lengths: `NSA_FA2_MIN_LEN_WIN`, `NSA_FA2_MIN_LEN_CMP` (bench‑tuned); defaults conservative on RTX 4090 per ADR.
- Fallbacks if unsupported or below thresholds: masked SDPA.

## Outcome
FA‑2 is integrated with safe guards and falls back to masked SDPA where not beneficial (notably RTX 4090, SM 8.9). Config defaults remain conservative unless explicit override.

