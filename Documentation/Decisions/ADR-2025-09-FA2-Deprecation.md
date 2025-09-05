ADR-2025-09: SDPA Flash as Default; FA‑2 Optional/Disabled by Default

Context
- Multiple GPU sweeps on H100 (SM90) and A100 (SM80) show PyTorch SDPA flash backend consistently matches or outperforms FlashAttention‑2 (FA‑2) for our NSA shapes and lengths.
- FA‑2 achieved 0.29–0.92× speedup (i.e., 2.4–3.4× slower) on H100 across tested regimes; A100 exhibited similar underperformance with additional head_dim constraints (>128 blocked by default).
- Stability/guarding improved: FA‑2 contracts (dtype, contiguity, head_dim vs SM), varlen cu_seqlens validation, and logging are in place. Startup audit and fallbacks are reliable.

Decision
- SDPA flash is the default attention backend on all supported GPUs.
- FA‑2 is optional and disabled by default for all branches; enable only when explicit perf sweeps demonstrate ≥1.2× speedup for specific shapes/lengths on a given GPU.
- Use large sentinel thresholds (`NSA_FA2_MIN_LEN_WIN=8192`, `NSA_FA2_MIN_LEN_CMP=8192`) when experimenting to prevent accidental use.
- On A100, keep `NSA_FA2_ALLOW_D_GT_128=0` by default.

Rationale
- SDPA flash is a well‑maintained, high‑performance path integrated in PyTorch 2.x. For dense/windowed use cases and our NSA regimes, SDPA flash already saturates GPU capabilities and avoids external kernel variance.
- Empirical data does not support enabling FA‑2 by default today. Guarded optional use remains for future regimes or kernel improvements.

Consequences
- Code defaults: `fa2_win_eff=False`, `fa2_cmp_eff=False` when env unset.
- Runbooks and docs state SDPA‑first; FA‑2 optional/experimental.
- Benches force SDPA flash baseline and sweep longer lengths to avoid bias.

Follow‑ups
- Revisit FA‑2 (or FA‑3 on Hopper) if future kernels demonstrate clear wins in our target regimes.
- Keep FA‑2 parity tests opt‑in (`NSA_TEST_FA2=1`) and exclude from CI by default.

Status
- Adopted and reflected in code, docs, and runbooks as of 2025‑09‑04.

