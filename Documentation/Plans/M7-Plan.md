# M7 — Training Showcase Plan

Status: Proposed

Objective
- Train a tiny, self-contained LLM-style decoder using NSAAttention to demonstrate end-to-end trainability with convergent loss and qualitative samples. Keep it fast, reproducible, and runnable on a single 4090/A100.

Scope
- Model: 0.5–5M params, depth 4–8, dim 128–512, heads 4–8, GQA groups 1–4; context 256–512; byte/char-level vocab.
- Attention: NSAAttention (cmp/sel/win) with PRD defaults (l=32, d=16, l’=64, n=16, w=512). FA‑2 optional; Triton off by default on 4090.
- Data: Tiny Shakespeare (~1MB) or similar; byte-level fallback if tokenizer unavailable.
- Training: Single GPU; BF16/FP32; fixed seeds; configurable batch size/steps.

Deliverables
- Config: configs/train_showcase.yaml (model, optimizer, data, runtime flags).
- Script: reuse scripts/train_toy.py (extend to accept showcase config) or add scripts/train_showcase.py.
- Artifacts:
  - Checkpoint: checkpoints/showcase_final.pt (weights + optimizer state + config).
  - Logs/curves: training.csv (loss, lr, ppl), curves.png (loss vs steps).
  - Samples: samples/before.txt, samples/after.txt (seeded generations at fixed prompt).
  - Report: Documentation/Test-Reports/M7-Training-Showcase-Report.md (env, config, metrics, curves, samples).

Acceptance Criteria
- A1: No NaNs; smooth decreasing loss; run time ≤ 2h on 4090 at ~1–3M params.
- A2: Validation PPL improves ≥ 30% vs init; suggested target PPL < 3.0 on Tiny Shakespeare char-level at ~1–3M params.
- A3: Deterministic: fixed seed; loss variance ≤ ±5% across two reruns.
- A4: Observability: gate mean/std finite; decode counters monotonic in eval; group-consistency/causality tests green.

Runtime Flags
- NSA_USE_FA2=1 (optional), NSA_FA2_MIN_LEN_{WIN,CMP} to avoid tiny-length slowdowns.
- NSA_TRITON_SEL_FORCE=0 (default on 4090). NSA_FORCE_BRANCH for ablations (cmp|sel|win).
- NSA_DEBUG_LOG=0 by default; enable selectively.

Tasks
1) Config + script wiring
- Add configs/train_showcase.yaml; ensure scripts/train_toy.py accepts it.
- Byte-level dataset loader (Tiny Shakespeare) with train/val split and fixed seed shuffling.

2) Sanity passes
- Evaluate forward/backward on a tiny batch; run a 200-step smoke to verify stability.

3) Full run
- Launch training per config; save artifacts and report.

4) Report
- Curves (CSV/PNG), final metrics (train/val loss, PPL), environment (GPU/driver, Torch/Triton/FA-2), samples before/after.

Risks & Mitigation
- FA‑2 instability on 4090 → default to SDPA; thresholds high by default.
- Overfit → val split, early stopping option; small weight decay.
- Throughput variance → fix seeds, stable batch shapes; report exact setup.

Timeline
- Wiring + config: 0.5 day
- Sanity + tuning: 0.5–1 day
- Full run + report: 0.5–1 day

