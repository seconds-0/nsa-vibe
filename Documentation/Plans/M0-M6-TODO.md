M0–M6 Closeout TODO

Status: actionable items to complete production readiness prior to M7

- FA‑2 (M1/M6):
  - Validate Torch 2.4 wheels on 4090; run FA‑2 varlen parity (GPU) and record PASS/SKIP.
  - Calibrate `fa2_min_len_win`/`fa2_min_len_cmp` on 4090 from benches; update `configs/profiles/sm89.yaml`.
- Varlen (M2):
  - GPU regression: run ragged parity `-k collate_varlen` and `-k gradcheck_varlen` (optional) with Torch 2.4 wheels.
- Long-context (M3):
  - Optional GPU smoke at 8k–16k to check trend invariants (attach logs if run).
- Triton (M4/M5):
  - Confirm forward parity + backward parity with force flags on 4090.
  - Decode bench: ensure CSV writes; attach summary.
- Routing/Profiles (M6):
  - Capture routing.json on 4090; ensure ADR respected (triton off by default).
  - Apply updated FA‑2 thresholds to `configs/profiles/sm89.yaml` (post‑bench CSVs).
- Automation:
  - Use `scripts/runner_oneshot.sh` to capture all artifacts under `artifacts/runner/<commit>/`.
  - Optionally run FA‑2 benches via `bench/bench_fa2.py` and generate threshold suggestions.

