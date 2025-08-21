Title: Add M7 Training Readiness Test Plan and Runbook

Summary
- Adds a detailed, actionable test plan to validate NSA correctness gates, long-context probes, and training harness readiness for M7C (~125M) training.
- Includes CPU/GPU command matrix, pass/fail expectations, deliverables, and artifacts layout.
- Documents DDP torchrun usage and logging expectations (tokens/sec in training.csv).

Changes
- New: Documentation/Test-Plans/M7-Training-Readiness-Test-Plan.md — step-by-step runbook with commands and deliverables.

How to Run (TL;DR)
- CPU correctness: `PYTHONPATH=. pytest -q -k "test_equiv_small or test_block_math or test_masks or test_group_consistency or test_decode_counters or test_selection_packed"`
- GPU routing + optional Triton/FA‑2 parity: see plan; write outputs under artifacts/test-reports.
- Long-context: `PYTHONPATH=. python scripts/demo_64k.py --S 65536 --prefill_tile 4096 --rope_scale 8.0` and `pytest -q nsa/tests/test_long_context_needle.py`
- DDP training (2 GPUs): `CONFIG=configs/m7c_125m.yaml PYTHONPATH=. torchrun --nproc_per_node=2 scripts/train_showcase.py --dataset fineweb_edu`

Expectations
- All CPU correctness tests PASS.
- GPU parity tests PASS when enabled; otherwise clean SKIPs.
- 64k demo completes with summary; needle test PASS on GPU.
- Trainer writes `training.csv` with `step,loss,lr,toks_per_s` and checkpoints on rank 0.

Deliverables
- Zip/tar of `artifacts/` including `test-reports/`, `bench/`, and training logs/checkpoints.
- Short environment + results summary in the PR comments.

Notes
- Triton selection on SM 8.9 is disabled by policy; force only for parity tests.
- FA‑2 tests require availability; otherwise skip.

