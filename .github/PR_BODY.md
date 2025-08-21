## Summary
Closeout for M0–M6: decode bench fix, routing/doc polish, runner one-shot, FA‑2 guidance, Triton parity.

## Highlights
- Fix: gate broadcasting in decode path; bench unblocked
- Stability: SDPA fallback nan_to_num; Triton/FA‑2 guards
- DX: one-shot runner script; remote GPU workflow; rich runbook
- Docs: Start-Here, FA‑2 install guide, routing, runner checklist

## Validation
- CPU tests green (3.10/3.11)
- GPU sanity via `scripts/gpu_sanity.py`
- Triton FWD/BWD parity on 4090 with force flags
- Decode bench CSV + summary produced on GPU

## Next
- Calibrate FA‑2 thresholds on 4090 from benches (then update profiles)
- (Optional) Self-hosted GPU CI lane when available
