# M2 — Trainability & Learnable ϕ (Consolidated Plan)

Status: Completed (per repo); consolidates training/gradcheck and learnable ϕ plans in alignment with PRD.md.

## Scope & Goals
- Replace avg pooling with learnable ϕ (depthwise Conv1d kernel=l, stride=d, causal), optionally followed by tiny MLP.
- Ensure end-to-end training stability; gates learn meaningful mixtures; preserve strict causality.
- Maintain Needle@64k competence.

## Deliverables
- Learnable ϕ codepath in NSAAttention with initialization equal to average pooling for parity.
- Gradcheck on tiny dims; deterministic training smokes; toy LM script.

## Acceptance (from PRD)
- Gradcheck FP32 passes on small shapes; tolerances documented.
- Toy convergence shows decreasing loss (qualitative trend akin to Figure 4).
- Needle@64k retrieval remains perfect with selection configured appropriately.

## Tests & Scripts
- Gradcheck/Backward: `test_gradcheck_varlen.py`, `test_backward_varlen.py` (opt‑in GPU where relevant).
- Training smokes: `test_train_smoke.py`, optional GPU tests behind env flags.
- Script: `scripts/train_toy.py` with base config.

## Flags
- Keep selection eager (no Triton) during training unless explicitly enabled; FA‑2 optional.

## Outcome
Learnable ϕ integrates without breaking causality or selection semantics; training and gradcheck pass on small rigs; parity preserved at init.

