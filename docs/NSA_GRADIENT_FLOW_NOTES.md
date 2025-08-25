# NSA Gradient Flow Notes (Paper Alignment)

Paper: "Native Sparse Attention" (arXiv:2502.11089v2)

This note clarifies where gradients should and should not flow in our implementation, referencing key equations and their practical implications.

- Selection is discrete (Eq. 11–12):
  - Top‑n block indices are selected per token position.
  - Gradients are not intended to flow through the combinatorial selection itself.
  - Implementation: Indices drive gathers/masks into K/V; gradients flow through attended Q/K/V tensors and the gate MLP, not through integer indices.

- Compressed branch (Eq. 7–8):
  - φ pooling is differentiable; gradients should flow through φ parameters (avg/conv) and K/V projections.
  - Emission schedule (num_cmp) is deterministic and non‑differentiable; only determines how many compressed tokens are visible.

- Mapping M (Eq. 9) and group reduction (Eq. 10):
  - These define how compressed probabilities map to selection blocks and how heads are reduced within a group.
  - When used only to compute indices, scores and mappings can be treated as non‑differentiable products; values used solely for indexing do not require gradients.

Training-oriented implications:

1) When computing selection indices, wrap scoring/mapping in `torch.no_grad()` or explicitly `detach()` before top‑k when gradients are not needed for the loss. This prevents large intermediate tensors (e.g., `p_cmp_all`, `p_slc_all`) from being saved for backward, reducing peak memory.

2) Keep gradients through Q/K/V projections and φ where they affect values consumed by SDPA.

3) Gate MLP should remain differentiable as it mixes branch outputs.

4) Avoid dense masks for large selection ranges during training; prefer packed/gather semantics to mitigate backward memory pressure.

This note documents intent only. Any implementation change to enforce these behaviors should be proposed and reviewed separately.

