# NSA Chunked Selection Scoring: Design Spec

Goal: Eliminate O(S²) memory in prefill selection scoring by computing scores and selections in chunks along the query (time) axis with bounded working set, preserving NSA semantics (Eq. 7–12).

Symbols
- B: batch size
- S: sequence length
- G: KV groups
- h: heads per group
- Dk/Dv: head dims
- S_cmp ~ ceil((S − l)/d) + 1 (compressed tokens)
- S_sel ~ ceil(S / l_sel) (selection blocks)
- T: chunk size along the query axis (S = sum of T_i)

Current Pain Points (O(S²))
- `p_cmp_all`: [B, S, G, h, S_cmp]
- `p_slc_all`: [B, S, G, h, S_sel]
- `p_grp_all`: [B, S, G, S_sel]
- Masked SDPA for selection/sliding can create [B, G*h, S, S_kv]

High‑Level Approach
1. Compute Q and branches’ K/V for the full sequence once (as today), but stream scoring per chunk of queries.
2. For each chunk t ∈ [t0, t1):
   - Build `Q_chunk = Q[:, t0:t1]` [B, T, G, h, Dk]
   - Compute `p_cmp_chunk = softmax(Q_chunk @ K_cmp^T)` → [B, T, G, h, S_cmp]
   - Map via Eq. 9 for chunk: `p_slc_chunk = p_cmp_chunk @ M_csl` → [B, T, G, h, S_sel]
   - Group-reduce heads → `p_grp_chunk` [B, T, G, S_sel]
   - For each time t in chunk: top‑n selection over valid ≤ t blocks, convert to ranges.
   - Run selection/sliding/compressed attention for this chunk (prefer packed/gather/FA‑2 varlen, not masked), compute gates for chunk, and produce chunk outputs.
   - Discard chunk intermediates before moving on.

Memory Characteristics
- Working set per chunk: O(T·S_cmp) and O(T·S_sel) instead of O(S·S_*).
- Peak memory becomes O(T·S) with small T (linear in S overall when T is fixed by memory budget).

Correctness & Semantics
- Exactness: Same math as full batched computation for each time index; chunk boundaries do not alter top‑k results because each t handles its full set of valid selection blocks.
- Determinism: Preserve tie‐breaking and forced blocks exactly as current implementation.
- Causality: Valid block mask per t remains identical (≤ t+1).

Integration Points (nsa/core/nsa_attention.py)
- `_forward_prefill_batched` and `_forward_prefill_sequential` both currently build full `p_*_all`. Replace with chunk loop:
  - Introduce a local chunk size `T = env_int("NSA_CHUNK_T", default=128)`.
  - Reuse existing functions (`compute_pcmp`, `map_pcmp_to_pslc`) on chunk slices.
  - Reuse existing selection conversion to ranges and attention kernels per chunk.
  - Keep gating and output projection per chunk, append to output buffer.

Backends
- Selection: default to packed or gather parity for training; avoid dense masked path.
- Sliding: prefer FA‑2 varlen; else gather parity.
- Compressed: FA‑2 compressed varlen if available; else gather parity.

Performance Notes
- Compute is still O(S·S_cmp) but memory and peak activation footprint reduce to O(T·S) with small T.
- Choose `T` to balance kernel efficiency and memory headroom (128 or 256 typical for A100 80GB at S=2048, dim=768).

Validation Plan
- Parity vs baseline for small S: compare outputs (MAE) and identical selection ranges under determinism flag.
- Memory scaling: measure peak reserved across S={512,1024,2048} for T={64,128,256}; show sub‑quadratic scaling.
- Throughput: report tokens/s vs current implementation at S=512.

Risks & Mitigations
- More kernel launches per chunk: amortize by using packed/varlen FA‑2; minimize Python overhead.
- Top‑k per t within chunks: ensure efficient vectorized top‑k across T.
- Bookkeeping complexity: encapsulate chunked scoring/mapping in a helper to keep `NSAAttention` readable.

