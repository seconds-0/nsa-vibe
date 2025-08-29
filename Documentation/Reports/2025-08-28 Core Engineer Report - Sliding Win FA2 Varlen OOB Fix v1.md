Subject: Fix CUDA illegal memory access in sliding-window FA-2 varlen path and honor FA‑2 disable flags

Summary
- Root cause: In `sliding_window_attention_fa2` varlen path, `cu_seqlens_q` (cuq) was allocated but never populated. Passing uninitialized `cuq` into `flash_attn_varlen` can lead to out‑of‑bounds reads and CUDA illegal memory access. This explains the crash observed at `nsa/core/attention_kernels.py:469` on FA‑2 varlen execution.
- Secondary issue: `NSA_USE_FA2=0` did not reliably disable FA‑2 in prefill because the code OR’d the env flag with `use_flash_default`, inadvertently enabling FA‑2 even when explicitly disabled. This masked the intended diagnostic and routed into the buggy FA‑2 path.

Changes
1) Safety fix: fill `cuq` correctly in varlen packing.
   - File: `nsa/core/attention_kernels.py`
   - Section: `sliding_window_attention_fa2` (varlen-all-rows path)
   - Patch: set `cuq = [0, 1, 2, ..., N]` via `cuq.copy_(torch.arange(0, N+1, dtype=int32))`; keep `cuk` as the cumsum of per-row window lengths.

2) Flag semantics fix: honor explicit FA‑2 disable even when defaults are on.
   - File: `nsa/core/nsa_attention.py`
   - `_cache_env_vars`: detect whether `NSA_USE_FA2`, `NSA_USE_FA2_WIN`, `NSA_USE_FA2_CMP` are explicitly set. Compute effective flags with hard‑disable semantics:
     - If `NSA_USE_FA2` is set to 0, force `fa2_all_eff = fa2_win_eff = fa2_cmp_eff = False`.
     - If not set, fall back to `use_flash_default` for `fa2_all_eff`; branch flags default to False unless explicitly set.
   - Use `fa2_*_eff` everywhere in prefill/decode routing instead of raw flags; remove the previous `or self.use_flash_default` leakage.

Why this fixes the issue
- With correct `cuq`, FA‑2 varlen receives valid sequence starts, preventing reads outside the packed buffers. The illegal memory access at line 469 was a symptom of this.
- With strict flag handling, setting `NSA_USE_FA2=0` now disables FA‑2 codepaths (both win/cmp) regardless of model defaults. This makes isolating SDPA paths and reproducing issues deterministic.

Validation (suggested)
- Minimal parity check (CPU, no FA‑2):
  - `python - <<'PY'` snippet compares `sliding_window_attention_fa2` vs `sliding_window_attention_masked` and should print `close True` for small shapes.
- GPU + FA‑2 varlen probe:
  - Ensure flash‑attn is installed per runbook; verify probe prints “FA2 varlen OK”.
  - Run `NSA_USE_FA2=0` to confirm FA‑2 paths are disabled (counters should show zero FA‑2 attempts).
  - Run `NSA_USE_FA2=1` and a short sliding‑window smoke (e.g., S=128) to verify no crashes; monitor `isfinite(o_pack)` check remains true.

Files touched
- `nsa/core/attention_kernels.py`: cuq initialization in FA‑2 sliding varlen path.
- `nsa/core/nsa_attention.py`: FA‑2 flag caching and effective gating; prefill/decode routing updated to use `fa2_*_eff`.

Acceptance checklist
- Causality and group consistency unaffected (mask semantics unchanged).
- CPU fallback preserved; masked SDPA paths untouched.
- FA‑2 remains opt‑in; `NSA_USE_FA2=0` now strictly disables it even if `use_flash_default=True`.

Next steps
- Run the 200‑step single‑A100 smoke with `NSA_USE_FA2=1` and verify throughput gate (≥ 300 tok/s) and stable fetch p95. If IO spikes, adjust `NSA_FWE_DOC_BATCH` / `NSA_FWE_Q` as planned.
- If any FA‑2 related fallback counters rise, capture logs under `artifacts/` and link them in a follow‑up report.

