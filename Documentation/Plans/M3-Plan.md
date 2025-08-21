# M3 — Full Decode Caches (Consolidated Plan)

Status: Completed (per repo); consolidates decode-cache fixes and full decode scheduling plans in alignment with PRD.md.

## Scope & Goals
- Maintain distinct caches: selection raw (`K_sel,V_sel`) and sliding (`K_win,V_win`).
- Compressed stream: emit every d steps after warmup l; ϕ runs on RoPE‑applied K (V unchanged).
- Exact decode read counters per step; observability of selection distances and gates.

## Deliverables
- Decode step updates for all branches; compressed emission schedule; meta growth.
- Counter logging + per‑branch breakdown; deterministic selection at decode.

## Acceptance (from PRD)
- Counters equal exact formula (num_cmp + n·l′ + min(w,S)).
- No future leakage in any branch at decode.

## Tests
- `test_decode_step.py`, `test_decode_counters.py`.
- Long‑context smoke (needle) optional.

## Outcome
Decode path is complete with correct emission, selection, and counters; instrumentation surfaces distances and gate stats.
