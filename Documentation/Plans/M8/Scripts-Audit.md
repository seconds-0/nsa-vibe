# Scripts Audit

Goals: inventory scripts; mark keep/fix/deprecate; prepare for runbook.

Heuristics
- Flag scripts with hardcoded hosts/paths, unguarded `ssh`, missing timeouts, or direct secrets.
- Prefer env‑driven parameters and shared helpers.

Initial Inventory (sample)
- keep: `scripts/start_tensorboard_debug.sh`, `scripts/monitor_training.sh` (now parameterized)
- fix: `scripts/automation/*` (audit for hardcoded paths); `scripts/run_tensorboard.sh` (ensure env overrides)
- deprecate: legacy one‑offs once runbook lands

Outputs
- See `scripts-audit-inventory.md` at repo root — PENDING

