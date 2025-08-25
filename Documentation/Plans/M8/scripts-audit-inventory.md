M8 Scripts & Docs Audit — Initial Inventory (v0)

Legend: keep = OK as-is; fix = parameterize/align; deprecate = remove or replace (add stub during migration)

Scripts (shell/Python)
- scripts/monitor_training.sh — fix: parameterized (DONE in P0); verify remote cmd prefix and CSV paths align
- scripts/start_tensorboard_debug.sh — fix: parameterized (DONE in P0); confirm TB_LOGDIR defaults per profile
- Makefile remote targets — fix: parameterized (DONE in P0); consider directing users to scripts/* instead
- scripts/run_tensorboard.sh — keep: local TB runner with port echo; ensure referenced by Makefile/README
- scripts/train_m7c_prime.sh — fix: verify env guard call, constraints use, dataset flag names; logging destinations
- scripts/automation/create_train_script.sh — fix: installs generic deps; align with constraints and `_env_guard.py`; tmux session naming
- scripts/automation/remote_train_setup.sh — fix: same as above; ensure no hardcoded branches/paths
- scripts/automation/create_setup_script.sh — fix: reduce to minimal setup; avoid global apt where possible
- scripts/bench_report.sh — keep: internal use; ensure markers and paths exist
- scripts/prime_bootstrap.sh — fix: align to constraints-cu121-torch24.txt; print env tuple; no host specifics
- scripts/cleanup_repo.sh — fix: contains rm -rf on patterns; ensure dry-run default and prompt/flag to confirm
- scripts/dev_quickcheck.sh — keep: dev only; ensure no remote calls
- scripts/runner_oneshot.sh — keep: dev helper; safe prints only
- scripts/run_m7_readiness.py — keep: parity tests aggregator; ensure tests paths correct
- scripts/run_milestone_smoke.py — fix: wire to new smoke script and CSV checker
- scripts/print_selection_ranges.py — keep: diagnostics
- scripts/gpu_sanity.py — keep: local GPU sanity
- scripts/run_tensorboard.sh — keep: already parameterized locally

Docs and Makefiles
- CLAUDE.md — fix: P0 done; removed hardcoded host/key; reference env-based setup
- AGENTS.md — fix: add M8 Roadmap; ensure no hardcoded hosts; update remote usage examples to scripts/*
- Makefile — fix: P0 done; confirm all remote targets read env vars and are optional

Reports/Guides with hardcoded hosts/keys (to neutralize or move examples to env-driven)
- NSA_TRAINING_STATUS_REPORT.md — fix refs to ubuntu@216.81.248.82 / ~/.ssh/primeintellect_ed25519
- NSA_PRODUCTION_TRAINING_REPORT.md — same
- NSA_BUG_ANALYSIS.md — same
- M7C_TRAINING_INVESTIGATION.md — same and remote ssh examples
- Documentation/Guides/Selection-Triton-Bench-4090.md — contains a legacy host example
- Documentation/prime-intellect-api-reference.md — keep API docs; ensure no secrets

Open questions / owners
- Training (owner: X): train_m7c_prime.sh alignment with M8 plans (env guard, telemetry, watchdog, data pipeline)
- Infra (owner: Y): automation scripts alignment with constraints and target images; tmux session naming
- Docs (owner: Z): scrub reports/guides for hardcoded hosts/keys and add env-based examples

Acceptance to close audit
- This inventory is reviewed; statuses agreed (keep/fix/deprecate)
- Patches queued for all “fix”; deprecations have stubs or are removed safely
- AGENTS.md and CLAUDE.md reference only parameterized scripts

