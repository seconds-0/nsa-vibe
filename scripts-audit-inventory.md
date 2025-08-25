# Scripts Audit Inventory (M8)

Status map for scripts folder. Update as audit progresses.

- scripts/start_tensorboard_debug.sh — KEEP (parameterized)
- scripts/monitor_training.sh — KEEP (parameterized TB path)
- scripts/run_tensorboard.sh — FIX (prefer env overrides; OK currently)
- scripts/automation/README.md — FIX (P0 - had hardcoded ubuntu@216.81.248.82, now fixed)
- scripts/automation/* — FIX (review remaining for hardcoded paths/hosts)  
- scripts/train_showcase.py — FIX (add telemetry + watchdog polling)
- scripts/automation/fwe_smoke.py — KEEP (migrated to nsa.data_pipeline)
- scripts/datasets/check_fwe_stream.py — KEEP (migrated to nsa.data_pipeline)
- scripts/test_1024_loader.py — KEEP (migrated to nsa.data_pipeline)
- scripts/datasets/fineweb_edu_loader.py — REMOVED (replaced by nsa.data_pipeline)
- scripts/prime_bootstrap.sh — REVIEW (ensure idempotent, parameterized)

Notes
- Use `.env.example` for env variables. Avoid embedding hosts/tokens.
- P0 security issues: Found 11 files with hardcoded ubuntu@216.81.248.82 or primeintellect_ed25519
- Non-script files (reports, guides) also need review for hardcoded values
