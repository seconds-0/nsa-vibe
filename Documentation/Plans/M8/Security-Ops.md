# Security & Ops Hygiene

Goals: remove hardcoded infra, parameterize ops, protect secrets.

What to implement
- Parameterize remote scripts via env: `REMOTE_HOST`, `SSH_KEY_PATH`, `SSH_OPTS`, `TB_PORT`, `TB_LOGDIR`.
- `.env.example` with safe defaults; document in CLAUDE.md.
- Secrets scanning patterns broadened (HF tokens, SSH keys, cloud creds).
- Access policy and rotation guidance; audit log of remote ops.

Deliverables
- P0: parameterized `monitor_training.sh`, `start_tensorboard_debug.sh` — DONE
- Update CLAUDE.md remote section — DONE

