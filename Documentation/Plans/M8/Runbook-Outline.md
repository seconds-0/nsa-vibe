# Runbook Outline (M8)

Purpose: step-by-step ops guide for training on A100 with NSA, from clean node to monitoring and emergency procedures.

- Overview: scope, roles, prerequisites (OS, CUDA, drivers, Python)
- Bootstrap: clone, venv, install (`requirements-gpu-cu121-torch24.txt`), env guard check
- Config: choose `CONFIG`, set precision (bf16/fp32), batch/seq
- Data: FineWeb‑Edu streaming notes; HF cache configuration (`HF_DATASETS_CACHE`); offline local data via `nsa.data_pipeline` JSONL/TXT
- Start: `python -u scripts/train_showcase.py --dataset fineweb_edu --ddp 0 --fwe-report-docs 500 --loader-timeout 120 --synthetic-on-fail`
- Monitor: TensorBoard tunnel (`scripts/run_tensorboard.sh`), `scripts/_watchdog.py --dir artifacts/m7c_125m --halt 1`, `scripts/monitor_training.sh`
- Telemetry: training.csv fields, heartbeat JSONL schema, `dt_fetch_s`
- Emergencies: `.HALT` file to stop, SIGUSR1 stack dump (`kill -USR1 <PID>`), watchdog anomalies (`.anomaly_type`)
- Validation: smokes (1k steps), expected loss trend, throughput checks
- Artifacts: where checkpoints, metrics, env snapshot live; how to resume
- Security/Ops: env vars, secrets hygiene, SSH config

Keep this lean and link to detailed guides where applicable.
