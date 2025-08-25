# Watchdog & Monitoring

Goals: detect stalls/collapses early and stop runs safely.

What to implement
- `scripts/_watchdog.py`: tail heartbeat CSV/JSONL; detect flatline, zero-grad streaks, throughput drop, gate collapse.
- Emit `.anomaly_type` and request `.profile_request` files on anomaly.
- Touch `.HALT` to request trainer graceful stop; trainer polls and exits.
- Robust CSV handling: partial lines, file rotation.

Deliverables
- `scripts/_watchdog.py` — DONE
- Trainer halt/poll hook — DONE

