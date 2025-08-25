# Data Pipeline Hardening

Goals: robust HF streaming on A100, clear diagnostics, safe fallbacks.

What to implement
- `FineWebStream(IterableDataset)`: retries with backoff; deterministic sharding; tokenizer swap (byte/gpt2); windowing to seq_len.
- First batch timeout with helpful error and synthetic fallback option.
- Progress hooks: docs/sec, tokens/sec; `_dt_fetch` metric to telemetry.
- Local/offline loaders: JSONL and TXT for emergency mode.

Deliverables
- `nsa/data_pipeline.py` with stream + local datasets — DONE
- CLI flags in `train_showcase.py` to choose dataset/tokenizer — DONE

