Start Here — NSA Vibe

What this repo provides
- NSA attention module with CPU fallbacks, FA‑2 wrappers, and a Triton selection path guarded by routing rules.
- Tests and benches you can run locally (CPU) or via a GPU runner.
- Docs and runbooks to validate on GPUs when available.

Quick local sanity
- Create venv, install: python3.10 -m venv .venv && . .venv/bin/activate && pip install -U pip && pip install -r requirements.txt
- Run tests: PYTHONPATH=. pytest -q (GPU tests skip by design)
- Print routing snapshot: python scripts/print_routing.py
- Tiny decode bench: PYTHONPATH=. python bench/bench_decode.py --S_list 8 --iters 3 --warmup 1 --csv decode_test.csv --branch_force_mode env; python bench/summarize_decode_csv.py decode_test.csv

Selection ranges inspection (debug)
- python scripts/print_selection_ranges.py --S 32 --heads 4 --groups 2 --dk 16 --dv 16 --l 8 --d 4 --l_sel 16 --n_sel 4 --w 16 --json

Training showcase (CPU/GPU)
- CONFIG=configs/train_showcase.yaml python scripts/train_showcase.py
- Outputs artifacts to artifacts/train_showcase/ (metrics.json, loss.txt)

GPU validation (when available)
- See Documentation/Runbooks/Runner-Engineer-Checklist.md and Documentation/Test-Plans/GPU-Test-Plan.md
- If using a hosted provider (e.g., Prime Intellect → RunPod), see Documentation/Guides/Remote-GPU-Runner.md

