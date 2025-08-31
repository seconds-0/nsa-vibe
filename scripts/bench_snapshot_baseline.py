#!/usr/bin/env python3
"""
Run the decode benchmark and snapshot a baseline JSON for perf-guard comparisons.

Usage:
  PYTHONPATH=. python scripts/bench_snapshot_baseline.py \
    --csv artifacts/decode_guard.csv \
    --out baselines/a100_decode_guard.json \
    --S_list 512,1024 --iters 16 --warmup 4

Notes:
  - Uses bench/bench_decode.py with --branch_force_mode env (no model weight edits)
  - Writes a minimal JSON mapping {context_len: total_ms}
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
from pathlib import Path


def run_decode_bench(csv_path: Path, S_list: str, iters: int, warmup: int) -> None:
    cmd = [
        "python",
        "bench/bench_decode.py",
        "--S_list",
        S_list,
        "--iters",
        str(iters),
        "--warmup",
        str(warmup),
        "--csv",
        str(csv_path),
        "--branch_force_mode",
        "env",
    ]
    env = os.environ.copy()
    env.setdefault("PYTHONPATH", ".")
    subprocess.run(cmd, check=True, env=env)


def parse_csv(csv_path: Path) -> dict[str, float]:
    values: dict[str, float] = {}
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    # Accept either header names from bench or printed table fields
    # Prefer CSV headers (ms_total)
    if rows and "S" in rows[0] and "ms_total" in rows[0]:
        for r in rows:
            ctx = r["S"].strip()
            try:
                values[ctx] = float(r["ms_total"])
            except Exception:
                continue
    else:
        # Fallback: try Total(ms) column if present
        for r in rows:
            if "S" in r and "Total(ms)" in r:
                ctx = r["S"].strip()
                try:
                    values[ctx] = float(r["Total(ms)"])
                except Exception:
                    continue
    return values


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, default=Path("artifacts/decode_guard.csv"))
    ap.add_argument("--out", type=Path, default=Path("baselines/a100_decode_guard.json"))
    ap.add_argument("--S_list", type=str, default="512,1024")
    ap.add_argument("--iters", type=int, default=16)
    ap.add_argument("--warmup", type=int, default=4)
    args = ap.parse_args()

    args.csv.parent.mkdir(parents=True, exist_ok=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    run_decode_bench(args.csv, args.S_list, args.iters, args.warmup)
    values = parse_csv(args.csv)
    if not values:
        print("[bench-snapshot] No values parsed from CSV; check bench output")
        return 2
    with open(args.out, "w") as f:
        json.dump(values, f, indent=2)
    print(f"[bench-snapshot] Baseline saved → {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

