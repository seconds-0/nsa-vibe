#!/usr/bin/env python3
"""Training watchdog: monitors heartbeat and CSV, halts on anomalies.

Usage:
  python scripts/_watchdog.py --dir artifacts/train_showcase --halt 1

Checks:
  - Heartbeat stall (> NSA_WATCH_HEARTBEAT_STALL_S, default 180s)
  - Throughput flatline (toks/s <= 0 for N consecutive checks)
  - Gate collapse (M8): low entropy, high max gate, or high collapse fraction
  - Optional zero-grad detection if present in heartbeat fields

Environment variables:
  - NSA_WATCH_GATE_COLLAPSE_N: consecutive collapses before halt (default 3)
  - NSA_WATCH_GATE_ENTROPY_MIN: minimum healthy gate entropy (default 0.2)
  - NSA_WATCH_GATE_MAX_THRESHOLD: maximum gate value before collapse (default 0.9)

On anomaly, writes `.anomaly_type` and touches `.HALT` if --halt is truthy.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Optional, Dict, Tuple


def read_last_heartbeat(path: Path) -> Optional[Dict]:
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            back = min(size, 4096)
            f.seek(-back, os.SEEK_SET)
            data = f.read().decode("utf-8", errors="ignore").strip().splitlines()
            for line in reversed(data):
                try:
                    return json.loads(line)
                except Exception:
                    continue
    except Exception:
        return None
    return None


def read_last_csv_row(path: Path) -> Optional[Tuple[int, float, float, float]]:
    try:
        with open(path) as f:
            lines = f.read().strip().splitlines()
            if not lines:
                return None
            last = lines[-1].strip()
            step_s, loss_s, lr_s, tps_s = last.split(",")[:4]
            return int(step_s), float(loss_s), float(lr_s), float(tps_s)
    except Exception:
        return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=str, default="artifacts/train_showcase")
    ap.add_argument("--halt", type=int, default=int(os.getenv("NSA_WATCH_HALT", "1")))
    ap.add_argument(
        "--interval", type=float, default=float(os.getenv("NSA_WATCH_INTERVAL_S", "30"))
    )
    args = ap.parse_args()

    out_dir = Path(args.dir)
    hb_path = None
    # Prefer rank0 heartbeat
    for cand in [out_dir / "heartbeat_rank0.jsonl", out_dir / "heartbeat.jsonl"]:
        if cand.exists():
            hb_path = cand
            break
    if hb_path is None:
        hb_path = out_dir / "heartbeat_rank0.jsonl"  # default path

    csv_path = out_dir / "training.csv"
    anomaly_path = out_dir / ".anomaly_type"
    halt_path = out_dir / ".HALT"

    last_hb_ts = None
    flatline_count = 0
    gate_collapse_count = 0
    HEARTBEAT_STALL = float(os.getenv("NSA_WATCH_HEARTBEAT_STALL_S", "180"))
    FLATLINE_N = int(os.getenv("NSA_WATCH_FLATLINE_N", "4"))
    GATE_COLLAPSE_N = int(os.getenv("NSA_WATCH_GATE_COLLAPSE_N", "3"))
    GATE_ENTROPY_MIN = float(os.getenv("NSA_WATCH_GATE_ENTROPY_MIN", "0.2"))
    GATE_MAX_THRESHOLD = float(os.getenv("NSA_WATCH_GATE_MAX_THRESHOLD", "0.9"))

    print(f"[watch] monitoring dir={out_dir} interval={args.interval}s halt={bool(args.halt)}")
    while True:
        time.sleep(max(1.0, args.interval))

        hb = read_last_heartbeat(hb_path) if hb_path else None
        csv = read_last_csv_row(csv_path)

        now = time.time()
        # Heartbeat stall detection
        if hb and isinstance(hb.get("ts"), (int, float)):
            last_hb_ts = hb["ts"]
        if last_hb_ts is not None and (now - last_hb_ts) > HEARTBEAT_STALL:
            anomaly_path.write_text("heartbeat_stall\n")
            if args.halt:
                halt_path.write_text("halt: heartbeat_stall\n")
            print("[watch] anomaly: heartbeat stall → HALT")
            continue

        # Throughput flatline detection
        if csv is not None:
            _step, _loss, _lr, tps = csv
            if tps <= 0.0:
                flatline_count += 1
            else:
                flatline_count = 0
            if flatline_count >= FLATLINE_N:
                anomaly_path.write_text("throughput_flatline\n")
                if args.halt:
                    halt_path.write_text("halt: throughput_flatline\n")
                print("[watch] anomaly: throughput flatline → HALT")
                continue

        # Gate collapse detection (M8)
        if hb:
            gate_collapsed = False
            gate_entropy_mean = hb.get("gate_entropy_mean")
            gate_max_gate = hb.get("gate_max_gate")
            gate_collapse_frac = hb.get("gate_collapse_frac", 0.0)

            # Check for gate collapse: low entropy OR high max gate OR high collapse fraction
            if gate_entropy_mean is not None and gate_entropy_mean < GATE_ENTROPY_MIN:
                gate_collapsed = True
            elif gate_max_gate is not None and gate_max_gate > GATE_MAX_THRESHOLD:
                gate_collapsed = True
            elif gate_collapse_frac > 0.5:  # More than 50% of gates collapsed
                gate_collapsed = True

            if gate_collapsed:
                gate_collapse_count += 1
            else:
                gate_collapse_count = 0

            if gate_collapse_count >= GATE_COLLAPSE_N:
                anomaly_path.write_text("gate_collapse\n")
                if args.halt:
                    halt_path.write_text("halt: gate_collapse\n")
                # Add gate health details for debugging
                details = {
                    "gate_entropy_mean": gate_entropy_mean,
                    "gate_max_gate": gate_max_gate,
                    "gate_collapse_frac": gate_collapse_frac,
                    "gate_branch_shares": hb.get("gate_branch_shares", []),
                }
                with open(out_dir / "gate_collapse_details.json", "w") as f:
                    json.dump(details, f, indent=2)
                print(
                    f"[watch] anomaly: gate collapse → HALT (entropy={gate_entropy_mean:.3f}, max_gate={gate_max_gate:.3f})"
                )
                continue

        # Optional zero grad detection if present in heartbeat
        if hb and hb.get("grad_norm") == 0:
            anomaly_path.write_text("zero_grad\n")
            if args.halt:
                halt_path.write_text("halt: zero_grad\n")
            print("[watch] anomaly: zero_grad → HALT")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
