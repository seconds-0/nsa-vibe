#!/usr/bin/env python3
"""
Profiler harness: runs a short bf16 A100 smoke with torch.profiler to capture
CUDA+CPU traces and summarize hotspots. Saves a Chrome trace and a text summary
under artifacts/profiler/.

Usage:
  NSA_PREFILL_BATCHED=1 NSA_DISABLE_AUX_STATS=1 \
  CONFIG=configs/m7c_125m_2xa100_production.yaml \
  python -u scripts/profiler_smoke.py --dataset synthetic --steps 200

Notes:
  - Uses default DDP=auto (single process if not launched with torchrun).
  - For multi-GPU, launch with torchrun and profile rank 0 only.
  - The schedule warms up for 10 steps and records the following 50 steps.
"""

import argparse
import os
from pathlib import Path

import torch
from torch.profiler import profile, ProfilerActivity, schedule, tensorboard_trace_handler


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, default="synthetic")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--out", type=str, default="artifacts/profiler")
    args, extra = ap.parse_known_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Ensure recommended stability envs are set unless user overrides
    os.environ.setdefault("NSA_PREFILL_BATCHED", "1")
    os.environ.setdefault("NSA_DISABLE_AUX_STATS", "1")
    # Keep TB/CSV off during profiling
    os.environ.setdefault("NSA_TB_DISABLE", "1")
    os.environ.setdefault("NSA_DISABLE_CSV_LOGS", "1")

    # Optional: enable NVTX ranges in NSA paths
    os.environ.setdefault("NSA_NVTX", "1")

    trace_root = out_dir / "trace"
    trace_root.mkdir(parents=True, exist_ok=True)

    prof_sched = schedule(wait=10, warmup=10, active=50, repeat=1)
    activities = [ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(ProfilerActivity.CUDA)

    with profile(
        activities=activities,
        schedule=prof_sched,
        on_trace_ready=tensorboard_trace_handler(str(trace_root)),
        record_shapes=True,
        with_stack=True,
        profile_memory=True,
    ) as prof:
        # Import here to ensure env vars are in place
        import subprocess, sys

        cmd = [
            sys.executable,
            "-u",
            "scripts/train_showcase.py",
            "--dataset",
            args.dataset,
            "--steps",
            str(args.steps),
        ] + extra
        env = os.environ.copy()
        # Run training as a child process so profiler can wrap this context
        # We keep stepping the profiler in a loop while subprocess runs.
        p = subprocess.Popen(cmd, env=env)
        try:
            import time

            while True:
                if p.poll() is not None:
                    break
                prof.step()
                time.sleep(0.5)
        finally:
            try:
                p.terminate()
            except Exception:
                pass

    # Save a simple text summary
    summary_path = out_dir / "summary.txt"
    try:
        # Re-run a tiny capture to get a table summary
        with profile(
            activities=activities, record_shapes=True, with_stack=False, profile_memory=True
        ) as prof2:
            pass
        # Use the earlier prof object for key_averages if available
        tbl = prof.key_averages().table(
            sort_by="self_cuda_time_total" if torch.cuda.is_available() else "self_cpu_time_total",
            row_limit=200,
        )
        summary_path.write_text(tbl)
    except Exception as e:
        summary_path.write_text(f"Failed to write summary: {e}")

    print(f"[profiler] trace written to {trace_root}; summary at {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
