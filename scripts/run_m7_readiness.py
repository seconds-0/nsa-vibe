#!/usr/bin/env python3
"""
Run the M7 Training Readiness Test Plan end-to-end and collect artifacts.

This script is resilient:
- Creates an artifacts/ tree with subfolders for test-reports, train, bench
- Tries GPU tests if CUDA is available; otherwise records SKIP
- Optional Triton / FA-2 parity tests via flags; skips cleanly if unavailable
- Captures outputs and produces a summary JSON with pass/fail/skips and exit codes

Usage examples:
  PYTHONPATH=. python scripts/run_m7_readiness.py
  PYTHONPATH=. python scripts/run_m7_readiness.py --out artifacts/run_$(date +%Y%m%d-%H%M) --enable-triton --enable-fa2
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple


def run_cmd(cmd: List[str], outfile: Path, env: Dict[str, str] | None = None) -> Tuple[int, float]:
    outfile.parent.mkdir(parents=True, exist_ok=True)
    ts = time.time()
    with open(outfile, "w", encoding="utf-8", errors="ignore") as f:
        try:
            proc = subprocess.run(
                cmd,
                stdout=f,
                stderr=subprocess.STDOUT,
                env=env,
                cwd=str(Path.cwd()),
                text=True,
            )
            rc = proc.returncode
        except Exception as e:
            f.write(f"[runner] exception: {e}\n")
            rc = 127
    dt = time.time() - ts
    return rc, dt


def has_module(name: str) -> bool:
    try:
        __import__(name)
        return True
    except Exception:
        return False


def cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def cuda_device_count() -> int:
    try:
        import torch

        return int(torch.cuda.device_count())
    except Exception:
        return 0


def is_sm89() -> bool:
    try:
        import torch

        if not torch.cuda.is_available():
            return False
        cap = torch.cuda.get_device_capability(0)
        return cap == (8, 9)
    except Exception:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out", type=str, default="artifacts/run", help="Base output directory for artifacts"
    )
    ap.add_argument(
        "--enable-triton", action="store_true", help="Run Triton selection parity tests if possible"
    )
    ap.add_argument(
        "--enable-fa2", action="store_true", help="Run FA-2 varlen parity tests if available"
    )
    ap.add_argument("--skip-long", action="store_true", help="Skip long-context demo/tests")
    args = ap.parse_args()

    out_base = Path(args.out)
    tr_dir = out_base / "test-reports"
    train_dir = out_base / "train"
    bench_dir = out_base / "bench"
    for p in (tr_dir, train_dir, bench_dir):
        p.mkdir(parents=True, exist_ok=True)

    summary = {"steps": []}
    env_base = os.environ.copy()
    env_base.setdefault("PYTHONPATH", ".")

    # 1) CPU correctness suite
    step = {"name": "cpu-correctness"}
    rc, dt = run_cmd(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "-k",
            "test_equiv_small or test_block_math or test_masks or test_group_consistency or test_decode_counters or test_selection_packed",
        ],
        tr_dir / "cpu-correctness.txt",
        env=env_base,
    )
    step["exit_code"] = rc
    step["duration_s"] = round(dt, 2)
    step["status"] = "ok" if rc == 0 else "failed"
    summary["steps"].append(step)

    # 2) GPU routing + optional Triton/FA-2
    gpu = cuda_available()
    step = {"name": "gpu-routing"}
    rc, dt = run_cmd(
        [sys.executable, "scripts/print_routing.py"], tr_dir / "routing.json", env=env_base
    )
    step["exit_code"] = rc
    step["duration_s"] = round(dt, 2)
    step["status"] = "ok" if rc == 0 else "failed"
    summary["steps"].append(step)

    if args.enable_triton and gpu:
        env_tri = env_base.copy()
        env_tri["NSA_USE_TRITON_SEL"] = "1"
        env_tri["NSA_TEST_TRITON_SEL"] = "1"
        if is_sm89():
            env_tri["NSA_TRITON_SEL_FORCE"] = "1"
        step = {"name": "triton-fwd"}
        rc, dt = run_cmd(
            [sys.executable, "-m", "pytest", "-q", "nsa/tests/test_triton_sel_parity_gpu.py"],
            tr_dir / "triton_fwd.txt",
            env=env_tri,
        )
        step.update(
            {"exit_code": rc, "duration_s": round(dt, 2), "status": "ok" if rc == 0 else "failed"}
        )
        summary["steps"].append(step)

        env_tri_b = env_tri.copy()
        env_tri_b["NSA_SEL_TRITON_ALLOW_GRAD"] = "1"
        step = {"name": "triton-bwd"}
        rc, dt = run_cmd(
            [sys.executable, "-m", "pytest", "-q", "nsa/tests/test_triton_sel_backward_gpu.py"],
            tr_dir / "triton_bwd.txt",
            env=env_tri_b,
        )
        step.update(
            {"exit_code": rc, "duration_s": round(dt, 2), "status": "ok" if rc == 0 else "failed"}
        )
        summary["steps"].append(step)
    else:
        summary["steps"].append({"name": "triton-fwd", "status": "skipped"})
        summary["steps"].append({"name": "triton-bwd", "status": "skipped"})

    if args.enable_fa2 and gpu and has_module("nsa.kernels.flash_wrappers"):
        # Probe FA-2 availability
        probe_txt = tr_dir / "fa2_probe.txt"
        with open(probe_txt, "w", encoding="utf-8") as f:
            try:
                from nsa.kernels.flash_wrappers import is_flash_varlen_available

                f.write(f"fa2_varlen_available {is_flash_varlen_available()}\n")
            except Exception as e:
                f.write(f"fa2_varlen_available exception {e}\n")
        step = {"name": "fa2-parity"}
        env_fa2 = env_base.copy()
        env_fa2["NSA_TEST_FA2"] = "1"
        env_fa2["NSA_USE_FA2"] = "1"
        rc, dt = run_cmd(
            [sys.executable, "-m", "pytest", "-q", "-k", "fa2_gpu_varlen"],
            tr_dir / "fa2_varlen.txt",
            env=env_fa2,
        )
        step.update(
            {"exit_code": rc, "duration_s": round(dt, 2), "status": "ok" if rc == 0 else "failed"}
        )
        summary["steps"].append(step)
    else:
        summary["steps"].append({"name": "fa2-parity", "status": "skipped"})

    # 3) Long-context probes (64k)
    if not args.skip_long and gpu:
        step = {"name": "demo-64k"}
        rc, dt = run_cmd(
            [
                sys.executable,
                "scripts/demo_64k.py",
                "--S",
                "65536",
                "--prefill_tile",
                "4096",
                "--rope_scale",
                "8.0",
                "--use_fa2",
                "0",
            ],
            tr_dir / "demo_64k.txt",
            env=env_base,
        )
        step.update(
            {"exit_code": rc, "duration_s": round(dt, 2), "status": "ok" if rc == 0 else "failed"}
        )
        summary["steps"].append(step)

        step = {"name": "needle-64k"}
        rc, dt = run_cmd(
            [
                sys.executable,
                "-m",
                "pytest",
                "-q",
                "nsa/tests/test_long_context_needle.py",
                "-k",
                "needle",
            ],
            tr_dir / "needle_64k.txt",
            env=env_base,
        )
        step.update(
            {"exit_code": rc, "duration_s": round(dt, 2), "status": "ok" if rc == 0 else "failed"}
        )
        summary["steps"].append(step)
    else:
        summary["steps"].append({"name": "demo-64k", "status": "skipped"})
        summary["steps"].append({"name": "needle-64k", "status": "skipped"})

    # 4) Trainer readiness
    step = {"name": "train-single"}
    env_train = env_base.copy()
    env_train["CONFIG"] = "configs/train_showcase.yaml"
    rc, dt = run_cmd(
        [sys.executable, "scripts/train_showcase.py"],
        train_dir / "train_showcase.txt",
        env=env_train,
    )
    step.update(
        {"exit_code": rc, "duration_s": round(dt, 2), "status": "ok" if rc == 0 else "failed"}
    )
    summary["steps"].append(step)

    # DDP only if >= 2 GPUs
    if cuda_device_count() >= 2:
        step = {"name": "train-ddp-m7c"}
        env_ddp = env_base.copy()
        env_ddp["CONFIG"] = "configs/m7c_125m.yaml"
        # Prefer torchrun; fallback to python -m torch.distributed.run
        cmd = ["torchrun", "--nproc_per_node=2", "scripts/train_showcase.py"]
        if has_module("datasets") and has_module("transformers"):
            cmd += ["--dataset", "fineweb_edu"]
        rc, dt = run_cmd(cmd, train_dir / "m7c_ddp.txt", env=env_ddp)
        if rc != 0:
            cmd = [
                sys.executable,
                "-m",
                "torch.distributed.run",
                "--nproc_per_node=2",
                "scripts/train_showcase.py",
            ]
            if has_module("datasets") and has_module("transformers"):
                cmd += ["--dataset", "fineweb_edu"]
            rc, dt = run_cmd(cmd, train_dir / "m7c_ddp.txt", env=env_ddp)
        step.update(
            {"exit_code": rc, "duration_s": round(dt, 2), "status": "ok" if rc == 0 else "failed"}
        )
        summary["steps"].append(step)
    else:
        summary["steps"].append({"name": "train-ddp-m7c", "status": "skipped"})

    # 5) Bench/telemetry
    bench_csv = bench_dir / "decode.csv"
    step = {"name": "bench-decode"}
    rc, dt = run_cmd(
        [
            sys.executable,
            "bench/bench_decode.py",
            "--S_list",
            "512,1024,2048,4096",
            "--iters",
            "32",
            "--warmup",
            "8",
            "--csv",
            str(bench_csv),
        ],
        bench_dir / "decode.txt",
        env=env_base,
    )
    step.update(
        {"exit_code": rc, "duration_s": round(dt, 2), "status": "ok" if rc == 0 else "failed"}
    )
    summary["steps"].append(step)
    # Summarize CSV if present
    step = {"name": "bench-decode-summary"}
    if bench_csv.exists():
        rc, dt = run_cmd(
            [sys.executable, "scripts/summarize_bench.py", str(bench_csv)],
            bench_dir / "decode_summary.txt",
            env=env_base,
        )
        step.update(
            {"exit_code": rc, "duration_s": round(dt, 2), "status": "ok" if rc == 0 else "failed"}
        )
    else:
        step.update({"status": "skipped"})
    summary["steps"].append(step)

    # Final summary JSON
    with open(out_base / "run_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Console summary
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
