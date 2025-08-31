#!/usr/bin/env python3
"""
Validate critical environment for the single-A100 production run.

Checks (default expectations):
  - FA-2 disabled: NSA_USE_FA2=0; NSA_FA2_MIN_LEN_{WIN,CMP} <= 0
  - Selection varlen disabled: NSA_USE_SEL_VARLEN=0
  - Triton selection disabled: NSA_USE_TRITON_SEL=0
  - Allocator + TF32 configured
  - GPU is A100-class (SM80)

Exit code 0 if OK; non-zero if problems found. Use --strict to fail on warnings.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass


@dataclass
class Issue:
    level: str  # WARN or FAIL
    msg: str


def env_bool(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).lower() in ("1", "true", "yes", "on")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--strict", action="store_true", help="Treat warnings as failures")
    args = ap.parse_args()

    issues: list[Issue] = []

    # FA-2 policy
    if env_bool("NSA_USE_FA2", "0"):
        issues.append(Issue("FAIL", "NSA_USE_FA2 is enabled; expected 0 for A100"))
    for k in ("NSA_FA2_MIN_LEN_WIN", "NSA_FA2_MIN_LEN_CMP"):
        v = os.getenv(k, "-1").strip() or "-1"
        try:
            if int(v) > 0:
                issues.append(Issue("WARN", f"{k} > 0; recommend -1 to hard-disable thresholds"))
        except Exception:
            issues.append(Issue("WARN", f"{k} not integer; recommend -1"))

    # Selection varlen & Triton selection
    if env_bool("NSA_USE_SEL_VARLEN", "0"):
        issues.append(Issue("FAIL", "NSA_USE_SEL_VARLEN is enabled; expected 0"))
    if env_bool("NSA_USE_TRITON_SEL", "0"):
        issues.append(Issue("WARN", "NSA_USE_TRITON_SEL is enabled; recommend 0 for production"))

    # Allocator & TF32
    alloc = os.getenv("PYTORCH_CUDA_ALLOC_CONF", "")
    if "expandable_segments:True" not in alloc:
        issues.append(Issue("WARN", "PYTORCH_CUDA_ALLOC_CONF missing 'expandable_segments:True'"))
    if os.getenv("TORCH_CUDNN_ALLOW_TF32", "0") not in ("1", "true", "yes"):
        issues.append(Issue("WARN", "TORCH_CUDNN_ALLOW_TF32 not enabled (set to 1)"))
    if os.getenv("TORCH_ALLOW_TF32_CUBLAS_OVERRIDE", "0") not in ("1", "true", "yes"):
        issues.append(Issue("WARN", "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE not enabled (set to 1)"))

    # GPU check (best-effort)
    try:
        import torch

        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            cap = torch.cuda.get_device_capability(0)
            if cap != (8, 0):
                issues.append(Issue("WARN", f"Device capability {cap} (name={name}); expected A100 (sm80)"))
        else:
            issues.append(Issue("FAIL", "CUDA not available"))
    except Exception as e:
        issues.append(Issue("WARN", f"torch probe failed: {e}"))

    # Report
    has_fail = any(i.level == "FAIL" for i in issues)
    if issues:
        for it in issues:
            print(f"[{it.level}] {it.msg}")
    else:
        print("[OK] Environment validated for single-A100 production run")

    if has_fail or (args.strict and issues):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

