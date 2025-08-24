#!/usr/bin/env python3
import json
import os
import sys
from dataclasses import dataclass, asdict
from typing import Optional, Tuple


@dataclass
class EnvReport:
    ok: bool
    reason: Optional[str]
    torch: Optional[str]
    cuda_available: bool
    device_name: Optional[str]
    capability: Optional[Tuple[int, int]]
    tf32_matmul: Optional[bool]
    tf32_cudnn: Optional[bool]
    dtype_policy: Optional[str]


def configure_env(dtype: str | None = None) -> EnvReport:
    try:
        import torch
    except Exception as e:
        return EnvReport(False, f"torch not importable: {e}", None, False, None, None, None, None, None)

    cuda_ok = torch.cuda.is_available()
    name = None
    cap = None
    if cuda_ok:
        try:
            name = torch.cuda.get_device_name(0)
            cap = torch.cuda.get_device_capability(0)
        except Exception:
            pass

    # Configure TF32 for Ampere+ by default (safe for training numerics with bf16/fp32)
    try:
        if cuda_ok:
            torch.backends.cuda.matmul.allow_tf32 = True  # type: ignore
            torch.backends.cudnn.allow_tf32 = True  # type: ignore
    except Exception:
        pass

    # Determine dtype policy
    dtype_policy = None
    if dtype:
        d = dtype.lower()
        if d in ("bf16", "bfloat16"):
            dtype_policy = "bf16"
            # bf16 needs Ampere+ (sm80)
            if cap and (cap[0] < 8 or (cap[0] == 8 and cap[1] < 0)):
                return EnvReport(False, f"bf16 requested but device capability {cap} < sm80", torch.__version__, cuda_ok, name, cap, getattr(torch.backends.cuda.matmul, 'allow_tf32', None), getattr(torch.backends.cudnn, 'allow_tf32', None), dtype_policy)
        elif d in ("fp16", "float16"):
            dtype_policy = "fp16"
        else:
            dtype_policy = "fp32"

    # Guard against known-problematic consumer Ada defaults (sm89) if requested via env
    if cap == (8, 9) and os.getenv("NSA_ALLOW_ADA", "0") not in ("1", "true", "yes"):
        # Not a hard error — training may still run; warn via reason
        return EnvReport(True, "consumer Ada (sm89) detected; prefer A100/H100 for training", torch.__version__, cuda_ok, name, cap, getattr(torch.backends.cuda.matmul, 'allow_tf32', None), getattr(torch.backends.cudnn, 'allow_tf32', None), dtype_policy)

    return EnvReport(True, None, torch.__version__, cuda_ok, name, cap, getattr(torch.backends.cuda.matmul, 'allow_tf32', None), getattr(torch.backends.cudnn, 'allow_tf32', None), dtype_policy)


def main() -> int:
    dtype = os.environ.get("NSA_DTYPE")
    rep = configure_env(dtype)
    print(json.dumps(asdict(rep), indent=2))
    return 0 if rep.ok else 2


if __name__ == "__main__":
    raise SystemExit(main())

