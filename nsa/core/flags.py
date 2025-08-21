import os
from typing import Optional

import torch


def env_true(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    v = v.strip().lower()
    return v in ("1", "true", "yes", "on")


def env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except Exception:
        return default


def is_sm89(device: Optional[torch.device] = None) -> bool:
    dev = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    if dev.type != "cuda":
        return False
    try:
        cap = torch.cuda.get_device_capability(dev)
        return cap == (8, 9)
    except Exception:
        return False


def torch_triton_version_pairing_ok() -> bool:
    try:
        import triton  # noqa: F401

        tv = triton.__version__
    except ImportError:
        tv = "<none>"
    except Exception:
        tv = "<unknown>"
    try:
        tt = torch.__version__
    except Exception:
        tt = "<unknown>"
    # Basic heuristic: 2.2.x ↔ triton 2.2.x; 2.3.x ↔ 2.3.x; 2.4+ ↔ 3.x
    try:
        major_minor = ".".join((tt or "").split("+")[0].split(".")[:2])
        parts = major_minor.split(".")
        t_major = int(parts[0])
        t_minor = int(parts[1])
        if t_major != 2:
            return True  # do not gate non-2.x
        if t_minor in (2, 3):
            return tv.startswith(f"{t_minor}.")
        if t_minor >= 4:
            return tv.startswith("3.")
        return True
    except (ValueError, IndexError):
        return True


def execution_routing_summary() -> dict:
    """Return a snapshot of routing-related flags and runtime probes."""
    info = {
        "cuda": torch.cuda.is_available(),
        "sm89": is_sm89(),
        "torch": torch.__version__,
    }
    try:
        import triton

        info["triton"] = triton.__version__
    except Exception:
        info["triton"] = "<none>"
    info["NSA_USE_TRITON_SEL"] = env_true("NSA_USE_TRITON_SEL", False)
    info["NSA_TRITON_SEL_FORCE"] = env_true("NSA_TRITON_SEL_FORCE", False)
    info["NSA_USE_FA2"] = env_true("NSA_USE_FA2", False)
    return info
