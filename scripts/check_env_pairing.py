#!/usr/bin/env python3
import json
import sys


def main():
    try:
        import torch
    except Exception as e:
        print(json.dumps({"ok": False, "error": f"torch import failed: {e}"}, indent=2))
        return 1
    try:
        import triton
        triton_ver = triton.__version__
    except Exception:
        triton_ver = None

    torch_ver = torch.__version__
    pairing_ok = True
    reason = ""
    try:
        major_minor = ".".join(torch_ver.split("+")[0].split(".")[:2])
        t_major, t_minor = map(int, major_minor.split("."))
        if t_major == 2 and t_minor in (2, 3):
            pairing_ok = triton_ver is not None and triton_ver.startswith(str(t_minor) + ".")
            if not pairing_ok:
                reason = f"expected triton {t_minor}.x for torch {torch_ver}, got {triton_ver}"
        elif t_major == 2 and t_minor >= 4:
            pairing_ok = triton_ver is not None and triton_ver.startswith("3.")
            if not pairing_ok:
                reason = f"expected triton 3.x for torch {torch_ver}, got {triton_ver}"
    except Exception as e:
        pairing_ok = False
        reason = f"pairing check failed: {e}"

    out = {
        "torch": torch_ver,
        "triton": triton_ver or "<none>",
        "pairing_ok": pairing_ok,
        "reason": reason,
    }
    print(json.dumps(out, indent=2))
    return 0 if pairing_ok else 2


if __name__ == "__main__":
    sys.exit(main())

