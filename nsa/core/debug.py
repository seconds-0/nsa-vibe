import os
from typing import Any, Dict


def _flag(name: str) -> bool:
    val = os.getenv(name, "0").lower()
    return val in ("1", "true", "yes")


def debug_enabled() -> bool:
    return _flag("NSA_DEBUG_LOG")


_COUNTS: Dict[str, int] = {}


def log(tag: str, **fields: Any) -> None:
    if not debug_enabled():
        return
    limit_env = os.getenv("NSA_LOG_LIMIT")
    if limit_env is not None:
        try:
            limit = int(limit_env)
        except Exception:
            limit = 0
        if limit > 0:
            cnt = _COUNTS.get(tag, 0)
            if cnt >= limit:
                return
            _COUNTS[tag] = cnt + 1
    parts = [f"{k}={_safe(v)}" for k, v in fields.items()]
    print(f"NSA-LOG {tag} " + " ".join(parts))


def _safe(v: Any) -> str:
    try:
        if isinstance(v, int | float | str):
            return str(v)
        if hasattr(v, "shape"):
            return str(tuple(int(x) for x in v.shape))
        return str(v)
    except Exception:
        return "<unrepr>"
