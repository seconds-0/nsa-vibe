import os
import torch


def _sm() -> int:
    if not torch.cuda.is_available():
        return 0
    try:
        c = torch.cuda.get_device_capability()
        return 10 * c[0] + c[1]
    except Exception:
        return 0


def _env_bool(name: str, default: bool = False) -> bool:
    v = os.getenv(name, "1" if default else "0").lower()
    return v in ("1", "true", "yes", "on")


def fa2_supported_verbose(
    q: torch.Tensor | None = None,
    k: torch.Tensor | None = None,
    v: torch.Tensor | None = None,
    qkv: torch.Tensor | None = None,
    head_dim: int | None = None,
    is_varlen: bool = False,
):
    sm = _sm()
    ten = qkv if qkv is not None else (q if q is not None else (k if k is not None else v))
    hd = head_dim if head_dim is not None else (ten.shape[-1] if ten is not None else None)  # type: ignore[index]
    dtype_ok = ten is not None and ten.dtype in (torch.float16, torch.bfloat16)
    contig_ok = (
        (qkv is not None and qkv.is_contiguous())
        or (
            q is not None
            and k is not None
            and v is not None
            and q.is_contiguous()
            and k.is_contiguous()
            and v.is_contiguous()
        )
    )
    allow_d_gt_128 = _env_bool("NSA_FA2_ALLOW_D_GT_128", False)
    hd_ok = (
        hd is not None
        and (hd % 8 == 0)
        and (
            (sm >= 90 and hd <= 256)
            or (sm < 90 and (hd <= 128 or (allow_d_gt_128 and hd <= 256)))
        )
    )
    reasons: list[str] = []
    if not dtype_ok:
        reasons.append("dtype must be fp16/bf16")
    if not contig_ok:
        reasons.append("non-contiguous tensor(s)")
    if not hd_ok:
        reasons.append(
            f"head_dim={hd} invalid for SM{sm}; need %8==0 and ≤{256 if sm>=90 else (128 if not allow_d_gt_128 else 256)}"
        )
    return (dtype_ok and contig_ok and hd_ok), {"sm": sm, "hd": hd, "reasons": reasons}


def check_cu_seqlens(cu: torch.Tensor, nnz: int, B: int) -> bool:
    if cu.dtype != torch.int32:
        return False
    if cu.numel() != B + 1:
        return False
    if int(cu[0].item()) != 0:
        return False
    if int(cu[-1].item()) != nnz:
        return False
    try:
        return bool(torch.all(cu[1:] >= cu[:-1]))
    except Exception:
        return False
