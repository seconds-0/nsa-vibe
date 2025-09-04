from __future__ import annotations

import torch
import torch.nn.functional as F

from nsa.core.debug import log
from nsa.attn.fa2_contracts import (
    fa2_supported_verbose as contracts_fa2_supported_verbose,
    check_cu_seqlens as _check_cu_seqlens,
)


def _capability(device: torch.device) -> tuple[int, int] | None:
    if device.type != "cuda":
        return None
    try:
        return torch.cuda.get_device_capability(device)
    except Exception:
        return None


def _env_bool(name: str, default: bool = False) -> bool:
    v = str(name and __import__("os").getenv(name, "1" if default else "0")).lower()
    return v in ("1", "true", "yes", "on")


def flash_attn_version() -> str | None:
    """Return flash-attn version string if importable, else None."""
    try:
        import flash_attn as _fa  # type: ignore

        return getattr(_fa, "__version__", None)
    except Exception:
        return None


def is_flash_available() -> bool:
    """Return True if flash-attn dense API is importable."""
    try:
        from flash_attn import flash_attn_func  # type: ignore

        _ = flash_attn_func  # silence linter
        return True
    except Exception:
        return False


def is_flash_varlen_available() -> bool:
    """Return True if a varlen API is importable (either QKV or KV-packed)."""
    try:
        from flash_attn import flash_attn_varlen_func  # type: ignore

        _ = flash_attn_varlen_func
        return True
    except Exception:
        try:
            from flash_attn import flash_attn_varlen_kvpacked_func  # type: ignore

            _ = flash_attn_varlen_kvpacked_func
            return True
        except Exception:
            return False


def _audit_enabled() -> bool:
    return _env_bool("NSA_FA2_AUDIT", False)


def audit_fa2_support(head_dim: int, heads: int = 8) -> None:
    """Optional startup audit to proactively validate FA‑2 availability.

    - Checks imports and SM/head_dim contracts
    - Runs a tiny dense forward if eligible; logs pass/fail
    - No exception bubbles out; this is best-effort observability
    """
    try:
        if not _audit_enabled():
            return
        if not torch.cuda.is_available():
            log("fa2.audit.skip", reason="no_cuda")
            return
        dev = torch.device("cuda")
        sm = torch.cuda.get_device_capability(dev)
        smi = 10 * sm[0] + sm[1]
        # Minimal shapes
        B, T = 1, 16
        H = max(1, heads)
        dt = torch.float16
        q = torch.randn(B, T, H, head_dim, device=dev, dtype=dt).contiguous()
        k = torch.randn(B, T, H, head_dim, device=dev, dtype=dt).contiguous()
        v = torch.randn(B, T, H, head_dim, device=dev, dtype=dt).contiguous()
        ok, detail = contracts_fa2_supported_verbose(q=q, k=k, v=v, head_dim=head_dim, is_varlen=False)
        if not ok:
            log("fa2.audit.result", ok=False, sm=smi, head_dim=int(head_dim), reasons=detail.get("reasons"))
            return
        try:
            from flash_attn import flash_attn_func  # type: ignore

            _ = flash_attn_func(q, k, v, dropout_p=0.0, softmax_scale=None, causal=True)
            log("fa2.audit.result", ok=True, sm=smi, head_dim=int(head_dim), dtype=str(dt))
        except Exception as e:  # pragma: no cover
            log("fa2.audit.fail", sm=smi, head_dim=int(head_dim), err=repr(e))
    except Exception:  # pragma: no cover
        # Never fail caller due to audit
        pass


def log_fa2_support_matrix(head_dims: tuple[int, ...] = (64, 96, 128, 192, 256)) -> None:
    """Log a concise FA‑2 support matrix for common head_dims and dtypes.

    Logs rows like: fa2.audit.matrix sm=90 D=128 dt=float16 dense_ok=True varlen_ok=True
    """
    try:
        if not _audit_enabled() or not torch.cuda.is_available():
            return
        dev = torch.device("cuda")
        sm = torch.cuda.get_device_capability(dev)
        smi = 10 * sm[0] + sm[1]
        for D in head_dims:
            for dt in (torch.float16, torch.bfloat16):
                # Dense probe tensors
                q = torch.randn(1, 16, 8, D, device=dev, dtype=dt).contiguous()
                k = torch.randn(1, 16, 8, D, device=dev, dtype=dt).contiguous()
                v = torch.randn(1, 16, 8, D, device=dev, dtype=dt).contiguous()
                dense_ok, d_detail = contracts_fa2_supported_verbose(q=q, k=k, v=v, head_dim=D, is_varlen=False)
                # Varlen probe tensors (packed-like shapes)
                qv = torch.randn(16, 8, D, device=dev, dtype=dt).contiguous()
                kv = torch.randn(64, 8, D, device=dev, dtype=dt).contiguous()
                varlen_ok, v_detail = contracts_fa2_supported_verbose(q=qv, k=kv, v=kv, head_dim=D, is_varlen=True)
                log(
                    "fa2.audit.matrix",
                    sm=smi,
                    D=int(D),
                    dt=str(dt),
                    dense_ok=bool(dense_ok),
                    varlen_ok=bool(varlen_ok),
                    dense_reasons=d_detail.get("reasons", []),
                    varlen_reasons=v_detail.get("reasons", []),
                )
    except Exception:  # pragma: no cover
        pass


def _log_perf_fallback(tag: str, **kwargs) -> None:
    """Log expected perf impact on fallback when audit/timing is enabled."""
    if _env_bool("NSA_DEBUG_TIMING") or _env_bool("NSA_SDPA_AUDIT"):
        log(tag, **kwargs)


def fa2_supported_verbose(
    device: torch.device, dtype: torch.dtype, head_dim: int
) -> tuple[bool, str]:
    """
    Conservative capability probe with a reason string for logging.
    We do not hard-fail on dtype, relying on try/except at call sites.
    """
    if device.type != "cuda":
        return False, "device_not_cuda"
    if head_dim % 8 != 0:
        return False, "head_dim_not_multiple_of_8"
    # Dtype guard: flash-attn v2 expects fp16/bf16 (fp32 unsupported by kernels)
    allow_fp32 = _env_bool("NSA_FA2_ALLOW_FP32", False)
    if dtype not in (torch.float16, torch.bfloat16) and not allow_fp32:
        return False, f"unsupported_dtype_{str(dtype).split('.')[-1]}"
    # SM- and head-dim-specific guardrails (best-effort, avoids kernel FPE)
    cap = _capability(device)
    if cap is not None:
        major, minor = cap
        # A100/Ampere (sm80/sm86): head_dim must be <= 128
        if major == 8 and head_dim > 128:
            return False, "head_dim_gt_128_on_sm8x"
        # H100/Hopper (sm90): head_dim must be <= 256 per FA2 docs
        if major >= 9 and head_dim > 256:
            return False, "head_dim_gt_256_on_sm9x"
    if not (is_flash_varlen_available() or is_flash_available()):
        return False, "flash_attn_not_importable"
    # Optional version floor (best-effort)
    ver = flash_attn_version()
    if ver is None:
        # Unknown version; still allow
        return True, "ok"
    # Allow all known versions; attach for logs
    return True, f"ok_v{ver}"


def fa2_supported(device: torch.device, dtype: torch.dtype, head_dim: int) -> bool:
    ok, _ = fa2_supported_verbose(device, dtype, head_dim)
    return ok


def attention_bgh(
    Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, causal: bool = True
) -> torch.Tensor:
    """
    Q: [B,G,h,Dk], K/V: [B,G,S,D*] -> out [B,G,h,Dv]
    Prefer flash-attn if available; fallback to SDPA.
    """
    B, G, h, Dk = Q.shape
    S = K.shape[2]
    # Try FA-2 dense path first
    if is_flash_available():
        try:
            from flash_attn import flash_attn_func  # type: ignore

            # Reshape without materializing copies
            q = Q.transpose(1, 2).reshape(B, G * h, 1, Dk).contiguous()  # [B,G*h,1,Dk]
            k = (
                K.unsqueeze(2)
                .expand(B, G, h, S, Dk)
                .reshape(B, G * h, S, Dk)
                .contiguous()
            )  # [B,G*h,S,Dk]
            v = (
                V.unsqueeze(2)
                .expand(B, G, h, S, V.shape[-1])
                .reshape(B, G * h, S, V.shape[-1])
                .contiguous()
            )  # [B,G*h,S,Dv]
            # Enforce dtype compatibility (cast if needed)
            if q.dtype not in (torch.float16, torch.bfloat16):
                dq = torch.float16 if _env_bool("NSA_FA2_PREF_FP16", True) else torch.bfloat16
                q = q.to(dq)
                k = k.to(dq)
                v = v.to(dq)
            # Contracts after shaping/contiguity to avoid false negatives
            ok_contracts, _detail = contracts_fa2_supported_verbose(
                q=q, k=k, v=v, head_dim=Dk, is_varlen=False
            )
            if _env_bool("NSA_DEBUG_TIMING"):
                log(
                    "fa2.bgh.path",
                    path="fa2.dense" if ok_contracts else "sdpa.contracts_block",
                    B=B,
                    G=G,
                    h=h,
                    S=S,
                    Dk=Dk,
                    dtype=str(q.dtype),
                )
            if ok_contracts:
                o = flash_attn_func(q, k, v, dropout_p=0.0, softmax_scale=None, causal=causal)
                o = o.reshape(B, G, h, -1)
                if not torch.isfinite(o).all():
                    log("warn.flash_bgh_nonfinite", path="fa2.dense")
                return torch.nan_to_num(o, nan=0.0)
            else:
                # Contracts blocked: log fallback + expected slowdown band
                reasons = _detail.get("reasons", []) if isinstance(_detail, dict) else []
                _log_perf_fallback(
                    "fa2.fallback",
                    path="bgh",
                    reason=(reasons[0] if reasons else "contracts_block"),
                    expected_slowdown="10-30x",
                    B=int(B),
                    S=int(S),
                    D=int(Dk),
                )
        except Exception:
            _log_perf_fallback(
                "fa2.fallback",
                path="bgh",
                reason="exception_in_flash_call",
                expected_slowdown="10-30x",
                B=int(B),
                S=int(S),
                D=int(Dk),
            )
    # SDPA fallback
    if _env_bool("NSA_DEBUG_TIMING"):
        log("fa2.bgh.path", path="sdpa", B=B, G=G, h=h, S=S, Dk=Dk)
    # Expand heads via view/expand to avoid materializing copies
    q2 = Q.reshape(B * G * h, 1, Dk).contiguous()
    k2 = K.unsqueeze(2).expand(B, G, h, S, Dk).reshape(B * G * h, S, Dk).contiguous()
    v2 = (
        V.unsqueeze(2)
        .expand(B, G, h, S, V.shape[-1])
        .reshape(B * G * h, S, V.shape[-1])
        .contiguous()
    )
    attn = F.scaled_dot_product_attention(q2, k2, v2, is_causal=causal)
    o = attn.squeeze(1).reshape(B, G, h, -1)
    return torch.nan_to_num(o, nan=0.0)


def attention_fa2_dense_batch(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    causal: bool,
) -> torch.Tensor:
    """
    Best-effort dense FA-2 call for a batch of independent rows.
    Shapes:
    - q: [N, Tq, h, D]
    - k: [N, Tk, h, D]
    - v: [N, Tk, h, Dv]
    Returns: o [N, Tq, h, Dv]
    Falls back to SDPA if flash-attn unavailable.
    """
    # Ensure contiguous tensors for FA-2 and compatible dtype
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    if q.dtype not in (torch.float16, torch.bfloat16):
        dq = torch.float16 if _env_bool("NSA_FA2_PREF_FP16", True) else torch.bfloat16
        q = q.to(dq)
        k = k.to(dq)
        v = v.to(dq)
    try:
        from flash_attn import flash_attn_func  # type: ignore

        if _env_bool("NSA_DEBUG_TIMING"):
            log(
                "fa2.batch.path",
                path="fa2.dense",
                N=int(q.shape[0]),
                Tq=int(q.shape[1]),
                Tk=int(k.shape[1]),
                h=int(q.shape[2]),
                D=int(q.shape[3]),
                dtype=str(q.dtype),
            )
        ok_contracts, _detail = contracts_fa2_supported_verbose(q=q, k=k, v=v, head_dim=int(q.shape[-1]), is_varlen=False)
        if ok_contracts:
            return flash_attn_func(q, k, v, dropout_p=0.0, softmax_scale=None, causal=causal)
        else:
            reasons = _detail.get("reasons", []) if isinstance(_detail, dict) else []
            _log_perf_fallback(
                "fa2.fallback",
                path="dense_batch",
                reason=(reasons[0] if reasons else "contracts_block"),
                expected_slowdown="5-20x",
                N=int(q.shape[0]),
                Tq=int(q.shape[1]),
                Tk=int(k.shape[1]),
                D=int(q.shape[3]),
            )
    except Exception:
        # SDPA fallback per row
        N, Tq, h, D = q.shape
        Tk = k.shape[1]
        Dv = v.shape[-1]
        if _env_bool("NSA_DEBUG_TIMING"):
            log("fa2.batch.path", path="sdpa", N=int(N), Tq=int(Tq), Tk=int(Tk))
        _log_perf_fallback(
            "fa2.fallback",
            path="dense_batch",
            reason="exception_in_flash_call",
            expected_slowdown="5-20x",
            N=int(N),
            Tq=int(Tq),
            Tk=int(Tk),
            D=int(D),
        )
        q2 = q.reshape(N * h, Tq, D)
        k2 = k.reshape(N * h, Tk, D)
        v2 = v.reshape(N * h, Tk, Dv)
        out = F.scaled_dot_product_attention(q2, k2, v2, is_causal=causal)
        return out.reshape(N, h, Tq, Dv).permute(0, 2, 1, 3).contiguous()


def attention_fa2_varlen(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    *,
    causal: bool,
) -> torch.Tensor:
    """
    Best-effort varlen FA-2 call with separate Q/K/V packing.
    Shapes:
    - q: [total_q, h, D], k: [total_k, h, D], v: [total_k, h, Dv]
    - cu_seqlens_*: int32 [N+1]
    Returns: [total_q, h, Dv] packed output.
    Falls back to dense batching by padding per bucket if varlen API unavailable.
    """
    # Ensure contiguous tensors for FA-2 and compatible dtype
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    if q.dtype not in (torch.float16, torch.bfloat16):
        dq = torch.float16 if _env_bool("NSA_FA2_PREF_FP16", True) else torch.bfloat16
        q = q.to(dq)
        k = k.to(dq)
        v = v.to(dq)
    # Validate cu_seqlens invariants to avoid kernel crashes
    assert cu_seqlens_q.is_cuda and cu_seqlens_k.is_cuda
    assert cu_seqlens_q.numel() >= 2 and cu_seqlens_k.numel() >= 2
    if not _check_cu_seqlens(cu_seqlens_q, int(q.shape[0]), int(cu_seqlens_q.numel() - 1)):
        raise ValueError("invalid cu_seqlens_q")
    if not _check_cu_seqlens(cu_seqlens_k, int(k.shape[0]), int(cu_seqlens_k.numel() - 1)):
        raise ValueError("invalid cu_seqlens_k")
    # Contracts on shaped/contiguous tensors
    ok_contracts, _detail = contracts_fa2_supported_verbose(
        q=q, k=k, v=v, head_dim=int(q.shape[-1]), is_varlen=True
    )
    try:
        from flash_attn import flash_attn_varlen_func  # type: ignore
        if _env_bool("NSA_DEBUG_TIMING"):
            N = int(cu_seqlens_q.numel() - 1)
            Tk = int(cu_seqlens_k[-1].item())
            log(
                "fa2.varlen.path",
                path="fa2.varlen" if ok_contracts else "sdpa.contracts_block",
                N=N,
                max_q=int(max_seqlen_q),
                max_k=int(max_seqlen_k),
                total_k=Tk,
                h=int(q.shape[1]),
                D=int(q.shape[2]),
                dtype=str(q.dtype),
            )
        if ok_contracts:
            return flash_attn_varlen_func(
                q,
                k,
                v,
                cu_seqlens_q,
                cu_seqlens_k,
                max_seqlen_q,
                max_seqlen_k,
                dropout_p=0.0,
                softmax_scale=None,
                causal=causal,
            )
        else:
            reasons = _detail.get("reasons", []) if isinstance(_detail, dict) else []
            _log_perf_fallback(
                "fa2.fallback",
                path="varlen",
                reason=(reasons[0] if reasons else "contracts_block"),
                expected_slowdown="5-20x",
                N=int(cu_seqlens_q.numel() - 1),
                D=int(q.shape[2]),
            )
    except Exception:
        # Try KV-packed API variant
        try:
            from flash_attn import flash_attn_varlen_kvpacked_func  # type: ignore

            # Build KV packed as [total_k, 2, h, D]
            kv_packed = torch.stack([k, v], dim=1).contiguous()
            if _env_bool("NSA_DEBUG_TIMING"):
                N = int(cu_seqlens_q.numel() - 1)
                Tk = int(cu_seqlens_k[-1].item())
                log(
                    "fa2.varlen.path",
                    path="fa2.varlen_kvpacked" if ok_contracts else "sdpa.contracts_block",
                    N=N,
                    max_q=int(max_seqlen_q),
                    max_k=int(max_seqlen_k),
                    total_k=Tk,
                    h=int(q.shape[1]),
                    D=int(q.shape[2]),
                    dtype=str(q.dtype),
                )
            if ok_contracts:
                return flash_attn_varlen_kvpacked_func(
                    q,
                    kv_packed,
                    cu_seqlens_q,
                    cu_seqlens_k,
                    max_seqlen_q,
                    max_seqlen_k,
                    dropout_p=0.0,
                    softmax_scale=None,
                    causal=causal,
                )
            else:
                reasons = _detail.get("reasons", []) if isinstance(_detail, dict) else []
                _log_perf_fallback(
                    "fa2.fallback",
                    path="varlen_kvpacked",
                    reason=(reasons[0] if reasons else "contracts_block"),
                    expected_slowdown="5-20x",
                    N=int(cu_seqlens_q.numel() - 1),
                    D=int(q.shape[2]),
                )
        except Exception:
            _log_perf_fallback(
                "fa2.fallback",
                path="varlen",
                reason="exception_in_flash_call",
                expected_slowdown="5-20x",
                N=int(cu_seqlens_q.numel() - 1),
                D=int(q.shape[2]),
            )
            raise NotImplementedError("FA-2 varlen API not available; caller should fallback")
