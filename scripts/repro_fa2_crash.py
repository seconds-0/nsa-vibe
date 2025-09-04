#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Minimal, deterministic FA-2 crash repro + guard suggestions.
Runs dense, varlen, and kvpacked kernels across known brittle preconditions.
Saves a JSON+TXT bundle under artifacts/2025-09-04/fa2_harden/repro/.
"""
import os, json, time, platform, traceback
import torch

# Try latest FA-2 entry points (works across 2.x):
try:
    from flash_attn import (
        flash_attn_func,
        flash_attn_qkvpacked_func,
        flash_attn_varlen_func,
        flash_attn_varlen_qkvpacked_func,
        flash_attn_varlen_kvpacked_func,
    )  # type: ignore
    FA_OK = True
    import flash_attn as _fa  # type: ignore
    FA_VERSION = getattr(_fa, "__version__", "unknown")
except Exception as e:  # pragma: no cover - import probe
    FA_OK = False
    FA_VERSION = f"import-error: {e}"

ARTDIR = "artifacts/2025-09-04/fa2_harden/repro"
os.makedirs(ARTDIR, exist_ok=True)


def env_summary():
    dev = torch.cuda.current_device() if torch.cuda.is_available() else None
    name = torch.cuda.get_device_name(dev) if dev is not None else "cpu"
    cc = torch.cuda.get_device_capability(dev) if dev is not None else (0, 0)
    sm = 10 * cc[0] + cc[1]
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_is_available": torch.cuda.is_available(),
        "device_name": name,
        "sm": sm,
        "driver": torch.version.cuda,
        "flash_attn": FA_VERSION,
        "env": {k: v for k, v in os.environ.items() if k.startswith("NSA_") or k.startswith("CUDA_")},
    }


def log(msg):
    print(f"[repro] {msg}")


def run_case(name, fn):
    rec = {"name": name, "ok": False, "error": None, "trace": None}
    t0 = time.time()
    try:
        fn()
        rec["ok"] = True
    except Exception as e:  # pragma: no cover - interactive repro script
        rec["error"] = repr(e)
        rec["trace"] = traceback.format_exc()
    rec["ms"] = (time.time() - t0) * 1000
    log(f"{name}: {'OK' if rec['ok'] else 'FAIL'} ({rec['ms']:.1f} ms)")
    return rec


def make_qkv(B=2, T=128, H=8, D=128, dtype=torch.float16, contig=True):
    qkv = torch.randn(B, T, 3, H, D, device="cuda", dtype=dtype, requires_grad=True)
    if not contig:
        # non-contiguous via transpose + narrow view (classic repro)
        qkv = qkv.transpose(1, 3)[:, :T]
    return qkv


def case_dense_fp32_crash():
    # fp32 is not supported by FA-2 CUDA kernels; expect failure or fallback
    qkv = make_qkv(dtype=torch.float32)
    flash_attn_qkvpacked_func(qkv, dropout_p=0.0, causal=True)  # type: ignore[name-defined]


def case_noncontig_qkv():
    qkv = make_qkv(dtype=torch.float16, contig=False)
    flash_attn_qkvpacked_func(qkv, dropout_p=0.0, causal=True)  # type: ignore[name-defined]


def _bad_cu_seqlens(nnz, B, dtype=torch.int64, monotonic=False):
    # wrong dtype + non-monotonic to trigger illegal memory access in buggy wrappers
    x = torch.zeros(B + 1, device="cuda", dtype=dtype)
    if monotonic:
        step = nnz // B
        for i in range(1, B + 1):
            x[i] = min(nnz, i * step)
    else:
        # make a violation
        x[1:] = 1
    return x


def case_varlen_invalid_cu_seqlens():
    B, T, H, D = 4, 64, 8, 64
    qkv = make_qkv(B=B, T=T, H=H, D=D, dtype=torch.float16)
    qkv = qkv.reshape(B * T, 3, H, D)  # (nnz, 3, H, D) for varlen qkvpacked
    cu = _bad_cu_seqlens(nnz=B * T, B=B, dtype=torch.int64, monotonic=False)  # int64 + non-monotonic
    flash_attn_varlen_qkvpacked_func(qkv, cu, T, 0.0, causal=True)  # type: ignore[name-defined]


def case_head_dim_sm_limit():
    # Try head_dim=256 on SM8x to demonstrate guard failure (or expect slow/fallback).
    # This should be gated by your fa2_supported_verbose, but we call directly here.
    H, D = 8, 256
    B, T = 1, 128
    qkv = make_qkv(B=B, T=T, H=H, D=D, dtype=torch.bfloat16)
    flash_attn_qkvpacked_func(qkv, 0.0, causal=True)  # type: ignore[name-defined]


def main():
    recs = []
    summ = env_summary()
    log(f"ENV: {json.dumps(summ, indent=2)}")
    if not torch.cuda.is_available():
        log("CUDA not available; nothing to repro.")
    else:
        torch.cuda.synchronize()
        for name, fn in [
            ("dense_fp32", case_dense_fp32_crash),
            ("noncontig_qkv", case_noncontig_qkv),
            ("varlen_invalid_cu_seqlens", case_varlen_invalid_cu_seqlens),
            ("head_dim_sm_limit", case_head_dim_sm_limit),
        ]:
            recs.append(run_case(name, fn))
            torch.cuda.synchronize()
    out = {"env": summ, "results": recs}
    with open(os.path.join(ARTDIR, "repro_results.json"), "w") as f:
        json.dump(out, f, indent=2)
    with open(os.path.join(ARTDIR, "repro_readme.txt"), "w") as f:
        f.write("Repro logs for FA-2 crashes / precondition violations.\n")
    log(f"Wrote artifacts to {ARTDIR}")


if __name__ == "__main__":
    main()

