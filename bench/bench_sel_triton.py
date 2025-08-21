import argparse
import os
import time

import torch


def _env_true(name: str, default: bool = False) -> bool:
    v = os.getenv(name, "1" if default else "0").lower()
    return v in ("1", "true", "yes", "on")


def make_single_span(N: int, L: int, device: torch.device) -> torch.Tensor:
    ranges = torch.zeros((N, 1, 2), device=device, dtype=torch.long)
    ranges[:, 0, 0] = 0
    ranges[:, 0, 1] = L
    return ranges


def make_multi_span(N: int, L: int, n: int, device: torch.device) -> torch.Tensor:
    # Split L into n contiguous spans of equal size (last may be longer)
    base = L // n
    rem = L - base * n
    starts = []
    pos = 0
    for i in range(n):
        ln = base + (1 if i < rem else 0)
        starts.append((pos, pos + ln))
        pos += ln
    ranges = torch.zeros((N, n, 2), device=device, dtype=torch.long)
    for i, (s, e) in enumerate(starts):
        ranges[:, i, 0] = s
        ranges[:, i, 1] = e
    return ranges


def _maybe_write_csv(csv_path: str | None, row: dict) -> None:
    if not csv_path:
        return
    import csv

    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            w.writeheader()
        w.writerow(row)


def run_dense(
    N: int,
    H: int,
    D: int,
    Dv: int,
    L: int,
    iters: int,
    warmup: int,
    streams: int,
    csv: str | None,
    device: torch.device,
) -> None:
    torch.manual_seed(0)
    Q = torch.randn(N, H, D, device=device, dtype=torch.float16)
    K = torch.randn(N, L, D, device=device, dtype=torch.float16)
    V = torch.randn(N, L, Dv, device=device, dtype=torch.float16)

    from nsa.kernels.triton_sel_kernel.sel_fwd import sel_attn_fwd_dense, sel_attn_fwd_dense_group

    # Warmup
    for _ in range(warmup):
        if streams <= 1:
            if _env_true("NSA_SEL_TRITON_GROUP", False):
                sel_attn_fwd_dense_group(Q, K, V)
            else:
                sel_attn_fwd_dense(Q, K, V)
        else:
            s = [torch.cuda.Stream() for _ in range(streams)]
            with torch.cuda.stream(s[0]):
                if _env_true("NSA_SEL_TRITON_GROUP", False):
                    sel_attn_fwd_dense_group(Q, K, V)
                else:
                    sel_attn_fwd_dense(Q, K, V)
            for st in s[1:]:
                with torch.cuda.stream(st):
                    # Launch identical work to exercise concurrent kernels
                    if _env_true("NSA_SEL_TRITON_GROUP", False):
                        sel_attn_fwd_dense_group(Q, K, V)
                    else:
                        sel_attn_fwd_dense(Q, K, V)
            torch.cuda.synchronize()
    # Timed runs
    if streams <= 1:
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            if _env_true("NSA_SEL_TRITON_GROUP", False):
                O_tri = sel_attn_fwd_dense_group(Q, K, V)
            else:
                O_tri = sel_attn_fwd_dense(Q, K, V)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        ms_tri = (t1 - t0) * 1e3 / iters
    else:
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            s = [torch.cuda.Stream() for _ in range(streams)]
            outs = []
            for st in s:
                with torch.cuda.stream(st):
                    if _env_true("NSA_SEL_TRITON_GROUP", False):
                        outs.append(sel_attn_fwd_dense_group(Q, K, V))
                    else:
                        outs.append(sel_attn_fwd_dense(Q, K, V))
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        ms_tri = (t1 - t0) * 1e3 / iters / streams

    # Reference: packed SDPA through wrapper-by-bucket
    from nsa.core.attention_kernels import grouped_selection_attention_packed

    # Build reference with B=N, S=1, G=1 so each row is independent
    Qw = Q.unsqueeze(1).unsqueeze(1)  # [N,1,1,H,D]
    Kw = K.unsqueeze(1)  # [N,1,L,D]
    Vw = V.unsqueeze(1)  # [N,1,L,Dv]
    ranges = make_single_span(N, L, device).unsqueeze(1).unsqueeze(1)  # [N,1,1,1,2]
    for _ in range(warmup):
        grouped_selection_attention_packed(Qw, Kw, Vw, ranges)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        O_ref = grouped_selection_attention_packed(Qw, Kw, Vw, ranges)
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    ms_ref = (t1 - t0) * 1e3 / iters

    mae = (O_tri - O_ref.squeeze(1).squeeze(1)).abs().float().mean().item()
    print(
        f"dense,N={N},H={H},D={D},Dv={Dv},L={L},streams={streams}, tri_ms={ms_tri:.3f}, ref_ms={ms_ref:.3f}, speedup={ms_ref / ms_tri:.2f}x, mae={mae:.4e}"
    )
    _maybe_write_csv(
        csv,
        {
            "mode": "dense",
            "N": N,
            "H": H,
            "D": D,
            "Dv": Dv,
            "L": L,
            "streams": streams,
            "tri_ms": f"{ms_tri:.4f}",
            "ref_ms": f"{ms_ref:.4f}",
            "speedup": f"{ms_ref / ms_tri:.4f}",
            "mae": f"{mae:.6e}",
        },
    )


def run_varlen(
    N: int,
    H: int,
    D: int,
    Dv: int,
    L: int,
    nspans: int,
    iters: int,
    warmup: int,
    streams: int,
    csv: str | None,
    device: torch.device,
) -> None:
    torch.manual_seed(0)
    Q = torch.randn(N, H, D, device=device, dtype=torch.float16)
    # Build concatenated K/V by spans
    ranges = make_multi_span(N, L, nspans, device)  # [N,n,2]
    total_L = N * L
    K_all = torch.empty(total_L, D, device=device, dtype=torch.float16)
    V_all = torch.empty(total_L, Dv, device=device, dtype=torch.float16)
    write = 0
    for i in range(N):
        # For benchmark, fill continuous block for row i
        K_row = torch.randn(L, D, device=device, dtype=torch.float16)
        V_row = torch.randn(L, Dv, device=device, dtype=torch.float16)
        K_all[write : write + L] = K_row
        V_all[write : write + L] = V_row
        write += L
    cu = torch.arange(0, (N + 1) * L, step=L, device=device, dtype=torch.int32)

    from nsa.kernels.triton_sel_kernel.sel_fwd import sel_attn_fwd_varlen, sel_attn_fwd_varlen_group

    # Warmup
    for _ in range(warmup):
        if streams <= 1:
            if _env_true("NSA_SEL_TRITON_GROUP", False):
                sel_attn_fwd_varlen_group(Q, K_all, V_all, cu)
            else:
                sel_attn_fwd_varlen(Q, K_all, V_all, cu)
        else:
            s = [torch.cuda.Stream() for _ in range(streams)]
            for st in s:
                with torch.cuda.stream(st):
                    if _env_true("NSA_SEL_TRITON_GROUP", False):
                        sel_attn_fwd_varlen_group(Q, K_all, V_all, cu)
                    else:
                        sel_attn_fwd_varlen(Q, K_all, V_all, cu)
            torch.cuda.synchronize()
    if streams <= 1:
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            if _env_true("NSA_SEL_TRITON_GROUP", False):
                O_tri = sel_attn_fwd_varlen_group(Q, K_all, V_all, cu)
            else:
                O_tri = sel_attn_fwd_varlen(Q, K_all, V_all, cu)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        ms_tri = (t1 - t0) * 1e3 / iters
    else:
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            s = [torch.cuda.Stream() for _ in range(streams)]
            for st in s:
                with torch.cuda.stream(st):
                    if _env_true("NSA_SEL_TRITON_GROUP", False):
                        sel_attn_fwd_varlen_group(Q, K_all, V_all, cu)
                    else:
                        sel_attn_fwd_varlen(Q, K_all, V_all, cu)
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        ms_tri = (t1 - t0) * 1e3 / iters / streams

    # Reference via packed SDPA wrapper
    from nsa.core.attention_kernels import grouped_selection_attention_packed

    # Build per-row K/V tensors; [N,L,D*]
    K_rows = torch.empty(N, L, D, device=device, dtype=torch.float16)
    V_rows = torch.empty(N, L, Dv, device=device, dtype=torch.float16)
    for i in range(N):
        K_rows[i] = K_all[i * L : (i + 1) * L]
        V_rows[i] = V_all[i * L : (i + 1) * L]
    # Reference with B=N, S=1, G=1
    Qw = Q.unsqueeze(1).unsqueeze(1)  # [N,1,1,H,D]
    Kw = K_rows.unsqueeze(1)  # [N,1,L,D]
    Vw = V_rows.unsqueeze(1)  # [N,1,L,Dv]
    rangesw = ranges.unsqueeze(1).unsqueeze(1)  # [N,1,1,n,2]
    for _ in range(warmup):
        grouped_selection_attention_packed(Qw, Kw, Vw, rangesw)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        O_ref = grouped_selection_attention_packed(Qw, Kw, Vw, rangesw)
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    ms_ref = (t1 - t0) * 1e3 / iters

    mae = (O_tri - O_ref.squeeze(1).squeeze(1)).abs().float().mean().item()
    print(
        f"varlen,N={N},H={H},D={D},Dv={Dv},L={L},n={nspans},streams={streams}, tri_ms={ms_tri:.3f}, ref_ms={ms_ref:.3f}, speedup={ms_ref / ms_tri:.2f}x, mae={mae:.4e}"
    )
    _maybe_write_csv(
        csv,
        {
            "mode": "varlen",
            "N": N,
            "H": H,
            "D": D,
            "Dv": Dv,
            "L": L,
            "nspans": nspans,
            "streams": streams,
            "tri_ms": f"{ms_tri:.4f}",
            "ref_ms": f"{ms_ref:.4f}",
            "speedup": f"{ms_ref / ms_tri:.4f}",
            "mae": f"{mae:.6e}",
        },
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--N", type=int, default=256)
    p.add_argument("--H", type=int, default=8)
    p.add_argument("--D", type=int, default=64)
    p.add_argument("--Dv", type=int, default=64)
    p.add_argument("--L_list", type=str, default="64,128,256,512")
    p.add_argument("--dist", type=str, choices=["few", "many", "mixed"], default="few")
    p.add_argument("--iters", type=int, default=50)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--streams", type=int, default=1)
    p.add_argument("--csv", type=str, default=None)
    p.add_argument("--decode", type=int, default=0)
    args = p.parse_args()

    assert torch.cuda.is_available(), "CUDA GPU required"
    device = torch.device("cuda")
    L_list: list[int] = [int(x) for x in args.L_list.split(",")]

    # Dense few-long case
    if args.dist == "few":
        for L in L_list:
            run_dense(
                args.N,
                args.H,
                args.D,
                args.Dv,
                L,
                args.iters,
                args.warmup,
                args.streams,
                args.csv,
                device,
            )
    else:
        # Varlen multi-span case
        for L in L_list:
            nspans = 8 if args.dist == "many" else 4
            run_varlen(
                args.N,
                args.H,
                args.D,
                args.Dv,
                L,
                nspans,
                args.iters,
                args.warmup,
                args.streams,
                args.csv,
                device,
            )


if __name__ == "__main__":
    main()
