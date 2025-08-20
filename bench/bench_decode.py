import time
import csv
import argparse
import numpy as np
import torch

from nsa.core.nsa_attention import NSAAttention
from nsa.cache.kv_cache import NSA_KV
from nsa.core.block_index import build_block_meta


def create_empty_kv(B: int, G: int, d_k: int, d_v: int, meta) -> NSA_KV:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return NSA_KV(
        K_sel=torch.zeros((B, G, 0, d_k), device=device),
        V_sel=torch.zeros((B, G, 0, d_v), device=device),
        K_win=torch.zeros((B, G, 0, d_k), device=device),
        V_win=torch.zeros((B, G, 0, d_v), device=device),
        K_cmp_raw_seq=torch.zeros((B, G, 0, d_k), device=device),
        V_cmp_raw_seq=torch.zeros((B, G, 0, d_v), device=device),
        K_cmp=torch.zeros((B, G, 0, d_k), device=device),
        V_cmp=torch.zeros((B, G, 0, d_v), device=device),
        win_ptr=torch.zeros((B, G), dtype=torch.int32, device=device),
        cmp_emit_next=torch.zeros((B, G), dtype=torch.int32, device=device),
        reads_pred=torch.zeros((0,), dtype=torch.int64, device=device),
        reads_act_total=torch.zeros((0,), dtype=torch.int64, device=device),
        reads_act_sel=torch.zeros((0,), dtype=torch.int64, device=device),
        reads_act_cmp=torch.zeros((0,), dtype=torch.int64, device=device),
        reads_act_win=torch.zeros((0,), dtype=torch.int64, device=device),
        meta=meta,
    )


def compute_expected_reads(S: int, l: int, d: int, n: int, l_sel: int, w: int) -> int:
    num_cmp = 0 if S < l else (S - l) // d + 1
    return num_cmp + n * l_sel + min(w, S)


def _force_branch(model: NSAAttention, which: str) -> None:
    # Force one-hot gating via large bias differences
    if which not in ("cmp", "sel", "win"):
        return
    with torch.no_grad():
        if which == "cmp":
            model.gate.fc2.bias.copy_(torch.tensor([1000.0, -1000.0, -1000.0], device=model.gate.fc2.bias.device))
        elif which == "sel":
            model.gate.fc2.bias.copy_(torch.tensor([-1000.0, 1000.0, -1000.0], device=model.gate.fc2.bias.device))
        else:
            model.gate.fc2.bias.copy_(torch.tensor([-1000.0, -1000.0, 1000.0], device=model.gate.fc2.bias.device))


def benchmark_decode_step():
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, default=1)
    ap.add_argument("--dim", type=int, default=256)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--groups", type=int, default=2)
    ap.add_argument("--dk", type=int, default=32)
    ap.add_argument("--dv", type=int, default=32)
    ap.add_argument("--l", type=int, default=32)
    ap.add_argument("--d", type=int, default=16)
    ap.add_argument("--l_sel", type=int, default=64)
    ap.add_argument("--n_sel", type=int, default=16)
    ap.add_argument("--w", type=int, default=512)
    ap.add_argument("--S_list", type=str, default="128,256,512,1024")
    ap.add_argument("--iters", type=int, default=32)
    ap.add_argument("--warmup", type=int, default=8)
    ap.add_argument("--csv", type=str, default=None)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    nsa = NSAAttention(dim=args.dim, n_heads=args.heads, n_kv_groups=args.groups, d_k=args.dk, d_v=args.dv,
                       l=args.l, d=args.d, l_sel=args.l_sel, n_sel=args.n_sel, w=args.w).to(device)

    S_values = [int(x) for x in args.S_list.split(",")]
    def run_all(writer):
        print(f"\n{'Context':<10} {'Total(ms)':<12} {'cmp(ms)':<10} {'sel(ms)':<10} {'win(ms)':<10} {'Reads':<12}")
        print("-" * 76)

        for S_ctx in S_values:
            x_ctx = torch.randn(args.B, S_ctx, args.dim, device=device)
            meta = build_block_meta(S_ctx + args.w, nsa.l, nsa.d, nsa.l_sel, n_sel=nsa.n_sel, w=nsa.w)
            kv = create_empty_kv(args.B, nsa.n_kv_groups, nsa.d_k, nsa.d_v, meta)
            with torch.no_grad():
                _, kv = nsa(x_ctx, kv, prefill=True)

            def run_decode(model: NSAAttention, kv_state: NSA_KV) -> float:
                times = []
                for step in range(args.iters):
                    x_tok = torch.randn(args.B, 1, args.dim, device=device)
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    with torch.no_grad():
                        _, kv_new = model(x_tok, kv_state, prefill=False)
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    times.append(time.perf_counter() - t0)
                    kv_state = kv_new
                return float(np.mean(times[args.warmup:]) * 1000)

            # total
            model_total = nsa
            ms_total = run_decode(model_total, kv)

            # cmp-only
            model_cmp = NSAAttention(args.dim, args.heads, args.groups, args.dk, args.dv, args.l, args.d, args.l_sel, args.n_sel, args.w).to(device)
            model_cmp.load_state_dict(nsa.state_dict(), strict=False)
            _force_branch(model_cmp, "cmp")
            ms_cmp = run_decode(model_cmp, kv)

            # sel-only
            model_sel = NSAAttention(args.dim, args.heads, args.groups, args.dk, args.dv, args.l, args.d, args.l_sel, args.n_sel, args.w).to(device)
            model_sel.load_state_dict(nsa.state_dict(), strict=False)
            _force_branch(model_sel, "sel")
            ms_sel = run_decode(model_sel, kv)

            # win-only
            model_win = NSAAttention(args.dim, args.heads, args.groups, args.dk, args.dv, args.l, args.d, args.l_sel, args.n_sel, args.w).to(device)
            model_win.load_state_dict(nsa.state_dict(), strict=False)
            _force_branch(model_win, "win")
            ms_win = run_decode(model_win, kv)

            S_current = S_ctx + args.iters
            expected = compute_expected_reads(S_current, nsa.l, nsa.d, nsa.n_sel, nsa.l_sel, nsa.w)
            actual = kv.reads_act_total[-1].item() if kv.reads_act_total.numel() else -1
            print(f"{S_ctx:<10} {ms_total:<12.2f} {ms_cmp:<10.2f} {ms_sel:<10.2f} {ms_win:<10.2f} {actual}/{expected}")
            if writer:
                writer.writerow([S_ctx, f"{ms_total:.3f}", f"{ms_cmp:.3f}", f"{ms_sel:.3f}", f"{ms_win:.3f}", int(actual), int(expected)])

    if args.csv:
        with open(args.csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["S", "ms_total", "ms_cmp", "ms_sel", "ms_win", "reads_actual", "reads_expected"])
            run_all(writer)
    else:
        run_all(None)


if __name__ == "__main__":
    benchmark_decode_step()
