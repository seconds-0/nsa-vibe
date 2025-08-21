#!/usr/bin/env python3
"""
Compute a recommended sel_triton_min_L from selection bench CSVs.

Inputs:
  - Dense CSV (few-span) with columns: mode,N,H,D,Dv,L,streams,tri_ms,ref_ms,speedup,mae
  - Varlen CSV (many-span) with columns: mode,N,H,D,Dv,L,nspans,streams,tri_ms,ref_ms,speedup,mae

Output:
  - Prints chosen threshold and summary
  - Optional markdown report via --out
  - Optional config update via --config
"""

import argparse
import csv
from pathlib import Path


def load_csv(path: str | None) -> list[dict]:
    rows: list[dict] = []
    if not path:
        return rows
    p = Path(path)
    if not p.exists():
        return rows
    with p.open() as f:
        r = csv.DictReader(f)
        for row in r:
            # Normalize types
            clean: dict[str, object] = {k: v for k, v in row.items()}
            for k in ("N", "H", "D", "Dv", "L", "nspans", "streams"):
                if k in clean and clean[k] != "":
                    try:
                        clean[k] = int(str(clean[k]))
                    except Exception:
                        pass
            for k in ("tri_ms", "ref_ms", "speedup", "mae"):
                if k in clean and clean[k] != "":
                    try:
                        clean[k] = float(str(clean[k]).replace("x", ""))
                    except Exception:
                        pass
            rows.append(clean)
    return rows


def choose_threshold(
    rows_dense: list[dict], rows_varlen: list[dict], margin: float = 1.2
) -> int | None:
    """Return minimal L where all rows for that L meet speedup >= margin across provided CSVs."""
    Ls = set()
    for r in rows_dense:
        if "L" in r:
            Ls.add(int(r["L"]))
    for r in rows_varlen:
        if "L" in r:
            Ls.add(int(r["L"]))
    if not Ls:
        return None
    for L in sorted(Ls):
        ok = True
        # dense rows
        for r in [
            x for x in rows_dense if int(x.get("L", -1)) == L and int(x.get("streams", 1)) == 1
        ]:
            if float(r.get("speedup", 0.0)) < margin:
                ok = False
                break
        # varlen rows
        if ok:
            for r in [
                x for x in rows_varlen if int(x.get("L", -1)) == L and int(x.get("streams", 1)) == 1
            ]:
                if float(r.get("speedup", 0.0)) < margin:
                    ok = False
                    break
        if ok:
            return L
    return None


def write_report(
    path: Path, dense: list[dict], varlen: list[dict], thr: int | None, margin: float
) -> None:
    with path.open("w") as f:
        f.write("# Selection Triton Threshold Report\n\n")
        f.write(f"Margin: {margin:.2f}x\n\n")
        f.write(f"Recommended sel_triton_min_L: {thr if thr is not None else 'N/A'}\n\n")

        def dump(title: str, rows: list[dict], cols: list[str]):
            f.write(f"## {title}\n\n")
            f.write("| " + " | ".join(cols) + " |\n")
            f.write("|" + "---|" * len(cols) + "\n")
            for r in rows:
                vals = []
                for c in cols:
                    vals.append(str(r.get(c, "")))
                f.write("| " + " | ".join(vals) + " |\n")
            f.write("\n")

        dump(
            "Dense (few)",
            dense,
            ["N", "H", "D", "Dv", "L", "streams", "tri_ms", "ref_ms", "speedup", "mae"],
        )
        dump(
            "Varlen (many)",
            varlen,
            ["N", "H", "D", "Dv", "L", "nspans", "streams", "tri_ms", "ref_ms", "speedup", "mae"],
        )


def maybe_update_config(config_path: str | None, thr: int | None) -> None:
    if not config_path or thr is None:
        return
    try:
        import yaml  # type: ignore
    except Exception:
        print("pyyaml not available; skipping config update")
        return
    p = Path(config_path)
    if not p.exists():
        print(f"Config not found: {config_path}")
        return
    cfg = yaml.safe_load(p.read_text()) or {}
    cfg.setdefault("runtime", {})
    cfg["runtime"]["sel_triton_min_L"] = int(thr)
    p.write_text(yaml.dump(cfg, default_flow_style=False, sort_keys=False))
    print(f"Updated {config_path} sel_triton_min_L={thr}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dense", required=True, help="Path to dense CSV (few)")
    ap.add_argument("--varlen", required=True, help="Path to varlen CSV (many)")
    ap.add_argument(
        "--margin", type=float, default=1.2, help="Minimum speedup to accept (default 1.2x)"
    )
    ap.add_argument("--out", help="Markdown report path")
    ap.add_argument("--config", help="Path to configs/base.yaml to update")
    args = ap.parse_args()

    dense = load_csv(args.dense)
    varlen = load_csv(args.varlen)
    thr = choose_threshold(dense, varlen, margin=args.margin)
    print(
        f"Recommended sel_triton_min_L: {thr if thr is not None else 'N/A'} (margin {args.margin:.2f}x)"
    )
    if args.out:
        write_report(Path(args.out), dense, varlen, thr, args.margin)
        print(f"Report written to {args.out}")
    if args.config and thr is not None:
        maybe_update_config(args.config, thr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
