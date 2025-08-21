#!/usr/bin/env python3
import sys
import csv


def summarize(csv_paths):
    for path in csv_paths:
        try:
            with open(path, newline="") as f:
                rows = list(csv.reader(f))
        except Exception as e:
            print(f"ERROR: cannot read {path}: {e}")
            continue
        if not rows:
            print(f"{path}: empty")
            continue
        header = rows[0]
        cols = {name: i for i, name in enumerate(header)}
        req = ["S", "ms_total", "ms_cmp", "ms_sel", "ms_win", "reads_actual", "reads_expected"]
        missing = [c for c in req if c not in cols]
        if missing:
            print(f"{path}: missing columns {missing}")
            continue
        print(f"\n=== {path} ===")
        print(
            f"{'S':>6}  {'ms_total':>8}  {'cmp':>8}  {'sel':>8}  {'win':>8}  {'reads':>12}  {'exp':>8}"
        )
        total_ms = []
        for r in rows[1:]:
            S = int(r[cols["S"]])
            mt = float(r[cols["ms_total"]])
            mc = float(r[cols["ms_cmp"]])
            ms = float(r[cols["ms_sel"]])
            mw = float(r[cols["ms_win"]])
            ra = int(r[cols["reads_actual"]])
            re = int(r[cols["reads_expected"]])
            total_ms.append(mt)
            print(f"{S:6d}  {mt:8.3f}  {mc:8.3f}  {ms:8.3f}  {mw:8.3f}  {ra:12d}  {re:8d}")
        if total_ms:
            avg_ms = sum(total_ms) / len(total_ms)
            print(f"avg ms_total: {avg_ms:.3f}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: summarize_bench.py bench1.csv [bench2.csv ...]")
        sys.exit(1)
    summarize(sys.argv[1:])

