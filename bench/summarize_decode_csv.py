#!/usr/bin/env python3
import csv
import argparse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", help="Input CSV from bench_decode.py --csv")
    args = ap.parse_args()

    rows = []
    with open(args.csv) as f:
        r = csv.DictReader(f)
        for row in r:
            row2 = {
                "S": int(row["S"]),
                "ms_total": float(row["ms_total"]),
                "ms_cmp": float(row["ms_cmp"]),
                "ms_sel": float(row["ms_sel"]),
                "ms_win": float(row["ms_win"]),
                "reads_actual": int(row["reads_actual"]),
                "reads_expected": int(row["reads_expected"]),
            }
            rows.append(row2)

    if not rows:
        print("No rows")
        return

    print(f"{'S':>6}  {'total':>8}  {'cmp%':>6}  {'sel%':>6}  {'win%':>6}  reads")
    for row in rows:
        total = row["ms_total"] or 1e-9
        cmp_p = 100.0 * row["ms_cmp"] / total
        sel_p = 100.0 * row["ms_sel"] / total
        win_p = 100.0 * row["ms_win"] / total
        reads = f"{row['reads_actual']}/{row['reads_expected']}"
        print(f"{row['S']:>6}  {row['ms_total']:>8.2f}  {cmp_p:>6.1f}  {sel_p:>6.1f}  {win_p:>6.1f}  {reads}")


if __name__ == "__main__":
    main()

