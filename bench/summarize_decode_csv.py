#!/usr/bin/env python3
import argparse
import csv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", help="Input CSV from bench_decode.py --csv")
    args = ap.parse_args()

    rows = []
    with open(args.csv) as f:
        r = csv.DictReader(f)
        for row in r:
            # Backward-compatible parsing with optional decode/total read breakdown
            row2 = {
                "S": int(row["S"]),
                "ms_total": float(row["ms_total"]),
                "ms_cmp": float(row["ms_cmp"]),
                "ms_sel": float(row["ms_sel"]),
                "ms_win": float(row["ms_win"]),
                # total reads (legacy names)
                "reads_actual_total": int(
                    row.get("reads_actual", row.get("reads_actual_total", 0)) or 0
                ),
                "reads_expected_total": int(
                    row.get("reads_expected", row.get("reads_expected_total", 0)) or 0
                ),
                # decode-only reads (new names)
                "reads_actual_decode": int(row.get("reads_actual_decode", 0) or 0),
                "reads_expected_decode": int(row.get("reads_expected_decode", 0) or 0),
            }
            rows.append(row2)

    if not rows:
        print("No rows")
        return

    print(
        f"{'S':>6}  {'total':>8}  {'cmp%':>6}  {'sel%':>6}  {'win%':>6}  {'reads(dec)':>14}  {'reads(tot)':>14}"
    )
    for row in rows:
        total = row["ms_total"] or 1e-9
        cmp_p = 100.0 * row["ms_cmp"] / total
        sel_p = 100.0 * row["ms_sel"] / total
        win_p = 100.0 * row["ms_win"] / total
        reads_dec = "-/-"
        if row["reads_expected_decode"] > 0 or row["reads_actual_decode"] > 0:
            reads_dec = f"{row['reads_actual_decode']}/{row['reads_expected_decode']}"
        reads_tot = f"{row['reads_actual_total']}/{row['reads_expected_total']}"
        print(
            f"{row['S']:>6}  {row['ms_total']:>8.2f}  {cmp_p:>6.1f}  {sel_p:>6.1f}  {win_p:>6.1f}  {reads_dec:>14}  {reads_tot:>14}"
        )


if __name__ == "__main__":
    main()
