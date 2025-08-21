#!/usr/bin/env python3
"""
Derive FA-2 min-length thresholds from a bench log and optionally update a profile YAML.

Usage:
  python scripts/apply_fa2_thresholds.py --fa2-bench artifacts/runner/<commit>/fa2_bench.txt \
      --profile configs/profiles/sm89.yaml --write

Heuristic:
  - For sliding: choose the smallest window size w where fa2_ms <= masked_ms.
  - For compressed: pick the smallest S (or effective L) where fa2 shows speedup.
Notes:
  - This is a conservative first pass; adjust as you collect more data.
"""
import argparse
import re
from pathlib import Path


def parse_fa2_bench(path: Path) -> tuple[int | None, int | None]:
    win_threshold = None
    cmp_threshold = None
    win_re = re.compile(r"^S=(\d+) w=(\d+) sliding masked ([0-9.]+) ms\s+fa2 ([0-9.]+) ms")
    cmp_re = re.compile(r"^S=(\d+) l=(\d+) d=(\d+) compressed masked ([0-9.]+) ms\s+fa2 ([0-9.]+) ms")
    with path.open() as f:
        for line in f:
            m = win_re.match(line.strip())
            if m:
                _S = int(m.group(1))
                w = int(m.group(2))
                masked = float(m.group(3))
                fa2 = float(m.group(4))
                if fa2 <= masked and win_threshold is None:
                    win_threshold = w
                continue
            m2 = cmp_re.match(line.strip())
            if m2:
                S = int(m2.group(1))
                masked = float(m2.group(4))
                fa2 = float(m2.group(5))
                if fa2 <= masked and cmp_threshold is None:
                    cmp_threshold = S
    return win_threshold, cmp_threshold


def update_profile(profile_path: Path, win_thr: int | None, cmp_thr: int | None, dry_run: bool) -> None:
    # Minimal in-place YAML update: replace lines for fa2_min_len_win/cmp if present; otherwise append under runtime.
    text = profile_path.read_text()
    changed = False
    def repl(key: str, val: int | None) -> str:
        nonlocal text, changed
        if val is None:
            return text
        pattern = re.compile(rf"(^\s*{key}:\s*)(\d+)(\s*$)", re.M)
        if pattern.search(text):
            text = pattern.sub(rf"\g<1>{val}\g<3>", text)
        else:
            # naive insert under 'runtime:' block
            text = re.sub(r"(^runtime:\s*$)", rf"\1\n  {key}: {val}", text, count=1, flags=re.M)
        changed = True
        return text
    text = repl("fa2_min_len_win", win_thr)
    text = repl("fa2_min_len_cmp", cmp_thr)
    if changed and not dry_run:
        profile_path.write_text(text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fa2-bench", required=True, help="Path to fa2_bench.txt")
    ap.add_argument("--profile", required=True, help="Profile YAML to update (e.g., configs/profiles/sm89.yaml)")
    ap.add_argument("--write", action="store_true", help="Apply changes (default: dry-run)")
    args = ap.parse_args()
    win_thr, cmp_thr = parse_fa2_bench(Path(args.fa2_bench))
    print({"fa2_min_len_win": win_thr, "fa2_min_len_cmp": cmp_thr})
    update_profile(Path(args.profile), win_thr, cmp_thr, dry_run=not args.write)


if __name__ == "__main__":
    main()

