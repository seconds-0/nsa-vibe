#!/usr/bin/env python3
import subprocess
import sys

SUITES = [
    # M0 correctness
    [
        "pytest",
        "-q",
        "-k",
        "test_equiv_small or test_block_math or test_masks or test_group_consistency",
    ],
    # M3 decode counters/order
    ["pytest", "-q", "-k", "test_decode_counters or test_decode_step"],
    # Selection packed parity (CPU)
    ["pytest", "-q", "-k", "test_selection_packed"],
    # Varlen packing + train smoke (CPU)
    ["pytest", "-q", "-k", "test_collate_varlen or test_train_smoke"],
]


def run(cmd):
    print("$", " ".join(cmd), flush=True)
    return subprocess.call(cmd)


def main():
    rc_total = 0
    for cmd in SUITES:
        rc = run(cmd)
        rc_total |= rc
        print("-- rc:", rc, flush=True)
    sys.exit(rc_total)


if __name__ == "__main__":
    main()
