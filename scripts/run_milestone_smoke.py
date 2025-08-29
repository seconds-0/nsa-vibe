#!/usr/bin/env python3
"""M8 Milestone smoke tests - comprehensive validation suite."""

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
    # M8 integration tests
    ["pytest", "-q", "-k", "test_m8_integration"],
    # M8 causality assertions
    ["pytest", "-q", "-k", "test_causality_asserts"],
    # M8 smoke tests (synthetic data)
    ["python", "scripts/run_smoke_tests.py", "--run-synthetic", "--smoke-steps", "100"],
]


def run(cmd):
    print("$", " ".join(cmd), flush=True)
    return subprocess.call(cmd)


def main():
    print("M8 Milestone Smoke Test Suite")
    print("==============================")
    rc_total = 0
    passed = 0
    failed = 0

    for i, cmd in enumerate(SUITES, 1):
        print(f"\n[{i}/{len(SUITES)}] Running: {' '.join(cmd)}")
        rc = run(cmd)
        if rc == 0:
            print(f"✅ Suite {i} PASSED")
            passed += 1
        else:
            print(f"❌ Suite {i} FAILED (rc={rc})")
            failed += 1
        rc_total |= rc

    print("\n==============================")
    print(f"Results: {passed} passed, {failed} failed")
    if rc_total == 0:
        print("🎉 All milestone smoke tests PASSED!")
    else:
        print("💥 Some milestone smoke tests FAILED!")

    sys.exit(rc_total)


if __name__ == "__main__":
    main()
