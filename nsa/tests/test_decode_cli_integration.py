import os
import sys
import csv
import subprocess
from pathlib import Path


def test_decode_cli_writes_csv(tmp_path):
    csv_path = tmp_path / "decode.csv"
    env = os.environ.copy()
    env["PYTHONPATH"] = "." + (os.pathsep + env.get("PYTHONPATH", ""))
    # Keep it tiny and CPU-friendly
    args = [
        sys.executable,
        "bench/bench_decode.py",
        "--B", "1",
        "--dim", "32",
        "--heads", "2",
        "--groups", "1",
        "--dk", "16",
        "--dv", "16",
        "--l", "8",
        "--d", "4",
        "--l_sel", "8",
        "--n_sel", "4",
        "--w", "16",
        "--S_list", "8",
        "--iters", "3",
        "--warmup", "1",
        "--csv", str(csv_path),
    ]
    proc = subprocess.run(args, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=60)
    assert proc.returncode == 0, f"bench failed: {proc.stderr}"
    assert csv_path.exists(), "CSV file not created"
    with csv_path.open("r", newline="") as f:
        reader = csv.reader(f)
        rows = list(reader)
    assert rows[0] == ["S", "ms_total", "ms_cmp", "ms_sel", "ms_win", "reads_actual", "reads_expected"], "CSV header mismatch"
    assert len(rows) >= 2, "Expected at least one data row"

