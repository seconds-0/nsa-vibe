import os
import sys
import subprocess


def test_selection_ranges_cli_runs(tmp_path):
    env = os.environ.copy()
    env["PYTHONPATH"] = "." + (os.pathsep + env.get("PYTHONPATH", ""))
    args = [
        sys.executable,
        "scripts/print_selection_ranges.py",
        "--S", "8",
        "--dim", "32",
        "--heads", "2",
        "--groups", "1",
        "--dk", "16",
        "--dv", "16",
        "--l", "4",
        "--d", "2",
        "--l_sel", "8",
        "--n_sel", "2",
        "--w", "8",
        "--json",
    ]
    proc = subprocess.run(args, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=60)
    assert proc.returncode == 0, proc.stderr
    # Expect at least one JSON line with key 't'
    lines = [ln for ln in proc.stdout.splitlines() if ln.strip()]
    assert lines, "no output"
    assert '"t":' in lines[0]

