import json
import subprocess
import sys


def test_check_env_pairing_runs():
    proc = subprocess.run([sys.executable, "scripts/check_env_pairing.py"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=30)
    # Script should always print JSON with keys even if versions unknown
    assert proc.returncode in (0, 2)
    data = json.loads(proc.stdout)
    assert "torch" in data and "triton" in data and "pairing_ok" in data

