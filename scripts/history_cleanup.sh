#!/usr/bin/env bash
set -euo pipefail

# One-time history rewrite to remove large/generated files from Git history.
# WARNING: This rewrites history. Coordinate with collaborators and force-push.

cat <<'NOTE'
This script prints recommended git filter-repo commands to remove:
- tools/**/node_modules
- decode_*.csv, decode_*.txt, *.tar.gz
- legacy artifact folders: artifacts-accuracy-224091b, artifacts-oneshot-4106f1b

Usage (manual steps):
  1) pip install git-filter-repo
  2) git checkout main   # or your default branch
  3) git filter-repo --invert-paths \
       --path tools/pricing-scraper/node_modules \
       --path-glob 'decode_*.csv' \
       --path-glob 'decode_*.txt' \
       --path-glob '*.tar.gz' \
       --path artifacts-accuracy-224091b \
       --path artifacts-oneshot-4106f1b
  4) git push --force-with-lease origin HEAD:main

Optionally, preserve tracked reports by moving their folders under artifacts/tracked/ before running.
NOTE

python - <<'PY'
print('\nRecommended command:\n')
cmd = (
    "git filter-repo --invert-paths "
    "--path tools/pricing-scraper/node_modules "
    "--path-glob 'decode_*.csv' "
    "--path-glob 'decode_*.txt' "
    "--path-glob '*.tar.gz' "
    "--path artifacts-accuracy-224091b "
    "--path artifacts-oneshot-4106f1b"
)
print(cmd)
PY

