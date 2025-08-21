#!/usr/bin/env bash
set -euo pipefail

# Cleanup generated artifacts and local clutter. Safe by default (dry run).
# Usage:
#   bash scripts/cleanup_repo.sh          # dry-run, prints what would be removed
#   bash scripts/cleanup_repo.sh --apply  # actually remove files

APPLY=0
if [[ ${1-} == "--apply" ]]; then
  APPLY=1
fi

red() { printf "\033[31m%s\033[0m\n" "$*"; }
green() { printf "\033[32m%s\033[0m\n" "$*"; }

declare -a PATTERNS=(
  "artifacts-*/"
  "artifacts/"
  "decode_*.csv"
  "decode_*.txt"
  "dense_final*.csv"
  "varlen_results.csv"
  "fa2_*.txt"
  "triton_*.txt"
  "sanity.out"
  "*.tar.gz"
  "tools/**/node_modules/"
  "tools/**/dist/"
)

FOUND=()
for pat in "${PATTERNS[@]}"; do
  # Use globbing via bash -O globstar for recursive patterns
  shopt -s nullglob globstar
  for f in $pat; do
    FOUND+=("$f")
  done
done

if [[ ${#FOUND[@]} -eq 0 ]]; then
  green "Nothing to clean."
  exit 0
fi

echo "Candidates to remove (matched by patterns):"
for f in "${FOUND[@]}"; do
  echo "  $f"
fi

if [[ $APPLY -eq 0 ]]; then
  red "\nDry run. Re-run with --apply to remove."
  exit 0
fi

echo
for f in "${FOUND[@]}"; do
  if [[ -d "$f" ]]; then
    rm -rf "$f"
  else
    rm -f "$f"
  fi
  echo "removed: $f"
done
green "\nCleanup complete."
