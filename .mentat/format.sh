#!/usr/bin/env bash
# NSA-Vibe Mentat format script
# Purpose: run fast, auto-fix linters before commit_and_push.
# Do not run tests here (keep commits fast); tests are handled via pre-commit optional hook or CI.

set -euo pipefail

# Prefer uv if available to use project venv; fallback to direct commands.
run() {
  if command -v uv >/dev/null 2>&1; then
    uv run "$@"
  else
    "$@"
  fi
}

echo "==> Running autofixers (ruff-format, ruff --fix, isort, black)"
# ruff format and fixes
run ruff format .
run ruff --fix .

# Keep isort/black in sync with repo's pre-commit config
if command -v uv >/dev/null 2>&1; then
  run isort --profile black .
  run black .
else
  if command -v isort >/dev/null 2>&1; then isort --profile black .; fi
  if command -v black >/dev/null 2>&1; then black .; fi
fi

echo "==> Format script completed"
