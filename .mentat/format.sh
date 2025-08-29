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

echo "==> Running autofixers (ruff-format, ruff check --fix)"
# Ensure ruff available quickly (prefer uv, fallback to pip)
if ! command -v ruff >/dev/null 2>&1; then
  if command -v uv >/dev/null 2>&1; then
    uv run python -m pip install -q --disable-pip-version-check ruff
  else
    python -m pip install -q --disable-pip-version-check ruff
  fi
fi
# ruff format and fixes (fast path; black/isort handled by pre-commit)
run ruff format .
run ruff check --fix .

echo "==> Format script completed"
