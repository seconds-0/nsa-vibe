#!/usr/bin/env bash
# NSA-Vibe Mentat format script
# Purpose: run fast, auto-fix linters before commit_and_push.
# Do not run tests here (keep commits fast); tests are handled via pre-commit optional hook or CI.

set -euo pipefail

# Prefer uvx/uv if available; fallback to python -m ruff.
ruff_cmd_format() {
  if command -v uvx >/dev/null 2>&1; then
    uvx ruff format .
  elif command -v uv >/dev/null 2>&1; then
    uv run python -m ruff format .
  else
    python -m ruff format . || true
  fi
}
ruff_cmd_fix() {
  if command -v uvx >/dev/null 2>&1; then
    uvx ruff check --fix .
  elif command -v uv >/dev/null 2>&1; then
    uv run python -m ruff check --fix .
  else
    python -m ruff check --fix . || true
  fi
}

echo "==> Running autofixers (ruff-format, ruff check --fix)"
# ruff format and fixes (fast path; black/isort handled by pre-commit)
ruff_cmd_format
ruff_cmd_fix

echo "==> Format script completed"
