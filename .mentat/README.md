## Mentat Agent Readme for NSA‑Vibe

This repo is Mentat‑friendly. Use the provided scripts to build your environment, format code, and run targeted tests quickly.

### Environment Setup

- Default workflow uses uv and Python 3.11:
  - `bash .mentat/setup.sh`
  - Activates `.venv` and syncs from `requirements-cpu.txt` by default.
  - GPU: install from `requirements-gpu-cu121-torch24.txt` on CUDA hosts if needed.

Common invocations:
- Lint: `uv run ruff check .`
- Type check: `uv run mypy -p nsa`
- Tests (fast): `PYTHONPATH=. uv run -q pytest`

Reference commands and toggles are in `AGENTS.md` (uv workflow section).

### Formatting and Autofix

- Mentat runs `.mentat/format.sh` automatically before committing. It applies:
  - `ruff format` and `ruff --fix` only (fast). Black/isort are enforced by pre-commit on actual commits.
- Keep comments minimal and suitable for merge; do not add temporary comments.

### Pre‑commit Hooks (lint + fast tests)

- Repo has `pre-commit` configured for: ruff, ruff-format, black, isort, mypy.
- We added a local hook `pytest-fast` that runs a small, CPU‑safe subset:
  - `test_masks.py`, `test_block_math.py`, `test_equiv_small.py`,
    `test_decode_counters.py`, `test_group_consistency.py`
- Enable hooks in a shell after setup:
  - `uv run pre-commit install`
- You can disable the pytest hook for large WIP commits:
  - `export PRECOMMIT_PYTEST=0`

Note: Do not run full test suites in `.mentat/format.sh` to keep commits snappy. CI and on‑demand runs cover larger matrices.

### Workflow for Agents

1. Inspect and plan changes (see `AGENTS.md` rules).
2. Make focused edits with minimal diff.
3. Run fast tests locally:
   - `PYTHONPATH=. uv run -q pytest -k "test_masks or test_block_math or test_equiv_small or test_decode_counters or test_group_consistency"`
4. Commit; Mentat formatting will auto‑fix trivial lint.
5. Iterate on CI feedback and Auto Review with judgement; avoid scope creep.

### Notes

- Keep CPU fallback working; avoid introducing Triton in M0 paths.
- Respect invariants: causality, group consistency, Eq.9/10 semantics.
- Prefer small, reversible PRs; document changes briefly in commit messages.
