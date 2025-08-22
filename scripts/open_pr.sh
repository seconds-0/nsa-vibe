#!/usr/bin/env bash
set -euo pipefail

# Open a GitHub PR using gh CLI if available; otherwise print instructions.
# Usage: bash scripts/open_pr.sh [base_branch] [head_branch] [title]

BASE=${1:-main}
HEAD=${2:-$(git rev-parse --abbrev-ref HEAD)}
TITLE=${3:-"NSA M0–M6 Closeout: benches, routing, docs"}

BODY_FILE=.github/PR_BODY.md
mkdir -p .github
cat > "$BODY_FILE" << 'EOF'
## Summary
Closeout for M0–M6: decode bench fix, routing/doc polish, runner one-shot, FA‑2 guidance, Triton parity.

## Highlights
- Fix: gate broadcasting in decode path; bench unblocked
- Stability: SDPA fallback nan_to_num; Triton/FA‑2 guards
- DX: one-shot runner script; remote GPU workflow; rich runbook
- Docs: Start-Here, FA‑2 install guide, routing, runner checklist

## Validation
- CPU tests green (3.11/3.12)
- GPU sanity via `scripts/gpu_sanity.py`
- Triton FWD/BWD parity on 4090 with force flags
- Decode bench CSV + summary produced on GPU

## Next
- Calibrate FA‑2 thresholds on 4090 from benches (then update profiles)
- (Optional) Self-hosted GPU CI lane when available
EOF

if command -v gh >/dev/null 2>&1; then
  echo "[pr] Using GitHub CLI to open PR" >&2
  gh pr create --base "$BASE" --head "$HEAD" --title "$TITLE" --body-file "$BODY_FILE"
else
  echo "[pr] gh CLI not found. Open a PR manually with the following details:" >&2
  echo "Base: $BASE" >&2
  echo "Head: $HEAD" >&2
  echo "Title: $TITLE" >&2
  echo "Body file prepared at $BODY_FILE" >&2
fi
