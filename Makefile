PYTHON := python
PIP := pip
VENV := .venv

# Prime Intellect automation (parameterized)
PRIME_HOST ?= $(REMOTE_HOST)
SSH_KEY ?= $(SSH_KEY_PATH)
REPO_URL = https://github.com/seconds-0/nsa-vibe.git
BRANCH = test-plan/m7-training-readiness
TB_PORT = 6006

.PHONY: venv install cpu-tests routing bench-decode bench-summarize bench-report triton-fwd triton-bwd lint oneshot pr env-pair clean
.PHONY: help-prime train-prime setup-prime monitor-prime logs-prime clean-prime status-prime help

help:
	@echo "NSA Development & Training Commands"
	@echo "=================================="
	@echo ""
	@echo "🚀 PRIME INTELLECT TRAINING:"
	@echo "  make train-prime    - One-command automated training start"
	@echo "  make monitor-prime  - TensorBoard tunnel (run in new terminal)"
	@echo "  make help-prime     - Full Prime Intellect help"
	@echo ""
	@echo "🔧 Local Development:"
	@echo "  make venv          - Create virtual environment"
	@echo "  make cpu-tests     - Run test suite"
	@echo "  make lint          - Run linting"
	@echo ""
	@echo "📊 Benchmarking:"
	@echo "  make bench-decode  - Decode benchmark"
	@echo "  make routing       - Show execution routing"

venv:
	$(PYTHON) -m venv $(VENV)
	. $(VENV)/bin/activate && $(PIP) install -U pip wheel setuptools
	. $(VENV)/bin/activate && $(PIP) install -r requirements.txt

install: venv
	@echo "Environment ready. Activate with: . $(VENV)/bin/activate"

cpu-tests:
	PYTHONPATH=. pytest -q

routing:
	$(PYTHON) scripts/print_routing.py || true

bench-decode:
	mkdir -p artifacts
	PYTHONPATH=. $(PYTHON) bench/bench_decode.py --S_list 512,1024 --iters 16 --warmup 4 --csv artifacts/decode_test.csv --branch_force_mode env

bench-summarize:
	$(PYTHON) bench/summarize_decode_csv.py artifacts/decode_test.csv

bench-report:
	bash scripts/bench_report.sh

triton-fwd:
	NSA_USE_TRITON_SEL=1 NSA_TRITON_SEL_FORCE=1 PYTHONPATH=. pytest -q nsa/tests/test_triton_sel_parity_gpu.py

triton-bwd:
	NSA_USE_TRITON_SEL=1 NSA_SEL_TRITON_ALLOW_GRAD=1 NSA_TRITON_SEL_FORCE=1 PYTHONPATH=. pytest -q nsa/tests/test_triton_sel_backward_gpu.py

lint:
	ruff check .
	mypy nsa || true

oneshot:
	bash scripts/runner_oneshot.sh

pr:
	bash scripts/open_pr.sh

env-pair:
	python scripts/check_env_pairing.py

clean:
	bash scripts/cleanup_repo.sh

# ============================================================================
# Prime Intellect M7C Training Automation
# ============================================================================

help-prime:
	@echo "🚀 NSA M7C Training Automation for Prime Intellect"
	@echo ""
	@echo "One-command training:"
	@echo "  make train-prime    - Complete automated setup & training start"
	@echo ""
	@echo "Monitoring:"
	@echo "  make monitor-prime  - Start TensorBoard tunnel (run in new terminal)"
	@echo "  make logs-prime     - Tail live training logs"
	@echo "  make status-prime   - Quick status check"
	@echo ""
	@echo "Manual control:"
	@echo "  make setup-prime    - Setup environment only (no training)"
	@echo "  make clean-prime    - Clean remote artifacts"
	@echo ""
	@echo "Target: $(PRIME_HOST)"
	@echo "TensorBoard: http://localhost:$(TB_PORT)"

train-prime:
	@echo "🚀 Starting automated M7C training on Prime Intellect..."
	@if [ -z "$(PRIME_HOST)" ]; then echo "Set REMOTE_HOST or PRIME_HOST"; exit 2; fi
	@echo "📡 Connecting to $(PRIME_HOST)..."
	@scripts/automation/create_train_script.sh
	ssh $(if [ -n "$(SSH_KEY)" ]; then echo -n "-i $(SSH_KEY)"; fi) $(PRIME_HOST) 'bash -s' < scripts/automation/remote_train_setup.sh
	@echo ""
	@echo "✅ Training started! Next steps:"
	@echo "   1. Run in NEW terminal: make monitor-prime"
	@echo "   2. Open: http://localhost:$(TB_PORT)"
	@echo "   3. Watch live graphs!"

monitor-prime:
	@echo "📊 Starting TensorBoard tunnel..."
	@echo "🔗 Will auto-open http://localhost:$(TB_PORT)..."
	@sleep 2 && (command -v open >/dev/null && open http://localhost:$(TB_PORT) &) || echo "Open http://localhost:$(TB_PORT) manually" &
	@if [ -z "$(PRIME_HOST)" ]; then echo "Set REMOTE_HOST or PRIME_HOST"; exit 2; fi
	ssh $(if [ -n "$(SSH_KEY)" ]; then echo -n "-i $(SSH_KEY)"; fi) -L $(TB_PORT):localhost:$(TB_PORT) $(PRIME_HOST) \
		'cd nsa-vibe && . .venv/bin/activate && bash scripts/run_tensorboard.sh'

logs-prime:
	@echo "📋 Tailing training logs..."
	@if [ -z "$(PRIME_HOST)" ]; then echo "Set REMOTE_HOST or PRIME_HOST"; exit 2; fi
	ssh $(if [ -n "$(SSH_KEY)" ]; then echo -n "-i $(SSH_KEY)"; fi) $(PRIME_HOST) 'cd nsa-vibe && tail -f artifacts/m7c_125m/training.csv 2>/dev/null || echo "No training.csv yet, checking run logs..." && find artifacts/train_runs -name "train.log" -exec tail -f {} \; 2>/dev/null || echo "No logs found yet"'

setup-prime:
	@echo "⚙️  Setting up environment only..."
	@scripts/automation/create_setup_script.sh
	@if [ -z "$(PRIME_HOST)" ]; then echo "Set REMOTE_HOST or PRIME_HOST"; exit 2; fi
	ssh $(if [ -n "$(SSH_KEY)" ]; then echo -n "-i $(SSH_KEY)"; fi) $(PRIME_HOST) 'bash -s' < scripts/automation/remote_setup_only.sh

clean-prime:
	@echo "🧹 Cleaning remote artifacts..."
	@if [ -z "$(PRIME_HOST)" ]; then echo "Set REMOTE_HOST or PRIME_HOST"; exit 2; fi
	ssh $(if [ -n "$(SSH_KEY)" ]; then echo -n "-i $(SSH_KEY)"; fi) $(PRIME_HOST) 'cd nsa-vibe && rm -rf artifacts/train_runs/* artifacts/m7c_125m/* 2>/dev/null || true && echo "✅ Artifacts cleaned"'

status-prime:
	@echo "📊 Prime Intellect Status Check"
	@echo "================================"
	@if [ -z "$(PRIME_HOST)" ]; then echo "Set REMOTE_HOST or PRIME_HOST"; exit 2; fi
	@ssh $(if [ -n "$(SSH_KEY)" ]; then echo -n "-i $(SSH_KEY)"; fi) $(PRIME_HOST) 'cd nsa-vibe 2>/dev/null && echo "✅ Repo exists" || echo "❌ Repo missing"' || echo "❌ SSH connection failed"
	@ssh $(if [ -n "$(SSH_KEY)" ]; then echo -n "-i $(SSH_KEY)"; fi) $(PRIME_HOST) 'cd nsa-vibe && test -f artifacts/m7c_125m/training.csv && echo "✅ Training active" || echo "⏳ No training.csv yet"' 2>/dev/null || echo "❓ Cannot check training status"
	@ssh $(if [ -n "$(SSH_KEY)" ]; then echo -n "-i $(SSH_KEY)"; fi) $(PRIME_HOST) 'nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader 2>/dev/null | head -2' || echo "❓ GPU status unknown"
	@ssh $(if [ -n "$(SSH_KEY)" ]; then echo -n "-i $(SSH_KEY)"; fi) $(PRIME_HOST) 'tmux list-sessions 2>/dev/null | grep m7c && echo "✅ tmux session active" || echo "⏳ No tmux session"' || echo "❓ Cannot check tmux"
env-guard:
	@echo "Running environment guard..."
	python scripts/_env_guard.py
