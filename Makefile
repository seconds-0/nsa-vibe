PYTHON := python
PIP := pip
VENV := .venv

.PHONY: venv install cpu-tests routing bench-decode bench-summarize bench-report triton-fwd triton-bwd lint oneshot pr env-pair clean

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
