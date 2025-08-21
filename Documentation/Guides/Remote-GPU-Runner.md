Remote GPU Runner (RunPod via Prime Intellect)

Goal
- Trigger GPU validation/benches from GitHub without hosting your own GPU, and collect artifacts.

Approach
- Use bench/run_automated_bench.py with a RunPod runner adapter (scaffolded) and a GitHub Actions workflow.
- You (or a teammate) add provider API keys as repository secrets when ready.

Steps
1) Create API credentials with your provider (RunPod) and obtain an API key.
2) In GitHub repo settings → Secrets and variables → Actions, add:
   - RUNPOD_API_KEY: your key
3) (Optional) Fork/enable the workflow .github/workflows/gpu-remote.yml
4) Use the “Run workflow” button (Actions → GPU Remote Validation) to dispatch a job with your GPU choice.

What the workflow does today
- Dry-run: calls bench/run_automated_bench.py --provider runpod with your inputs.
- When secrets are present and the adapter is completed, it will:
  - Launch a GPU task
  - Run parity tests and decode bench
  - Upload logs/CSVs back as job artifacts

Manual alternative
- You can run the same command locally (no network executed):
  - python bench/run_automated_bench.py --provider runpod --gpu A100 --output results.json
  - Without RUNPOD_API_KEY it will print a helpful message and exit.

Notes
- Keep secrets in GitHub; never commit them.
- Limit scope of the key to only what is needed for running benches.

