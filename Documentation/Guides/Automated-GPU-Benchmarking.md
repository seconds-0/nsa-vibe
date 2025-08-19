# Automated GPU Benchmarking System

This guide explains the automated GPU benchmarking system for NSA FlashAttention-2 threshold tuning, which replaces the manual process described in the M1 GPU Benchmark Playbook.

## Overview

The automated benchmarking system eliminates manual GPU provisioning, SSH access, and result parsing. It provides:

- **One-command benchmarking** across multiple GPU types
- **Automatic threshold optimization** based on performance data
- **GitHub Actions integration** for CI/CD pipelines
- **Multiple provider support** (Modal, local GPU, RunPod, Lambda Cloud)
- **Automatic config updates** and PR creation

## Quick Start

### Prerequisites

1. **Create Modal Account** (recommended provider):
   - Sign up at [modal.com](https://modal.com)
   - **Add payment method** (required for GPU usage)
   - Modal provides **$10/month free credits** for new users
   - GPU usage is billed per-second after free credits

2. Install Modal CLI:
```bash
pip install modal
```

3. Set up Modal authentication:
```bash
# Interactive setup (recommended)
modal token new

# Or set environment variables:
export MODAL_TOKEN_ID="your-token-id"
export MODAL_TOKEN_SECRET="your-token-secret"
```

### Running Benchmarks

#### Option 1: Local CLI
```bash
# Run on Modal with T4 GPU (cheapest option)
python bench/run_automated_bench.py --provider modal --gpu T4

# Run on more powerful GPU
python bench/run_automated_bench.py --provider modal --gpu A100

# Run and automatically update config
python bench/run_automated_bench.py --provider modal --gpu L4 --update-config

# Generate detailed report
python bench/run_automated_bench.py --provider modal --gpu T4 --report benchmark_report.md
```

#### Option 2: GitHub Actions

1. **Manual trigger** via GitHub UI:
   - Go to Actions → GPU Benchmark → Run workflow
   - Select GPU type and options
   - Click "Run workflow"

2. **PR comment trigger**:
   ```
   /benchmark T4
   ```
   or
   ```
   /benchmark A100
   ```

The workflow will:
- Provision the requested GPU
- Run benchmarks
- Parse results
- Create a PR with updated thresholds

#### Option 3: Direct Modal Execution
```bash
# Run Modal app directly
modal run bench/modal_gpu_bench.py --gpu-type L4 --output results.json

# Parse results and update config
python bench/threshold_optimizer.py results.json --config configs/base.yaml
```

## Architecture

### Components

1. **`bench/modal_gpu_bench.py`**
   - Modal serverless app for GPU execution
   - Handles environment setup, benchmark runs, and result parsing
   - Returns structured JSON with recommendations

2. **`bench/threshold_optimizer.py`**
   - Parses benchmark outputs from any source
   - Determines optimal thresholds based on safety margins
   - Updates YAML configs automatically
   - Generates markdown reports

3. **`bench/run_automated_bench.py`**
   - Unified CLI interface for all providers
   - Handles provider-specific setup
   - Manages result aggregation and reporting

4. **`.github/workflows/gpu-bench.yml`**
   - GitHub Actions workflow for CI/CD
   - Supports manual and comment triggers
   - Creates PRs with optimized configs

## Providers

### Modal (Recommended)
- **Pros**: Serverless, pay-per-second, no setup
- **Cons**: Requires Modal account
- **Cost**: ~$0.10-1.00 per benchmark run
- **GPUs**: T4, L4, A10, A100, H100

### Local GPU
- **Pros**: Free if you have hardware
- **Cons**: Requires local CUDA setup
- **Use**: Development and testing

### RunPod (Future)
- **Pros**: Wide GPU selection, competitive pricing
- **Cons**: Implementation pending
- **Cost**: ~$0.50-2.00 per benchmark

### Lambda Cloud (Future)
- **Pros**: High-end GPUs, good availability
- **Cons**: Implementation pending
- **Cost**: ~$1.00-3.00 per benchmark

## Configuration

### Safety Margins

The `--safety-margin` parameter controls threshold selection:

```bash
# Conservative: FA-2 must be 50% faster
python bench/run_automated_bench.py --safety-margin 1.5

# Balanced: FA-2 must be 20% faster (default)
python bench/run_automated_bench.py --safety-margin 1.2

# Aggressive: FA-2 must be 10% faster
python bench/run_automated_bench.py --safety-margin 1.1
```

### Environment Variables

```bash
# Modal (required for Modal provider)
export MODAL_TOKEN_ID="..."
export MODAL_TOKEN_SECRET="..."

# RunPod (future)
export RUNPOD_API_KEY="..."

# Lambda Cloud (future)
export LAMBDA_API_KEY="..."
```

### GitHub Secrets

For GitHub Actions, add these secrets to your repository:

1. Go to Settings → Secrets and variables → Actions
2. Add:
   - `MODAL_TOKEN_ID`
   - `MODAL_TOKEN_SECRET`

## Workflow Examples

### 1. Development Workflow
```bash
# Test on cheap GPU first
python bench/run_automated_bench.py --provider modal --gpu T4 --output t4_results.json

# If promising, test on target GPU
python bench/run_automated_bench.py --provider modal --gpu A100 --output a100_results.json

# Compare results
python bench/run_automated_bench.py --compare t4_results.json a100_results.json

# Update config with best results
python bench/threshold_optimizer.py a100_results.json --update-config
```

### 2. CI/CD Workflow
```yaml
# In your GitHub workflow
- name: Benchmark on PR
  if: contains(github.event.pull_request.labels.*.name, 'needs-benchmark')
  uses: ./.github/workflows/gpu-bench.yml
  with:
    gpu_type: L4
    create_pr: true
```

### 3. Multi-GPU Comparison
```bash
# Run on multiple GPUs
for gpu in T4 L4 A100; do
  python bench/run_automated_bench.py --provider modal --gpu $gpu \
    --output "results_${gpu}.json" \
    --report "report_${gpu}.md"
done

# Generate comparison report
python bench/run_automated_bench.py --compare results_*.json
```

## Output Formats

### JSON Results
```json
{
  "device_info": {
    "device_name": "NVIDIA L4",
    "torch_version": "2.3.0",
    "cuda_version": "12.1"
  },
  "recommendation": {
    "fa2_min_len_win": 64,
    "fa2_min_len_cmp": 16
  },
  "results": [...],
  "parity_passed": true
}
```

### Markdown Report
The system generates detailed markdown reports with:
- Device information
- Performance tables with speedup indicators
- Recommended thresholds
- Analysis and rationale

### Config Updates
Automatically updates `configs/base.yaml`:
```yaml
runtime:
  fa2_min_len_win: 64  # Updated by benchmark
  fa2_min_len_cmp: 16  # Updated by benchmark
```

## Cost Analysis

### Modal Pricing (after free credits)

| GPU | Cost/Hour | Container Build | Benchmark Run | Total Cost |
|-----|-----------|-----------------|---------------|------------|
| T4 | $0.59 | ~1 min | ~2 min | ~$0.03 |
| L4 | $0.76 | ~1 min | ~2 min | ~$0.04 |
| A100-40GB | $3.73 | ~1 min | ~1 min | ~$0.12 |
| H100 | $8.58 | ~1 min | ~1 min | ~$0.29 |

**Notes:**
- First $10/month is free (enough for ~300 T4 runs or ~80 A100 runs)
- Container image is cached after first build
- Actual costs include container build time + execution time
- Storage costs are minimal (~$0.001 per run)

## Troubleshooting

### Modal Issues

**"Modal not found"**
```bash
pip install modal
modal token new
```

**"No payment method"**
- Go to [modal.com/settings/billing](https://modal.com/settings/billing)
- Add a credit card (required for GPU usage)
- You still get $10 free credits monthly

**"No GPU available" or "Quota exceeded"**
- Modal may be at capacity; try a different GPU type
- Check your usage at [modal.com/usage](https://modal.com/usage)
- Free tier has concurrent GPU limits
- Consider upgrading to a paid plan for higher limits

**"Authentication failed"**
```bash
# Re-authenticate
modal token new

# Verify authentication
modal config show
```

### Local GPU Issues

**"CUDA not available"**
```bash
# Check PyTorch CUDA
python -c "import torch; print(torch.cuda.is_available())"

# Reinstall with CUDA support
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

**"Flash-attn not available"**
```bash
# Install pre-built wheel
pip install flash-attn --no-build-isolation
```

### GitHub Actions Issues

**"Secrets not found"**
- Ensure MODAL_TOKEN_ID and MODAL_TOKEN_SECRET are set in repository secrets
- Check secret names match exactly

**"PR creation failed"**
- Ensure the workflow has write permissions
- Check branch protection rules

## Comparison with Manual Process

| Aspect | Manual (M1 Playbook) | Automated |
|--------|---------------------|-----------|
| Time | 30-60 minutes | 2-5 minutes |
| Complexity | High (SSH, manual setup) | Low (single command) |
| Reproducibility | Variable | Exact |
| Cost | Higher (instance minimum) | Lower (pay-per-second) |
| Error-prone | Yes (manual parsing) | No (automated) |
| CI/CD Integration | Difficult | Native |

## Advanced Features

### Custom Benchmark Configurations
```python
# In modal_gpu_bench.py, modify bench parameters
for S in [256, 512, 1024, 2048, 4096]:  # Extended range
    for w in [32, 64, 128, 256, 512]:   # More window sizes
        bench_once(S=S, w=w, ...)
```

### Parallel Multi-GPU Benchmarking
```bash
# Run benchmarks in parallel using GNU parallel
parallel -j 4 python bench/run_automated_bench.py \
  --provider modal --gpu {} --output results_{}.json \
  ::: T4 L4 A10 A100
```

### Automated Regression Detection
```python
# Compare against baseline
baseline = load_baseline("configs/baseline_thresholds.json")
current = run_benchmark()
if current["fa2_min_len_win"] > baseline["fa2_min_len_win"]:
    print("Performance regression detected!")
```

## Future Enhancements

1. **RunPod Integration**: Direct API support for RunPod serverless
2. **Lambda Cloud Integration**: Native Lambda Cloud API support
3. **Performance Tracking**: Historical benchmark database
4. **Auto-tuning**: Gradient-based threshold optimization
5. **Multi-config Testing**: Test multiple configurations in parallel
6. **Cost Optimization**: Automatic provider selection based on availability/cost

## Contributing

To add a new provider:

1. Create a runner class in `bench/run_automated_bench.py`:
```python
class NewProviderRunner(BenchmarkRunner):
    def is_available(self) -> bool:
        # Check if provider is configured
        pass
    
    def run(self, gpu_type: str) -> Dict:
        # Run benchmark and return results
        pass
```

2. Register in the runners dictionary:
```python
runners = {
    "newprovider": NewProviderRunner(),
    ...
}
```

3. Update documentation with setup instructions

## Summary

The automated GPU benchmarking system provides a modern, efficient alternative to manual benchmark processes. Key benefits:

- **5-10x faster** than manual process
- **Fully reproducible** results
- **CI/CD ready** with GitHub Actions
- **Cost-effective** with serverless execution
- **Zero infrastructure** management

For most users, the Modal-based approach with the CLI wrapper provides the best balance of simplicity, cost, and performance.