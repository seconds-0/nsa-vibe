# Prime Intellect API Reference

This document contains the complete API reference for Prime Intellect's GPU cloud platform, compiled for the NSA benchmarking automation.

## Overview

Prime Intellect provides a unified API for managing GPU resources across multiple cloud providers. The platform focuses on making GPU compute accessible and affordable for AI/ML workloads.

## Authentication

All API requests require authentication via Bearer token.

### Getting an API Key
1. Sign up at [app.primeintellect.ai](https://app.primeintellect.ai)
2. Navigate to Settings → API Keys
3. Click "Generate New Key +"
4. Assign appropriate permissions (Pods: Create/Read/Delete, Availability: Read)

### Using the API Key
Include in all requests:
```
Authorization: Bearer <your-api-key>
```

Environment variable:
```bash
export PRIME_API_KEY="your-api-key"
```

## Base URL
```
https://api.primeintellect.ai/api/v1/
```

## Endpoints

### 1. Check GPU Availability

**GET** `/availability/`

Query available GPUs across providers.

**Query Parameters:**
- `regions` (array): Geographic regions (e.g., "united_states", "canada", "europe")
- `gpu_count` (integer): Number of GPUs required
- `gpu_type` (string): Specific GPU model (see GPU Types section)

**Example Request:**
```bash
curl --request GET \
  --url 'https://api.primeintellect.ai/api/v1/availability/?regions=united_states&gpu_count=1&gpu_type=H100_80GB' \
  --header 'Authorization: Bearer $PRIME_API_KEY'
```

**Response Fields:**
- `cloudId`: Unique provider identifier (use for pod creation)
- `provider`: Cloud service provider (e.g., "runpod", "datacrunch")
- `dataCenter`: Physical location
- `prices`: Pricing information
  - `hourly`: Cost per hour
  - `monthly`: Monthly rate if available
- `stockStatus`: Current availability
- `images`: Available OS/environment images

### 2. Create Pod

**POST** `/pods/`

Create a new GPU pod/instance.

**Request Body:**
```json
{
  "pod": {
    "name": "benchmark-pod",
    "cloudId": "<from-availability-api>",
    "gpuType": "H100_80GB",
    "socket": "PCIe",
    "gpuCount": 1,
    "diskSize": 100,
    "vcpus": 8,
    "memory": 32,
    "maxPrice": 5.0,
    "image": "ubuntu_22_cuda_12",
    "autoRestart": false,
    "envVars": [
      {
        "key": "GITHUB_REPO",
        "value": "https://github.com/seconds-0/nsa-vibe.git"
      }
    ]
  },
  "provider": {
    "type": "runpod"
  }
}
```

**Parameters:**
- **Required:**
  - `cloudId`: Provider-specific identifier from availability API
  - `gpuType`: GPU model (see GPU Types)
  - `socket`: "PCIe" or "SXM"
  - `gpuCount`: Number of GPUs

- **Optional:**
  - `name`: Human-readable pod name
  - `diskSize`: Storage in GB (default: 50)
  - `vcpus`: Number of vCPUs (default: provider-specific)
  - `memory`: RAM in GB (default: provider-specific)
  - `maxPrice`: Maximum hourly price willing to pay
  - `image`: Container image (see Images section)
  - `customTemplateId`: Custom Docker image ID
  - `autoRestart`: Auto-restart on failure
  - `envVars`: Environment variables array
  - `jupyterPassword`: Password for Jupyter access

**Response (200 OK):**
```json
{
  "id": "pod-xyz123",
  "name": "benchmark-pod",
  "status": "starting",
  "provider": "runpod",
  "cloudId": "cloud-abc",
  "gpuType": "H100_80GB",
  "gpuCount": 1,
  "sshConnection": {
    "host": "123.45.67.89",
    "port": 22,
    "user": "root"
  },
  "price": {
    "hourly": 2.50
  },
  "createdAt": "2024-01-01T00:00:00Z"
}
```

### 3. Get Pod Status

**GET** `/pods/{pod_id}`

Get detailed information about a specific pod.

**Response Fields:**
- `id`: Pod identifier
- `status`: Current status ("starting", "running", "stopped", "error")
- `sshConnection`: SSH details when running
- `runtime`: Time pod has been running
- `costs`: Accumulated costs

### 4. List Pods

**GET** `/pods/`

List all pods for the authenticated user/team.

**Query Parameters:**
- `status`: Filter by status
- `provider`: Filter by provider
- `limit`: Number of results
- `offset`: Pagination offset

### 5. Delete Pod

**DELETE** `/pods/{pod_id}`

Terminate and delete a pod.

**Response (200 OK):**
```json
{
  "message": "Pod terminated successfully",
  "finalCost": 2.50
}
```

### 6. Get Pod History

**GET** `/pods/history`

Get historical pod usage and costs.

## GPU Types

Common GPU types available:
- `T4`: NVIDIA T4 (16GB)
- `L4`: NVIDIA L4 (24GB)
- `L40`: NVIDIA L40 (48GB)
- `RTX_3090`: GeForce RTX 3090 (24GB)
- `RTX_4090`: GeForce RTX 4090 (24GB)
- `A10`: NVIDIA A10 (24GB)
- `A30`: NVIDIA A30 (24GB)
- `A40`: NVIDIA A40 (48GB)
- `A100_40GB`: NVIDIA A100 (40GB)
- `A100_80GB`: NVIDIA A100 (80GB)
- `H100_80GB`: NVIDIA H100 (80GB)
- `H100_NVL`: NVIDIA H100 NVL (94GB)

## Images

Standard images available:
- `ubuntu_22_cuda_12`: Ubuntu 22.04 with CUDA 12.x
- `ubuntu_20_cuda_11`: Ubuntu 20.04 with CUDA 11.x
- `pytorch_2_cuda_12`: PyTorch 2.x with CUDA 12.x
- `tensorflow_2_cuda_12`: TensorFlow 2.x with CUDA 12.x

## Python SDK / CLI

### Installation

```bash
# Install CLI
pip install prime-cli

# Or with uv (recommended)
uv tool install prime
```

### CLI Commands

```bash
# Authenticate
prime login

# List available GPUs
prime availability list
prime availability list --gpu-type H100_80GB --region united_states

# Pod management
prime pods create --name my-pod --gpu-type A100_40GB --gpu-count 1
prime pods list
prime pods status <pod-id>
prime pods ssh <pod-id>
prime pods terminate <pod-id>

# Teams (if applicable)
prime teams list
prime teams switch <team-id>
```

### Python Usage Examples

#### Check Availability
```python
import requests
import os

headers = {
    'Authorization': f'Bearer {os.environ["PRIME_API_KEY"]}'
}

response = requests.get(
    'https://api.primeintellect.ai/api/v1/availability/',
    params={
        'regions': ['united_states'],
        'gpu_type': 'A100_40GB',
        'gpu_count': 1
    },
    headers=headers
)

availability = response.json()
for option in availability:
    print(f"{option['provider']}: ${option['prices']['hourly']}/hr in {option['dataCenter']}")
```

#### Create and Manage Pod
```python
import requests
import time
import os

api_key = os.environ['PRIME_API_KEY']
headers = {
    'Authorization': f'Bearer {api_key}',
    'Content-Type': 'application/json'
}

# 1. Check availability
availability_resp = requests.get(
    'https://api.primeintellect.ai/api/v1/availability/',
    params={'gpu_type': 'T4', 'gpu_count': 1},
    headers=headers
)
options = availability_resp.json()
cloud_id = options[0]['cloudId']  # Pick first available

# 2. Create pod
pod_data = {
    "pod": {
        "name": "nsa-benchmark",
        "cloudId": cloud_id,
        "gpuType": "T4",
        "socket": "PCIe",
        "gpuCount": 1,
        "diskSize": 50,
        "image": "ubuntu_22_cuda_12",
        "envVars": [
            {"key": "REPO_URL", "value": "https://github.com/seconds-0/nsa-vibe.git"}
        ]
    },
    "provider": {
        "type": options[0]['provider']
    }
}

create_resp = requests.post(
    'https://api.primeintellect.ai/api/v1/pods/',
    json=pod_data,
    headers=headers
)
pod = create_resp.json()
pod_id = pod['id']

# 3. Wait for pod to be ready
while True:
    status_resp = requests.get(
        f'https://api.primeintellect.ai/api/v1/pods/{pod_id}',
        headers=headers
    )
    status = status_resp.json()
    if status['status'] == 'running':
        break
    time.sleep(5)

# 4. Get SSH details
ssh_info = status['sshConnection']
print(f"SSH: ssh {ssh_info['user']}@{ssh_info['host']} -p {ssh_info['port']}")

# 5. Run your workload here...

# 6. Cleanup
delete_resp = requests.delete(
    f'https://api.primeintellect.ai/api/v1/pods/{pod_id}',
    headers=headers
)
```

#### Run Script on Pod via SSH
```python
import paramiko
import os

def run_benchmark_on_pod(ssh_info, private_key_path):
    """Execute benchmark script on Prime Intellect pod via SSH."""
    
    # Setup SSH client
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    
    # Connect
    ssh.connect(
        hostname=ssh_info['host'],
        port=ssh_info['port'],
        username=ssh_info['user'],
        key_filename=private_key_path
    )
    
    # Run commands
    commands = [
        "git clone https://github.com/seconds-0/nsa-vibe.git",
        "cd nsa-vibe && pip install -r requirements.txt",
        "cd nsa-vibe && python bench/bench_fa2.py"
    ]
    
    for cmd in commands:
        stdin, stdout, stderr = ssh.exec_command(cmd)
        print(stdout.read().decode())
        if stderr:
            print(stderr.read().decode())
    
    ssh.close()
```

## Pricing

Approximate hourly rates (varies by provider and availability):

| GPU | Hourly Rate | Notes |
|-----|------------|-------|
| T4 | $0.25-0.40 | Good for testing |
| L4 | $0.35-0.50 | Newer, efficient |
| RTX 4090 | $0.40-0.60 | Consumer GPU |
| A10 | $0.60-0.80 | Professional |
| A100 40GB | $1.00-1.50 | High-end training |
| A100 80GB | $1.50-2.00 | Large models |
| H100 80GB | $2.50-3.50 | Latest generation |

**Billing:**
- Minimum billing: 1 hour
- Billed per hour (not per-second like some providers)
- Prices vary by provider and region
- Spot/preemptible instances may be cheaper

## Error Codes

- `401`: Invalid or missing API key
- `403`: Insufficient permissions
- `404`: Resource not found
- `422`: Invalid request parameters
- `429`: Rate limit exceeded
- `500`: Internal server error
- `503`: Provider temporarily unavailable

## Rate Limits

- API calls: 1000 per hour
- Pod creation: 10 per hour
- Concurrent pods: Varies by account type

## Best Practices

1. **Cost Management:**
   - Always set `maxPrice` to avoid unexpected charges
   - Terminate pods immediately after use
   - Use smaller GPUs for testing

2. **Reliability:**
   - Implement retry logic for transient failures
   - Save work frequently (pods can be preempted)
   - Use `autoRestart` for long-running jobs

3. **Security:**
   - Never commit API keys to version control
   - Use environment variables for sensitive data
   - Rotate API keys regularly

4. **Performance:**
   - Choose data centers close to your location
   - Use NVLink/SXM GPUs for multi-GPU workloads
   - Pre-build custom Docker images for faster startup

## Support

- Documentation: https://docs.primeintellect.ai/
- Support: support@primeintellect.ai
- Status: https://status.primeintellect.ai/

## Migration from Modal

Key differences when migrating from Modal:

| Feature | Modal | Prime Intellect |
|---------|-------|-----------------|
| Model | Serverless functions | Persistent pods |
| Billing | Per-second | Per-hour (min 1hr) |
| Setup | Decorators | API/CLI |
| Execution | Automatic | Manual SSH/script |
| Cleanup | Automatic | Manual termination |
| Cost | Higher | 50-60% cheaper |

## Example: NSA Benchmark Workflow

Complete workflow for running NSA benchmarks:

```python
import requests
import json
import time
import os
from typing import Dict

class PrimeIntellectBenchmark:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.primeintellect.ai/api/v1"
        self.headers = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        }
    
    def find_cheapest_gpu(self, gpu_type: str) -> Dict:
        """Find cheapest available GPU of specified type."""
        resp = requests.get(
            f"{self.base_url}/availability/",
            params={'gpu_type': gpu_type, 'gpu_count': 1},
            headers=self.headers
        )
        options = resp.json()
        return min(options, key=lambda x: x['prices']['hourly'])
    
    def create_benchmark_pod(self, gpu_type: str = "T4") -> str:
        """Create pod for benchmarking."""
        # Find cheapest option
        option = self.find_cheapest_gpu(gpu_type)
        
        # Create pod
        pod_config = {
            "pod": {
                "name": f"nsa-bench-{gpu_type}",
                "cloudId": option['cloudId'],
                "gpuType": gpu_type,
                "socket": "PCIe",
                "gpuCount": 1,
                "diskSize": 50,
                "vcpus": 8,
                "memory": 32,
                "image": "ubuntu_22_cuda_12",
                "maxPrice": option['prices']['hourly'] * 1.2,  # 20% buffer
                "envVars": [
                    {"key": "PYTHONPATH", "value": "."},
                    {"key": "NSA_USE_FA2", "value": "1"}
                ]
            },
            "provider": {"type": option['provider']}
        }
        
        resp = requests.post(
            f"{self.base_url}/pods/",
            json=pod_config,
            headers=self.headers
        )
        return resp.json()['id']
    
    def wait_for_pod(self, pod_id: str, timeout: int = 300) -> Dict:
        """Wait for pod to be ready."""
        start = time.time()
        while time.time() - start < timeout:
            resp = requests.get(
                f"{self.base_url}/pods/{pod_id}",
                headers=self.headers
            )
            pod = resp.json()
            if pod['status'] == 'running':
                return pod
            time.sleep(5)
        raise TimeoutError(f"Pod {pod_id} not ready after {timeout}s")
    
    def run_benchmark(self, gpu_type: str = "T4") -> Dict:
        """Complete benchmark workflow."""
        pod_id = None
        try:
            # Create pod
            pod_id = self.create_benchmark_pod(gpu_type)
            print(f"Created pod: {pod_id}")
            
            # Wait for ready
            pod = self.wait_for_pod(pod_id)
            print(f"Pod ready: {pod['sshConnection']}")
            
            # Here you would SSH and run benchmarks
            # For now, return mock results
            results = {
                "gpu_type": gpu_type,
                "pod_id": pod_id,
                "provider": pod['provider'],
                "cost_per_hour": pod['price']['hourly']
            }
            
            return results
            
        finally:
            # Always cleanup
            if pod_id:
                requests.delete(
                    f"{self.base_url}/pods/{pod_id}",
                    headers=self.headers
                )
                print(f"Terminated pod: {pod_id}")

# Usage
if __name__ == "__main__":
    benchmark = PrimeIntellectBenchmark(os.environ['PRIME_API_KEY'])
    results = benchmark.run_benchmark("T4")
    print(json.dumps(results, indent=2))
```

This reference document provides everything needed to interact with Prime Intellect's API for GPU benchmarking purposes.