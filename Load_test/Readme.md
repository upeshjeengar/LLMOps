# Load testing an LLM using vLLM
This folder has the learning steps for deploying a large language model on AWS EC2-GPU instance. For the demo purpose I will be using qwen-3 0.6B parameter model on g4dn.xlarge instance of EC2, which is Tesla L4 GPUs and has 16GB vRAM and 125GB storage.

## How Many Requests Can the System Handle at Once?
Let's estimate the number of request our machine can handle concurrently without breaking(this is just a theoretical estimate, in reality various other external factors like input and output size, embedding sizes extra will play a significant role ). 
So let's assume an average number of input tokens to be 500 and average number of output tokens to be 2500, so in total 2500 tokens per request, 
for this we'll be required to calculate KV Cache sizes, vLLMs uses KV caching as optimization technique for faster inference(skips calculating attention matrices again and again for previous tokens), but that leads to KV cache overhead.

### Total memory distribution
* Total VRAM: 16 GB
* Model Weights (FP16): ~1.2 GB (0.6 Billion parameters $\times$ 2 bytes)
* vLLM Framework Overhead: ~1.5 GB
* Remaining VRAM for KV Cache: ~13.3 GB

### Qwen3-0.6B specs:
* Layers: 28
* Key-Value (KV) Heads: 8
* Head Dimension: 64
* Bytes per parameter (using FP16): 2

so total cache size per token will be = 2 (Key and Value) $\times$ Layers  $\times$ KV Heads  $\times$ Head Dimension  $\times$ Bytes per parameter
which will be 2 $\times$  28 $\times$  8 $\times$  64 $\times$ 2 = 57344

So total cache size per token will be 57344 bytes, one request will have 2500 tokens on average so one request will take 57344 $\times$ 2500 bytes or (57344 $\times$ 2500)/(1024*1024) = 136 MB per request and we have approx 13GB remaining: 13Gb/136Mb = 98 requests


# Observations
## Quick setup guide
run this commands: 
```
curl -LsSf https://astral.sh/uv/install.sh | sh #install uv
uv pip install vllm --torch-backend=auto # vllm with uv to auto select backend
```

Serving model as http request endpoint(for production ready setup)
```
!python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-0.6B \  # defining model name
    --max-model-len 512 \      # model will not use more than 512 tokens per request
    --max-num-seqs 2000 \      # max sequence this endpoint will be able to handle
    --enforce-eager            # for compatibility and debugging
```

Let's move on to practicals, I will be using vLLM framework and will run on different GPUs for load_testing

### Model
I am running Qwen3-0.6B on all the GPUs with a fix input size of 4 and setting the max output tokens 100, so according to above calculations, our model will require 57344(total cache size in byte per token) * 106(assuming 4 words will become 6 tokens on average and max 100 output tokens) ≈ 5.8 MB per request

## Tesla T4 GPU
Configurations:
VRAM: 15GB
Storage: 112GB

### Expected requests we can handle
Memory we have: 12.3 GB(calculations [here](#total-memory-distribution))

12.3*1024/5.8 ≈ 2171 requests we can theoretically handle

### Actual

Let's see some logs:    

### 500 requests: 
🚀  Starting load test with 500 requests
📊  Results
Time: 9.96s
Success: 500/500
Throughput: 50.21 req/s
Token Throughput: 5021.16 tok/s
💾  Saved results to output.csv

### 1000 requests:
🚀 Starting load test with 1000 requests
📊  Results
Time: 22.20s   
Success: 1000/1000  
Throughput: 45.04 req/s  
Token Throughput: 4503.76 tok/s  
💾  Saved results to output.csv  

### 1500 requests:
🚀  Starting load test with 1500 requests
📊  Results
Time: 31.02s   
Success: 957/1500  
Throughput: 30.85 req/s  
Token Throughput: 3084.64 tok/s  
💾  Saved results to output.csv 

so practical breaking point is in between 1000 to 1500, let's find it out with binary search technique...
So upon experimentating, we are able to serve 1346 requests when I have average 5 input tokens and 100 output tokens per request.


# On AWS g4dn.xlarge
## setting up cuda kernels on this machine:
```
sudo apt update
sudo apt install python3 python3-dev python3-pip
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv
source .venv/bin/activate
uv pip install vllm --torch-backend=auto
# Install vLLM with a specific CUDA version (e.g., 13.0).
export VLLM_VERSION=$(curl -s https://api.github.com/repos/vllm-project/vllm/releases/latest | jq -r .tag_name | sed 's/^v//')
export CUDA_VERSION=130 # or other
export CPU_ARCH=$(uname -m) # x86_64 or aarch64

#start server at 8000:
python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen3-0.6B --max-model-len 512 --max-num-seqs 2000 --enforce-eager
 
python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen3-0.6B --max-model-len 512 --max-num-seqs 2000 --enforce-eager --dtype float16   --attention-backend TRITON_ATTN # use triton attention if flash attention doesn't support
 
vllm serve unsloth/Llama-3.2-1B-Instruct
curl http://localhost:8000/v1/chat/completions -H "Content-Type: application/json" -d '{"model":"unsloth/Llama-3.2-1B-Instruct","messages":[{"role":"user","content":"Hello!"}]}'

python load_test.py # to laod test
```

## Few logs

❌ Concurrency 2500 failed (breakpoint reached). Trying a lower load...
❌ Concurrency 1250 failed (breakpoint reached). Trying a lower load...

--------------------------------------------------
⚡ Testing Concurrency Level: 625
--------------------------------------------------
🚀 Starting load test with 625 requests
📊 Results
Time: 17.33s   
Success: 625/625  
Throughput: 36.06 req/s  
Token Throughput: 3606.20 tok/s  
💾 Saved results to output.csv  
✅ Concurrency 625 succeeded! Trying a higher load...

--------------------------------------------------
⚡ Testing Concurrency Level: 937
--------------------------------------------------
🚀 Starting load test with 937 requests
📊 Results
Time: 28.99s   
Success: 937/937  
Throughput: 32.33 req/s  
Token Throughput: 3232.64 tok/s  
💾 Saved results to output.csv  
✅ Concurrency 937 succeeded! Trying a higher load...

--------------------------------------------------
⚡ Testing Concurrency Level: 1015
--------------------------------------------------
🚀 Starting load test with 1015 requests
📊 Results
Time: 29.68s   
Success: 1015/1015  
Throughput: 34.20 req/s  
Token Throughput: 3419.78 tok/s  
💾 Saved results to output.csv  
✅ Concurrency 1015 succeeded! Trying a higher load...

--------------------------------------------------
⚡ Testing Concurrency Level: 1054
--------------------------------------------------
🚀 Starting load test with 1054 requests

❌ Concurrency 1041 failed (breakpoint reached). Trying a lower load... 

--------------------------------------------------
⚡ Testing Concurrency Level: 1040
--------------------------------------------------
🚀 Starting load test with 1040 requests
📊 Results
Time: 30.73s   
Success: 1040/1040  
Throughput: 33.84 req/s  
Token Throughput: 3384.30 tok/s  
💾 Saved results to output.csv  
✅ Concurrency 1040 succeeded! Trying a higher load...

==================================================
🏆 MAXIMUM STABLE CONCURRENCY (BREAKPOINT): 1040
==================================================


## Conclusion: 
The reason for difference in theoretical and practical concurrency: theoretical concurrency is a memory-bound upper limit, while practical concurrency is constrained by multiple interacting bottlenecks—compute, scheduling, fragmentation, and system overhead.

Why our results actually make sense
Theoretical: ~2171
Practical stable: ~1000–1300
👉 That’s a very normal ratio (~50%) for LLM serving systems
In production systems, engineers often:
target only 60–70% of theoretical capacity

to avoid instability spikes
The reason google colab's GPU is showing higher concurrency is because it was already optimized, but on g4dn I was installing all kernels by myself with base settings, so no optimization.