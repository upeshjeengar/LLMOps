# AGENTS.md

LLM deployment repository with load testing, quantization guides, and AWS EC2-GPU deployment patterns.

## Setup Commands

AWS EC2 g4dn.xlarge setup:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv && source .venv/bin/activate
uv pip install vllm --torch-backend=auto
```

Start vLLM server (required before load testing):
```bash
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3-0.6B \
  --max-model-len 512 \
  --max-num-seqs 2000 \
  --enforce-eager
```

Run load test (from `Load_test/` directory):
```bash
cd Load_test && python load_test.py
```

## Key Constraints

- vLLM server must be running on localhost:8000 before load testing
- Load test modifies `CONCURRENT_REQUESTS` variable in `load_test.py:11` to test different loads
- MLflow UI accessed via `ssh -L 5000:localhost:5000 remote` when running on remote server

## Quantization Documentation

`Quantization/Readme.md` covers:
- CPU: llama.cpp with GGUF format
- GPU: TensorRT-ModelOpt (speed), AutoGPTQ/AutoAWQ (ease-of-use), QLoRA (training)