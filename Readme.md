# Deploying an LLM using vLLM
This repo has the learning steps for deploying a large language model on AWS EC2-GPU instance. For the demo purpose I will be using qwen-3 0.6B parameter model on g4dn.xlarge instance of EC2, which is Tesla L4 GPUs and has 16GB vRAM and 125GB storage.

## Load Testing
Load testing an GPU for how many concurrent request it can handle, see Load_test folder for the detailed theoretical calculation and experiments result.