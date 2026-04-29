There are two types of quantization methods on the basis of type of machine you are using:
1. CPU
The best quantization framework for CPU inference is llama.cpp, which utilizes the GGUF.
2. GPU
For GPU quantization, top frameworks include NVIDIA TensorRT-ModelOpt for maximum inference speed, and AutoGPTQ/AutoAWQ for ease-of-use with LLMs. QLoRA (via bitsandbytes) is best for training, while GPTQ/AWQ are ideal for post-training 4-bit inference, ensuring minimal accuracy loss with high throughput on NVIDIA hardware

# GGUF 
GGUF quantization implements Post-Training Quantization (PTQ): given an already-trained Llama-like model in high precision, it reduces the bit width of each individual weight. The resulting checkpoint requires less memory and thus facillitates inference on consumer-grade hardware.

There are 3 quantization methods

## Legacy Quantization Methods
Legacy quantization methods are the first generation of quantization algorithms in llama.cpp. Even though they are deprecated as standalone quantization methods, they remain relevant because their successors (K-quants and I-quants) build upon them rather than replace them entirely.
Legacy quants use affine quantization, mapping each floating-point scalar weight to a lower-bit integer.

Legacy quants come in two main subcategories that correspond to symmetric and asymmetric quantization:
### Type 0 (Symmetric QUantization)
- Type 0 (symmetric)

![alt text](<Symmetric Quantization.png>)
![alt text](<Symmetric Quantization 2.png>)

- Type 1 (asymmetric)
Type 1 quantization uses asymmetric quantization: it uses integer bins more effectively, even when the weight clipping range is not symmetric (e.g. [-1, +2]).
![alt text](<Asymmetric Quantization.png>)

Block Quantization
![alt text](<Block Quantization.png>)

## K-Quants


## 