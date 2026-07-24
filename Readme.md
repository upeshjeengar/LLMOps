# My 3-Year LLM Engineering Learning Notes

> A practical, first-principles path from tokens and transformers to fine-tuning, reasoning models, inference systems, RAG, MCP, and agents.

Hi, I'm **Upesh Jeengar**. Over the last three years, I have kept notes while learning how large language models work and how to build applications around them. This repository is my attempt to turn those scattered notes, papers, videos, notebooks, and implementation lessons into a learning path that somebody else can actually follow.

The goal is not to memorise every acronym in the LLM ecosystem. It is to build a mental model strong enough that you can read a paper, implement a small version of an idea, understand its trade-offs, and decide when it is useful in a real product.

This guide begins with the mechanics of a decoder-only language model: tokenisation, embeddings, attention, transformer blocks, pre-training, and generation. It then moves into the work that turns a base model into an engineering system: fine-tuning, reinforcement learning, evaluation, efficient inference, retrieval, tool use, MCP, multi-agent systems, and multimodality.

Most of these resources are the exact links I used while learning. I have added context around each one so you know *why* it is worth opening and *when* to use it.

## Table of contents

- [How I would use this repository](#how-i-would-use-this-repository)
- [The learning roadmap](#the-learning-roadmap)
- [Part 1: Foundations and the language-model lifecycle](#part-1-foundations-and-the-language-model-lifecycle)
- [Part 2: Text, tokenisation, and embeddings](#part-2-text-tokenisation-and-embeddings)
- [Part 3: Attention and transformer architecture](#part-3-attention-and-transformer-architecture)
- [Part 4: Pre-training and building an LLM](#part-4-pre-training-and-building-an-llm)
- [Part 5: Fine-tuning, alignment, and reasoning](#part-5-fine-tuning-alignment-and-reasoning)
- [Part 6: Evaluation](#part-6-evaluation)
- [Part 7: Efficient inference and serving](#part-7-efficient-inference-and-serving)
- [Part 8: MoE and multimodal models](#part-8-moe-and-multimodal-models)
- [Part 9: RAG, tool use, MCP, and agents](#part-9-rag-tool-use-mcp-and-agents)
- [Part 10: Papers and research directions](#part-10-papers-and-research-directions)
- [Resource directory](#resource-directory)
- [A practical project sequence](#a-practical-project-sequence)
- [Contributing](#contributing)

## How I would use this repository

I do not recommend trying to finish every resource before building anything. That creates the familiar feeling of "learning a lot" without gaining engineering judgement. Instead, run a loop:

1. Learn one concept well enough to explain it in plain language.
2. Implement a deliberately small version of it.
3. Measure or inspect what changes when you remove it.
4. Read the original paper or a deeper resource only after the implementation gives you questions.

For example, do not begin by reading every paper about FlashAttention. First implement ordinary scaled dot-product attention, notice the `O(n^2)` attention-score matrix, and profile its memory behaviour. Then the ideas of tiling, online softmax, and recomputation become concrete rather than magical.

I also recommend keeping your own notes. Write down:

- What problem does this technique solve?
- What computation, memory, latency, data-quality, or safety cost does it introduce?
- What assumption does it rely on?
- How would I know it worked in my system?

Those four questions have helped me much more than a collection of definitions.

## The learning roadmap

This is the order I would follow today. Treat the time ranges as flexible; the outcome matters more than the calendar.

| Phase | Main outcome | Suggested focus |
| --- | --- | --- |
| 1. Foundations | Explain the base-model lifecycle and transformer vocabulary | Pre-training, tokenisation, embeddings, attention |
| 2. Build a tiny GPT | Train and sample from a small decoder-only model | Dataset windows, causal masking, transformer blocks |
| 3. Post-training | Adapt a model to a useful task safely and economically | Continued pre-training, SFT, LoRA/QLoRA, preference optimisation |
| 4. Evaluation | Measure quality rather than relying on demos | Task benchmarks, retrieval metrics, human and rule-based evaluation |
| 5. Inference | Understand why production serving is a systems problem | KV cache, GQA, PagedAttention, FlashAttention, quantisation |
| 6. Applications | Build systems that use knowledge and take bounded actions | RAG, tool calling, MCP, routing, agents |
| 7. Research | Form independent opinions about new model ideas | MoE, JEPA, multimodality, reasoning RL |

## Part 1: Foundations and the language-model lifecycle

### What an LLM is - and is not

At its simplest, an autoregressive LLM is trained to predict the next token. It reads a sequence of token IDs and learns a probability distribution for the token that should follow. This looks simple, but at enough scale and with enough diverse data, the same objective supports completion, summarisation, translation, extraction, coding assistance, and many other behaviours.

The useful distinction is between a **base model** and a **post-trained model**:

- A base or foundation model is produced through pre-training on large-scale, mostly unlabeled text. The labels are created from the text itself: given a prefix, predict its continuation. This is self-supervised learning.
- An instruction model is a base model adapted to follow requests using high-quality instruction/response data.
- A classification model is adapted to predict an explicit label, such as spam/not-spam or positive/negative sentiment.
- An agentic application is not merely a model. It is a system around a model: it can retrieve context, choose tools, perform actions, keep state, and be evaluated against a task.

Not every transformer is an LLM, and not every LLM must be a transformer. Transformers are also used in vision and other modalities; language models have historically used recurrent, convolutional, and other architectures. The point of learning the transformer deeply is that it is the dominant practical architecture for modern LLM work, not that it is the only imaginable one.

### Start here: the resources that shaped my foundation

#### [Build a Large Language Model (From Scratch) - Sebastian Raschka](https://sebastianraschka.com/books/#build-a-large-language-model-from-scratch)

This was one of my main first-principles resources. It is the best starting point in this list if you want to go from text processing to a working GPT-style implementation without treating the architecture as a black box. Use it as the spine for the first half of this guide: read, code, then modify the implementation.

#### [Vizuara: Building LLMs from Scratch playlist](https://www.youtube.com/playlist?list=PLPTV0NXA_ZSgsLAr8YCgCwhPIJNNtexWu)

Use this alongside the book when you want a visual or intuitive explanation before opening the code. It is particularly helpful for attention, transformer shapes, and the progression from token IDs to model outputs.

#### [Attention Is All You Need](https://arxiv.org/abs/1706.03762)

This is the original Transformer paper. Do not worry if every detail does not make sense on the first read. Read the architecture diagram first, then return after you have implemented attention, masking, multi-head attention, normalisation, and feed-forward layers. At that point, the paper becomes an engineering document rather than a wall of notation.

#### [Language Models are Few-Shot Learners (GPT-3)](https://arxiv.org/pdf/2005.14165)

Read this to understand why few-shot prompting became a major capability discussion. The useful takeaway is not only the model size; it is the evaluation framing: one model can be prompted to perform many tasks without task-specific gradient updates. Compare zero-shot, one-shot, and few-shot settings yourself whenever you evaluate a model.

#### [OpenAI language-understanding paper](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf)

This is the OpenAI paper linked from my early GPT notes. Use it as a historical decoder-only reference: GPT-style models generate continuations autoregressively and can transfer learned language patterns to downstream tasks.

## Part 2: Text, tokenisation, and embeddings

Before a model can learn language, text has to become numbers. This part of the pipeline sounds mundane, but choices here shape context length, cost, multilingual behaviour, code handling, and even which strings can be represented efficiently.

### 1. Tokenisation

The usual path is:

```text
raw text -> tokens/subwords -> token IDs -> embedding vectors
```

Modern LLMs generally use subword tokenisation rather than pure word tokenisation. A tokenizer should keep frequent pieces intact when useful while splitting uncommon words into reusable pieces. That means a model can share information among related words such as `token`, `tokens`, and `tokenization`, and it can represent previously unseen words without creating an unlimited word-level vocabulary.

### 2. Byte Pair Encoding (BPE)

BPE repeatedly merges frequent pairs into a learned vocabulary. It began as a compression idea, but it is a natural fit for subword tokenisation. The important engineering lesson is that the tokenizer is learned from data too: change the corpus or vocabulary size and you change how text is segmented, how many tokens a prompt consumes, and which patterns are easy for the model to represent.

#### [OpenAI tiktoken](https://github.com/openai/tiktoken)

This is a practical open-source implementation of OpenAI-style BPE tokenisation. Use it to inspect token counts, experiment with encodings, and develop an instinct for why "one word equals one token" is only a rough approximation.

#### [Breaking Down Text: How BPE Tokenization Works](https://ruby-scowl-263.notion.site/Breaking-Down-Text-How-BPE-Tokenization-Works-19bb5dd44d5a806b95defbe96ac464e0)

Read this for an intuitive walkthrough of the merge process. It is a good bridge between the compression algorithm and the behaviour you see in real tokenizers.

#### [Tokenizer implementation notebook](https://colab.research.google.com/drive/1yzRlhATL3QSZdkp3jaNI-XKFqVv9u90R?usp=sharing)

This is the notebook I used while working through a tokenizer implementation. Use it as a companion exercise: change the vocabulary size, inspect token boundaries, and see how special tokens affect the stream.

#### [The Verdict - Edith Wharton](https://public.wsu.edu/~campbelld/wharton/books/verdict.htm)

This public-domain text was used in my notes as a small corpus. It is excellent for a tiny end-to-end experiment because you can train quickly, inspect every stage, and avoid pretending that a toy dataset will produce a general-purpose model.

### 3. Input-target pairs and causal training

For a decoder-only model, a text sequence is shifted by one token:

```text
input:  [t0, t1, t2, t3]
target: [t1, t2, t3, t4]
```

Training examples are often created with a sliding window. The model predicts the next token at each position but is prevented from looking into the future. This is why the causal mask is not an optional implementation detail: without it, the model could cheat during training by seeing the answer token.

### 4. Token and position embeddings

An embedding table maps each token ID to a dense vector. The values begin as learned parameters, usually randomly initialised, and become useful only through training. Token embeddings encode the identity and learned usage of a token; they do not tell the model where the token occurs.

Position information solves that problem. A model needs to distinguish "dog bites man" from "man bites dog." In a classic absolute-position design, a learned position vector is added to each token vector. Relative-position approaches instead emphasise distance and direction between tokens, which can generalise better across sequence lengths and locations.

#### [word2vec Google News embeddings](https://huggingface.co/fse/word2vec-google-news-300)

This is a useful historical and practical reference for dense word vectors. Explore it to build intuition for semantic similarity, but remember that static embeddings give each word one fixed vector. Transformer representations are contextual: the representation of `bank` can change with the words around it.

## Part 3: Attention and transformer architecture

### Why attention replaced the old bottleneck

Before transformers, sequence-to-sequence RNN encoder-decoder systems passed the source sequence through a recurrent encoder and tried to capture its meaning in a final hidden state. That became a bottleneck on long sequences: the decoder had limited direct access to early source information.

Attention changes this. At a decoding step, the model can score all relevant positions and build a weighted combination of their values. The important word is *weighted*: the model does not simply copy every token equally; it learns which information is useful for the current representation.

#### [Visualizing Neural Machine Translation Mechanics of Seq2seq Models with Attention](https://jalammar.github.io/visualizing-neural-machine-translation-mechanics-of-seq2seq-models-with-attention/)

This is a strong visual explanation of the encoder-decoder attention problem that preceded transformers. Read it before or alongside self-attention if the motivation for attention feels abstract.

### The attention calculation

Each token representation is projected into three learned spaces:

- **Query (Q):** what this position is looking for.
- **Key (K):** what each candidate position offers for matching.
- **Value (V):** the information to be aggregated once relevance is determined.

In scaled dot-product attention, the core expression is:

```text
Attention(Q, K, V) = softmax((QK^T / sqrt(d_k)) + mask) V
```

The dot product scores query-key compatibility. Dividing by `sqrt(d_k)` prevents the score magnitude from growing too aggressively as the key dimension grows, which helps softmax and gradients stay well behaved. The softmax normalises each row into attention weights. The weighted sum of values produces contextualised token representations.

### Causal attention

GPT-style generation needs a causal (masked) version of self-attention. Position `i` can attend to tokens up to `i`, never to `i + 1` and beyond. A common implementation places negative infinity above the diagonal before softmax, making future positions receive zero probability.

This aligns training with generation. At inference time, the model does not know future tokens, so it should not have access to them during training either.

### Multi-head attention

One attention head learns one set of projections. Multi-head attention runs several attention heads in parallel, allowing different subspaces to specialise in different relationships, then concatenates and projects their outputs. It is not "several models"; it is a structured way to let one layer represent multiple patterns at once.

### The rest of a transformer block

A useful mental model for a transformer block is:

```text
input
  +- layer normalisation -> multi-head causal attention -> residual addition
  +- layer normalisation -> feed-forward network (with GELU) -> residual addition
output
```

- **Layer normalisation** stabilises the scale of activations and makes optimisation more reliable.
- **Feed-forward layers** transform each token position independently after attention has mixed information across positions.
- **GELU** is a smooth activation function commonly used in transformer feed-forward layers.
- **Residual/shortcut connections** create direct gradient paths and preserve useful information through depth.
- **Dropout** is a regularisation tool used during training; it is generally disabled at inference.

Implementation milestone: write the shapes down at every operation. Most attention bugs are shape bugs, mask-broadcasting bugs, or head-layout bugs disguised as modelling problems.

## Part 4: Pre-training and building an LLM

Pre-training is next-token prediction over a large corpus. The loss is typically cross-entropy between the model's logits and the shifted target token IDs. The output is a base model: it has learned language patterns and broad statistical knowledge, but it is not yet necessarily helpful, truthful, safe, or aligned with a specific task.

When implementing a small model, I would make the following checkpoints explicit:

1. A deterministic tokenizer and vocabulary.
2. A dataset that yields correctly shifted, fixed-length input-target windows.
3. A causal attention mask that is tested independently.
4. A transformer block that preserves the expected dimensions.
5. A training loop with validation loss, checkpointing, and sample generation.
6. A generation function with temperature, top-k/top-p sampling, and an end-of-text condition.

Tiny models trained on small corpora will overfit quickly. That is not a failure: it is a diagnostic tool. You are learning to validate data flow, loss behaviour, masks, and sampling before the cost of errors becomes large.

## Part 5: Fine-tuning, alignment, and reasoning

Pre-training gives a general distribution over text; post-training makes the model more useful for a desired interaction or task. This is where data quality, task definition, and evaluation often matter more than architectural novelty.

### Continued pre-training

Before paying for task-specific labels, you can continue self-supervised training on domain text. For example, a legal assistant may benefit from continued pre-training on a carefully licensed legal corpus before supervised instruction fine-tuning. This teaches domain terminology and distributional patterns using comparatively cheap unlabelled data.

### Supervised fine-tuning (SFT)

SFT trains on curated input-output examples. Its success depends on the quality, diversity, and consistency of those examples. Define the desired behaviour before you collect data: tone, formats, refusal boundaries, tool-call schemas, and what a correct answer looks like.

### Full fine-tuning, LoRA, and QLoRA

Full fine-tuning updates every trainable parameter. It can be powerful, but it is expensive, slow, and more prone to overfitting when the dataset is small. Parameter-efficient fine-tuning methods such as **LoRA** add small trainable low-rank adapters while keeping the base weights frozen. **QLoRA** combines a quantised base model with adapter training, making experimentation much more accessible.

My rule of thumb: start with prompting or retrieval; then try LoRA/QLoRA if the deficiency is stable and repeated; only consider full fine-tuning when the evidence and budget justify it.

### Reinforcement learning and preference optimisation

Reinforcement learning treats the model as a policy that produces actions/tokens in response to a state or prompt. A reward signal tells the training process which outcomes are preferred. This is useful when a simple supervised target does not capture the objective, such as verified code execution, a structured format, human preference, or multi-step task success.

#### [Stanford CS336 language modelling lectures](https://www.youtube.com/playlist?list=PLoROMvodv4rOCXd21gf0CF4xr35yINeOy)

I used this lecture series as a source for the reasoning and reinforcement-learning notes. In particular, [lecture 6](https://www.youtube.com/watch?v=k5Fh-UgTuCo&list=PLoROMvodv4rOCXd21gf0CF4xr35yINeOy&index=6) is a useful place to deepen the connection between language modelling and RL-style post-training.

### PPO and GRPO

**PPO (Proximal Policy Optimisation)** constrains policy updates so the fine-tuned model does not move too far from the previous/reference policy in a single update. The clipping/KL-control idea is important: aggressively optimising a noisy reward can destroy useful behaviour.

**GRPO (Group Relative Policy Optimisation)** compares multiple sampled outputs for the same prompt and estimates which samples are better relative to the group. It can avoid relying on a learned value model for the advantage estimate. In reasoning settings, this fits naturally with several rollouts, verifiable rewards, and selecting the stronger reasoning trajectory.

For both methods, reward design is the real product requirement. A reward can combine:

- **Accuracy:** does a program pass tests, is a math answer correct, or did a tool action succeed?
- **Format:** did the output meet a required schema or delimiter convention?
- **Safety/policy:** did the system respect constraints?
- **Preference:** do qualified reviewers prefer the response?

Avoid optimising visible process text merely because it looks like reasoning. Reward outcomes that can be checked, and evaluate whether the model is gaming the signal.

### Reasoning models

Reasoning-oriented models benefit from sampling, verification, and RL training on tasks with objective signals such as math, coding, or structured puzzles. Benchmarks commonly used in this area include HumanEval, Codeforces-style problems, SWE-bench, AIME, and GSM8K.

The metric `pass@k` asks: if I sample `k` attempts, what is the probability that at least one succeeds? This is often more revealing than judging a single deterministic sample, but it must be paired with compute-aware reporting: a model that needs 100 attempts to solve a task has a very different product profile from one that solves it on the first attempt.

## Part 6: Evaluation

Evaluation should be designed before a fine-tuning or agent project begins. A polished demo is not an evaluation plan.

### Use multiple measurement layers

- **Task outcome:** exact match, executable tests, policy compliance, or business-process completion.
- **Retrieval quality:** recall at candidate selection, then precision after reranking.
- **Model behaviour:** instruction following, groundedness, latency, format validity, and robustness to adversarial prompts.
- **Human evaluation:** valuable for usefulness and nuance, but expensive and subject to rater disagreement.
- **Rule-based evaluation:** cheap and repeatable for schemas, citations, arithmetic, code tests, or tool outcomes, but narrow.

For retrieval, I separate the pipeline into candidate generation and reranking. Candidate generation should maximise recall: the correct document needs to be present. Reranking should maximise precision: the best evidence should reach the model. Hybrid retrieval combines semantic/vector similarity with lexical methods such as BM25, because each catches failures that the other misses.

## Part 7: Efficient inference and serving

Training teaches the weights; inference determines whether people can use the model economically. Long prompts, concurrent users, output length, memory bandwidth, and scheduling can dominate the practical experience.

### KV caching

During autoregressive generation, attention needs keys and values for prior tokens. Recomputing them for the entire prefix at each generated token is wasteful. A **KV cache** stores those past key/value tensors, so the model computes K/V only for the new token and attends over the cached history.

There are two phases:

1. **Prefill:** process the input prompt and build the initial KV cache.
2. **Decode:** generate new tokens, extending the cache one step at a time.

KV caching substantially improves speed but creates a memory-management problem, especially under concurrent requests and long contexts.

#### [KV caching video](https://www.youtube.com/watch?v=CxRGWfcGVbs)

Use this as an intuitive introduction to why naive decoding repeats work and how cached keys/values change the computation.

### Grouped-query attention (GQA)

GQA allows multiple query heads to share key/value heads. It reduces the KV-cache footprint and memory bandwidth demand while retaining many of the benefits of multi-head attention. It is one of the important reasons modern decoder models can serve larger contexts more efficiently.

#### [Grouped-query attention video](https://www.youtube.com/watch?v=mtsY7JsGQjw)

This resource focuses on the relationship between attention-head design and inference cost. Watch it after you understand normal multi-head attention and KV caching.

### PagedAttention

PagedAttention manages the KV cache in fixed-size blocks rather than requiring every request to reserve one large contiguous region. This reduces fragmentation and enables continuous batching: requests at different generation stages can share the server efficiently.

#### [PagedAttention video](https://www.youtube.com/watch?v=-AB6m0Spo6c)

Use this to develop intuition for block-based KV-cache allocation and why request scheduling is part of LLM serving.

#### [Operating-system paging primer](https://www.geeksforgeeks.org/operating-systems/paging-in-operating-system/)

PagedAttention is easier to understand if you already understand the paging analogy: virtual blocks can map to physical blocks, allowing flexible allocation without assuming one contiguous memory region.

### FlashAttention

Standard attention materialises an `N x N` score matrix, which becomes expensive for long sequences. FlashAttention reorganises the calculation around GPU memory hierarchy. It tiles Q, K, and V into blocks, performs work in fast on-chip memory where possible, uses an online-softmax calculation to preserve correctness, and recomputes some values during backpropagation instead of storing a huge intermediate matrix.

The central lesson is not merely "FlashAttention is faster." It is that I/O between memory levels can be the bottleneck even when arithmetic is cheap. In modern ML systems, data movement is often the real algorithmic cost.

#### [Decoding Flash Attention](https://outcomeschool.com/blog/decoding-flash-attention)

Read this after implementing ordinary attention. It gives a practical explanation of tiling, online softmax, and the memory trade-off behind the algorithm.

### Quantisation, distillation, and caching

**Quantisation** stores and/or computes model values at lower precision to reduce memory use and potentially increase throughput. **Distillation** trains a smaller student model to reproduce useful behaviour from a larger teacher. **Prefix caching** reuses the KV cache of common prompt prefixes, such as a large stable system prompt or shared document context.

#### [My LLMOps quantisation notes](https://github.com/upeshjeengar/LLMOps/tree/main/Quantization)

This is my repository section for model quantisation. It is the practical companion to this README when you want to go beyond definitions and work through the deployment trade-offs.

#### [Model distillation video](https://www.youtube.com/watch?v=jrJKRYAdh7I)

Use this to understand the teacher-student idea and why matching a teacher's distributions or outputs can produce smaller, cheaper models.

## Part 8: MoE and multimodal models

### Mixture of Experts (MoE)

MoE architectures replace some dense feed-forward layers with many experts and a router/gating network. The router sends each token to only a small subset of experts. This can scale total parameter count without activating every parameter for every token.

The trade-off is not free. MoE training needs load balancing; routing can be unstable; and inference still needs sufficient memory to hold the model's parameters. When evaluating an MoE, distinguish **total parameters**, **active parameters per token**, memory requirements, and actual serving throughput.

#### [Mixture of Experts video](https://www.youtube.com/watch?v=v7U21meXd6Y)

This is a useful visual resource for building the router/expert mental model before reading implementation details.

### Multimodal LLMs

Multimodal systems turn inputs such as text, images, audio, or video into compatible learned representations, then allow a model to reason across them. The details differ across architectures, but the recurring idea is alignment: different modalities must become representations that can be used together.

#### [Vision Transformers (ViT) overview](https://viso.ai/deep-learning/vision-transformer-vit/)

This is a useful introductory reference for how the Transformer idea extends beyond language. It explains the patch-based view of images that makes the later ViT versus NaViT comparison easier to follow.

### Kimi K2.5, NaViT, and visual reasoning

My notes also explore Kimi K2.5 and the contrast between fixed-resolution ViT-style processing and NaViT-style variable-resolution packing. Standard ViT pipelines often resize images to a fixed size and split them into fixed patches. NaViT-like approaches can pack variable-resolution images more flexibly, which is valuable when native detail and efficiency both matter.

#### [My Kimi K2.5 architecture article](https://upesh-jeengar.medium.com/seeing-thinking-acting-inside-kimi-k2-5s-architecture-47d9ae239671)

I wrote this article to unpack the model's visual and agentic architecture at a more applied level.

#### [Kimi K2.5 research paper](https://arxiv.org/abs/2602.02276)

Read the primary paper after the article if you want the architectural claims, training details, and evaluation evidence in the authors' words.

## Part 9: RAG, tool use, MCP, and agents

The model is only one component of an LLM product. Many valuable applications work because they give the model access to current, private, or structured information and bounded ways to act.

### Retrieval-augmented generation (RAG)

RAG has three broad stages:

```text
retrieve relevant evidence -> add it to the model context -> generate a grounded response
```

The hard part is retrieval quality, not simply adding a vector database. Start with document parsing, chunking, metadata, and a retrieval benchmark. Use embeddings for semantic recall, BM25 for lexical precision, and hybrid retrieval when it performs better on your evaluation set. Then rerank a manageable candidate set with a stronger model.

Do not call an answer grounded simply because retrieval happened. Track whether the evidence was retrieved, whether it supports the answer, whether the answer cites it correctly, and whether the system abstains when evidence is missing.

### Tool calling

Tool calling lets a model request actions from deterministic systems: search, database queries, calculations, ticketing, email drafts, code execution, and business workflows. A robust system validates tool inputs, enforces authorization outside the model, limits side effects, logs decisions, and returns structured tool results to the model.

Tool selection is an engineering problem of its own. Routing to a smaller relevant tool subset can reduce latency, cost, and accidental use of dangerous capabilities.

### Model Context Protocol (MCP)

MCP standardises the connection between AI applications and external tools or data sources. I think of it as a common interface: instead of every AI client building a custom integration for every service, a host can use protocol-speaking clients to connect to servers that advertise their tools, resources, and prompts.

The core pieces are:

- **Host:** the user-facing AI application or agent environment.
- **Client:** a connection manager maintained by the host for an MCP server.
- **Server:** the component that exposes access to a capability or data source, such as a repository, database, or workspace service.

MCP does not remove the need for security. It makes integration more consistent; you still need authentication, authorization, input validation, least privilege, audit logging, and clear confirmation boundaries for consequential actions.

### Agents and multi-agent systems

An agent is a system that pursues a goal on a user's behalf. In practice, that means it can decide whether it needs context, choose a tool or subtask, observe results, and continue until a stop condition. More agents are not automatically better. Begin with one agent, a clear task state, explicit tool boundaries, and measurable success criteria. Add routing or specialised agents only when one component is demonstrably overloaded or needs different permissions.

#### [Enterprise AI Automation](https://github.com/upeshjeengar/enterprise-ai-automation)

This is my multi-agent enterprise-automation project. It routes employee requests through intent recognition, applies policy-driven approval paths, and can block prompt-injection attempts. The repository currently uses mock services for systems such as SAP Ariba, GitHub, and Slack, but the architecture is designed to be integrated with enterprise services. It is a concrete example of why agent design must include access control and business rules, not only model prompts.

## Part 10: Papers and research directions

### Attention residuals

Standard residual connections add earlier activations directly. The attention-residual idea in my notes asks whether a deeper layer could learn to weight information from prior layer representations rather than uniformly accumulating them. It is a useful research direction to keep in mind: residual paths solve optimisation problems, but their design can itself be a representational choice.

### JEPA: Joint Embedding Predictive Architecture

JEPA is a self-supervised approach that predicts representations of hidden or target content from visible context rather than reconstructing raw pixels or relying only on contrastive objectives. A simplified picture is:

1. A context encoder represents visible input.
2. A target encoder represents the hidden target region, often with a slowly updated teacher-like mechanism.
3. A predictor learns to predict the target representation from the context representation.

The motivation is to encourage abstraction and world modelling rather than literal pixel reconstruction. This is worth studying if you are interested in alternatives to next-token or next-pixel prediction, especially in vision and multimodal learning.

## Resource directory

This compact directory makes every resource in the notes easy to find again.

| Topic | Resource | Why I used it |
| --- | --- | --- |
| End-to-end LLM implementation | [Sebastian Raschka's book](https://sebastianraschka.com/books/#build-a-large-language-model-from-scratch) | First-principles route from tokens to a GPT-style model |
| Visual foundation | [Vizuara playlist](https://www.youtube.com/playlist?list=PLPTV0NXA_ZSgsLAr8YCgCwhPIJNNtexWu) | Visual companion for the core concepts |
| Transformer architecture | [Attention Is All You Need](https://arxiv.org/abs/1706.03762) | Original Transformer paper |
| GPT history | [OpenAI language-understanding paper](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf) | Decoder-only language-model reference from my notes |
| Few-shot learning | [GPT-3 paper](https://arxiv.org/pdf/2005.14165) | Prompt-based generalisation and evaluation framing |
| Tiny text corpus | [The Verdict](https://public.wsu.edu/~campbelld/wharton/books/verdict.htm) | Small public-domain corpus for experiments |
| Tokenisation code | [Colab notebook](https://colab.research.google.com/drive/1yzRlhATL3QSZdkp3jaNI-XKFqVv9u90R?usp=sharing) | Implementation exercise for tokenisation |
| Practical BPE | [tiktoken](https://github.com/openai/tiktoken) | Inspect and use production-grade BPE encodings |
| BPE intuition | [BPE explainer](https://ruby-scowl-263.notion.site/Breaking-Down-Text-How-BPE-Tokenization-Works-19bb5dd44d5a806b95defbe96ac464e0) | Step-by-step merge-process explanation |
| Embeddings | [word2vec vectors](https://huggingface.co/fse/word2vec-google-news-300) | Historical baseline for semantic vector intuition |
| Pre-transformer attention | [Seq2seq attention visualisation](https://jalammar.github.io/visualizing-neural-machine-translation-mechanics-of-seq2seq-models-with-attention/) | Why attention solved the encoder bottleneck |
| Reasoning/RL | [Stanford CS336 lectures](https://www.youtube.com/playlist?list=PLoROMvodv4rOCXd21gf0CF4xr35yINeOy) | Background for language modelling and RL concepts |
| KV caching | [Video](https://www.youtube.com/watch?v=CxRGWfcGVbs) | Understand cached key/value tensors during decoding |
| GQA | [Video](https://www.youtube.com/watch?v=mtsY7JsGQjw) | KV-cache and head-sharing trade-offs |
| PagedAttention | [Video](https://www.youtube.com/watch?v=-AB6m0Spo6c) | Block-based cache management and continuous batching |
| FlashAttention | [Decoding Flash Attention](https://outcomeschool.com/blog/decoding-flash-attention) | Tiling, online softmax, and memory efficiency |
| Quantisation | [My LLMOps notes](https://github.com/upeshjeengar/LLMOps/tree/main/Quantization) | Practical deployment-oriented quantisation guide |
| Distillation | [Video](https://www.youtube.com/watch?v=jrJKRYAdh7I) | Teacher-student compression intuition |
| MoE | [Video](https://www.youtube.com/watch?v=v7U21meXd6Y) | Sparse routing and expert networks |
| Transformers in vision | [ViT overview](https://viso.ai/deep-learning/vision-transformer-vit/) | Patch-based image transformers |
| OS memory analogy | [Paging primer](https://www.geeksforgeeks.org/operating-systems/paging-in-operating-system/) | Background for understanding PagedAttention |
| Agentic application | [Enterprise AI Automation](https://github.com/upeshjeengar/enterprise-ai-automation) | My policy-aware multi-agent project |
| Multimodality | [My Kimi K2.5 article](https://upesh-jeengar.medium.com/seeing-thinking-acting-inside-kimi-k2-5s-architecture-47d9ae239671) | Applied walkthrough of a multimodal, agentic model |
| Multimodal research | [Kimi K2.5 paper](https://arxiv.org/abs/2602.02276) | Primary technical source |

## A practical project sequence

If you want to turn this reading list into portfolio-quality skill, build in this order:

1. **Tiny tokenizer and language model.** Train a very small decoder-only model on a public-domain corpus. Visualise input-target windows and test the causal mask.
2. **Transformer from scratch.** Implement Q/K/V projections, scaled dot-product attention, multi-head attention, residual connections, layer normalisation, and a feed-forward network. Unit-test shapes and masking.
3. **Domain adaptation experiment.** Compare prompting, continued pre-training, and a LoRA fine-tune for a narrow domain. Keep a held-out evaluation set before training.
4. **RAG system with an evaluation harness.** Build ingestion, chunking, hybrid retrieval, reranking, citations, and a set of answerable/unanswerable questions. Measure retrieval recall separately from final-answer quality.
5. **Tool-using assistant.** Add only a few deterministic tools first. Validate schemas, restrict permissions, require approvals for consequential actions, and log every tool call.
6. **Inference benchmark.** Measure time-to-first-token, tokens per second, peak memory, and quality for different context lengths, quantisation choices, and concurrent loads. Explain the changes using KV caching and batching.
7. **Agent workflow.** Build a bounded workflow with planning, retrieval, tools, stop conditions, and recovery paths. Add multiple agents only if routing creates a measurable improvement.

The finished project should answer more than "does it work?" It should answer "what fails, how do I measure that failure, and what trade-off did I choose?"

## Contributing

These notes are intentionally open for correction and expansion. If you spot an error, a stale link, an unclear explanation, or a missing practical resource, please open an issue or submit a pull request with:

- the concept or section being improved;
- the reason for the change;
- a primary source where possible; and
- a short explanation that helps a learner understand the trade-off, not just the definition.

External resources belong to their respective authors and retain their own licenses. Please use datasets, model weights, APIs, and software in accordance with their terms and applicable law.

# LLMOps
This repo has the Complete guide for deploying a large language model.

## Load Testing(optional)
This is just a work of my curiosity that what would be difference between theoretical and actual concurrency we can achieve, the content of my experiment is in `Load_test/` folder,I am Load testing an 15GB GPU for how many concurrent request it can handle, see Load_test folder for the detailed theoretical calculation and experiments result. The LLM used is [Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B)


## Model Performance
Before deploying an LLM, we must need to look at model accuracy:
1. **Robustness:** A robust model adeptly handles unexpected or noisy data, maintaining performance in diverse conditions, including adversarial examples and edge cases.
2. **Generalisability**: The model should perform well on new, unseen data, avoiding overfitting to training data and adapting to varied situations.
3. **Fairness and Bias**: Ensure the model does not perpetuate biases, and remains equitable across different demographic groups.
4. **Interpretability and Explainability**: The ability to clearly trace and articulate the decision-making process and its operational patterns.
5. **Compliance and Ethical considerations**: Focus on adhering to legal standards in data privacy, security and usage rights.

### Speed and performance
In deploying models for real time applications, inference speed is crucial. So along with the accuracy of the model we also need to maintain speed of results.
Latency(RTT for the first token to appear(if we are using continuous stream of tokens)) and Throughput(ML model's capactiy to process units of inference over time).

- Scalability: Ensuring model performs well as data volume expands(higher context window)
- Cost Effectiveness: Balancing the economic aspect of model training, deployment and upkeep.
- Deployment readiness: Focusing on integration ease into existing environments.   
Cost-Quality-Latency triangle is the essential framework for every inference decision. 

### Model optimization techniques
1. Model Distillation: Employing a faster, smaller model trained by larger model balancing speed & accuracy throught knowledge transfer and optimizations.
2. Model Quantization: 32 bit to 16 bit or optimization like NF16, NF8(which preserves more accuracy than simple bit cutting)
3. KV Caching:  
**Problem**: Without caching, as an LLM generates tokens, it recomputes attention for all previous tokens in every step (e.g., "The," "The quick," "The quick brown"), causing a bottleneck.  
**Solution**: The model computes and vectors for the new token and fetches the stored and vectors for all past tokens from GPU memory.  
**Result**: Only the latest token's attention needs calculation, significantly increasing speed. 
4. Grouped Query attention: Grouped-query attention is an attention variant derived from standard multi-head attention. Instead of giving every query head its own keys and values, it lets several query heads share the same key-value projections, which makes KV caching much cheaper without changing the overall decoder recipe very much.  
It optimizes KV-cache size, memory bandwidth, and long-context inference cost
5. Paged attention: an efficient memory management algorithm for Large Language Model (LLM) serving that reduces GPU memory waste by partitioning the Key-Value (KV) cache into fixed-size blocks.  
6. FlashAttention is a high-speed, memory-efficient algorithm,  
   Instead of:  
   compute full QK^T → store → softmax → multiply V  
   FlashAttention does something like:  
   - take block of Q  
   - take block of K  
   - compute partial attention 
   - update result  
   - discard temporary matrix 
   - move to next block  
   ![](https://towardsdatascience.com/wp-content/uploads/2025/01/03odj61pQsdUpvVsK.png)
   So at any moment, the GPU only holds a small tile of the attention matrix.
7. Mixture of experts(Only a subset of parameters active per token)
8. Prefix caching: Caching Q,k,v for initial system prompt or instruction.


# Parallelism: Scaling model to multiple CPU/GPUs
We can scale our model to multiple devices, if it is bigger than whole VRAM or just for sack of parallel processing we want to scale our Model, then we can do so by following methods:
## 1. Data Parallelism
Same setup of model is replicated on multiple devices and each device is fed with a slice of data, processing happens parallely and all setups are synchronized later

## 2.ZeRO(Zero Reduandancy optimization) Data Parallel
ZeRO reduces the memory consumption of each GPU by partitioning the various model training states (weights, gradients, and optimizer states) across the available devices (GPUs and CPUs) in the distributed training hardware, main difference between DP and ZeRO DP is that DP stores full model while ZeRO stores small parts of models on various GPU

## 3. TensorParallel
Tensor Parallelism partitions tensors (typically weights of a layer) across multiple GPUs, allowing each GPU to compute a portion of the same operation in parallel. Intermediate results are combined using communication primitives like all-reduce or all-gather to produce the final output.


# Model Management & ML-Ops

Below is a guide and Bash commands to set up MLflow on an Ubuntu system. This setup includes installing MLflow, setting up a backend store for experiments and runs, and launching the MLflow UI.

### MLflow Setup Guide for Ubuntu

#### Prerequisites:
- Python 3.6 or higher
- Pip (Python package installer)
- Ubuntu system (or a similar Linux distribution)




### Step 1: Create custom user (optional)

Setting up a custom user for MLflow and a dedicated Python environment is a good practice, especially for ensuring that the MLflow service runs securely and isolated from other system processes. Here's how you can set it up on an Ubuntu system:

1. **Create the User**:
   Open a terminal and run the following command to create a new user called `mlflow`.
   ```bash
   sudo adduser mlflow
   ```

2. **Grant Sudo Privileges (Optional)**:
   If this user needs to perform administrative tasks, you can grant it sudo privileges. Otherwise, you can skip this step.
   ```bash
   sudo usermod -aG sudo mlflow
   ```

### Step 2: Install Python and Create an Environment

1. **Switch to the mlflow User**:
   Switch to the new user account.
   ```bash
   su - mlflow
   ```

2. **Install Python3 and Pip**:
   Ensure Python3 and Pip are installed. Most Ubuntu versions come with Python3 by default, but you might need to install pip.
   ```bash
   sudo apt update
   sudo apt install python3 python3-pip
   ```

3. **Install Virtualenv**:
   Virtualenv is a tool to create isolated Python environments.
   ```bash
   pip3 install virtualenv
   ```

4. **Create a Virtual Environment**:
   Create a new directory for the MLflow server and navigate into it. Then create a virtual environment.
   ```bash
   mkdir ~/mlflow_server
   cd ~/mlflow_server
   virtualenv mlflow_env
   ```

5. **Activate the Virtual Environment**:
   Before installing MLflow and other dependencies, activate the virtual environment.
   ```bash
   source mlflow_env/bin/activate
   ```

### Step 3: Install MLflow
With the virtual environment activated, install MLflow. If you want to ensure compatibility, use the same versions I use in the course. If you'd like to use the latest, make sure to use matching versions for the other libraries.

```bash
pip install mlflow==2.7.1
```


### Step 4: Install Backend Store (Optional)
MLflow uses a tracking server to log experiment data. By default, it logs to the local filesystem, but for more robust use, you may want to set up a database like MySQL or SQLite.

**For SQLite (Simpler Option):**
- SQLite comes pre-installed on many systems, including Ubuntu.
- Decide on a directory where you want your SQLite database to reside
```bash
cd ~/mlflow_server
mkdir metrics_store
```

**For MySQL:**
- Install MySQL Server:
  ```bash
  sudo apt update
  sudo apt install mysql-server
  ```
- Secure your installation and set up your user (follow the prompt after the command):
  ```bash
  sudo mysql_secure_installation
  ```
- Log into MySQL to create a database for MLflow:
  ```bash
  sudo mysql -u root -p
  ```
- Once inside MySQL, create a database:
  ```mysql
  CREATE DATABASE mlflow_db;
  EXIT;
  ```

### Step 5: Set Backend Store for MLflow
- **For SQLite**, you'll use a URI like: `sqlite:////home/mlflow/mlflow_server/metrics_store/mlflow.db`
- **For MySQL**, the URI will be: `mysql://<username>:<password>@localhost/mlflow_db`


### Step 6: Install Artifact Store
The artifact store is where MLflow saves model artifacts like models and plots. You can use S3, Azure Blob Storage, Google Cloud Storage, or even a shared filesystem.

- **For local storage (simplest for getting started)**, use a local directory.
```bash
cd ~/mlflow_server
mkdir artifact_store
```




### Step 7: Launch MLflow Tracking Server
Open a terminal and run the following command, replacing the URIs with your chosen backend and artifact store paths:

```bash
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlflow-artifacts
```

Replace `sqlite:///mlflow.db` with your MySQL URI if you're using MySQL, and adjust `./mlflow-artifacts` to the path where you want artifacts stored.

### Step 8: Accessing the MLflow UI
- Once the tracking server is running, it will display a URL, typically `http://127.0.0.1:5000`. Open this URL in a web browser to access the MLflow UI.
- You can now navigate the UI to see your experiments, runs, metrics, and artifacts.
```bash
ssh -L 5000:localhost:5000 remote
```


#### Additional Tips:
- **Service**: For a more permanent setup, you might want to set up MLflow to run as a service or use a process manager like `supervisor` to manage the server process.
- **Security**: If you're setting this up on a cloud server or an exposed machine, ensure you configure proper security settings, including firewalls and authentication for the MLflow server.

### Conclusion
You now have MLflow set up on your Ubuntu system with a backend store for tracking experiments and an artifact store for saving model artifacts. You can start running experiments and tracking them using the MLflow Python library, and all your experiment details will be accessible through the MLflow UI.

# Advanced model deployment techniques
## Batching and Dynamic batches 
Batching boost efficiency and speed by parallel processing requests on GPUs 
Instead of single request -> a Batch of request that will do there work concurrently

Dynamic Batching(adaptive batching): Technique which adjusts the batch size of data based on current conditions and requirements.

Challenges: Finding optimal batch size(comes throught experimentation, around theoretical estimated value), Varying input sizes of each request in the batch, balancing latency and throghput.

