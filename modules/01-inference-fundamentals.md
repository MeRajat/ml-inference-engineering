# Module 1 — Inference fundamentals and performance mental models

**Weeks 1–2 · about 22 hours (10 reading, 12 labs)**

## Why this module

Every later decision (which engine, which quantization, which GPU, what batch
size) depends on being able to predict performance from first principles. By
the end of this module you can look at a model and a GPU and say whether a
workload is limited by compute, by memory bandwidth or by overhead, and estimate
tokens per second before you run it.

## Learning objectives

- Derive KV-cache size, prefill cost and per-token decode cost for a transformer
- Explain why decode is memory-bandwidth-bound and prefill is compute-bound
- Use the roofline model to place an operator on a given GPU
- Define time to first token (TTFT), time per output token (TPOT), inter-token
  latency (ITL) and throughput, and explain how batching trades between them
- Read a PyTorch profiler trace and find graph breaks and recompiles in `torch.compile`

## Core resources

| # | Resource | Type | What to study | Hours |
|---|---|---|---|---|
| 1 | [How To Scale Your Model](https://jax-ml.github.io/scaling-book/) (Google DeepMind) | Online book | [Rooflines](https://jax-ml.github.io/scaling-book/roofline/), [Transformer math](https://jax-ml.github.io/scaling-book/transformers/), [Inference](https://jax-ml.github.io/scaling-book/inference/), [GPUs](https://jax-ml.github.io/scaling-book/gpus/). Do the worked problems. | 12 |
| 2 | [Transformer Inference Arithmetic](https://kipp.ly/transformer-inference-arithmetic/) (kipply) | Blog | All of it. Redo the numbers for a current GPU. | 1.5 |
| 3 | [Making Deep Learning Go Brrrr From First Principles](https://horace.io/brrr_intro.html) (Horace He) | Blog | All of it: compute, memory bandwidth, overhead. | 1 |
| 4 | [Stanford CS229S: Systems for Machine Learning](https://cs229s.stanford.edu/fall2023/calendar/) | Course slides | Oct 4 hardware-aware algorithm design, Oct 9 analysing transformer performance, Oct 11 FlashAttention. | 5 |
| 5 | [MIT 6.5940 TinyML and Efficient Deep Learning Computing](https://hanlab.mit.edu/courses/2024-fall-65940) ([videos](https://www.youtube.com/playlist?list=PL80kAHvQbh-qGtNc54A6KW4i4bkTPjiRF)) | Course | Lectures 12 (Transformer and LLM) and 13 (Efficient LLM Deployment). | 3 |
| 6 | [Efficiently Scaling Transformer Inference](https://arxiv.org/abs/2211.05102) (Pope et al.) | Paper | Sections 2–3: latency against throughput, prefill against decode. | 2 |
| 7 | [LLM Inference Unveiled: Survey and Roofline Model Insights](https://arxiv.org/abs/2402.16363) with [LLM-Viewer](https://github.com/hahnyuan/LLM-Viewer) | Paper + tool | Sections 2–3, then run the tool on your lab model. | 3 |
| 8 | [FlashAttention](https://arxiv.org/abs/2205.14135) and [FlashAttention-2](https://arxiv.org/abs/2307.08691) | Papers | v1 section 3 (tiling and online softmax); v2 work partitioning. | 3 |
| 9 | [Coding the KV Cache in LLMs from Scratch](https://magazine.sebastianraschka.com/p/coding-the-kv-cache-in-llms) (Sebastian Raschka) and [Hugging Face cache strategies](https://huggingface.co/docs/transformers/kv_cache) | Blog + docs | Code along; then read the static and offloaded cache variants. | 2 |
| 10 | [Accelerating Generative AI with PyTorch II: GPT, Fast](https://pytorch.org/blog/accelerating-generative-ai-2/) and [gpt-fast](https://github.com/meta-pytorch/gpt-fast) | Blog + repo | Read the post, then `generate.py` and `model.py`. | 3 |
| 11 | [torch.compile tutorial](https://docs.pytorch.org/tutorials/intermediate/torch_compile_tutorial.html) and [Ways to use torch.compile](https://blog.ezyang.com/2024/11/ways-to-use-torch-compile/) (Edward Yang) | Docs + blog | Guards, graph breaks, when compilation helps inference. | 2 |
| 12 | [PyTorch profiler recipe](https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html) and [performance tuning guide](https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html) | Docs | Operator-level CPU and CUDA time, trace export, the inference checklist. | 2 |
| 13 | Metric definitions: [Databricks](https://www.databricks.com/blog/llm-inference-performance-engineering-best-practices), [Anyscale](https://www.anyscale.com/blog/reproducible-performance-metrics-for-llm-inference), [NVIDIA NIM](https://docs.nvidia.com/nim/benchmarking/llm/metrics) | Blogs + docs | TTFT, TPOT, ITL and throughput under concurrency. | 2 |
| 14 | GPU architecture: [What every developer should know about GPU computing](https://blog.codingconfessions.com/p/gpu-computing) and the [Modal GPU glossary](https://modal.com/gpu-glossary) | Blog + reference | Streaming multiprocessors, warps, memory hierarchy. Keep the glossary open as a reference. | 2 |

That table is more than the ten reading hours in the budget. Resources 1–3, 6, 9,
10 and 13 are the minimum; fit the rest in as the labs raise questions.

## Deep-dive resources

| Resource | Type | Notes |
|---|---|---|
| [Efficient Memory Management for LLM Serving with PagedAttention](https://arxiv.org/abs/2309.06180) | Paper | Sections 2–4. The bridge to Module 3. |
| [FlashAttention-3](https://arxiv.org/abs/2407.08608) | Paper | Hopper-specific asynchrony and FP8. |
| [Roofline: An Insightful Visual Performance Model](https://www2.eecs.berkeley.edu/Pubs/TechRpts/2008/EECS-2008-134.html) | Paper | The original roofline report. Short. |
| [PyTorch internals](https://blog.ezyang.com/2019/05/pytorch-internals/) (Edward Yang) | Blog | Dispatcher, strides, storage. |
| [Dynamo deep dive](https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/torch.compiler_dynamo_deepdive.html) and [TorchInductor design note](https://dev-discuss.pytorch.org/t/torchinductor-a-pytorch-native-compiler-with-define-by-run-ir-and-symbolic-shapes/747) | Docs | How `torch.compile` traces and generates code. |
| [MIT 6.172 Performance Engineering of Software Systems](https://ocw.mit.edu/courses/6-172-performance-engineering-of-software-systems-fall-2018/) ([videos](https://www.youtube.com/playlist?list=PLUl4u3cNGP63VIBQVWguXxZZi0566y7Wf)) | Course | Lectures 1, 2, 10 and 14. |
| [Mastering LLM Techniques: Inference Optimization](https://developer.nvidia.com/blog/mastering-llm-techniques-inference-optimization/) (NVIDIA) | Blog | Overview of batching and model-level techniques. |
| [The KV Cache: Memory Usage in Transformers](https://www.youtube.com/watch?v=80bIUggRJf4) | Video | Visual walkthrough. |
| [How GPU Computing Works](https://www.youtube.com/watch?v=3l10o0DYJXg) (GTC 2021) | Video | Third-party upload of the NVIDIA talk. |

## Optional

- [Performance Analysis and Tuning on Modern CPUs](https://github.com/dendibakh/perf-book) (Denis Bakhvalov, free book): measurement methodology
- [Systems Performance, 2nd edition](https://www.brendangregg.com/systems-performance-2nd-edition-book.html) (Brendan Gregg, Addison-Wesley): methodologies and the USE method
- [Algorithms for Modern Hardware](https://en.algorithmica.org/hpc/) (Sergey Slotin, free book): memory and cache chapters
- [AI Engineering](https://github.com/chiphuyen/aie-book) (Chip Huyen, O'Reilly, 2025): the inference optimization chapter
- [Large Transformer Model Inference Optimization](https://lilianweng.github.io/posts/2023-01-10-inference-optimization/) (Lilian Weng): vocabulary map
- [Fast LLM Inference From Scratch](https://andrewkchan.dev/posts/yalm.html): a C++/CUDA engine build log

## Labs

### Lab 1.1 — KV-cache and roofline audit

**Tools:** PyTorch, a small open model (GPT-2 or a 1–3B model), `torch.profiler`, LLM-Viewer.

Implement greedy decoding with and without a KV cache. Measure TTFT, TPOT and
peak memory across prompt lengths from 128 to 8k tokens and batch sizes from 1 to 64.

**Deliverable:** plots plus a one-page table of predicted against measured tokens
per second. Aim for predictions within about 30%, and label each regime as
memory-bound, compute-bound or overhead-bound.

### Lab 1.2 — The gpt-fast ladder

**Tools:** gpt-fast or your own loop, `torch.compile(mode="reduce-overhead")`,
a static KV cache, the scaled-dot-product-attention fast path.

Apply the optimizations one at a time.

**Deliverable:** a waterfall chart of tokens per second after each step, a
profiler trace before and after, and graph breaks and recompiles driven to zero
(`TORCH_LOGS=recompiles,graph_breaks`).

### Lab 1.3 — Latency–throughput curve

**Tools:** a simple `asyncio` load generator against your own batched decode
loop, first with static batching, then continuous batching.

**Deliverable:** throughput against p50 and p99 TTFT and ITL for concurrency 1
to 128. Mark the knee and explain it.

## Checkpoint

You are ready for Module 2 when you can answer these without notes:

1. How many bytes of KV cache does one token cost for your lab model, and why?
2. Why does doubling batch size nearly double decode throughput up to a point, and what sets that point?
3. Your profiler shows 40% of time outside CUDA kernels. What are three likely causes?
