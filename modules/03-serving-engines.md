# Module 3 — Serving engines and LLM inference systems

**Weeks 4–6 · about 36 hours (14 reading, 22 labs)**

## Why this module

A serving engine is a scheduler, a memory manager and a model runner. Knowing
the flags is not enough for an inference role: you are expected to read the
scheduler, explain why it made a decision, and change it. This module takes you
through the ideas (continuous batching, paged KV cache, prefix caching, chunked
prefill, disaggregation) and then into the code that implements them.

"Triton" in this module means the NVIDIA Triton **Inference Server**. The Triton
kernel language is Module 5.

## Learning objectives

- Explain static, dynamic and continuous batching and when each applies
- Explain paged KV-cache management and prefix caching from the source code of
  vLLM and SGLang
- Explain chunked prefill, token budgets and why they bound tail latency
- Deploy the same model on two engines and account for the differences
- Configure Triton dynamic batching and instance groups for a non-LLM model
- Measure speculative decoding, structured output and multi-adapter serving
  under load and say when each is a net loss

## Core resources

### Start here

| # | Resource | Type | What to study | Hours |
|---|---|---|---|---|
| 1 | [Inference Engineering](https://www.baseten.co/inference-engineering/) (Philip Kiely, Baseten Books, 2026; free online, [contents](https://www.baseten.co/inference-engineering/llms.txt)) | Book | The Software chapter (inference engines, benchmarking) and the Techniques chapter (speculative decoding, caching, disaggregation). | 8 |
| 2 | [Inside vLLM: Anatomy of a High-Throughput LLM Inference System](https://vllm.ai/blog/2025-09-05-anatomy-of-vllm) (Aleksa Gordić) | Blog | The best single walkthrough of the engine. Read it with the code open. | 4 |
| 3 | [How continuous batching enables 23x throughput in LLM inference](https://www.anyscale.com/blog/continuous-batching-llm-inference) (Anyscale) | Blog | Static, dynamic and continuous batching with benchmarks. | 1 |
| 4 | [CMU 11-868 LLM Systems](https://llmsystem.github.io/llmsystem2026spring/docs/Syllabus/) | Course slides | The weeks on generation and speculative decoding, LLM serving with SGLang, PagedAttention and vLLM, and DistServe and KV-cache systems. Slides only. | 8 |

### Papers

| # | Resource | What to study | Hours |
|---|---|---|---|
| 5 | [Orca](https://www.usenix.org/conference/osdi22/presentation/yu) (OSDI '22) | Iteration-level scheduling: the origin of continuous batching. | 1.5 |
| 6 | [Efficient Memory Management for LLM Serving with PagedAttention](https://arxiv.org/abs/2309.06180) (SOSP '23) | Sections 3–4: fragmentation, block tables, copy-on-write. | 2 |
| 7 | [Sarathi-Serve](https://arxiv.org/abs/2403.02310) (OSDI '24) | Chunked prefill and stall-free batching. | 1.5 |
| 8 | [DistServe](https://arxiv.org/abs/2401.09670) (OSDI '24), with [Splitwise](https://arxiv.org/abs/2311.18677) | Prefill/decode disaggregation and goodput. | 2.5 |
| 9 | [SGLang](https://arxiv.org/abs/2312.07104) | RadixAttention and compressed finite-state-machine constrained decoding. | 1.5 |
| 10 | Speculative decoding: [Leviathan et al.](https://arxiv.org/abs/2211.17192) and [Chen et al.](https://arxiv.org/abs/2302.01318) | The lossless accept/reject rule and the speed-up analysis. | 1.5 |
| 11 | [Towards Efficient Generative LLM Serving](https://arxiv.org/abs/2312.15234) | Survey. Read the system-level sections as a map. | 2 |

### Code to read

| # | Repository | Read | Hours |
|---|---|---|---|
| 12 | [nano-vllm](https://github.com/GeeeekExplorer/nano-vllm) | Read this first: `nanovllm/engine/scheduler.py`, `block_manager.py`, `sequence.py`, `llm_engine.py`, `model_runner.py`. | 3 |
| 13 | [vLLM](https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/sched/scheduler.py) | `vllm/v1/core/sched/scheduler.py` (`schedule()` and the token budget), `vllm/v1/core/kv_cache_manager.py`, `vllm/v1/core/block_pool.py`, `vllm/v1/engine/core.py`. | 6 |
| 14 | [SGLang](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/mem_cache/radix_cache.py) and [mini-sglang](https://github.com/sgl-project/mini-sglang) | In mini-sglang: `python/minisgl/kvcache/radix_cache.py` and `engine/engine.py`. In SGLang: `srt/mem_cache/radix_cache.py`, `srt/managers/scheduler.py`, `srt/managers/schedule_policy.py`. | 5 |
| 15 | [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) and the [Triton backend](https://github.com/triton-inference-server/tensorrtllm_backend) | Engine build flow, in-flight batching, KV-cache reuse. | 4 |

File paths were checked against the default branches in November 2026. Projects
this active move files around; if a path is missing, search the repository for
the class name.

### Documentation

| # | Resource | What to study | Hours |
|---|---|---|---|
| 16 | vLLM: [architecture overview](https://docs.vllm.ai/en/latest/design/arch_overview/), [prefix caching](https://docs.vllm.ai/en/latest/design/prefix_caching/), [paged attention](https://docs.vllm.ai/en/latest/design/paged_attention/), [optimization and tuning](https://docs.vllm.ai/en/latest/configuration/optimization/) | The design pages. | 4 |
| 17 | [SGLang docs](https://docs.sglang.io/) | Server arguments. | 1 |
| 18 | Triton Inference Server: [architecture](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/architecture.html), [batchers](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/batcher.html), [model configuration](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/model_configuration.html), [conceptual guide part 2](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/tutorials/Conceptual_Guide/Part_2-improving_resource_utilization/README.html) | Dynamic batching, instance groups, queue delay, ensembles. | 5 |
| 19 | [TensorRT](https://docs.nvidia.com/deeplearning/tensorrt/latest/index.html) and [TensorRT-LLM](https://nvidia.github.io/TensorRT-LLM/) docs | Engine building, precision, optimization profiles. | 3 |

### Talks

| # | Talk | Hours |
|---|---|---|
| 20 | [Fast LLM Serving with vLLM and PagedAttention](https://www.youtube.com/watch?v=5ZlavKF_98U) (Ray Summit 2023, the original authors) | 0.5 |
| 21 | [Accelerating LLM Inference with vLLM (and SGLang)](https://www.youtube.com/watch?v=PcvxdWJOyUE) (Ion Stoica) | 1 |
| 22 | vLLM office hours ([index](https://vllm.ai/office-hours), [playlist](https://www.youtube.com/playlist?list=PLbMP1JcGBmSHxp4-lubU5WYmJ9YgAQcf3)): [Intro to vLLM V1](https://www.youtube.com/watch?v=jmzIvQZCLZM), [Advanced techniques for maximizing performance](https://www.youtube.com/watch?v=vJA1BGkAdNo), [Disaggregated prefill and KV cache storage](https://www.youtube.com/watch?v=FPr37jCOvrA) | 3 |

The reading budget covers resources 1–3, 5–7, 12, 13, 16 and 20–21. The rest
supports the labs you choose.

## Deep-dive resources

| Topic | Resources |
|---|---|
| vLLM design rationale | [V1 architecture](https://vllm.ai/blog/2025-01-27-v1-alpha-release), [structured decoding](https://vllm.ai/blog/2025-01-14-struct-decode-intro), [speculative decoding](https://vllm.ai/blog/2024-10-17-spec-decode), [original launch post](https://vllm.ai/blog/2023-06-20-vllm) |
| vLLM features | [speculative decoding](https://docs.vllm.ai/en/latest/features/speculative_decoding/), [structured outputs](https://docs.vllm.ai/en/latest/features/structured_outputs/), [LoRA adapters](https://docs.vllm.ai/en/latest/features/lora/), [disaggregated prefill](https://docs.vllm.ai/en/latest/features/disagg_prefill/), [benchmark CLI](https://docs.vllm.ai/en/latest/benchmarking/cli/) |
| More office hours | [Speculative decoding](https://www.youtube.com/watch?v=eVJBFajJRIU), [Structured outputs](https://www.youtube.com/watch?v=CGEkEwg0V9U), [Intelligent inference scheduling with llm-d](https://www.youtube.com/watch?v=gaoNMdsfOPg), [Production stack deep dive](https://www.youtube.com/watch?v=0ZVu0A4wWQg) |
| SGLang | [RadixAttention post](https://www.lmsys.org/blog/2024-01-17-sglang/), [compressed FSM for JSON decoding](https://www.lmsys.org/blog/2024-02-05-compressed-fsm/), [runtime comparison](https://www.lmsys.org/blog/2024-07-25-sglang-llama3/), [prefill/decode disaggregation](https://docs.sglang.io/docs/advanced_features/pd_disaggregation) |
| Scheduling and fairness | [FastServe](https://arxiv.org/abs/2305.05920), [Fairness in Serving LLMs](https://arxiv.org/abs/2401.00588), [Llumnix](https://arxiv.org/abs/2406.03243) |
| Pre-LLM serving classics | [Clipper](https://arxiv.org/abs/1612.03079), [InferLine](https://arxiv.org/abs/1812.01776), [AlpaServe](https://arxiv.org/abs/2302.11665) |
| Structured decoding | [Efficient Guided Generation](https://arxiv.org/abs/2307.09702), [XGrammar](https://arxiv.org/abs/2411.15100) |
| Multi-adapter serving | [S-LoRA](https://arxiv.org/abs/2311.03285) |
| KV-cache systems | [Mooncake](https://arxiv.org/abs/2407.00079), [CacheGen](https://arxiv.org/abs/2310.07240), [CacheBlend](https://arxiv.org/abs/2405.16444) |
| llama.cpp | [repository](https://github.com/ggml-org/llama.cpp), [server README](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md), [server internals](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README-dev.md), [GGUF specification](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md) |
| TensorRT-LLM | [disaggregated serving](https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/features/disagg-serving.md) |
| Triton | [tutorials](https://github.com/triton-inference-server/tutorials), [perf_analyzer](https://github.com/triton-inference-server/perf_analyzer) |
| Ray Serve | [serving LLMs](https://docs.ray.io/en/latest/serve/llm/index.html), [dynamic request batching](https://docs.ray.io/en/latest/serve/advanced-guides/dyn-req-batch.html), [source](https://github.com/ray-project/ray/tree/master/python/ray/llm), [disaggregated serving with vLLM](https://www.anyscale.com/blog/ray-serve-llm-anyscale-apis-wide-ep-disaggregated-serving-vllm) |
| Engine comparisons | [Modal LLM Engine Advisor](https://modal.com/llm-almanac/advisor), [BentoML backend benchmark](https://www.bentoml.com/blog/benchmarking-llm-inference-backends), [Hugging Face TGI benchmarking](https://huggingface.co/blog/tgi-benchmarking) |
| Surveys | [LLM Inference Serving: Survey of Recent Advances](https://arxiv.org/abs/2407.12391), [A Survey on Efficient Inference for LLMs](https://arxiv.org/abs/2404.14294) |

## Optional

- [LLM Inference Handbook](https://handbook.modular.com/) ([source](https://github.com/modular/llm-inference-handbook)): practitioner reference
- [LLMs in Production](https://www.manning.com/books/llms-in-production) (Brousseau and Sharp, Manning): broader deployment view
- [AI Engineering chapter summaries](https://github.com/chiphuyen/aie-book/blob/main/chapter-summaries.md) (Chip Huyen): the inference optimization chapter
- [Efficient Inference with SGLang](https://www.deeplearning.ai/courses/efficient-inference-with-sglang-text-and-image-generation) (DeepLearning.AI short course)
- [NVIDIA Dynamo introduction](https://developer.nvidia.com/blog/introducing-nvidia-dynamo-a-low-latency-distributed-inference-framework-for-scaling-reasoning-ai-models/), [A guide to LLM inference and performance](https://www.baseten.co/blog/llm-transformer-inference-guide/) (Baseten), [Assisted generation](https://huggingface.co/blog/assisted-generation) (Hugging Face), [Together Inference Engine 2.0](https://www.together.ai/blog/together-inference-engine-2)
- [Text Generation Inference router](https://github.com/huggingface/text-generation-inference/tree/main/router): a clean Rust reference design. The project is in maintenance mode, so read it as history.
- [KServe](https://kserve.github.io/website/) ([generative runtime](https://kserve.github.io/website/docs/model-serving/generative-inference/overview)), [LitServe](https://github.com/Lightning-AI/LitServe) ([docs](https://lightning.ai/docs/litserve)), [BentoML](https://docs.bentoml.com/en/latest/)

## Labs

### Lab 3.1 — Continuous batching and scheduler tuning

**Tools:** vLLM with one 7–8B model on a single GPU; GuideLLM or `vllm bench serve`.

Drive at five or more request rates with a mixed prompt-length workload. Vary
`--max-num-seqs`, `--max-num-batched-tokens` and prefix caching.

**Deliverable:** throughput against p50 and p99 TTFT and ITL, the maximum
goodput at a stated target (for example p99 TTFT under 500 ms, TPOT under
50 ms), and a one-page explanation tied to specific lines of `scheduler.py`.

### Lab 3.2 — Prefix cache comparison: vLLM and SGLang

**Tools:** both engines, identical load.

Build a workload with a shared system prompt, few-shot examples and multi-turn
chats at 0%, 50% and 90% prefix sharing.

**Deliverable:** cache-hit rate (from each engine's Prometheus metrics), TTFT
and tokens per second per engine and sharing level. Stretch: add LRU eviction or
a prefix-hit counter to nano-vllm's `block_manager.py` and show the change.

### Lab 3.3 — Triton dynamic batching with a TensorRT engine

**Tools:** an encoder (BERT or a ViT) exported to ONNX, a TensorRT FP16 engine,
Triton, perf_analyzer.

Sweep `max_queue_delay_microseconds`, `preferred_batch_size` and instance-group count.

**Deliverable:** a latency–throughput Pareto plot against a PyTorch baseline
behind LitServe or Ray Serve `@serve.batch`, and your chosen configuration with reasons.

### Lab 3.4 — Decoding features under load

**Tools:** vLLM.

Measure (a) speculative decoding speed-up and acceptance rate at 1, 8 and 32
requests per second; (b) JSON-schema constrained decoding overhead per token;
(c) 1, 8 and 32 concurrent LoRA adapters against merged weights.

**Deliverable:** tokens per second and p99 latency for each, and the load at
which speculative decoding turns net-negative, with the reason.

### Stretch — Disaggregated prefill and decode

On two GPUs, compare colocated serving with SGLang's prefill/decode
disaggregation on a long-prompt short-output workload and the reverse.

**Deliverable:** target attainment against request rate, with KV transfer
overhead quantified.

## Checkpoint

1. A long prompt arrives while 30 sequences are decoding. What does the scheduler do, and what happens to their inter-token latency?
2. How does a paged KV cache make prefix sharing cheap?
3. When does speculative decoding slow a busy server down?
