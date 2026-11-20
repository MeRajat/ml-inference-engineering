# Schedule

Twenty weeks at 10–12 hours a week. Tick items as you go. Each lab links to
its brief; write the report from [`labs/REPORT_TEMPLATE.md`](labs/REPORT_TEMPLATE.md).

If you have 20 or more hours a week, the plan compresses to about twelve weeks.
Do not compress by skipping labs.

## Before week 1

- [ ] [Module 0](modules/00-orientation.md): environment, GPU access, baseline check

## Module 1 — Fundamentals

**Week 1**
- [ ] Read: Transformer Inference Arithmetic; Making Deep Learning Go Brrrr; scaling-book rooflines and transformer math
- [ ] Lab 1.1: KV-cache and roofline audit

**Week 2**
- [ ] Read: scaling-book inference and GPUs; Efficiently Scaling Transformer Inference; GPT, Fast
- [ ] Lab 1.2: the gpt-fast ladder
- [ ] Lab 1.3: latency–throughput curve
- [ ] [Checkpoint](modules/01-inference-fundamentals.md#checkpoint)

## Module 1B — Transformer internals and architectural optimizations

**Week 3**
- [ ] Watch: CS336 Lecture 3 (Architectures); LLaMA explained
- [ ] Read: the Big LLM Architecture Comparison; rotary embeddings; multi-query and grouped-query attention papers
- [ ] Code: nanoGPT, nanochat and the Llama reference `model.py`
- [ ] Lab 1B.1: attention-head structure against KV cache
- [ ] Lab 1B.3: nanoGPT to a Llama-style block (start)

**Week 4**
- [ ] Read: visual guide to attention variants; Mixtral and the DeepSeek-V3 architecture section; one audio architecture (Whisper, Moshi, CSM or MusicGen)
- [ ] Lab 1B.3: finish
- [ ] Lab 1B.2: positional scheme against extrapolation and cache reuse
- [ ] Lab 1B.4: mixture-of-experts routing cost
- [ ] [Checkpoint](modules/01b-transformer-architecture.md#checkpoint)

## Module 2 — Benchmarking and observability

**Week 5**
- [ ] Read: Open Versus Closed; coordinated omission; goodput
- [ ] Lab 2.1: coordinated omission in your own data
- [ ] Lab 2.2: goodput curve and tool cross-check
- [ ] Lab 2.3: observability stack (keep it running for the rest of the course)
- [ ] [Checkpoint](modules/02-benchmarking-and-observability.md#checkpoint)

## Module 3 — Serving engines

**Week 6**
- [ ] Read: continuous batching post; Orca; PagedAttention; Inside vLLM
- [ ] Code: nano-vllm end to end
- [ ] Lab 3.1: scheduler tuning sweep

**Week 7**
- [ ] Read: Sarathi-Serve; SGLang; vLLM design docs
- [ ] Code: vLLM scheduler and KV-cache manager; SGLang radix cache
- [ ] Lab 3.2: prefix cache comparison

**Week 8**
- [ ] Read: Triton Inference Server batching docs; TensorRT docs; speculative decoding papers
- [ ] Lab 3.3: Triton dynamic batching with TensorRT
- [ ] Lab 3.4: decoding features under load
- [ ] [Checkpoint](modules/03-serving-engines.md#checkpoint)

## Module 4 — Model optimization

**Week 9**
- [ ] Watch: MIT 6.5940 lectures 5, 6, 13
- [ ] Read: Lilian Weng's overview; GPTQ, AWQ, SmoothQuant; "Give Me BF16 or Give Me Death"
- [ ] Lab 4.1: quantization Pareto sweep

**Week 10**
- [ ] Read: TensorRT quantized types and best practices; ONNX Runtime optimizations
- [ ] Lab 4.2: compilation ladder
- [ ] Lab 4.3: speculative decoding and KV-cache compression

**Week 11**
- [ ] Watch: MIT 6.5940 lectures 9, 18
- [ ] Read: the paradox of diffusion distillation; Consistency Models; Distil-Whisper
- [ ] Lab 4.4: distillation for speech or diffusion
- [ ] [Checkpoint](modules/04-model-optimization.md#checkpoint)

## Module 5 — GPU programming and kernels

**Week 12**
- [ ] Read: PMPP opening chapters; CUDA best practices
- [ ] Watch: How CUDA Programming Works; GPU MODE L1, L4, L8
- [ ] Lab 5.1: matmul ladder in CUDA

**Week 13**
- [ ] Do: Triton tutorials and Triton-Puzzles
- [ ] Watch: GPU MODE L14, L16, L18
- [ ] Lab 5.2: a fused Triton kernel

**Week 14**
- [ ] Watch: GPU MODE L12; FlashAttention derivation
- [ ] Read: Nsight guides; CUDA graphs; custom operators
- [ ] Lab 5.3: FlashAttention forward pass and CUDA graphs
- [ ] [Checkpoint](modules/05-gpu-programming-and-kernels.md#checkpoint)

## Module 6 — Real-time and audio

**Week 15**
- [ ] Read: Voice AI primer; The Tail at Scale; Inference Engineering ASR and TTS sections
- [ ] Read: one model family in depth (Moshi, Kyutai streaming models, or streaming Whisper)
- [ ] Lab 6.1: streaming text-to-speech server and harness

**Week 16**
- [ ] Read: WebSocket and WebRTC chapters; the Modal voice-bot architecture
- [ ] Lab 6.2: streaming speech recognition
- [ ] Lab 6.3: voice agent latency budget
- [ ] [Checkpoint](modules/06-realtime-streaming-and-audio.md#checkpoint)

## Module 7 — Scale and operations

**Week 17**
- [ ] Read: GPU Operator docs; autoscaling docs and the GKE guide; cold-start material
- [ ] Lab 7.1: GPU sharing bake-off
- [ ] Lab 7.2: queue-depth autoscaling and cold start

**Week 18**
- [ ] Read: Gateway API Inference Extension; llm-d; canary rollouts; SRE chapters; inference economics
- [ ] Lab 7.3: cache-aware routing and safe rollout
- [ ] [Checkpoint](modules/07-production-scale-and-operations.md#checkpoint)

## Module 8 — Capstone

**Week 19**
- [ ] Choose [option A or B](capstone/README.md); write predictions; build the baseline
- [ ] Apply and measure optimizations

**Week 20**
- [ ] Scale, failure and cost measurements
- [ ] Publish the repository and write-up
- [ ] [Portfolio checklist](career/README.md#portfolio-checklist)

## Running alongside, from week 6

- [ ] Follow one open-source inference project's issue tracker and pull requests
- [ ] Open a first pull request by week 12 (documentation or a benchmark fix is fine)
- [ ] Open a performance-related pull request by week 18
