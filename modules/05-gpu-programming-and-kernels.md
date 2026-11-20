# Module 5 — GPU programming and custom kernels

**Weeks 12–14 · about 36 hours (14 reading and lectures, 22 labs)**

## Why this module

This is the skill that separates an inference specialist from a platform
engineer who deploys models. Serving frameworks get you most of the way; the
last factor of two usually comes from understanding what the GPU is doing and
writing or fixing the kernel yourself.

"Triton" in this module means the Triton **language**, a Python DSL for GPU
kernels. The NVIDIA Triton Inference Server is a different project, covered in
Module 3.

## Learning objectives

- Explain the CUDA execution model: threads, blocks, warps, occupancy, coalescing
- Write, verify and benchmark kernels in both CUDA C++ and Triton
- Profile with Nsight Systems (timeline) and Nsight Compute (per-kernel metrics)
- Implement tiled attention with online softmax
- Remove launch overhead with CUDA graphs
- Register a custom operator so that it works under `torch.compile`

## Core resources

| # | Resource | Type | What to study | Hours |
|---|---|---|---|---|
| 1 | [Programming Massively Parallel Processors, 4th edition](https://shop.elsevier.com/books/programming-massively-parallel-processors/hwu/978-0-323-91231-0) (Hwu, Kirk, El Hajj; Morgan Kaufmann) with the [co-author's lecture playlist](https://www.youtube.com/playlist?list=PLRRuQYjFhpmubuwx-w8X964ofVkW1T8O4) | Book + videos | The opening chapters on the execution model, memory and performance, then convolution, reduction and prefix sum. Watch the lectures at speed if you prefer video. | 25 (spread over the module) |
| 2 | [GPU MODE lectures](https://github.com/gpu-mode/lectures) ([channel](https://www.youtube.com/@GPUMODE)) | Lecture series | Start with [Lecture 1: profiling and integrating CUDA kernels in PyTorch](https://www.youtube.com/watch?v=LuhJEEJQgUM). Then L4 (compute and memory architecture), L8 (CUDA performance checklist), L12 (Flash Attention), L14 (practitioner's guide to Triton), L16 (hands-on profiling), L18 (fused kernels). | 12 |
| 3 | [Triton tutorials](https://triton-lang.org/main/getting-started/tutorials/index.html) and [repository](https://github.com/triton-lang/triton) | Docs + repo | Work through in order: vector add, fused softmax, matmul, layer norm, [fused attention](https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html). | 8 |
| 4 | [Triton-Puzzles](https://github.com/gpu-mode/Triton-Puzzles) and [GPU-Puzzles](https://github.com/srush/GPU-Puzzles) (Sasha Rush) | Exercises | Do all of them. | 6 |
| 5 | [How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance](https://siboehm.com/articles/22/CUDA-MMM) ([code](https://github.com/siboehm/SGEMM_CUDA)) (Simon Boehm) | Blog + repo | The best worked example of iterative kernel optimization. Read the kernels in order. | 4 |
| 6 | [CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/) and [An Even Easier Introduction to CUDA](https://developer.nvidia.com/blog/even-easier-introduction-cuda/) | Docs | Memory optimizations, execution configuration, performance metrics. | 4 |
| 7 | [How CUDA Programming Works](https://www.youtube.com/watch?v=QQceTDjA4f4) (Stephen Jones, GTC 2022) | Video | Why CUDA is shaped the way it is. Third-party upload. | 1 |
| 8 | Stanford CS149: [GPU architecture and CUDA programming](https://gfxcourses.stanford.edu/cs149/fall25/lecture/gpuarch/) and [Efficiently evaluating DNNs on GPUs](https://gfxcourses.stanford.edu/cs149/fall25/lecture/dnninference/) | Lectures | Both lectures. [2023 videos](https://www.youtube.com/playlist?list=PLoROMvodv4rMp7MTFr4hQsDEcX7Bx6Odp). | 4 |
| 9 | FlashAttention in code: [Umar Jamil's derivation and Triton walkthrough](https://www.youtube.com/watch?v=zy8ChVd_oTM), then [flash-attention-minimal](https://github.com/tspeterkim/flash-attention-minimal), then the [production repository](https://github.com/Dao-AILab/flash-attention) | Video + repos | In that order. | 10 |
| 10 | Profilers: [Nsight Systems user guide](https://docs.nvidia.com/nsight-systems/UserGuide/index.html), [Nsight Compute profiling guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html), [torch.profiler](https://docs.pytorch.org/docs/stable/profiler.html), [py-spy](https://github.com/benfred/py-spy) | Docs | Timeline and CPU–GPU gaps in Nsight Systems; per-kernel metrics and roofline in Nsight Compute; host-side stalls in py-spy. | 6 |
| 11 | CUDA graphs: [NVIDIA introduction](https://developer.nvidia.com/blog/cuda-graphs/), [PyTorch post](https://pytorch.org/blog/accelerating-pytorch-with-cuda-graphs/), [CUDA semantics notes](https://docs.pytorch.org/docs/stable/notes/cuda.html) | Blogs + docs | The constraints matter most: static shapes, no CPU synchronization. | 2.5 |
| 12 | [PyTorch custom operators](https://docs.pytorch.org/tutorials/advanced/custom_ops_landing_page.html) and [FlexAttention](https://pytorch.org/blog/flexattention/) | Docs + blog | Registering a kernel that survives `torch.compile`; compiler-generated attention kernels. | 3 |
| 13 | Mixed precision and memory formats: [torch.amp](https://docs.pytorch.org/docs/stable/amp.html), [NVIDIA mixed precision guide](https://docs.nvidia.com/deeplearning/performance/mixed-precision-training/index.html), [channels-last tutorial](https://docs.pytorch.org/tutorials/intermediate/memory_format_tutorial.html) | Docs | Autocast rules, tensor-core eligibility, memory layout. | 2.5 |

The book is the long pole. Read it alongside the labs across all three weeks
instead of finishing it first.

## Deep-dive resources

| Resource | Type | Notes |
|---|---|---|
| [CUDA Training Series](https://www.olcf.ornl.gov/cuda-training-series/) ([exercises](https://github.com/olcf/cuda-training-series)) (NVIDIA and Oak Ridge) | Course | 13 sessions with homework. Prioritise optimization, managed memory, streams and performance analysis. |
| [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/) | Reference | Look things up here. |
| [GPUs Go Brrr](https://hazyresearch.stanford.edu/blog/2024-05-12-tk), the [ThunderKittens paper](https://arxiv.org/abs/2410.20399) and [repository](https://github.com/HazyResearch/ThunderKittens) (Hazy Research) | Blog + paper + repo | Tensor cores, shared-memory tiles and occupancy on recent hardware. |
| [Modern GPU Programming for MLSys](https://mlc.ai/modern-gpu-programming-for-mlsys/) ([repository](https://github.com/mlc-ai/modern-gpu-programming-for-mlsys)) | Online book | What makes a kernel fast, tiled to state-of-the-art GEMM, attention kernels, measuring kernel performance. |
| [CMU 10-414/714 Deep Learning Systems](https://dlsyscourse.org/lectures/) ([assignments](https://dlsyscourse.org/assignments/)) | Course | Hardware acceleration lectures and the NDArray CUDA backend homework. |
| [Triton: An Intermediate Language and Compiler for Tiled Neural Network Computations](https://www.eecs.harvard.edu/~htk/publication/2019-mapl-tillet-kung-cox.pdf) | Paper | Design rationale for the block-based model. |
| More GPU MODE lectures | Videos | L15 (CUTLASS), L23 (tensor cores), L29 (Triton internals), L34 (low-bit Triton kernels). Find them in the [lecture index](https://github.com/gpu-mode/lectures). |
| Tensor compilers: [TVM paper](https://arxiv.org/abs/1802.04799), [Machine Learning Compilation](https://book.mlc.ai/), [TVM docs](https://tvm.apache.org/docs/), [PyTorch 2 paper](https://docs.pytorch.org/assets/pytorch2-2.pdf) | Papers + course | The schedule/compute split, autotuning, and how PyTorch 2 compiles. |

## Optional

- [MLIR paper](https://arxiv.org/abs/2002.11054) and the [Toy tutorial](https://mlir.llvm.org/docs/Tutorials/Toy/)
- [Stanford CS149 assignment 3](https://github.com/stanford-cs149/asst3): a CUDA renderer
- Reference code: [CUTLASS](https://github.com/NVIDIA/cutlass), [llm.c](https://github.com/karpathy/llm.c), [cuda-samples](https://github.com/NVIDIA/cuda-samples)

## Labs

### Lab 5.1 — Matmul ladder in CUDA

**Tools:** CUDA C++, Nsight Compute, cuBLAS as the baseline.

Following the Boehm worklog, write a naive kernel, then coalesced, then
shared-memory tiled, then register-blocked and vectorized SGEMM at 4096 × 4096.

**Deliverable:** a GFLOPS table per kernel reaching at least half of cuBLAS,
with Nsight Compute screenshots (memory throughput, occupancy, roofline) that
explain each jump.

### Lab 5.2 — A fused Triton kernel for an inference hot path

**Tools:** Triton, PyTorch custom-operator registration, `torch.profiler`.

Write a fused RMSNorm + linear kernel, or a fused sampling kernel (temperature
and top-k). Verify against PyTorch with `torch.allclose` and integrate it so that
it works under `torch.compile`.

**Deliverable:** a benchmark across shapes and dtypes (fp32, fp16, bf16) against
eager and compiled PyTorch, targeting at least 1.5× over eager, with achieved
memory bandwidth as a percentage of the device peak.

### Lab 5.3 — FlashAttention forward pass and CUDA graphs

**Tools:** Triton or CUDA, `F.scaled_dot_product_attention` as the reference,
Nsight Systems, `torch.cuda.CUDAGraph`.

Implement tiled causal attention with online softmax.

**Deliverable:**
1. Correctness within 1e-3 and a latency and peak-memory curve against naive
   attention and the PyTorch reference for sequence lengths 1k to 16k, within 2×
   of the reference.
2. A small-batch decode step captured in a CUDA graph, with Nsight Systems
   timelines before and after showing the launch gaps closing, and the change in
   per-token latency.

## Checkpoint

1. What is occupancy, and when does raising it not help?
2. Why does a tiled attention kernel use less memory traffic than the naive one?
3. Your kernel is correct but reaches 20% of peak memory bandwidth. What do you check first?
