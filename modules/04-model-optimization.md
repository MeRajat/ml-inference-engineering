# Module 4 — Model-level optimization

**Weeks 7–9 · about 36 hours (14 reading and lectures, 22 labs)**

## Why this module

The engine decides how well you use the GPU. The model decides how much work
there is to do. Quantization, distillation, compilation and decoding tricks
change the work itself, and each one trades quality for speed. The skill is
measuring both sides of that trade honestly.

As a senior ML engineer you have probably used some of these for training or
fine-tuning. Here the lens is different: which formats a serving engine
actually accelerates, what breaks under load, and how to evaluate quality loss.

## Learning objectives

- Choose between FP8, INT8 and 4-bit weight formats for a given engine and GPU
- Explain GPTQ, AWQ and SmoothQuant well enough to debug a bad result
- Build a compilation ladder (eager → `torch.compile` → ONNX Runtime → TensorRT)
  and locate the layer responsible for an accuracy drop
- Predict speculative decoding speed-up from acceptance rate and cost ratio
- Explain step distillation for diffusion and flow models, and how it applies to speech
- Report every speed-up next to a quality metric

## Core resources

### Start here

| # | Resource | Type | What to study | Hours |
|---|---|---|---|---|
| 1 | [Large Transformer Model Inference Optimization](https://lilianweng.github.io/posts/2023-01-10-inference-optimization/) (Lilian Weng) | Blog | Read first, as a map. | 2 |
| 2 | [Inference Engineering](https://www.baseten.co/inference-engineering/) (Philip Kiely) | Book | Techniques chapter: quantization (number formats, approaches, measuring quality impact) and speculative decoding. | 3 |
| 3 | [MIT 6.5940 TinyML and Efficient Deep Learning Computing](https://hanlab.mit.edu/courses/2024-fall-65940) (Song Han) | Course | See the lecture list below. | 12 |

**MIT 6.5940 lectures for this module** ([full playlist](https://www.youtube.com/playlist?list=PL80kAHvQbh-qGtNc54A6KW4i4bkTPjiRF))

| Lecture | Topic |
|---|---|
| [3](https://www.youtube.com/watch?v=EjsB0WgIfUM), [4](https://youtu.be/upaZrpXkELc) | Pruning and sparsity |
| [5](https://www.youtube.com/watch?v=ymAzUz3qlIA), [6](https://youtu.be/wrcgWm_nUeE) | Quantization |
| [9](https://www.youtube.com/watch?v=Ubj3QXv4rjw) | Knowledge distillation |
| [13](https://youtu.be/sTz2tXG1T0c) | Efficient LLM deployment |
| [14](https://youtu.be/OCdwWfVoQ-Q) | LLM post-training |
| [15](https://youtu.be/kgTWKjbnrBA) | Long-context LLM |
| [18](https://youtu.be/LXrqmQrscf0) | Diffusion models |

Skip the lectures on architecture search and microcontrollers.

### Quantization

| # | Resource | What to study | Hours |
|---|---|---|---|
| 4 | [A Survey of Quantization Methods for Efficient Neural Network Inference](https://arxiv.org/abs/2103.13630) | The taxonomy: uniform and non-uniform, symmetric and asymmetric, post-training and quantization-aware. | 3 |
| 5 | [A White Paper on Neural Network Quantization](https://arxiv.org/abs/2106.08295) (Qualcomm) | The practical debugging pipeline. | 3 |
| 6 | [GPTQ](https://arxiv.org/abs/2210.17323), [AWQ](https://arxiv.org/abs/2306.00978), [SmoothQuant](https://arxiv.org/abs/2211.10438) | The three post-training algorithms behind current LLM tooling. | 5 |
| 7 | ["Give Me BF16 or Give Me Death"?](https://arxiv.org/abs/2411.02355) with the [summary post](https://developers.redhat.com/articles/2024/10/17/we-ran-over-half-million-evaluations-quantized-llms) | The best reference on how to evaluate quality loss. | 2 |
| 8 | [A Visual Guide to Quantization](https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-quantization) and the [GGUF format docs](https://huggingface.co/docs/hub/gguf) | Intuition for GPTQ and GGUF block quantization. | 1.5 |
| 9 | [llm-compressor examples](https://github.com/vllm-project/llm-compressor/tree/main/examples) ([docs](https://docs.vllm.ai/projects/llm-compressor/en/latest/)) and [vLLM quantization support](https://docs.vllm.ai/en/latest/features/quantization/index.html) | Run the FP8, W4A16, AWQ, INT8 and KV-cache examples. | 5 |
| 10 | llama.cpp tools: [quantize](https://github.com/ggml-org/llama.cpp/blob/master/tools/quantize/README.md), [imatrix](https://github.com/ggml-org/llama.cpp/tree/master/tools/imatrix), [perplexity](https://github.com/ggml-org/llama.cpp/tree/master/tools/perplexity) | GGUF quant types, importance matrices, built-in KL-divergence measurement. | 3 |

### Compilation and graph optimization

| # | Resource | What to study | Hours |
|---|---|---|---|
| 11 | TensorRT: [quantized types](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/work-with-quantized-types.html) and [best practices](https://docs.nvidia.com/deeplearning/tensorrt/latest/performance/best-practices.html) | Explicit quantize/dequantize semantics; benchmarking, layer fusion, profiling. | 4 |
| 12 | [Torch-TensorRT](https://docs.pytorch.org/TensorRT/) ([examples](https://github.com/pytorch/TensorRT/tree/main/examples)) and [Polygraphy](https://github.com/NVIDIA/TensorRT/tree/main/tools/Polygraphy) | Compilation paths from PyTorch; per-layer accuracy debugging. | 4 |
| 13 | ONNX Runtime: [quantization](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html) and [graph optimizations](https://onnxruntime.ai/docs/performance/model-optimizations/graph-optimizations.html) | Formats, static and dynamic, optimization levels. | 2 |
| 14 | PyTorch: [quantization](https://docs.pytorch.org/docs/stable/quantization.html), [torchao](https://docs.pytorch.org/ao/stable/index.html) ([repository](https://github.com/pytorch/ao)), [torch.export](https://docs.pytorch.org/tutorials/intermediate/torch_export_tutorial.html) | The native quantization library and the export path. | 3 |
| 15 | [NVIDIA Model Optimizer examples](https://github.com/NVIDIA/Model-Optimizer/tree/main/examples) | Post-training quantization for Hugging Face and ONNX models, diffusion, speculative decoding benchmarks. | 4 |

### Decoding and cache

| # | Resource | What to study | Hours |
|---|---|---|---|
| 16 | [Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192) | The acceptance rule and expected speed-up. | 1.5 |
| 17 | Hugging Face: [assisted generation](https://huggingface.co/blog/assisted-generation), [speculative decoding for Whisper](https://huggingface.co/blog/whisper-speculative-decoding) | Runnable walkthroughs; the second covers audio. | 1.5 |
| 18 | [Flash-Decoding](https://pytorch.org/blog/flash-decoding/) and [GPT, Fast](https://pytorch.org/blog/accelerating-generative-ai-2/) | Stacking optimizations with tokens per second at each step: the template for your lab reports. | 2 |
| 19 | [A Survey on LLM Acceleration based on KV Cache Management](https://arxiv.org/abs/2412.19442) | Token-, model- and system-level methods. | 2 |
| 20 | Hugging Face: [LLM inference optimization](https://huggingface.co/docs/transformers/main/en/llm_optims), [quantization overview](https://huggingface.co/docs/transformers/quantization/overview) | Backend comparison table. | 1.5 |

### Distillation

| # | Resource | What to study | Hours |
|---|---|---|---|
| 21 | [Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531) and [DistilBERT](https://arxiv.org/abs/1910.01108) | Soft targets, then the standard transformer recipe. | 1.5 |
| 22 | [The paradox of diffusion distillation](https://sander.ai/2024/02/28/paradox.html) (Sander Dieleman) | Read before the step-distillation papers. | 1.5 |
| 23 | [Consistency Models](https://arxiv.org/abs/2303.01469) and [DMD2](https://arxiv.org/abs/2405.14867) | Read these two closely. | 3 |
| 24 | [Distil-Whisper](https://arxiv.org/abs/2311.00430) with the [training code](https://github.com/huggingface/distil-whisper/tree/main/training) | An end-to-end recipe for a speech model. | 4 |
| 25 | [diffusers consistency distillation](https://github.com/huggingface/diffusers/tree/main/examples/consistency_distillation) and [few-step inference docs](https://huggingface.co/docs/diffusers/main/en/using-diffusers/inference_with_lcm) | Runnable scripts. | 3 |

Resources 1–3, 6, 7, 9, 11, 16 and 22 fit the reading budget. Choose the rest
by lab.

## Deep-dive resources

| Topic | Resources |
|---|---|
| Quantization internals | [Integer-only inference](https://arxiv.org/abs/1712.05877), [LLM.int8()](https://arxiv.org/abs/2208.07339), [QLoRA](https://arxiv.org/abs/2305.14314), reference code for [GPTQ](https://github.com/IST-DASLab/gptq) and [AWQ](https://github.com/mit-han-lab/llm-awq), [GGUF quantization design discussion](https://github.com/ggml-org/llama.cpp/discussions/5063), [community KL-divergence comparison](https://gist.github.com/Artefact2/b5f810600771265fc1e39442288e8ec9) |
| KV-cache compression | [KIVI](https://arxiv.org/abs/2402.02750), [H2O](https://arxiv.org/abs/2306.14048), [KV cache quantization in Transformers](https://huggingface.co/blog/kv-cache-quantization) |
| Speculative decoding and early exit | [Chen et al.](https://arxiv.org/abs/2302.01318), [Medusa](https://arxiv.org/abs/2401.10774), [EAGLE](https://arxiv.org/abs/2401.15077), [LayerSkip](https://arxiv.org/abs/2404.16710) ([code](https://github.com/facebookresearch/LayerSkip)), [assisted decoding docs](https://huggingface.co/docs/transformers/main/en/assisted_decoding) |
| Step distillation lineage | [Progressive Distillation](https://arxiv.org/abs/2202.00512), [Latent Consistency Models](https://arxiv.org/abs/2310.04378) ([LCM-LoRA post](https://huggingface.co/blog/lcm_lora)), [Adversarial Diffusion Distillation](https://arxiv.org/abs/2311.17042), [DMD](https://arxiv.org/abs/2311.18828) |
| Speech distillation | [DMOSpeech](https://arxiv.org/abs/2410.11097), [CoMoSpeech](https://arxiv.org/abs/2305.06908) |
| Pruning | [Deep Compression](https://arxiv.org/abs/1510.00149), [Lottery Ticket Hypothesis](https://arxiv.org/abs/1803.03635), [SparseGPT](https://arxiv.org/abs/2301.00774), [Wanda](https://arxiv.org/abs/2306.11695) |
| Efficient attention | [Mistral 7B](https://arxiv.org/abs/2310.06825) for sliding-window attention with a rolling buffer |
| Courses | [Stanford CS229S](https://cs229s.stanford.edu/fall2023/calendar/) lectures on compression and ML compilers; [CMU 11-868](https://llmsystem.github.io/llmsystem2026spring/docs/Syllabus/) quantization and attention-optimization decks; [Machine Learning Compilation](https://mlc.ai/summer22/) (Tianqi Chen) |
| Books | [Efficient Processing of Deep Neural Networks](https://doi.org/10.1007/978-3-031-01766-7) (Sze, Chen, Yang, Emer; Springer), with its [free tutorial-paper precursor](https://arxiv.org/abs/1703.09039) |

## Optional

- [Quantization explained with PyTorch](https://www.youtube.com/watch?v=0VdNflU08yA) (Umar Jamil): a refresher if you are rusty
- [Machine Learning Systems, model compression chapter](https://mlsysbook.ai/vol1/10_model_compression/10_model_compression.html) (Vijay Janapa Reddi, MIT Press, open access)
- [Efficient Deep Learning Book](https://efficientdlbook.com/) (Menghani and Singh, draft) with [codelabs](https://github.com/EfficientDL/book/tree/main/codelabs)
- [Model Compression and Efficient Inference for LLMs: A Survey](https://arxiv.org/abs/2402.09748)
- [LoRA](https://arxiv.org/abs/2106.09685): adapter merging for zero-overhead inference
- bitsandbytes: [repository](https://github.com/bitsandbytes-foundation/bitsandbytes), [docs](https://huggingface.co/docs/transformers/main/en/quantization/bitsandbytes), [8-bit](https://huggingface.co/blog/hf-bitsandbytes-integration) and [4-bit](https://huggingface.co/blog/4bit-transformers-bitsandbytes) posts
- Hugging Face [quantization concepts](https://huggingface.co/docs/transformers/main/en/quantization/concept_guide), [Optimum](https://huggingface.co/docs/optimum/index) ([quantization guide](https://huggingface.co/docs/optimum/concept_guides/quantization), [ONNX Runtime usage](https://huggingface.co/docs/optimum-onnx/onnxruntime/usage_guides/quantization))
- Other compilers: [ExecuTorch](https://docs.pytorch.org/executorch/stable/index.html), [OpenVINO model optimization](https://docs.openvino.ai/2025/openvino-workflow/model-optimization.html) and [NNCF](https://github.com/openvinotoolkit/nncf), [Apache TVM](https://tvm.apache.org/docs/get_started/overview.html)
- [TensorRT diffusion demo](https://github.com/NVIDIA/TensorRT/tree/release/10.9/demo/Diffusion) (release branch) and [NVIDIA's post-training quantization walkthrough](https://developer.nvidia.com/blog/optimizing-llms-for-performance-and-accuracy-with-post-training-quantization/)

## Labs

### Lab 4.1 — Quantization Pareto sweep

**Tools:** llm-compressor, bitsandbytes, llama.cpp, vLLM, lm-evaluation-harness.

Take an 8B-class open model. Produce BF16, FP8, INT8 (SmoothQuant + GPTQ),
4-bit GPTQ and 4-bit AWQ variants; add NF4 and GGUF variants with and without an
importance matrix. Serve the safetensors variants in vLLM and the GGUF ones in llama.cpp.

**Deliverable:** a table and Pareto plot of TTFT, tokens per second at
concurrency 1, 8 and 32, and GPU memory, against perplexity, KL-divergence from
BF16 and two or three task evaluations. State which format you would ship and
why, and compare with the published results in resource 7.

### Lab 4.2 — Compilation ladder

**Tools:** PyTorch, ONNX Runtime, TensorRT, Polygraphy.

For an encoder or ViT, measure eager FP32 → mixed precision → `torch.compile` →
ONNX Runtime with graph optimizations on and off → TensorRT FP16 → TensorRT INT8
with calibration.

**Deliverable:** p50 and p99 latency and throughput at batch 1, 8 and 32, engine
build time and accuracy change per rung; a Polygraphy per-layer comparison that
locates the layer with the largest INT8 error; and a mixed-precision fix.

### Lab 4.3 — Speculative decoding and KV-cache compression

**Tools:** vLLM or Hugging Face assisted generation.

Measure the target model alone, with a draft model, and with n-gram lookup.

**Deliverable:** acceptance rate, mean accepted tokens per step and end-to-end
speed-up on three workloads (code, summarization, chat), compared with the
analytical prediction from resource 16. Extension: enable KV-cache quantization
and report memory saved, context or batch gained, and quality change on a
long-context task.

### Lab 4.4 — Distillation for speech or diffusion (pick one)

- **Speech recognition:** follow the Distil-Whisper training code to distil
  Whisper to a two-layer decoder on a small dataset, then add speculative
  decoding with the teacher. Report real-time factor against word error rate on
  in-distribution and out-of-distribution sets.
- **Text-to-speech:** benchmark a flow-matching model such as F5-TTS at
  different step counts. Report real-time factor against word error rate and
  speaker similarity.
- **Diffusion:** run the consistency distillation script or compare public
  few-step checkpoints. Report images per second at 1, 2, 4, 8 and 25 steps
  against a quality score and a small human-preference grid.

## Checkpoint

1. A 4-bit model is no faster than FP8 on your engine. Give two reasons.
2. Draft model acceptance is 60% and the draft costs 10% of the target per token. Roughly what speed-up do you expect, and what erodes it under load?
3. Your INT8 engine loses three points of accuracy. How do you find the layer responsible?
