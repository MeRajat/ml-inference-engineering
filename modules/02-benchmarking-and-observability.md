# Module 2 — Benchmarking, profiling and observability

**Week 3 · about 12 hours (5 reading, 7 labs)**

## Why this module comes early

Every later module asks you to make something faster and prove it. Most
published inference numbers are wrong in the same few ways: closed-loop load
that hides queueing, clients that under-report tail latency, and throughput
figures with no latency target attached. Learn to measure properly once, build
the harness, and reuse it for the rest of the course.

## Learning objectives

- Choose between open-loop and closed-loop load and explain what each hides
- Explain coordinated omission and show it in your own data
- Report goodput (requests per second that meet a latency target) instead of raw throughput
- Run a rate sweep and find the saturation knee
- Stand up GPU and engine telemetry with dashboards, traces and one useful alert

## Core resources

| # | Resource | Type | What to study | Hours |
|---|---|---|---|---|
| 1 | Inference Engineering (Philip Kiely, Baseten): [Measuring latency and throughput](https://www.baseten.co/inference-engineering/book/01-prerequisites/measuring-latency-and-throughput/) and [Performance benchmarking and load testing](https://www.baseten.co/inference-engineering/book/04-software/performance-benchmarking-and-load-testing/) | Book | Both sections. | 1.5 |
| 2 | [Open Versus Closed: A Cautionary Tale](https://www.usenix.org/conference/nsdi-06/open-versus-closed-cautionary-tale) ([PDF](https://www.usenix.org/legacy/event/nsdi06/tech/full_papers/schroeder/schroeder.pdf)) and the [k6 explanation](https://grafana.com/docs/k6/latest/using-k6/scenarios/concepts/open-vs-closed/) | Paper + docs | The most important methodology paper for load testing. | 2.5 |
| 3 | [How NOT to Measure Latency](https://www.youtube.com/watch?v=lJ8ydIuPFeU) (Gil Tene) and [On Coordinated Omission](https://www.scylladb.com/2021/04/22/on-coordinated-omission/) (ScyllaDB) | Talk + blog | Why naive clients under-report tails. | 2 |
| 4 | Metric definitions: [NVIDIA NIM benchmarking overview](https://docs.nvidia.com/nim/benchmarking/llm/overview) and [metrics](https://docs.nvidia.com/nim/benchmarking/llm/metrics), [Baseten on LLM benchmarks](https://www.baseten.co/blog/understanding-performance-benchmarks-for-llm-inference/) | Docs + blog | Note where tools disagree, for example whether the first token counts in inter-token latency. | 1.5 |
| 5 | Goodput: [DistServe](https://arxiv.org/abs/2401.09670) sections 2–3 and [GenAI-Perf goodput docs](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/perf_analyzer/genai-perf/docs/goodput.html) | Paper + docs | Why throughput at a latency target is the number that matters. | 2 |
| 6 | [GuideLLM](https://github.com/vllm-project/guidellm) with the [vLLM office hours walkthrough](https://www.youtube.com/watch?v=crwoAti-mOI) | Tool + talk | Sweep, constant-rate and Poisson modes. | 3 |
| 7 | [AIPerf](https://github.com/ai-dynamo/aiperf) (successor to [GenAI-Perf](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/perf_analyzer/genai-perf/README.html)) | Tool | Concurrency against request-rate modes, goodput flags. | 2 |
| 8 | Telemetry: [DCGM exporter](https://github.com/NVIDIA/dcgm-exporter) ([docs](https://docs.nvidia.com/datacenter/cloud-native/gpu-telemetry/latest/dcgm-exporter.html)), [vLLM metrics](https://docs.vllm.ai/en/latest/usage/metrics/), [Prometheus histograms](https://prometheus.io/docs/practices/histograms/) | Docs | What "GPU utilisation" really measures; bucket and quantile pitfalls. | 3 |
| 9 | Systems Performance (Brendan Gregg): [the USE method](https://www.brendangregg.com/usemethod.html), [methodology](https://www.brendangregg.com/methodology.html), [active benchmarking](https://www.brendangregg.com/activebenchmarking.html) | Web pages | These three pages cover the ideas you need from the [book](https://www.brendangregg.com/systems-performance-2nd-edition-book.html)'s methodology and benchmarking chapters. | 2 |

## Deep-dive resources

| Resource | Type | Notes |
|---|---|---|
| [MLPerf Inference paper](https://arxiv.org/abs/1911.02549), [datacenter benchmark](https://mlcommons.org/benchmarks/inference-datacenter/), [docs](https://docs.mlcommons.org/inference/), [repository](https://github.com/mlcommons/inference), [LLM workloads in v5](https://mlcommons.org/2025/04/llm-inference-v5/) | Paper + docs | Scenarios, LoadGen, latency constraints on LLM workloads. |
| [Revisiting SLO and System Level Metrics in LLM Serving](https://arxiv.org/abs/2410.14257) | Paper | Critique of common metrics. |
| Other load generators: [LLMPerf](https://github.com/ray-project/llmperf), [inference-perf](https://github.com/kubernetes-sigs/inference-perf), [inference-benchmarker](https://github.com/huggingface/inference-benchmarker), [vLLM benchmarking](https://docs.vllm.ai/en/latest/benchmarking/), [k6](https://grafana.com/docs/k6/latest/), [Locust](https://docs.locust.io/en/stable/) | Tools | Know one well and the rest by trade-off. You will need k6 or Locust for custom WebSocket and audio protocols in Module 6. |
| OpenTelemetry: [GenAI semantic conventions](https://github.com/open-telemetry/semantic-conventions/tree/main/docs/gen-ai), [vLLM OpenTelemetry example](https://docs.vllm.ai/en/latest/examples/observability/opentelemetry/), [OpenLLMetry](https://github.com/traceloop/openllmetry) | Spec + docs | Span and metric names for end-to-end traces. |
| [wrk2](https://github.com/giltene/wrk2), [HdrHistogram](https://github.com/HdrHistogram/HdrHistogram), [Marc Brooker on latency](https://brooker.co.za/blog/2021/04/19/latency.html) | Tools + blog | Correcting for coordinated omission. |
| [DCGM user guide](https://docs.nvidia.com/datacenter/dcgm/latest/user-guide/index.html) and the [DCGM Grafana dashboard](https://grafana.com/grafana/dashboards/12239-nvidia-dcgm-exporter-dashboard/) | Docs | Field reference and a starting dashboard. |
| [AI flame graphs](https://www.brendangregg.com/blog/2024-10-29/ai-flame-graphs.html) and [Performance Analysis Methodology](https://www.youtube.com/watch?v=abLan0aXJkw) (Brendan Gregg) | Blog + talk | Methodology applied to accelerators. |

## Optional

- Public methodologies to critique: [Artificial Analysis](https://artificialanalysis.ai/methodology), [InferenceX](https://github.com/SemiAnalysisAI/InferenceX)
- [Monitoring distributed systems](https://sre.google/sre-book/monitoring-distributed-systems/) (Google SRE book): the four golden signals
- [Performance Modeling and Design of Computer Systems](https://www.cs.cmu.edu/~harchol/PerformanceModeling/book.html) (Mor Harchol-Balter, Cambridge University Press): queueing theory reference
- [GenAI-Perf / AIPerf video](https://www.youtube.com/watch?v=OuQCSOIR-Y8) and [perf_analyzer](https://github.com/triton-inference-server/perf_analyzer)

## Labs

### Lab 2.1 — Show coordinated omission in your own data

**Tools:** one vLLM endpoint; a closed-loop fixed-concurrency client and an
open-loop Poisson client (GuideLLM, AIPerf rate mode, or k6 constant arrival rate).

Run both at the same nominal throughput while injecting a two-second stall.

**Deliverable:** side-by-side p50 and p99 TTFT and end-to-end latency showing
by what factor the closed-loop client under-reports, plus a one-page methodology note.

### Lab 2.2 — Goodput curve and tool cross-check

**Tools:** GuideLLM and AIPerf on the same model and prompt distribution.

Sweep request rate with both.

**Deliverable:** throughput against latency and goodput against rate (target:
TTFT p95 under 300 ms, ITL p95 under 50 ms), the saturation knee, and an
explanation of every metric where the two tools disagree.

### Lab 2.3 — Observability stack

**Tools:** DCGM exporter, vLLM `/metrics`, Prometheus, Grafana, an OpenTelemetry
collector with Tempo or Jaeger.

**Deliverable:** a dashboard with rate, errors and duration, queue depth,
KV-cache usage, GPU utilisation and memory, and TTFT histograms; one trace from
gateway through engine to first token; one multi-window burn-rate alert that
fires during a load test.

Keep this stack and the harness. Every later lab uses them.

## Checkpoint

1. Your load test uses 32 concurrent workers and reports p99 of 400 ms. Why might real users see seconds?
2. GPU utilisation reads 100%. Is the GPU saturated? What else do you look at?
3. Two tools report different inter-token latency for the same run. Name two definitional reasons.
