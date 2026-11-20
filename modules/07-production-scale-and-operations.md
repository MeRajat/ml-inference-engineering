# Module 7 — Production scale and operations

**Weeks 17–18 · about 24 hours (9 reading, 15 labs)**

## Why this module

One fast replica is a benchmark. A service is many replicas that scale with
traffic, start quickly, route requests to the right place, survive a bad
checkpoint and cost a known amount. The posting this course targets asks for
tooling that lets researchers "ship new models to production quickly, safely";
this module is where you build that.

## Learning objectives

- Run GPUs on Kubernetes and choose between time-slicing, MPS and MIG
- Autoscale on the right signal (queue depth or in-flight requests, not GPU utilisation)
- Break a cold start into its stages and cut the largest one
- Explain why round-robin load balancing fails for language models and what
  cache-aware routing does instead
- Roll out a new checkpoint with a metric-gated canary and automatic rollback
- Model cost per unit of work and plan GPU capacity for a target load

## Core resources

| # | Resource | Type | What to study | Hours |
|---|---|---|---|---|
| 1 | Inference Engineering (Philip Kiely, Baseten): [Production](https://www.baseten.co/inference-engineering/book/07-production/) and [Autoscaling](https://www.baseten.co/inference-engineering/book/07-production/autoscaling/) | Book | Containers, autoscaling, cold starts, capacity, canary and blue-green. Vendor-written, so read critically. | 3 |
| 2 | [NVIDIA GPU Operator](https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/latest/index.html): [time-slicing](https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/latest/gpu-sharing.html), [MIG](https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/latest/gpu-operator-mig.html), and the [device plugin](https://github.com/NVIDIA/k8s-device-plugin) README for MPS | Docs | Each component and the three sharing modes. | 4 |
| 3 | Kubernetes: [scheduling GPUs](https://kubernetes.io/docs/tasks/manage-gpus/scheduling-gpus/) and [dynamic resource allocation](https://kubernetes.io/docs/concepts/resource-management/dynamic-resource-allocation/) | Docs | Extended resources against DRA. | 2 |
| 4 | [Which GPU Sharing Strategy Is Right for You?](https://www.youtube.com/watch?v=nOgxv_R13Dg) (KubeCon NA 2024) | Talk | Benchmarked comparison of the sharing modes. | 1 |
| 5 | Autoscaling: [HPA](https://kubernetes.io/docs/concepts/workloads/autoscaling/horizontal-pod-autoscale/), [KEDA concepts](https://keda.sh/docs/2.21/concepts/) and [Prometheus scaler](https://keda.sh/docs/2.21/scalers/prometheus/), [GKE best practices for autoscaling LLM inference](https://docs.cloud.google.com/kubernetes-engine/docs/best-practices/machine-learning/inference/autoscaling) | Docs | The GKE guide is the best write-up on which signal to scale on. | 4 |
| 6 | [Optimizing Load Balancing and Autoscaling for LLM Inference on Kubernetes](https://www.youtube.com/watch?v=TSEGAh1bs4A) | Talk | Measured comparisons of scaling metrics. | 1 |
| 7 | [Gateway API Inference Extension](https://gateway-api-inference-extension.sigs.k8s.io/) ([introduction](https://kubernetes.io/blog/2025/06/05/introducing-gateway-api-inference-extension/), [repository](https://github.com/kubernetes-sigs/gateway-api-inference-extension)) | Docs + blog | Inference pools and the endpoint picker: routing on queue depth, KV-cache and adapter affinity. | 3 |
| 8 | [llm-d architecture](https://llm-d.ai/docs/architecture) ([repository](https://github.com/llm-d/llm-d)) and [KV-Cache Wins You Can Feel](https://www.youtube.com/watch?v=RWOtYFMAnHI) | Docs + talk | Prefix-cache-aware routing and prefill/decode disaggregation. Run the quickstart. | 5 |
| 9 | Ray Serve: [overview](https://docs.ray.io/en/latest/serve/index.html), [autoscaling guide](https://docs.ray.io/en/latest/serve/autoscaling-guide.html), [serving LLMs](https://docs.ray.io/en/latest/serve/llm/index.html), [production guide](https://docs.ray.io/en/latest/serve/production-guide/index.html) | Docs | Target ongoing requests; model composition. | 5 |
| 10 | Cold starts: [ServerlessLLM](https://arxiv.org/abs/2401.14351), [Modal on cold starts](https://modal.com/docs/guide/cold-start) and [GPU memory snapshots](https://modal.com/blog/gpu-mem-snapshots), [Tensorizer](https://github.com/coreweave/tensorizer) | Paper + docs + repo | The stages (node, image, weights, load, warm-up) and the main mitigations. | 4 |
| 11 | Rollouts: [Argo Rollouts canary](https://argoproj.github.io/argo-rollouts/features/canary/) and [analysis](https://argo-rollouts.readthedocs.io/en/stable/features/analysis/), [Canarying releases](https://sre.google/workbook/canarying-releases/) (Google SRE Workbook) | Docs + book chapter | Metric-gated canaries. | 3 |
| 12 | Google SRE: [Handling overload](https://sre.google/sre-book/handling-overload/), [Addressing cascading failures](https://sre.google/sre-book/addressing-cascading-failures/), [Implementing SLOs](https://sre.google/workbook/implementing-slos/), [Alerting on SLOs](https://sre.google/workbook/alerting-on-slos/) | Book chapters | Load shedding, retry budgets, burn-rate alerts. Read only these chapters. | 5 |
| 13 | [LLM Inference Economics from First Principles](https://www.tensoreconomics.com/p/llm-inference-economics-from-first) (Tensor Economics) | Blog | Cost per token from hardware up. | 2 |

Resources 1, 2, 5, 7, 10, 11 and 13 fit the reading budget. Add the rest
according to which stack you choose for the labs.

## Deep-dive resources

| Resource | Type | Notes |
|---|---|---|
| [vLLM production stack](https://github.com/vllm-project/production-stack) ([docs](https://docs.vllm.ai/projects/production-stack/en/latest/), [talk](https://www.youtube.com/watch?v=ccyquE4jzcc)) and [vLLM on Kubernetes](https://docs.vllm.ai/en/latest/deployment/k8s/) | Repo + docs | One reference control plane. |
| [AIBrix](https://github.com/vllm-project/aibrix) ([paper](https://arxiv.org/abs/2504.03648)) | Repo + paper | Router, adapter management, distributed KV cache. Read the paper for the design space. |
| [NVIDIA Dynamo](https://github.com/ai-dynamo/dynamo) | Repo | Distributed inference framework. |
| [KServe canary rollouts](https://kserve.github.io/website/docs/model-serving/predictive-inference/rollout-strategies/canary) and [Knative autoscaling](https://knative.dev/docs/serving/autoscaling/) | Docs | The KServe and Knative approach to the same problems. |
| [SkyServe](https://arxiv.org/abs/2411.01438) ([docs](https://docs.skypilot.ai/en/latest/serving/sky-serve.html)) | Paper + docs | Multi-region and spot replica placement. |
| [Model Streamer](https://github.com/dsx-ai-factory/model-streamer) | Repo | Streaming weights from object storage. |
| [Load balancing in the datacenter](https://sre.google/sre-book/load-balancing-datacenter/) (Google SRE book) | Book chapter | Background for routing. |
| [NVIDIA DRA driver](https://github.com/kubernetes-sigs/dra-driver-nvidia-gpu) and [GKE Inference Gateway](https://docs.cloud.google.com/kubernetes-engine/docs/concepts/about-gke-inference-gateway) | Repo + docs | Where GPU allocation and managed gateways are heading. |
| [Ray Serve: faster first token with custom routing](https://www.anyscale.com/blog/ray-serve-faster-first-token-custom-routing) | Blog | Custom request routing. |

## Optional talks

- [Mastering GPU Management in Kubernetes Using the Operator Pattern](https://www.youtube.com/watch?v=jbpIFCkEEng)
- [GPUs on Kubernetes: what actually happens when you request one](https://www.youtube.com/watch?v=nu6bLhuvlWM)
- [Best Practices for LLM Autoscaling from the llm-d Ecosystem](https://www.youtube.com/watch?v=BiNeepp_4rg)
- [Rolling Out LLM Inference Engines on Kubernetes](https://www.youtube.com/watch?v=Kdl9o5U_gFA)
- [llm-d: Multi-Accelerator LLM Inference on Kubernetes](https://www.youtube.com/watch?v=g8_snJA_ESU)
- [Gateway API Inference Extension keynote](https://www.youtube.com/watch?v=BBqDpqATcI0) (KubeCon EU 2025) and the [Google Cloud write-up](https://cloud.google.com/blog/products/containers-kubernetes/google-bytedance-and-red-hat-improve-ai-on-kubernetes)
- Ray Summit 2025: [Ray Serve roadmap](https://www.youtube.com/watch?v=S3CaJ36xl_U) and [Ray Serve with vLLM at Apple](https://www.youtube.com/watch?v=QcmA5zsIvjU)
- [Scaling AI Inference Workloads with GPUs and Kubernetes](https://www.youtube.com/watch?v=13p4sZRKtwE)
- [llm-d observability](https://www.youtube.com/watch?v=TFCx-CixBAM)

## Labs

### Lab 7.1 — GPU sharing bake-off

**Tools:** a single GPU node with the GPU Operator; two or three small model
replicas under time-slicing, MPS and (if the card supports it) MIG.

**Deliverable:** throughput and p99 latency per mode, plus an interference
measurement with a noisy neighbour on and off.

### Lab 7.2 — Queue-depth autoscaling and cold start

**Tools:** vLLM on Kubernetes; KEDA scaling on `vllm:num_requests_waiting`,
compared with scaling on GPU utilisation and with Ray Serve's target ongoing
requests; a step load and a ramp load.

**Deliverable:** minutes in violation of your latency target and GPU-hours used
per policy, plus a cold-start waterfall (schedule, image pull, weight load,
first token) before and after one mitigation.

### Lab 7.3 — Cache-aware routing and safe rollout

**Tools:** three vLLM replicas behind the Gateway API Inference Extension or
llm-d; Argo Rollouts with an analysis template.

Compare round-robin with the endpoint picker on a multi-turn, prefix-heavy
workload. Then canary a new checkpoint, gating on p95 TTFT and error rate.

**Deliverable:** the change in p95 TTFT and throughput from routing, and a
recorded automatic rollback of a deliberately bad checkpoint.

## Checkpoint

1. Why is GPU utilisation a poor autoscaling signal for a language model server?
2. A cold start takes four minutes. Which stage is usually largest, and what are two ways to cut it?
3. What should gate promotion of a new checkpoint from 5% to 50% of traffic?
