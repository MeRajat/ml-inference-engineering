# Module 0 — Orientation and setup

**Time:** 3–4 hours, before week 1.

## Who this course is for

You are a senior ML engineer. You can train and fine-tune models, you read PyTorch
fluently, and you know how a transformer works. What you have not done is own the
path from a research checkpoint to a production endpoint that is fast, cheap and
reliable under real traffic. This course covers that path.

It does not cover training, fine-tuning methods, prompt engineering, RAG or
agent frameworks.

## What "done" looks like

By the end you can:

1. Predict a model's latency and throughput on a given GPU before running it, and
   explain the gap when the measurement disagrees.
2. Deploy a model on at least two serving engines and explain their scheduler,
   batching and memory design from the source code.
3. Apply quantization, compilation and decoding-level optimizations and report the
   speed gain next to the quality loss.
4. Write and profile a custom GPU kernel, and read a profiler trace from Python
   down to the kernel level.
5. Serve a streaming speech model with stated tail-latency targets.
6. Run an autoscaled multi-replica deployment with dashboards, load tests and a
   safe rollout path for new checkpoints.

## How to work through it

- **Pace:** 18 weeks at 10–12 hours a week. See [`SCHEDULE.md`](../SCHEDULE.md).
- **Ratio:** about one third reading and lectures, two thirds labs. If you are
  short on time, cut optional reading, never the lab.
- **Resource levels:** each module marks resources as **core** (do these),
  **deep-dive** (do when the topic matters for your target role) or **optional**.
- **Write everything down:** every lab ends in a short report with numbers. Those
  reports become your portfolio. Use [`labs/REPORT_TEMPLATE.md`](../labs/REPORT_TEMPLATE.md).

## Environment

### Local machine

- Python 3.11 or newer, `uv` or `conda` for environments
- Docker
- `kubectl`, `helm`, and `kind` or `k3d` for local Kubernetes practice (Module 7)
- A load-testing tool is introduced in Module 2

### GPU access

Nearly every lab needs an NVIDIA GPU. You do not need to own one.

| Need | Suggested hardware | Used in |
|---|---|---|
| Small models, profiling, kernels | One 24 GB card (for example an RTX 4090 or A10G class) | Modules 1, 2, 5 |
| Serving 7B–30B models | One 48–80 GB card (for example an L40S, A100 or H100 class) | Modules 3, 4, 6 |
| Multi-replica and autoscaling | Two or more GPU nodes for a few hours | Module 7, capstone |

Rent by the hour from any cloud or GPU marketplace and shut instances down when
you stop. Keep model weights on a persistent volume so you do not pay to
re-download them. Record what each lab cost you: cost per unit of work is one of
the metrics this course teaches.

### Accounts

- A Hugging Face account and access token, with licences accepted for the gated
  models you plan to use
- A container registry (Docker Hub or GitHub Container Registry)

## Baseline check

Before week 1, confirm you can do the following without looking anything up. If
not, spend a few extra hours here first.

- Write the forward pass of a decoder-only transformer block from memory
- Explain what `torch.no_grad()`, `torch.inference_mode()` and `model.eval()` each change
- Explain half precision versus bfloat16 and when each loses accuracy
- Build and run a Docker image that uses a GPU
- Explain the difference between p50 and p99 latency and why the mean hides problems
