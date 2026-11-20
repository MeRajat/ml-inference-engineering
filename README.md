# ML Inference Engineering

A self-study course for senior ML engineers who want to specialise in inference:
taking a research checkpoint to a production endpoint that is fast, cheap and
reliable under real traffic.

**18 weeks · 10–12 hours a week · about two thirds hands-on**

The course is a curated path through public material (books, university
lectures, papers, talks, source code and documentation) with labs that each end
in a measured result. It was built around the requirements of research-engineer
inference roles at model labs and voice-AI companies: production serving for
latency-sensitive workloads, GPU programming and inference optimization, and
profiling across the stack from model architecture to kernels to orchestration.

## Who it is for

You already train and fine-tune models, read PyTorch fluently and know how a
transformer works. You have not yet owned serving performance in production.
Training, fine-tuning methods, prompt engineering and agent frameworks are out
of scope.

## Syllabus

| # | Module | Weeks | Hours | You finish with |
|---|---|---|---|---|
| 0 | [Orientation and setup](modules/00-orientation.md) | before week 1 | 3–4 | A working environment and GPU access |
| 1 | [Inference fundamentals and performance mental models](modules/01-inference-fundamentals.md) | 1–2 | 22 | Predicted against measured tokens per second, within about 30% |
| 2 | [Benchmarking, profiling and observability](modules/02-benchmarking-and-observability.md) | 3 | 12 | A load-test harness and dashboard you reuse all course |
| 3 | [Serving engines and LLM inference systems](modules/03-serving-engines.md) | 4–6 | 36 | Two engines benchmarked and explained from their source |
| 4 | [Model-level optimization](modules/04-model-optimization.md) | 7–9 | 36 | A speed-against-quality Pareto report |
| 5 | [GPU programming and custom kernels](modules/05-gpu-programming-and-kernels.md) | 10–12 | 36 | A custom kernel, profiled and integrated with PyTorch |
| 6 | [Real-time, streaming and audio inference](modules/06-realtime-streaming-and-audio.md) | 13–14 | 24 | A streaming speech service with tail-latency numbers |
| 7 | [Production scale and operations](modules/07-production-scale-and-operations.md) | 15–16 | 24 | Autoscaling, cache-aware routing and an automatic rollback |
| 8 | [Capstone](capstone/README.md) | 17–18 | 24 | A public project and write-up |

Then: [turning the course into a job](career/README.md).

The week-by-week plan is in [`SCHEDULE.md`](SCHEDULE.md).

## How each module is organised

- **Why this module** and **learning objectives**
- **Core resources**: do these. Each row says which part to study and roughly how long it takes.
- **Deep-dive resources**: for topics that matter to your target role
- **Optional**: background and breadth
- **Labs**: each has tools and a measurable deliverable. Write them up with
  [`labs/REPORT_TEMPLATE.md`](labs/REPORT_TEMPLATE.md).
- **Checkpoint**: questions you should be able to answer before moving on

Core resource tables often list more than the reading budget allows. Each
module says which subset is the minimum.

## If you only read ten things

1. [How To Scale Your Model](https://jax-ml.github.io/scaling-book/) (Google DeepMind): rooflines, transformer arithmetic, inference
2. [Inference Engineering](https://www.baseten.co/inference-engineering/) (Philip Kiely): the one book written for this job
3. [Making Deep Learning Go Brrrr From First Principles](https://horace.io/brrr_intro.html) (Horace He)
4. [Inside vLLM: Anatomy of a High-Throughput LLM Inference System](https://vllm.ai/blog/2025-09-05-anatomy-of-vllm) (Aleksa Gordić)
5. [Efficient Memory Management for LLM Serving with PagedAttention](https://arxiv.org/abs/2309.06180)
6. [MIT 6.5940 TinyML and Efficient Deep Learning Computing](https://hanlab.mit.edu/courses/2024-fall-65940) (Song Han)
7. [GPU MODE lectures](https://github.com/gpu-mode/lectures)
8. [How to Optimize a CUDA Matmul Kernel](https://siboehm.com/articles/22/CUDA-MMM) (Simon Boehm)
9. [Open Versus Closed: A Cautionary Tale](https://www.usenix.org/conference/nsdi-06/open-versus-closed-cautionary-tale)
10. [The Tail at Scale](https://research.google/pubs/the-tail-at-scale/) (Dean and Barroso)

## Shorter paths

| If your goal is | Do |
|---|---|
| Deploy and operate models (platform role), 10 weeks | Modules 1, 2, 3, 7, then capstone option B |
| Voice and real-time roles, 14 weeks | Modules 1, 2, 3, 4, 6, then capstone option A |
| Performance and kernel work, 12 weeks | Modules 1, 2, 4, 5, plus an upstream performance contribution |

## About the resources

- Every link was checked in November 2026. Run `python3 scripts/check_links.py`
  to re-check; a few publishers block automated requests and are reported as
  "blocked" instead of failing.
- Pointers to specific chapters, sections and files are guidance. Books get new
  editions and active repositories move files, so search by title or class name
  if a pointer is stale.
- Hour estimates are rough and assume the audience above.
- Some resources are written by vendors. They are included because the content
  is good; read their benchmarks critically.
- This field moves quickly. Prefer a project's current documentation over any
  blog post, including the ones listed here, when they disagree.

## Repository layout

```
modules/     one page per module: resources, labs, checkpoint
labs/        report template; put your reports in labs/reports/
capstone/    the two capstone options and a self-grading rubric
career/      portfolio checklist and interview preparation
scripts/     link checker
SCHEDULE.md  week-by-week plan and progress tracker
```
