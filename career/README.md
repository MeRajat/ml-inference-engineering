# Turning the course into a job

## What hiring teams look for

Inference postings usually say some version of "show us hard problems you have
solved, with artifacts". The posting this course was built around asks for:

- experience deploying and serving ML models in production, ideally for
  latency-sensitive or real-time applications
- GPU programming and inference optimization skills (CUDA, Triton, TensorRT,
  or serving frameworks such as vLLM or SGLang)
- the ability to profile, diagnose and remove bottlenecks across the serving
  stack on your own, from model architecture to kernels to orchestration, and to
  build the tooling to measure it

Map your evidence to those three lines directly.

| Requirement | Evidence from this course |
|---|---|
| Production serving, real-time | Module 6 streaming speech service, Module 7 autoscaled deployment, capstone |
| GPU programming and optimization | Module 4 optimization reports, Module 5 custom kernel, an upstream PR |
| Profiling across the stack, building tooling | Module 2 benchmark harness, profiler traces in every report |

## Portfolio checklist

- [ ] One public repository with the capstone, runnable from a README
- [ ] One long write-up with real numbers: baseline, each change, final result, cost
- [ ] Three or four short lab reports linked from the capstone README
- [ ] At least one merged pull request to an open-source inference project
      (vLLM, SGLang, llama.cpp, a quantization library or a benchmarking tool).
      Documentation and benchmark fixes count as a start; a performance fix is better.
- [ ] A one-page summary of the above for recruiters who will not read the repository

## Interview preparation

Practise answering these out loud, with numbers.

**Mental models**
- A 13B-parameter model in half precision on a 24 GB card: how much room is left
  for KV cache, and how many concurrent 4k-token sequences does that allow?
- Why is decode memory-bound and prefill compute-bound? What does that imply for batching?
- Throughput doubled but p99 time to first token tripled. What happened?

**Systems design**
- Design a serving stack for a streaming text-to-speech model with a 200 ms
  time-to-first-audio target at p95.
- Traffic is 10× higher for two hours each day. How do you autoscale GPU replicas
  when a cold start takes four minutes?
- How do you roll out a new checkpoint to 100% of traffic safely?

**Debugging**
- GPU utilization reads 100% but throughput is half of what you predicted. List
  your first five checks.
- Latency is fine at 10 requests per second and collapses at 12. Why, and what
  do you measure to confirm?
- A quantized model matches the baseline on your benchmark but users report worse
  output. How do you investigate?

**Optimization trade-offs**
- When does quantization make inference slower?
- When is speculative decoding a net loss?
- Distil, quantize or compile: which first, for which workload?

## Where to apply

Roles are titled Inference Engineer, ML Performance Engineer, Research Engineer
(Inference), ML Infrastructure Engineer or Model Serving Engineer. Look at model
labs, voice and speech companies, inference providers, GPU clouds and the
platform teams of larger product companies. Remote policies vary by company and
by role, so check each posting.
