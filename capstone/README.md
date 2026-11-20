# Capstone

**Weeks 19–20 · about 24 hours, plus whatever you choose to add**

Pick one project. Both combine every module. The first suits roles at voice and
real-time companies; the second suits platform and research-engineering roles.
The output is a public repository and a write-up with numbers.

## Option A — Self-hosted real-time voice stack with latency targets

**Build**
- Streaming speech-to-text (faster-whisper or Kyutai) → vLLM → streaming
  text-to-speech (Orpheus or Kokoro)
- Orchestrated with Pipecat or LiveKit Agents over WebRTC
- On Kubernetes with the GPU Operator

**Operate**
- Autoscaling on queue depth or active streams
- Cache-aware routing for the language model tier
- GPU, engine and trace telemetry in one dashboard

**Optimise**
- At least one model-level change from Module 4 (for example a quantized
  language model or a compiled text-to-speech decoder), with quality measured
- At least one engine or kernel-level change from Module 3 or 5

**Deliverables**
- An open-loop load test with simulated concurrent callers sending audio paced
  at real time
- A report with voice-to-voice p50, p95 and p99, time to first audio, real-time
  factor, maximum concurrent sessions per GPU at your target, cost per
  conversation-minute, and behaviour during scale-up and cold start
- A waterfall showing what each optimisation bought

## Option B — A "ship a checkpoint safely" platform

**Build** a command-line tool or CI pipeline that takes a new language-model or
text-to-speech checkpoint and:

1. Packages it with fast weight loading (Tensorizer or Model Streamer)
2. Runs an automated benchmark gate: open-loop load, goodput at your latency
   target, compared against the current baseline, plus a quality check
3. Deploys it as shadow, then canary, through Argo Rollouts or KServe, with
   analysis on p95 time to first token (or first audio), error rate and GPU memory
4. Promotes or rolls back automatically

**Deliverables**
- A regression report generated per checkpoint
- A recorded automatic rollback of a deliberately slow checkpoint
- A capacity and cost model: GPUs needed at a target request rate or stream
  count, and cost per million tokens or per audio-hour
- A short document with latency targets and a runbook

## Grading yourself

| Criterion | Evidence |
|---|---|
| Predictions | You wrote down expected numbers before measuring, and explained the gaps |
| Measurement | Open-loop load, tail percentiles, stated input distribution, warm-up excluded |
| Quality | Every speed-up is reported next to a quality metric |
| Depth | At least one finding came from a profiler trace, not from trying settings |
| Operations | Scale-up, cold start and a failed rollout are each demonstrated |
| Cost | A cost per unit of work, with the arithmetic shown |
| Reproducibility | Someone else can run it from the README |

## Write-up outline

1. The problem and the latency targets
2. Architecture diagram
3. Baseline numbers
4. Each optimisation: hypothesis, change, result, cost in quality
5. Scale and failure behaviour
6. Cost model
7. What you would do with another month

Then follow [`career/README.md`](../career/README.md).
