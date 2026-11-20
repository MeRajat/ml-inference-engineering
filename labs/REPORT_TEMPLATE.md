# Lab report template

Copy this file into `labs/reports/<module>-<lab>.md` and fill it in. Keep it to
one or two pages. A reader should be able to reproduce your numbers.

## 1. Question

One sentence: what did you set out to measure or build?

## 2. Setup

| Item | Value |
|---|---|
| Model and revision | |
| Precision / quantization | |
| Engine and version | |
| GPU, driver, CUDA version | |
| Instance type and hourly cost | |
| Load pattern (open or closed loop, arrival rate, concurrency) | |
| Prompt / input distribution (lengths in and out) | |
| Run duration and warm-up | |

## 3. Prediction

Write this **before** you run anything. What do you expect, and why? Include a
back-of-envelope calculation where you can.

## 4. Results

| Configuration | TTFT p50 / p99 | Per-token latency p50 / p99 | Throughput | Peak GPU memory | Quality metric | Cost per unit |
|---|---|---|---|---|---|---|
| Baseline | | | | | | |
| Change 1 | | | | | | |
| Change 2 | | | | | | |

Add one plot if it makes the result clearer (latency against load is usually the
right one).

## 5. Explanation

Where did the measurement differ from your prediction? What does the profiler
say the time is spent on?

## 6. What you would do next

Two or three bullets.

## 7. Reproduce

Exact commands, config files and commit hashes.
