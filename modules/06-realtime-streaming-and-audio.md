# Module 6 — Real-time, streaming and audio inference

**Weeks 13–14 · about 24 hours (9 reading, 15 labs)**

## Why this module

Text generation tolerates a second of delay. A voice conversation does not. This
module covers workloads where the output must be produced faster than it is
consumed, the first chunk must arrive within a few hundred milliseconds, and the
tail matters more than the median. It is the closest match to roles at voice and
speech companies.

## Learning objectives

- Define time to first audio (TTFA) and real-time factor (RTF), and work out how
  many concurrent real-time streams one GPU can carry
- Explain how codec frame rate and codebook count set the token budget for a
  speech model
- Explain the three text-to-speech serving archetypes: language model plus codec,
  small non-autoregressive, and flow matching
- Choose a transport per hop (WebRTC, WebSockets, gRPC streaming, server-sent
  events) and justify it
- Build a per-stage latency budget for a voice pipeline and apply tail-latency
  techniques such as hedged requests

## Core resources

| # | Resource | Type | What to study | Hours |
|---|---|---|---|---|
| 1 | [Voice AI and Voice Agents: An Illustrated Primer](https://voiceaiandvoiceagents.com/) | Online book | Latency, speech-to-text and text-to-speech, network transport, turn detection, hosting and scaling. | 5 |
| 2 | Inference Engineering (Philip Kiely, Baseten): [ASR models](https://www.baseten.co/inference-engineering/book/06-modalities/asr-models/) and [TTS models](https://www.baseten.co/inference-engineering/book/06-modalities/tts-models/) | Book | The serving-side view of both. | 2 |
| 3 | [The Tail at Scale](https://research.google/pubs/the-tail-at-scale/) (Dean and Barroso) | Paper | Fan-out amplification, hedged and tied requests. Map them onto a speech-to-text → language model → text-to-speech chain. | 2 |
| 4 | [Moshi](https://arxiv.org/abs/2410.00037) ([repository](https://github.com/kyutai-labs/moshi)) (Kyutai) | Paper + repo | The Mimi codec and multi-stream sections; then run the server and trace the WebSocket frame loop. | 6 |
| 5 | [Delayed Streams Modeling](https://arxiv.org/abs/2509.08753) ([repository](https://github.com/kyutai-labs/delayed-streams-modeling)) (Kyutai) | Paper + repo | Streaming-by-design speech models with batched streaming serving. | 4 |
| 6 | Streaming Whisper: [Whisper paper](https://arxiv.org/abs/2212.04356), [Whisper-Streaming paper](https://arxiv.org/abs/2307.14743) ([code](https://github.com/ufal/whisper_streaming)), [faster-whisper](https://github.com/SYSTRAN/faster-whisper), [WhisperLive](https://github.com/collabora/WhisperLive) | Papers + repos | How a 30-second-window offline model becomes a streaming one. | 5 |
| 7 | Open text-to-speech models, pick two or three: [Orpheus](https://github.com/canopyai/Orpheus-TTS), [Sesame CSM](https://github.com/SesameAILabs/csm) ([write-up](https://www.sesame.com/journal/crossing-the-uncanny-valley-of-voice)), [Kokoro](https://huggingface.co/hexgrad/Kokoro-82M) ([FastAPI server](https://github.com/remsky/Kokoro-FastAPI)), [F5-TTS](https://github.com/SWivid/F5-TTS) ([paper](https://arxiv.org/abs/2410.06885)) | Repos + papers | Read each streaming path and compare TTFA and RTF across the three archetypes. | 8 |
| 8 | [One-second voice-to-voice latency with Modal, Pipecat and open models](https://modal.com/blog/low-latency-voice-bot) | Blog | A concrete reference architecture with a measurement method. | 1.5 |
| 9 | Orchestration, pick one: [LiveKit Agents](https://docs.livekit.io/agents/) ([turn detection](https://docs.livekit.io/agents/logic/turns/), [repository](https://github.com/livekit/agents)) or [Pipecat](https://github.com/pipecat-ai/pipecat) ([pipeline docs](https://docs.pipecat.ai/pipecat/learn/pipeline)) | Docs + repos | Turn detection, interruptions, the frame and pipeline model. | 4 |
| 10 | High Performance Browser Networking (Ilya Grigorik, O'Reilly, free online): [WebSocket](https://hpbn.co/websocket/) and [WebRTC](https://hpbn.co/webrtc/) | Book | Head-of-line blocking, jitter buffers. | 4 |
| 11 | Talks: [Optimizing inference for voice models in production](https://www.youtube.com/watch?v=gmTHs5T_YAE) (Philip Kiely) and [Engineering voice agents: latency, quality and scale](https://www.youtube.com/watch?v=N7b1PJc7SFc) (Rishabh Bhargava) | Videos | Practitioner view. | 1.5 |

The reading budget covers resources 1–3, 8, 11 and one model family from each of
4–7. Read the rest as your lab choice requires.

## Deep-dive resources

| Resource | Type | Notes |
|---|---|---|
| Neural audio codecs: [SoundStream](https://arxiv.org/abs/2107.03312), [EnCodec](https://arxiv.org/abs/2210.13438) ([code](https://github.com/facebookresearch/encodec)), [Descript Audio Codec](https://arxiv.org/abs/2306.06546) ([code](https://github.com/descriptinc/descript-audio-codec)), [Mimi weights](https://huggingface.co/kyutai/mimi) | Papers + repos | Read SoundStream fully, skim the others. |
| [WebRTC for the Curious](https://webrtcforthecurious.com/) | Online book | ICE, RTP, Opus. |
| [gRPC core concepts](https://grpc.io/docs/what-is-grpc/core-concepts/) and [server-sent events](https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events/Using_server-sent_events) | Docs | Bidirectional streaming and one-way streaming. |
| [Triton decoupled models](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/decoupled_models.html) | Docs | How the Triton Inference Server streams responses. |
| Turn detection: [LiveKit's transformer end-of-turn model](https://livekit.com/blog/using-a-transformer-to-improve-end-of-turn-detection), [Pipecat smart-turn](https://github.com/pipecat-ai/smart-turn), [Silero VAD](https://github.com/snakers4/silero-vad) | Blog + repos | The part of the latency budget that is not inference. |
| [The world's fastest voice bot](https://www.daily.co/blog/the-worlds-fastest-voice-bot/) (Daily) and [Reduce the latency of your voice agent](https://www.youtube.com/watch?v=jnedfk5PL2A) (LiveKit) | Blog + video | Per-stage latency budgets. |
| [Unmute](https://github.com/kyutai-labs/unmute) (Kyutai) | Repo | A complete cascaded system to read. |

## Optional

- [XTTS paper](https://arxiv.org/abs/2406.04904), [Coqui TTS](https://github.com/coqui-ai/TTS) and [streaming XTTS on Baseten](https://www.baseten.co/blog/streaming-real-time-text-to-speech-with-xtts-v2/)
- [Hugging Face speech-to-speech](https://github.com/huggingface/speech-to-speech)
- [Hugging Face Audio Course](https://huggingface.co/learn/audio-course/chapter0/introduction) and [Stanford CS224S](https://web.stanford.edu/class/cs224s/) if you are new to speech
- [Open ASR Leaderboard](https://huggingface.co/spaces/hf-audio/open_asr_leaderboard): reports speed alongside accuracy
- [LiveKit observability](https://docs.livekit.io/testing/observability/)

## Labs

### Lab 6.1 — Streaming text-to-speech server and harness

**Tools:** Kokoro (Kokoro-FastAPI) and Orpheus (on a vLLM backend), each behind a
WebSocket endpoint that streams audio chunks; an open-loop async client you write.

**Deliverable:** a table of TTFA p50, p95 and p99, RTF, and the maximum number
of concurrent real-time streams per GPU (RTF below 1 at p95) at 1, 4, 16 and 64 streams.

### Lab 6.2 — Streaming speech recognition

**Tools:** faster-whisper with whisper_streaming or WhisperLive, against Kyutai
speech-to-text. Feed audio paced at real time over a WebSocket.

**Deliverable:** word error rate against finalisation latency (p50 and p95) for
at least three chunk settings, and streams per GPU.

### Lab 6.3 — Voice agent latency budget

**Tools:** a Pipecat or LiveKit Agents pipeline (voice activity detection and
turn detection → speech-to-text → vLLM → text-to-speech) over WebRTC, with
timestamps at every stage.

**Deliverable:** a voice-to-voice latency waterfall (p50 and p95) over 100
scripted turns, and one optimisation (for example co-location or hedged
text-to-speech requests) with the measured before and after.

## Checkpoint

1. A text-to-speech model emits 12.5 codec frames per second with 8 codebooks. How many tokens per second must it generate to stay real-time, and what does that imply for batch size?
2. Why can a batch size that maximises throughput break a real-time stream?
3. Where would you hedge requests in a voice pipeline, and what does it cost?
