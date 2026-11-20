# Module 1B — Transformer internals and architectural optimizations

**Weeks 3–4 · about 24 hours (10 reading and lectures, 14 labs)**

## Why this module

Module 1 gave you a cost model. This module applies it to the architecture
itself. Almost every structural change made to the transformer since 2017 was
motivated by cost: grouped-query attention shrinks the KV cache, rotary
positions make cached keys reusable, mixture-of-experts separates memory
footprint from compute per token, and depth transformers make multi-codebook
speech generation fit a real-time budget.

You already know the vanilla transformer. The goal here is to be able to open
any model's configuration file and say what it will cost to serve, and why its
designers made each choice.

## Learning objectives

- Describe the modern decoder block (pre-norm, RMSNorm, SwiGLU, rotary
  positions, grouped-query attention) and what each change did to compute and memory
- Compute KV-cache bytes per token from a configuration file for multi-head,
  grouped-query, multi-query and latent attention
- Explain how each positional scheme affects length extrapolation and whether
  cached keys and values stay valid when the window slides
- Explain why approximate attention (sparse, linear, low-rank) mostly lost to
  exact attention with a better implementation
- Explain why a mixture-of-experts model's memory follows total parameters while
  its compute follows active parameters, and what expert imbalance does to latency
- Compare the serving profile of decoder-only, encoder-only, encoder-decoder,
  diffusion and multi-codebook audio transformers

## Core resources

### The block and what changed

| # | Resource | Type | What to study | Hours |
|---|---|---|---|---|
| 1 | [Attention Is All You Need](https://arxiv.org/abs/1706.03762) | Paper | Re-read section 3 and the complexity table with a cost lens. This is the baseline. | 1.5 |
| 2 | [Stanford CS336, Lecture 3: Architectures](https://www.youtube.com/watch?v=lVynu4bo1rY) ([course](https://cs336.stanford.edu/)) | Lecture | The best single lecture on what modern language models changed: norm placement, RMSNorm, SwiGLU, rotary positions, hyperparameter ratios. | 1.5 |
| 3 | [A Mathematical Framework for Transformer Circuits](https://transformer-circuits.pub/2021/framework/index.html) (Anthropic) | Paper | The residual-stream view. Read the architecture overview sections. | 3 |
| 4 | [The Big LLM Architecture Comparison](https://magazine.sebastianraschka.com/p/the-big-llm-architecture-comparison) and [From GPT-2 to gpt-oss](https://magazine.sebastianraschka.com/p/from-gpt-2-to-gpt-oss-analyzing-the) (Sebastian Raschka) | Blogs | What each open model family changed and why. | 3 |
| 5 | [LLaMA explained: KV-Cache, Rotary Positional Embedding, RMS Norm, Grouped Query Attention, SwiGLU](https://www.youtube.com/watch?v=Mn_9W1nCFLo) (Umar Jamil) | Video | One video covering the modern block in inference terms. | 1.2 |
| 6 | [ModernBERT](https://arxiv.org/abs/2412.13663) ([write-up](https://huggingface.co/blog/modernbert)) | Paper + blog | An encoder rebuilt with modern parts. The serving case for encoder-only models. | 2 |

### Positions

| # | Resource | Type | What to study | Hours |
|---|---|---|---|---|
| 7 | [Rotary Embeddings: A Relative Revolution](https://blog.eleuther.ai/rotary-embeddings/) (EleutherAI) | Blog | The clearest explanation, with reference implementations. | 1 |
| 8 | [RoFormer](https://arxiv.org/abs/2104.09864) | Paper | The derivation. Note that rotation is applied to queries and keys in every layer, so nothing positional is stored in the embeddings. | 1.5 |

### Attention structure

| # | Resource | Type | What to study | Hours |
|---|---|---|---|---|
| 9 | [Fast Transformer Decoding: One Write-Head is All You Need](https://arxiv.org/abs/1911.02150) (Shazeer) | Paper | The original memory-bandwidth argument for multi-query attention. | 1 |
| 10 | [GQA](https://arxiv.org/abs/2305.13245) | Paper | The interpolation between multi-head and multi-query that most open models use. | 0.75 |
| 11 | [DeepSeek-V2](https://arxiv.org/abs/2405.04434) and the [DeepSeek-V3 reference model](https://github.com/deepseek-ai/DeepSeek-V3/blob/main/inference/model.py) | Paper + code | Multi-head latent attention: the paper's section on it, then the `MLA`, `Gate`, `Expert`, `MoE` and `Block` classes. | 3 |
| 12 | [Mistral 7B](https://arxiv.org/abs/2310.06825) and [mistral-inference](https://github.com/mistralai/mistral-inference/blob/main/src/mistral_inference/transformer.py) | Paper + code | Sliding-window attention with a rolling-buffer cache. Read `transformer.py`, `cache.py` and `rope.py`. | 2 |
| 13 | [A Visual Guide to Attention Variants in Modern LLMs](https://magazine.sebastianraschka.com/p/visual-attention-variants) with the [chapter 4 bonus code](https://github.com/rasbt/LLMs-from-scratch/tree/main/ch04) (Sebastian Raschka) | Blog + code | Current overview with runnable code for KV cache, grouped-query, latent attention, sliding window and mixture-of-experts. | 4 |
| 14 | [The Transformer Family Version 2.0](https://lilianweng.github.io/posts/2023-01-27-the-transformer-family-v2/) (Lilian Weng) | Blog | Dense catalogue of positional, attention-span and efficient-attention variants. | 2.5 |
| 15 | [Stanford CS336, Lecture 4: Attention Alternatives](https://www.youtube.com/watch?v=cKSwj_qZ8Jg) | Lecture | Attention variants and mixture-of-experts. | 1.5 |

FlashAttention is the implementation-level answer to attention cost. You read
the papers in [Module 1](01-inference-fundamentals.md) and implement it in
[Module 5](05-gpu-programming-and-kernels.md).

### Mixture-of-experts

| # | Resource | Type | What to study | Hours |
|---|---|---|---|---|
| 16 | [Mixtral of Experts](https://arxiv.org/abs/2401.04088) and the architecture section of the [DeepSeek-V3 technical report](https://arxiv.org/abs/2412.19437) | Papers | Routing, shared experts, load balancing. | 2.5 |
| 17 | [Mixture of Experts Explained](https://huggingface.co/blog/moe) (Hugging Face) or [Mixture-of-Experts LLMs](https://cameronrwolfe.substack.com/p/moe-llms) (Cameron Wolfe) | Blog | Pick one. Routing, load balancing, and the gap between active and total parameters. | 1.5 |

### Beyond text

| # | Resource | Type | What to study | Hours |
|---|---|---|---|---|
| 18 | [Whisper](https://arxiv.org/abs/2212.04356) and [`model.py`](https://github.com/openai/whisper/blob/main/whisper/model.py) | Paper + code | Encoder-decoder audio: the encoder runs once; the decoder keeps self-attention and cross-attention caches. | 2.5 |
| 19 | [Moshi](https://arxiv.org/abs/2410.00037) and [`lm.py`](https://github.com/kyutai-labs/moshi/blob/main/moshi/moshi/models/lm.py) | Paper + code | A temporal transformer plus a small depth transformer over audio codebooks. | 4 |
| 20 | Sesame CSM: [write-up](https://www.sesame.com/journal/crossing-the-uncanny-valley-of-voice), [`models.py`](https://github.com/SesameAILabs/csm/blob/main/models.py), [model docs](https://huggingface.co/docs/transformers/en/model_doc/csm) | Blog + code | A backbone predicts the first codebook; a small decoder predicts the rest. | 2 |
| 21 | [MusicGen](https://arxiv.org/abs/2306.05284) and [`codebooks_patterns.py`](https://github.com/facebookresearch/audiocraft/blob/main/audiocraft/modules/codebooks_patterns.py) ([model docs](https://github.com/facebookresearch/audiocraft/blob/main/docs/MUSICGEN.md)) | Paper + code | Codebook interleaving: delayed, parallel, unrolled and coarse-first patterns. | 2.5 |

### Code to read and write

| # | Repository | Read | Hours |
|---|---|---|---|
| 22 | [nanoGPT](https://github.com/karpathy/nanoGPT/blob/master/model.py) | `model.py`: the GPT-2-style baseline for the labs. | 1 |
| 23 | [nanochat](https://github.com/karpathy/nanochat/blob/master/nanochat/gpt.py) | `gpt.py`: a modernised single-file model with rotary positions, RMS norm and a KV-head setting. Diff it against nanoGPT. | 1.5 |
| 24 | Meta Llama reference: [Llama 2](https://github.com/meta-llama/llama/blob/main/llama/model.py), [Llama 3](https://github.com/meta-llama/llama3/blob/main/llama/model.py) | `model.py`: about 300 lines with RMSNorm, rotary positions, KV cache and SwiGLU. | 1.5 |
| 25 | LLMs-from-scratch chapter 5 bonus: [GPT to Llama](https://github.com/rasbt/LLMs-from-scratch/tree/main/ch05/07_gpt_to_llama), [Qwen3](https://github.com/rasbt/LLMs-from-scratch/tree/main/ch05/11_qwen3), [Gemma 3](https://github.com/rasbt/LLMs-from-scratch/tree/main/ch05/12_gemma3) | The conversion notebooks, and the standalone notebooks with and without a KV cache. | 5 |

The reading budget covers resources 1, 2, 4, 5, 7, 9, 10, 13, 16 and 22–24, plus
one of 18–21. If you are aiming at voice roles, do all of 18–21 and trim
elsewhere; they set up [Module 6](06-realtime-streaming-and-audio.md).

## Deep-dive resources

| Topic | Resources |
|---|---|
| Normalization and MLP | [Pre-norm against post-norm](https://arxiv.org/abs/2002.04745), [RMSNorm](https://arxiv.org/abs/1910.07467), [GLU variants (SwiGLU)](https://arxiv.org/abs/2002.05202), [GELU](https://arxiv.org/abs/1606.08415), [weight tying](https://arxiv.org/abs/1608.05859) |
| Positions | [ALiBi](https://arxiv.org/abs/2108.12409), [relative position representations](https://arxiv.org/abs/1803.02155), [T5](https://arxiv.org/abs/1910.10683) |
| Why approximate attention lost | Skim abstracts and method figures: [Longformer](https://arxiv.org/abs/2004.05150), [Reformer](https://arxiv.org/abs/2001.04451), [Performer](https://arxiv.org/abs/2009.14794), [Linformer](https://arxiv.org/abs/2006.04768), [Big Bird](https://arxiv.org/abs/2007.14062). Then the surveys: [Efficient Transformers](https://arxiv.org/abs/2009.06732), [A Survey of Transformers](https://arxiv.org/abs/2106.04554). |
| Mixture-of-experts | [Switch Transformers](https://arxiv.org/abs/2101.03961), [Mistral and Mixtral explained](https://www.youtube.com/watch?v=UiX8K-xBUpE) (Umar Jamil) |
| Sharing and skipping | [ALBERT](https://arxiv.org/abs/1909.11942) (sharing saves memory, not compute), [LayerSkip](https://arxiv.org/abs/2404.16710) |
| Current open models | [A Technical Tour of the DeepSeek Models](https://magazine.sebastianraschka.com/p/technical-deepseek), [Qwen3 From Scratch](https://magazine.sebastianraschka.com/p/qwen3-from-scratch), [Recent Developments in LLM Architectures](https://magazine.sebastianraschka.com/p/recent-developments-in-llm-architectures). Architecture sections and config tables only: [Llama 2](https://arxiv.org/abs/2307.09288), [Gemma 2](https://arxiv.org/abs/2408.00118), [Gemma 3](https://arxiv.org/abs/2503.19786), [Qwen2](https://arxiv.org/abs/2407.10671), [Qwen3](https://arxiv.org/abs/2505.09388). |
| Vision and diffusion | [Vision Transformer](https://arxiv.org/abs/2010.11929), [Diffusion Transformer](https://arxiv.org/abs/2212.09748) ([`models.py`](https://github.com/facebookresearch/DiT/blob/main/models.py)). A diffusion transformer has no KV cache: cost is steps times a full forward pass. |
| Speech | [VALL-E](https://arxiv.org/abs/2301.02111): autoregressive first codebook, non-autoregressive rest |
| Production reference code | Hugging Face [Llama](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py), [Mistral](https://github.com/huggingface/transformers/blob/main/src/transformers/models/mistral/modeling_mistral.py) and [ModernBERT](https://github.com/huggingface/transformers/blob/main/src/transformers/models/modernbert/modeling_modernbert.py) model files |
| Build-along | [Coding LLaMA 2 from scratch](https://www.youtube.com/watch?v=oM4VmoabDAI) (Umar Jamil), [Let's reproduce GPT-2](https://www.youtube.com/watch?v=l8pRSuU81PU) (Andrej Karpathy), [CS336 assignment 1](https://github.com/stanford-cs336/assignment1-basics) |
| Inference lecture | [Stanford CS336, Lecture 10: Inference](https://www.youtube.com/watch?v=EfM546A79aM) |

## Optional

- Books: [Speech and Language Processing, 3rd edition draft](https://web.stanford.edu/~jurafsky/slp3/) (Jurafsky and Martin; [transformers chapter](https://web.stanford.edu/~jurafsky/slp3/7.pdf)), [Understanding Deep Learning](https://udlbook.github.io/udlbook/) (Simon Prince, MIT Press; the transformers chapter), [Build a Large Language Model (From Scratch)](https://www.manning.com/books/build-a-large-language-model-from-scratch) (Sebastian Raschka, Manning; [code](https://github.com/rasbt/LLMs-from-scratch))
- [The Annotated Transformer](https://nlp.seas.harvard.edu/annotated-transformer/) and [Formal Algorithms for Transformers](https://arxiv.org/abs/2207.09238): the original architecture in code and in pseudocode
- Tokenization: [Stanford CS336, Lecture 1](https://www.youtube.com/watch?v=JuoVZkPBiKk) and [Let's build the GPT Tokenizer](https://www.youtube.com/watch?v=zduSFxRajkE) (Andrej Karpathy)
- Other lecture treatments: [Stanford CS224N](https://web.stanford.edu/class/cs224n/), [CMU Advanced NLP](https://cmu-l3.github.io/anlp-fall2025/)
- Refreshers: [Let's build GPT](https://www.youtube.com/watch?v=kCc8FmEb1nY) (Andrej Karpathy), [Attention is all you need, explained](https://www.youtube.com/watch?v=bCz4OMemCcA) (Umar Jamil), 3Blue1Brown on [transformers](https://www.youtube.com/watch?v=wjZofJX0v4M) and [attention](https://www.youtube.com/watch?v=eMlx5fFNoYc)
- [Self-attention, multi-head, causal and cross-attention in code](https://magazine.sebastianraschka.com/p/understanding-and-coding-self-attention) (Sebastian Raschka)
- [Differential Transformer](https://arxiv.org/abs/2410.05258): note what the doubled attention maps cost
- Alternative backbones: [Mamba](https://arxiv.org/abs/2312.00752), [Mamba-2 explained](https://tridao.me/blog/2024/mamba2-part1-model/), [RWKV](https://arxiv.org/abs/2305.13048), [xLSTM](https://arxiv.org/abs/2405.04517), and Albert Gu's talk [On the Tradeoffs of State Space Models and Transformers](https://www.youtube.com/watch?v=bHSDPgZYie0)
- Audio codec background: [AudioLM](https://arxiv.org/abs/2209.03143), [SoundStream](https://arxiv.org/abs/2107.03312)
- Lookup references: [x-transformers](https://github.com/lucidrains/x-transformers/blob/main/x_transformers/x_transformers.py), [minGPT](https://github.com/karpathy/minGPT/blob/master/mingpt/model.py), [annotated paper implementations](https://nn.labml.ai/)

## Labs

### Lab 1B.1 — Attention-head structure against KV cache

**Tools:** PyTorch; nanoGPT `model.py` as the base; `torch.profiler`.

Write one attention module with a KV-head count parameter (1 is multi-query,
equal to the head count is multi-head), plus a latent-attention variant, each
with an explicit KV cache.

**Deliverable:**
- KV-cache bytes per token per layer, calculated and measured
- Decode tokens per second at batch 1, 8 and 32 with 1k, 4k and 16k context
- Peak GPU memory
- One paragraph on where decoding becomes memory-bandwidth-bound
- The same formula applied to a real model's `config.json`

### Lab 1B.2 — Positional scheme against extrapolation and cache reuse

**Tools:** nanoGPT on TinyStories or enwik8, with learned-absolute, sinusoidal,
rotary and ALiBi positions.

**Deliverable:**
- Validation loss at 1×, 2× and 4× the training context for each scheme
- A cache-reuse experiment: drop the first *k* tokens and test whether the
  remaining cached keys and values are still valid without recomputation. Report
  the maximum logit error against a full recompute for each scheme.
- Per-step overhead of applying rotary positions

### Lab 1B.3 — nanoGPT to a Llama-style block, one change at a time

**Tools:** nanoGPT; the Meta Llama `model.py` and nanochat `gpt.py` as targets.

Apply in order: norm placement → RMSNorm → GELU to SwiGLU at matched parameter
count → learned positions to rotary → grouped-query attention → bias removal.

**Deliverable:**
- An ablation table with one row per change: parameter count, FLOPs per token,
  validation loss at fixed steps, prefill time, decode tokens per second with the cache
- Final check: load real weights for a small open Llama-family model into your
  block and match the reference logits to 1e-4

### Lab 1B.4 — Mixture-of-experts routing cost

**Tools:** PyTorch; the DeepSeek-V3 `Gate`, `Expert` and `MoE` classes as reference.

Replace the dense MLP with a top-*k*-of-*N* routed layer (N of 8 and 64, k of 2
and 8), once at equal active parameters and once at equal total parameters.

**Deliverable:**
- Tokens per second and peak memory against dense, at batch 1 and batch 64
- A histogram of expert load per batch, and the share of time in routing and
  gather/scatter against the expert matrix multiplies
- A naive per-expert loop compared with grouped dispatch
- A short note on what batch-level expert imbalance does to latency

### Optional Lab 1B.5 — Codebook decoding patterns for speech

**Tools:** audiocraft `codebooks_patterns.py`; CSM `models.py` or Moshi `lm.py`
as reference. Random weights are enough for cost measurement.

For 8 to 32 codebooks at 12.5 frames per second, compare flattened
autoregressive decoding, the delay pattern, and a backbone with a small depth transformer.

**Deliverable:**
- Backbone steps and total forward passes per second of audio, and KV-cache
  growth per second of audio
- Real-time factor and time to first audio on one GPU
- The streaming latency implied by the delay pattern

## Checkpoint

1. A model has 32 layers, 32 attention heads, 8 KV heads and a head dimension of 128. How many bytes of KV cache does one token cost in half precision? What would it cost with full multi-head attention?
2. With a sliding context window, why can cached keys be reused under rotary positions but not under learned absolute positions?
3. A mixture-of-experts model has 8 experts and routes each token to 2. What fraction of MLP parameters sits in GPU memory, and what fraction does compute per token touch?
4. Why does a diffusion transformer not benefit from a KV cache, and what is the equivalent lever?
