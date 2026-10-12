# Architecture

GLM-5.3-Flash has 45 layers: 34 KDA linear-attention layers and 11 sparse MLA (DSA) layers whose top-k token
selection comes from a k-pool indexer, mHC hyper-connections (4 streams, 20 Sinkhorn iterations) between layers,
and 288 routed experts per MoE layer with top-8 sigmoid routing. The EXL3 3.05 bpw checkpoint is about 117 GiB, so
on two 24 GB GPUs most routed expert weights must live in system memory. Everything below runs as one vLLM
process group with tensor parallelism 2 (no expert parallelism) and one request at a time.

## Where the weights live

| Part | Location |
|---|---|
| Attention, mHC, router, shared expert, embeddings (UVA), LM head | GPU, split by tensor parallelism |
| Hottest 20% of each layer's LRU capacity of routed experts ("pinned") | GPU, permanently; no host copy |
| Other routed experts | Host memory: one exact-size registered (UVA) copy per GPU half |
| Per-layer LRU cache of routed experts | GPU; capacity per layer from `configs/lru-plan-2x3090-13312m-k3.json` |
| KV cache (MLA latent + indexer k-pool) and KDA recurrent state | GPU, 3.1 GiB per GPU for 262,144 tokens |

Pinning happens at load time (`EXL3_PINNED_FRAC=0.2`): the hottest experts of the seed ranking
(`configs/hot-expert-rankings.json`) are written straight into VRAM slots outside the LRU, and the host tensors
are created without their rows. That is what leaves about 6 GB of a 128 GB machine free while serving 256K.

## Decode: one token

1. Attention and mHC run on the GPUs (fused mHC post/pre-norm, sparse MLA with 32 KV splits, fused router top-k).
2. For each MoE layer the router picks 8 experts per token. Experts in the GPU cache (pinned or LRU) run on the
   GPU with the EXL3 MoE GEMV kernels.
3. Cold experts are split: the GPU takes 30% of them (`EXL3_HYBRID_GPU_FRAC`), fetching their trellises from host
   memory over PCIe into the LRU, and the rest go to the CPU. A graph-safe bridge (`runtime/glm-hybrid`) writes
   the activations to pinned host memory and bumps a sequence number; a CPU thread pool decodes the EXL3 trellis
   of those experts on the fly (AVX2) and writes the result back; the GPU adds it to the layer output.
4. Each GPU has its own CPU pool. `runtime/glm53_host.py` gives each pool one thread per physical core on every
   CCD and keeps the pool cores' SMT siblings idle, because the pools synchronise with spin barriers.

The whole decode step is captured as one CUDA graph (`FULL_DECODE_ONLY`).

## Prefill

Prompts are processed in 8,192-token chunks. For each MoE layer every routed expert used by the chunk must run on
the GPU; experts not in VRAM are staged over PCIe eight at a time into a ring of temporary slots and run with the
batched EXL3 kernels. Prefill speed is bounded by host-to-GPU bandwidth: on the validation machine both GPUs sit
behind one PCIe root complex, which caps concurrent transfers at about 18.4 GB/s per GPU.

## Long context

The 11 DSA layers keep an FP8 k-pool index of every token and attend only to the tokens the indexer selects, so
attention cost stays roughly flat with context length. The KDA layers keep a fixed-size recurrent state. A
262,144-token window needs 3.1 GiB of KV cache per GPU.

## Agents and reasoning

- **Bounded reasoning.** Every request gets a thinking budget of 4,096 tokens. When it runs out, the runtime
  forces "I have thought about this enough; I will now act on my best plan." and `</think>`.
  `configs/chat-template-bounded.jinja` is the upstream template with OpenAI-style `reasoning_effort` names
  (`minimal`, `medium`, `xhigh`, ...) mapped to GLM's `low` / `high` / `max`.
- **Preserved reasoning.** Most agent clients send earlier assistant turns back without their reasoning. GLM's
  template then renders them as empty `<think></think>` blocks, and the model imitates that: it stops thinking and
  loops. The server caches each tool-calling response's reasoning by tool-call id and re-attaches it to the last 4
  assistant turns of later requests (`GLM53_PRESERVE_REASONING_TURNS=4`).
- **Prefix caching** reuses the KV of an unchanged conversation prefix, so a tool-call turn only prefills the new
  tool output.

See [agents.md](agents.md) for measurements and the experimental switches.

## Components

The base image is the vLLM vendor build tagged `qwen38-flash-next` (vLLM `0.1.dev20073+g8e685d198`), pinned by
digest. It is used for its compiled vLLM extensions and Python environment, which match those the runtime was
developed with byte for byte; its vLLM Python package is replaced by the overlay below, so nothing Qwen-specific
runs.

| Path | What it is |
|---|---|
| `runtime/vllm-overlay/` | The vLLM Python package files that differ from the pinned vendor image: GLM-5.3 model support from [spele1100/vllm-glm53](https://github.com/spele1100/vllm-glm53) plus this project's EXL3 integration (`model_executor/layers/quantization/exl3.py`), kernels and serving changes. Installed over the image's vLLM; the compiled vLLM extensions are the image's own. |
| `runtime/exllamav3-overlay/` | 28 files over ExLlamaV3 `499890c`: EXL3 MoE GEMV decode kernels, bf16 in/out, deterministic MoE accumulation, LRU gather/prefetch kernels and the CPU MoE path. |
| `runtime/glm-hybrid/` | `glm_hybrid` (CPU expert pools, bridge kernels, fused split + LRU update) and `glm_fast` (fused mHC, router and top-k kernels). |
| `runtime/glm53_host.py` | CPU pool planner. |
| `runtime/model/checkpoint.lock.json` | Size and SHA-256 of every file of the pinned upstream checkpoint. |
