---
license: mit
base_model:
  - zai-org/GLM-5.3-Flash
  - turboderp/GLM-5.3-Flash-exl3
pipeline_tag: text-generation
tags:
  - glm-5.3
  - exl3
  - exllamav3
  - vllm
  - 256k-context
  - rtx-3090
  - dual-gpu
  - 24gb-vram
  - 128gb-ram
  - consumer-gpu
  - cpu-offload
  - moe
  - local-llm
  - agents
---

# GLM-5.3-Flash on two RTX 3090s with 128 GB RAM: EXL3 3.05 bpw, 256K context

Run **GLM-5.3-Flash locally** on two 24 GB GPUs with its full 262,144-token context. This page holds the model
card, the runtime snapshot and a checksum lock. The weights are turboderp's
[EXL3 3.05 bpw quantization](https://huggingface.co/turboderp/GLM-5.3-Flash-exl3/tree/3.05bpw), used unchanged
from revision `332ab457b709b7ba30dd9a448be5de03b80a7ac9`; the runtime downloads and verifies them.

**Runtime: [github.com/DominikBucko/glm53-flash-2x3090](https://github.com/DominikBucko/glm53-flash-2x3090)**

| Two RTX 3090s + 128 GB RAM, one request, best of 2 | First token | Input tok/s | Decode tok/s |
|---|---:|---:|---:|
| 131,072 + 2,048, prefill profile | **113.5 s** | **1,155** | 49.4 |
| 260,096 + 2,048, prefill profile | 235.3 s | 1,106 | 48.2 |
| 131,072 + 2,048, agent profile (prefix caching) | 236.2 s | 555 | 49.3 |
| 8,192 + 2,048, agent profile | 9.5 s | 859 | 50.4 |

Decode figures come from the benchmark's repeated seed prompt; real agent turns decoded at 37.7 tok/s on average.
With prefix caching (the default agent profile) cold prefill of long prompts is about 2x slower, but follow-up
turns only prefill new tokens: in a 15-task agent benchmark 72.9% of prompt tokens came from the cache. Three
needles at 32K, 131K and 260K tokens were retrieved as exact JSON in both profiles.
[Full report](https://github.com/DominikBucko/glm53-flash-2x3090/blob/main/benchmarks/2026-10-12/README.md).

## What the runtime adds

- **Fits 128 GB.** Routed experts live in host memory with one exact-size registered copy per GPU half; the
  hottest 20% of each layer's GPU cache are pinned in VRAM and get no host copy at all.
- **CPU + GPU decode.** Each GPU keeps a per-layer LRU of routed experts in VRAM. Cold experts are split: the GPU
  fetches 30% over PCIe, a CPU thread pool per GPU computes the rest directly from the EXL3 trellis, all inside
  one CUDA graph per decode step.
- **Prefill in 8,192-token chunks** with cold experts staged to the GPUs over PCIe.
- **Agent fixes.** Bounded reasoning (4,096 tokens), reasoning re-attached to the last 4 tool-calling turns when
  clients drop it (GLM otherwise stops thinking and loops), prefix caching, and an optional strict tool-call
  grammar.

## The one checkpoint fix

turboderp's 3.05 bpw branch ships the 22 sparse-attention indexer tensors of the 11 DSA layers in
`kpool_aux.safetensors` but does not list them in `model.safetensors.index.json`. A loader that discovers
tensors through the index then runs the indexer without them, which looks fast and breaks long-context retrieval.
`scripts/install_model.py` adds the 22 entries and checks the result against `runtime/model/checkpoint.lock.json`
(size and SHA-256 of all 29 upstream files). No weight file is modified.

## Setup

```bash
git clone https://github.com/DominikBucko/glm53-flash-2x3090 && cd glm53-flash-2x3090
scripts/download_model.sh /models/glm53-flash-exl3-3.05bpw      # ~117 GiB, verified
cp .env.example .env                                             # set MODEL_DIR
make build-image && make preflight && make serve                 # OpenAI-compatible API on 127.0.0.1:8000
```

Linux, Docker with the NVIDIA Container Toolkit, two RTX 3090s, 128 GB RAM. See the GitHub repository for the
prebuilt image, the hardware and memory guides, and the benchmark reports.

## Limitations and license

- Validated on one machine: 2x RTX 3090, Threadripper PRO 5975WX, 128 GB DDR4-3200 8-channel. Decode speed
  depends heavily on CPU memory bandwidth; prefill on PCIe bandwidth.
- One request at a time. No speculative decoding.
- The 3.05 bpw quantization costs quality relative to the BF16 model; no claim is made about that gap.
- GLM-5.3-Flash is MIT-licensed (Z.AI); turboderp's EXL3 quantization is MIT-licensed. The runtime is
  Apache-2.0. Review the upstream terms before use.
