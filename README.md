# GLM-5.3-Flash on 2× RTX 3090

<h2 align="center">Up to 1,155 tok/s prefill · up to 50 tok/s decode</h2>
<p align="center"><strong>262,144-token context · 2× RTX 3090 (24 GB) · 128 GB system memory</strong></p>
<p align="center">One request: 131,072 input tokens in 113.5 s with the <a href="configs/prefill-256k.env">prefill profile</a>; 47–50 tok/s decode on the benchmark prompts (37.7 tok/s averaged over real agent turns). The default <a href="configs/2x3090-256k-agent.env">agent profile</a> adds prefix caching, so follow-up turns only prefill what is new.</p>
<p align="center"><a href="https://huggingface.co/albucino/GLM-5.3-Flash-EXL3-3.05bpw"><strong>Model card</strong></a> · <a href="benchmarks/2026-10-12/README.md">Benchmark report</a> · <a href="docs/agents.md">Agents</a></p>

[GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash) is a 320B-parameter mixture-of-experts model
with 18B active parameters, 288 routed experts per layer and a 262,144-token context. This runtime serves
turboderp's [EXL3 3.05 bpw quantization](https://huggingface.co/turboderp/GLM-5.3-Flash-exl3/tree/3.05bpw)
(117 GiB, weights unchanged) on two 24 GB GPUs: routed experts live in system memory, each GPU caches the active
ones per layer in VRAM, and a CPU thread pool per GPU computes the cold experts during decode.

The serving code is a pinned vLLM image plus the GLM-5.3 support of
[spele1100/vllm-glm53](https://github.com/spele1100/vllm-glm53), a patched
[ExLlamaV3](https://github.com/turboderp-org/exllamav3) and this repository's kernels, all as checksummed overlays.

## Results

Two RTX 3090s, Threadripper PRO 5975WX, 128 GB DDR4-3200 (8 channels). One request, best of 2, from the
published image:

| Profile | Input + output tokens | First token | Input tok/s | Decode tok/s |
|---|---:|---:|---:|---:|
| Prefill | 8,192 + 2,048 | 7.3 s | 1,125 | 50.4 |
| Prefill | 131,072 + 2,048 | **113.5 s** | **1,155** | 49.4 |
| Prefill | 260,096 + 2,048 | **235.3 s** | 1,106 | 48.2 |
| Agent (default) | 8,192 + 2,048 | 9.5 s | 859 | **50.4** |
| Agent | 131,072 + 2,048 | 236.2 s | 555 | 49.3 |
| Agent | 260,096 + 2,048 | 506.7 s | 513 | 48.1 |

Input tok/s is prompt tokens / time to first token, with a fresh `cache_salt` so nothing is reused. The agent
profile's prefix caching makes cold prefill of long prompts about 2x slower, but in a 15-task agent benchmark
72.9% of all prompt tokens came from the cache and a tool-call turn took 7.6 s (median). The benchmark prompt
routes to a small set of experts; real agent turns decoded at 37.7 tok/s on average. Long-context retrieval
(three needles at 32K, 131K and 260K tokens, exact JSON) and a coherence check passed in both profiles. See the
[benchmark report](benchmarks/2026-10-12/README.md).

## Agents

GLM-5.3-Flash is a reasoning model, and at 3 bits behind a typical agent loop it tends to stop reasoning and
repeat tool calls. The default profile fixes the two causes found:

- **Bounded reasoning**: a 4,096-token thinking budget per turn, closed with a fixed sentence when exhausted.
  OpenAI-style `reasoning_effort` values (`minimal` … `xhigh`) are mapped to GLM's `low` / `high` / `max`.
- **Preserved reasoning**: most clients send earlier assistant turns back without their reasoning, GLM then
  imitates the empty `<think></think>` blocks and loops. The server re-attaches the reasoning of the last 4
  tool-calling turns.

On an internal 15-task engineering-agent benchmark this scored 9/15 strict passes, mean 94.6 (single run).
Two more switches are available but not on by default; see [docs/agents.md](docs/agents.md).

## How it fits

| Where | What |
|---|---|
| 2× 24 GB GPUs | Attention, mHC, router, shared expert, LM head (tensor parallel); per-layer LRU of routed experts, 20% of it pinned; KV cache for 262,144 tokens (3.1 GiB per GPU) |
| 128 GB RAM | Every routed expert not pinned on a GPU, one exact-size registered copy per GPU half (~110 GiB pinned) |
| CPU | One expert pool per GPU computing cold experts during decode, planned from the host topology |

Prefill streams cold experts to the GPUs over PCIe in 8,192-token chunks. Details: [docs/architecture.md](docs/architecture.md).

## Run it

You need Linux, Docker with the NVIDIA Container Toolkit, two 24 GB RTX 3090s, 128 GB RAM and about 120 GB of
disk for the checkpoint. A many-core CPU with several memory channels matters for decode ([docs/hardware.md](docs/hardware.md)).

```bash
git clone https://github.com/DominikBucko/glm53-flash-2x3090 && cd glm53-flash-2x3090
scripts/download_model.sh /models/glm53-flash-exl3-3.05bpw   # turboderp's files at the pinned revision, verified
cp .env.example .env                                          # set MODEL_DIR
make build-image && make preflight && make serve
```

`download_model.sh` checks every file against `runtime/model/checkpoint.lock.json` and links the sparse-attention
indexer's auxiliary shard into the index (`scripts/install_model.py`; the upstream index omits it). The server
listens on `http://127.0.0.1:8000/v1` (OpenAI-compatible, model `GLM-5.3-Flash`, tool calling and reasoning
enabled). The first start takes about 6 minutes; Triton and TileLang kernels compile during the first requests
(set `JIT_CACHE_DIR` in `.env` to keep them).

To use the prebuilt image instead of building, set `IMAGE=ghcr.io/dominikbucko/glm53-flash-2x3090@sha256:<digest>`
from the release notes and run `make preflight && IMAGE=... ./scripts/docker_serve.sh`.

### Profiles

| File | Use |
|---|---|
| [`configs/2x3090-256k-agent.env`](configs/2x3090-256k-agent.env) | Default, baked into the image: 262,144 tokens, prefix caching, bounded + preserved reasoning |
| [`configs/prefill-256k.env`](configs/prefill-256k.env) | One-shot long prompts: prefix caching off, about 2x faster cold prefill at 131K+ |
| [`configs/agent-experimental.env`](configs/agent-experimental.env) | Strict tool-call grammar and minimum reasoning (opt-in, see [docs/agents.md](docs/agents.md)) |

Add a profile with `cat configs/<profile>.env >> .env`. If memory runs out, see [docs/memory.md](docs/memory.md).

## Data and caveats

- Validated on one machine. Decode depends on CPU memory bandwidth and core count, prefill on PCIe bandwidth;
  other hardware will differ.
- One request at a time; no speculative decoding.
- The image decodes about 5% slower than the native research build it was made from (prefill matches); see the
  benchmark report.
- The weights are a 3.05 bpw quantization; no claim is made about quality relative to the BF16 model beyond the
  gates and the agent benchmark. Agent benchmark fixtures and traces are not published.
- None of the results is a speed guarantee.

Engineering notes for contributors: [AGENTS.md](AGENTS.md). Correctness methodology:
[docs/correctness.md](docs/correctness.md).

## License

The runtime code is Apache-2.0 ([LICENSE](LICENSE)). The overlays carry vLLM / spele1100/vllm-glm53 (Apache-2.0)
and ExLlamaV3 (MIT) code; GLM-5.3-Flash and turboderp's quantization are MIT-licensed. See
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
