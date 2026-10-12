# Performance

Read every published number with its request shape. This runtime serves one request at a time; its speed depends
on prompt length, on how many routed experts of a step are already in VRAM, on PCIe bandwidth (prefill) and on
CPU memory bandwidth (decode).

## Protocol

`scripts/benchmark_serving.py` is the public client (shared with
[qwen38-flash-next-2x3090](https://github.com/DominikBucko/qwen38-flash-next-2x3090)):

- token-ID prompts built by repeating a fixed public seed text through the server's own tokenizer, so the prompt
  length is exact;
- `/v1/completions`, streaming, greedy, `ignore_eos=true`, so every run produces exactly the requested output;
- a fresh random `cache_salt` per request, so prefix caching cannot shortcut prefill (model weights, expert caches
  and CUDA graphs stay warm);
- **input tok/s** = prompt tokens / time to first token (includes queueing and the first decode step);
- **decode tok/s** = (output tokens - 1) / time after the first token, as observed through the API.

```bash
python3 scripts/benchmark_serving.py --input-tokens 131072 --output-tokens 2048 --runs 2 --warmup 0 \
  --output result.json
```

Standard shapes: 8,192 + 2,048, 131,072 + 2,048, and 260,096 + 2,048 (the full 262,144-token window).

`scripts/longctx_gate.py` runs the long-context correctness gates (see [correctness.md](correctness.md)).

## Reading the numbers

- Prefill throughput falls slowly with context length because the sparse-attention indexer scores every cached
  token. Most of the cost is copying cold experts over PCIe: an 8,192-token chunk uses almost every routed expert.
- Decode speed depends on the prompt: experts used by the prompt's last chunk are in VRAM when decoding starts,
  and the LRU adapts within a few dozen tokens. Short synthetic prompts are not representative of agent traffic.
- Agent turns are dominated by prefix-cache hits and decode; see [agents.md](agents.md) for measured turn times.
