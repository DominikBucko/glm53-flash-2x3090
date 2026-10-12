# Runtime snapshot

This directory is copied from the tagged GitHub release of
[glm53-flash-2x3090](https://github.com/DominikBucko/glm53-flash-2x3090). It records the exact vLLM base image,
overlay hashes, the launch profile and the checkpoint lock.

Recommended use is to clone the matching GitHub tag, run `scripts/download_model.sh` (it downloads turboderp's
EXL3 3.05 bpw files at the pinned revision, verifies them against `model/checkpoint.lock.json` and links the k-pool
auxiliary shard into the index), then `make serve` or the prebuilt image named in the release notes.

```text
2x RTX 3090, TP2, 262,144-token context, 8,192-token prefill chunks, routed experts in host memory with a
per-layer VRAM LRU (20% pinned), CPU expert pools for decode, bounded reasoning (4,096 tokens),
preserved reasoning for the last 4 tool-calling turns, prefix caching.
```
