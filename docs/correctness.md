# Correctness

The weights are turboderp's EXL3 3.05 bpw files, unchanged. Everything this repository changes is in the runtime:
kernels, caching, scheduling and serving. Those changes were checked in two ways.

## While developing: exact and teacher-forced comparisons

The research build has a deterministic mode (fixed-point int64 accumulation for the routed experts, sorted
sparse-attention top-k, pinned launch geometry) in which a forward pass is bitwise reproducible. Changes that
should not alter arithmetic, such as CUDA graphs, the dynamic expert cache, expert pinning, DMA staging and the
hybrid CPU path, were required to produce bitwise identical logits to the static reference in that mode. Changes
that do alter arithmetic, such as fused kernels, merged projections and bf16 I/O, were compared by mean
teacher-forced negative log-likelihood against a captured reference, with a noise floor of about +-0.04 measured
between repeated production-mode runs. The ExLlamaV3 reference implementation agreed with this runtime per
position (correlation 0.984).

The model amplifies tiny numerical differences: two non-deterministic runs of the same build differ by a KL of
about 0.4 per position. Comparisons therefore use teacher-forced NLL, not per-position KL or sampled text.

## Every release: long-context gates

`scripts/longctx_gate.py` runs against the served endpoint:

- **needles** at 32,768, 131,072 and 260,000 prompt tokens: a seeded synthetic log with three facts at 5%, 50%
  and 95% depth; the model must return exactly `{"alpha": ..., "middle": ..., "omega": ...}`;
- **coherence**: 1,024 greedy tokens of an essay; the unique 4-gram ratio must stay at or above 0.9.

The results for the published image are in [`benchmarks/`](../benchmarks). The gates catch broken attention,
cache corruption and degenerate decoding; they are not a quality benchmark.

## Known limits

- The 3.05 bpw quantization itself costs quality relative to the BF16 model; no claim is made about that gap.
- The bounded reasoning budget (4,096 tokens) and the forced closing sentence change outputs on problems that
  need longer reasoning.
- Preserved reasoning changes the prompt the model sees in agent conversations (it adds back the model's own
  earlier reasoning); see [agents.md](agents.md).
