# Contributing hardware reports, fixes, and benchmarks

The most useful contributions are independent reproductions, clear failure reports, and measured improvements.
You do not need to write CUDA code to help.

## Share a run on your hardware

Use the [hardware report form](https://github.com/DominikBucko/glm53-flash-2x3090/issues/new?template=hardware-report.yml).
Successful dual RTX 3090 reproductions, other GPUs and CPUs, and failed attempts all help establish what works. A
report on other hardware starts as community evidence; it does not become a validated profile by itself.

Include the GPU models, CPU (model, cores, CCDs), RAM capacity and channels, driver, PCIe topology
(`nvidia-smi topo -m`), runtime commit, image digest, checkpoint revision, environment overrides, prompt/output
token counts, warmup state, and measurements. Keep prefill, time to first token and decode speed separate. The
CPU expert pools make decode depend on the CPU and memory bandwidth, so the CPU plan printed at startup
(`[glm53-host] ...`) is part of the report. A completed short prompt does not prove full-context support.

Use public synthetic prompts (`scripts/benchmark_serving.py`, `scripts/longctx_gate.py`) and redact tokens,
private prompts, user paths, and machine identifiers from logs. See the [performance guide](docs/performance.md)
for the comparison protocol and the [hardware guide](docs/hardware.md) for compatibility boundaries.

## Ask a question or report a bug

Use [Discussions](https://github.com/DominikBucko/glm53-flash-2x3090/discussions) for setup questions and tuning
ideas. For crashes, hangs or incorrect output, use the
[bug report form](https://github.com/DominikBucko/glm53-flash-2x3090/issues/new?template=bug-report.yml). Check
the [memory guide](docs/memory.md) first for OOMs.

## Submit a pull request

For runtime, kernel, cache-plan or benchmark changes, read [AGENTS.md](AGENTS.md) first; it explains the coupled
memory, performance and correctness constraints. Regenerate overlay manifests with `scripts/build_overlays.py` and
keep `repro.lock.json` consistent.

Describe the problem, the resulting behaviour and the evidence. For performance changes, compare identical
workloads and record full-context behaviour as well as short decode. For kernel or numerics changes, include a
correctness check: the long-context gates at least, and a teacher-forced comparison for anything that changes
arithmetic. GPU-free CI cannot verify CUDA correctness or speed.

Before submitting source changes:

```bash
make validate
make test
for script in scripts/*.sh; do bash -n "$script" || exit; done
```
