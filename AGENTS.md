# Engineering notes

Read this before changing kernels, cache plans, launch settings or benchmarks. It records what the published
numbers depend on and the failure modes that cost the most time.

## Ground truth

- Checkpoint: `turboderp/GLM-5.3-Flash-exl3`, branch `3.05bpw`, revision
  `332ab457b709b7ba30dd9a448be5de03b80a7ac9`, weights unchanged. The upstream index omits the 22 k-pool tensors of
  the sparse-attention indexer; `scripts/install_model.py` adds them. Without that the indexer silently runs
  without its weights and long-context retrieval quality collapses while speed looks better.
- Runtime: the vendor vLLM image pinned in `repro.lock.json` plus three overlays (`runtime/`). Regenerate them only
  with `scripts/build_overlays.py`; never hand-edit a manifest.
- Validated hardware: 2x RTX 3090 24 GB (SM86), Threadripper PRO 5975WX, 128 GB DDR4-3200 8-channel.
- Default profile: `configs/2x3090-256k-agent.env`, maximum context 262144.

## Invariants

1. **Host memory is the binding constraint.** About 110 GiB is pinned while serving 256K. Anything that keeps a
   second host copy of expert banks (an unpinned checkpoint, a debug dump, PyTorch's caching host allocator, which
   rounds 576 MiB banks up to 1 GiB) runs a 128 GB machine out of memory. Keep
   `VLLM_UVA_OFFLOAD_EXACT_HOST_REGISTER=1`.
2. **CPU expert pools own whole cores.** They synchronise with spin barriers. Any other busy thread on a pool core
   or its SMT sibling stalls the whole pool (1.8-4x slower decode). vLLM's own processes are confined to
   `HYBRID_MAIN_CPUS` with `taskset`; keep it that way when adding threads.
3. **The shared expert stays on the main stream** (`VLLM_DISABLE_SHARED_EXPERTS_STREAM=1`). EXL3 linears are
   cooperative full-grid kernels sharing a workspace with the routed experts; running the shared expert on a
   second stream deadlocked both GPUs in the image (first decode step after a 32K prefill, every time) and is
   timing-dependent elsewhere.
4. **Cache plans are checkpoint- and budget-specific.** `configs/lru-plan-2x3090-13312m-k3.json` assigns per-layer
   capacities for a 13,312 MiB per-GPU expert budget next to a 262,144-token KV cache with 8,192-token chunks. A
   larger plan with 8K chunks ran out of GPU memory during prefill.
5. **Correctness gates before speed claims.** Run `scripts/longctx_gate.py` (needles at 32K/131K/260K and the
   coherence check) on every candidate; any kernel or numerics change also needs a teacher-forced NLL comparison
   (see `docs/correctness.md`). The model amplifies ULP-level noise, so do not judge an A/B from sampled text or
   per-position KL.

## Failure modes seen

| Symptom | Cause |
|---|---|
| Fast but wrong long-context answers | k-pool tensors missing from the index (invariant above) |
| Decode hangs, GPUs at 100% / ~180 W, CPU pools idle, `RPC call to sample_tokens timed out` after 300 s | Shared expert on a second stream (invariant 3) |
| Engine hang during long agent runs, CPU busy, GPUs idle | Race in the CPU MoE pool's dispatch (fixed: generation and worker count published in one atomic word) |
| Agent loops on tiny tool calls, zero reasoning | Client dropped reasoning from history; preserved reasoning fixes it (`docs/agents.md`) |
| A worker OOM-killed at load or while serving | Host memory: other processes, a full RAM-backed `/tmp`, or a second expert copy |
| CPU at 90 C after an hour of agent load | Sustained 24-core pool load; agent runs at a 230 W CPU package limit still decoded at about 37-38 tok/s |

## How to run a useful experiment

- Change one thing; keep the profile, prompt shapes and image identical otherwise.
- Use `scripts/benchmark_serving.py` shapes (8,192 / 131,072 / 260,096 input, 2,048 output) with fresh
  `cache_salt`, and report time to first token, input tok/s and decode tok/s separately.
- Speed depends on which experts a prompt activates. Compare candidates on the same prompts, not against numbers
  measured with other prompts.
- Watch temperatures: abort sustained runs at CPU Tctl >= 90 C or GPU >= 82 C.

## Files to inspect

| Topic | File |
|---|---|
| EXL3 integration, expert caches, pinning, hybrid decode, prefill staging | `runtime/vllm-overlay/model_executor/layers/quantization/exl3.py` |
| GLM-5.3 model, mHC, sparse attention | `runtime/vllm-overlay/models/glm5next/nvidia/` |
| Bounded reasoning, minimum reasoning | `runtime/vllm-overlay/v1/worker/gpu/sample/thinking_budget.py` |
| Preserved reasoning | `runtime/vllm-overlay/entrypoints/openai/chat_completion/serving.py` |
| Strict tool grammar | `runtime/vllm-overlay/parser/glm47_moe.py` |
| CPU expert pools and bridge | `runtime/glm-hybrid/csrc/moe_mul1.cpp`, `runtime/glm-hybrid/csrc/hybrid_bridge.cu` |
| EXL3 MoE GEMV kernels | `runtime/exllamav3-overlay/exllamav3/exllamav3_ext/quant/exl3_moe_gemv.cu` |
| CPU plan | `runtime/glm53_host.py` |
