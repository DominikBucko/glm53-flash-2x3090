# Memory

## Host memory

Loading pins about 110 GiB of host memory (registered with the CUDA driver, so it cannot be swapped). With the
default profile a 128 GB machine has about 6 GB left while serving. Before starting:

- Stop other large memory users, including other model servers.
- Make sure `/tmp` is not filled: on many distributions it is a RAM-backed tmpfs, and files there take memory
  away from the server.
- Swap does not help this runtime; the pinned memory cannot be swapped and swapping anything else under load
  slows the CPU expert pools.

If the host runs out of memory during load, the kernel OOM killer stops the largest process, which is usually one
vLLM worker; the container then exits.

## GPU memory (per GPU)

| Use | Size |
|---|---|
| Routed-expert cache, pinned experts included (`configs/lru-plan-2x3090-13312m-k3.json`) | 13 GiB budget |
| KV cache and KDA state for 262,144 tokens | 3.1 GiB |
| Attention, mHC, router, shared experts, LM head (tensor-parallel half), prefill staging, activations, CUDA graphs | the rest |

About 21.5 GiB of each 24 GiB card is in use after loading. The cache plan assigns each MoE layer a capacity
proportional to how often its experts miss; it is specific to this checkpoint and to a 13,312 MiB budget.

## Shorter contexts

`MAX_MODEL_LEN` can be lowered, but the KV budget (`KV_CACHE_MEMORY_BYTES`) and the expert cache plan were sized
together. Freed KV memory is not automatically given to the expert cache; a larger plan needs a new plan file and
measurement.
