# Hardware

## Validated

| Part | Validation machine |
|---|---|
| GPUs | 2x NVIDIA GeForce RTX 3090 24 GB (SM86), PCIe 4.0 x16 each, no NVLink |
| CPU | AMD Threadripper PRO 5975WX, 32 cores / 64 threads, 4 CCDs |
| Memory | 128 GB DDR4-3200, 8 channels |
| OS | Linux, Docker with the NVIDIA Container Toolkit |

The image is built for compute capability 8.6 only (`TORCH_CUDA_ARCH_LIST=8.6`). Other GPUs need a rebuild and
are not validated.

## What matters

**System memory: 128 GB.** The runtime keeps about 110 GiB pinned in host memory: one copy of every routed expert
that is not pinned on a GPU, per GPU half. The validation machine had about 6 GB free while serving. 96 GB or 64 GB
machines cannot load this profile; there is no lower-memory profile yet.

**CPU and memory bandwidth decide decode speed.** During decode the cold experts of every layer are computed by a
CPU thread pool per GPU, reading their weights at memory bandwidth. The validation numbers come from 8-channel
DDR4 and 2 x 12 Zen 3 cores. A desktop CPU with dual-channel memory will decode much slower. The pools are planned
from the host topology at startup (`runtime/glm53_host.py`): one thread per physical core, spread over all
L3 domains, SMT siblings of pool cores left idle. The plan is printed as `[glm53-host] ...` lines; set
`EXL3_HYBRID_CPUS_R0` / `EXL3_HYBRID_CPUS_R1` / `HYBRID_MAIN_CPUS` / `EXL3_HYBRID_THREADS` to override it.

**PCIe bandwidth decides prefill speed.** Prefill copies every cold expert of every chunk to the GPU. On the
validation machine both GPUs hang off the same PCIe root complex, which caps concurrent host-to-GPU copies at
about 18.4 GB/s per GPU (26.4 GB/s with one GPU copying). Two GPUs on separate root complexes should prefill faster;
x8 slots will prefill slower. Check with `nvidia-smi topo -m`.

**Cooling.** Long agent sessions keep 24 CPU cores and both GPUs busy for hours. On the validation machine the CPU
reached 90 C at a 280 W package limit during a multi-hour agent benchmark; at 230 W it peaked at 88 C and agent
turns still decoded at about 37-38 tok/s. The GPUs ran with a 275-300 W limit.

## Not validated

RTX 4090 (SM89) and RTX 5090 (SM120) need a rebuild of the image for their architecture; the GLM-5.3 vLLM base
(spele1100/vllm-glm53) was developed on Ada, but this runtime's EXL3 kernels and cache plans were only measured on
RTX 3090s. Intel CPUs, single-socket desktop CPUs and machines with less than 128 GB are untested. Reports are
welcome via the hardware report form.
