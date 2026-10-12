# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""UVA-based CPU offloading using Unified Virtual Addressing."""

from collections.abc import Generator
import os

import torch
import torch.nn as nn
from torch.func import functional_call

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.model_executor.offloader.base import BaseOffloader, should_pin_memory
from vllm.utils.mem_utils import format_gib
from vllm.utils.platform_utils import is_uva_available
from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor

logger = init_logger(__name__)


class UVAOffloader(BaseOffloader):
    """Offloader using Unified Virtual Addressing (UVA) for zero-copy access.

    This offloader moves parameters to pinned CPU memory and creates CUDA views
    using UVA. The GPU can then directly access the CPU memory without explicit
    transfers, at the cost of PCIe bandwidth (slower than GPU memory).

    When UVA is disabled via env var, falls back to a functional_call-based
    approach that moves parameters on-demand.

    Args:
        cpu_offload_max_bytes: Maximum bytes to offload to CPU.
        cpu_offload_params: Set of parameter name segments to selectively
            offload. If empty, all parameters are eligible up to the byte limit.
    """

    supports_tower_offload = True

    def __init__(
        self,
        cpu_offload_max_bytes: int,
        cpu_offload_params: set[str] | None = None,
    ):
        self.cpu_offload_max_bytes = cpu_offload_max_bytes
        self.cpu_offload_bytes = 0
        self.cpu_offload_params = cpu_offload_params or set()

        self.pin_memory = should_pin_memory()
        self.uva_offloading = (
            is_uva_available() and not envs.VLLM_WEIGHT_OFFLOADING_DISABLE_UVA
        )
        # torch.Tensor.pin_memory() uses PyTorch's CUDA host caching allocator.
        # Its large allocation bins round a 576 MiB EXL3 trellis bank to 1 GiB
        # and a 288 MiB bank to 512 MiB.  A model with hundreds of such banks
        # therefore consumes ~1.78x the requested host bytes.  This opt-in path
        # allocates exact-sized pageable storage and registers it in place with
        # CUDA.  It is valid only while constructing parameters whose contents
        # will immediately be replaced by the checkpoint loader.
        self.exact_host_register = os.environ.get(
            "VLLM_UVA_OFFLOAD_EXACT_HOST_REGISTER", "0"
        ).lower() in ("1", "true", "yes")
        self._registered_cpu_tensors: list = []
        self._diag_interval_bytes = int(
            float(os.environ.get("VLLM_UVA_OFFLOAD_DIAG_GIB", "0")) * 1024**3
        )
        self._diag_next_bytes = self._diag_interval_bytes

    def _maybe_log_diagnostics(self, name: str, tensor_bytes: int) -> None:
        if (
            self._diag_interval_bytes <= 0
            or self.cpu_offload_bytes < self._diag_next_bytes
        ):
            return

        proc_stats: dict[str, str] = {}
        try:
            with open(f"/proc/{os.getpid()}/smaps_rollup") as handle:
                for line in handle:
                    if ":" in line:
                        key, value = line.split(":", 1)
                        proc_stats[key] = value.strip()
        except OSError as exc:
            proc_stats["error"] = repr(exc)

        import torch

        logger.warning(
            "UVA_DIAG requested=%.3fGiB cap=%.3fGiB last=%s "
            "last_bytes=%d rss=%s pss_shmem=%s private_dirty=%s "
            "cuda_alloc=%.3fGiB cuda_reserved=%.3fGiB",
            self.cpu_offload_bytes / 1024**3,
            self.cpu_offload_max_bytes / 1024**3,
            name,
            tensor_bytes,
            proc_stats.get("Rss", "?"),
            proc_stats.get("Pss_Shmem", "?"),
            proc_stats.get("Private_Dirty", "?"),
            torch.cuda.memory_allocated() / 1024**3,
            torch.cuda.memory_reserved() / 1024**3,
        )
        while self._diag_next_bytes <= self.cpu_offload_bytes:
            self._diag_next_bytes += self._diag_interval_bytes

    def wrap_modules(
        self,
        modules_generator: Generator[nn.Module, None, None],
        prefix: str = "",
    ) -> list[nn.Module]:
        """Wrap modules with UVA offloading."""
        if prefix:
            prefix = f"{prefix}."
        modules = [
            self._maybe_offload_to_cpu(module, prefix) for module in modules_generator
        ]
        if self.cpu_offload_bytes > 0:
            logger.info(
                "Total CPU offloaded parameters: %s",
                format_gib(self.cpu_offload_bytes),
            )
        return modules

    def _maybe_offload_to_cpu(self, module: nn.Module, prefix: str = "") -> nn.Module:
        """Offload module parameters to CPU using UVA if budget allows."""
        if (params := next(module.parameters(), None)) is None:
            return module

        device = params.device

        if device == torch.device("cpu"):
            return module

        if self.cpu_offload_bytes >= self.cpu_offload_max_bytes:
            return module

        # offload parameters to CPU
        # use pin_memory if possible, which helps cudagraph capture speed
        offloaded_parameters = False
        for name, p in module.named_parameters():
            if self.cpu_offload_bytes >= self.cpu_offload_max_bytes:
                # we use per-parameter offloading
                # one module might have some parameters offloaded and some not
                break

            # Skip parameters an earlier wrap_modules call already offloaded.
            # The UVA path leaves p.device as the accelerator (a view of CPU
            # memory), so the marker is the only way to recognize those.
            if p.device.type == "cpu" or getattr(p, "_vllm_is_uva_offloaded", False):
                continue

            if self.cpu_offload_params:
                # Check if parameter belongs to the offloading set
                # Add dots here to ensure we match full segments only
                # e.g., "experts.w2_weight" matches "mlp.experts.w2_weight"
                # but not "mlp.experts.w2_weight_scale"
                should_offload = any(
                    f".{param}." in f".{prefix}{name}."
                    for param in self.cpu_offload_params
                )
                if not should_offload:
                    continue

            if self.exact_host_register:
                if not self.uva_offloading:
                    raise RuntimeError(
                        "VLLM_UVA_OFFLOAD_EXACT_HOST_REGISTER requires UVA"
                    )
                if not p.data.is_contiguous():
                    raise RuntimeError(
                        "Exact host registration currently requires contiguous "
                        f"parameters; got {prefix}{name} stride={p.data.stride()}"
                    )
                cpu_data = torch.empty(
                    p.data.shape, dtype=p.data.dtype, device="cpu"
                )
                tensor_bytes = cpu_data.numel() * cpu_data.element_size()
                if tensor_bytes:
                    status = torch.cuda.cudart().cudaHostRegister(
                        cpu_data.data_ptr(), tensor_bytes, 2
                    )
                    if int(status) != 0:
                        raise RuntimeError(
                            "cudaHostRegister failed for exact UVA allocation "
                            f"{prefix}{name} bytes={tensor_bytes}: {status}"
                        )
                # Keep the registered CPU storage alive for exactly as long as
                # this offloader/model. Worker process teardown releases the
                # CUDA registrations; failed loads also terminate the worker.
                self._registered_cpu_tensors.append(cpu_data)
            else:
                cpu_data = p.data.to(device="cpu")
                if self.pin_memory:
                    cpu_data = cpu_data.pin_memory()

            if not self.uva_offloading:
                p.data = cpu_data
            else:
                p.data = get_accelerator_view_from_cpu_tensor(cpu_data)
                p._vllm_is_uva_offloaded = True

            self.cpu_offload_bytes += p.data.numel() * p.data.element_size()
            self._maybe_log_diagnostics(
                f"{prefix}{name}", p.data.numel() * p.data.element_size()
            )
            offloaded_parameters = True

        if offloaded_parameters and not self.uva_offloading:
            original_forward = module.forward

            def forward(*args, **kwargs):
                module.forward = original_forward
                device_state = {
                    # here we blindly call `to(device)`
                    # if the parameter is already on the device,
                    # it will be a no-op
                    k: v.to(device, non_blocking=True)
                    for k, v in module.state_dict().items()
                }

                # set `tie_weights=False` as tied weights in original model
                # become untied when calling .to(device) individually
                output = functional_call(
                    module,
                    device_state,
                    args=args,
                    kwargs=kwargs,
                    tie_weights=False,
                )
                module.forward = forward
                return output

            module.forward = forward

        return module
