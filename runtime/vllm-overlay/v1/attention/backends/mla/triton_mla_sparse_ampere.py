# SPDX-License-Identifier: Apache-2.0
"""Portable sparse-MLA backend for GLM-5.3 on Ampere.

The upstream sparse CUDA backends use FA3/FlashMLA/FlashInfer kernels that
start at SM89 or SM90.  GLM-5.3's no-RoPE MLA can use vLLM's existing Triton
latent decode kernel instead: turn each selected token into a one-row page and
run the normal paged latent-attention kernel over that compact page table.
This preserves the trained top-k selection exactly and avoids dense decode.
"""

import os
from typing import Any, ClassVar

import torch

import vllm.envs as envs
from vllm.config.cache import CacheDType
from vllm.model_executor.layers.attention.sparse_mla_attention import (
    SparseMLACommonImpl,
)
from vllm.platforms import current_platform
from vllm.platforms.interface import DeviceCapability
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionLayer,
    MLAAttentionImpl,
    MultipleOf,
)
from vllm.v1.attention.backends.mla.flashattn_mla_sparse import (
    FlashAttnMLASparseMetadata,
    FlashAttnMLASparseMetadataBuilder,
)
from vllm.v1.attention.backends.mla.sparse_utils import (
    flat_kv_row_view,
    triton_convert_req_index_compact_ordered,
    triton_convert_req_index_to_global_index,
)
from vllm.v1.attention.backends.mla.triton_mla import _compute_num_kv_splits
from vllm.v1.attention.ops.triton_decode_attention import decode_attention_fwd



_GLM53_MLA_DUMP_N: dict[int, int] = {}
_GLM53_MLA_LAYER_NO: dict[int, int] = {}


def _glm53_mla_dump(impl, topk, rows, counts, md, kv_rows, q) -> None:
    dev = q.device.index
    key = id(impl)
    layer_no = _GLM53_MLA_LAYER_NO.setdefault(key, len(_GLM53_MLA_LAYER_NO))
    n = _GLM53_MLA_DUMP_N.get(key, 0)
    _GLM53_MLA_DUMP_N[key] = n + 1
    d = "/home/bot/llm/probe/mla_dump"
    os.makedirs(d, exist_ok=True)
    safe = rows.clamp(min=0).long()
    gathered = kv_rows[safe.reshape(-1)].reshape(*rows.shape, -1)
    blob = {
        "topk": topk.cpu(), "rows": rows.cpu(), "counts": counts.cpu(),
        "block_table": md.block_table.cpu(),
        "req_id_per_token": md.req_id_per_token[: q.shape[0]].cpu(),
        "block_size": md.block_size,
        "slot_mapping": getattr(md, "slot_mapping", None).cpu()
        if getattr(md, "slot_mapping", None) is not None else None,
        "kv": gathered.float().cpu(),
        "q": q.float().cpu(),
    }
    torch.save(blob, f"{d}/dev{dev}_mla{layer_no:02d}_{n}.pt")

class AmpereTritonMLASparseBackend(AttentionBackend):
    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.float16, torch.bfloat16]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "auto",
        "float16",
        "bfloat16",
        "fp8_e5m2",
    ]

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        return [64]

    @staticmethod
    def get_name() -> str:
        return "TRITON_MLA_SPARSE_AMPERE"

    @staticmethod
    def get_builder_cls() -> type[FlashAttnMLASparseMetadataBuilder]:
        return FlashAttnMLASparseMetadataBuilder

    @staticmethod
    def get_impl_cls() -> type[MLAAttentionImpl[Any]]:
        return AmpereTritonMLASparseImpl

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        return []

    @classmethod
    def is_mla(cls) -> bool:
        return True

    @classmethod
    def is_sparse(cls) -> bool:
        return True

    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
        return capability.major == 8 and capability.minor in (0, 6)


class AmpereTritonMLASparseImpl(
    SparseMLACommonImpl[FlashAttnMLASparseMetadata]
):
    can_return_lse_for_decode: bool = True

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: list[float] | None,
        sliding_window: int | None,
        kv_cache_dtype: str,
        logits_soft_cap: float | None,
        attn_type: str,
        kv_sharing_target_layer_name: str | None,
        topk_indices_buffer: torch.Tensor | None = None,
        indexer: Any | None = None,
        **mla_args: Any,
    ) -> None:
        unsupported_features = [alibi_slopes, sliding_window, logits_soft_cap]
        if any(unsupported_features):
            raise NotImplementedError(
                "Ampere Triton sparse MLA does not support alibi, sliding "
                "window, or logits soft cap."
            )
        if kv_cache_dtype not in ("auto", "float16", "bfloat16", "fp8_e5m2"):
            raise NotImplementedError(
                "Ampere Triton sparse MLA currently requires FP16/BF16 KV cache."
            )

        super().__init__(
            num_heads,
            head_size,
            scale,
            num_kv_heads,
            alibi_slopes,
            sliding_window,
            kv_cache_dtype,
            logits_soft_cap,
            attn_type,
            kv_sharing_target_layer_name,
            indexer=indexer,
            topk_indices_buffer=topk_indices_buffer,
            **mla_args,
        )
        assert self.topk_indices_buffer is not None
        self.supports_quant_query_input = False
        self._sm_count = current_platform.num_compute_units()
        self._prefill_query_chunk_tokens = int(
            os.environ.get("VLLM_GLM53_SM86_MLA_QUERY_CHUNK", "0")
        )
        if self._prefill_query_chunk_tokens < 0:
            raise ValueError("VLLM_GLM53_SM86_MLA_QUERY_CHUNK must be >= 0")

    def forward_mqa(
        self,
        q: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: FlashAttnMLASparseMetadata,
        layer: AttentionLayer,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if isinstance(q, tuple):
            q = torch.cat(q, dim=-1)
        assert isinstance(q, torch.Tensor)
        if q.shape[-1] != self.kv_lora_rank:
            raise ValueError(
                "Ampere Triton sparse MLA currently supports no-RoPE MLA only: "
                f"query dim {q.shape[-1]} != latent dim {self.kv_lora_rank}."
            )

        num_tokens, num_heads, _ = q.shape
        if (
            getattr(self, "kv_cache_dtype", "auto") == "fp8_e5m2"
            and kv_c_and_k_pe_cache.dtype != torch.float8_e5m2
        ):
            # The cache holds e5m2 bytes; vLLM's generic fp8 view is e4m3, which SM86 Triton
            # cannot lower. Reinterpret the bytes with the configured format.
            kv_c_and_k_pe_cache = kv_c_and_k_pe_cache.view(torch.uint8).view(torch.float8_e5m2)
        topk_indices = self.topk_indices_buffer[:num_tokens]
        kv_rows, block_stride_rows = flat_kv_row_view(
            kv_c_and_k_pe_cache, attn_metadata.block_size
        )
        if os.environ.get("VLLM_GLM53_DETERMINISTIC_TOPK", "0") == "1":
            # Stable order: the upstream remap compacts through racing atomics.
            physical_rows, valid_counts = triton_convert_req_index_compact_ordered(
                attn_metadata.req_id_per_token[:num_tokens],
                attn_metadata.block_table,
                topk_indices,
                BLOCK_SIZE=attn_metadata.block_size,
                BLOCK_STRIDE_ROWS=block_stride_rows,
            )
        else:
            physical_rows, valid_counts = triton_convert_req_index_to_global_index(
                attn_metadata.req_id_per_token[:num_tokens],
                attn_metadata.block_table,
                topk_indices,
                BLOCK_SIZE=attn_metadata.block_size,
                BLOCK_STRIDE_ROWS=block_stride_rows,
                NUM_TOPK_TOKENS=topk_indices.shape[1],
                return_valid_counts=True,
            )

        if 3 <= num_tokens <= 31 and not torch.cuda.is_current_stream_capturing() \
                and os.path.exists("/home/bot/llm/probe/MLA_DUMP_ON"):
            _glm53_mla_dump(
                self, topk_indices, physical_rows, valid_counts, attn_metadata,
                kv_rows, q,
            )
        # Treat each selected physical KV row as a page of size one.  The
        # existing Triton latent-attention kernel then reads exactly the compact
        # valid prefix produced by the remap kernel.
        paged_rows = kv_rows.unsqueeze(1).unsqueeze(2)
        output = torch.zeros(
            num_tokens,
            num_heads,
            self.kv_lora_rank,
            dtype=q.dtype,
            device=q.device,
        )
        lse = torch.zeros(num_tokens, num_heads, dtype=q.dtype, device=q.device)

        if envs.VLLM_BATCH_INVARIANT or num_tokens > 1:
            num_kv_splits = 1
        else:
            # The remap kernel compacts at most index_topk rows.  Size the
            # split-KV launch from that static upper bound instead of reading
            # valid_counts back to the CPU.  Besides making full CUDA-graph
            # capture legal, this removes one device-wide synchronization per
            # sparse layer and decode step.  Extra splits are semantically
            # empty when the live context is shorter and are ignored by the
            # stage-2 reduction.
            num_kv_splits = _compute_num_kv_splits(
                topk_indices.shape[1], self._sm_count
            )
            _splits_override = int(os.environ.get("GLM53_SPARSE_MLA_KV_SPLITS", "0"))
            if _splits_override > 0:
                num_kv_splits = _splits_override
        # Sparse MLA queries are independent.  Chunk large prefills so the
        # FP32 split-K reduction scratch scales with the configured query
        # chunk rather than the entire prompt.  Decode (num_tokens == 1) and
        # the attention math for every query are unchanged.
        query_chunk_tokens = num_tokens
        if num_tokens > 1 and self._prefill_query_chunk_tokens:
            query_chunk_tokens = min(
                num_tokens, self._prefill_query_chunk_tokens
            )
        for query_start in range(0, num_tokens, query_chunk_tokens):
            query_end = min(num_tokens, query_start + query_chunk_tokens)
            scratch = torch.empty(
                (
                    query_end - query_start,
                    num_heads,
                    num_kv_splits,
                    self.kv_lora_rank + 1,
                ),
                dtype=torch.float32,
                device=q.device,
            )

            decode_attention_fwd(
                q[query_start:query_end],
                paged_rows,
                paged_rows,
                output[query_start:query_end],
                lse[query_start:query_end],
                physical_rows[query_start:query_end],
                valid_counts[query_start:query_end],
                scratch,
                num_kv_splits,
                self.scale,
                1,
                k_scale=layer._k_scale,
                v_scale=layer._k_scale,
                is_mla=True,
            )
        return output, lse
