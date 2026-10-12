# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Triton fallback kernels used by the local DeepSeek V4 path."""

import torch

from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton


def _view_packed_fp8_paged_mqa_kv_cache(
    kv_cache: torch.Tensor,
    head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return FP8 values and fp32 scales from indexer cache block storage."""
    if kv_cache.dtype != torch.uint8:
        raise TypeError(f"Expected uint8 kv_cache, got {kv_cache.dtype}")
    if kv_cache.dim() == 3:
        num_blocks, block_size, head_dim_with_scale = kv_cache.shape
        num_kv_heads = 1
    elif kv_cache.dim() == 4:
        num_blocks, block_size, num_kv_heads, head_dim_with_scale = kv_cache.shape
    else:
        raise ValueError(f"Expected 3D or 4D kv_cache, got {kv_cache.dim()} dimensions")
    if num_kv_heads != 1:
        raise ValueError(f"Expected one KV head, got {num_kv_heads}")

    scale_bytes = head_dim_with_scale - head_dim
    if scale_bytes <= 0 or scale_bytes % torch.float32.itemsize != 0:
        raise ValueError(
            "Expected kv_cache last dimension to contain FP8 values followed "
            f"by fp32 scale bytes; got head_dim={head_dim}, "
            f"last_dim={head_dim_with_scale}"
        )

    block_stride = kv_cache.stride(0)
    base_storage_offset = kv_cache.storage_offset()
    scale_elems = scale_bytes // torch.float32.itemsize
    kv_values = torch.as_strided(
        kv_cache,
        size=(num_blocks, block_size, 1, head_dim),
        stride=(block_stride, head_dim, head_dim, 1),
        storage_offset=base_storage_offset,
    ).view(torch.float8_e4m3fn)
    kv_scale = torch.as_strided(
        kv_cache,
        size=(num_blocks, block_size, 1, scale_bytes),
        stride=(block_stride, scale_bytes, scale_bytes, 1),
        storage_offset=base_storage_offset + block_size * head_dim,
    ).view(torch.float32)
    return kv_values, kv_scale[..., :scale_elems]


@triton.jit
def _fp8_mqa_logits_kernel(
    q_ptr,
    k_ptr,
    scale_ptr,
    weights_ptr,
    cu_seqlen_ks_ptr,
    cu_seqlen_ke_ptr,
    logits_ptr,
    num_q,
    seq_len_kv,
    num_heads: tl.constexpr,
    head_dim: tl.constexpr,
    stride_qm: tl.constexpr,
    stride_qh: tl.constexpr,
    stride_qd: tl.constexpr,
    stride_kn: tl.constexpr,
    stride_kd: tl.constexpr,
    stride_wm: tl.constexpr,
    stride_wh: tl.constexpr,
    stride_lm,
    stride_ln: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    NATIVE_FP8: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)

    valid_m = offs_m < num_q
    valid_n = offs_n < seq_len_kv
    seq_start = tl.load(cu_seqlen_ks_ptr + offs_m, mask=valid_m, other=0)
    seq_end = tl.load(cu_seqlen_ke_ptr + offs_m, mask=valid_m, other=0)
    seq_mask = (offs_n[None, :] >= seq_start[:, None]) & (
        offs_n[None, :] < seq_end[:, None]
    )

    logits = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for h in tl.range(0, num_heads):
        scores = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for d0 in tl.range(0, head_dim, BLOCK_D):
            d = d0 + offs_d
            q = tl.load(
                q_ptr
                + offs_m[:, None] * stride_qm
                + h * stride_qh
                + d[None, :] * stride_qd,
                mask=valid_m[:, None] & (d[None, :] < head_dim),
                other=0.0,
            )
            k = tl.load(
                k_ptr + offs_n[:, None] * stride_kn + d[None, :] * stride_kd,
                mask=valid_n[:, None] & (d[None, :] < head_dim),
                other=0.0,
            )
            if NATIVE_FP8:
                scores += tl.dot(q, tl.trans(k), out_dtype=tl.float32)
            else:
                scores += tl.dot(
                    q.to(tl.float32),
                    tl.trans(k.to(tl.float32)),
                    input_precision="tf32",
                )
        scale = tl.load(scale_ptr + offs_n, mask=valid_n, other=0.0)
        weighted = tl.maximum(scores * scale[None, :], 0.0)
        weight = tl.load(
            weights_ptr + offs_m * stride_wm + h * stride_wh,
            mask=valid_m,
            other=0.0,
        )
        logits += weighted * weight[:, None]

    store_mask = valid_m[:, None] & valid_n[None, :]
    logits = tl.where(seq_mask & store_mask, logits, float("-inf"))
    tl.store(
        logits_ptr + offs_m[:, None] * stride_lm + offs_n[None, :] * stride_ln,
        logits,
        mask=store_mask,
    )


def fp8_mqa_logits_triton(
    q: torch.Tensor,
    kv: tuple[torch.Tensor, torch.Tensor],
    weights: torch.Tensor,
    cu_seqlen_ks: torch.Tensor,
    cu_seqlen_ke: torch.Tensor,
    *,
    native_fp8: bool | None = None,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    k_fp8, scale = kv
    num_q, num_heads, head_dim = q.shape
    seq_len_kv = k_fp8.shape[0]
    if out is None:
        logits = torch.empty(
            (num_q, seq_len_kv),
            device=q.device,
            dtype=torch.float32,
        )
    else:
        if (
            out.shape != (num_q, seq_len_kv)
            or out.dtype != torch.float32
            or out.device != q.device
        ):
            raise ValueError(
                "FP8 MQA logits output must have shape "
                f"{(num_q, seq_len_kv)}, dtype float32, and device {q.device}"
            )
        logits = out
    if num_q == 0 or seq_len_kv == 0:
        return logits

    if native_fp8 is None:
        native_fp8 = current_platform.is_device_capability_family(120)
    block_m = _fp8_mqa_logits_block_m(num_q, seq_len_kv)
    grid = (triton.cdiv(num_q, block_m), triton.cdiv(seq_len_kv, 128))
    _fp8_mqa_logits_kernel[grid](
        q,
        k_fp8,
        scale,
        weights,
        cu_seqlen_ks,
        cu_seqlen_ke,
        logits,
        num_q,
        seq_len_kv,
        num_heads,
        head_dim,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k_fp8.stride(0),
        k_fp8.stride(1),
        weights.stride(0),
        weights.stride(1),
        logits.stride(0),
        logits.stride(1),
        BLOCK_M=block_m,
        BLOCK_N=128,
        BLOCK_D=64,
        NATIVE_FP8=native_fp8,
        num_warps=4,
    )
    return logits


def _fp8_mqa_logits_block_m(num_q: int, seq_len_kv: int) -> int:
    if seq_len_kv <= 16 * 1024:
        return 16
    return 64


@triton.jit
def _fp8_paged_mqa_logits_kernel(
    q_ptr,
    kv_ptr,
    scale_ptr,
    weights_ptr,
    context_lens_ptr,
    block_tables_ptr,
    logits_ptr,
    token_start,
    num_rows,
    logits_width,
    next_n: tl.constexpr,
    num_heads: tl.constexpr,
    head_dim: tl.constexpr,
    block_size: tl.constexpr,
    stride_qb: tl.constexpr,
    stride_qn: tl.constexpr,
    stride_qh: tl.constexpr,
    stride_qd: tl.constexpr,
    stride_kvb: tl.constexpr,
    stride_kvs: tl.constexpr,
    stride_kvd: tl.constexpr,
    stride_sb: tl.constexpr,
    stride_ss: tl.constexpr,
    stride_wm: tl.constexpr,
    stride_wh: tl.constexpr,
    stride_clb: tl.constexpr,
    stride_cln: tl.constexpr,
    stride_btb: tl.constexpr,
    stride_btk: tl.constexpr,
    stride_lm,
    stride_ln: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_local_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_n = token_start + offs_local_n
    offs_d = tl.arange(0, BLOCK_D)

    valid_m = offs_m < num_rows
    valid_n = offs_local_n < logits_width
    batch = offs_m // next_n
    q_pos = offs_m - batch * next_n
    context_len = tl.load(
        context_lens_ptr + batch * stride_clb + q_pos * stride_cln,
        mask=valid_m,
        other=0,
    )
    context_mask = valid_n[None, :] & (offs_n[None, :] < context_len[:, None])

    block_rank = offs_n // block_size
    block_offset = offs_n - block_rank * block_size
    block_idx = tl.load(
        block_tables_ptr
        + batch[:, None] * stride_btb
        + block_rank[None, :] * stride_btk,
        mask=valid_m[:, None] & valid_n[None, :],
        other=0,
    ).to(tl.int64)

    logits = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    scale = tl.load(
        scale_ptr + block_idx * stride_sb + block_offset[None, :] * stride_ss,
        mask=context_mask,
        other=0.0,
    )
    for h in tl.range(0, num_heads):
        scores = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for d0 in tl.range(0, head_dim, BLOCK_D):
            d = d0 + offs_d
            q = tl.load(
                q_ptr
                + batch[:, None] * stride_qb
                + q_pos[:, None] * stride_qn
                + h * stride_qh
                + d[None, :] * stride_qd,
                mask=valid_m[:, None] & (d[None, :] < head_dim),
                other=0.0,
            ).to(tl.float32)
            k = tl.load(
                kv_ptr
                + block_idx[:, :, None] * stride_kvb
                + block_offset[None, :, None] * stride_kvs
                + d[None, None, :] * stride_kvd,
                mask=context_mask[:, :, None] & (d[None, None, :] < head_dim),
                other=0.0,
            ).to(tl.float32)
            scores += tl.sum(q[:, None, :] * k, axis=2)
        weighted = tl.maximum(scores * scale, 0.0)
        weight = tl.load(
            weights_ptr + offs_m * stride_wm + h * stride_wh,
            mask=valid_m,
            other=0.0,
        )
        logits += weighted * weight[:, None]

    store_mask = valid_m[:, None] & valid_n[None, :]
    logits = tl.where(context_mask & store_mask, logits, float("-inf"))
    tl.store(
        logits_ptr + offs_m[:, None] * stride_lm + offs_local_n[None, :] * stride_ln,
        logits,
        mask=store_mask,
    )


@triton.jit
def _fp8_paged_mqa_logits_rowwise_kernel(
    q_ptr,
    kv_ptr,
    scale_ptr,
    weights_ptr,
    context_lens_ptr,
    block_tables_ptr,
    logits_ptr,
    token_start,
    num_rows,
    logits_width,
    next_n: tl.constexpr,
    num_heads: tl.constexpr,
    head_dim: tl.constexpr,
    block_size: tl.constexpr,
    stride_qb: tl.constexpr,
    stride_qn: tl.constexpr,
    stride_qh: tl.constexpr,
    stride_qd: tl.constexpr,
    stride_kvb: tl.constexpr,
    stride_kvs: tl.constexpr,
    stride_kvd: tl.constexpr,
    stride_sb: tl.constexpr,
    stride_ss: tl.constexpr,
    stride_wm: tl.constexpr,
    stride_wh: tl.constexpr,
    stride_clb: tl.constexpr,
    stride_cln: tl.constexpr,
    stride_btb: tl.constexpr,
    stride_btk: tl.constexpr,
    stride_lm,
    stride_ln: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    """Per-row paged-MQA logits kernel optimised for long ``token_count``.

    Each Triton program handles one logical row (``batch * next_n + q_pos``)
    across a ``BLOCK_N``-wide window of token positions. Q is loaded once per
    head tile and reused for every K element in the window, which preserves
    L2 / register locality and avoids the M-axis padding waste of the
    generic 2D-tiled kernel at long contexts (mt-bench c=1 MTP=2 num_rows=3
    with token_count=131072 launches 12k programs of 128 logits each rather
    than 8k programs of 64 logits with 25 % M-axis waste).
    """
    row = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_local_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_n = token_start + offs_local_n
    offs_d = tl.arange(0, BLOCK_D)

    valid_row = row < num_rows
    valid_n = offs_local_n < logits_width
    batch = row // next_n
    q_pos = row - batch * next_n
    context_len = tl.load(
        context_lens_ptr + batch * stride_clb + q_pos * stride_cln,
        mask=valid_row,
        other=0,
    )
    if token_start + pid_n * BLOCK_N >= context_len:
        logits = tl.full((BLOCK_N,), float("-inf"), dtype=tl.float32)
        tl.store(
            logits_ptr + row * stride_lm + offs_local_n * stride_ln,
            logits,
            mask=valid_row & valid_n,
        )
        return
    context_mask = valid_n & (offs_n < context_len)

    block_rank = offs_n // block_size
    block_offset = offs_n - block_rank * block_size
    block_idx = tl.load(
        block_tables_ptr + batch * stride_btb + block_rank * stride_btk,
        mask=valid_row & context_mask,
        other=0,
    ).to(tl.int64)

    scale = tl.load(
        scale_ptr + block_idx * stride_sb + block_offset * stride_ss,
        mask=context_mask,
        other=0.0,
    )
    logits = tl.zeros((BLOCK_N,), dtype=tl.float32)

    for h0 in tl.range(0, num_heads, BLOCK_H):
        heads = h0 + tl.arange(0, BLOCK_H)
        valid_h = heads < num_heads
        scores = tl.zeros((BLOCK_H, BLOCK_N), dtype=tl.float32)
        for d0 in tl.range(0, head_dim, BLOCK_D):
            d = d0 + offs_d
            q = tl.load(
                q_ptr
                + batch * stride_qb
                + q_pos * stride_qn
                + heads[:, None] * stride_qh
                + d[None, :] * stride_qd,
                mask=valid_row & valid_h[:, None] & (d[None, :] < head_dim),
                other=0.0,
            ).to(tl.float32)
            k = tl.load(
                kv_ptr
                + block_idx[None, :] * stride_kvb
                + block_offset[None, :] * stride_kvs
                + d[:, None] * stride_kvd,
                mask=context_mask[None, :] & (d[:, None] < head_dim),
                other=0.0,
            ).to(tl.float32)
            scores += tl.dot(q, k, input_precision="tf32")

        weighted = tl.maximum(scores * scale[None, :], 0.0)
        weight = tl.load(
            weights_ptr + row * stride_wm + heads * stride_wh,
            mask=valid_row & valid_h,
            other=0.0,
        )
        logits += tl.sum(weighted * weight[:, None], axis=0)

    logits = tl.where(context_mask & valid_row, logits, float("-inf"))
    tl.store(
        logits_ptr + row * stride_lm + offs_local_n * stride_ln,
        logits,
        mask=valid_row & valid_n,
    )


def fp8_paged_mqa_logits_rowwise_triton(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    weights: torch.Tensor,
    context_lens: torch.Tensor,
    block_tables: torch.Tensor,
    max_model_len: int,
    token_start: int = 0,
    token_count: int | None = None,
    logits_out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Rowwise paged-MQA logits wrapper.

    Pre-condition: ``head_dim % 64 == 0`` and ``num_heads % 4 == 0`` so the
    ``tl.dot`` inside ``_fp8_paged_mqa_logits_rowwise_kernel`` lands on
    tensor-core friendly tile shapes. DSv4-Flash (head_dim=128,
    num_heads=64) satisfies both and is the only model that exercises this
    path today; the generic 2D kernel below remains the fallback for
    misaligned shapes.
    """
    batch_size, next_n, num_heads, head_dim = q.size()
    kv_values, kv_scale = _view_packed_fp8_paged_mqa_kv_cache(kv_cache, head_dim)
    _, block_size, _, _ = kv_values.size()
    num_rows = batch_size * next_n
    if token_count is None:
        token_count = max_model_len - token_start
    assert token_start >= 0
    assert token_count >= 0
    assert token_start + token_count <= max_model_len
    if logits_out is not None:
        assert logits_out.shape == (num_rows, token_count)
        assert logits_out.dtype == torch.float32 and logits_out.device == q.device
        logits = logits_out
    else:
        logits = torch.empty(
            (num_rows, token_count),
            device=q.device,
            dtype=torch.float32,
        )
    if num_rows == 0 or token_count == 0:
        return logits

    context_lens_2d = context_lens.reshape(batch_size, -1)
    if context_lens_2d.shape[1] == 1 and next_n != 1:
        context_lens_2d = context_lens_2d.expand(batch_size, next_n).contiguous()
    block_n = 128
    grid = (num_rows, triton.cdiv(token_count, block_n))
    _fp8_paged_mqa_logits_rowwise_kernel[grid](
        q,
        kv_values,
        kv_scale,
        weights,
        context_lens_2d,
        block_tables,
        logits,
        token_start,
        num_rows,
        token_count,
        next_n,
        num_heads,
        head_dim,
        block_size,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        kv_values.stride(0),
        kv_values.stride(1),
        kv_values.stride(3),
        kv_scale.stride(0),
        kv_scale.stride(1),
        weights.stride(0),
        weights.stride(1),
        context_lens_2d.stride(0),
        context_lens_2d.stride(1),
        block_tables.stride(0),
        block_tables.stride(1),
        logits.stride(0),
        logits.stride(1),
        BLOCK_N=block_n,
        BLOCK_D=64,
        BLOCK_H=8,
        num_warps=4,
    )
    return logits


def fp8_paged_mqa_logits_triton(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    weights: torch.Tensor,
    context_lens: torch.Tensor,
    block_tables: torch.Tensor,
    max_model_len: int,
    token_start: int = 0,
    token_count: int | None = None,
    logits_out: torch.Tensor | None = None,
) -> torch.Tensor:
    batch_size, next_n, num_heads, head_dim = q.size()
    # Aligned head shapes (DSv4-Flash and any future MQA model with
    # ``head_dim % 64 == 0`` and ``num_heads % 4 == 0``) get the rowwise
    # kernel, which keeps long-context decode (>100K tokens) on a per-row
    # grid that re-uses Q across the full token window. The generic 2D
    # kernel below still handles misaligned shapes and remains the canonical
    # reference for the rowwise variant.
    if head_dim % 64 == 0 and num_heads % 4 == 0:
        return fp8_paged_mqa_logits_rowwise_triton(
            q,
            kv_cache,
            weights,
            context_lens,
            block_tables,
            max_model_len,
            token_start=token_start,
            token_count=token_count,
            logits_out=logits_out,
        )

    kv_values, kv_scale = _view_packed_fp8_paged_mqa_kv_cache(kv_cache, head_dim)
    _, block_size, _, _ = kv_values.size()
    num_rows = batch_size * next_n
    if token_count is None:
        token_count = max_model_len - token_start
    assert token_start >= 0
    assert token_count >= 0
    assert token_start + token_count <= max_model_len
    if logits_out is not None:
        assert logits_out.shape == (num_rows, token_count)
        assert logits_out.dtype == torch.float32 and logits_out.device == q.device
        logits = logits_out
    else:
        logits = torch.empty(
            (num_rows, token_count),
            device=q.device,
            dtype=torch.float32,
        )
    if num_rows == 0 or token_count == 0:
        return logits

    context_lens_2d = context_lens.reshape(batch_size, -1)
    if context_lens_2d.shape[1] == 1 and next_n != 1:
        context_lens_2d = context_lens_2d.expand(batch_size, next_n).contiguous()
    # Adaptive BLOCK_M: the kernel masks off positions >= num_rows, so a fixed
    # BLOCK_M=4 wastes ~75% of M-axis work in the common single-stream decode
    # case (num_rows=1). Pick the smallest power-of-2 tile that still covers
    # num_rows so we keep one grid-program for typical decode while still
    # benefiting from larger tiles when batch / MTP push num_rows higher.
    if num_rows <= 1:
        block_m = 1
    elif num_rows <= 2:
        block_m = 2
    elif num_rows <= 4:
        block_m = 4
    else:
        block_m = 8
    grid = (triton.cdiv(num_rows, block_m), triton.cdiv(token_count, 64))
    _fp8_paged_mqa_logits_kernel[grid](
        q,
        kv_values,
        kv_scale,
        weights,
        context_lens_2d,
        block_tables,
        logits,
        token_start,
        num_rows,
        token_count,
        next_n,
        num_heads,
        head_dim,
        block_size,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        kv_values.stride(0),
        kv_values.stride(1),
        kv_values.stride(3),
        kv_scale.stride(0),
        kv_scale.stride(1),
        weights.stride(0),
        weights.stride(1),
        context_lens_2d.stride(0),
        context_lens_2d.stride(1),
        block_tables.stride(0),
        block_tables.stride(1),
        logits.stride(0),
        logits.stride(1),
        BLOCK_M=block_m,
        BLOCK_N=64,
        BLOCK_D=64,
        num_warps=4,
    )
    return logits


@triton.jit
def _tf32_hc_prenorm_gemm_kernel(
    x_ptr,
    fn_ptr,
    out_ptr,
    sqrsum_ptr,
    M,
    K: tl.constexpr,
    N: tl.constexpr,
    stride_xm: tl.constexpr,
    stride_xk: tl.constexpr,
    stride_fnn: tl.constexpr,
    stride_fnk: tl.constexpr,
    stride_outs,
    stride_outm: tl.constexpr,
    stride_outn: tl.constexpr,
    stride_sqs,
    stride_sqm: tl.constexpr,
    NUM_SPLIT: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_s = tl.program_id(2)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    split_k = tl.cdiv(K, NUM_SPLIT)
    split_begin = pid_s * split_k
    split_end = tl.minimum(split_begin + split_k, K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    sq = tl.zeros((BLOCK_M,), dtype=tl.float32)

    for k0 in tl.range(0, split_k, BLOCK_K):
        k = split_begin + k0 + offs_k
        k_mask = k < split_end
        x = tl.load(
            x_ptr + offs_m[:, None] * stride_xm + k[None, :] * stride_xk,
            mask=(offs_m[:, None] < M) & k_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        fn = tl.load(
            fn_ptr + offs_n[None, :] * stride_fnn + k[:, None] * stride_fnk,
            mask=(offs_n[None, :] < N) & k_mask[:, None],
            other=0.0,
        ).to(tl.float32)

        acc += tl.dot(x, fn, input_precision="tf32", out_dtype=tl.float32)
        sq += tl.sum(x * x, axis=1)

    tl.store(
        out_ptr
        + pid_s * stride_outs
        + offs_m[:, None] * stride_outm
        + offs_n[None, :] * stride_outn,
        acc,
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )

    if pid_n == 0:
        tl.store(
            sqrsum_ptr + pid_s * stride_sqs + offs_m * stride_sqm,
            sq,
            mask=offs_m < M,
        )


def tf32_hc_prenorm_gemm_triton(
    x: torch.Tensor,
    fn: torch.Tensor,
    out: torch.Tensor,
    sqrsum: torch.Tensor,
    num_split: int,
) -> None:
    assert x.dim() == 2
    assert fn.dim() == 2
    assert out.dim() == 3
    assert sqrsum.dim() == 2

    m, k = x.shape
    n = fn.shape[0]
    assert fn.shape[1] == k
    assert out.shape == (num_split, m, n)
    assert sqrsum.shape == (num_split, m)

    if m == 0:
        return

    block_m = 16
    block_n = triton.next_power_of_2(n)
    block_n = min(max(block_n, 16), 32)
    block_k = 64
    grid = (triton.cdiv(m, block_m), triton.cdiv(n, block_n), num_split)
    _tf32_hc_prenorm_gemm_kernel[grid](
        x,
        fn,
        out,
        sqrsum,
        m,
        k,
        n,
        x.stride(0),
        x.stride(1),
        fn.stride(0),
        fn.stride(1),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        sqrsum.stride(0),
        sqrsum.stride(1),
        num_split,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        num_warps=4,
    )



@triton.jit
def _q16_mqa_logits_kernel(
    q_ptr,
    k_ptr,
    scale_ptr,
    weights_ptr,
    cu_seqlen_ks_ptr,
    cu_seqlen_ke_ptr,
    logits_ptr,
    num_q,
    seq_len_kv,
    num_heads,
    stride_qm,
    stride_qh,
    stride_qd: tl.constexpr,
    stride_kn,
    stride_kd: tl.constexpr,
    stride_wm,
    stride_wh: tl.constexpr,
    stride_lm,
    stride_ln: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEAD_DIM)
    valid_m = offs_m < num_q
    valid_n = offs_n < seq_len_kv
    seq_start = tl.load(cu_seqlen_ks_ptr + offs_m, mask=valid_m, other=2147483647)
    seq_end = tl.load(cu_seqlen_ke_ptr + offs_m, mask=valid_m, other=0)
    store_mask = valid_m[:, None] & valid_n[None, :]
    lo = pid_n * BLOCK_N
    if (tl.max(seq_end, axis=0) <= lo) | (tl.min(seq_start, axis=0) >= lo + BLOCK_N):
        tl.store(
            logits_ptr + offs_m[:, None] * stride_lm + offs_n[None, :] * stride_ln,
            tl.full((BLOCK_M, BLOCK_N), float("-inf"), tl.float32),
            mask=store_mask,
        )
        return
    # K arrives as raw e4m3fn bytes (SM86 Triton has no fp8e4nv type): decode exactly to fp16.
    kb = tl.load(
        k_ptr + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd,
        mask=valid_n[:, None],
        other=0,
    ).to(tl.int32)
    k_sign = (kb & 0x80) << 8
    k_exp = (kb >> 3) & 0xF
    k_man = kb & 0x7
    k_norm = (k_sign | ((k_exp + 8) << 10) | (k_man << 7)).to(tl.int16).to(tl.float16, bitcast=True)
    k_sub = k_man.to(tl.float32) * 0.001953125
    k_sub = tl.where(k_sign != 0, -k_sub, k_sub).to(tl.float16)
    k = tl.where(k_exp == 0, k_sub, k_norm)
    kt = tl.trans(k)
    scale = tl.load(scale_ptr + offs_n, mask=valid_n, other=0.0)
    logits = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for h in tl.range(0, num_heads):
        q = tl.load(
            q_ptr + offs_m[:, None] * stride_qm + h * stride_qh + offs_d[None, :] * stride_qd,
            mask=valid_m[:, None],
            other=0.0,
        ).to(tl.float16)
        scores = tl.dot(q, kt, out_dtype=tl.float32)
        weighted = tl.maximum(scores * scale[None, :], 0.0)
        w = tl.load(weights_ptr + offs_m * stride_wm + h * stride_wh, mask=valid_m, other=0.0)
        logits += weighted * w[:, None]
    seq_mask = (offs_n[None, :] >= seq_start[:, None]) & (offs_n[None, :] < seq_end[:, None])
    logits = tl.where(seq_mask, logits, float("-inf"))
    tl.store(
        logits_ptr + offs_m[:, None] * stride_lm + offs_n[None, :] * stride_ln,
        logits,
        mask=store_mask,
    )


def q16_mqa_logits_triton(q, kv, weights, cu_seqlen_ks, cu_seqlen_ke, num_heads=None):
    """bf16/fp16 Q x fp8 K indexer logits (see patches/patch_q16_mqa.py)."""
    k_fp8, scale = kv
    num_q, heads, head_dim = q.shape
    if num_heads is None:
        num_heads = heads
    seq_len_kv = k_fp8.shape[0]
    logits = torch.empty((num_q, seq_len_kv), device=q.device, dtype=torch.float32)
    if num_q == 0 or seq_len_kv == 0:
        return logits
    block_m, block_n = 64, 128
    grid = (triton.cdiv(num_q, block_m), triton.cdiv(seq_len_kv, block_n))
    k_bytes = k_fp8.view(torch.uint8)
    _q16_mqa_logits_kernel[grid](
        q, k_bytes, scale, weights, cu_seqlen_ks, cu_seqlen_ke, logits,
        num_q, seq_len_kv, num_heads,
        q.stride(0), q.stride(1), q.stride(2),
        k_bytes.stride(0), k_bytes.stride(1),
        weights.stride(0), weights.stride(1),
        logits.stride(0), logits.stride(1),
        HEAD_DIM=head_dim, BLOCK_M=block_m, BLOCK_N=block_n,
        num_warps=8, num_stages=2,
    )
    return logits



@triton.jit
def _q16_paged_mqa_logits_kernel(
    q_ptr,            # [rows, heads, D] 16-bit
    kv_ptr,           # uint8 packed cache base
    weights_ptr,      # [rows, heads] fp32
    ctx_ptr,          # [rows] int32 live context (tokens)
    bt_ptr,           # [batch, max_blocks] int32
    logits_ptr,       # [rows, max_tokens] fp32
    next_n,
    num_heads,
    max_tokens,
    max_blocks,
    block_stride,     # bytes per physical block
    stride_qm,
    stride_qh,
    stride_wm,
    stride_btb,
    stride_lm,
    HEAD_DIM: tl.constexpr,
    BLOCK_KV: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    pid_n = tl.program_id(1)
    batch = row // next_n
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    valid_n = offs_n < max_tokens
    ctx = tl.load(ctx_ptr + row)
    if pid_n * BLOCK_N >= ctx:
        tl.store(logits_ptr + row * stride_lm + offs_n,
                 tl.full((BLOCK_N,), float("-inf"), tl.float32), mask=valid_n)
        return
    offs_d = tl.arange(0, HEAD_DIM)
    lb = offs_n // BLOCK_KV
    tib = offs_n - lb * BLOCK_KV
    live = valid_n & (offs_n < ctx)
    pb = tl.load(bt_ptr + batch * stride_btb + lb, mask=live & (lb < max_blocks), other=0).to(tl.int64)
    base = pb * block_stride
    kb = tl.load(kv_ptr + base[:, None] + (tib * HEAD_DIM)[:, None] + offs_d[None, :],
                 mask=live[:, None], other=0).to(tl.int32)
    k_sign = (kb & 0x80) << 8
    k_exp = (kb >> 3) & 0xF
    k_man = kb & 0x7
    k_norm = (k_sign | ((k_exp + 8) << 10) | (k_man << 7)).to(tl.int16).to(tl.float16, bitcast=True)
    k_sub = k_man.to(tl.float32) * 0.001953125
    k_sub = tl.where(k_sign != 0, -k_sub, k_sub).to(tl.float16)
    k = tl.where(k_exp == 0, k_sub, k_norm)
    sb = base + BLOCK_KV * HEAD_DIM + tib * 4
    scale = tl.load((kv_ptr + sb).to(tl.pointer_type(tl.float32)), mask=live, other=0.0)
    offs_h = tl.arange(0, BLOCK_H)
    valid_h = offs_h < num_heads
    q = tl.load(q_ptr + row * stride_qm + offs_h[:, None] * stride_qh + offs_d[None, :],
                mask=valid_h[:, None], other=0.0).to(tl.float16)
    scores = tl.dot(q, tl.trans(k), out_dtype=tl.float32)          # [BLOCK_H, BLOCK_N]
    w = tl.load(weights_ptr + row * stride_wm + offs_h, mask=valid_h, other=0.0)
    weighted = tl.maximum(scores * scale[None, :], 0.0) * w[:, None]
    logits = tl.sum(weighted, axis=0)
    logits = tl.where(live, logits, float("-inf"))
    tl.store(logits_ptr + row * stride_lm + offs_n, logits, mask=valid_n)


def q16_paged_mqa_logits_triton(q, kv_cache, weights, context_lens, block_tables,
                                max_model_len, num_heads=None):
    """q [B, next_n, H, D] 16-bit; kv_cache uint8 [blocks, block_kv, (1,) D + 4]."""
    batch, next_n, heads, head_dim = q.shape
    if num_heads is None:
        num_heads = heads
    rows = batch * next_n
    qf = q.reshape(rows, heads, head_dim)
    block_kv = kv_cache.shape[1]
    kvb = kv_cache.view(torch.uint8)
    block_stride = kvb.stride(0)
    max_tokens = min(max_model_len, block_tables.shape[1] * block_kv)
    logits = torch.empty((rows, max_model_len), device=q.device, dtype=torch.float32)
    if max_model_len > max_tokens:
        logits[:, max_tokens:].fill_(float("-inf"))
    ctx = context_lens.reshape(rows).to(torch.int32).contiguous()
    bt = block_tables.to(torch.int32)
    block_n = 128
    block_h = max(16, triton.next_power_of_2(num_heads))
    grid = (rows, triton.cdiv(max_tokens, block_n))
    _q16_paged_mqa_logits_kernel[grid](
        qf, kvb, weights, ctx, bt, logits,
        next_n, num_heads, max_tokens, bt.shape[1], block_stride,
        qf.stride(0), qf.stride(1), weights.stride(0), bt.stride(0), logits.stride(0),
        HEAD_DIM=head_dim, BLOCK_KV=block_kv, BLOCK_H=block_h, BLOCK_N=block_n,
        num_warps=4,
    )
    return logits
