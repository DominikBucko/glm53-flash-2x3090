# SPDX-License-Identifier: Apache-2.0
"""EXL3 trellis quantization for GLM-5.3-Flash routed experts.

Checkpoint ABI (turboderp/GLM-5.3-Flash-exl3):
  quant_method=exl3, codebook=mcg, scope=glm53_routed_experts_only
  per expert matrix: trellis (int16) + suh/svh (fp16) + codebook marker

Non-routed tensors stay native (UnquantizedLinearMethod). Experts never
expand to a persistent BF16 weight; LinearEXL3 / exllamav3_ext runs the
trellis GEMM. TP shards projections; EP instead assigns complete experts
to each rank and all-reduces one combined hidden vector per routed layer.
"""

from __future__ import annotations

import importlib
import importlib.util
import json
import os
import re
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig
from vllm.model_executor.layers.fused_moe.fused_moe_method_base import (
    FusedMoEMethodBase,
)
from vllm.model_executor.layers.linear import (
    LinearBase,
    LinearMethodBase,
    UnquantizedLinearMethod,
)
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.model_executor.layers.quantization import register_quantization_config
from vllm.model_executor.utils import replace_parameter, set_weight_attrs
from vllm.triton_utils import tl, triton

if TYPE_CHECKING:
    from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts
    from vllm.model_executor.layers.fused_moe.runner.shared_experts import (
        SharedExperts,
    )

logger = init_logger(__name__)

EXLLAMAV3_COMMIT = "c5d9c657966ffeeaa9353f0cc899f18629da4a13"
EXLLAMAV3_VERSION = "0.0.43"
MCG_MULTIPLIER = 0xCBAC1FED
MCG_MARKER_SIGNED_INT32 = -877912083
MUL1_MULTIPLIER = 0x83DCD12D
MUL1_MARKER_SIGNED_INT32 = -2082680531
EXL3_SUFFIXES = ("trellis", "suh", "svh", "mcg", "mul1")
SWIGLU_LIMIT_DEFAULT = 10.0
# Default fused-kernel temp rows/expert. 1024 covers MNBT=1024 in one launch
# but measured slower than 128+fallback (P2b). Override with EXL3_TEMP_ROWS_FUSED.
TEMP_ROWS_FUSED = 128
MOE_ACT_SILU = 0
# Shared fused scratch: decode is sequential across layers.
_FUSED_TEMP_CACHE: dict[tuple, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = {}
_FAT_SCRATCH_CACHE: dict[tuple, dict[str, torch.Tensor]] = {}
_FAT_COUNT_CACHE: dict[tuple, tuple[torch.Tensor, torch.cuda.Stream]] = {}
_FAT_BUCKET_EDGES = (16, 32, 64, 128, 256, 512, 1024, 2048)
_FAT_STATS: dict[str, Any] = {
    "layers": 0,
    "fat_layers": 0,
    "fat_experts": 0,
    "max_rows": 0,
    "sum_max_rows": 0,
    "hist": [0] * (len(_FAT_BUCKET_EDGES) + 1),
}
_FAT_TIERS = ("kernel", "batched", "sorted", "legacy")
# Machine-checkable E2 (fat-expert) diagnostics. Load-time fields mirror the
# most recent weight load; counters are monotonic per process and are the
# ground truth for what actually ran. The key set is contract — extend it only
# together with a bump of EXL3_FAT_DIAG_SCHEMA.
EXL3_FAT_DIAG_SCHEMA = 1
EXL3_FAT_DIAG_KEYS = (
    "schema",
    "configured_tier",
    "effective_tier",
    "tier_reason",
    "shared_suh",
    "shared_suh_layers",
    "moe_layers_loaded",
    "sym_exl3_moe",
    "sym_fat_gemm",
    "sym_fat_gemm_scatter",
    "cap_major",
    "cap_minor",
    "cap_ok",
    "tp_rank",
    "tp_size",
    "fused_temps_allocs",
    "fused_temps_bytes",
    "fat_scratch_allocs",
    "fat_scratch_bytes",
    "fat_scratch_peak_bytes",
    "prefill_layer_calls",
    "thin_calls",
    "row_tile_calls",
    "fallback_calls",
    "fallback_reasons",
    "fat_expert_runs",
    "direct_calls",
    "scatter_calls",
    "fat_stat_layers",
    "fat_layers",
    "fat_expert_slots",
    "max_rows",
)
_EXL3_FAT_DIAG: dict[str, Any] = {
    "schema": EXL3_FAT_DIAG_SCHEMA,
    "configured_tier": "legacy",
    "effective_tier": "legacy",
    "tier_reason": "unresolved",
    "shared_suh": False,
    "shared_suh_layers": 0,
    "moe_layers_loaded": 0,
    "sym_exl3_moe": False,
    "sym_fat_gemm": False,
    "sym_fat_gemm_scatter": False,
    "cap_major": -1,
    "cap_minor": -1,
    "cap_ok": False,
    "tp_rank": -1,
    "tp_size": 1,
    "fused_temps_allocs": 0,
    "fused_temps_bytes": 0,
    "fat_scratch_allocs": 0,
    "fat_scratch_bytes": 0,
    "fat_scratch_peak_bytes": 0,
    "prefill_layer_calls": 0,
    "thin_calls": 0,
    "row_tile_calls": 0,
    "fallback_calls": {tier: 0 for tier in _FAT_TIERS},
    "fallback_reasons": {},
    "fat_expert_runs": 0,
    "direct_calls": 0,
    "scatter_calls": 0,
}
_FAT_SCRATCH_BYTES: dict[tuple, int] = {}
_exl3_fat_tier_logged = False


@triton.jit
def _permute_exl3_expert_rows_kernel(
    source_ptr,
    output_ptr,
    new_to_old_ptr,
    row_size: tl.constexpr,
    block_size: tl.constexpr,
):
    """Copy one hot-first expert permutation without a full-size GPU temporary."""
    new_row = tl.program_id(0)
    block = tl.program_id(1)
    offsets = block * block_size + tl.arange(0, block_size)
    mask = offsets < row_size
    old_row = tl.load(new_to_old_ptr + new_row)
    values = tl.load(source_ptr + old_row * row_size + offsets, mask=mask)
    tl.store(output_ptr + new_row * row_size + offsets, values, mask=mask)


@triton.jit
def _update_exl3_lru_map_kernel(
    global_ids_ptr,
    source_map_ptr,
    cache_map_ptr,
    slot_global_ids_ptr,
    slot_ages_ptr,
    clock_ptr,
    miss_local_ids_ptr,
    miss_slots_ptr,
    counters_ptr,
    num_ids: tl.constexpr,
    global_num_experts: tl.constexpr,
    capacity: tl.constexpr,
    id_block: tl.constexpr,
    capacity_block: tl.constexpr,
):
    """Exact, graph-safe LRU update for one routed row or small verify batch."""
    id_offsets = tl.arange(0, id_block)
    requested = tl.load(
        global_ids_ptr + id_offsets, mask=id_offsets < num_ids, other=-1
    )
    slot_offsets = tl.arange(0, capacity_block)
    valid_slots = slot_offsets < capacity
    slot_global_ids = tl.load(
        slot_global_ids_ptr + slot_offsets, mask=valid_slots, other=-1
    )
    slot_ages = tl.load(slot_ages_ptr + slot_offsets, mask=valid_slots, other=0)
    protected = ~valid_slots
    for request_index in tl.static_range(0, num_ids):
        requested_global_id = tl.sum(
            tl.where(id_offsets == request_index, requested, 0), axis=0
        )
        protected = protected | (slot_global_ids == requested_global_id)

    clock = tl.load(clock_ptr)
    hits = 0
    misses = 0
    for request_index in tl.static_range(0, num_ids):
        global_id = tl.sum(
            tl.where(id_offsets == request_index, requested, 0), axis=0
        )
        valid_global = (global_id >= 0) & (global_id < global_num_experts)
        global_safe = tl.minimum(
            tl.maximum(global_id, 0), global_num_experts - 1
        )
        local_id = tl.load(
            source_map_ptr + global_safe, mask=valid_global, other=-1
        )
        is_local = valid_global & (local_id >= 0)
        cache_slot = tl.load(
            cache_map_ptr + global_safe, mask=is_local, other=-1
        )
        is_hit = is_local & (cache_slot >= 0)
        appeared_before = tl.sum(
            (requested == global_id) & (id_offsets < request_index)
        ) > 0

        victim_ages = tl.where(protected, 0x7FFFFFFF, slot_ages)
        victim = tl.argmin(victim_ages, axis=0)
        is_miss = is_local & ~is_hit & ~appeared_before
        assigned_slot = tl.where(is_hit, cache_slot, victim)
        new_age = clock + request_index + 1

        old_global_id = tl.sum(
            tl.where(slot_offsets == victim, slot_global_ids, 0), axis=0
        )
        old_valid = is_miss & (old_global_id >= 0)
        tl.store(cache_map_ptr + old_global_id, -1, mask=old_valid)
        tl.store(cache_map_ptr + global_safe, victim, mask=is_miss)

        slot_global_ids = tl.where(
            is_miss & (slot_offsets == victim), global_id, slot_global_ids
        )
        slot_ages = tl.where(
            (is_hit | is_miss) & (slot_offsets == assigned_slot),
            new_age,
            slot_ages,
        )
        protected = protected | (is_miss & (slot_offsets == victim))
        tl.store(
            miss_local_ids_ptr + request_index,
            tl.where(is_miss, local_id, -1),
        )
        tl.store(
            miss_slots_ptr + request_index,
            tl.where(is_miss, victim, -1),
        )
        hits += is_hit
        misses += is_miss

    tl.store(slot_global_ids_ptr + slot_offsets, slot_global_ids, mask=valid_slots)
    tl.store(slot_ages_ptr + slot_offsets, slot_ages, mask=valid_slots)
    tl.store(clock_ptr, clock + num_ids)
    tl.atomic_add(counters_ptr, hits)
    tl.atomic_add(counters_ptr + 1, misses)


@triton.jit
def _update_exl3_slru_map_kernel(
    global_ids_ptr,
    source_map_ptr,
    cache_map_ptr,
    slot_global_ids_ptr,
    slot_ages_ptr,
    slot_segments_ptr,
    clock_ptr,
    miss_local_ids_ptr,
    miss_slots_ptr,
    counters_ptr,
    num_ids: tl.constexpr,
    global_num_experts: tl.constexpr,
    capacity: tl.constexpr,
    protected_capacity: tl.constexpr,
    id_block: tl.constexpr,
    capacity_block: tl.constexpr,
):
    """Graph-safe segmented LRU with request-set-safe victim selection."""
    id_offsets = tl.arange(0, id_block)
    requested = tl.load(
        global_ids_ptr + id_offsets, mask=id_offsets < num_ids, other=-1
    )
    slot_offsets = tl.arange(0, capacity_block)
    valid_slots = slot_offsets < capacity
    slot_global_ids = tl.load(
        slot_global_ids_ptr + slot_offsets, mask=valid_slots, other=-1
    )
    slot_ages = tl.load(slot_ages_ptr + slot_offsets, mask=valid_slots, other=0)
    slot_segments = tl.load(
        slot_segments_ptr + slot_offsets, mask=valid_slots, other=0
    )
    request_protected = ~valid_slots
    for request_index in tl.static_range(0, num_ids):
        requested_global_id = tl.sum(
            tl.where(id_offsets == request_index, requested, 0), axis=0
        )
        request_protected = request_protected | (
            slot_global_ids == requested_global_id
        )

    clock = tl.load(clock_ptr)
    hits = 0
    misses = 0
    for request_index in tl.static_range(0, num_ids):
        global_id = tl.sum(
            tl.where(id_offsets == request_index, requested, 0), axis=0
        )
        valid_global = (global_id >= 0) & (global_id < global_num_experts)
        global_safe = tl.minimum(
            tl.maximum(global_id, 0), global_num_experts - 1
        )
        local_id = tl.load(
            source_map_ptr + global_safe, mask=valid_global, other=-1
        )
        is_local = valid_global & (local_id >= 0)
        cache_slot = tl.load(
            cache_map_ptr + global_safe, mask=is_local, other=-1
        )
        is_hit = is_local & (cache_slot >= 0)
        appeared_before = tl.sum(
            (requested == global_id) & (id_offsets < request_index)
        ) > 0

        hit_segment = tl.sum(
            tl.where(slot_offsets == cache_slot, slot_segments, 0), axis=0
        )
        promote = is_hit & (hit_segment == 0)
        protected_count = tl.sum(
            valid_slots & (slot_segments != 0), axis=0
        )
        demote_mask = (
            valid_slots
            & (slot_segments != 0)
            & (slot_offsets != cache_slot)
        )
        demote_ages = tl.where(demote_mask, slot_ages, 0x7FFFFFFF)
        demote_slot = tl.argmin(demote_ages, axis=0)
        demote = promote & (protected_count >= protected_capacity)
        slot_segments = tl.where(
            demote & (slot_offsets == demote_slot), 0, slot_segments
        )
        slot_segments = tl.where(
            promote & (slot_offsets == cache_slot), 1, slot_segments
        )

        # New entries enter probation. Request-set protection guarantees every
        # expert selected by this token remains resident through the update.
        probation_candidate = (
            valid_slots & (slot_segments == 0) & ~request_protected
        )
        victim_ages = tl.where(probation_candidate, slot_ages, 0x7FFFFFFF)
        victim = tl.argmin(victim_ages, axis=0)
        is_miss = is_local & ~is_hit & ~appeared_before
        assigned_slot = tl.where(is_hit, cache_slot, victim)
        new_age = clock + request_index + 1

        old_global_id = tl.sum(
            tl.where(slot_offsets == victim, slot_global_ids, 0), axis=0
        )
        old_valid = is_miss & (old_global_id >= 0)
        tl.store(cache_map_ptr + old_global_id, -1, mask=old_valid)
        tl.store(cache_map_ptr + global_safe, victim, mask=is_miss)

        slot_global_ids = tl.where(
            is_miss & (slot_offsets == victim), global_id, slot_global_ids
        )
        slot_ages = tl.where(
            (is_hit | is_miss) & (slot_offsets == assigned_slot),
            new_age,
            slot_ages,
        )
        slot_segments = tl.where(
            is_miss & (slot_offsets == victim), 0, slot_segments
        )
        request_protected = request_protected | (
            is_miss & (slot_offsets == victim)
        )
        tl.store(
            miss_local_ids_ptr + request_index,
            tl.where(is_miss, local_id, -1),
        )
        tl.store(
            miss_slots_ptr + request_index,
            tl.where(is_miss, victim, -1),
        )
        hits += is_hit
        misses += is_miss

    tl.store(slot_global_ids_ptr + slot_offsets, slot_global_ids, mask=valid_slots)
    tl.store(slot_ages_ptr + slot_offsets, slot_ages, mask=valid_slots)
    tl.store(slot_segments_ptr + slot_offsets, slot_segments, mask=valid_slots)
    tl.store(clock_ptr, clock + num_ids)
    tl.atomic_add(counters_ptr, hits)
    tl.atomic_add(counters_ptr + 1, misses)


@triton.jit
def _gather_exl3_lru_rows_kernel(
    source_ptr,
    cache_ptr,
    miss_local_ids_ptr,
    miss_slots_ptr,
    row_size: tl.constexpr,
    num_ids: tl.constexpr,
    block_size: tl.constexpr,
):
    request_index = tl.program_id(0)
    block = tl.program_id(1)
    offsets = block * block_size + tl.arange(0, block_size)
    source_row = tl.load(miss_local_ids_ptr + request_index)
    cache_row = tl.load(miss_slots_ptr + request_index)
    mask = (source_row >= 0) & (cache_row >= 0) & (offsets < row_size)
    values = tl.load(source_ptr + source_row * row_size + offsets, mask=mask)
    tl.store(cache_ptr + cache_row * row_size + offsets, values, mask=mask)


@triton.jit
def _gather_exl3_lru_expert_kernel(
    source0_ptr,
    cache0_ptr,
    source1_ptr,
    cache1_ptr,
    source2_ptr,
    cache2_ptr,
    source3_ptr,
    cache3_ptr,
    source4_ptr,
    cache4_ptr,
    source5_ptr,
    cache5_ptr,
    miss_local_ids_ptr,
    miss_slots_ptr,
    row0: tl.constexpr,
    row1: tl.constexpr,
    row2: tl.constexpr,
    row3: tl.constexpr,
    row4: tl.constexpr,
    row5: tl.constexpr,
    chunks0: tl.constexpr,
    chunks1: tl.constexpr,
    chunks2: tl.constexpr,
    chunks3: tl.constexpr,
    chunks4: tl.constexpr,
    chunks5: tl.constexpr,
    block_size: tl.constexpr,
):
    """Gather all six 16-bit EXL3 rows with one launch per layer."""
    request_index = tl.program_id(0)
    chunk = tl.program_id(1)
    source_row = tl.load(miss_local_ids_ptr + request_index)
    cache_row = tl.load(miss_slots_ptr + request_index)
    valid = (source_row >= 0) & (cache_row >= 0)
    offsets = tl.arange(0, block_size)

    if chunk < chunks0:
        local_chunk = chunk
        local_offsets = local_chunk * block_size + offsets
        mask = valid & (local_offsets < row0)
        values = tl.load(source0_ptr + source_row * row0 + local_offsets, mask=mask)
        tl.store(cache0_ptr + cache_row * row0 + local_offsets, values, mask=mask)
    elif chunk < chunks0 + chunks1:
        local_chunk = chunk - chunks0
        local_offsets = local_chunk * block_size + offsets
        mask = valid & (local_offsets < row1)
        values = tl.load(source1_ptr + source_row * row1 + local_offsets, mask=mask)
        tl.store(cache1_ptr + cache_row * row1 + local_offsets, values, mask=mask)
    elif chunk < chunks0 + chunks1 + chunks2:
        local_chunk = chunk - chunks0 - chunks1
        local_offsets = local_chunk * block_size + offsets
        mask = valid & (local_offsets < row2)
        values = tl.load(source2_ptr + source_row * row2 + local_offsets, mask=mask)
        tl.store(cache2_ptr + cache_row * row2 + local_offsets, values, mask=mask)
    elif chunk < chunks0 + chunks1 + chunks2 + chunks3:
        local_chunk = chunk - chunks0 - chunks1 - chunks2
        local_offsets = local_chunk * block_size + offsets
        mask = valid & (local_offsets < row3)
        values = tl.load(source3_ptr + source_row * row3 + local_offsets, mask=mask)
        tl.store(cache3_ptr + cache_row * row3 + local_offsets, values, mask=mask)
    elif chunk < chunks0 + chunks1 + chunks2 + chunks3 + chunks4:
        local_chunk = chunk - chunks0 - chunks1 - chunks2 - chunks3
        local_offsets = local_chunk * block_size + offsets
        mask = valid & (local_offsets < row4)
        values = tl.load(source4_ptr + source_row * row4 + local_offsets, mask=mask)
        tl.store(cache4_ptr + cache_row * row4 + local_offsets, values, mask=mask)
    else:
        local_chunk = chunk - chunks0 - chunks1 - chunks2 - chunks3 - chunks4
        local_offsets = local_chunk * block_size + offsets
        mask = valid & (local_offsets < row5)
        values = tl.load(source5_ptr + source_row * row5 + local_offsets, mask=mask)
        tl.store(cache5_ptr + cache_row * row5 + local_offsets, values, mask=mask)


@triton.jit
def _build_exl3_cold_map_kernel(
    source_map_ptr,
    cache_map_ptr,
    cold_map_ptr,
    num_experts: tl.constexpr,
    block_size: tl.constexpr,
):
    offsets = tl.arange(0, block_size)
    mask = offsets < num_experts
    source_slot = tl.load(source_map_ptr + offsets, mask=mask, other=-1)
    cache_slot = tl.load(cache_map_ptr + offsets, mask=mask, other=-1)
    cold_slot = tl.where((source_slot >= 0) & (cache_slot < 0), source_slot, -1)
    tl.store(cold_map_ptr + offsets, cold_slot, mask=mask)


@triton.jit
def _refresh_exl3_hybrid_ptrs_kernel(
    local_to_global_ptr,
    cache_map_ptr,
    source0_ptr,
    cache0_ptr,
    hybrid0_ptr,
    source1_ptr,
    cache1_ptr,
    hybrid1_ptr,
    source2_ptr,
    cache2_ptr,
    hybrid2_ptr,
    source3_ptr,
    cache3_ptr,
    hybrid3_ptr,
    source4_ptr,
    cache4_ptr,
    hybrid4_ptr,
    source5_ptr,
    cache5_ptr,
    hybrid5_ptr,
    source6_ptr,
    cache6_ptr,
    hybrid6_ptr,
    source7_ptr,
    cache7_ptr,
    hybrid7_ptr,
    source8_ptr,
    cache8_ptr,
    hybrid8_ptr,
    num_local: tl.constexpr,
    block_size: tl.constexpr,
):
    """Point every local expert at VRAM when resident, UVA otherwise."""
    offsets = tl.arange(0, block_size)
    valid = offsets < num_local
    global_id = tl.load(local_to_global_ptr + offsets, mask=valid, other=0)
    cache_slot = tl.load(cache_map_ptr + global_id, mask=valid, other=-1)
    resident = valid & (cache_slot >= 0)
    cache_safe = tl.maximum(cache_slot, 0)

    source0 = tl.load(source0_ptr + offsets, mask=valid, other=0)
    cached0 = tl.load(cache0_ptr + cache_safe, mask=resident, other=0)
    tl.store(hybrid0_ptr + offsets, tl.where(resident, cached0, source0), mask=valid)
    source1 = tl.load(source1_ptr + offsets, mask=valid, other=0)
    cached1 = tl.load(cache1_ptr + cache_safe, mask=resident, other=0)
    tl.store(hybrid1_ptr + offsets, tl.where(resident, cached1, source1), mask=valid)
    source2 = tl.load(source2_ptr + offsets, mask=valid, other=0)
    cached2 = tl.load(cache2_ptr + cache_safe, mask=resident, other=0)
    tl.store(hybrid2_ptr + offsets, tl.where(resident, cached2, source2), mask=valid)
    source3 = tl.load(source3_ptr + offsets, mask=valid, other=0)
    cached3 = tl.load(cache3_ptr + cache_safe, mask=resident, other=0)
    tl.store(hybrid3_ptr + offsets, tl.where(resident, cached3, source3), mask=valid)
    source4 = tl.load(source4_ptr + offsets, mask=valid, other=0)
    cached4 = tl.load(cache4_ptr + cache_safe, mask=resident, other=0)
    tl.store(hybrid4_ptr + offsets, tl.where(resident, cached4, source4), mask=valid)
    source5 = tl.load(source5_ptr + offsets, mask=valid, other=0)
    cached5 = tl.load(cache5_ptr + cache_safe, mask=resident, other=0)
    tl.store(hybrid5_ptr + offsets, tl.where(resident, cached5, source5), mask=valid)
    source6 = tl.load(source6_ptr + offsets, mask=valid, other=0)
    cached6 = tl.load(cache6_ptr + cache_safe, mask=resident, other=0)
    tl.store(hybrid6_ptr + offsets, tl.where(resident, cached6, source6), mask=valid)
    source7 = tl.load(source7_ptr + offsets, mask=valid, other=0)
    cached7 = tl.load(cache7_ptr + cache_safe, mask=resident, other=0)
    tl.store(hybrid7_ptr + offsets, tl.where(resident, cached7, source7), mask=valid)
    source8 = tl.load(source8_ptr + offsets, mask=valid, other=0)
    cached8 = tl.load(cache8_ptr + cache_safe, mask=resident, other=0)
    tl.store(hybrid8_ptr + offsets, tl.where(resident, cached8, source8), mask=valid)


@triton.jit
def _split_exl3_lru_work_maps_kernel(
    global_ids_ptr,
    source_map_ptr,
    cache_map_ptr,
    miss_slots_ptr,
    hit_map_ptr,
    miss_map_ptr,
    num_ids: tl.constexpr,
    global_num_experts: tl.constexpr,
    global_block: tl.constexpr,
    id_block: tl.constexpr,
):
    offsets = tl.arange(0, global_block)
    tl.store(hit_map_ptr + offsets, -1, mask=offsets < global_num_experts)
    tl.store(miss_map_ptr + offsets, -1, mask=offsets < global_num_experts)
    request_offsets = tl.arange(0, id_block)
    valid_request = request_offsets < num_ids
    global_ids = tl.load(
        global_ids_ptr + request_offsets, mask=valid_request, other=-1
    )
    valid_global = (
        valid_request
        & (global_ids >= 0)
        & (global_ids < global_num_experts)
    )
    global_safe = tl.minimum(
        tl.maximum(global_ids, 0), global_num_experts - 1
    )
    source_slot = tl.load(
        source_map_ptr + global_safe, mask=valid_global, other=-1
    )
    cache_slot = tl.load(
        cache_map_ptr + global_safe, mask=valid_global, other=-1
    )
    fill_slot = tl.load(
        miss_slots_ptr + request_offsets, mask=valid_request, other=-1
    )
    is_local = valid_global & (source_slot >= 0) & (cache_slot >= 0)
    tl.store(
        hit_map_ptr + global_safe,
        cache_slot,
        mask=is_local & (fill_slot < 0),
    )
    tl.store(
        miss_map_ptr + global_safe,
        cache_slot,
        mask=is_local & (fill_slot >= 0),
    )


@dataclass
class _Exl3MixedVMMAllocation:
    """Keep CUDA VMM mappings and their non-owning PyTorch storage alive."""

    storage: Any
    address: Any
    handles: tuple[Any, ...]
    mapped_bytes: int
    gpu_bytes: int
    host_bytes: int


_EXL3_MIXED_VMM_ALLOCATIONS: list[_Exl3MixedVMMAllocation] = []
_EXL3_VMM_RANKINGS_CACHE: dict[str, dict[str, Any]] = {}


@dataclass
class _Exl3SpecStage:
    """One device-wide EXL3 scratch arena reused by every target MoE layer."""

    w13_trellis: torch.Tensor
    w13_suh: torch.Tensor
    w13_svh: torch.Tensor
    w13_marker: torch.Tensor
    w2_trellis: torch.Tensor
    w2_suh: torch.Tensor
    w2_svh: torch.Tensor
    w2_marker: torch.Tensor
    expert_map: torch.Tensor
    layer: Any
    inners: list[dict[str, Any]]
    capacity: int


@dataclass
class _Exl3LRUCache:
    """Per-layer fixed-address VRAM cache backed by complete UVA expert rows."""

    w13_trellis: torch.Tensor
    w13_suh: torch.Tensor
    w13_svh: torch.Tensor
    w13_marker: torch.Tensor
    w2_trellis: torch.Tensor
    w2_suh: torch.Tensor
    w2_svh: torch.Tensor
    w2_marker: torch.Tensor
    source_map: torch.Tensor
    cache_map: torch.Tensor
    cold_map: torch.Tensor
    hit_map: torch.Tensor
    miss_map: torch.Tensor
    slot_global_ids: torch.Tensor
    slot_ages: torch.Tensor
    slot_segments: torch.Tensor
    clock: torch.Tensor
    miss_local_ids: torch.Tensor
    miss_slots: torch.Tensor
    counters: torch.Tensor
    copy_stream: torch.cuda.Stream
    layer: Any
    hybrid_layer: Any
    inners: list[dict[str, Any]]
    local_to_global: torch.Tensor
    capacity: int
    protected_capacity: int
    max_update_tokens: int
    expert_bytes: int
    layer_name: str


_EXL3_LRU_CACHES: list[_Exl3LRUCache] = []
_EXL3_LRU_COPY_STREAMS: dict[str, torch.cuda.Stream] = {}
_EXL3_LRU_CAPACITY_PLANS: dict[str, dict[str, Any]] = {}
_EXL3_LRU_APPLY_CALLS = 0
_EXL3_LRU_TRACE_HANDLES: dict[str, Any] = {}
_EXL3_MOE_POINTER_NAMES = (
    "gate_trellis",
    "gate_suh",
    "gate_svh",
    "up_trellis",
    "up_suh",
    "up_svh",
    "down_trellis",
    "down_suh",
    "down_svh",
)


_EXL3_SPEC_STAGE_CACHE: dict[tuple[Any, ...], _Exl3SpecStage] = {}


def _resolve_exl3_lru_capacity(
    default_capacity: int, layer_name: str, device: torch.device
) -> int:
    """Resolve an optional byte-budgeted, per-layer expert-cache plan.

    The optimizer emits one plan per physical GPU for EP and one global plan
    for TP.  TP rank 1 deliberately falls back to device 0's global plan,
    because both ranks cache complementary halves of the same expert IDs.
    """
    path = os.environ.get("EXL3_LRU_CAPACITY_PLAN", "")
    if not path:
        return default_capacity
    raw = _EXL3_LRU_CAPACITY_PLANS.get(path)
    if raw is None:
        with open(path, encoding="utf-8") as handle:
            raw = json.load(handle)
        _EXL3_LRU_CAPACITY_PLANS[path] = raw
    match = re.search(r"(?:^|\.)layers\.(\d+)(?:\.|$)", layer_name)
    if match is None:
        raise ValueError(f"cannot resolve EXL3 cache-plan layer: {layer_name}")
    layer_id = match.group(1)
    device_id = str(device.index if device.index is not None else torch.cuda.current_device())
    devices = raw.get("devices", {})
    device_plan = devices.get(device_id)
    if device_plan is None and raw.get("ownership") == "global":
        device_plan = devices.get("0")
    if device_plan is None:
        raise KeyError(
            f"EXL3 cache plan has no device {device_id}: {path}"
        )
    capacities = device_plan.get("capacity_plan", device_plan)
    if layer_id not in capacities:
        raise KeyError(
            f"EXL3 cache plan has no layer {layer_id} for device {device_id}: {path}"
        )
    capacity = int(capacities[layer_id])
    if capacity <= 0:
        raise ValueError(
            f"EXL3 cache plan capacity must be positive, got {capacity} for {layer_name}"
        )
    return capacity


@triton.jit
def _build_exl3_spec_stage_map_kernel(
    global_ids_ptr,
    stage_map_ptr,
    num_ids: tl.constexpr,
    global_num_experts: tl.constexpr,
    block_size: tl.constexpr,
):
    route = tl.program_id(0)
    global_id = tl.load(global_ids_ptr + route)
    prior = tl.arange(0, block_size)
    prior_ids = tl.load(global_ids_ptr + prior, mask=prior < route, other=-1)
    is_first = tl.sum((prior < route) & (prior_ids == global_id), axis=0) == 0
    valid = is_first & (global_id >= 0) & (global_id < global_num_experts)
    tl.store(stage_map_ptr + global_id, route, mask=valid)


@triton.jit
def _gather_unique_exl3_spec_stage_rows_kernel(
    source_ptr,
    output_ptr,
    physical_ids_ptr,
    global_ids_ptr,
    row_size: tl.constexpr,
    num_ids: tl.constexpr,
    scan_size: tl.constexpr,
    block_size: tl.constexpr,
):
    route = tl.program_id(0)
    block = tl.program_id(1)
    global_id = tl.load(global_ids_ptr + route)
    prior = tl.arange(0, scan_size)
    prior_ids = tl.load(global_ids_ptr + prior, mask=prior < route, other=-1)
    is_first = tl.sum((prior < route) & (prior_ids == global_id), axis=0) == 0
    offsets = block * block_size + tl.arange(0, block_size)
    mask = is_first & (offsets < row_size)
    physical_id = tl.load(physical_ids_ptr + route)
    values = tl.load(source_ptr + physical_id * row_size + offsets, mask=mask)
    tl.store(output_ptr + route * row_size + offsets, values, mask=mask)


def fused_moe_row_tile_enabled() -> bool:
    """GPU row tiles instead of LinearEXL3 fallback. Prefill-only; decode stays one launch.

    Measured slower than the 128-row fallback at MNBT=1024 (8 full-grid launches).
    Default off; keep for MNBT > temp rows if a later bump still overflows.
    """
    return os.environ.get("EXL3_MOE_ROW_TILE", "0") != "0"


def temp_rows_fused() -> int:
    raw = os.environ.get("EXL3_TEMP_ROWS_FUSED", "").strip()
    if not raw:
        return int(TEMP_ROWS_FUSED)
    return max(1, int(raw))

def sorted_fat_fallback_enabled() -> bool:
    """Use the existing expert-sorted buffers for oversized prefill experts."""
    return os.environ.get("EXL3_FAT_SORTED", "0") != "0"

def batched_fat_fallback_enabled() -> bool:
    """Enable E1 batched fat experts; implies expert-sorted routing."""
    return os.environ.get("EXL3_FAT_BATCHED", "0") != "0"

def fat_kernel_enabled() -> bool:
    """Enable the E2 fat kernel; implies E1 batching and sorted routing."""
    return os.environ.get("EXL3_FAT_KERNEL", "0") != "0"


def fat_expert_log_enabled() -> bool:
    return os.environ.get("EXL3_FAT_EXPERT_LOG", "1") != "0"

def configured_fat_tier() -> str:
    """Highest fat tier the env requests: kernel > batched > sorted > legacy."""
    if fat_kernel_enabled():
        return "kernel"
    if batched_fat_fallback_enabled():
        return "batched"
    if sorted_fat_fallback_enabled():
        return "sorted"
    return "legacy"


def exl3_fat_symbols() -> tuple[bool, bool, bool]:
    """(exl3_moe, exl3_fat_gemm, exl3_fat_gemm_scatter) availability."""
    try:
        ext = load_exllamav3_ext()
    except Exception:
        return False, False, False
    return (
        hasattr(ext, "exl3_moe"),
        hasattr(ext, "exl3_fat_gemm"),
        hasattr(ext, "exl3_fat_gemm_scatter"),
    )


def exl3_device_capability() -> tuple[int, int]:
    """CUDA capability of the current device; (-1, -1) without a GPU."""
    if not torch.cuda.is_available():
        return -1, -1
    try:
        return tuple(int(v) for v in torch.cuda.get_device_capability())
    except Exception:
        return -1, -1


def _exl3_tp_rank_size() -> tuple[int, int]:
    """TP identity so each rank's diag line is attributable; -1 outside vLLM."""
    try:
        from vllm.distributed import (
            get_tensor_model_parallel_rank,
            get_tensor_model_parallel_world_size,
        )

        return (
            int(get_tensor_model_parallel_rank()),
            int(get_tensor_model_parallel_world_size()),
        )
    except Exception:
        return -1, 1


def resolve_exl3_fat_tier(
    shared_suh: bool,
    symbols: tuple[bool, bool, bool] | None = None,
    *,
    fat_kernel_compatible: bool = True,
) -> tuple[str, str]:
    """Map the configured fat tier onto what this image + checkpoint can run.

    Order matters: the checkpoint cap comes first. E1 batched and the E2
    kernel run gate+up as one stacked GEMM behind a single input Hadamard
    (gate.suh), so a checkpoint without shared SUH caps the tier at sorted —
    a legitimate lower tier that needs no fat symbols, whatever the image.
    Only when the kernel would actually run does a missing symbol become an
    image/flag mismatch, failing closed here at load instead of mid-prefill.
    """
    configured = configured_fat_tier()
    if configured == "legacy":
        return configured, "none_requested"
    if not shared_suh and configured in ("kernel", "batched"):
        return "sorted", "shared_suh_absent"
    if configured == "kernel" and not fat_kernel_compatible:
        return "batched", "fat_kernel_requires_k4_mcg"
    if symbols is None:
        symbols = exl3_fat_symbols()
    if configured == "kernel":
        missing = [
            name
            for name, present in zip(
                ("exl3_fat_gemm", "exl3_fat_gemm_scatter"), symbols[1:]
            )
            if not present
        ]
        if missing:
            raise RuntimeError(
                "EXL3_FAT_KERNEL=1 requires exllamav3_ext."
                + "/".join(missing)
                + "; this image was built without the fat kernel — unset "
                "EXL3_FAT_KERNEL or serve an E2 image"
            )
    return configured, f"{configured}_ok"


def exl3_fat_diag() -> dict[str, Any]:
    """Snapshot of the E2 diagnostics; the key set is EXL3_FAT_DIAG_KEYS."""
    diag = dict(_EXL3_FAT_DIAG)
    diag["fallback_calls"] = dict(_EXL3_FAT_DIAG["fallback_calls"])
    diag["fallback_reasons"] = dict(_EXL3_FAT_DIAG["fallback_reasons"])
    diag.update(
        fat_stat_layers=_FAT_STATS["layers"],
        fat_layers=_FAT_STATS["fat_layers"],
        fat_expert_slots=_FAT_STATS["fat_experts"],
        max_rows=_FAT_STATS["max_rows"],
    )
    return diag


def _exl3_fat_diag_line() -> str:
    d = exl3_fat_diag()
    parts = [
        f"schema={d['schema']}",
        f"configured_tier={d['configured_tier']}",
        f"effective_tier={d['effective_tier']}",
        f"tier_reason={d['tier_reason']}",
        f"shared_suh={int(d['shared_suh'])}",
        f"shared_suh_layers={d['shared_suh_layers']}/{d['moe_layers_loaded']}",
        f"sym_exl3_moe={int(d['sym_exl3_moe'])}",
        f"sym_fat_gemm={int(d['sym_fat_gemm'])}",
        f"sym_fat_gemm_scatter={int(d['sym_fat_gemm_scatter'])}",
        f"cap={d['cap_major']}.{d['cap_minor']}",
        f"cap_ok={int(d['cap_ok'])}",
        f"tp_rank={d['tp_rank']} tp_size={d['tp_size']}",
        f"prefill_layer_calls={d['prefill_layer_calls']}",
        f"thin_calls={d['thin_calls']}",
        f"row_tile_calls={d['row_tile_calls']}",
        "fallback_calls="
        + ",".join(f"{t}={d['fallback_calls'][t]}" for t in _FAT_TIERS),
        "fallback_reasons="
        + (
            ",".join(f"{r}={n}" for r, n in sorted(d["fallback_reasons"].items()))
            or "none"
        ),
        f"fat_expert_runs={d['fat_expert_runs']}",
        f"direct_calls={d['direct_calls']}",
        f"scatter_calls={d['scatter_calls']}",
        f"fat_layers={d['fat_layers']}",
        f"fat_expert_slots={d['fat_expert_slots']}",
        f"max_rows={d['max_rows']}",
        f"fused_temps_allocs={d['fused_temps_allocs']}",
        f"fused_temps_bytes={d['fused_temps_bytes']}",
        f"fat_scratch_allocs={d['fat_scratch_allocs']}",
        f"fat_scratch_bytes={d['fat_scratch_bytes']}",
        f"fat_scratch_peak_bytes={d['fat_scratch_peak_bytes']}",
    ]
    return " ".join(parts)


def _record_exl3_fat_reason(reason: str) -> None:
    reasons = _EXL3_FAT_DIAG["fallback_reasons"]
    reasons[reason] = reasons.get(reason, 0) + 1


def _record_exl3_fat_tier(layer: torch.nn.Module, tier: str, reason: str) -> None:
    _EXL3_FAT_DIAG["fallback_calls"][tier] += 1
    _record_exl3_fat_reason(reason)
    layer._exl3_last_fat_fallback = tier
    layer._exl3_last_fat_reason = reason


def _record_exl3_fat_resolution(layer: torch.nn.Module) -> None:
    """Resolve the E2 tier once per MoE layer at weight load and log once.

    Per-layer truth lands on layer._exl3_fat_effective_tier; the module state
    mirrors the most recent load. A resolution that changes between layers of
    one model is a checkpoint property worth a loud line, not a silent one.
    """
    global _exl3_fat_tier_logged
    shared_suh = bool(getattr(layer, "_exl3_shared_w13_suh", False))
    fat_kernel_compatible = (
        int(getattr(layer, "_exl3_bits", 4)) == 4
        and str(getattr(layer, "_exl3_codebook", "mcg")) == "mcg"
    )
    effective_tier, tier_reason = resolve_exl3_fat_tier(
        shared_suh, fat_kernel_compatible=fat_kernel_compatible
    )
    layer._exl3_fat_effective_tier = effective_tier
    layer._exl3_fat_tier_reason = tier_reason

    diag = _EXL3_FAT_DIAG
    sym_moe, sym_gemm, sym_scatter = exl3_fat_symbols()
    cap_major, cap_minor = exl3_device_capability()
    diag["moe_layers_loaded"] += 1
    if shared_suh:
        diag["shared_suh_layers"] += 1
    diag["shared_suh"] = diag["shared_suh_layers"] == diag["moe_layers_loaded"]
    diag["configured_tier"] = configured_fat_tier()
    diag["sym_exl3_moe"] = sym_moe
    diag["sym_fat_gemm"] = sym_gemm
    diag["sym_fat_gemm_scatter"] = sym_scatter
    diag["cap_major"] = cap_major
    diag["cap_minor"] = cap_minor
    # LinearEXL3 (and the fat GEMM built on it) needs >= Ampere; GB10 is SM121.
    diag["cap_ok"] = (cap_major, cap_minor) >= (8, 0)
    diag["tp_rank"], diag["tp_size"] = _exl3_tp_rank_size()

    if diag["tier_reason"] == "unresolved":
        diag["effective_tier"] = effective_tier
        diag["tier_reason"] = tier_reason
    elif diag["effective_tier"] != effective_tier:
        logger.warning(
            "exl3 e2 diag tier changed %s -> %s (%s): %s",
            diag["effective_tier"],
            effective_tier,
            tier_reason,
            _exl3_fat_diag_line(),
        )
        diag["effective_tier"] = effective_tier
        diag["tier_reason"] = tier_reason
    if not _exl3_fat_tier_logged:
        _exl3_fat_tier_logged = True
        if (
            diag["effective_tier"] != diag["configured_tier"]
            and diag["configured_tier"] != "legacy"
        ):
            logger.warning("exl3 e2 diag degraded %s", _exl3_fat_diag_line())
        else:
            logger.info("exl3 e2 diag %s", _exl3_fat_diag_line())


def reset_exl3_fat_diag_counters() -> None:
    """Zero the E2 runtime counters; load-time fields and live bytes stay.

    Scratch current/peak restart from the resident cache so a windowed read
    (e.g. exactly one cold request) still reports honest byte counts.
    """
    diag = _EXL3_FAT_DIAG
    for key in (
        "prefill_layer_calls",
        "thin_calls",
        "row_tile_calls",
        "fat_expert_runs",
        "direct_calls",
        "scatter_calls",
        "fat_scratch_allocs",
    ):
        diag[key] = 0
    diag["fallback_calls"] = {tier: 0 for tier in _FAT_TIERS}
    diag["fallback_reasons"] = {}
    diag["fat_scratch_bytes"] = sum(_FAT_SCRATCH_BYTES.values())
    diag["fat_scratch_peak_bytes"] = diag["fat_scratch_bytes"]


def reset_exl3_fat_expert_stats() -> None:
    _FAT_STATS["layers"] = 0
    _FAT_STATS["fat_layers"] = 0
    _FAT_STATS["fat_experts"] = 0
    _FAT_STATS["max_rows"] = 0
    _FAT_STATS["sum_max_rows"] = 0
    _FAT_STATS["hist"] = [0] * (len(_FAT_BUCKET_EDGES) + 1)


def _fat_bucket(n: int) -> int:
    for i, edge in enumerate(_FAT_BUCKET_EDGES):
        if n <= edge:
            return i
    return len(_FAT_BUCKET_EDGES)


def record_exl3_fat_expert_stats(
    counts: torch.Tensor,
    *,
    max_rows: int | None = None,
    counts_host: list[int] | None = None,
) -> dict[str, Any]:
    """Prefill-only routing stats. Reuse an existing host copy when available."""
    if counts_host is None:
        if max_rows is None:
            max_rows = int(counts.max().item())
        n_fat = int((counts > temp_rows_fused()).sum().item())
    else:
        if max_rows is None:
            max_rows = max(counts_host, default=0)
        cap = temp_rows_fused()
        n_fat = sum(n > cap for n in counts_host)
    st = _FAT_STATS
    st["layers"] += 1
    st["sum_max_rows"] += max_rows
    st["hist"][_fat_bucket(max_rows)] += 1
    if max_rows > st["max_rows"]:
        st["max_rows"] = max_rows
    if n_fat:
        st["fat_layers"] += 1
        st["fat_experts"] += n_fat
    # 42 routed-MoE layers per engine step (MoE from layer 3 of 45).
    if st["layers"] % 42 == 0:
        avg = st["sum_max_rows"] / st["layers"]
        le128 = sum(st["hist"][:4])
        gt128 = sum(st["hist"][4:])
        logger.info(
            "exl3 fat-expert P0: layers=%d fat_layers=%d (%.1f%%) fat_expert_slots=%d "
            "max_rows=%d avg_max_rows=%.1f hist_le128=%d hist_gt128=%d hist=%s",
            st["layers"],
            st["fat_layers"],
            100.0 * st["fat_layers"] / st["layers"],
            st["fat_experts"],
            st["max_rows"],
            avg,
            le128,
            gt128,
            st["hist"],
        )
        logger.info("exl3 e2 diag %s", _exl3_fat_diag_line())
    return {
        "max_rows": max_rows,
        "n_fat": n_fat,
        "layers": st["layers"],
        "fat_layers": st["fat_layers"],
    }


def _narrow_tp(tensor: torch.Tensor, dim: int, tp_rank: int, tp_size: int) -> torch.Tensor:
    if tp_size <= 1:
        return tensor
    size = int(tensor.shape[dim])
    if size % tp_size:
        raise ValueError(
            f"EXL3 TP shard: dim {dim} size {size} is not divisible by tp={tp_size}"
        )
    chunk = size // tp_size
    return tensor.narrow(dim, chunk * tp_rank, chunk).contiguous()


def shard_exl3_col(loaded: torch.Tensor, suffix: str, tp_rank: int, tp_size: int) -> torch.Tensor:
    """Gate/up: trellis dim 1 and svh dim 0 are column-parallel."""
    if suffix == "trellis":
        return _narrow_tp(loaded, 1, tp_rank, tp_size)
    if suffix == "svh":
        return _narrow_tp(loaded, 0, tp_rank, tp_size)
    return loaded.contiguous()


def shard_exl3_row(loaded: torch.Tensor, suffix: str, tp_rank: int, tp_size: int) -> torch.Tensor:
    """Down: trellis dim 0 and suh dim 0 are row-parallel."""
    if suffix == "trellis":
        return _narrow_tp(loaded, 0, tp_rank, tp_size)
    if suffix == "suh":
        return _narrow_tp(loaded, 0, tp_rank, tp_size)
    return loaded.contiguous()


def _install_exllamav3_namespace() -> None:
    """Load LinearEXL3 without running exllamav3/__init__.py (FlashAttention)."""
    if "exllamav3.modules.quant.exl3" in sys.modules:
        return
    import exllamav3_ext  # noqa: F401  — compiled extension must exist

    spec = importlib.util.find_spec("exllamav3")
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError("exllamav3 package is not installed in this image")
    package_root = Path(list(spec.submodule_search_locations)[0])

    # Stub only packages whose __init__.py pulls FlashAttention / serving extras.
    # Leave .ext, .util, and .modules.quant as real modules so LinearEXL3 loads.
    for name, path in (
        ("exllamav3", package_root),
        ("exllamav3.modules", package_root / "modules"),
        ("exllamav3.model", package_root / "model"),
    ):
        if name in sys.modules:
            continue
        module = types.ModuleType(name)
        module.__file__ = str(path / "__init__.py")
        module.__package__ = name
        module.__path__ = [str(path)]
        sys.modules[name] = module

    if "exllamav3.model.config" not in sys.modules:
        config = types.ModuleType("exllamav3.model.config")
        config.__file__ = str(package_root / "model/config.py")
        config.__package__ = "exllamav3.model"
        config.Config = type("Config", (), {})

        class _InferParams:
            """Minimal current-ExLlamaV3 inference policy for standalone linears."""

            def __init__(self) -> None:
                self.no_reconstruct = False
                self.mgemm_K_threshold = int(
                    os.environ.get("EXL3_MGEMM_K_THRESHOLD", "6")
                )
                self.mgemm_n_threshold = int(
                    os.environ.get("EXL3_MGEMM_N_THRESHOLD", "8192")
                )

            def use_mgemm(
                self,
                k: int,
                out_features: int,
                mul1: bool = False,
                device: torch.device | None = None,
            ) -> bool:
                del device
                if not mul1:
                    return True
                return k >= self.mgemm_K_threshold or (
                    self.mgemm_n_threshold > 0
                    and out_features < self.mgemm_n_threshold
                )

        class _NullConfig:
            def __init__(self) -> None:
                self.infer_params = _InferParams()

        config.InferParams = _InferParams
        config.NullConfig = _NullConfig
        sys.modules[config.__name__] = config


def load_linear_exl3_cls():
    _install_exllamav3_namespace()
    return importlib.import_module("exllamav3.modules.quant.exl3").LinearEXL3


def make_linear_exl3(
    trellis: torch.Tensor,
    suh: torch.Tensor,
    svh: torch.Tensor,
    marker: torch.Tensor,
    *,
    codebook: str = "mcg",
    out_dtype: torch.dtype = torch.float16,
):
    """Build a LinearEXL3 over already-sharded packed tensors. No BF16 expand."""
    cls = load_linear_exl3_cls()
    if codebook not in ("mcg", "mul1"):
        raise ValueError(f"unsupported EXL3 codebook={codebook!r}")
    return cls(
        config=None,
        in_features=int(suh.numel()),
        out_features=int(svh.numel()),
        trellis=trellis.contiguous(),
        suh=suh.contiguous(),
        svh=svh.contiguous(),
        mcg=marker.contiguous() if codebook == "mcg" else None,
        mul1=marker.contiguous() if codebook == "mul1" else None,
        out_dtype=out_dtype,
        transformers_fix=True,
    )


def execute_exl3_linear(
    x: torch.Tensor,
    trellis: torch.Tensor,
    suh: torch.Tensor,
    svh: torch.Tensor,
    marker: torch.Tensor,
    *,
    codebook: str = "mcg",
    out_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Real EXL3 expert GEMM entry (LinearEXL3 / exllamav3_ext)."""
    inner = make_linear_exl3(
        trellis, suh, svh, marker, codebook=codebook, out_dtype=torch.float16
    )
    return inner.forward(x.contiguous().half(), {}, out_dtype=out_dtype)


def fused_moe_enabled() -> bool:
    return os.environ.get("EXL3_FUSED_MOE", "1") != "0"


def load_exllamav3_ext():
    import exllamav3_ext

    return exllamav3_ext


def _exl3_moe_accepts_num_active(fn) -> bool:
    try:
        import inspect

        if "num_active" in inspect.signature(fn).parameters:
            return True
    except (TypeError, ValueError):
        pass
    doc = getattr(fn, "__doc__", None) or ""
    return "num_active" in doc or "arg29" in doc or doc.count("arg") >= 30


def pin_exl3_expert_map(
    layer: torch.nn.Module, device: torch.device
) -> torch.Tensor | None:
    """Move expert_map onto `device` once. CUDA graph capture forbids a CPU→GPU copy."""
    emap = getattr(layer, "expert_map", None)
    if emap is None:
        return None
    if emap.device != device or emap.dtype != torch.long:
        emap = emap.to(device=device, dtype=torch.long)
        # Current RoutedExperts exposes ``expert_map`` as a read-only
        # property backed by the registered ``_expert_map`` buffer.  Replace
        # that buffer directly so future property reads stay graph-safe.
        if "_expert_map" in getattr(layer, "_buffers", {}):
            layer._buffers["_expert_map"] = emap
        else:
            setattr(layer, "_expert_map", emap)
    return emap


def map_topk_to_local(
    ids: torch.Tensor,
    n_local: int,
    expert_map: torch.Tensor | None,
) -> torch.Tensor:
    """ids (T, K) global expert ids → local ids, invalid/non-local → n_local sentinel.

    `expert_map` must already live on `ids.device` (see pin_exl3_expert_map).
    """
    flat = ids.reshape(-1)
    if expert_map is None:
        invalid = (flat < 0) | (flat >= n_local)
        return torch.where(invalid, flat.new_full(flat.shape, n_local), flat)
    if expert_map.device != flat.device or expert_map.dtype != torch.long:
        raise RuntimeError(
            "EXL3 expert_map is not pinned to the hidden-state device; "
            "call pin_exl3_expert_map before fused apply (CUDA graphs forbid the copy)"
        )
    n_global = int(expert_map.numel())
    safe = flat.clamp(min=0, max=max(n_global - 1, 0))
    mapped = expert_map[safe] if n_global else flat.new_full(flat.shape, n_local)
    invalid = (flat < 0) | (flat >= n_global) | (mapped < 0) | (mapped >= n_local)
    return torch.where(invalid, flat.new_full(flat.shape, n_local), mapped)


@triton.jit
def _prepare_exl3_decode_routes_kernel(
    ids_ptr,
    weights_ptr,
    expert_map_ptr,
    expert_count_ptr,
    token_sorted_ptr,
    weight_sorted_ptr,
    n_ids: tl.constexpr,
    topk: tl.constexpr,
    n_local: tl.constexpr,
    n_global: tl.constexpr,
    has_map: tl.constexpr,
    id_block: tl.constexpr,
    count_block: tl.constexpr,
):
    """Build the three fused-MoE routing buffers in one graph-safe CTA."""
    count_offsets = tl.arange(0, count_block)
    tl.store(
        expert_count_ptr + count_offsets,
        0,
        mask=count_offsets <= n_local,
    )
    tl.debug_barrier()

    offsets = tl.arange(0, id_block)
    valid = offsets < n_ids
    global_id = tl.load(ids_ptr + offsets, mask=valid, other=-1)
    if has_map:
        safe = tl.minimum(tl.maximum(global_id, 0), n_global - 1)
        local_id = tl.load(expert_map_ptr + safe, mask=valid, other=-1)
        valid_local = (
            valid
            & (global_id >= 0)
            & (global_id < n_global)
            & (local_id >= 0)
            & (local_id < n_local)
        )
    else:
        local_id = global_id
        valid_local = valid & (global_id >= 0) & (global_id < n_local)
    local_id = tl.where(valid_local, local_id, n_local).to(tl.int32)
    tl.atomic_add(expert_count_ptr + local_id, 1, mask=valid)

    # The source position is a deterministic tie-breaker and also encodes the
    # permutation needed for token IDs and weights.
    encoded = local_id * id_block + offsets
    encoded = tl.where(
        valid,
        encoded,
        (n_local + 1) * id_block + offsets,
    )
    ordered = tl.sort(encoded, dim=0)
    source_offset = ordered % id_block
    sorted_valid = offsets < n_ids
    sorted_weight = tl.load(
        weights_ptr + source_offset,
        mask=sorted_valid,
    )
    tl.store(
        token_sorted_ptr + offsets,
        source_offset // topk,
        mask=sorted_valid,
    )
    tl.store(weight_sorted_ptr + offsets, sorted_weight, mask=sorted_valid)


def _prepare_exl3_decode_routes(
    ids: torch.Tensor,
    weights: torch.Tensor,
    expert_map: torch.Tensor | None,
    n_local: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    n_ids = int(ids.numel())
    expert_count = torch.empty(
        n_local + 1,
        dtype=torch.long,
        device=ids.device,
    )
    token_sorted = torch.empty(n_ids, dtype=torch.long, device=ids.device)
    weight_sorted = torch.empty(
        n_ids,
        dtype=torch.float16,
        device=ids.device,
    )
    dummy_map = ids if expert_map is None else expert_map
    _prepare_exl3_decode_routes_kernel[(1,)](
        ids,
        weights,
        dummy_map,
        expert_count,
        token_sorted,
        weight_sorted,
        n_ids=n_ids,
        topk=int(ids.shape[-1]),
        n_local=n_local,
        n_global=(0 if expert_map is None else int(expert_map.numel())),
        has_map=(expert_map is not None),
        id_block=triton.next_power_of_2(n_ids),
        count_block=triton.next_power_of_2(n_local + 1),
        num_warps=8,
    )
    return expert_count, token_sorted, weight_sorted


def apply_exl3_python_loop(
    x2d: torch.Tensor,
    ids: torch.Tensor,
    weights: torch.Tensor,
    inners: list[dict[str, Any]],
    expert_map: torch.Tensor | None,
    limit: float,
    *,
    only_experts: set[int] | None = None,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Unique-expert LinearEXL3 loop. `only_experts` is local ids (fat-expert fallback)."""
    tokens, hidden = x2d.shape
    if out is None:
        out = _exl3_moe_out_zeros(tokens, hidden, x2d.device)
    unique = torch.unique(ids)
    for raw in unique.tolist():
        e_raw = int(raw)
        if e_raw < 0:
            continue
        e = e_raw
        if expert_map is not None:
            mapped = int(expert_map[e].item()) if expert_map.numel() > e else e
            if mapped < 0:
                continue
            e = mapped
        if e >= len(inners):
            continue
        if only_experts is not None and e not in only_experts:
            continue
        token_idx, k_pos = (ids == int(raw)).nonzero(as_tuple=True)
        h = x2d.index_select(0, token_idx)
        pack = inners[e]
        gate = pack["gate"].forward(h.contiguous().half(), {}, out_dtype=torch.float32)
        up = pack["up"].forward(h.contiguous().half(), {}, out_dtype=torch.float32)
        act = F.silu(gate.clamp(max=limit)) * up.clamp(min=-limit, max=limit)
        down = pack["down"].forward(act.contiguous().half(), {}, out_dtype=torch.float32)
        scale = weights[token_idx, k_pos].unsqueeze(-1).to(dtype=torch.float32)
        _exl3_fx_index_add_(out, token_idx, down * scale)
    return out

def apply_exl3_sorted_fat(
    xh: torch.Tensor,
    token_sorted: torch.Tensor,
    weight_sorted: torch.Tensor,
    counts_host: list[int],
    inners: list[dict[str, Any]],
    limit: float,
    cap: int,
    out: torch.Tensor,
) -> torch.Tensor:
    """Run oversized experts from contiguous slices of the existing sort.

    One `counts.tolist()` in the caller replaces the legacy per-expert
    `unique.tolist()`, expert-map `.item()`, and `(ids == expert).nonzero()`
    synchronizations. Slices are views; only the LinearEXL3 inputs and outputs
    allocate, as they do in the legacy fallback.
    """
    offset = 0
    for e, n_rows in enumerate(counts_host):
        start = offset
        offset += n_rows
        if n_rows <= cap:
            continue
        _EXL3_FAT_DIAG["fat_expert_runs"] += 1
        token_idx = token_sorted[start:offset]
        h = xh.index_select(0, token_idx)
        pack = inners[e]
        gate = pack["gate"].forward(h, {}, out_dtype=torch.float32)
        up = pack["up"].forward(h, {}, out_dtype=torch.float32)
        act = F.silu(gate.clamp(max=limit)) * up.clamp(min=-limit, max=limit)
        down = pack["down"].forward(
            act.contiguous().half(), {}, out_dtype=torch.float32
        )
        scale = weight_sorted[start:offset].unsqueeze(-1).to(dtype=torch.float32)
        _exl3_fx_index_add_(out, token_idx, down * scale)
    return out


def _fat_scratch(
    device: torch.device,
    rows: int,
    gate: Any,
) -> dict[str, torch.Tensor]:
    """Return shared prefill scratch, growing once if the configured chunk grows."""
    hidden = int(gate.in_features)
    intermediate = int(gate.out_features)
    configured = int(
        os.environ.get(
            "EXL3_FAT_SCRATCH_ROWS",
            os.environ.get("MAX_NUM_BATCHED_TOKENS", "0"),
        )
        or 0
    )
    needed = max(256, rows, configured)
    capacity = 1 << (needed - 1).bit_length()
    key = (
        str(device),
        hidden,
        intermediate,
        int(gate.K),
        int(gate.trellis.shape[-1]),
    )
    scratch = _FAT_SCRATCH_CACHE.get(key)
    if scratch is not None and int(scratch["h"].shape[0]) >= rows:
        return scratch

    in_tiles, out_tiles, k_words = map(int, gate.trellis.shape)
    scratch = {
        "packed13": torch.empty(
            (in_tiles, 2 * out_tiles, k_words),
            dtype=torch.int16,
            device=device,
        ),
        "svh13": torch.empty(
            2 * intermediate, dtype=torch.float16, device=device
        ),
        "w13": torch.empty(
            (hidden, 2 * intermediate), dtype=torch.float16, device=device
        ),
        "w2": torch.empty(
            (intermediate, hidden), dtype=torch.float16, device=device
        ),
        "h": torch.empty(
            (capacity, hidden), dtype=torch.float16, device=device
        ),
        "h13": torch.empty(
            (capacity, hidden), dtype=torch.float16, device=device
        ),
        "gate_up": torch.empty(
            (capacity, 2 * intermediate), dtype=torch.float32, device=device
        ),
        "act": torch.empty(
            (capacity, intermediate), dtype=torch.float32, device=device
        ),
        "act_h": torch.empty(
            (capacity, intermediate), dtype=torch.float16, device=device
        ),
        "h2": torch.empty(
            (capacity, intermediate), dtype=torch.float16, device=device
        ),
        "down": torch.empty(
            (capacity, hidden), dtype=torch.float32, device=device
        ),
    }
    _FAT_SCRATCH_CACHE[key] = scratch
    _FAT_SCRATCH_BYTES[key] = sum(
        t.numel() * t.element_size() for t in scratch.values()
    )
    diag = _EXL3_FAT_DIAG
    diag["fat_scratch_allocs"] += 1
    diag["fat_scratch_bytes"] = sum(_FAT_SCRATCH_BYTES.values())
    if diag["fat_scratch_bytes"] > diag["fat_scratch_peak_bytes"]:
        diag["fat_scratch_peak_bytes"] = diag["fat_scratch_bytes"]
    return scratch


def _stage_counts_to_host(
    counts: torch.Tensor,
) -> tuple[torch.Tensor, torch.cuda.Stream]:
    """Copy routing counts on a side stream before launching thin experts."""
    key = (str(counts.device), int(counts.numel()))
    cached = _FAT_COUNT_CACHE.get(key)
    if cached is None:
        host = torch.empty(
            int(counts.numel()), dtype=counts.dtype, device="cpu", pin_memory=True
        )
        stream = torch.cuda.Stream(device=counts.device)
        cached = (host, stream)
        _FAT_COUNT_CACHE[key] = cached
    host, stream = cached
    current = torch.cuda.current_stream(counts.device)
    with torch.cuda.stream(stream):
        stream.wait_stream(current)
        host.copy_(counts, non_blocking=True)
    return host, stream


def apply_exl3_batched_fat(
    xh: torch.Tensor,
    token_sorted: torch.Tensor,
    weight_sorted: torch.Tensor,
    counts_host: list[int],
    inners: list[dict[str, Any]],
    limit: float,
    cap: int,
    out: torch.Tensor,
    use_kernel: bool = False,
) -> torch.Tensor:
    """Run fat experts with persistent buffers and optional direct trellis GEMM."""
    ext = load_exllamav3_ext()
    offset = 0
    for e, n_rows in enumerate(counts_host):
        start = offset
        offset += n_rows
        if n_rows <= cap:
            continue
        _EXL3_FAT_DIAG["fat_expert_runs"] += 1

        token_idx = token_sorted[start:offset]
        gate = inners[e]["gate"]
        up = inners[e]["up"]
        down = inners[e]["down"]
        scratch = _fat_scratch(xh.device, n_rows, gate)
        intermediate = int(gate.out_features)

        h = scratch["h"][:n_rows]
        h13 = scratch["h13"][:n_rows]
        torch.index_select(xh, 0, token_idx, out=h)
        ext.had_r_128(h, h13, gate.suh, None, 1.0)

        packed13 = scratch["packed13"]
        out_tiles = int(gate.trellis.shape[1])
        packed13[:, :out_tiles].copy_(gate.trellis)
        packed13[:, out_tiles:].copy_(up.trellis)
        gate_up = scratch["gate_up"][:n_rows]
        svh13 = scratch["svh13"]
        svh13[:intermediate].copy_(gate.svh)
        svh13[intermediate:].copy_(up.svh)
        if use_kernel:
            if not hasattr(ext, "exl3_fat_gemm"):
                raise RuntimeError(
                    "EXL3_FAT_KERNEL=1 requires exllamav3_ext.exl3_fat_gemm"
                )
            ext.exl3_fat_gemm(
                h13, packed13, gate_up, svh13, gate.K, gate.mcg, gate.mul1
            )
            _EXL3_FAT_DIAG["direct_calls"] += 1
        else:
            w13 = scratch["w13"]
            ext.reconstruct(w13, packed13, gate.K, gate.mcg, gate.mul1)
            ext.hgemm(h13, w13, gate_up)
            ext.had_r_128(gate_up, gate_up, None, svh13, 1.0)

        gate_out = gate_up[:, :intermediate]
        up_out = gate_up[:, intermediate:]
        gate_out.clamp_(max=limit)
        up_out.clamp_(min=-limit, max=limit)
        act = scratch["act"][:n_rows]
        torch.sigmoid(gate_out, out=act)
        act.mul_(gate_out).mul_(up_out)
        act_h = scratch["act_h"][:n_rows]
        act_h.copy_(act)

        h2 = scratch["h2"][:n_rows]
        ext.had_r_128(act_h, h2, down.suh, None, 1.0)
        if use_kernel:
            if out.dtype == torch.int64:
                raise RuntimeError(
                    "EXL3_FAT_KERNEL scatter does not support EXL3_DETERMINISTIC_MOE"
                )
            if not hasattr(ext, "exl3_fat_gemm_scatter"):
                raise RuntimeError(
                    "EXL3_FAT_KERNEL=1 requires "
                    "exllamav3_ext.exl3_fat_gemm_scatter"
                )
            ext.exl3_fat_gemm_scatter(
                h2,
                down.trellis,
                out,
                down.svh,
                token_idx,
                weight_sorted[start:offset],
                down.K,
                down.mcg,
                down.mul1,
            )
            _EXL3_FAT_DIAG["scatter_calls"] += 1
        else:
            w2 = scratch["w2"]
            ext.reconstruct(w2, down.trellis, down.K, down.mcg, down.mul1)
            down_out = scratch["down"][:n_rows]
            ext.hgemm(h2, w2, down_out)
            ext.had_r_128(down_out, down_out, None, down.svh, 1.0)
            down_out.mul_(weight_sorted[start:offset].unsqueeze(-1))
            _exl3_fx_index_add_(out, token_idx, down_out)
    return out


def build_exl3_fused_state(layer: torch.nn.Module, inners: list[dict[str, Any]]) -> None:
    """Pointer tables + fused temps, once after load. No per-token alloc."""
    import exllamav3_ext

    device = layer.w13_trellis.device
    n_exp = len(inners)
    hidden = int(layer._exl3_hidden_size)
    intermediate = int(layer._exl3_intermediate_local)

    def _ptrs(which: str, attr: str) -> torch.Tensor:
        return torch.tensor(
            [int(getattr(pack[which], attr).data_ptr()) for pack in inners],
            dtype=torch.int64,
            device=device,
        )

    layer._exl3_ptrs = {
        "gate_trellis": _ptrs("gate", "trellis"),
        "gate_suh": _ptrs("gate", "suh"),
        "gate_svh": _ptrs("gate", "svh"),
        "up_trellis": _ptrs("up", "trellis"),
        "up_suh": _ptrs("up", "suh"),
        "up_svh": _ptrs("up", "svh"),
        "down_trellis": _ptrs("down", "trellis"),
        "down_suh": _ptrs("down", "suh"),
        "down_svh": _ptrs("down", "svh"),
        "mcg": bool(inners[0]["gate"].mcg),
        "mul1": bool(inners[0]["gate"].mul1),
    }
    for pack in inners:
        for which in ("gate", "up", "down"):
            inner = pack[which]
            if (
                bool(inner.mcg) != layer._exl3_ptrs["mcg"]
                or bool(inner.mul1) != layer._exl3_ptrs["mul1"]
            ):
                raise RuntimeError("EXL3 fused MoE requires one uniform codebook")
    idx = int(device.index) if device.index is not None else 0
    concurrency = int(exllamav3_ext.exl3_moe_max_concurrency(idx))
    if concurrency < 1:
        concurrency = 1
    rows = temp_rows_fused()
    key = (str(device), hidden, intermediate, concurrency, rows)
    temps = _FUSED_TEMP_CACHE.get(key)
    if temps is None:
        temps = (
            torch.empty((concurrency, rows, hidden), dtype=torch.float16, device=device),
            torch.empty((concurrency, rows, hidden), dtype=torch.float16, device=device),
            torch.empty((concurrency, rows, intermediate), dtype=torch.float16, device=device),
            torch.empty((concurrency, rows, intermediate), dtype=torch.float16, device=device),
        )
        _FUSED_TEMP_CACHE[key] = temps
        _EXL3_FAT_DIAG["fused_temps_allocs"] += 1
    # Layers share one cache entry, so assign (never accumulate) the bytes.
    _EXL3_FAT_DIAG["fused_temps_bytes"] = sum(
        t.numel() * t.element_size() for t in temps
    )
    layer._exl3_fused_temps = temps
    layer._exl3_fused_concurrency = concurrency
    layer._exl3_k = int(layer._exl3_bits)


def _get_exl3_spec_stage(
    layer: torch.nn.Module,
    capacity: int,
    global_num_experts: int,
    cache_tag: str = "spec",
) -> _Exl3SpecStage:
    """Allocate one stable GPU staging arena for small speculative batches."""
    device = layer.w13_trellis.device
    marker_suffix = str(getattr(layer, "_exl3_codebook", "mcg"))
    key = (
        str(device),
        capacity,
        global_num_experts,
        tuple(layer.w13_trellis.shape[1:]),
        tuple(layer.w13_suh.shape[1:]),
        tuple(layer.w13_svh.shape[1:]),
        tuple(layer.w2_trellis.shape[1:]),
        tuple(layer.w2_suh.shape[1:]),
        tuple(layer.w2_svh.shape[1:]),
        marker_suffix,
        int(layer._exl3_bits),
        cache_tag,
    )
    cached = _EXL3_SPEC_STAGE_CACHE.get(key)
    if cached is not None:
        return cached

    def empty_like_rows(source: torch.Tensor) -> torch.Tensor:
        return torch.empty(
            (capacity, *source.shape[1:]),
            dtype=source.dtype,
            device=source.device,
        )

    w13_trellis = empty_like_rows(layer.w13_trellis)
    w13_suh = empty_like_rows(layer.w13_suh)
    w13_svh = empty_like_rows(layer.w13_svh)
    w2_trellis = empty_like_rows(layer.w2_trellis)
    w2_suh = empty_like_rows(layer.w2_suh)
    w2_svh = empty_like_rows(layer.w2_svh)
    marker_value = (
        MCG_MARKER_SIGNED_INT32
        if marker_suffix == "mcg"
        else MUL1_MARKER_SIGNED_INT32
    )
    w13_marker = torch.full(
        (capacity, 2, 1), marker_value, dtype=torch.int32, device=device
    )
    w2_marker = torch.full(
        (capacity, 1), marker_value, dtype=torch.int32, device=device
    )
    expert_map = torch.full(
        (global_num_experts,), -1, dtype=torch.long, device=device
    )
    inners: list[dict[str, Any]] = []
    for expert in range(capacity):
        inners.append(
            {
                "gate": make_linear_exl3(
                    w13_trellis[expert, 0],
                    w13_suh[expert, 0],
                    w13_svh[expert, 0],
                    w13_marker[expert, 0],
                    codebook=marker_suffix,
                ),
                "up": make_linear_exl3(
                    w13_trellis[expert, 1],
                    w13_suh[expert, 1],
                    w13_svh[expert, 1],
                    w13_marker[expert, 1],
                    codebook=marker_suffix,
                ),
                "down": make_linear_exl3(
                    w2_trellis[expert],
                    w2_suh[expert],
                    w2_svh[expert],
                    w2_marker[expert],
                    codebook=marker_suffix,
                ),
            }
        )
    stage_layer = types.SimpleNamespace(
        w13_trellis=w13_trellis,
        _exl3_hidden_size=int(layer._exl3_hidden_size),
        _exl3_intermediate_local=int(layer._exl3_intermediate_local),
        _exl3_bits=int(layer._exl3_bits),
    )
    build_exl3_fused_state(stage_layer, inners)
    stage = _Exl3SpecStage(
        w13_trellis=w13_trellis,
        w13_suh=w13_suh,
        w13_svh=w13_svh,
        w13_marker=w13_marker,
        w2_trellis=w2_trellis,
        w2_suh=w2_suh,
        w2_svh=w2_svh,
        w2_marker=w2_marker,
        expert_map=expert_map,
        layer=stage_layer,
        inners=inners,
        capacity=capacity,
    )
    _EXL3_SPEC_STAGE_CACHE[key] = stage
    resident_bytes = sum(
        tensor.numel() * tensor.element_size()
        for tensor in (
            w13_trellis,
            w13_suh,
            w13_svh,
            w13_marker,
            w2_trellis,
            w2_suh,
            w2_svh,
            w2_marker,
            expert_map,
        )
    )
    logger.info(
        "Allocated shared EXL3 speculative stage: slots=%d bytes=%.2f MiB",
        capacity,
        resident_bytes / (1024 * 1024),
    )
    return stage


def _copy_exl3_stage_interval(
    source_layer: torch.nn.Module,
    stage: _Exl3SpecStage,
    begin: int,
    end: int,
) -> None:
    """DMA-copy one contiguous physical-expert interval into a VRAM stage."""
    width = end - begin
    for name in (
        "w13_trellis",
        "w13_suh",
        "w13_svh",
        "w2_trellis",
        "w2_suh",
        "w2_svh",
    ):
        getattr(stage, name)[:width].copy_(
            getattr(source_layer, name)[begin:end],
            non_blocking=True,
        )



def exl3_prefill_fixed_tiles_enabled() -> bool:
    return os.environ.get("EXL3_PREFILL_FIXED_TILES", "0").lower() in ("1", "true", "yes")


@triton.jit
def _exl3_build_row_tiles_kernel(
    token_sorted_ptr,
    weight_sorted_ptr,
    counts_ptr,
    prefix_ptr,
    tile_prefix_ptr,
    tile_tok_ptr,
    tile_w_ptr,
    n_exp,
    R,
    CAP: tl.constexpr,
):
    t = tl.program_id(0)
    e = tl.program_id(1)
    cnt = tl.load(counts_ptr + e)
    tc = tl.minimum(tl.maximum(cnt - t * CAP, 0), CAP)
    j = tl.arange(0, CAP)
    m = j < tc
    src = tl.load(prefix_ptr + e) + t * CAP + j
    dst = tl.load(tile_prefix_ptr + t * n_exp + e) + j
    tok = tl.load(token_sorted_ptr + src, mask=m, other=0)
    w = tl.load(weight_sorted_ptr + src, mask=m, other=0)
    tl.store(tile_tok_ptr + t * R + dst, tok, mask=m)
    tl.store(tile_w_ptr + t * R + dst, w, mask=m)


def _exl3_moe_fixed_tiles(
    fn: Any,
    xh: torch.Tensor,
    out: torch.Tensor,
    counts: torch.Tensor,
    token_sorted: torch.Tensor,
    weight_sorted: torch.Tensor,
    temps: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ptrs: dict[str, Any],
    k: int,
    limit: float,
    n_active_host: int | None,
    num_tiles: int,
) -> None:
    """Run every routed row through the fused kernel in num_tiles row tiles of `cap` rows per
    expert.  counts: [n_exp] int64 per-expert rows of the expert-sorted token_sorted; no host
    synchronisation (routing for all tiles is built by one Triton launch)."""
    n = int(counts.numel())
    R = int(token_sorted.numel())
    dev = counts.device
    cap = int(temps[0].shape[1])
    if num_tiles <= 0 or R == 0:
        return
    counts = counts.to(torch.long)
    prefix = torch.cumsum(counts, 0) - counts
    t_off = torch.arange(num_tiles, device=dev, dtype=torch.long)[:, None] * cap
    tile_ec = (counts[None, :] - t_off).clamp(min=0, max=cap)
    tile_prefix = (torch.cumsum(tile_ec, 1) - tile_ec).contiguous()
    if cap != triton.next_power_of_2(cap):
        raise RuntimeError("EXL3_PREFILL_FIXED_TILES requires a power-of-two EXL3_TEMP_ROWS_FUSED")
    tile_tok = torch.empty(num_tiles, R, dtype=token_sorted.dtype, device=dev)
    tile_w = torch.empty(num_tiles, R, dtype=weight_sorted.dtype, device=dev)
    _exl3_build_row_tiles_kernel[(num_tiles, n)](
        token_sorted, weight_sorted, counts, prefix, tile_prefix, tile_tok, tile_w,
        n, R, CAP=cap, num_warps=4,
    )
    ec_full = torch.zeros(num_tiles, n + 1, dtype=torch.long, device=dev)
    ec_full[:, :n] = tile_ec
    for t in range(num_tiles):
        _exl3_moe_launch(
            fn, xh, out, ec_full[t], tile_tok[t], tile_w[t],
            temps, ptrs, k, limit, n_active_host,
        )


def _exl3_dma_prefill_routes(x2d, ids, weights, source_layer, source_map):
    """Cold-expert routing for the staged prefill + ASYNC count copy to pinned host memory."""
    n_exp = int(source_layer.w13_trellis.shape[0])
    tokens = int(x2d.shape[0])
    local = map_topk_to_local(ids, n_exp, source_map)
    topk = int(ids.shape[-1])
    flat_token = torch.arange(tokens, device=x2d.device, dtype=torch.long).repeat_interleave(topk)
    flat_weight = weights.reshape(-1).to(dtype=torch.float16)
    order = local.argsort()
    token_sorted = flat_token[order]
    weight_sorted = flat_weight[order]
    expert_count = torch.zeros(n_exp + 1, dtype=torch.long, device=x2d.device)
    expert_count.scatter_add_(0, local.long(), torch.ones(local.shape, dtype=torch.long, device=x2d.device))
    host, stream = _stage_counts_to_host(expert_count)
    event = torch.cuda.Event()
    event.record(stream)
    return (token_sorted, weight_sorted, expert_count, host, event)


def _apply_exl3_dma_staged_prefill(
    x2d: torch.Tensor,
    ids: torch.Tensor,
    weights: torch.Tensor,
    source_layer: torch.nn.Module,
    source_map: torch.Tensor,
    copy_stream: torch.cuda.Stream,
    limit: float,
    routes: tuple | None = None,
) -> torch.Tensor | None:
    """Copy each cold expert interval once, then consume its sorted prompt rows.

    Direct UVA execution can reread a compressed expert once per 16-row kernel
    iteration and spends SM issue/L2 resources on PHB loads.  This eager-only
    prefill path sorts routes once, double-buffers contiguous expert intervals
    through CUDA's copy engine, and accumulates every interval into one output.
    Fat experts are evaluated only after their weights are resident in the
    staging arena, avoiding repeated PHB reads.  ``EXL3_PREFILL_DMA_FAT_MODE``
    selects either the fused row-tile path (default) or the regular sorted-fat
    fallback for a numerically closer A/B.  The supplied map may mask hot
    experts already evaluated from the decode LRU; this path deliberately does
    not mutate that cache.
    """
    group_size = int(os.environ.get("EXL3_PREFILL_DMA_STAGE_EXPERTS", "0"))
    if group_size <= 0:
        return None
    tokens = int(x2d.shape[0])
    min_tokens = int(os.environ.get("EXL3_PREFILL_DMA_MIN_TOKENS", "2"))
    if tokens < min_tokens:
        return None
    fat_mode = os.environ.get(
        "EXL3_PREFILL_DMA_FAT_MODE", "sorted"
    ).strip().lower()
    if fat_mode not in {"row_tile", "sorted"}:
        raise ValueError(
            "EXL3_PREFILL_DMA_FAT_MODE must be 'row_tile' or 'sorted', "
            f"got {fat_mode!r}"
        )

    import exllamav3_ext

    n_exp = int(source_layer.w13_trellis.shape[0])
    group_size = min(group_size, n_exp)
    if routes is not None:
        token_sorted, weight_sorted, expert_count, _host_counts, _count_event = routes
        _count_event.synchronize()
        counts_host = [int(value) for value in _host_counts[:n_exp].tolist()]
    else:
        local = map_topk_to_local(ids, n_exp, source_map)
        topk = int(ids.shape[-1])
        flat_token = torch.arange(
            tokens, device=x2d.device, dtype=torch.long
        ).repeat_interleave(topk)
        flat_weight = weights.reshape(-1).to(dtype=torch.float16)
        order = local.argsort()
        token_sorted = flat_token[order]
        weight_sorted = flat_weight[order]
        expert_count = torch.zeros(
            n_exp + 1, dtype=torch.long, device=x2d.device
        )
        expert_count.scatter_add_(
            0,
            local.long(),
            torch.ones(local.shape, dtype=torch.long, device=x2d.device),
        )

        # This is one intentional eager-prefill D2H dependency per routed layer.
        # It replaces repeated direct-UVA trellis fetches and is never used during
        # graph-captured decode.
        counts_host = [int(value) for value in expert_count[:n_exp].tolist()]
    offsets = [0]
    for count in counts_host:
        offsets.append(offsets[-1] + count)
    intervals = [
        (begin, min(begin + group_size, n_exp))
        for begin in range(0, n_exp, group_size)
        if offsets[min(begin + group_size, n_exp)] > offsets[begin]
    ]
    if not intervals:
        return _exl3_moe_out_zeros(tokens, x2d.shape[-1], x2d.device)

    global_num_experts = int(source_map.numel())
    stages = [
        _get_exl3_spec_stage(
            source_layer,
            group_size,
            global_num_experts,
            cache_tag=f"dma-prefill-{slot}",
        )
        for slot in range(min(2, len(intervals)))
    ]
    stage_counts = [
        torch.empty(group_size + 1, dtype=torch.long, device=x2d.device)
        for _ in stages
    ]
    ready = [torch.cuda.Event() for _ in stages]
    reusable = [torch.cuda.Event() for _ in stages]
    main_stream = torch.cuda.current_stream(x2d.device)
    for event in reusable:
        event.record(main_stream)

    def schedule_copy(index: int) -> None:
        begin, end = intervals[index]
        slot = index % len(stages)
        with torch.cuda.stream(copy_stream):
            copy_stream.wait_event(reusable[slot])
            _copy_exl3_stage_interval(
                source_layer, stages[slot], begin, end
            )
            ready[slot].record(copy_stream)

    output = _exl3_moe_out_zeros(tokens, x2d.shape[-1], x2d.device)
    xh = x2d.contiguous().half()
    fn = exllamav3_ext.exl3_moe
    accepts_num_active = _exl3_moe_accepts_num_active(fn)
    schedule_copy(0)
    for index, (begin, end) in enumerate(intervals):
        slot = index % len(stages)
        stage = stages[slot]
        main_stream.wait_event(ready[slot])
        counts = stage_counts[slot]
        counts.zero_()
        width = end - begin
        counts[:width].copy_(expert_count[begin:end])
        route_begin = offsets[begin]
        route_end = offsets[end]
        num_active = sum(count > 0 for count in counts_host[begin:end])
        interval_counts = counts_host[begin:end]
        interval_max_rows = max(interval_counts, default=0)
        cap = int(stage.layer._exl3_fused_temps[0].shape[1])
        route_tokens = token_sorted[route_begin:route_end]
        route_weights = weight_sorted[route_begin:route_end]
        if exl3_prefill_fixed_tiles_enabled() and interval_max_rows > cap:
            _exl3_moe_fixed_tiles(
                fn, xh, output, counts[:group_size], route_tokens, route_weights,
                stage.layer._exl3_fused_temps, stage.layer._exl3_ptrs,
                int(source_layer._exl3_k), limit,
                num_active if accepts_num_active else None,
                (interval_max_rows + cap - 1) // cap,
            )
        elif interval_max_rows <= cap:
            _exl3_moe_launch(
                fn,
                xh,
                output,
                counts,
                route_tokens,
                route_weights,
                stage.layer._exl3_fused_temps,
                stage.layer._exl3_ptrs,
                int(source_layer._exl3_k),
                limit,
                num_active if accepts_num_active else None,
            )
        elif fat_mode == "row_tile":
            _exl3_moe_row_tiles(
                fn,
                xh,
                output,
                counts[:width],
                route_tokens,
                route_weights,
                stage.layer._exl3_fused_temps,
                stage.layer._exl3_ptrs,
                int(source_layer._exl3_k),
                limit,
                num_active if accepts_num_active else None,
                interval_max_rows,
            )
        else:
            # Launch every thin expert with the fused kernel; it deliberately
            # skips counts above ``cap``.  The ordinary sorted fallback then
            # consumes those fat slices from the same staged HBM weights.
            _exl3_moe_launch(
                fn,
                xh,
                output,
                counts,
                route_tokens,
                route_weights,
                stage.layer._exl3_fused_temps,
                stage.layer._exl3_ptrs,
                int(source_layer._exl3_k),
                limit,
                num_active if accepts_num_active else None,
            )
            apply_exl3_sorted_fat(
                xh,
                route_tokens,
                route_weights,
                interval_counts,
                stage.inners[:width],
                limit,
                cap,
                output,
            )
        reusable[slot].record(main_stream)
        next_index = index + 1
        if next_index < len(intervals):
            schedule_copy(next_index)
    return output


def _gather_exl3_spec_stage_rows(
    source: torch.Tensor,
    output: torch.Tensor,
    physical_ids: torch.Tensor,
    global_ids: torch.Tensor,
) -> None:
    num_ids = int(physical_ids.numel())
    row_size = int(source[0].numel())
    _gather_unique_exl3_spec_stage_rows_kernel[
        (num_ids, triton.cdiv(row_size, 1024))
    ](
        source,
        output,
        physical_ids,
        global_ids,
        row_size=row_size,
        num_ids=num_ids,
        scan_size=triton.next_power_of_2(num_ids),
        block_size=1024,
        num_warps=8,
    )


def _apply_exl3_spec_stage(
    x2d: torch.Tensor,
    ids: torch.Tensor,
    weights: torch.Tensor,
    layer: torch.nn.Module,
    expert_map: torch.Tensor | None,
    limit: float,
) -> torch.Tensor | None:
    """Stage each distinct routed row for a small verify pass into VRAM."""
    max_tokens = int(os.environ.get("EXL3_SPEC_STAGE_MAX_TOKENS", "0"))
    tokens = int(x2d.shape[0])
    if (
        max_tokens <= 0
        or tokens < 2
        or tokens > max_tokens
        or not getattr(layer, "_exl3_mixed_vmm_active", False)
    ):
        return None
    topk = int(ids.shape[-1])
    num_ids = tokens * topk
    capacity = max_tokens * topk
    global_num_experts = (
        int(expert_map.numel())
        if expert_map is not None
        else int(getattr(layer, "global_num_experts", layer.w13_trellis.shape[0]))
    )
    stage = _get_exl3_spec_stage(layer, capacity, global_num_experts)
    flat_global_ids = ids.reshape(-1)
    physical_ids = map_topk_to_local(
        ids, int(layer.w13_trellis.shape[0]), expert_map
    ).reshape(-1)
    for source, output in (
        (layer.w13_trellis, stage.w13_trellis),
        (layer.w13_suh, stage.w13_suh),
        (layer.w13_svh, stage.w13_svh),
        (layer.w2_trellis, stage.w2_trellis),
        (layer.w2_suh, stage.w2_suh),
        (layer.w2_svh, stage.w2_svh),
    ):
        _gather_exl3_spec_stage_rows(
            source, output[:num_ids], physical_ids, flat_global_ids
        )
    stage.expert_map.fill_(-1)
    _build_exl3_spec_stage_map_kernel[(num_ids,)](
        flat_global_ids,
        stage.expert_map,
        num_ids=num_ids,
        global_num_experts=global_num_experts,
        block_size=triton.next_power_of_2(num_ids),
        num_warps=4,
    )
    return apply_exl3_fused_moe(
        x2d,
        ids,
        weights,
        stage.layer,
        stage.inners,
        stage.expert_map,
        limit,
    )


def exl3_deterministic_moe_enabled() -> bool:
    return os.environ.get("EXL3_DETERMINISTIC_MOE", "0").lower() in (
        "1",
        "true",
        "yes",
    )


@triton.jit
def _exl3_fixed64_accumulate_kernel(out_ptr, acc_ptr, n, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    acc = tl.load(acc_ptr + offsets, mask=mask, other=0)
    value = (acc.to(tl.float64) * 2.3283064365386963e-10).to(tl.float32)
    out = tl.load(out_ptr + offsets, mask=mask, other=0.0)
    tl.store(out_ptr + offsets, out + value, mask=mask)


@triton.jit
def _exl3_fixed64_to_float_kernel(out_ptr, acc_ptr, n, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    acc = tl.load(acc_ptr + offsets, mask=mask, other=0)
    value = (acc.to(tl.float64) * 2.3283064365386963e-10).to(tl.float32)
    tl.store(out_ptr + offsets, value, mask=mask)


def _exl3_fixed64_to_float(acc: torch.Tensor) -> torch.Tensor:
    out = torch.empty(acc.shape, dtype=torch.float32, device=acc.device)
    n = out.numel()
    if n:
        block = 1024
        _exl3_fixed64_to_float_kernel[(triton.cdiv(n, block),)](
            out, acc, n, BLOCK=block, num_warps=4
        )
    return out


def _exl3_moe_out_zeros(tokens: int, hidden: int, device) -> torch.Tensor:
    """Routed-MoE accumulator: int64 fixed point in deterministic mode."""
    dtype = torch.int64 if exl3_deterministic_moe_enabled() else torch.float32
    return torch.zeros(tokens, hidden, dtype=dtype, device=device)


def _exl3_fx_index_add_(
    out: torch.Tensor, index: torch.Tensor, value: torch.Tensor
) -> None:
    """out.index_add_ that also accepts the int64 fixed-point accumulator."""
    if out.dtype == torch.int64:
        fixed = torch.round(value.to(torch.float32) * 4294967296.0).to(torch.int64)
        out.index_add_(0, index, fixed)
    else:
        out.index_add_(0, index, value)


def _exl3_fixed64_accumulate(out: torch.Tensor, acc: torch.Tensor) -> None:
    """out += acc * 2^-32 (exllamav3 MOE_OUT_FIXED64, MOE_FIXED64_SHIFT=32)."""
    n = out.numel()
    block = 1024
    _exl3_fixed64_accumulate_kernel[(triton.cdiv(n, block),)](
        out, acc, n, BLOCK=block, num_warps=4
    )


def _exl3_moe_launch(
    fn: Any,
    xh: torch.Tensor,
    out: torch.Tensor,
    expert_count: torch.Tensor,
    token_sorted: torch.Tensor,
    weight_sorted: torch.Tensor,
    temps: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ptrs: dict[str, Any],
    k: int,
    limit: float,
    n_active_host: int | None,
) -> None:
    # Deterministic mode: the kernel scatter-adds the top-k expert outputs of
    # each token with integer atomics into a fixed-point scratch, so the sum
    # is independent of expert-group scheduling order; the scratch is then
    # added to ``out`` in stream order.
    deterministic = exl3_deterministic_moe_enabled()
    target = out
    if deterministic:
        assert out.is_contiguous()
        if n_active_host is not None:
            # Stream-K slice boundaries depend on the group width; pin it.
            n_active_host = -1
        if out.dtype != torch.int64:
            assert out.dtype == torch.float32
            target = torch.zeros(out.shape, dtype=torch.int64, device=out.device)
    args = (
        xh,
        target,
        expert_count,
        token_sorted,
        weight_sorted,
        temps[0],
        temps[1],
        temps[2],
        temps[3],
        MOE_ACT_SILU,
        k,
        k,
        k,
        ptrs["gate_trellis"],
        ptrs["gate_suh"],
        ptrs["gate_svh"],
        ptrs["up_trellis"],
        ptrs["up_suh"],
        ptrs["up_svh"],
        ptrs["down_trellis"],
        ptrs["down_suh"],
        ptrs["down_svh"],
        ptrs["mcg"],
        ptrs["mul1"],
        ptrs["mcg"],
        ptrs["mul1"],
        ptrs["mcg"],
        ptrs["mul1"],
        float(limit),
    )
    if n_active_host is not None:
        fn(*args, n_active_host)
    else:
        fn(*args)
    if deterministic and target is not out:
        _exl3_fixed64_accumulate(out, target)


def _exl3_moe_row_tiles(
    fn: Any,
    xh: torch.Tensor,
    out: torch.Tensor,
    counts: torch.Tensor,
    token_sorted: torch.Tensor,
    weight_sorted: torch.Tensor,
    temps: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ptrs: dict[str, Any],
    k: int,
    limit: float,
    n_active_host: int | None,
    max_rows: int,
) -> None:
    """Launch exl3_moe once per 128-row slice of the sorted expert-token buffer.

    Prefill-only (host syncs). Decode never reaches here: tokens <= temp rows.
    Kernel skips experts with count > temp rows; tiles keep every expert inside temps.
    """
    n_exp = int(counts.shape[0])
    device = counts.device
    tile = int(temps[0].shape[1])
    n_tiles = (max_rows + tile - 1) // tile
    prefix = torch.empty(n_exp, dtype=torch.long, device=device)
    prefix[0] = 0
    if n_exp > 1:
        prefix[1:] = counts[:-1].cumsum(0)
    for t in range(n_tiles):
        row0 = t * tile
        tile_counts = (counts - row0).clamp(min=0, max=tile)
        n_tile = int(tile_counts.sum().item())
        if n_tile == 0:
            continue
        cum = tile_counts.cumsum(0)
        idx = torch.arange(n_tile, device=device)
        expert_id = torch.searchsorted(cum, idx, right=True)
        local_row = idx - (cum[expert_id] - tile_counts[expert_id])
        src = prefix[expert_id] + row0 + local_row
        tile_ec = torch.zeros(n_exp + 1, dtype=torch.long, device=device)
        tile_ec[:n_exp] = tile_counts
        _exl3_moe_launch(
            fn,
            xh,
            out,
            tile_ec,
            token_sorted.index_select(0, src),
            weight_sorted.index_select(0, src),
            temps,
            ptrs,
            k,
            limit,
            n_active_host,
        )



_EXL3_MOE_GEMV_WS: dict[str, torch.Tensor] = {}


def exl3_moe_gemv_enabled() -> bool:
    return os.environ.get("EXL3_MOE_GEMV", "0").lower() in ("1", "true", "yes")


def _exl3_moe_gemv_ws(device: torch.device) -> torch.Tensor:
    import exllamav3_ext

    key = str(device)
    ws = _EXL3_MOE_GEMV_WS.get(key)
    if ws is None:
        ws = torch.zeros(
            int(exllamav3_ext.exl3_moe_gemv_ws_ints()), dtype=torch.int32, device=device
        )
        _EXL3_MOE_GEMV_WS[key] = ws
    return ws


def apply_exl3_gemv_moe(
    x2d: torch.Tensor,
    ids: torch.Tensor,
    weights: torch.Tensor,
    layer: torch.nn.Module,
    n_exp: int,
    expert_map: torch.Tensor | None,
    limit: float,
) -> torch.Tensor:
    """Single-token routed MoE via exl3_moe_gemv (see patches/patch_moe_gemv_integration.py)."""
    import exllamav3_ext as ext

    ptrs = layer._exl3_ptrs
    k = int(getattr(layer, "_exl3_k", 4))
    hidden = int(x2d.shape[-1])
    inter = int(layer._exl3_intermediate_local)
    tokens = int(x2d.shape[0])
    per_token = int(ids.shape[-1])
    topk = int(ids.numel())          # (token, expert) pairs
    dev = x2d.device
    tables = getattr(layer, "_exl3_gemv_tables", None)
    if tables is None:
        tables = [
            ptrs[f"{m}_{c}"] for m in ("gate", "up", "down") for c in ("trellis", "suh", "svh")
        ]
        layer._exl3_gemv_tables = tables
    gu_tab = torch.empty(3, 2 * topk, dtype=torch.long, device=dev)
    d_tab = torch.empty(3, topk, dtype=torch.long, device=dev)
    w = torch.empty(topk, dtype=torch.float32, device=dev)
    wts = weights.reshape(-1)
    if wts.dtype not in (torch.float32, torch.bfloat16):
        wts = wts.float()
    ext.exl3_moe_gemv_prep(
        ids.reshape(-1).to(torch.long).contiguous(), wts.contiguous(), expert_map, n_exp,
        tables, gu_tab, d_tab, w,
    )
    residual = os.environ.get("EXL3_MOE_GEMV_RESIDUAL", "1") != "0"
    ws = _exl3_moe_gemv_ws(dev)
    xh = x2d.reshape(tokens, hidden)
    if xh.dtype == torch.bfloat16 and _EXL3_BF16_IO:
        xh = xh.contiguous()   # bf16 converted to half on load inside the kernel (bit-identical)
    else:
        xh = xh.contiguous() if xh.dtype == torch.float16 else xh.to(torch.float16)
    gu = torch.empty(2 * topk, inter, dtype=torch.float32, device=dev)
    # matrix m (gate m < P, up m >= P) reads token row (m % P) // per_token
    ext.exl3_moe_gemv(xh, topk, per_token, gu_tab[0], gu_tab[1], gu_tab[2], gu, 2 * topk,
                      hidden, inter, k, residual, ws)
    act = torch.empty(topk, inter, dtype=torch.float16, device=dev)
    ext.exl3_moe_swiglu(gu, act, topk, float(limit))
    y = torch.empty(topk, hidden, dtype=torch.float32, device=dev)
    ext.exl3_moe_gemv(act, topk, 1, d_tab[0], d_tab[1], d_tab[2], y, topk, inter, hidden, k, residual, ws)
    layer._exl3_last_apply = "gemv"
    if exl3_deterministic_moe_enabled() or x2d.dtype != torch.bfloat16:
        out = _exl3_moe_out_zeros(tokens, hidden, dev)
        ext.exl3_moe_combine(y, w, out, per_token, True)
        return out
    out = torch.empty(tokens, hidden, dtype=torch.bfloat16, device=dev)
    ext.exl3_moe_combine_bf16(y, w, out, per_token)
    return out


def _exl3_gemv_moe_eligible(x2d, ids, layer) -> bool:
    """Same gate as apply_exl3_fused_moe uses to pick apply_exl3_gemv_moe."""
    ptrs = getattr(layer, "_exl3_ptrs", None)
    tokens = int(x2d.shape[0])
    return bool(
        ptrs
        and tokens <= int(os.environ.get("EXL3_MOE_GEMV_MAX_TOKENS", "4"))
        and tokens * int(ids.shape[-1]) <= 32
        and exl3_moe_gemv_enabled()
        and bool(ptrs.get("mul1"))
        and int(getattr(layer, "_exl3_k", 0)) in (2, 3, 4, 5)
        and int(layer._exl3_intermediate_local) % 256 == 0
    )


def _apply_exl3_gemv_moe_overlap(x2d, ids, weights, layer, n_exp, expert_map, limit, miss_slots, gather_event):
    """apply_exl3_gemv_moe split around an in-flight GPU-subset gather (see EXL3_HYBRID_OVERLAP_GATHER)."""
    import exllamav3_ext as ext
    import glm_fast

    ptrs = layer._exl3_ptrs
    k = int(getattr(layer, "_exl3_k", 4))
    hidden = int(x2d.shape[-1])
    inter = int(layer._exl3_intermediate_local)
    tokens = int(x2d.shape[0])
    per_token = int(ids.shape[-1])
    topk = int(ids.numel())
    dev = x2d.device
    tables = getattr(layer, "_exl3_gemv_tables", None)
    if tables is None:
        tables = [
            ptrs[f"{m}_{c}"] for m in ("gate", "up", "down") for c in ("trellis", "suh", "svh")
        ]
        layer._exl3_gemv_tables = tables
    gu_tab = torch.empty(3, 2 * topk, dtype=torch.long, device=dev)
    d_tab = torch.empty(3, topk, dtype=torch.long, device=dev)
    w = torch.empty(topk, dtype=torch.float32, device=dev)
    wts = weights.reshape(-1)
    if wts.dtype not in (torch.float32, torch.bfloat16):
        wts = wts.float()
    ext.exl3_moe_gemv_prep(
        ids.reshape(-1).to(torch.long).contiguous(), wts.contiguous(), expert_map, n_exp,
        tables, gu_tab, d_tab, w,
    )
    gu_a = torch.empty(2 * topk, dtype=torch.long, device=dev)
    gu_b = torch.empty_like(gu_a)
    d_a = torch.empty(topk, dtype=torch.long, device=dev)
    d_b = torch.empty_like(d_a)
    glm_fast.moe_split_tables(gu_tab[0], d_tab[0], miss_slots, gu_a, gu_b, d_a, d_b)
    residual = os.environ.get("EXL3_MOE_GEMV_RESIDUAL", "1") != "0"
    ws = _exl3_moe_gemv_ws(dev)
    xh = x2d.reshape(tokens, hidden)
    if xh.dtype == torch.bfloat16 and _EXL3_BF16_IO:
        xh = xh.contiguous()
    else:
        xh = xh.contiguous() if xh.dtype == torch.float16 else xh.to(torch.float16)
    gu = torch.empty(2 * topk, inter, dtype=torch.float32, device=dev)
    act = torch.empty(topk, inter, dtype=torch.float16, device=dev)
    y = torch.empty(topk, hidden, dtype=torch.float32, device=dev)
    # A: resident experts, overlapping the gather on the side stream
    ext.exl3_moe_gemv(xh, topk, per_token, gu_a, gu_tab[1], gu_tab[2], gu, 2 * topk, hidden, inter, k, residual, ws)
    ext.exl3_moe_swiglu(gu, act, topk, float(limit))
    ext.exl3_moe_gemv(act, topk, 1, d_a, d_tab[1], d_tab[2], y, topk, inter, hidden, k, residual, ws)
    # B: experts admitted this step, after their weights landed in VRAM
    torch.cuda.current_stream(dev).wait_event(gather_event)
    ext.exl3_moe_gemv(xh, topk, per_token, gu_b, gu_tab[1], gu_tab[2], gu, 2 * topk, hidden, inter, k, residual, ws)
    ext.exl3_moe_swiglu(gu, act, topk, float(limit))
    ext.exl3_moe_gemv(act, topk, 1, d_b, d_tab[1], d_tab[2], y, topk, inter, hidden, k, residual, ws)
    layer._exl3_last_apply = "gemv-overlap"
    if exl3_deterministic_moe_enabled() or x2d.dtype != torch.bfloat16:
        out = _exl3_moe_out_zeros(tokens, hidden, dev)
        ext.exl3_moe_combine(y, w, out, per_token, True)
        return out
    out = torch.empty(tokens, hidden, dtype=torch.bfloat16, device=dev)
    ext.exl3_moe_combine_bf16(y, w, out, per_token)
    return out


def apply_exl3_fused_moe(
    x2d: torch.Tensor,
    ids: torch.Tensor,
    weights: torch.Tensor,
    layer: torch.nn.Module,
    inners: list[dict[str, Any]],
    expert_map: torch.Tensor | None,
    limit: float,
    decode_group_hint_override: int | None = None,
) -> torch.Tensor:
    """One exl3_moe launch per layer when tokens or hottest expert fit temp rows.

    Cap is temps dim1 (`EXL3_TEMP_ROWS_FUSED`, default 128). Overflow uses GPU
    row tiles if `EXL3_MOE_ROW_TILE=1`; otherwise fat experts use the highest
    enabled tier: kernel implies batched, batched implies sorted, then legacy.
    Decode (tokens ≤ cap) stays a single graph-safe launch.
    """
    import exllamav3_ext

    tokens, hidden = x2d.shape
    n_exp = len(inners)
    ptrs = getattr(layer, "_exl3_ptrs", None)
    temps = getattr(layer, "_exl3_fused_temps", None)
    if not ptrs or temps is None:
        raise RuntimeError("EXL3 fused pointer tables were not built after weight load")

    topk = int(ids.shape[-1])
    if (
        tokens <= int(os.environ.get("EXL3_MOE_GEMV_MAX_TOKENS", "4"))
        and tokens * topk <= 32
        and exl3_moe_gemv_enabled()
        and bool(ptrs.get("mul1"))
        and int(getattr(layer, "_exl3_k", 0)) in (2, 3, 4, 5)
        and int(layer._exl3_intermediate_local) % 256 == 0
    ):
        return apply_exl3_gemv_moe(x2d, ids, weights, layer, n_exp, expert_map, limit)
    fused_route_prep = os.environ.get(
        "EXL3_FUSED_DECODE_ROUTE_PREP", "0"
    ).lower() in ("1", "true", "yes")
    if fused_route_prep and tokens == 1:
        expert_count, token_sorted, weight_sorted = (
            _prepare_exl3_decode_routes(ids, weights, expert_map, n_exp)
        )
    else:
        local = map_topk_to_local(ids, n_exp, expert_map)
        flat_token = torch.arange(
            tokens, device=x2d.device, dtype=torch.long
        ).repeat_interleave(topk)
        flat_weight = weights.reshape(-1).to(dtype=torch.float16)
        order = local.argsort()
        token_sorted = flat_token[order]
        weight_sorted = flat_weight[order]
        # scatter_add stays on GPU. torch.bincount can host-stage and break
        # CUDA graphs.
        expert_count = torch.zeros(
            n_exp + 1, dtype=torch.long, device=local.device
        )
        expert_count.scatter_add_(
            0,
            local.long(),
            torch.ones(local.shape, dtype=torch.long, device=local.device),
        )
    out = _exl3_moe_out_zeros(tokens, hidden, x2d.device)
    xh = x2d.contiguous().half()

    counts = expert_count[:n_exp]
    fn = exllamav3_ext.exl3_moe
    # -1 = unknown active count: max-concurrency grid, no .item() host sync.
    # EP decode commonly has only ~top_k / ep_size local experts.  The fused
    # kernel's dynamic ticket scheduler remains correct when ``num_active`` is
    # a fixed group-count hint rather than the exact count: fewer, wider groups
    # simply process additional active experts in later tickets.  This recovers
    # SM occupancy without a graph-breaking device→host count read.  Keep it
    # opt-in until the per-layer/full-model sweep establishes the best hint.
    n_active_host = -1 if _exl3_moe_accepts_num_active(fn) else None
    if n_active_host is not None and decode_group_hint_override is not None:
        explicit_hint = int(decode_group_hint_override)
        if explicit_hint > 0:
            n_active_host = min(
                explicit_hint, int(layer._exl3_fused_concurrency)
            )
    elif n_active_host is not None:
        decode_group_hint = int(os.environ.get("EXL3_MOE_DECODE_GROUPS", "0"))
        decode_group_tokens = int(
            os.environ.get("EXL3_MOE_DECODE_GROUPS_MAX_TOKENS", "1")
        )
        if decode_group_hint > 0 and tokens <= decode_group_tokens:
            n_active_host = min(
                decode_group_hint, int(layer._exl3_fused_concurrency)
            )
    k = int(getattr(layer, "_exl3_k", 4))
    # Actual kernel cap is the allocated temp dim1 (env-selected at load).
    cap = int(temps[0].shape[1])

    # Reset per call so a "kernel" label from an earlier prefill cannot
    # masquerade through later decode/thin/row-tile calls.
    layer._exl3_last_fat_fallback = "none"
    layer._exl3_last_fat_reason = "no_fat_experts"

    if tokens <= cap:
        _exl3_moe_launch(
            fn, xh, out, expert_count, token_sorted, weight_sorted,
            temps, ptrs, k, limit, n_active_host,
        )
        return out

    if exl3_prefill_fixed_tiles_enabled() and cap == triton.next_power_of_2(cap):
        _exl3_moe_fixed_tiles(
            fn, xh, out, counts, token_sorted, weight_sorted,
            temps, ptrs, k, limit, n_active_host, (tokens + cap - 1) // cap,
        )
        layer._exl3_last_fat_fallback = "fixed_tiles"
        return out

    # Prefill larger than temps. E1 copies routing counts on a side stream and
    # launches thin experts immediately, overlapping the D2H synchronization.
    # Decode never reaches here (capture sizes << cap).
    _EXL3_FAT_DIAG["prefill_layer_calls"] += 1
    want_fat_kernel = fat_kernel_enabled()
    want_batched_fat = batched_fat_fallback_enabled() or want_fat_kernel
    use_sorted_fat = sorted_fat_fallback_enabled() or want_batched_fat
    use_batched_fat = (
        want_batched_fat
        and bool(getattr(layer, "_exl3_shared_w13_suh", False))
    )
    use_fat_kernel = use_batched_fat and want_fat_kernel
    use_row_tiles = fused_moe_row_tile_enabled()
    launched = False
    counts_host = None
    if use_batched_fat and not use_row_tiles:
        counts_cpu, count_stream = _stage_counts_to_host(counts)
        _exl3_moe_launch(
            fn, xh, out, expert_count, token_sorted, weight_sorted,
            temps, ptrs, k, limit, n_active_host,
        )
        launched = True
        count_stream.synchronize()
        counts_host = counts_cpu.tolist()
    elif use_sorted_fat:
        counts_host = counts.tolist()

    max_rows = (
        max(counts_host, default=0)
        if counts_host is not None
        else int(counts.max().item())
    )
    if fat_expert_log_enabled():
        record_exl3_fat_expert_stats(
            counts, max_rows=max_rows, counts_host=counts_host
        )
    if max_rows <= cap:
        _EXL3_FAT_DIAG["thin_calls"] += 1
        _record_exl3_fat_reason("thin_only")
        if not launched:
            _exl3_moe_launch(
                fn, xh, out, expert_count, token_sorted, weight_sorted,
                temps, ptrs, k, limit, n_active_host,
            )
        return out

    if use_row_tiles:
        _EXL3_FAT_DIAG["row_tile_calls"] += 1
        layer._exl3_last_fat_fallback = "row_tile"
        layer._exl3_last_fat_reason = "row_tile_preempts_fat"
        _record_exl3_fat_reason("row_tile_preempts_fat")
        _exl3_moe_row_tiles(
            fn, xh, out, counts, token_sorted, weight_sorted,
            temps, ptrs, k, limit, n_active_host, max_rows,
        )
        return out

    if not launched:
        _exl3_moe_launch(
            fn, xh, out, expert_count, token_sorted, weight_sorted,
            temps, ptrs, k, limit, n_active_host,
        )
    if use_batched_fat:
        if use_fat_kernel:
            _record_exl3_fat_tier(layer, "kernel", "kernel_ok")
        else:
            _record_exl3_fat_tier(layer, "batched", "batched_ok")
        assert counts_host is not None
        apply_exl3_batched_fat(
            xh,
            token_sorted,
            weight_sorted,
            counts_host,
            inners,
            limit,
            cap,
            out,
            use_kernel=use_fat_kernel,
        )
    elif use_sorted_fat:
        _record_exl3_fat_tier(
            layer,
            "sorted",
            "degraded_shared_suh" if want_batched_fat else "sorted_ok",
        )
        assert counts_host is not None
        apply_exl3_sorted_fat(
            xh,
            token_sorted,
            weight_sorted,
            counts_host,
            inners,
            limit,
            cap,
            out,
        )
    else:
        _record_exl3_fat_tier(layer, "legacy", "legacy_default")
        fat = (counts > cap).nonzero(as_tuple=False).view(-1)
        if fat.numel():
            apply_exl3_python_loop(
                x2d,
                ids,
                weights,
                inners,
                expert_map,
                limit,
                only_experts=set(int(i) for i in fat.tolist()),
                out=out,
            )
            _EXL3_FAT_DIAG["fat_expert_runs"] += int(fat.numel())
    return out


def apply_exl3_experts(
    x: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    layer: torch.nn.Module,
    *,
    limit: float = SWIGLU_LIMIT_DEFAULT,
    fused: bool | None = None,
) -> torch.Tensor:
    """Shipped routed-expert apply. `fused=None` honors EXL3_FUSED_MOE."""
    inners = getattr(layer, "_exl3_inners", None)
    if not inners:
        raise RuntimeError("EXL3 experts were not built after weight load")
    tokens, hidden = x.shape[-2], x.shape[-1]
    x2d = x.reshape(tokens, hidden)
    ids = topk_ids.reshape(tokens, -1).to(torch.long)
    weights = topk_weights.reshape(tokens, -1)
    lru_cache = getattr(layer, "_exl3_dynamic_lru_cache", None)
    if lru_cache is not None:
        out = _apply_exl3_dynamic_lru(
            x2d,
            ids,
            weights,
            layer,
            inners,
            lru_cache,
            limit,
        )
        layer._exl3_last_apply = (
            "dynamic_lru"
            if tokens <= lru_cache.max_update_tokens
            else "lru_hybrid_prefill"
        )
        if out.dtype == torch.int64:
            out = _exl3_fixed64_to_float(out)
        return out.to(dtype=x.dtype)
    expert_map = pin_exl3_expert_map(layer, x2d.device)
    have_ptrs = bool(getattr(layer, "_exl3_ptrs", None))
    if fused is True and not have_ptrs:
        raise RuntimeError("EXL3 fused apply requested but pointer tables are missing")
    use_fused = (fused_moe_enabled() if fused is None else bool(fused)) and have_ptrs
    if use_fused:
        try:
            import exllamav3_ext

            use_fused = hasattr(exllamav3_ext, "exl3_moe")
        except Exception:
            use_fused = False
    if use_fused:
        out = _apply_exl3_spec_stage(
            x2d, ids, weights, layer, expert_map, limit
        )
        if out is None:
            out = apply_exl3_fused_moe(
                x2d, ids, weights, layer, inners, expert_map, limit
            )
            layer._exl3_last_apply = "fused"
        else:
            layer._exl3_last_apply = "spec_stage"
    else:
        out = apply_exl3_python_loop(x2d, ids, weights, inners, expert_map, limit)
        layer._exl3_last_apply = "loop"
    if out.dtype == torch.int64:
        out = _exl3_fixed64_to_float(out)
    return out.to(dtype=x.dtype)


def _suffix_from_mapped_name(weight_name: str) -> str:
    tail = weight_name.rsplit(".", 1)[-1]
    for suffix in EXL3_SUFFIXES:
        if tail == suffix or tail.endswith("_" + suffix):
            return suffix
    raise ValueError(f"not an EXL3 packed name: {weight_name}")


def _cuda_driver_check(result: tuple[Any, ...], operation: str) -> Any:
    from cuda.bindings import driver

    error, *values = result
    if error != driver.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"{operation} failed: {error}")
    if not values:
        return None
    return values[0] if len(values) == 1 else tuple(values)


def _allocate_exl3_empty_mixed_vmm_tensor(
    shape: tuple[int, ...],
    dtype: torch.dtype,
    hot_experts: int,
    device: torch.device,
) -> tuple[torch.Tensor, _Exl3MixedVMMAllocation]:
    """Allocate an empty expert-major tensor directly across VRAM and RAM.

    Loading directly into this tensor avoids the otherwise fatal peak of first
    materializing every TP rank's full expert tensor in pinned host memory and
    then allocating the cold VMM copy. Checkpoint expert IDs are remapped by
    the layer's expert map, so the loader writes hot experts into the VRAM
    prefix and cold experts into the host-backed suffix in one pass.
    """
    from cuda.bindings import driver

    if len(shape) < 1 or not 0 < hot_experts <= int(shape[0]):
        raise ValueError(f"invalid EXL3 VMM shape/hot capacity: {shape}/{hot_experts}")
    device = torch.device(device)
    if device.type != "cuda":
        raise ValueError(f"EXL3 direct VMM requires a CUDA device, got {device}")
    device_index = device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
        device = torch.device("cuda", device_index)

    _cuda_driver_check(driver.cuInit(0), "cuInit")
    cuda_device = _cuda_driver_check(
        driver.cuDeviceGet(device_index), "cuDeviceGet"
    )
    numa_id = _cuda_driver_check(
        driver.cuDeviceGetAttribute(
            driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_HOST_NUMA_ID,
            cuda_device,
        ),
        "cuDeviceGetAttribute(HOST_NUMA_ID)",
    )

    def allocation_prop(location_type: Any, location_id: int) -> Any:
        prop = driver.CUmemAllocationProp()
        prop.type = driver.CUmemAllocationType.CU_MEM_ALLOCATION_TYPE_PINNED
        prop.location.type = location_type
        prop.location.id = location_id
        prop.requestedHandleTypes = (
            driver.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_NONE
        )
        return prop

    device_prop = allocation_prop(
        driver.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE, device_index
    )
    host_prop = allocation_prop(
        driver.CUmemLocationType.CU_MEM_LOCATION_TYPE_HOST_NUMA, numa_id
    )
    minimum = driver.CUmemAllocationGranularity_flags.CU_MEM_ALLOC_GRANULARITY_MINIMUM
    alignment = max(
        _cuda_driver_check(
            driver.cuMemGetAllocationGranularity(device_prop, minimum),
            "cuMemGetAllocationGranularity(device)",
        ),
        _cuda_driver_check(
            driver.cuMemGetAllocationGranularity(host_prop, minimum),
            "cuMemGetAllocationGranularity(host)",
        ),
    )
    element_size = torch.empty((), dtype=dtype).element_size()
    row_elements = 1
    for dimension in shape[1:]:
        row_elements *= int(dimension)
    row_bytes = row_elements * element_size
    original_bytes = int(shape[0]) * row_bytes
    mapped_bytes = ((original_bytes + alignment - 1) // alignment) * alignment
    gpu_data_bytes = hot_experts * row_bytes
    gpu_bytes = min(
        ((gpu_data_bytes + alignment - 1) // alignment) * alignment,
        mapped_bytes,
    )
    host_bytes = mapped_bytes - gpu_bytes

    address = _cuda_driver_check(
        driver.cuMemAddressReserve(mapped_bytes, alignment, 0, 0),
        "cuMemAddressReserve",
    )
    handles: list[Any] = []
    if gpu_bytes:
        gpu_handle = _cuda_driver_check(
            driver.cuMemCreate(gpu_bytes, device_prop, 0),
            "cuMemCreate(device)",
        )
        handles.append(gpu_handle)
        _cuda_driver_check(
            driver.cuMemMap(address, gpu_bytes, 0, gpu_handle, 0),
            "cuMemMap(device)",
        )
    if host_bytes:
        host_handle = _cuda_driver_check(
            driver.cuMemCreate(host_bytes, host_prop, 0),
            "cuMemCreate(host)",
        )
        handles.append(host_handle)
        _cuda_driver_check(
            driver.cuMemMap(int(address) + gpu_bytes, host_bytes, 0, host_handle, 0),
            "cuMemMap(host)",
        )

    access = driver.CUmemAccessDesc()
    access.location.type = driver.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
    access.location.id = device_index
    access.flags = driver.CUmemAccess_flags.CU_MEM_ACCESS_FLAGS_PROT_READWRITE
    _cuda_driver_check(
        driver.cuMemSetAccess(address, mapped_bytes, [access], 1),
        "cuMemSetAccess",
    )

    stride = [1] * len(shape)
    for index in range(len(shape) - 2, -1, -1):
        stride[index] = stride[index + 1] * int(shape[index + 1])
    storage = torch._C._construct_storage_from_data_pointer(
        int(address), device, mapped_bytes
    )
    metadata = {
        "nbytes": mapped_bytes,
        "data_ptr": int(address),
        "size": shape,
        "stride": tuple(stride),
        "dtype": dtype,
        "device": device,
        "storage_offset": 0,
    }
    output = torch._C._construct_CUDA_Tensor_From_Storage_And_Metadata(
        metadata, storage
    )
    allocation = _Exl3MixedVMMAllocation(
        storage=storage,
        address=address,
        handles=tuple(handles),
        mapped_bytes=mapped_bytes,
        gpu_bytes=gpu_bytes,
        host_bytes=host_bytes,
    )
    _EXL3_MIXED_VMM_ALLOCATIONS.append(allocation)
    return output, allocation


def _allocate_exl3_mixed_vmm_tensor(
    source: torch.Tensor,
    new_to_old: torch.Tensor,
    hot_experts: int,
) -> tuple[torch.Tensor, _Exl3MixedVMMAllocation]:
    """Build a hot-first tensor with a VRAM prefix and host-RAM suffix.

    ``source`` is expected to be a CUDA view of pinned host memory created by
    vLLM's UVA offloader.  The resulting tensor keeps the normal stacked EXL3
    expert geometry, so LinearEXL3 and the fused ``exl3_moe`` pointer tables do
    not need a separate cold path.  CUDA page tables choose VRAM or host memory
    for each contiguous range.
    """
    from cuda.bindings import driver

    if not source.is_cuda or not source.is_contiguous() or source.ndim == 0:
        raise ValueError(
            "Mixed EXL3 VMM requires a contiguous CUDA/UVA tensor; "
            f"got device={source.device}, shape={tuple(source.shape)}"
        )
    if not 0 < hot_experts <= int(source.shape[0]):
        raise ValueError(
            f"invalid EXL3 VMM hot capacity {hot_experts} for {source.shape[0]} experts"
        )

    device_index = source.device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    _cuda_driver_check(driver.cuInit(0), "cuInit")
    device = _cuda_driver_check(driver.cuDeviceGet(device_index), "cuDeviceGet")
    numa_id = _cuda_driver_check(
        driver.cuDeviceGetAttribute(
            driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_HOST_NUMA_ID,
            device,
        ),
        "cuDeviceGetAttribute(HOST_NUMA_ID)",
    )

    def allocation_prop(location_type: Any, location_id: int) -> Any:
        prop = driver.CUmemAllocationProp()
        prop.type = driver.CUmemAllocationType.CU_MEM_ALLOCATION_TYPE_PINNED
        prop.location.type = location_type
        prop.location.id = location_id
        prop.requestedHandleTypes = (
            driver.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_NONE
        )
        return prop

    device_prop = allocation_prop(
        driver.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE,
        device_index,
    )
    host_prop = allocation_prop(
        driver.CUmemLocationType.CU_MEM_LOCATION_TYPE_HOST_NUMA,
        numa_id,
    )
    minimum = driver.CUmemAllocationGranularity_flags.CU_MEM_ALLOC_GRANULARITY_MINIMUM
    alignment = max(
        _cuda_driver_check(
            driver.cuMemGetAllocationGranularity(device_prop, minimum),
            "cuMemGetAllocationGranularity(device)",
        ),
        _cuda_driver_check(
            driver.cuMemGetAllocationGranularity(host_prop, minimum),
            "cuMemGetAllocationGranularity(host)",
        ),
    )
    original_bytes = source.numel() * source.element_size()
    mapped_bytes = ((original_bytes + alignment - 1) // alignment) * alignment
    row_bytes = source[0].numel() * source.element_size()
    gpu_data_bytes = hot_experts * row_bytes
    gpu_bytes = min(
        ((gpu_data_bytes + alignment - 1) // alignment) * alignment,
        mapped_bytes,
    )
    host_bytes = mapped_bytes - gpu_bytes

    address = _cuda_driver_check(
        driver.cuMemAddressReserve(mapped_bytes, alignment, 0, 0),
        "cuMemAddressReserve",
    )
    handles: list[Any] = []
    if gpu_bytes:
        gpu_handle = _cuda_driver_check(
            driver.cuMemCreate(gpu_bytes, device_prop, 0),
            "cuMemCreate(device)",
        )
        handles.append(gpu_handle)
        _cuda_driver_check(
            driver.cuMemMap(address, gpu_bytes, 0, gpu_handle, 0),
            "cuMemMap(device)",
        )
    if host_bytes:
        host_handle = _cuda_driver_check(
            driver.cuMemCreate(host_bytes, host_prop, 0),
            "cuMemCreate(host)",
        )
        handles.append(host_handle)
        _cuda_driver_check(
            driver.cuMemMap(int(address) + gpu_bytes, host_bytes, 0, host_handle, 0),
            "cuMemMap(host)",
        )

    access = driver.CUmemAccessDesc()
    access.location.type = driver.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
    access.location.id = device_index
    access.flags = driver.CUmemAccess_flags.CU_MEM_ACCESS_FLAGS_PROT_READWRITE
    _cuda_driver_check(
        driver.cuMemSetAccess(address, mapped_bytes, [access], 1),
        "cuMemSetAccess",
    )

    storage = torch._C._construct_storage_from_data_pointer(
        int(address), source.device, mapped_bytes
    )
    metadata = {
        "nbytes": mapped_bytes,
        "data_ptr": int(address),
        "size": tuple(source.shape),
        "stride": tuple(source.stride()),
        "dtype": source.dtype,
        "device": source.device,
        "storage_offset": 0,
    }
    output = torch._C._construct_CUDA_Tensor_From_Storage_And_Metadata(
        metadata, storage
    )
    row_size = source[0].numel()
    grid = (source.shape[0], triton.cdiv(row_size, 1024))
    _permute_exl3_expert_rows_kernel[grid](
        source,
        output,
        new_to_old,
        row_size=row_size,
        block_size=1024,
        num_warps=8,
    )
    torch.cuda.synchronize(source.device)
    allocation = _Exl3MixedVMMAllocation(
        storage=storage,
        address=address,
        handles=tuple(handles),
        mapped_bytes=mapped_bytes,
        gpu_bytes=gpu_bytes,
        host_bytes=host_bytes,
    )
    _EXL3_MIXED_VMM_ALLOCATIONS.append(allocation)
    return output, allocation


def _ranked_experts_for_layer(
    rankings: dict[str, Any], layer_name: str, num_experts: int
) -> list[int]:
    """Accept either ExLlama count vectors or an already-ranked ID list."""
    match = re.search(r"(?:^|\.)layers\.(\d+)(?:\.|$)", layer_name)
    if match is None:
        return []
    layer_id = match.group(1)
    values = rankings.get(layer_name)
    if values is None:
        values = rankings.get(f"model.language_model.layers.{layer_id}.mlp")
    if values is None:
        values = rankings.get(layer_id)
    if not isinstance(values, list) or len(values) != num_experts:
        return []
    ints = [int(value) for value in values]
    if sorted(ints) == list(range(num_experts)):
        return ints
    return sorted(range(num_experts), key=lambda expert: (-ints[expert], expert))


def _install_exl3_expert_map(layer: torch.nn.Module, expert_map: torch.Tensor) -> None:
    """Update the RoutedExperts map and its manager across vLLM revisions."""
    installed = False
    if hasattr(layer, "_expert_map"):
        layer._expert_map = expert_map
        installed = True
    manager = getattr(layer, "expert_map_manager", None)
    if manager is not None and hasattr(manager, "_expert_map"):
        manager._expert_map = expert_map
        installed = True
    if not installed:
        try:
            layer.expert_map = expert_map
            installed = True
        except (AttributeError, RuntimeError):
            pass
    if not installed:
        raise RuntimeError("unable to install hot-first EXL3 expert map")


def _exl3_vmm_hot_plan(
    layer: torch.nn.Module,
    layer_name: str,
    num_experts: int,
    device: torch.device,
) -> tuple[list[int], torch.Tensor]:
    capacity = int(os.environ.get("EXL3_VMM_HOT_EXPERTS", "0"))
    rankings_path = os.environ.get("EXL3_VMM_HOT_RANKINGS")
    if capacity <= 0:
        raise ValueError("EXL3_VMM_HOT_EXPERTS must be positive when VMM is enabled")
    if not rankings_path or not os.path.isfile(rankings_path):
        raise FileNotFoundError(f"EXL3 VMM rankings not found: {rankings_path}")
    rankings = _EXL3_VMM_RANKINGS_CACHE.get(rankings_path)
    if rankings is None:
        with open(rankings_path) as source_file:
            rankings = json.load(source_file)
        _EXL3_VMM_RANKINGS_CACHE[rankings_path] = rankings
    original_map = getattr(layer, "expert_map", None)
    global_experts = int(getattr(layer, "global_num_experts", num_experts))
    ranked_global = _ranked_experts_for_layer(
        rankings, layer_name, global_experts
    )
    if not ranked_global:
        raise KeyError(f"no {global_experts}-expert ranking for {layer_name}")
    if original_map is None:
        ranked_ids = ranked_global
    else:
        original_map_cpu = [
            int(value) for value in original_map.detach().cpu().tolist()
        ]
        ranked_ids = [
            original_map_cpu[global_id]
            for global_id in ranked_global
            if original_map_cpu[global_id] >= 0
        ]
    if sorted(ranked_ids) != list(range(num_experts)):
        raise RuntimeError(
            f"EXL3 local ranking is not a permutation for {layer_name}: "
            f"local={num_experts} ranked={len(ranked_ids)}"
        )
    hot_ids = ranked_ids[: min(capacity, num_experts)]
    hot_set = set(hot_ids)
    new_to_old = hot_ids + [
        expert for expert in range(num_experts) if expert not in hot_set
    ]
    old_to_new = [0] * num_experts
    for new_id, old_id in enumerate(new_to_old):
        old_to_new[old_id] = new_id

    if original_map is None:
        original_map_cpu = list(range(global_experts))
        map_dtype = torch.int32
    else:
        original_map_cpu = [
            int(value) for value in original_map.detach().cpu().tolist()
        ]
        map_dtype = original_map.dtype
    new_map_cpu = [
        -1 if old_id < 0 else old_to_new[old_id] for old_id in original_map_cpu
    ]
    new_map = torch.tensor(new_map_cpu, dtype=map_dtype, device=device)
    return hot_ids, new_map


def _allocate_exl3_direct_vmm_trellises(
    layer: torch.nn.Module,
    layer_name: str,
    num_experts: int,
    w13_shape: tuple[int, ...],
    w2_shape: tuple[int, ...],
) -> tuple[torch.Tensor, torch.Tensor] | None:
    enabled = os.environ.get("EXL3_MIXED_VMM_HOT_CACHE", "0").lower()
    direct = os.environ.get("EXL3_VMM_DIRECT_ALLOC", "1").lower()
    dynamic = os.environ.get("EXL3_DYNAMIC_LRU", "0").lower()
    if dynamic in ("1", "true", "yes"):
        # Dynamic replacement needs a complete host master for every row.
        # The direct VMM layout has only a host suffix, so let vLLM's UVA
        # offloader retain all source rows and allocate a separate VRAM cache.
        return None
    if enabled not in ("1", "true", "yes") or direct in ("0", "false", "no"):
        return None
    device = torch.device("cuda", torch.cuda.current_device())
    hot_ids, new_map = _exl3_vmm_hot_plan(
        layer, layer_name, num_experts, device
    )
    w13, allocation13 = _allocate_exl3_empty_mixed_vmm_tensor(
        w13_shape, torch.int16, len(hot_ids), device
    )
    w2, allocation2 = _allocate_exl3_empty_mixed_vmm_tensor(
        w2_shape, torch.int16, len(hot_ids), device
    )
    _install_exl3_expert_map(layer, new_map)
    layer._exl3_mixed_vmm_active = True
    layer._exl3_vmm_direct_alloc = True
    layer._exl3_vmm_hot_experts = len(hot_ids)
    layer._exl3_vmm_gpu_bytes = allocation13.gpu_bytes + allocation2.gpu_bytes
    layer._exl3_vmm_host_bytes = allocation13.host_bytes + allocation2.host_bytes
    logger.info(
        "Direct mixed EXL3 VMM: layer=%s hot=%d gpu=%.2f MiB host=%.2f MiB",
        layer_name,
        len(hot_ids),
        layer._exl3_vmm_gpu_bytes / (1024 * 1024),
        layer._exl3_vmm_host_bytes / (1024 * 1024),
    )
    return w13, w2


def maybe_init_exl3_mixed_vmm_hot_cache(
    layer: torch.nn.Module, layer_name: str
) -> None:
    """Move hot EXL3 expert rows to VRAM and keep cold rows in pinned RAM."""
    enabled = os.environ.get("EXL3_MIXED_VMM_HOT_CACHE", "0").lower()
    if enabled not in ("1", "true", "yes"):
        return
    if getattr(layer, "_exl3_vmm_direct_alloc", False):
        return
    if getattr(layer, "_exl3_mixed_vmm_active", False):
        raise RuntimeError(f"mixed EXL3 VMM cache already initialized: {layer_name}")
    capacity = int(os.environ.get("EXL3_VMM_HOT_EXPERTS", "0"))
    rankings_path = os.environ.get("EXL3_VMM_HOT_RANKINGS")
    if capacity <= 0:
        raise ValueError("EXL3_VMM_HOT_EXPERTS must be positive when VMM cache is enabled")
    if not rankings_path or not os.path.isfile(rankings_path):
        raise FileNotFoundError(f"EXL3 VMM rankings not found: {rankings_path}")

    w13 = layer.w13_trellis
    w2 = layer.w2_trellis
    if not (
        getattr(w13, "_vllm_is_uva_offloaded", False)
        and getattr(w2, "_vllm_is_uva_offloaded", False)
    ):
        raise RuntimeError(
            "EXL3 mixed VMM requires UVA-offloaded trellises; launch with "
            "--offload-backend uva --cpu-offload-params experts"
        )
    num_experts = int(w13.shape[0])
    rankings = _EXL3_VMM_RANKINGS_CACHE.get(rankings_path)
    if rankings is None:
        with open(rankings_path) as source_file:
            rankings = json.load(source_file)
        _EXL3_VMM_RANKINGS_CACHE[rankings_path] = rankings
    original_map = getattr(layer, "expert_map", None)
    global_experts = int(getattr(layer, "global_num_experts", num_experts))
    ranked_global = _ranked_experts_for_layer(
        rankings, layer_name, global_experts
    )
    if not ranked_global:
        raise KeyError(f"no {global_experts}-expert ranking for {layer_name}")
    if original_map is None:
        ranked_ids = ranked_global
    else:
        original_map_cpu = [
            int(value) for value in original_map.detach().cpu().tolist()
        ]
        ranked_ids = [
            original_map_cpu[global_id]
            for global_id in ranked_global
            if original_map_cpu[global_id] >= 0
        ]
    if sorted(ranked_ids) != list(range(num_experts)):
        raise RuntimeError(
            f"EXL3 local ranking is not a permutation for {layer_name}: "
            f"local={num_experts} ranked={len(ranked_ids)}"
        )
    hot_ids = ranked_ids[: min(capacity, num_experts)]
    hot_set = set(hot_ids)
    new_to_old_list = hot_ids + [
        expert for expert in range(num_experts) if expert not in hot_set
    ]
    old_to_new = [0] * num_experts
    for new_id, old_id in enumerate(new_to_old_list):
        old_to_new[old_id] = new_id

    if original_map is None:
        original_map_cpu = list(range(global_experts))
        map_dtype = torch.int32
        map_device = w13.device
    else:
        original_map_cpu = [int(value) for value in original_map.detach().cpu().tolist()]
        map_dtype = original_map.dtype
        map_device = original_map.device
    new_map_cpu = [
        -1 if old_id < 0 else old_to_new[old_id] for old_id in original_map_cpu
    ]
    new_map = torch.tensor(new_map_cpu, dtype=map_dtype, device=map_device)
    new_to_old = torch.tensor(new_to_old_list, dtype=torch.long, device=w13.device)

    marker_suffix = str(getattr(layer, "_exl3_codebook", "mcg"))
    tensor_names = (
        "w13_trellis",
        "w2_trellis",
        "w13_suh",
        "w13_svh",
        f"w13_{marker_suffix}",
        "w2_suh",
        "w2_svh",
        f"w2_{marker_suffix}",
    )
    vmm_gpu_bytes = 0
    vmm_host_bytes = 0
    regular_gpu_bytes = 0
    with torch.no_grad():
        for name in tensor_names:
            tensor = getattr(layer, name)
            if tensor.shape[0] != num_experts:
                raise RuntimeError(
                    f"EXL3 expert-indexed tensor mismatch: {name} {tuple(tensor.shape)}"
                )
            if name in ("w13_trellis", "w2_trellis"):
                reordered, allocation = _allocate_exl3_mixed_vmm_tensor(
                    tensor, new_to_old, len(hot_ids)
                )
                vmm_gpu_bytes += allocation.gpu_bytes
                vmm_host_bytes += allocation.host_bytes
            else:
                reordered = torch.index_select(tensor, 0, new_to_old).contiguous()
                regular_gpu_bytes += reordered.numel() * reordered.element_size()
            replace_parameter(
                layer, name, Parameter(reordered, requires_grad=False)
            )
        _install_exl3_expert_map(layer, new_map)

    del w13, w2
    torch.accelerator.empty_host_cache()
    layer._exl3_mixed_vmm_active = True
    layer._exl3_vmm_hot_experts = len(hot_ids)
    layer._exl3_vmm_gpu_bytes = vmm_gpu_bytes
    layer._exl3_vmm_host_bytes = vmm_host_bytes
    logger.info(
        "Mixed EXL3 VMM cache: layer=%s hot=%d gpu=%.2f MiB host=%.2f MiB "
        "aux_gpu=%.2f MiB",
        layer_name,
        len(hot_ids),
        vmm_gpu_bytes / (1024 * 1024),
        vmm_host_bytes / (1024 * 1024),
        regular_gpu_bytes / (1024 * 1024),
    )


def _gather_exl3_lru_rows(
    source: torch.Tensor,
    output: torch.Tensor,
    miss_local_ids: torch.Tensor,
    miss_slots: torch.Tensor,
    num_ids: int,
) -> None:
    row_size = int(source[0].numel())
    block_size = int(os.environ.get("EXL3_LRU_GATHER_BLOCK", "1024"))
    num_warps = int(os.environ.get("EXL3_LRU_GATHER_WARPS", "8"))
    if block_size < 256 or block_size > 65536 or block_size & (block_size - 1):
        raise ValueError(
            "EXL3_LRU_GATHER_BLOCK must be a power of two in [256, 65536]"
        )
    if num_warps not in (1, 2, 4, 8):
        raise ValueError("EXL3_LRU_GATHER_WARPS must be one of 1, 2, 4, 8")
    grid = (num_ids, triton.cdiv(row_size, block_size))
    _gather_exl3_lru_rows_kernel[grid](
        source,
        output,
        miss_local_ids,
        miss_slots,
        row_size=row_size,
        num_ids=num_ids,
        block_size=block_size,
        num_warps=num_warps,
    )


def _gather_exl3_lru_transfers(
    transfers: tuple[tuple[torch.Tensor, torch.Tensor], ...],
    miss_local_ids: torch.Tensor,
    miss_slots: torch.Tensor,
    num_ids: int,
) -> None:
    if os.environ.get("EXL3_LRU_GATHER_IMPL", "triton") == "cuda":
        import exllamav3_ext

        exllamav3_ext.exl3_lru_gather(
            [source for source, _ in transfers],
            [output for _, output in transfers],
            miss_local_ids,
            miss_slots,
            num_ids,
            int(os.environ.get("EXL3_LRU_GATHER_CUDA_BLOCKS", "4")),
            1024,
        )
        return
    fused = os.environ.get("EXL3_LRU_FUSED_GATHER", "0").lower() in (
        "1",
        "true",
        "yes",
    )
    if not fused:
        for source, output in transfers:
            _gather_exl3_lru_rows(
                source, output, miss_local_ids, miss_slots, num_ids
            )
        return
    if len(transfers) != 6:
        raise ValueError("EXL3 fused gather requires exactly six component rows")
    block_size = int(os.environ.get("EXL3_LRU_GATHER_BLOCK", "1024"))
    num_warps = int(os.environ.get("EXL3_LRU_GATHER_WARPS", "8"))
    if block_size < 256 or block_size > 65536 or block_size & (block_size - 1):
        raise ValueError(
            "EXL3_LRU_GATHER_BLOCK must be a power of two in [256, 65536]"
        )
    if num_warps not in (1, 2, 4, 8):
        raise ValueError("EXL3_LRU_GATHER_WARPS must be one of 1, 2, 4, 8")
    rows = tuple(int(source[0].numel()) for source, _ in transfers)
    chunks = tuple(triton.cdiv(row, block_size) for row in rows)
    flat = tuple(
        value.view(torch.int16)
        for source, output in transfers
        for value in (source, output)
    )
    grid = (num_ids, sum(chunks))
    _gather_exl3_lru_expert_kernel[grid](
        *flat,
        miss_local_ids,
        miss_slots,
        row0=rows[0],
        row1=rows[1],
        row2=rows[2],
        row3=rows[3],
        row4=rows[4],
        row5=rows[5],
        chunks0=chunks[0],
        chunks1=chunks[1],
        chunks2=chunks[2],
        chunks3=chunks[3],
        chunks4=chunks[4],
        chunks5=chunks[5],
        block_size=block_size,
        num_warps=num_warps,
    )


def _refresh_exl3_hybrid_ptrs(
    cache: _Exl3LRUCache, source_layer: torch.nn.Module
) -> None:
    source_ptrs = source_layer._exl3_ptrs
    cache_ptrs = cache.layer._exl3_ptrs
    hybrid_ptrs = cache.hybrid_layer._exl3_ptrs
    pointer_args: list[torch.Tensor] = []
    for name in _EXL3_MOE_POINTER_NAMES:
        pointer_args.extend(
            (source_ptrs[name], cache_ptrs[name], hybrid_ptrs[name])
        )
    num_local = int(cache.local_to_global.numel())
    _refresh_exl3_hybrid_ptrs_kernel[(1,)](
        cache.local_to_global,
        cache.cache_map,
        *pointer_args,
        num_local=num_local,
        block_size=triton.next_power_of_2(num_local),
        num_warps=4,
    )


_EXL3_EXPERT_SOURCE_NAMES = ("w13_trellis", "w13_suh", "w13_svh", "w2_trellis", "w2_suh", "w2_svh")



def _exl3_plan_pinned_experts(layer_name: str, num_experts: int, layer: torch.nn.Module) -> list[int]:
    """Local ids of the experts pinned in VRAM for this layer (hottest first), or [] when pinning is off."""
    frac = float(os.environ.get("EXL3_PINNED_FRAC", "0") or 0)
    if frac <= 0 or os.environ.get("EXL3_DYNAMIC_LRU", "0").lower() not in ("1", "true", "yes"):
        return []
    if bool(getattr(layer, "use_ep", False)):
        return []
    match = re.search(r"(?:^|\.)layers\.(\d+)(?:\.|$)", layer_name)
    if match is None:
        return []
    max_layer_id = int(os.environ.get("EXL3_DYNAMIC_LRU_MAX_LAYER_ID", "-1"))
    if max_layer_id >= 0 and int(match.group(1)) > max_layer_id:
        return []
    base = int(os.environ.get("EXL3_VMM_HOT_EXPERTS", "0"))
    path = os.environ.get("EXL3_VMM_HOT_RANKINGS")
    if base <= 0 or not path or not os.path.isfile(path):
        return []
    device = torch.device("cuda", torch.cuda.current_device())
    capacity = min(_resolve_exl3_lru_capacity(base, layer_name, device), num_experts)
    min_dynamic = max(32, max(1, int(os.environ.get("EXL3_DYNAMIC_LRU_MAX_TOKENS", "1"))) * 8)
    n_pin = max(0, min(int(round(frac * capacity)), capacity - min_dynamic))
    if n_pin <= 0:
        return []
    rankings = _EXL3_VMM_RANKINGS_CACHE.get(path)
    if rankings is None:
        with open(path) as handle:
            rankings = json.load(handle)
        _EXL3_VMM_RANKINGS_CACHE[path] = rankings
    ranked = _ranked_experts_for_layer(rankings, layer_name, num_experts)
    return ranked[:n_pin] if ranked else []

def maybe_init_exl3_dynamic_lru_cache(
    layer: torch.nn.Module, layer_name: str
) -> None:
    """Create a fixed-address VRAM LRU backed by all local experts in UVA RAM."""
    enabled = os.environ.get("EXL3_DYNAMIC_LRU", "0").lower()
    if enabled not in ("1", "true", "yes"):
        return
    max_layer_id = int(os.environ.get("EXL3_DYNAMIC_LRU_MAX_LAYER_ID", "-1"))
    if max_layer_id >= 0:
        match = re.search(r"(?:^|\.)layers\.(\d+)(?:\.|$)", layer_name)
        if match is None:
            raise ValueError(f"cannot resolve EXL3 LRU layer: {layer_name}")
        if int(match.group(1)) > max_layer_id:
            logger.info(
                "Dynamic EXL3 LRU bypass: layer=%s exceeds max layer id %d",
                layer_name,
                max_layer_id,
            )
            return
    if getattr(layer, "_exl3_dynamic_lru_cache", None) is not None:
        raise RuntimeError(f"EXL3 dynamic LRU already initialized: {layer_name}")
    if not fused_moe_enabled():
        raise RuntimeError("EXL3 dynamic LRU requires EXL3_FUSED_MOE=1")

    capacity = int(os.environ.get("EXL3_VMM_HOT_EXPERTS", "0"))
    rankings_path = os.environ.get("EXL3_VMM_HOT_RANKINGS")
    if capacity <= 0:
        raise ValueError("EXL3_VMM_HOT_EXPERTS must be positive for dynamic LRU")
    if not rankings_path or not os.path.isfile(rankings_path):
        raise FileNotFoundError(f"EXL3 LRU rankings not found: {rankings_path}")

    source_tensors = (
        layer.w13_trellis,
        layer.w13_suh,
        layer.w13_svh,
        layer.w2_trellis,
        layer.w2_suh,
        layer.w2_svh,
    )
    if not all(
        getattr(tensor, "_vllm_is_uva_offloaded", False)
        for tensor in source_tensors
    ):
        raise RuntimeError(
            "EXL3 dynamic LRU requires every expert source tensor in UVA RAM; "
            "launch with --offload-backend uva --cpu-offload-params experts"
        )

    num_local = int(layer.w13_trellis.shape[0])
    capacity = _resolve_exl3_lru_capacity(
        capacity, layer_name, layer.w13_trellis.device
    )
    capacity = min(capacity, num_local + len(getattr(layer, "_exl3_pinned_local", None) or []))
    device = layer.w13_trellis.device
    global_experts = int(getattr(layer, "global_num_experts", num_local))
    original_map = getattr(layer, "expert_map", None)
    if original_map is None:
        source_map = torch.arange(global_experts, dtype=torch.long, device=device)
    else:
        source_map = original_map.to(device=device, dtype=torch.long)
    source_map_cpu = [int(value) for value in source_map.detach().cpu().tolist()]
    pinned_global: list[int] = []
    if getattr(layer, "_exl3_pinned_local", None):
        # global -> compact host row; pinned experts have no host row (-1) and live in fixed VRAM slots
        local_to_global_full = {local: g for g, local in enumerate(source_map_cpu) if local >= 0}
        pinned_global = [local_to_global_full[local] for local in layer._exl3_pinned_local]
        host_row = layer._exl3_host_row
        source_map_cpu = [host_row.get(local, -1) if local >= 0 else -1 for local in source_map_cpu]
        source_map = torch.tensor(source_map_cpu, dtype=torch.long, device=device)

    rankings = _EXL3_VMM_RANKINGS_CACHE.get(rankings_path)
    if rankings is None:
        with open(rankings_path) as source_file:
            rankings = json.load(source_file)
        _EXL3_VMM_RANKINGS_CACHE[rankings_path] = rankings
    ranked_global = _ranked_experts_for_layer(
        rankings, layer_name, global_experts
    )
    if not ranked_global:
        raise KeyError(f"no {global_experts}-expert ranking for {layer_name}")
    n_pin = len(pinned_global)
    dyn_capacity = capacity - n_pin
    seed_global = [
        global_id
        for global_id in ranked_global
        if source_map_cpu[global_id] >= 0
    ][:dyn_capacity]
    if len(seed_global) != dyn_capacity:
        raise RuntimeError(
            f"EXL3 LRU could seed only {len(seed_global)}/{dyn_capacity} local "
            f"experts for {layer_name}"
        )
    seed_local = torch.tensor(
        [source_map_cpu[global_id] for global_id in seed_global],
        dtype=torch.long,
        device=device,
    )
    seed_global = seed_global + pinned_global   # slot order: dynamic LRU, then fixed pinned slots

    def empty_rows(source: torch.Tensor) -> torch.Tensor:
        return torch.empty(
            (capacity, *source.shape[1:]), dtype=source.dtype, device=device
        )

    w13_trellis = empty_rows(layer.w13_trellis)
    w13_suh = empty_rows(layer.w13_suh)
    w13_svh = empty_rows(layer.w13_svh)
    w2_trellis = empty_rows(layer.w2_trellis)
    w2_suh = empty_rows(layer.w2_suh)
    w2_svh = empty_rows(layer.w2_svh)
    with torch.no_grad():
        for source, output in zip(
            source_tensors,
            (w13_trellis, w13_suh, w13_svh, w2_trellis, w2_suh, w2_svh),
            strict=True,
        ):
            output[:dyn_capacity].copy_(source.index_select(0, seed_local))
        if n_pin > 0:
            for name, output in zip(
                _EXL3_EXPERT_SOURCE_NAMES,
                (w13_trellis, w13_suh, w13_svh, w2_trellis, w2_suh, w2_svh),
                strict=True,
            ):
                output[dyn_capacity:].copy_(getattr(layer, "_exl3_pin_" + name))
    if n_pin > 0:
        for name in _EXL3_EXPERT_SOURCE_NAMES:
            del layer._buffers["_exl3_pin_" + name]
        logger.info(
            "EXL3 pinned experts: layer=%s pinned=%d dynamic=%d host_rows=%d",
            layer_name, n_pin, dyn_capacity, num_local,
        )

    marker_suffix = str(getattr(layer, "_exl3_codebook", "mcg"))
    marker_value = (
        MCG_MARKER_SIGNED_INT32
        if marker_suffix == "mcg"
        else MUL1_MARKER_SIGNED_INT32
    )
    w13_marker = torch.full(
        (capacity, 2, 1), marker_value, dtype=torch.int32, device=device
    )
    w2_marker = torch.full(
        (capacity, 1), marker_value, dtype=torch.int32, device=device
    )
    inners: list[dict[str, Any]] = []
    for expert in range(capacity):
        inners.append(
            {
                "gate": make_linear_exl3(
                    w13_trellis[expert, 0],
                    w13_suh[expert, 0],
                    w13_svh[expert, 0],
                    w13_marker[expert, 0],
                    codebook=marker_suffix,
                ),
                "up": make_linear_exl3(
                    w13_trellis[expert, 1],
                    w13_suh[expert, 1],
                    w13_svh[expert, 1],
                    w13_marker[expert, 1],
                    codebook=marker_suffix,
                ),
                "down": make_linear_exl3(
                    w2_trellis[expert],
                    w2_suh[expert],
                    w2_svh[expert],
                    w2_marker[expert],
                    codebook=marker_suffix,
                ),
            }
        )
    cache_layer = types.SimpleNamespace(
        w13_trellis=w13_trellis,
        _exl3_hidden_size=int(layer._exl3_hidden_size),
        _exl3_intermediate_local=int(layer._exl3_intermediate_local),
        _exl3_bits=int(layer._exl3_bits),
    )
    build_exl3_fused_state(cache_layer, inners)

    source_ptrs = getattr(layer, "_exl3_ptrs", None)
    source_temps = getattr(layer, "_exl3_fused_temps", None)
    if not source_ptrs or source_temps is None:
        raise RuntimeError(
            "EXL3 dynamic LRU requires source fused state before cache init"
        )
    hybrid_ptrs = {
        name: source_ptrs[name].clone() for name in _EXL3_MOE_POINTER_NAMES
    }
    hybrid_ptrs["mcg"] = source_ptrs["mcg"]
    hybrid_ptrs["mul1"] = source_ptrs["mul1"]
    hybrid_layer = types.SimpleNamespace(
        _exl3_ptrs=hybrid_ptrs,
        _exl3_fused_temps=source_temps,
        _exl3_fused_concurrency=int(layer._exl3_fused_concurrency),
        _exl3_k=int(layer._exl3_k),
    )
    local_to_global_cpu = [-1] * num_local
    for global_id, local_id in enumerate(source_map_cpu):
        if local_id >= 0:
            local_to_global_cpu[local_id] = global_id
    if any(value < 0 for value in local_to_global_cpu):
        raise RuntimeError(f"EXL3 source map is not invertible for {layer_name}")
    local_to_global = torch.tensor(
        local_to_global_cpu, dtype=torch.long, device=device
    )

    cache_map = torch.full(
        (global_experts,), -1, dtype=torch.long, device=device
    )
    cache_map[
        torch.tensor(seed_global, dtype=torch.long, device=device)
    ] = torch.arange(capacity, dtype=torch.long, device=device)
    max_update_tokens = int(os.environ.get("EXL3_DYNAMIC_LRU_MAX_TOKENS", "1"))
    max_requests = max(1, max_update_tokens) * int(getattr(layer, "top_k", 8))
    protected_pct = int(os.environ.get("EXL3_LRU_SLRU_PROTECTED_PCT", "0"))
    if protected_pct < 0 or protected_pct >= 100:
        raise ValueError("EXL3_LRU_SLRU_PROTECTED_PCT must be in [0, 99]")
    protected_capacity = dyn_capacity * protected_pct // 100
    copy_stream = _EXL3_LRU_COPY_STREAMS.get(str(device))
    if copy_stream is None:
        # SM-issued UVA gathers compete with routed and shared-expert kernels;
        # a high-priority stream can reserve newly available blocks for the
        # demand fill without changing cache policy or numerics.  Keep the
        # default priority as the control and resolve the option once at load
        # so CUDA-graph replay has a stable stream topology.
        copy_priority = int(os.environ.get("EXL3_LRU_COPY_STREAM_PRIORITY", "0"))
        copy_stream = torch.cuda.Stream(device=device, priority=copy_priority)
        _EXL3_LRU_COPY_STREAMS[str(device)] = copy_stream
    cache = _Exl3LRUCache(
        w13_trellis=w13_trellis,
        w13_suh=w13_suh,
        w13_svh=w13_svh,
        w13_marker=w13_marker,
        w2_trellis=w2_trellis,
        w2_suh=w2_suh,
        w2_svh=w2_svh,
        w2_marker=w2_marker,
        source_map=source_map,
        cache_map=cache_map,
        cold_map=source_map.clone(),
        hit_map=torch.full_like(source_map, -1),
        miss_map=torch.full_like(source_map, -1),
        slot_global_ids=torch.tensor(
            seed_global[:dyn_capacity], dtype=torch.long, device=device
        ),
        slot_ages=torch.arange(
            dyn_capacity, 0, -1, dtype=torch.long, device=device
        ),
        slot_segments=(
            torch.arange(dyn_capacity, device=device) < protected_capacity
        ).to(torch.int32),
        clock=torch.tensor([dyn_capacity], dtype=torch.long, device=device),
        miss_local_ids=torch.empty(
            max_requests, dtype=torch.long, device=device
        ),
        miss_slots=torch.empty(max_requests, dtype=torch.long, device=device),
        counters=torch.zeros(2, dtype=torch.long, device=device),
        copy_stream=copy_stream,
        layer=cache_layer,
        hybrid_layer=hybrid_layer,
        inners=inners,
        local_to_global=local_to_global,
        capacity=dyn_capacity,
        protected_capacity=protected_capacity,
        max_update_tokens=max_update_tokens,
        expert_bytes=sum(
            int(source[0].numel()) * source.element_size()
            for source in source_tensors
        ),
        layer_name=layer_name,
    )
    layer._exl3_dynamic_lru_cache = cache
    _EXL3_LRU_CACHES.append(cache)
    cache._pf_sources = (
        layer.w13_trellis, layer.w13_suh, layer.w13_svh,
        layer.w2_trellis, layer.w2_suh, layer.w2_svh,
    )
    cache._pf_request = None
    cache._pf_pending = None
    _exl3_pf_reserve(w13_trellis.device, cache._pf_sources)
    cache_bytes = sum(
        tensor.numel() * tensor.element_size()
        for tensor in (
            w13_trellis,
            w13_suh,
            w13_svh,
            w13_marker,
            w2_trellis,
            w2_suh,
            w2_svh,
            w2_marker,
        )
    )
    logger.info(
        "Dynamic EXL3 LRU: layer=%s local=%d capacity=%d protected=%d "
        "cache=%.2f MiB update_tokens<=%d source=UVA",
        layer_name,
        num_local,
        capacity,
        protected_capacity,
        cache_bytes / (1024 * 1024),
        max_update_tokens,
    )



_EXL3_PF: dict[str, dict[str, Any]] = {}


def exl3_prefetch_k() -> int:
    return int(os.environ.get("EXL3_PREFETCH_K", "0"))


def exl3_prefetch_slots() -> int:
    k = exl3_prefetch_k()
    return int(os.environ.get("EXL3_PREFETCH_SLOTS", str(k))) if k > 0 else 0


def _exl3_pf_state(device: torch.device) -> dict[str, Any]:
    key = str(device)
    st = _EXL3_PF.get(key)
    if st is None:
        st = {"row_bytes": [0] * 6, "staging": [None, None], "rows": [None, None],
              "events": [None, None], "stream": None}
        _EXL3_PF[key] = st
    return st


def _exl3_pf_reserve(device: torch.device, sources: tuple[torch.Tensor, ...]) -> None:
    """Grow the two parity staging buffers to fit this layer's rows (load time only)."""
    k = exl3_prefetch_slots()
    if k <= 0:
        return
    st = _exl3_pf_state(device)
    rb = [int(t[0].numel() * t.element_size()) for t in sources]
    grow = any(r > old for r, old in zip(rb, st["row_bytes"]))
    if not grow and st["staging"][0] is not None:
        return
    st["row_bytes"] = [max(r, old) for r, old in zip(rb, st["row_bytes"])]
    for b in range(2):
        st["staging"][b] = [
            torch.empty(k * r + 16, dtype=torch.uint8, device=device) for r in st["row_bytes"]
        ]
        st["rows"][b] = torch.full((k,), -1, dtype=torch.long, device=device)


def _exl3_find_lru_cache(module: torch.nn.Module):
    cache = getattr(module, "_exl3_dynamic_lru_cache", None)
    if cache is not None:
        return cache
    for sub in module.modules():
        cache = getattr(sub, "_exl3_dynamic_lru_cache", None)
        if cache is not None:
            return cache
    return None


def exl3_request_prefetch(cur_module, next_module, pred: torch.Tensor) -> None:
    """Called by the model for single-token forwards: prefetch pred (global ids) for next layer."""
    cur = _exl3_find_lru_cache(cur_module)
    nxt = _exl3_find_lru_cache(next_module)
    if cur is None or nxt is None or getattr(nxt, "_pf_sources", None) is None:
        return
    cur._pf_request = (nxt, pred.reshape(-1).to(torch.long))


def _exl3_layer_parity(cache) -> int:
    name = str(getattr(cache, "layer_name", ""))
    try:
        return int(name.split("layers.")[1].split(".")[0]) % 2
    except (IndexError, ValueError):
        return 0



def _exl3_lru_integrity(cache, source_layer) -> None:
    """Debug audit; see patches/patch_lru_integrity.py."""
    if torch.cuda.is_current_stream_capturing():
        return
    torch.cuda.synchronize()
    pairs = (
        (source_layer.w13_trellis, cache.w13_trellis), (source_layer.w13_suh, cache.w13_suh),
        (source_layer.w13_svh, cache.w13_svh), (source_layer.w2_trellis, cache.w2_trellis),
        (source_layer.w2_suh, cache.w2_suh), (source_layer.w2_svh, cache.w2_svh),
    )
    slots = cache.slot_global_ids.tolist()
    cmap = cache.cache_map.tolist()
    smap = cache.source_map.tolist()
    bad, checked, map_bad = [], 0, 0
    for slot, g in enumerate(slots):
        if g < 0:
            continue
        if cmap[g] != slot:
            map_bad += 1
        row = smap[g]
        if row < 0:
            continue
        checked += 1
        for c, (src, dst) in enumerate(pairs):
            a = src[row].reshape(-1).view(torch.int16) if src.element_size() == 2 else src[row].reshape(-1)
            b = dst[slot].reshape(-1).view(torch.int16) if dst.element_size() == 2 else dst[slot].reshape(-1)
            if not torch.equal(a, b):
                bad.append((slot, g, c))
                break
    logger.warning(
        "EXL3 LRU integrity %s: checked=%d bad=%d map_mismatch=%d first_bad=%s",
        cache.layer_name, checked, len(bad), map_bad, bad[:4],
    )



# ----------------------------------------------------------------------------------------------
# Hybrid CPU+GPU decode (patches/patch_hybrid_cpu_moe.py)
# ----------------------------------------------------------------------------------------------
_EXL3_HYBRID: dict[str, Any] = {}


def exl3_hybrid_cpu_enabled() -> bool:
    return os.environ.get("EXL3_HYBRID_CPU", "0").lower() in ("1", "true", "yes")


def _exl3_hybrid_state(device: torch.device) -> dict[str, Any]:
    st = _EXL3_HYBRID.get("state")
    if st is None:
        import glm_hybrid

        rank = device.index or 0
        cpus = [int(c) for c in os.environ.get(f"EXL3_HYBRID_CPUS_R{rank}", "").split(",") if c.strip()]
        st = {
            "ext": glm_hybrid,
            "region": glm_hybrid.create(64, 8192),
            "dev_seq": torch.zeros(64, dtype=torch.int32, device=device),
            "slots": {},
            "threads": int(os.environ.get("EXL3_HYBRID_THREADS", "12")),
            "cpus": cpus,
            "timeout": float(os.environ.get("EXL3_HYBRID_TIMEOUT_S", "5")),
            "started": False,
        }
        _EXL3_HYBRID["state"] = st
        logger.info("EXL3 hybrid CPU decode: rank %d, %d threads on cpus %s", rank, st["threads"], cpus)
    return st


def _exl3_hybrid_register(st: dict[str, Any], cache: Any, source_layer: torch.nn.Module, limit: float) -> int:
    key = id(cache)
    slot = st["slots"].get(key)
    if slot is not None:
        return slot
    gh = st["ext"]
    slot = len(st["slots"])
    if slot >= 64:
        raise RuntimeError("EXL3 hybrid: more than 64 MoE layers")
    # CPU aliases of this rank's host masters (UVA-offloaded parameters); slice on the CPU side
    w13t = gh.cpu_view(source_layer.w13_trellis)
    w13s = gh.cpu_view(source_layer.w13_suh)
    w13v = gh.cpu_view(source_layer.w13_svh)
    w2t = gh.cpu_view(source_layer.w2_trellis)
    w2s = gh.cpu_view(source_layer.w2_suh)
    w2v = gh.cpu_view(source_layer.w2_svh)
    E = int(w13t.shape[0])
    gt = [w13t[e, 0] for e in range(E)]
    ut = [w13t[e, 1] for e in range(E)]
    gs = [w13s[e, 0] for e in range(E)]
    us = [w13s[e, 1] for e in range(E)]
    gv = [w13v[e, 0] for e in range(E)]
    uv = [w13v[e, 1] for e in range(E)]
    dt = [w2t[e] for e in range(E)]
    ds = [w2s[e] for e in range(E)]
    dv = [w2v[e] for e in range(E)]
    act_limit = float(limit) if limit is not None and limit == limit and abs(float(limit)) != float("inf") else 0.0
    handle = gh.make_layer(gt, gs, gv, ut, us, uv, dt, ds, dv, [], [], [], 0, act_limit, 0)
    gh.set_layer(st["region"], slot, handle)
    st["slots"][key] = slot
    cache._hyb_fallback = cache.slot_global_ids[:1]
    if not st["started"]:
        gh.start(st["region"], st["threads"], st["cpus"])
        st["started"] = True
    return slot


def _exl3_hyb_lru_update(cache, ids_flat, num_ids):
    _update_exl3_lru_map_kernel[(1,)](
        ids_flat,
        cache.source_map,
        cache.cache_map,
        cache.slot_global_ids,
        cache.slot_ages,
        cache.clock,
        cache.miss_local_ids,
        cache.miss_slots,
        cache.counters,
        num_ids=num_ids,
        global_num_experts=int(cache.cache_map.numel()),
        capacity=cache.capacity,
        id_block=triton.next_power_of_2(num_ids),
        capacity_block=triton.next_power_of_2(cache.capacity),
        num_warps=4,
    )


def _apply_exl3_hybrid_decode(x2d, ids, weights, source_layer, cache, limit):
    st = _exl3_hybrid_state(x2d.device)
    slot = _exl3_hybrid_register(st, cache, source_layer, limit)
    gh = st["ext"]
    current = torch.cuda.current_stream(x2d.device)
    # (a) this layer's experts prefetched during the previous layer: wait before deciding residency
    if getattr(cache, "_hyb_pf_pending", False):
        current.wait_event(cache._hyb_pf_event)
        cache._hyb_pf_pending = False
    # (b) prefetch the next layer's predicted experts into its VRAM cache on a side stream
    req = getattr(cache, "_pf_request", None)
    cache._pf_request = None
    if req is not None and os.environ.get("EXL3_HYBRID_PREFETCH", "1").lower() in ("1", "true", "yes"):
        nxt, pred = req
        n_pred = int(pred.numel())
        srcs = getattr(nxt, "_pf_sources", None)
        if srcs is not None and 0 < n_pred <= int(nxt.miss_local_ids.numel()) and n_pred <= nxt.capacity:
            pf = st.get("pf_stream")
            if pf is None:
                pf = torch.cuda.Stream(device=x2d.device)
                st["pf_stream"] = pf
            if getattr(nxt, "_hyb_pf_event", None) is None:
                nxt._hyb_pf_event = torch.cuda.Event()
            pf.wait_stream(current)
            pred.record_stream(pf)
            with torch.cuda.stream(pf):
                _exl3_hyb_lru_update(nxt, pred, n_pred)
                nxt_transfers = (
                    (srcs[0], nxt.w13_trellis), (srcs[1], nxt.w13_suh), (srcs[2], nxt.w13_svh),
                    (srcs[3], nxt.w2_trellis), (srcs[4], nxt.w2_suh), (srcs[5], nxt.w2_svh),
                )
                _gather_exl3_lru_transfers(nxt_transfers, nxt.miss_local_ids, nxt.miss_slots, n_pred)
                nxt._hyb_pf_event.record(pf)
            nxt._hyb_pf_pending = True
    x = x2d.contiguous()
    ids_c = ids.contiguous()
    w = weights.to(torch.float32).contiguous()
    gpu_ids = torch.empty_like(ids_c)
    gpu_w = torch.empty_like(w)
    gpu_admit = torch.empty_like(ids_c)
    frac = float(os.environ.get("EXL3_HYBRID_GPU_FRAC", "0"))
    global_ids = ids_c.reshape(-1)
    num_ids = int(global_ids.numel())
    admit = (
        os.environ.get("EXL3_HYBRID_ADMIT", "1").lower() in ("1", "true", "yes")
        and num_ids <= int(cache.miss_local_ids.numel())
        and num_ids <= cache.capacity
    )
    if not admit:
        frac = 0.0   # the GPU subset needs admission to land in VRAM
    fused = (
        frac > 0.0
        and os.environ.get("EXL3_HYBRID_FUSED_SUBMIT", "0") == "1"
        and hasattr(gh, "submit_lru")
        and gh.submit_lru(st["region"], slot, x, ids_c, w, cache._hyb_fallback, st["dev_seq"], gpu_ids, gpu_w,
                          gpu_admit, frac, cache.source_map, cache.cache_map, cache.slot_global_ids,
                          cache.slot_ages, cache.clock, cache.miss_local_ids, cache.miss_slots, cache.counters,
                          int(cache.capacity))
    )
    if not fused:
        gh.submit(st["region"], slot, x, ids_c, w, cache.cache_map, cache.source_map,
                  cache._hyb_fallback, st["dev_seq"], gpu_ids, gpu_w, gpu_admit, frac)
    transfers = (
        (source_layer.w13_trellis, cache.w13_trellis),
        (source_layer.w13_suh, cache.w13_suh),
        (source_layer.w13_svh, cache.w13_svh),
        (source_layer.w2_trellis, cache.w2_trellis),
        (source_layer.w2_suh, cache.w2_suh),
        (source_layer.w2_svh, cache.w2_svh),
    )
    current = torch.cuda.current_stream(x2d.device)
    overlap = (
        frac > 0.0
        and os.environ.get("EXL3_HYBRID_OVERLAP_GATHER", "0") == "1"
        and _exl3_gemv_moe_eligible(x2d, gpu_ids, cache.layer)
    )
    if frac > 0.0:
        # GPU subset: admit + copy now, so this GEMV computes it from VRAM
        if not fused:
            _exl3_hyb_lru_update(cache, gpu_admit.reshape(-1), num_ids)
        if overlap:
            gs = st.get("gather_stream")
            if gs is None:
                gs = torch.cuda.Stream(device=x2d.device)
                st["gather_stream"] = gs
                st["gather_event"] = torch.cuda.Event()
            gs.wait_stream(current)
            with torch.cuda.stream(gs):
                _gather_exl3_lru_transfers(transfers, cache.miss_local_ids, cache.miss_slots, num_ids)
                st["gather_event"].record(gs)
        else:
            _gather_exl3_lru_transfers(transfers, cache.miss_local_ids, cache.miss_slots, num_ids)
    if overlap:
        out = _apply_exl3_gemv_moe_overlap(
            x2d, gpu_ids, gpu_w, cache.layer, len(cache.inners), cache.cache_map, limit,
            cache.miss_slots, st["gather_event"],
        )
    else:
        out = apply_exl3_fused_moe(x2d, gpu_ids, gpu_w, cache.layer, cache.inners, cache.cache_map, limit)
    admit_cpu = os.environ.get("EXL3_HYBRID_ADMIT_CPU", "1").lower() in ("1", "true", "yes")
    if admit and (admit_cpu or frac <= 0.0):
        # CPU subset (and LRU touch of everything used): admit now, copy in the background
        _exl3_hyb_lru_update(cache, global_ids, num_ids)
        side = st.get("copy_stream")
        if side is None:
            side = torch.cuda.Stream(device=x2d.device)
            st["copy_stream"] = side
        side.wait_stream(current)
        with torch.cuda.stream(side):
            _gather_exl3_lru_transfers(transfers, cache.miss_local_ids, cache.miss_slots, num_ids)
        st["pending_join"] = True
    gh.wait_add(st["region"], slot, st["dev_seq"], out, st["timeout"])
    n_layers = st.get("n_layers") or int(os.environ.get("EXL3_HYBRID_LAYERS", "0"))
    if n_layers <= 0 and slot == 0 and len(st["slots"]) > 1:
        st["n_layers"] = n_layers = len(st["slots"])
    if st.get("pending_join") and (n_layers <= 0 or slot == n_layers - 1):
        current.wait_stream(st["copy_stream"])
        st["pending_join"] = False
    return out

def _apply_exl3_dynamic_lru(
    x2d: torch.Tensor,
    ids: torch.Tensor,
    weights: torch.Tensor,
    source_layer: torch.nn.Module,
    source_inners: list[dict[str, Any]],
    cache: _Exl3LRUCache,
    limit: float,
) -> torch.Tensor:
    global _EXL3_LRU_APPLY_CALLS
    _EXL3_LRU_APPLY_CALLS += 1
    # Prefetch must read the SAME host masters as the demand fill (the UVA offloader may re-wrap
    # parameters after cache init), so refresh them on every call.
    cache._pf_sources = (
        source_layer.w13_trellis, source_layer.w13_suh, source_layer.w13_svh,
        source_layer.w2_trellis, source_layer.w2_suh, source_layer.w2_svh,
    )
    log_every = int(os.environ.get("EXL3_LRU_LOG_EVERY", "0"))
    if log_every > 0 and _EXL3_LRU_APPLY_CALLS % log_every == 0:
        peers = [
            value
            for value in _EXL3_LRU_CACHES
            if value.counters.device == cache.counters.device
        ]
        totals = torch.stack([value.counters for value in peers]).sum(dim=0)
        missed_bytes = torch.stack(
            [value.counters[1] * value.expert_bytes for value in peers]
        ).sum()
        hits, misses, missed_bytes_host = (
            int(value)
            for value in torch.cat((totals, missed_bytes.reshape(1))).tolist()
        )
        logger.info(
            "EXL3 LRU counters device=%s calls=%d layers=%d hits=%d misses=%d "
            "hit_rate=%.6f missed_bytes=%d",
            cache.counters.device,
            _EXL3_LRU_APPLY_CALLS,
            len(peers),
            hits,
            misses,
            hits / max(hits + misses, 1),
            missed_bytes_host,
        )
    tokens = int(x2d.shape[0])
    if (
        exl3_hybrid_cpu_enabled()
        and tokens <= int(os.environ.get("EXL3_HYBRID_MAX_TOKENS", "4"))
        and not exl3_deterministic_moe_enabled()
    ):
        return _apply_exl3_hybrid_decode(x2d, ids, weights, source_layer, cache, limit)
    global_ids = ids.reshape(-1)
    num_ids = int(global_ids.numel())
    trace_path = os.environ.get("EXL3_LRU_TRACE_PATH", "")
    trace_max_calls = int(os.environ.get("EXL3_LRU_TRACE_MAX_CALLS", "0"))
    if (
        trace_path
        and tokens <= cache.max_update_tokens
        and (trace_max_calls <= 0 or _EXL3_LRU_APPLY_CALLS <= trace_max_calls)
    ):
        # Deliberately synchronizing and opt-in: route-policy collection runs in
        # eager mode and is never used for throughput measurements.  Device and
        # PID substitutions keep EP workers from sharing a JSONL writer.
        device_index = cache.counters.device.index
        resolved_path = trace_path.format(
            device=device_index if device_index is not None else 0,
            pid=os.getpid(),
        )
        handle = _EXL3_LRU_TRACE_HANDLES.get(resolved_path)
        if handle is None:
            Path(resolved_path).parent.mkdir(parents=True, exist_ok=True)
            handle = open(resolved_path, "a", encoding="utf-8", buffering=1)
            _EXL3_LRU_TRACE_HANDLES[resolved_path] = handle
        handle.write(
            json.dumps(
                {
                    "schema": 1,
                    "pid": os.getpid(),
                    "device": device_index,
                    "call": _EXL3_LRU_APPLY_CALLS,
                    "layer": cache.layer_name,
                    "tokens": tokens,
                    "ids": [int(value) for value in global_ids.tolist()],
                },
                separators=(",", ":"),
            )
            + "\n"
        )
    can_update = (
        tokens <= cache.max_update_tokens
        and num_ids <= int(cache.miss_local_ids.numel())
        and num_ids <= cache.capacity
    )
    if can_update:
        use_slru = (
            cache.protected_capacity > 0
            and num_ids <= cache.capacity - cache.protected_capacity
        )
        if use_slru:
            _update_exl3_slru_map_kernel[(1,)](
                global_ids,
                cache.source_map,
                cache.cache_map,
                cache.slot_global_ids,
                cache.slot_ages,
                cache.slot_segments,
                cache.clock,
                cache.miss_local_ids,
                cache.miss_slots,
                cache.counters,
                num_ids=num_ids,
                global_num_experts=int(cache.cache_map.numel()),
                capacity=cache.capacity,
                protected_capacity=cache.protected_capacity,
                id_block=triton.next_power_of_2(num_ids),
                capacity_block=triton.next_power_of_2(cache.capacity),
                num_warps=4,
            )
        else:
            _update_exl3_lru_map_kernel[(1,)](
                global_ids,
                cache.source_map,
                cache.cache_map,
                cache.slot_global_ids,
                cache.slot_ages,
                cache.clock,
                cache.miss_local_ids,
                cache.miss_slots,
                cache.counters,
                num_ids=num_ids,
                global_num_experts=int(cache.cache_map.numel()),
                capacity=cache.capacity,
                id_block=triton.next_power_of_2(num_ids),
                capacity_block=triton.next_power_of_2(cache.capacity),
                num_warps=4,
            )
        transfers = (
            (source_layer.w13_trellis, cache.w13_trellis),
            (source_layer.w13_suh, cache.w13_suh),
            (source_layer.w13_svh, cache.w13_svh),
            (source_layer.w2_trellis, cache.w2_trellis),
            (source_layer.w2_suh, cache.w2_suh),
            (source_layer.w2_svh, cache.w2_svh),
        )
        overlap = os.environ.get("EXL3_LRU_OVERLAP", "0").lower() in (
            "1",
            "true",
            "yes",
        )
        if overlap:
            global_experts = int(cache.cache_map.numel())
            _split_exl3_lru_work_maps_kernel[(1,)](
                global_ids,
                cache.source_map,
                cache.cache_map,
                cache.miss_slots,
                cache.hit_map,
                cache.miss_map,
                num_ids=num_ids,
                global_num_experts=global_experts,
                global_block=triton.next_power_of_2(global_experts),
                id_block=triton.next_power_of_2(num_ids),
                num_warps=4,
            )
            current_stream = torch.cuda.current_stream(x2d.device)
            with torch.cuda.stream(cache.copy_stream):
                cache.copy_stream.wait_stream(current_stream)
                _gather_exl3_lru_transfers(
                    transfers,
                    cache.miss_local_ids,
                    cache.miss_slots,
                    num_ids,
                )
            base_hint = int(os.environ.get("EXL3_MOE_DECODE_GROUPS", "0"))
            hit_hint = int(os.environ.get("EXL3_MOE_HIT_GROUPS", str(base_hint)))
            miss_hint = int(os.environ.get("EXL3_MOE_MISS_GROUPS", str(base_hint)))
            hit = apply_exl3_fused_moe(
                x2d,
                ids,
                weights,
                cache.layer,
                cache.inners,
                cache.hit_map,
                limit,
                decode_group_hint_override=hit_hint,
            )
            current_stream.wait_stream(cache.copy_stream)
            miss = apply_exl3_fused_moe(
                x2d,
                ids,
                weights,
                cache.layer,
                cache.inners,
                cache.miss_map,
                limit,
                decode_group_hint_override=miss_hint,
            )
            return hit + miss

        pf_k = exl3_prefetch_slots()
        pending = getattr(cache, "_pf_pending", None)
        cache._pf_pending = None
        current_stream = torch.cuda.current_stream(x2d.device)
        if pf_k > 0 and pending is not None:
            import exllamav3_ext

            st = _exl3_pf_state(x2d.device)
            current_stream.wait_event(st["events"][pending])
            exllamav3_ext.exl3_lru_gather_staged(
                [source for source, _ in transfers],
                [output for _, output in transfers],
                st["staging"][pending],
                st["rows"][pending],
                pf_k,
                cache.miss_local_ids,
                cache.miss_slots,
                num_ids,
                int(os.environ.get("EXL3_LRU_GATHER_CUDA_BLOCKS", "4")),
                1024,
            )
        else:
            _gather_exl3_lru_transfers(
                transfers,
                cache.miss_local_ids,
                cache.miss_slots,
                num_ids,
            )
        request = getattr(cache, "_pf_request", None)
        cache._pf_request = None
        if pf_k > 0 and request is not None:
            import exllamav3_ext

            nxt, pred = request
            st = _exl3_pf_state(x2d.device)
            b = _exl3_layer_parity(nxt)
            if st["stream"] is None:
                st["stream"] = torch.cuda.Stream(device=x2d.device)
                st["events"] = [torch.cuda.Event(), torch.cuda.Event()]
            pf_stream = st["stream"]
            pf_stream.wait_stream(current_stream)
            # pred was allocated on the main stream; keep its block alive for the side stream
            pred.record_stream(pf_stream)
            with torch.cuda.stream(pf_stream):
                exllamav3_ext.exl3_lru_prefetch(
                    list(nxt._pf_sources),
                    st["staging"][b],
                    pred[:pf_k],
                    nxt.cache_map,
                    nxt.source_map,
                    st["rows"][b],
                    min(pf_k, int(pred.numel())),
                    int(os.environ.get("EXL3_PREFETCH_BLOCKS", "4")),
                    1024,
                )
                st["events"][b].record(pf_stream)
            nxt._pf_pending = b
        return apply_exl3_fused_moe(
            x2d,
            ids,
            weights,
            cache.layer,
            cache.inners,
            cache.cache_map,
            limit,
        )

    # The hybrid pointer-table path keeps one expert namespace and selects the
    # VRAM slot
    # or UVA master independently for every expert.  This lets the ticket
    # scheduler interleave hot compute with cold PHB reads in one fused launch,
    # avoiding duplicate routing/sorting and a second MoE launch.  Keep the
    # original two-pass path as an A/B fallback until real prefill validates it.
    hybrid_prefill = os.environ.get("EXL3_LRU_HYBRID_PREFILL", "0").lower() in (
        "1",
        "true",
        "yes",
    )
    if hybrid_prefill:
        _refresh_exl3_hybrid_ptrs(cache, source_layer)
        hybrid_groups = int(
            os.environ.get("EXL3_MOE_PREFILL_HYBRID_GROUPS", "0")
        )
        return apply_exl3_fused_moe(
            x2d,
            ids,
            weights,
            cache.hybrid_layer,
            source_inners,
            cache.source_map,
            limit,
            decode_group_hint_override=(hybrid_groups or None),
        )

    if os.path.exists("/home/bot/llm/probe/LRU_CHECK_ON"):
        _exl3_lru_integrity(cache, source_layer)
    # Broad prompt scans never mutate the decode-local LRU.  Build a map that
    # masks resident experts out of the UVA master.  The optional DMA path
    # evaluates the same hot pass from VRAM, then copies only cold physical
    # expert intervals into a small double buffer.  Unlike the original stage
    # experiment, fat experts remain eligible and are row-tiled after staging.
    global_experts = int(cache.cache_map.numel())
    _build_exl3_cold_map_kernel[(1,)](
        cache.source_map,
        cache.cache_map,
        cache.cold_map,
        num_experts=global_experts,
        block_size=triton.next_power_of_2(global_experts),
        num_warps=4,
    )
    hot_groups = int(os.environ.get("EXL3_MOE_PREFILL_HOT_GROUPS", "0"))
    cold_groups = int(os.environ.get("EXL3_MOE_PREFILL_COLD_GROUPS", "0"))
    cold_routes = None
    if (
        exl3_prefill_fixed_tiles_enabled()
        and int(os.environ.get("EXL3_PREFILL_DMA_STAGE_EXPERTS", "0")) > 0
        and int(x2d.shape[0]) >= int(os.environ.get("EXL3_PREFILL_DMA_MIN_TOKENS", "2"))
    ):
        cold_routes = _exl3_dma_prefill_routes(x2d, ids, weights, source_layer, cache.cold_map)
    hot = apply_exl3_fused_moe(
        x2d,
        ids,
        weights,
        cache.layer,
        cache.inners,
        cache.cache_map,
        limit,
        decode_group_hint_override=(hot_groups or None),
    )
    staged = _apply_exl3_dma_staged_prefill(
        x2d,
        ids,
        weights,
        source_layer,
        cache.cold_map,
        cache.copy_stream,
        limit,
        routes=cold_routes,
    )
    if staged is not None:
        return hot + staged
    cold = apply_exl3_fused_moe(
        x2d,
        ids,
        weights,
        source_layer,
        source_inners,
        cache.cold_map,
        limit,
        decode_group_hint_override=(cold_groups or None),
    )
    return hot + cold


@register_quantization_config("exl3")
class Exl3Config(QuantizationConfig):
    """Routed-experts-only EXL3. Dense / shared / attention stay native."""

    def __init__(
        self,
        bits: int = 4,
        head_bits: int = 6,
        vision_bits: int = 3,
        codebook: str = "mcg",
        scope: str = "glm53_routed_experts_only",
        tensor_bits: dict[str, int] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.bits = int(bits)
        self.head_bits = int(head_bits)
        self.vision_bits = int(vision_bits)
        self.codebook = str(codebook)
        self.scope = str(scope)
        self.tensor_bits = {
            str(name): int(value) for name, value in (tensor_bits or {}).items()
        }
        self.raw_config = dict(kwargs)
        if self.codebook not in ("mcg", "mul1"):
            raise ValueError(f"unsupported EXL3 codebook={self.codebook!r}")
        for label, value in (
            ("bits", self.bits),
            ("head_bits", self.head_bits),
            ("vision_bits", self.vision_bits),
        ):
            if value not in (2, 3, 4, 5, 6, 8):
                raise ValueError(f"unsupported EXL3 {label}={value}")

    def get_name(self) -> str:
        return "exl3"

    def get_supported_act_dtypes(self) -> list[torch.dtype]:
        return [torch.bfloat16, torch.float16, torch.float32]

    @classmethod
    def get_min_capability(cls) -> int:
        # LinearEXL3 uses CUDA >= Ampere; GB10 is SM121.
        return 80

    @staticmethod
    def get_config_filenames() -> list[str]:
        return ["quantization_config.json"]

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "Exl3Config":
        tensor_storage = config.get("tensor_storage", {})
        if not tensor_storage:
            # A speculative drafter has its own quantization config, but vLLM
            # instantiates both quant configs in one worker.  Do not feed the
            # target model's 37 MiB ledger to the much smaller draft model.
            # DFlash2's published EXL3 checkpoint is currently identifiable by
            # its K8 head; keep an explicit override for future checkpoints.
            draft_ledger_path = os.getenv("EXL3_DRAFT_TENSOR_LEDGER", "")
            is_dflash_draft = (
                bool(draft_ledger_path)
                and int(config.get("head_bits", 6)) == 8
            )
            ledger_path = (
                draft_ledger_path
                if is_dflash_draft
                else os.getenv("EXL3_TENSOR_LEDGER", "")
            )
            if ledger_path:
                ledger_config = json.loads(Path(ledger_path).read_text())
                if ledger_config.get("quant_method") != "exl3":
                    raise ValueError(
                        f"EXL3 tensor ledger {ledger_path} has quant_method="
                        f"{ledger_config.get('quant_method')!r}"
                    )
                tensor_storage = ledger_config.get("tensor_storage", {})
        tensor_bits = {
            str(name): int(entry["bits_per_weight"])
            for name, entry in tensor_storage.items()
            if isinstance(entry, dict)
            and entry.get("quant_format") == "exl3"
            and "bits_per_weight" in entry
        }
        # The converted GLM checkpoint stores native NextN weights in a
        # separate mtp.safetensors file, so its main 37-MiB tensor ledger has
        # no layer-45 entries.  Merge a compact ledger derived directly from
        # the sidecar's trellis shapes; reject conflicts rather than silently
        # changing an existing target tensor's K width.
        mtp_bits_path = os.getenv("EXL3_MTP_TENSOR_BITS", "")
        if mtp_bits_path:
            mtp_record = json.loads(Path(mtp_bits_path).read_text())
            mtp_bits = mtp_record.get("tensor_bits", {})
            if not isinstance(mtp_bits, dict):
                raise ValueError(
                    f"EXL3 MTP tensor ledger {mtp_bits_path} lacks tensor_bits"
                )
            for name, value in mtp_bits.items():
                bits = int(value)
                previous = tensor_bits.get(str(name))
                if previous is not None and previous != bits:
                    raise ValueError(
                        f"EXL3 tensor K conflict for {name}: "
                        f"target=K{previous} mtp=K{bits}"
                    )
                tensor_bits[str(name)] = bits
        skip = {
            "bits",
            "head_bits",
            "vision_bits",
            "codebook",
            "scope",
            "quant_method",
            # tr3 ships a 37 MiB per-tensor ledger; keep it off the config object.
            "tensor_storage",
        }
        return cls(
            bits=int(config.get("bits", 4)),
            head_bits=int(config.get("head_bits", 6)),
            vision_bits=int(config.get("vision_bits", config.get("bits", 4))),
            codebook=str(config.get("codebook", "mcg")),
            scope=str(config.get("scope", "glm53_routed_experts_only")),
            tensor_bits=tensor_bits,
            **{k: v for k, v in config.items() if k not in skip},
        )

    def linear_bits(self, prefix: str) -> int:
        """Resolve the checkpoint's per-tensor K value for a vLLM prefix.

        The advertised 3.05 bpw is an average: dense, shared, attention, and
        head tensors use K4/K5/K6 while routed experts use K3.  The Hub ledger
        names the backbone ``model.language_model`` whereas vLLM constructs it
        as ``language_model.model``; merged gate/up modules also need both
        source entries checked.
        """
        if not self.tensor_bits:
            if prefix == "lm_head" or prefix.endswith(".lm_head"):
                return self.head_bits
            if ".visual." in prefix:
                return self.vision_bits
            return self.bits

        if (
            prefix == "lm_head"
            or prefix.endswith(".lm_head")
            # Native MTP constructs a temporary per-layer SharedHead before
            # load_eagle_model replaces it with the already-loaded target LM
            # head.  The sidecar intentionally contains no duplicate head
            # weights, so give that temporary module the target head's EXL3
            # layout instead of requiring a nonexistent ledger entry.
            or re.fullmatch(r"model\.layers\.\d+\.head", prefix) is not None
        ):
            candidates = ["lm_head"]
        else:
            ledger_prefix = prefix
            if ledger_prefix.startswith("language_model.model."):
                ledger_prefix = (
                    "model.language_model."
                    + ledger_prefix.removeprefix("language_model.model.")
                )
            # vLLM mounts a DFlash2 drafter after the target's layer range
            # (45..49 for GLM-5.3-Flash), while its checkpoint ledger uses
            # draft-local names (layers.0..4).  Detect the draft ledger by its
            # local layer namespace and translate the runtime prefix.
            layer_match = re.match(r"model\.layers\.(\d+)(\..+)", ledger_prefix)
            if (
                layer_match is not None
                and any(name.startswith("layers.0.") for name in self.tensor_bits)
            ):
                layer_offset = int(os.getenv("EXL3_DFLASH_LAYER_OFFSET", "45"))
                draft_layer = int(layer_match.group(1)) - layer_offset
                if draft_layer >= 0:
                    ledger_prefix = f"layers.{draft_layer}{layer_match.group(2)}"
            elif layer_match is not None:
                # Native GLM MTP is constructed as text-only model.layers.45,
                # while its sidecar preserves the multimodal checkpoint's
                # model.language_model.layers.45 namespace.
                native_prefix = (
                    "model.language_model.layers."
                    f"{layer_match.group(1)}{layer_match.group(2)}"
                )
                native_layer_stem = (
                    "model.language_model.layers."
                    f"{layer_match.group(1)}."
                )
                if any(
                    name.startswith(native_layer_stem)
                    for name in self.tensor_bits
                ):
                    ledger_prefix = native_prefix
            elif (
                ledger_prefix.startswith("model.")
                and ledger_prefix.removeprefix("model.") in self.tensor_bits
            ):
                ledger_prefix = ledger_prefix.removeprefix("model.")
            candidates = [ledger_prefix]
            if ledger_prefix.endswith(".gate_up_proj"):
                stem = ledger_prefix.removesuffix(".gate_up_proj")
                candidates = [f"{stem}.gate_proj", f"{stem}.up_proj"]
            elif (
                ledger_prefix.endswith(".qkv_proj")
                and ledger_prefix not in self.tensor_bits
            ):
                stem = ledger_prefix.removesuffix(".qkv_proj")
                candidates = [
                    f"{stem}.q_proj",
                    f"{stem}.k_proj",
                    f"{stem}.v_proj",
                ]
            elif ledger_prefix.endswith(".fused_qkv_a_proj"):
                # vLLM fuses the DSA q_a and kv_a projections at runtime,
                # while the EXL3 checkpoint stores two independently packed
                # tensors.  DeepSeekV2FusedQkvAProjLinear exposes them as two
                # output shards, so accept the fusion only when their K values
                # agree (the 3.05-bpw GLM checkpoint uses K5 for both).
                stem = ledger_prefix.removesuffix(".fused_qkv_a_proj")
                candidates = [
                    f"{stem}.q_a_proj",
                    f"{stem}.kv_a_proj_with_mqa",
                ]

        found = [self.tensor_bits[name] for name in candidates if name in self.tensor_bits]
        if not found:
            raise KeyError(
                f"No EXL3 tensor-storage entry for vLLM linear {prefix!r}; "
                f"tried {candidates}"
            )
        if len(set(found)) != 1:
            raise ValueError(
                f"Merged EXL3 linear {prefix!r} has incompatible K values: "
                f"{dict(zip(candidates, found))}"
            )
        return found[0]

    def moe_bits(self, prefix: str) -> int:
        """Resolve the uniform routed-expert K value for one MoE layer."""
        if not self.tensor_bits:
            return self.bits
        match = re.search(r"(?:^|\.)layers\.(\d+)(?:\.|$)", prefix)
        if match is None:
            raise KeyError(f"Cannot resolve EXL3 MoE layer from prefix {prefix!r}")
        stem = f"model.language_model.layers.{match.group(1)}.mlp.experts."
        found = {
            bits for name, bits in self.tensor_bits.items() if name.startswith(stem)
        }
        if len(found) != 1:
            raise ValueError(
                f"EXL3 routed layer {prefix!r} requires one uniform K, found "
                f"{sorted(found)}"
            )
        return found.pop()

    @classmethod
    def override_quantization_method(
        cls,
        hf_quant_cfg: dict[str, Any],
        user_quant: str | None,
        hf_config: Any = None,
    ) -> str | None:
        method = str((hf_quant_cfg or {}).get("quant_method", "")).lower()
        if method == "exl3":
            return "exl3"
        return None

    def get_quant_method(self, layer: torch.nn.Module, prefix: str):
        from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts
        from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead

        if isinstance(layer, RoutedExperts):
            return Exl3MoEMethod(layer.moe_config, self, layer_name=prefix)
        if isinstance(layer, ParallelLMHead):
            # ParallelLMHead is a VocabParallelEmbedding rather than a
            # LinearBase, but its non-embedding path intentionally accepts a
            # linear quant method and passes the same create/apply contract.
            return Exl3LinearMethod(self, prefix=prefix)
        if isinstance(layer, LinearBase):
            if _is_exl3_linear_prefix(prefix):
                return Exl3LinearMethod(self, prefix=prefix)
            return UnquantizedLinearMethod()
        return None


def _is_exl3_linear_prefix(prefix: str) -> bool:
    """Match the tensors actually serialized as EXL3 in Unsloth's checkpoint."""
    return (
        prefix == "lm_head"
        or prefix.endswith(".lm_head")
        or prefix == "fc"
        or prefix.endswith(".fc")
        or prefix.endswith(".eh_proj")
        or prefix.endswith(".qkv_proj")
        or prefix.endswith(".self_attn.o_proj")
        or prefix.endswith(".self_attn.q_a_proj")
        or prefix.endswith(".self_attn.q_b_proj")
        or prefix.endswith(".self_attn.kv_a_proj_with_mqa")
        or prefix.endswith(".self_attn.fused_qkv_a_proj")
        or prefix.endswith(".self_attn.indexer.wq_b")
        or prefix.endswith(".mlp.gate_up_proj")
        or prefix.endswith(".mlp.down_proj")
        or prefix.endswith(".mlp.shared_experts.gate_up_proj")
        or prefix.endswith(".mlp.shared_experts.down_proj")
        or ".visual." in prefix
    )


_EXL3_BF16_IO = os.environ.get("EXL3_BF16_IO", "0") == "1"


def _exl3_linear_bf16(layer, x):
    """One activation row, bf16 in/out through the int8 sq GEMV; None if any inner is not covered."""
    inners = layer._exl3_linear_inners
    x2 = x.reshape(1, x.shape[-1])
    if not x2.is_contiguous():
        x2 = x2.contiguous()
    total = sum(int(inner.out_features) for inner in inners)
    out = torch.empty((1, total), dtype=torch.bfloat16, device=x.device)
    off = 0
    for inner in inners:
        n = int(inner.out_features)
        bc = getattr(inner, "bc", None)
        if bc is None or not hasattr(bc, "run_bf16") or not bc.run_bf16(x2, out[:, off:off + n]):
            layer._exl3_bf16_ok = False
            return None
        off += n
    return out


class Exl3LinearMethod(LinearMethodBase):
    """Packed EXL3 for dense/shared/attention/head LinearBase modules."""

    def __init__(self, quant_config: Exl3Config, *, prefix: str) -> None:
        self.quant_config = quant_config
        self.prefix = prefix
        self.bits = quant_config.linear_bits(prefix)

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs: Any,
    ) -> None:
        del params_dtype
        if input_size_per_partition % 16 or any(
            size % 16 for size in output_partition_sizes
        ):
            raise ValueError(
                "EXL3 linear dimensions must be 16-aligned: "
                f"input_local={input_size_per_partition} "
                f"outputs_local={output_partition_sizes}"
            )
        n_shards = len(output_partition_sizes)
        merged_qkv = (
            self.prefix.endswith(".qkv_proj")
            and n_shards == 3
            and os.environ.get("EXL3_MERGED_QKV", "0").lower()
            in ("1", "true", "yes")
        )
        in_tiles = input_size_per_partition // 16
        out_tiles = [size // 16 for size in output_partition_sizes]
        max_out_tiles = max(out_tiles)
        max_output = max(output_partition_sizes)
        k_words = self.bits * 16
        extra = {
            key: value
            for key, value in extra_weight_attrs.items()
            if key != "weight_loader"
        }
        if merged_qkv:
            # The checkpoint serializes KDA Q/K/V as one EXL3 matrix with one
            # shared input Hadamard/sign transform.  Store the TP-local output
            # thirds directly in one packed projection, avoiding both three
            # kernel launches and a duplicate merged tensor at load time.
            tensors = {
                "trellis": torch.empty(
                    1,
                    in_tiles,
                    sum(out_tiles),
                    k_words,
                    dtype=torch.int16,
                ),
                "suh": torch.empty(
                    1, input_size_per_partition, dtype=torch.float16
                ),
                "svh": torch.empty(
                    1, sum(output_partition_sizes), dtype=torch.float16
                ),
                self.quant_config.codebook: torch.empty(
                    1, 1, dtype=torch.int32
                ),
            }
        else:
            tensors = {
                "trellis": torch.empty(
                    n_shards,
                    in_tiles,
                    max_out_tiles,
                    k_words,
                    dtype=torch.int16,
                ),
                "suh": torch.empty(
                    n_shards, input_size_per_partition, dtype=torch.float16
                ),
                "svh": torch.empty(n_shards, max_output, dtype=torch.float16),
                self.quant_config.codebook: torch.empty(
                    n_shards, 1, dtype=torch.int32
                ),
            }
        for name, tensor in tensors.items():
            param = Parameter(tensor, requires_grad=False)
            layer.register_parameter(name, param)
            set_weight_attrs(param, extra)
            param.weight_loader = self._load_exl3_linear
            param._exl3_owner = layer
            param._exl3_suffix = name
        layer._exl3_linear_input_size = input_size_per_partition
        layer._exl3_linear_output_sizes = tuple(output_partition_sizes)
        layer._exl3_linear_out_tiles = tuple(out_tiles)
        layer._exl3_linear_merged_qkv = merged_qkv
        layer._exl3_linear_row_parallel = input_size_per_partition < input_size
        layer._exl3_linear_col_parallel = (
            sum(output_partition_sizes) < output_size
        )
        layer._exl3_linear_codebook = self.quant_config.codebook
        layer._exl3_linear_bits = self.bits

    def _load_exl3_linear(
        self,
        param: Parameter,
        loaded_weight: torch.Tensor,
        shard_id: int | str | None = None,
    ) -> None:
        from vllm.distributed import (
            get_tensor_model_parallel_rank,
            get_tensor_model_parallel_world_size,
        )

        layer = param._exl3_owner
        suffix = str(param._exl3_suffix)
        output_sizes = layer._exl3_linear_output_sizes
        if (
            shard_id is None
            and len(output_sizes) == 3
            and self.prefix.endswith(".qkv_proj")
        ):
            # GLM KDA serializes a single EXL3 [Q | K | V] matrix, whereas
            # MergedColumnParallelLinear owns three local output shards. Split
            # the quantized output axis before TP sharding; the input scale and
            # scalar codebook marker are shared by all three projections.
            if suffix == "trellis":
                if loaded_weight.shape[1] % 3:
                    raise ValueError(
                        f"packed EXL3 QKV trellis has invalid shape "
                        f"{tuple(loaded_weight.shape)} for {self.prefix}"
                    )
                packed_shards = loaded_weight.chunk(3, dim=1)
            elif suffix == "svh":
                if loaded_weight.shape[0] % 3:
                    raise ValueError(
                        f"packed EXL3 QKV svh has invalid shape "
                        f"{tuple(loaded_weight.shape)} for {self.prefix}"
                    )
                packed_shards = loaded_weight.chunk(3, dim=0)
            elif suffix in ("suh", "mcg", "mul1"):
                packed_shards = (loaded_weight,) * 3
            else:
                raise ValueError(
                    f"unsupported packed EXL3 QKV suffix {suffix!r}"
                )
            for packed_shard_id, packed_shard in enumerate(packed_shards):
                self._load_exl3_linear(
                    param, packed_shard.contiguous(), packed_shard_id
                )
            return
        if shard_id is None:
            shard_idx = 0
        elif isinstance(shard_id, int):
            shard_idx = shard_id
        else:
            shard_idx = {"q": 0, "k": 1, "v": 2}.get(shard_id, -1)
        if not 0 <= shard_idx < len(output_sizes):
            raise ValueError(
                f"invalid EXL3 linear shard {shard_id!r} for {self.prefix}"
            )
        loaded = loaded_weight.detach().contiguous()
        tp_rank = get_tensor_model_parallel_rank()
        tp_size = get_tensor_model_parallel_world_size()
        if layer._exl3_linear_row_parallel:
            sharded = shard_exl3_row(loaded, suffix, tp_rank, tp_size)
        elif layer._exl3_linear_col_parallel:
            sharded = shard_exl3_col(loaded, suffix, tp_rank, tp_size)
        else:
            sharded = loaded

        if layer._exl3_linear_merged_qkv:
            tile_offset = sum(layer._exl3_linear_out_tiles[:shard_idx])
            output_offset = sum(layer._exl3_linear_output_sizes[:shard_idx])
            if suffix == "trellis":
                out_tiles = layer._exl3_linear_out_tiles[shard_idx]
                dest = param.data[
                    0, :, tile_offset : tile_offset + out_tiles, :
                ]
            elif suffix == "svh":
                output_size = layer._exl3_linear_output_sizes[shard_idx]
                dest = param.data[
                    0, output_offset : output_offset + output_size
                ]
            elif suffix == "suh":
                dest = param.data[0]
            else:
                dest = param.data[0, 0]
                sharded = sharded.reshape(())
        elif suffix == "trellis":
            out_tiles = layer._exl3_linear_out_tiles[shard_idx]
            dest = param.data[shard_idx, :, :out_tiles, :]
        elif suffix == "svh":
            output_size = layer._exl3_linear_output_sizes[shard_idx]
            dest = param.data[shard_idx, :output_size]
        elif suffix == "suh":
            dest = param.data[shard_idx]
        else:
            dest = param.data[shard_idx, 0]
            sharded = sharded.reshape(())
        if tuple(dest.shape) != tuple(sharded.shape):
            raise RuntimeError(
                f"EXL3 linear load mismatch {self.prefix}.{suffix} "
                f"shard={shard_id}: dest={tuple(dest.shape)} "
                f"loaded={tuple(sharded.shape)}"
            )
        dest.copy_(sharded)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        marker_name = layer._exl3_linear_codebook
        marker = getattr(layer, marker_name)
        expected_marker = (
            MCG_MARKER_SIGNED_INT32
            if marker_name == "mcg"
            else MUL1_MARKER_SIGNED_INT32
        )
        if not torch.all(marker == expected_marker):
            raise RuntimeError(
                f"EXL3 {marker_name} marker mismatch in {self.prefix}: "
                f"expected {expected_marker}"
            )
        inners = []
        if layer._exl3_linear_merged_qkv:
            inner = make_linear_exl3(
                layer.trellis[0],
                layer.suh[0],
                layer.svh[0],
                marker[0],
                codebook=marker_name,
            )
            inners.append(inner)
        else:
            for shard_idx, output_size in enumerate(
                layer._exl3_linear_output_sizes
            ):
                out_tiles = layer._exl3_linear_out_tiles[shard_idx]
                inner = make_linear_exl3(
                    layer.trellis[shard_idx, :, :out_tiles, :],
                    layer.suh[shard_idx],
                    layer.svh[shard_idx, :output_size],
                    marker[shard_idx],
                    codebook=marker_name,
                )
                inners.append(inner)
        layer._exl3_linear_inners = inners
        layer._exl3_linear_multi = None
        use_multilinear = os.environ.get(
            "EXL3_MULTILINEAR_GATE_UP", "0"
        ).lower() in ("1", "true", "yes")
        if (
            use_multilinear
            and self.prefix.endswith(".gate_up_proj")
            and len(inners) == 2
        ):
            gate, up = inners
            if (
                gate.in_features != up.in_features
                or gate.out_features != up.out_features
                or gate.K != up.K
                or gate.mcg != up.mcg
                or gate.mul1 != up.mul1
            ):
                raise RuntimeError(
                    f"EXL3 multilinear-incompatible gate/up pair: {self.prefix}"
                )
            device = gate.trellis.device
            layer._exl3_linear_multi = types.SimpleNamespace(
                ptrs_trellis=torch.tensor(
                    [gate.trellis.data_ptr(), up.trellis.data_ptr()],
                    dtype=torch.long,
                    device=device,
                ),
                ptrs_suh=torch.tensor(
                    [gate.suh.data_ptr(), up.suh.data_ptr()],
                    dtype=torch.long,
                    device=device,
                ),
                ptrs_svh=torch.tensor(
                    [gate.svh.data_ptr(), up.svh.data_ptr()],
                    dtype=torch.long,
                    device=device,
                ),
                in_features=gate.in_features,
                out_features=gate.out_features,
                k=gate.K,
                mcg=gate.mcg,
                mul1=gate.mul1,
            )
        logger.info(
            "EXL3 %s linear engaged: layer=%s bits=%d shards=%d in=%d out=%s "
            "multilinear=%s",
            marker_name,
            self.prefix,
            self.bits,
            len(inners),
            layer._exl3_linear_input_size,
            layer._exl3_linear_output_sizes,
            layer._exl3_linear_multi is not None,
        )

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        shape = x.shape
        if (
            _EXL3_BF16_IO
            and x.dtype == torch.bfloat16
            and bias is None
            and x.numel() == shape[-1]
            and getattr(layer, "_exl3_bf16_ok", True)
        ):
            out_bf16 = _exl3_linear_bf16(layer, x)
            if out_bf16 is not None:
                return out_bf16.reshape(*shape[:-1], -1)
        xh = x.reshape(-1, shape[-1]).contiguous().half()
        multi = getattr(layer, "_exl3_linear_multi", None)
        max_rows = int(os.environ.get("EXL3_MULTILINEAR_MAX_ROWS", "32"))
        if multi is not None and int(xh.shape[0]) <= max_rows:
            rows = int(xh.shape[0])
            a_had = torch.empty(
                (2, rows, multi.in_features),
                dtype=torch.float16,
                device=xh.device,
            )
            planes = torch.empty(
                (2, rows, multi.out_features),
                dtype=torch.float16,
                device=xh.device,
            )
            load_exllamav3_ext().exl3_mgemm(
                xh.view(1, rows, multi.in_features),
                multi.ptrs_trellis,
                planes,
                multi.ptrs_suh,
                a_had,
                multi.ptrs_svh,
                None,
                None,
                multi.k,
                -1,
                multi.mcg,
                multi.mul1,
                -1,
                -1,
                0,
                1,
                None,
                None,
            )
            output_2d = torch.cat((planes[0], planes[1]), dim=-1)
        else:
            outputs = [
                inner.forward(xh, {}, out_dtype=torch.float16)
                for inner in layer._exl3_linear_inners
            ]
            output_2d = outputs[0] if len(outputs) == 1 else torch.cat(
                outputs, dim=-1
            )
        output = output_2d.reshape(*shape[:-1], -1)
        output = output.to(dtype=x.dtype)
        if bias is not None:
            output = output + bias
        return output


class Exl3MoEMethod(FusedMoEMethodBase):
    """Packed trellis experts: create/load packed tensors, LinearEXL3 apply."""

    def __init__(
        self, moe, quant_config: Exl3Config, *, layer_name: str = ""
    ) -> None:
        super().__init__(moe)
        self.quant_config = quant_config
        self.bits = quant_config.moe_bits(layer_name)
        self.layer_name = layer_name
        self._logged = False

    def get_fused_moe_quant_config(self, layer: "RoutedExperts") -> FusedMoEQuantConfig | None:
        return None

    def create_weights(
        self,
        layer: "RoutedExperts",
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        del params_dtype
        if hidden_size % 16 or intermediate_size_per_partition % 16:
            raise ValueError(
                "EXL3 trellis tiles are 16-wide; "
                f"hidden={hidden_size} intermediate_local={intermediate_size_per_partition}"
            )
        k_words = self.bits * 16
        in_tiles = hidden_size // 16
        out_tiles = intermediate_size_per_partition // 16

        extra = {k: v for k, v in extra_weight_attrs.items() if k != "weight_loader"}

        # Experts pinned permanently in VRAM (EXL3_PINNED_FRAC) never get host rows.
        pinned = _exl3_plan_pinned_experts(self.layer_name, num_experts, layer)
        all_experts = num_experts
        if pinned:
            pin_set = set(pinned)
            layer._exl3_pinned_local = list(pinned)
            layer._exl3_pin_row = {local: row for row, local in enumerate(pinned)}
            layer._exl3_host_row = {
                local: row
                for row, local in enumerate(e for e in range(all_experts) if e not in pin_set)
            }
            num_experts = all_experts - len(pinned)
            pin_dev = torch.device("cuda", torch.cuda.current_device())
            n_pin = len(pinned)
            for name, shape, dtype in (
                ("w13_trellis", (n_pin, 2, in_tiles, out_tiles, k_words), torch.int16),
                ("w13_suh", (n_pin, 2, hidden_size), torch.float16),
                ("w13_svh", (n_pin, 2, intermediate_size_per_partition), torch.float16),
                ("w2_trellis", (n_pin, out_tiles, in_tiles, k_words), torch.int16),
                ("w2_suh", (n_pin, intermediate_size_per_partition), torch.float16),
                ("w2_svh", (n_pin, hidden_size), torch.float16),
            ):
                layer.register_buffer(
                    "_exl3_pin_" + name, torch.empty(shape, dtype=dtype, device=pin_dev), persistent=False
                )
        # w13_* : stacked [expert, {gate=0, up=1}, ...] so the stock
        # expert_params_mapping (experts.w13_ + suffix) hits these names.
        w13_shape = (num_experts, 2, in_tiles, out_tiles, k_words)
        w2_shape = (num_experts, out_tiles, in_tiles, k_words)
        direct_vmm = None if pinned else _allocate_exl3_direct_vmm_trellises(
            layer, self.layer_name, num_experts, w13_shape, w2_shape
        )
        if direct_vmm is None:
            w13_tensor = torch.empty(w13_shape, dtype=torch.int16)
            w2_tensor = torch.empty(w2_shape, dtype=torch.int16)
        else:
            w13_tensor, w2_tensor = direct_vmm
        w13_trellis = Parameter(w13_tensor, requires_grad=False)
        w13_suh = Parameter(
            torch.empty(num_experts, 2, hidden_size, dtype=torch.float16),
            requires_grad=False,
        )
        w13_svh = Parameter(
            torch.empty(
                num_experts, 2, intermediate_size_per_partition, dtype=torch.float16
            ),
            requires_grad=False,
        )
        w13_marker = Parameter(
            torch.empty(num_experts, 2, 1, dtype=torch.int32),
            requires_grad=False,
        )
        w2_trellis = Parameter(w2_tensor, requires_grad=False)
        w2_suh = Parameter(
            torch.empty(
                num_experts, intermediate_size_per_partition, dtype=torch.float16
            ),
            requires_grad=False,
        )
        w2_svh = Parameter(
            torch.empty(num_experts, hidden_size, dtype=torch.float16),
            requires_grad=False,
        )
        w2_marker = Parameter(
            torch.empty(num_experts, 1, dtype=torch.int32),
            requires_grad=False,
        )

        packed = {
            "w13_trellis": w13_trellis,
            "w13_suh": w13_suh,
            "w13_svh": w13_svh,
            f"w13_{self.quant_config.codebook}": w13_marker,
            "w2_trellis": w2_trellis,
            "w2_suh": w2_suh,
            "w2_svh": w2_svh,
            f"w2_{self.quant_config.codebook}": w2_marker,
        }
        for name, param in packed.items():
            layer.register_parameter(name, param)
            set_weight_attrs(param, extra)
            param.weight_loader = self._load_exl3
            param._exl3_owner = layer
            param._exl3_name = name
        if hasattr(layer, "w13_weight") or hasattr(layer, "w2_weight"):
            raise RuntimeError("EXL3 create_weights must not allocate dense expert weights")

        layer._exl3_hidden_size = hidden_size
        layer._exl3_intermediate_local = intermediate_size_per_partition
        layer._exl3_k_words = k_words
        layer._exl3_bits = self.bits
        layer._exl3_codebook = self.quant_config.codebook

    def _load_exl3(
        self,
        param: Parameter,
        loaded_weight: torch.Tensor,
        weight_name: str,
        shard_id: str = "w1",
        expert_id: int = 0,
        return_success: bool = False,
    ) -> bool | None:
        from vllm.distributed import (
            get_tensor_model_parallel_rank,
            get_tensor_model_parallel_world_size,
        )

        layer = param
        # param is the Parameter; expert_id is already physical. Map to local
        # via the owning module if present on the weight_loader closure... we
        # look up from param's __dict__ after register. RoutedExperts.weight_loader
        # maps global→local; glm5next calls *our* loader, so map here.
        owner = getattr(param, "_exl3_owner", None)
        if owner is not None:
            local_id = owner._map_global_expert_id_to_local_expert_id(expert_id)
            if local_id == -1:
                return False if return_success else None
            expert_id = local_id
        target = param.data
        pin_row = getattr(owner, "_exl3_pin_row", None) if owner is not None else None
        if pin_row is not None:
            if expert_id in pin_row:
                buf = getattr(owner, "_exl3_pin_" + str(getattr(param, "_exl3_name", "")), None)
                if buf is None:
                    # codebook markers are one constant per tensor; pinned rows keep none
                    return True if return_success else None
                target = buf
                expert_id = pin_row[expert_id]
            else:
                expert_id = owner._exl3_host_row[expert_id]

        if owner is not None and bool(getattr(owner, "use_ep", False)):
            tp_rank, tp_size = 0, 1
        else:
            tp_rank = get_tensor_model_parallel_rank()
            tp_size = get_tensor_model_parallel_world_size()
        suffix = _suffix_from_mapped_name(weight_name)
        loaded = loaded_weight.detach().contiguous()

        if shard_id in ("w1", "w3"):
            shard_idx = 0 if shard_id == "w1" else 1
            sharded = shard_exl3_col(loaded, suffix, tp_rank, tp_size)
            dest = target[expert_id, shard_idx]
        elif shard_id == "w2":
            sharded = shard_exl3_row(loaded, suffix, tp_rank, tp_size)
            dest = target[expert_id]
        else:
            raise ValueError(f"unknown EXL3 shard_id={shard_id}")

        if sharded.ndim == 0 and dest.numel() == 1:
            sharded = sharded.reshape(dest.shape)
        if tuple(dest.shape) != tuple(sharded.shape):
            raise RuntimeError(
                f"EXL3 load shape mismatch {weight_name} shard={shard_id} "
                f"expert={expert_id}: dest {tuple(dest.shape)} != "
                f"loaded {tuple(sharded.shape)}"
            )
        dest.copy_(sharded)
        return True if return_success else None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if not hasattr(layer, "w13_trellis"):
            return
        dynamic_lru = os.environ.get("EXL3_DYNAMIC_LRU", "0").lower() in (
            "1",
            "true",
            "yes",
        )
        if not dynamic_lru:
            maybe_init_exl3_mixed_vmm_hot_cache(layer, self.layer_name)
        # Bind owner for any late loads; stitch LinearEXL3 handles.
        marker_suffix = str(getattr(layer, "_exl3_codebook", "mcg"))
        for name in (
            "w13_trellis",
            "w13_suh",
            "w13_svh",
            f"w13_{marker_suffix}",
            "w2_trellis",
            "w2_suh",
            "w2_svh",
            f"w2_{marker_suffix}",
        ):
            getattr(layer, name)._exl3_owner = layer

        marker13 = getattr(layer, f"w13_{marker_suffix}").reshape(-1)
        marker2 = getattr(layer, f"w2_{marker_suffix}").reshape(-1)
        expected_marker = (
            MCG_MARKER_SIGNED_INT32
            if marker_suffix == "mcg"
            else MUL1_MARKER_SIGNED_INT32
        )
        if not torch.all(marker13 == expected_marker) or not torch.all(
            marker2 == expected_marker
        ):
            raise RuntimeError(
                f"EXL3 {marker_suffix} marker is not expected int32 "
                f"{expected_marker}; packed ABI mismatch"
            )

        n_exp = int(layer.w13_trellis.shape[0])
        layer._exl3_shared_w13_suh = bool(
            torch.equal(layer.w13_suh[:, 0], layer.w13_suh[:, 1])
        )
        _record_exl3_fat_resolution(layer)
        inners: list[dict[str, Any]] = []
        for e in range(n_exp):
            gate = make_linear_exl3(
                layer.w13_trellis[e, 0],
                layer.w13_suh[e, 0],
                layer.w13_svh[e, 0],
                marker13.view(n_exp, 2, 1)[e, 0],
                codebook=marker_suffix,
            )
            up = make_linear_exl3(
                layer.w13_trellis[e, 1],
                layer.w13_suh[e, 1],
                layer.w13_svh[e, 1],
                marker13.view(n_exp, 2, 1)[e, 1],
                codebook=marker_suffix,
            )
            down = make_linear_exl3(
                layer.w2_trellis[e],
                layer.w2_suh[e],
                layer.w2_svh[e],
                marker2.view(n_exp, 1)[e],
                codebook=marker_suffix,
            )
            inners.append({"gate": gate, "up": up, "down": down})
        layer._exl3_inners = inners
        fused_ok = False
        fused_err = None
        if fused_moe_enabled():
            try:
                import exllamav3_ext

                if hasattr(exllamav3_ext, "exl3_moe"):
                    build_exl3_fused_state(layer, inners)
                    fused_ok = True
                else:
                    fused_err = "exllamav3_ext.exl3_moe missing"
            except Exception as exc:
                fused_err = repr(exc)
                layer._exl3_ptrs = None
        if not self._logged:
            if fused_ok:
                logger.info(
                    "EXL3 %s trellis engaged for routed experts: bits=%s "
                    "experts_local=%s hidden=%s intermediate_local=%s "
                    "fused_moe=exl3_moe concurrency=%s "
                    "(no BF16 expert reconstruct at load)",
                    marker_suffix,
                    self.bits,
                    n_exp,
                    layer._exl3_hidden_size,
                    layer._exl3_intermediate_local,
                    getattr(layer, "_exl3_fused_concurrency", "?"),
                )
            else:
                logger.info(
                    "EXL3 %s trellis engaged for routed experts: bits=%s "
                    "experts_local=%s hidden=%s intermediate_local=%s "
                    "fused_moe=python_loop (%s) "
                    "(no BF16 expert reconstruct at load)",
                    marker_suffix,
                    self.bits,
                    n_exp,
                    layer._exl3_hidden_size,
                    layer._exl3_intermediate_local,
                    fused_err or "EXL3_FUSED_MOE=0",
                )
            self._logged = True
        if dynamic_lru:
            if not fused_ok:
                raise RuntimeError(
                    "EXL3 dynamic LRU could not initialize without fused MoE: "
                    f"{fused_err}"
                )
            source_names = (
                "w13_trellis",
                "w13_suh",
                "w13_svh",
                f"w13_{marker_suffix}",
                "w2_trellis",
                "w2_suh",
                "w2_svh",
                f"w2_{marker_suffix}",
            )
            source_is_fully_resident = all(
                not bool(
                    getattr(
                        getattr(layer, name), "_vllm_is_uva_offloaded", False
                    )
                )
                for name in source_names
            )
            if source_is_fully_resident:
                # A byte-limited UVA placement can deliberately leave complete
                # tail layers in HBM.  Their source trellises already form a
                # 100%-hit expert bank, so allocating a second LRU bank wastes
                # scarce VRAM and can prevent the model from loading at all.
                # The normal fused source path is both faster and exact.
                logger.info(
                    "Dynamic EXL3 LRU bypass: layer=%s source is fully HBM-resident",
                    self.layer_name,
                )
            else:
                maybe_init_exl3_dynamic_lru_cache(layer, self.layer_name)

    def apply(
        self,
        layer: "RoutedExperts",
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts: "SharedExperts | None",
        shared_experts_input: torch.Tensor | None,
    ) -> torch.Tensor:
        del shared_experts, shared_experts_input
        limit = getattr(self.moe, "swiglu_limit", None) or SWIGLU_LIMIT_DEFAULT
        return apply_exl3_experts(
            x, topk_ids, topk_weights, layer, limit=float(limit)
        )
