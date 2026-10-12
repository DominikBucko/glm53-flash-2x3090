#pragma once

#include <ATen/Tensor.h>

// Batched m == 1 int8-activation GEMV over several independent EXL3 (mul1 codebook) matrices in
// ONE regular launch: the decode-time routed-MoE building block. Every matrix i has its own trellis
// B_ptrs[i], input sign vector suh_ptrs[i] and output scale svh_ptrs[i]; all share size_k/size_n.
// Input row for matrix i is A + ((i % a_mod) / a_div) * size_k. Output
// C[i, 0:size_n] (fp32). Work is split into (matrix x k-slice x 256-column) units over a single
// wave; per-(matrix, column-group) completion counters elect the last contributor, which runs the
// fixed-order (deterministic) epilogue of exl3_gemv_int8_sq_kernel. ws: int32 workspace from
// exl3_moe_gemv_ws_ints(); counters self-reset, so the buffer must start zeroed and must not be
// shared by concurrent launches.
void exl3_moe_gemv
(
    const at::Tensor& A,
    int64_t a_mod,
    int64_t a_div,
    const at::Tensor& B_ptrs,
    const at::Tensor& suh_ptrs,
    const at::Tensor& svh_ptrs,
    at::Tensor& C,
    int64_t num_mats,
    int64_t size_k,
    int64_t size_n,
    int64_t K,
    bool residual,
    at::Tensor& ws
);

int64_t exl3_moe_gemv_ws_ints();

// act[e, j] = min(silu(g[e, j]), limit) * clamp(u[e, j], -limit, limit) as fp16, where
// g = gu[e, :], u = gu[E + e, :] (fp32 GEMV outputs). limit == 0 disables clamping.
void exl3_moe_swiglu
(
    const at::Tensor& gu,
    at::Tensor& act,
    int64_t num_experts,
    double limit
);

// out[h] (+)= sum_e w[e] * y[e, h] in fixed expert order (deterministic). y fp32 [E, H], w fp32 [E]
// (or fp16), out fp32 [H]. accumulate=false overwrites.
void exl3_moe_combine
(
    const at::Tensor& y,
    const at::Tensor& w,
    at::Tensor& out,
    int64_t num_experts,
    bool accumulate
);

#include <vector>

// Few-SM persistent LRU miss fill (see exl3_moe_gemv.cu). src[c]/dst[c]: [rows, ...] tensors whose
// row c-bytes are copied for every i with miss_rows[i] >= 0 and miss_slots[i] >= 0.
void exl3_lru_gather
(
    const std::vector<at::Tensor>& src,
    const std::vector<at::Tensor>& dst,
    const at::Tensor& miss_rows,
    const at::Tensor& miss_slots,
    int64_t num_ids,
    int64_t blocks,
    int64_t threads
);

// Cross-layer prefetch into VRAM staging, and the staging-aware miss fill (see .cu)
void exl3_lru_prefetch
(
    const std::vector<at::Tensor>& src,
    const std::vector<at::Tensor>& staging,
    const at::Tensor& pred,
    const at::Tensor& cache_map,
    const at::Tensor& source_map,
    at::Tensor& staged_rows,
    int64_t K,
    int64_t blocks,
    int64_t threads
);

void exl3_lru_gather_staged
(
    const std::vector<at::Tensor>& src,
    const std::vector<at::Tensor>& dst,
    const std::vector<at::Tensor>& staging,
    const at::Tensor& staged_rows,
    int64_t K,
    const at::Tensor& miss_rows,
    const at::Tensor& miss_slots,
    int64_t num_ids,
    int64_t blocks,
    int64_t threads
);

#include <c10/util/Optional.h>

// Decode MoE glue: id mapping + pointer-table gather + weight masking in one launch
void exl3_moe_gemv_prep
(
    const at::Tensor& ids,
    const at::Tensor& weights,
    const c10::optional<at::Tensor>& expert_map,
    int64_t n_local,
    const std::vector<at::Tensor>& tables,
    at::Tensor& gu_tab,
    at::Tensor& d_tab,
    at::Tensor& w_out
);

void exl3_moe_combine_bf16(const at::Tensor& y, const at::Tensor& w, at::Tensor& out, int64_t num_experts);
