#include <cuda_fp16.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include "exl3_moe_gemv.cuh"
#include "exl3_gemv_int8_kernel.cuh"
#include "exl3_devctx.cuh"
#include "../util.h"
#include "../util.cuh"
#include <map>
#include <set>
#include <type_traits>
#include <cuda_bf16.h>

// Workspace layout (int32 units):
//   [counters: MG_MAX_MATS x MG_COUNTERS_PER_MAT]
//   [per-matrix per-slice {q, s1, s2, pad}: MG_MAX_MATS x SQ_KSPLIT_CAP x 4]
//   [partials: num_mats x ksplit x pstride]
#define MG_MAX_MATS 64
#define MG_COUNTERS_PER_MAT 64                  // size_n <= 64 * 256 = 16384
#define MG_COUNTERS (MG_MAX_MATS * MG_COUNTERS_PER_MAT)
#define MG_QSUMS (MG_MAX_MATS * SQ_KSPLIT_CAP * 4)
#define MG_PARTIALS_OFF (MG_COUNTERS + MG_QSUMS)
#define MG_WS_INTS ((int64_t) 8 * 1024 * 1024)  // 32 MB

int64_t exl3_moe_gemv_ws_ints() { return MG_WS_INTS; }

template <int bits, bool residual, bool a_bf16 = false>
__global__ __launch_bounds__(NUM_THREADS)
void exl3_moe_gemv_kernel
(
    const half* __restrict__ A,
    const int a_mod,
    const int a_div,
    const uint16_t* const* __restrict__ B_ptrs,
    const half* const* __restrict__ suh_ptrs,
    const half* const* __restrict__ svh_ptrs,
    float* __restrict__ C,
    const int num_mats,
    const int size_k,
    const int size_n,
    const int rows_per,
    int* __restrict__ ws
)
{
    extern __shared__ uint32_t shmem[];

    const int rows_total = size_k >> 4;
    const int nb256_total = size_n / 256;
    const int ksplit = CEIL_DIVIDE(rows_total, rows_per);
    const int units_per_mat = nb256_total * ksplit;
    const int units = units_per_mat * num_mats;
    const int slice_stride = rows_per * 16;
    const int pstride = size_n * (residual ? 2 : 1);

    int* counters = ws;
    float* qsums_all = (float*) (ws + MG_COUNTERS);
    int* partials_all = ws + MG_PARTIALS_OFF;

    // Shared layout identical to exl3_gemv_int8_sq_kernel at M = 1
    half* sh_ah = (half*) shmem;
    uint32_t* sh_as = shmem + rows_per * 8;
    uint32_t* sh_b = sh_as + slice_stride * (residual ? 2 : 1);
    float* sh_tmp = (float*) (sh_b + (gemv_int8_stage_smem(bits) ? 8 * GEMV_STAGE_D * 16 * bits : 0));
    __shared__ float sh_red[33];
    __shared__ int sh_last;

    const int t = threadIdx.x;
    int prev_key = -1;
    for (int unit = blockIdx.x; unit < units; unit += gridDim.x)
    {
        const int mat = unit / units_per_mat;
        const int u = unit - mat * units_per_mat;
        const int slice = u / nb256_total;
        const int nb256 = u - slice * nb256_total;
        const int kb0 = slice * rows_per;
        const int nrows = MIN(rows_per, rows_total - kb0);

        const half* Am = A + (size_t) ((mat % a_mod) / a_div) * size_k;
        const uint16_t* Bm = B_ptrs[mat];
        // silent skip (pointer value 1): another launch over the same tables owns this matrix
        if (Bm == reinterpret_cast<const uint16_t*>(1)) continue;
        // skip weight-0 matrices (prep nulls their trellis pointer): no staging, no partials, no counter
        // traffic (counters stay 0); the slice-0 unit zeroes this matrix's 256-wide output block
        if (Bm == nullptr)
        {
            if (slice == 0)
                for (int j = t; j < 256; j += NUM_THREADS) C[(size_t) mat * size_n + nb256 * 256 + j] = 0.0f;
            continue;
        }
        const half* suh = suh_ptrs[mat];
        const half* svh = svh_ptrs[mat];
        float* qsums = qsums_all + (size_t) mat * SQ_KSPLIT_CAP * 4;
        int* partials = partials_all + (size_t) mat * ksplit * pstride;

        const int key = mat * SQ_KSPLIT_CAP + slice;
        if (key != prev_key)
        {
            gemv_int8_stage_slice<1, residual, a_bf16>(Am, 1, size_k, suh, qsums + 4 * slice,
                                               sh_ah, sh_as, slice_stride, sh_red, kb0, nrows);
            prev_key = key;
        }

        int* pacc = partials + (size_t) slice * pstride;
        if constexpr (bits == 4)
            gemv_int8_unit_wide<1, residual, false>(Bm, pacc, pstride, sh_as, slice_stride, nb256, kb0, nrows, size_n);
        else if constexpr (gemv_int8_stage_smem(bits))
            gemv_int8_unit_smem<bits, 1, residual, false>(Bm, pacc, pstride, sh_as, slice_stride, sh_b, nb256, kb0, nrows, size_n);
        else
            gemv_int8_unit_narrow<bits, 1, residual, false>(Bm, pacc, pstride, sh_as, slice_stride, nb256, kb0, nrows, size_n);

        // Last of the ksplit contributors to (mat, nb256) runs the fixed-order epilogue
        __threadfence();
        __syncthreads();
        int* ctr = counters + mat * MG_COUNTERS_PER_MAT + nb256;
        if (t == 0) sh_last = (atomicAdd(ctr, 1) == ksplit - 1) ? 1 : 0;
        __syncthreads();
        if (sh_last)
        {
            gemv_int8_epilogue_group_sq<1, true, residual>(partials, qsums, pstride, ksplit, 1,
                                                           C + (size_t) mat * size_n, svh, sh_tmp, nb256, size_n);
            if (t == 0) *ctr = 0;
        }
    }
}

typedef void (*moe_gemv_fn)(const half*, int, int, const uint16_t* const*, const half* const*, const half* const*,
                            float*, int, int, int, int, int*);

template <int bits>
static void* select_moe_gemv(bool residual, bool a_bf16)
{
    if (a_bf16)
        return residual ? (void*) exl3_moe_gemv_kernel<bits, true, true> : (void*) exl3_moe_gemv_kernel<bits, false, true>;
    return residual ? (void*) exl3_moe_gemv_kernel<bits, true> : (void*) exl3_moe_gemv_kernel<bits, false>;
}

static void* select_moe_gemv_k(int K, bool residual, bool a_bf16 = false)
{
    switch (K)
    {
        case 2: return select_moe_gemv<2>(residual, a_bf16);
        case 3: return select_moe_gemv<3>(residual, a_bf16);
        case 4: return select_moe_gemv<4>(residual, a_bf16);
        case 5: return select_moe_gemv<5>(residual, a_bf16);
    }
    return nullptr;
}

static std::set<void*> moe_gemv_attr_set[MAX_DEVICES];
static std::map<std::pair<void*, size_t>, int> moe_gemv_occ[MAX_DEVICES];

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
)
{
    const at::cuda::OptionalCUDAGuard device_guard(A.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    const bool a_bf16 = A.scalar_type() == at::kBFloat16;
    if (!a_bf16) TORCH_CHECK_DTYPE(A, kHalf);
    TORCH_CHECK(A.is_contiguous(), "exl3_moe_gemv: A must be contiguous");
    TORCH_CHECK_DTYPE(C, kFloat);
    TORCH_CHECK_DTYPE(ws, kInt);
    TORCH_CHECK(B_ptrs.dtype() == at::kLong && suh_ptrs.dtype() == at::kLong && svh_ptrs.dtype() == at::kLong,
                "exl3_moe_gemv: pointer tables must be int64");
    TORCH_CHECK(B_ptrs.numel() >= num_mats && suh_ptrs.numel() >= num_mats && svh_ptrs.numel() >= num_mats,
                "exl3_moe_gemv: pointer tables shorter than num_mats");
    TORCH_CHECK(num_mats >= 1 && num_mats <= MG_MAX_MATS, "exl3_moe_gemv: num_mats out of range");
    TORCH_CHECK(size_k % 128 == 0 && size_n % 256 == 0, "exl3_moe_gemv: size_k % 128 / size_n % 256");
    TORCH_CHECK(size_n / 256 <= MG_COUNTERS_PER_MAT, "exl3_moe_gemv: size_n too large");
    TORCH_CHECK(C.numel() >= num_mats * size_n, "exl3_moe_gemv: C too small");
    TORCH_CHECK(ws.numel() >= MG_WS_INTS, "exl3_moe_gemv: workspace too small");

    int device;
    cudaGetDevice(&device);
    int num_sms = DevCtx::instance().get_num_sms(device);

    void* fn = select_moe_gemv_k((int) K, residual, a_bf16);
    TORCH_CHECK(fn, "exl3_moe_gemv: unsupported K");

    const int rows_max = gemv_int8_sq_rows_max(1, residual);
    auto smem_for = [&] (int rows_per) -> size_t
    {
        size_t stage = gemv_int8_stage_smem((int) K) ? (size_t) 8 * GEMV_STAGE_D * 16 * K * 4 : 0;
        return (size_t) rows_per * 16 * 2 + (size_t) rows_per * 16 * 4 * (residual ? 2 : 1)
               + stage + (size_t) 2 * 128 * 4;
    };
    if (moe_gemv_attr_set[device].find(fn) == moe_gemv_attr_set[device].end())
    {
        cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) smem_for(rows_max));
        cudaFuncSetAttribute(fn, cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxShared);
        moe_gemv_attr_set[device].insert(fn);
        cuda_check(cudaPeekAtLastError());
    }

    const int rows_total = (int) (size_k / 16);
    const int nb = (int) (size_n / 256);
    auto decomp = [&] (int grid_) -> int
    {
        int r = CEIL_DIVIDE(rows_total * nb * (int) num_mats, grid_);
        int rows_per = (MAX(r, MIN(2 * r, 32)) + 7) & ~7;
        rows_per = MAX(rows_per, SQ_MINROWS);
        rows_per = MIN(rows_per, rows_max);
        rows_per = MIN(rows_per, (rows_total + 7) & ~7);
        return rows_per;
    };
    int rows_per = decomp(6 * num_sms);
    size_t smem_guess = smem_for(rows_per);
    int maxb;
    auto key = std::make_pair(fn, smem_guess);
    auto it = moe_gemv_occ[device].find(key);
    if (it != moe_gemv_occ[device].end()) maxb = it->second;
    else
    {
        maxb = 1;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxb, fn, NUM_THREADS, smem_guess);
        moe_gemv_occ[device][key] = maxb;
    }
    int grid = MIN(MAX(maxb, 1) * num_sms, 1024);
    rows_per = decomp(grid);
    int ksplit = CEIL_DIVIDE(rows_total, rows_per);
    TORCH_CHECK(ksplit <= SQ_KSPLIT_CAP, "exl3_moe_gemv: ksplit cap");
    int64_t pstride = size_n * (residual ? 2 : 1);
    TORCH_CHECK(MG_PARTIALS_OFF + num_mats * ksplit * pstride <= MG_WS_INTS, "exl3_moe_gemv: workspace cap");
    int units = nb * ksplit * (int) num_mats;
    grid = MIN(grid, units);

    const half* A_ptr = (const half*) A.data_ptr();
    TORCH_CHECK(a_mod >= 1 && a_div >= 1, "exl3_moe_gemv: a_mod/a_div");
    int a_mod_i = (int) a_mod, a_div_i = (int) a_div;
    TORCH_CHECK(A.numel() >= ((num_mats - 1) % a_mod / a_div + 1) * size_k, "exl3_moe_gemv: A too small");
    const uint16_t* const* B_p = (const uint16_t* const*) B_ptrs.data_ptr();
    const half* const* suh_p = (const half* const*) suh_ptrs.data_ptr();
    const half* const* svh_p = (const half* const*) svh_ptrs.data_ptr();
    float* C_ptr = (float*) C.data_ptr();
    int nm = (int) num_mats, sk = (int) size_k, sn = (int) size_n;
    int* ws_ptr = (int*) ws.data_ptr();
    void* args[] = { (void*) &A_ptr, (void*) &a_mod_i, (void*) &a_div_i, (void*) &B_p, (void*) &suh_p, (void*) &svh_p,
                     (void*) &C_ptr, (void*) &nm, (void*) &sk, (void*) &sn, (void*) &rows_per, (void*) &ws_ptr };
    cuda_check(cudaLaunchKernel(fn, dim3(grid), dim3(NUM_THREADS), args, smem_for(rows_per), stream));
}

// ------------------------------------------------------------------------------------------------

__global__ void exl3_moe_swiglu_kernel
(
    const float* __restrict__ gu,
    half* __restrict__ act,
    const int num_experts,
    const int n,
    const float limit
)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_experts * n) return;
    int e = idx / n;
    int j = idx - e * n;
    float g = gu[(size_t) e * n + j];
    float u = gu[(size_t) (num_experts + e) * n + j];
    float s = g / (1.0f + __expf(-g));
    if (limit != 0.0f)
    {
        s = fminf(s, limit);
        u = fminf(fmaxf(u, -limit), limit);
    }
    act[(size_t) e * n + j] = __float2half_rn(s * u);
}

void exl3_moe_swiglu(const at::Tensor& gu, at::Tensor& act, int64_t num_experts, double limit)
{
    const at::cuda::OptionalCUDAGuard device_guard(gu.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    TORCH_CHECK_DTYPE(gu, kFloat);
    TORCH_CHECK_DTYPE(act, kHalf);
    int n = (int) (act.numel() / num_experts);
    TORCH_CHECK(gu.numel() >= 2 * num_experts * n, "exl3_moe_swiglu: gu too small");
    int total = (int) (num_experts * n);
    exl3_moe_swiglu_kernel<<<CEIL_DIVIDE(total, 256), 256, 0, stream>>>(
        (const float*) gu.data_ptr(), (half*) act.data_ptr(), (int) num_experts, n, (float) limit);
    cuda_check(cudaPeekAtLastError());
}

// ------------------------------------------------------------------------------------------------

template <typename OutT, typename WT>
__global__ void exl3_moe_combine_kernel
(
    const float* __restrict__ y,
    const WT* __restrict__ w,
    OutT* __restrict__ out,
    const int num_experts,
    const int h,
    const int num_tokens,
    const bool accumulate
)
{
    int jj = blockIdx.x * blockDim.x + threadIdx.x;
    if (jj >= h * num_tokens) return;
    const int t = jj / h;
    const int j = jj - t * h;
    const WT* wt = w + (size_t) t * num_experts;
    const float* yt = y + (size_t) t * num_experts * h;
    float acc = 0.0f;
    if constexpr (std::is_same<OutT, float>::value)
    {
        for (int e = 0; e < num_experts; ++e)
            acc = fmaf((float) wt[e], yt[(size_t) e * h + j], acc);
        out[jj] = accumulate ? out[jj] + acc : acc;
    }
    else
    {
        // int64 fixed point (2^-32), matching exllamav3 MOE_OUT_FIXED64: per-expert terms are
        // quantized exactly as the fused kernel's scatter, so partition invariance holds
        long long s = accumulate ? (long long) out[jj] : 0ll;
        for (int e = 0; e < num_experts; ++e)
            s += __float2ll_rn(((float) wt[e] * yt[(size_t) e * h + j]) * 4294967296.0f);
        out[jj] = (OutT) s;
    }
}

void exl3_moe_combine(const at::Tensor& y, const at::Tensor& w, at::Tensor& out, int64_t num_experts, bool accumulate)
{
    const at::cuda::OptionalCUDAGuard device_guard(y.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    TORCH_CHECK_DTYPE(y, kFloat);
    TORCH_CHECK_DTYPE(w, kFloat);
    int h = (int) out.size(-1);
    int nt = (int) (out.numel() / h);
    TORCH_CHECK(y.numel() >= (int64_t) nt * num_experts * h && w.numel() >= nt * num_experts, "exl3_moe_combine: y/w too small");
    dim3 grid(CEIL_DIVIDE(h * nt, 256));
    if (out.dtype() == at::kFloat)
        exl3_moe_combine_kernel<float, float><<<grid, 256, 0, stream>>>(
            (const float*) y.data_ptr(), (const float*) w.data_ptr(), (float*) out.data_ptr(), (int) num_experts, h, nt, accumulate);
    else if (out.dtype() == at::kLong)
        exl3_moe_combine_kernel<long long, float><<<grid, 256, 0, stream>>>(
            (const float*) y.data_ptr(), (const float*) w.data_ptr(), (long long*) out.data_ptr(), (int) num_experts, h, nt, accumulate);
    else
        TORCH_CHECK(false, "exl3_moe_combine: out must be float32 or int64");
    cuda_check(cudaPeekAtLastError());
}

// ------------------------------------------------------------------------------------------------
// LRU miss fill: copy every component row of every missed expert from (pinned, UVA-mapped) host
// masters into its VRAM slot in ONE launch on a few persistent blocks. A PCIe read is latency-bound
// (~1.5-2 us), so a handful of SMs with deep per-thread ILP (EXL3_GATHER_ILP 16-byte loads in
// flight per thread) saturate the link while leaving the remaining SMs to concurrent compute.
// Component c of miss i: src_base[c] + src_row * row_bytes[c] -> dst_base[c] + slot * row_bytes[c].
// Hits (src_row < 0 or slot < 0) are skipped. Row sizes must be multiples of 16 bytes.

#define EXL3_GATHER_COMPONENTS 6
#define EXL3_GATHER_ILP 8

struct Exl3GatherArgs
{
    const char* src[EXL3_GATHER_COMPONENTS];
    char* dst[EXL3_GATHER_COMPONENTS];
    long long row_bytes[EXL3_GATHER_COMPONENTS];
    int num_components;
};

__global__ void __launch_bounds__(1024) exl3_lru_gather_kernel
(
    const Exl3GatherArgs args,
    const long long* __restrict__ miss_rows,
    const long long* __restrict__ miss_slots,
    const int num_ids
)
{
    const long long stride = (long long) gridDim.x * blockDim.x;
    const long long tid = (long long) blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = 0; i < num_ids; ++i)
    {
        const long long sr = miss_rows[i];
        const long long ds = miss_slots[i];
        if (sr < 0 || ds < 0) continue;
        for (int c = 0; c < args.num_components; ++c)
        {
            const long long nchunks = args.row_bytes[c] >> 4;
            const uint4* s = (const uint4*) (args.src[c] + sr * args.row_bytes[c]);
            uint4* d = (uint4*) (args.dst[c] + ds * args.row_bytes[c]);
            long long j = tid;
            for (; j + (EXL3_GATHER_ILP - 1) * stride < nchunks; j += EXL3_GATHER_ILP * stride)
            {
                uint4 v[EXL3_GATHER_ILP];
                #pragma unroll
                for (int u = 0; u < EXL3_GATHER_ILP; ++u) v[u] = __ldcg(s + j + u * stride);
                #pragma unroll
                for (int u = 0; u < EXL3_GATHER_ILP; ++u) __stcg(d + j + u * stride, v[u]);
            }
            for (; j < nchunks; j += stride) __stcg(d + j, __ldcg(s + j));
        }
    }
}

void exl3_lru_gather
(
    const std::vector<at::Tensor>& src,
    const std::vector<at::Tensor>& dst,
    const at::Tensor& miss_rows,
    const at::Tensor& miss_slots,
    int64_t num_ids,
    int64_t blocks,
    int64_t threads
)
{
    const at::cuda::OptionalCUDAGuard device_guard(miss_rows.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    TORCH_CHECK(src.size() == dst.size() && src.size() >= 1 && src.size() <= EXL3_GATHER_COMPONENTS,
                "exl3_lru_gather: 1..6 matching src/dst components");
    TORCH_CHECK(miss_rows.dtype() == at::kLong && miss_slots.dtype() == at::kLong, "exl3_lru_gather: int64 ids");
    Exl3GatherArgs args = {};
    args.num_components = (int) src.size();
    for (size_t c = 0; c < src.size(); ++c)
    {
        int64_t rb = src[c][0].numel() * src[c].element_size();
        TORCH_CHECK(rb == dst[c][0].numel() * dst[c].element_size(), "exl3_lru_gather: row size mismatch");
        TORCH_CHECK(rb % 16 == 0, "exl3_lru_gather: row bytes must be a multiple of 16");
        TORCH_CHECK(((uintptr_t) src[c].data_ptr()) % 16 == 0 && ((uintptr_t) dst[c].data_ptr()) % 16 == 0,
                    "exl3_lru_gather: 16-byte alignment");
        args.src[c] = (const char*) src[c].data_ptr();
        args.dst[c] = (char*) dst[c].data_ptr();
        args.row_bytes[c] = rb;
    }
    exl3_lru_gather_kernel<<<(int) blocks, (int) threads, 0, stream>>>(
        args, (const long long*) miss_rows.data_ptr(), (const long long*) miss_slots.data_ptr(), (int) num_ids);
    cuda_check(cudaPeekAtLastError());
}

// ------------------------------------------------------------------------------------------------
// Cross-layer prefetch (pre-gating) support. Staging buffers hold up to K predicted experts of the
// NEXT MoE layer as raw bytes: component c of staged entry k lives at staging[c] + k * row_bytes[c].
// staged_rows[k] = local source row that entry k holds, or -1.
//
// exl3_lru_prefetch: for each predicted global id g = pred[k]: if cache_map[g] >= 0 (resident) or
// source_map[g] < 0 (not local), staged_rows[k] = -1; else copy source row source_map[g] of every
// component into staging entry k and set staged_rows[k] = that row.
//
// exl3_lru_gather_staged: the LRU miss fill, but a miss whose source row is staged is copied from
// VRAM staging instead of host memory. Cache contents and LRU decisions are unchanged, so outputs are
// bit-identical to the unstaged path; only the transport differs.

#define EXL3_PREFETCH_MAX_K 32

struct Exl3StageArgs
{
    const char* stage[EXL3_GATHER_COMPONENTS];
};

// Decide once which predicted experts to stage (single thread): racing reads of cache_map (the
// main stream may be updating the next layer's LRU concurrently) or pred must not let different
// copy blocks disagree about an entry, or a partially written entry could be consumed.
__global__ void exl3_lru_prefetch_decide_kernel
(
    const long long* __restrict__ pred,
    const long long* __restrict__ cache_map,
    const long long* __restrict__ source_map,
    const int num_global,
    long long* __restrict__ staged_rows,
    const int K,
    const int K_total
)
{
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    // Every slot is (re)written: entries >= K are invalidated so a stale row from an earlier,
    // larger prefetch (possibly of another layer) can never match a miss.
    for (int k = 0; k < K_total; ++k)
    {
        const long long g = k < K ? pred[k] : -1;
        long long row = -1;
        if (g >= 0 && g < num_global && cache_map[g] < 0) row = source_map[g];
        for (int j = 0; j < k; ++j) if (staged_rows[j] == row) row = -1;   // de-duplicate
        staged_rows[k] = row;
    }
}

__global__ void __launch_bounds__(1024) exl3_lru_prefetch_kernel
(
    const Exl3GatherArgs args,       // src = host masters, dst = staging bases
    const long long* __restrict__ staged_rows,
    const int K
)
{
    const long long stride = (long long) gridDim.x * blockDim.x;
    const long long tid = (long long) blockIdx.x * blockDim.x + threadIdx.x;
    for (int k = 0; k < K; ++k)
    {
        const long long row = staged_rows[k];
        if (row < 0) continue;
        for (int c = 0; c < args.num_components; ++c)
        {
            const long long nchunks = args.row_bytes[c] >> 4;
            const uint4* s = (const uint4*) (args.src[c] + row * args.row_bytes[c]);
            uint4* d = (uint4*) (args.dst[c] + (long long) k * args.row_bytes[c]);
            long long j = tid;
            for (; j + (EXL3_GATHER_ILP - 1) * stride < nchunks; j += EXL3_GATHER_ILP * stride)
            {
                uint4 v[EXL3_GATHER_ILP];
                #pragma unroll
                for (int u = 0; u < EXL3_GATHER_ILP; ++u) v[u] = __ldcg(s + j + u * stride);
                #pragma unroll
                for (int u = 0; u < EXL3_GATHER_ILP; ++u) __stcg(d + j + u * stride, v[u]);
            }
            for (; j < nchunks; j += stride) __stcg(d + j, __ldcg(s + j));
        }
    }
}

__global__ void __launch_bounds__(1024) exl3_lru_gather_staged_kernel
(
    const Exl3GatherArgs args,
    const Exl3StageArgs stage,
    const long long* __restrict__ staged_rows,
    const int K,
    const long long* __restrict__ miss_rows,
    const long long* __restrict__ miss_slots,
    const int num_ids
)
{
    const long long stride = (long long) gridDim.x * blockDim.x;
    const long long tid = (long long) blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = 0; i < num_ids; ++i)
    {
        const long long sr = miss_rows[i];
        const long long ds = miss_slots[i];
        if (sr < 0 || ds < 0) continue;
        int kk = -1;
        for (int k = 0; k < K; ++k) if (staged_rows[k] == sr) { kk = k; break; }
        for (int c = 0; c < args.num_components; ++c)
        {
            const long long nchunks = args.row_bytes[c] >> 4;
            const uint4* s = kk >= 0
                ? (const uint4*) (stage.stage[c] + (long long) kk * args.row_bytes[c])
                : (const uint4*) (args.src[c] + sr * args.row_bytes[c]);
            uint4* d = (uint4*) (args.dst[c] + ds * args.row_bytes[c]);
            long long j = tid;
            for (; j + (EXL3_GATHER_ILP - 1) * stride < nchunks; j += EXL3_GATHER_ILP * stride)
            {
                uint4 v[EXL3_GATHER_ILP];
                #pragma unroll
                for (int u = 0; u < EXL3_GATHER_ILP; ++u) v[u] = __ldcg(s + j + u * stride);
                #pragma unroll
                for (int u = 0; u < EXL3_GATHER_ILP; ++u) __stcg(d + j + u * stride, v[u]);
            }
            for (; j < nchunks; j += stride) __stcg(d + j, __ldcg(s + j));
        }
    }
}

static Exl3GatherArgs make_gather_args(const std::vector<at::Tensor>& src, const std::vector<at::Tensor>& dst_rows_like,
                                       const std::vector<int64_t>* dst_ptrs)
{
    Exl3GatherArgs args = {};
    args.num_components = (int) src.size();
    for (size_t c = 0; c < src.size(); ++c)
    {
        int64_t rb = src[c][0].numel() * src[c].element_size();
        TORCH_CHECK(rb % 16 == 0, "exl3 gather: row bytes must be a multiple of 16");
        TORCH_CHECK(((uintptr_t) src[c].data_ptr()) % 16 == 0, "exl3 gather: 16-byte alignment (src)");
        args.src[c] = (const char*) src[c].data_ptr();
        if (dst_ptrs) args.dst[c] = (char*) (*dst_ptrs)[c];
        else
        {
            TORCH_CHECK(rb == dst_rows_like[c][0].numel() * dst_rows_like[c].element_size(), "exl3 gather: row size mismatch");
            TORCH_CHECK(((uintptr_t) dst_rows_like[c].data_ptr()) % 16 == 0, "exl3 gather: 16-byte alignment (dst)");
            args.dst[c] = (char*) dst_rows_like[c].data_ptr();
        }
        args.row_bytes[c] = rb;
    }
    return args;
}

// staging[c]: flat uint8 CUDA tensors, each >= K * row_bytes[c], 16-byte aligned
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
)
{
    const at::cuda::OptionalCUDAGuard device_guard(pred.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    TORCH_CHECK(src.size() == staging.size() && src.size() >= 1 && src.size() <= EXL3_GATHER_COMPONENTS, "exl3_lru_prefetch: components");
    TORCH_CHECK(K >= 1 && K <= staged_rows.numel() && staged_rows.numel() <= EXL3_PREFETCH_MAX_K, "exl3_lru_prefetch: K");
    TORCH_CHECK(pred.numel() >= K, "exl3_lru_prefetch: pred shorter than K");
    TORCH_CHECK(pred.dtype() == at::kLong && cache_map.dtype() == at::kLong && source_map.dtype() == at::kLong
                && staged_rows.dtype() == at::kLong, "exl3_lru_prefetch: int64 maps");
    std::vector<int64_t> dptrs;
    for (size_t c = 0; c < src.size(); ++c)
    {
        int64_t rb = src[c][0].numel() * src[c].element_size();
        TORCH_CHECK(staging[c].numel() * staging[c].element_size() >= staged_rows.numel() * rb, "exl3_lru_prefetch: staging too small");
        TORCH_CHECK(((uintptr_t) staging[c].data_ptr()) % 16 == 0, "exl3_lru_prefetch: staging alignment");
        dptrs.push_back((int64_t) staging[c].data_ptr());
    }
    Exl3GatherArgs args = make_gather_args(src, src, &dptrs);
    exl3_lru_prefetch_decide_kernel<<<1, 32, 0, stream>>>(
        (const long long*) pred.data_ptr(), (const long long*) cache_map.data_ptr(),
        (const long long*) source_map.data_ptr(), (int) cache_map.numel(),
        (long long*) staged_rows.data_ptr(), (int) K, (int) staged_rows.numel());
    exl3_lru_prefetch_kernel<<<(int) blocks, (int) threads, 0, stream>>>(
        args, (const long long*) staged_rows.data_ptr(), (int) staged_rows.numel());
    cuda_check(cudaPeekAtLastError());
}

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
)
{
    const at::cuda::OptionalCUDAGuard device_guard(miss_rows.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    TORCH_CHECK(src.size() == dst.size() && src.size() == staging.size() && src.size() <= EXL3_GATHER_COMPONENTS, "exl3_lru_gather_staged: components");
    TORCH_CHECK(miss_rows.dtype() == at::kLong && miss_slots.dtype() == at::kLong && staged_rows.dtype() == at::kLong, "int64 ids");
    Exl3GatherArgs args = make_gather_args(src, dst, nullptr);
    Exl3StageArgs st = {};
    for (size_t c = 0; c < staging.size(); ++c)
    {
        TORCH_CHECK(staging[c].numel() * staging[c].element_size() >= K * args.row_bytes[c], "staging too small");
        st.stage[c] = (const char*) staging[c].data_ptr();
    }
    exl3_lru_gather_staged_kernel<<<(int) blocks, (int) threads, 0, stream>>>(
        args, st, (const long long*) staged_rows.data_ptr(), (int) K,
        (const long long*) miss_rows.data_ptr(), (const long long*) miss_slots.data_ptr(), (int) num_ids);
    cuda_check(cudaPeekAtLastError());
}

// ------------------------------------------------------------------------------------------------
// Decode MoE glue in one launch: map top-k global ids through expert_map (or identity) to local
// experts, gather the nine pointer tables and mask the routing weights (non-local -> weight 0,
// pointers of local expert 0). Outputs (int64): gu_tab[3][2K] = {trellis, suh, svh} for gate then
// up; d_tab[3][K]; w (fp32 [K]).

__global__ void exl3_moe_gemv_prep_kernel
(
    const long long* __restrict__ ids,
    const float* __restrict__ weights_f32,
    const __nv_bfloat16* __restrict__ weights_bf16,
    const long long* __restrict__ expert_map,
    const int n_global_map,
    const int n_local,
    const long long* __restrict__ gt, const long long* __restrict__ gs, const long long* __restrict__ gv,
    const long long* __restrict__ ut, const long long* __restrict__ us, const long long* __restrict__ uv,
    const long long* __restrict__ dt, const long long* __restrict__ ds, const long long* __restrict__ dv,
    long long* __restrict__ gu_tab,
    long long* __restrict__ d_tab,
    float* __restrict__ w_out,
    const int K
)
{
    int k = threadIdx.x;
    if (k >= K) return;
    long long g = ids[k];
    long long l = g;
    bool valid = g >= 0;
    if (expert_map)
    {
        valid = valid && g < n_global_map;
        l = valid ? expert_map[g] : -1;
    }
    valid = valid && l >= 0 && l < n_local;
    if (!valid) l = 0;
    float w = weights_f32 ? weights_f32[k] : __bfloat162float(weights_bf16[k]);
    w_out[k] = valid ? w : 0.0f;
    // weight exactly 0 (invalid id, or a hybrid-decode CPU-side expert): null trellis -> GEMV skips it
    const bool skip = !(valid && w != 0.0f);
    gu_tab[0 * 2 * K + k] = skip ? 0 : gt[l];  gu_tab[0 * 2 * K + K + k] = skip ? 0 : ut[l];
    gu_tab[1 * 2 * K + k] = gs[l];  gu_tab[1 * 2 * K + K + k] = us[l];
    gu_tab[2 * 2 * K + k] = gv[l];  gu_tab[2 * 2 * K + K + k] = uv[l];
    d_tab[0 * K + k] = skip ? 0 : dt[l];
    d_tab[1 * K + k] = ds[l];
    d_tab[2 * K + k] = dv[l];
}

void exl3_moe_gemv_prep
(
    const at::Tensor& ids,
    const at::Tensor& weights,
    const c10::optional<at::Tensor>& expert_map,
    int64_t n_local,
    const std::vector<at::Tensor>& tables,   // gate t/s/v, up t/s/v, down t/s/v
    at::Tensor& gu_tab,
    at::Tensor& d_tab,
    at::Tensor& w_out
)
{
    const at::cuda::OptionalCUDAGuard device_guard(ids.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    TORCH_CHECK(ids.dtype() == at::kLong, "prep: ids int64");
    TORCH_CHECK(tables.size() == 9, "prep: nine pointer tables");
    int K = (int) ids.numel();
    TORCH_CHECK(K <= 64, "prep: K <= 64");
    TORCH_CHECK(gu_tab.numel() >= 6 * K && d_tab.numel() >= 3 * K && w_out.numel() >= K, "prep: outputs");
    const float* wf = weights.dtype() == at::kFloat ? (const float*) weights.data_ptr() : nullptr;
    const __nv_bfloat16* wb = weights.dtype() == at::kBFloat16 ? (const __nv_bfloat16*) weights.data_ptr() : nullptr;
    TORCH_CHECK(wf || wb, "prep: weights fp32 or bf16");
    const long long* em = expert_map.has_value() ? (const long long*) expert_map->data_ptr() : nullptr;
    int n_map = expert_map.has_value() ? (int) expert_map->numel() : 0;
    auto P = [&](int i) { return (const long long*) tables[i].data_ptr(); };
    exl3_moe_gemv_prep_kernel<<<1, 64, 0, stream>>>(
        (const long long*) ids.data_ptr(), wf, wb, em, n_map, (int) n_local,
        P(0), P(1), P(2), P(3), P(4), P(5), P(6), P(7), P(8),
        (long long*) gu_tab.data_ptr(), (long long*) d_tab.data_ptr(), (float*) w_out.data_ptr(), K);
    cuda_check(cudaPeekAtLastError());
}

// bf16 output variant of the combine (overwrite), for the production path
__global__ void exl3_moe_combine_bf16_kernel
(
    const float* __restrict__ y, const float* __restrict__ w, __nv_bfloat16* __restrict__ out,
    const int num_experts, const int h, const int num_tokens
)
{
    int jj = blockIdx.x * blockDim.x + threadIdx.x;
    if (jj >= h * num_tokens) return;
    const int t = jj / h;
    const int j = jj - t * h;
    const float* wt = w + (size_t) t * num_experts;
    const float* yt = y + (size_t) t * num_experts * h;
    float acc = 0.0f;
    for (int e = 0; e < num_experts; ++e) acc = fmaf(wt[e], yt[(size_t) e * h + j], acc);
    out[jj] = __float2bfloat16_rn(acc);
}

void exl3_moe_combine_bf16(const at::Tensor& y, const at::Tensor& w, at::Tensor& out, int64_t num_experts)
{
    const at::cuda::OptionalCUDAGuard device_guard(y.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    TORCH_CHECK(out.dtype() == at::kBFloat16 && y.dtype() == at::kFloat && w.dtype() == at::kFloat, "combine_bf16 dtypes");
    int h = (int) out.size(-1);
    int nt = (int) (out.numel() / h);
    exl3_moe_combine_bf16_kernel<<<CEIL_DIVIDE(h * nt, 256), 256, 0, stream>>>(
        (const float*) y.data_ptr(), (const float*) w.data_ptr(), (__nv_bfloat16*) out.data_ptr(), (int) num_experts, h, nt);
    cuda_check(cudaPeekAtLastError());
}
