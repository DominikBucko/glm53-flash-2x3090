// glm_fast: decode-path fusions for GLM-5.3-Flash on SM86.
//
// mhc_post_pre_norm: one kernel for an mHC boundary at decode time (replaces mhc_post_tilelang +
// hc_prenorm_gemm_tilelang + mhc_pre_big_fuse_tilelang + rms_norm_kernel). Same rounding points as the
// split path: the new residual is rounded to bf16 before the pre block reads it, layer_input is rounded to
// bf16 before the RMSNorm, RMSNorm output = bf16((x * rsqrt(mean(x^2) + eps)) * w).
// All blocks do the post update + a slice of the prenorm GEMM; the last block to finish (atomic ticket)
// reduces the partials in a fixed order, runs the sigmoid/sinkhorn mixes, then layer_input + RMSNorm.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>

namespace {

constexpr int HC = 4;
constexpr int NMIX = 2 * HC + HC * HC;  // 24
constexpr int NRED = NMIX + 1;          // + sum of squares

__device__ __forceinline__ float warp_sum(float v)
{
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

__device__ __forceinline__ float sigmoidf_(float x) { return 1.0f / (1.0f + expf(-x)); }

template <int H, int NB>
__global__ void __launch_bounds__(HC * H / NB + 32) mhc_post_pre_norm_kernel(
    const __nv_bfloat16* __restrict__ x,       // [T, H]       layer output
    const __nv_bfloat16* __restrict__ res,     // [T, HC, H]   previous residual streams
    const float* __restrict__ post,            // [T, HC]      previous post mix
    const float* __restrict__ comb,            // [T, HC, HC]  previous comb mix (row = input stream)
    const float* __restrict__ fn,              // [NMIX, HC*H]
    const float* __restrict__ hc_scale,        // [3]
    const float* __restrict__ hc_base,         // [NMIX]
    const __nv_bfloat16* __restrict__ norm_w,  // [H]
    __nv_bfloat16* __restrict__ res_out,       // [T, HC, H]
    float* __restrict__ post_out,              // [T, HC]
    float* __restrict__ comb_out,              // [T, HC, HC]
    __nv_bfloat16* __restrict__ y,             // [T, H]       normalized layer input
    float* __restrict__ partials,              // [T, NB, NRED]
    int* __restrict__ counters,                // [T], zero between launches (self-resetting)
    float rms_eps, float pre_eps, float sk_eps, float post_mult, int sk_repeat, float norm_eps)
{
    constexpr int NT = HC * H / NB;  // threads per block, one residual element each
    constexpr int HS = H / NB;       // hidden positions per block
    constexpr int NW = NT / 32;
    constexpr int NQ = H / NT;       // finalize: hidden positions per thread
    static_assert(NT % 32 == 0 && H % NT == 0 && NW <= 32, "bad config");

    const int t = blockIdx.y;
    const int b = blockIdx.x;
    const int tid = threadIdx.x;
    const int lane = tid & 31, warp = tid >> 5;
    const bool worker = tid < NT;   // the extra warp (warp NW) only runs the sinkhorn in the last block
    const int j = worker ? tid / HS : 0;
    const int h = b * HS + tid % HS;

    __shared__ float s_red[NW + 1][NRED];
    __shared__ float s_tot[NRED];
    __shared__ float s_pre[HC];
    __shared__ float s_wsum[NW];
    __shared__ float s_rsn;
    __shared__ int s_last;

    // ---- 1. post: new_r[j, h] = post[j] * x[h] + sum_i comb[i, j] * res[i, h]
    const __nv_bfloat16* rt = res + (size_t)t * HC * H;
    if (worker) {
    float acc = post[t * HC + j] * __bfloat162float(x[(size_t)t * H + h]);
#pragma unroll
    for (int i = 0; i < HC; ++i)
        acc += comb[t * HC * HC + i * HC + j] * __bfloat162float(rt[(size_t)i * H + h]);
    const __nv_bfloat16 nb = __float2bfloat16_rn(acc);
    res_out[(size_t)t * HC * H + (size_t)j * H + h] = nb;
    const float xv = __bfloat162float(nb);

    // ---- 2. this block's slice of mixes = flat(new_r) @ fn^T and of sum(new_r^2)
    float v[NRED];
    const float* fcol = fn + (size_t)j * H + h;
#pragma unroll
    for (int k = 0; k < NMIX; ++k) v[k] = xv * __ldg(fcol + (size_t)k * HC * H);
    v[NMIX] = xv * xv;
#pragma unroll
    for (int k = 0; k < NRED; ++k) {
        const float s = warp_sum(v[k]);
        if (lane == 0) s_red[warp][k] = s;
    }
    }
    __syncthreads();
    if (tid < NRED) {
        float s = 0.0f;
#pragma unroll
        for (int w = 0; w < NW; ++w) s += s_red[w][tid];
        partials[((size_t)t * NB + b) * NRED + tid] = s;
    }
    __threadfence();
    __syncthreads();
    if (tid == 0) s_last = (atomicAdd(&counters[t], 1) == NB - 1);
    __syncthreads();
    if (!s_last) return;
    __threadfence();

    // ---- 3. last block: reduce partials (fixed order: NW groups of NB/NW blocks, then the groups in order)
    static_assert(NB % NW == 0, "NB must be a multiple of the warp count");
    {
        constexpr int PER = NB / NW;
        if (warp < NW && lane < NRED) {
            float s = 0.0f;
#pragma unroll
            for (int q = 0; q < PER; ++q) s += __ldcg(&partials[((size_t)t * NB + warp * PER + q) * NRED + lane]);
            s_red[warp][lane] = s;
        }
    }
    __syncthreads();
    if (tid < NRED) {
        float s = 0.0f;
#pragma unroll
        for (int w = 0; w < NW; ++w) s += s_red[w][tid];
        s_tot[tid] = s;
    }
    __syncthreads();
    const float rms = rsqrtf(s_tot[NMIX] / (float)(HC * H) + rms_eps);
    if (warp == 1 && lane < HC) {
        s_pre[lane] = sigmoidf_(s_tot[lane] * rms * hc_scale[0] + hc_base[lane]) + pre_eps;
    } else if (warp == 1 && lane >= HC && lane < 2 * HC) {
        const int q = lane - HC;
        post_out[t * HC + q] = sigmoidf_(s_tot[HC + q] * rms * hc_scale[1] + hc_base[HC + q]) * post_mult;
    }
    __syncthreads();
    if (warp == NW) {
        // extra warp: sinkhorn, off the critical path (comb is only consumed by the next boundary)
        if (lane == 0) counters[t] = 0;
        // every lane runs the whole 4x4 sinkhorn in registers (16 independent divides per step, no shuffles)
        float cm[HC][HC];
#pragma unroll
        for (int r = 0; r < HC; ++r)
#pragma unroll
            for (int c = 0; c < HC; ++c)
                cm[r][c] = s_tot[2 * HC + r * HC + c] * rms * hc_scale[2] + hc_base[2 * HC + r * HC + c];
#pragma unroll
        for (int r = 0; r < HC; ++r) {
            float mx = cm[r][0];
#pragma unroll
            for (int c = 1; c < HC; ++c) mx = fmaxf(mx, cm[r][c]);
            float sm = 0.0f;
#pragma unroll
            for (int c = 0; c < HC; ++c) { cm[r][c] = expf(cm[r][c] - mx); sm += cm[r][c]; }
#pragma unroll
            for (int c = 0; c < HC; ++c) cm[r][c] = cm[r][c] / sm + sk_eps;
        }
        float cs[HC], rs[HC];
#pragma unroll
        for (int c = 0; c < HC; ++c) cs[c] = cm[0][c] + cm[1][c] + cm[2][c] + cm[3][c] + sk_eps;
#pragma unroll
        for (int r = 0; r < HC; ++r)
#pragma unroll
            for (int c = 0; c < HC; ++c) cm[r][c] = cm[r][c] / cs[c];
        for (int it = 1; it < sk_repeat; ++it) {
#pragma unroll
            for (int r = 0; r < HC; ++r) rs[r] = cm[r][0] + cm[r][1] + cm[r][2] + cm[r][3] + sk_eps;
#pragma unroll
            for (int r = 0; r < HC; ++r) {
                const float inv = __fdividef(1.0f, rs[r]);
#pragma unroll
                for (int c = 0; c < HC; ++c) cm[r][c] *= inv;
            }
#pragma unroll
            for (int c = 0; c < HC; ++c) cs[c] = __fdividef(1.0f, cm[0][c] + cm[1][c] + cm[2][c] + cm[3][c] + sk_eps);
#pragma unroll
            for (int r = 0; r < HC; ++r)
#pragma unroll
                for (int c = 0; c < HC; ++c) cm[r][c] *= cs[c];
        }
        if (lane < HC * HC) {
            float v = 0.0f;
#pragma unroll
            for (int r = 0; r < HC; ++r)
#pragma unroll
                for (int c = 0; c < HC; ++c)
                    if (lane == r * HC + c) v = cm[r][c];
            comb_out[t * HC * HC + lane] = v;
        }
    }

    // ---- 4. layer_input = bf16(sum_i pre[i] * new_r[i]); y = RMSNorm(layer_input) * w   (warps 0..NW-1)
    static_assert(NQ == 16, "phase 4 assumes 16 elements per thread");
    if (!worker) return;
    const int hb = tid * NQ;
    const __nv_bfloat16* ro = res_out + (size_t)t * HC * H + hb;
    uint4 wv[2];
    wv[0] = __ldg(reinterpret_cast<const uint4*>(norm_w + hb));
    wv[1] = __ldg(reinterpret_cast<const uint4*>(norm_w + hb + 8));
    uint4 rv[HC][2];
#pragma unroll
    for (int i = 0; i < HC; ++i) {
        rv[i][0] = __ldcg(reinterpret_cast<const uint4*>(ro + (size_t)i * H));
        rv[i][1] = __ldcg(reinterpret_cast<const uint4*>(ro + (size_t)i * H + 8));
    }
    float li[NQ];
    float sq = 0.0f;
#pragma unroll
    for (int q = 0; q < NQ; ++q) {
        float o = 0.0f;
#pragma unroll
        for (int i = 0; i < HC; ++i)
            o += s_pre[i] * __bfloat162float(reinterpret_cast<const __nv_bfloat16*>(&rv[i][q / 8])[q % 8]);
        li[q] = __bfloat162float(__float2bfloat16_rn(o));
        sq += li[q] * li[q];
    }
    sq = warp_sum(sq);
    if (lane == 0) s_wsum[warp] = sq;
    asm volatile("bar.sync 1, %0;" :: "r"(NT));   // named barrier over the NT worker threads only
    if (tid == 0) {
        float s = 0.0f;
#pragma unroll
        for (int w = 0; w < NW; ++w) s += s_wsum[w];
        s_rsn = rsqrtf(s / (float)H + norm_eps);
    }
    asm volatile("bar.sync 1, %0;" :: "r"(NT));
    const float rsn = s_rsn;
    uint4 yv[2];
#pragma unroll
    for (int q = 0; q < NQ; ++q) {
        const float w = __bfloat162float(reinterpret_cast<const __nv_bfloat16*>(&wv[q / 8])[q % 8]);
        reinterpret_cast<__nv_bfloat16*>(&yv[q / 8])[q % 8] = __float2bfloat16_rn(li[q] * rsn * w);
    }
    *reinterpret_cast<uint4*>(y + (size_t)t * H + hb) = yv[0];
    *reinterpret_cast<uint4*>(y + (size_t)t * H + hb + 8) = yv[1];
}


// ------------------------------------------------------------------------------------------------------------------
// topk_sorted: exact top-k of each row of fp32 logits [R, W] (non-finite entries never selected), written as the
// selected column ids in ascending order, -1 padded to k. Ties at the k-th value take the lowest ids.
// Radix select on the order-preserving uint32 key: 3 histogram passes (bits 31..20, 19..8, 7..0; the last block of
// each pass picks the digit), then a per-block count pass and an ordered compaction pass (block-local ballot scans,
// cross-block offsets from the counts). Replaces torch.topk (sorted radix select + sort) + mask + copy.

constexpr int TK_NB = 64;
constexpr int TK_BD = 512;
constexpr int TK_BINS = 4096;
constexpr int TK_WS_ROW = TK_BINS + 8 + 2 * TK_NB;  // hist | state[8] | cnt_gt[NB] | cnt_eq[NB]

__device__ __forceinline__ bool tk_key(float x, uint32_t& key)
{
    const uint32_t u = __float_as_uint(x);
    if ((u & 0x7f800000u) == 0x7f800000u) return false;  // inf / nan
    key = (u & 0x80000000u) ? ~u : (u | 0x80000000u);
    return true;
}

// exclusive scan of a per-thread int over the block (thread order); returns the exclusive prefix, sets *total
__device__ __forceinline__ int tk_block_scan(int v, int* s_warp, int* total)
{
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    constexpr int NW = TK_BD / 32;
    int incl = v;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
        const int n = __shfl_up_sync(0xffffffffu, incl, o);
        if (lane >= o) incl += n;
    }
    if (lane == 31) s_warp[warp] = incl;
    __syncthreads();
    if (warp == 0) {
        int w = lane < NW ? s_warp[lane] : 0;
        int wi = w;
#pragma unroll
        for (int o = 1; o < 32; o <<= 1) {
            const int n = __shfl_up_sync(0xffffffffu, wi, o);
            if (lane >= o) wi += n;
        }
        if (lane < NW) s_warp[lane] = wi - w;  // exclusive warp offsets
        if (lane == NW - 1) s_warp[NW] = wi;
    }
    __syncthreads();
    const int r = s_warp[warp] + incl - v;
    *total = s_warp[NW];
    __syncthreads();
    return r;
}

template <int PASS>
__global__ void __launch_bounds__(TK_BD) topk_hist_kernel(const float* __restrict__ logits, int W, int64_t stride,
                                                          int k, int* __restrict__ ws)
{
    constexpr int SHIFT = PASS == 0 ? 20 : (PASS == 1 ? 8 : 0);
    constexpr int NBINS = PASS == 2 ? 256 : 4096;
    constexpr int PER = NBINS / TK_BD > 0 ? NBINS / TK_BD : 1;  // bins per thread in the digit search
    constexpr int ACTIVE = NBINS / PER;
    const int r = blockIdx.y, b = blockIdx.x, tid = threadIdx.x;
    int* hist = ws + (size_t)r * TK_WS_ROW;
    int* st = hist + TK_BINS;  // 0 prefix, 1 remaining k, 2 mode (1 = select all finite), 3 ticket
    __shared__ int sh[NBINS];
    __shared__ int s_warp[TK_BD / 32 + 1];
    __shared__ int s_last;
    const int mode = PASS == 0 ? 0 : st[2];
    const uint32_t prefix = PASS == 0 ? 0u : (uint32_t)st[0];
    const uint32_t hmask = PASS == 0 ? 0u : (PASS == 1 ? 0xfff00000u : 0xffffff00u);

    for (int i = tid; i < NBINS; i += TK_BD) sh[i] = 0;
    __syncthreads();
    if (mode == 0) {
        const int slice = (W + TK_NB - 1) / TK_NB;
        const int s0 = b * slice, s1 = min(W, s0 + slice);
        const float* row = logits + (size_t)r * stride;
        for (int i = s0 + tid; i < s1; i += TK_BD) {
            uint32_t key;
            if (!tk_key(row[i], key)) continue;
            if (PASS > 0 && (key & hmask) != prefix) continue;
            atomicAdd(&sh[(key >> SHIFT) & (NBINS - 1)], 1);
        }
        __syncthreads();
        for (int i = tid; i < NBINS; i += TK_BD)
            if (sh[i]) atomicAdd(&hist[i], sh[i]);
    }
    __threadfence();
    __syncthreads();
    if (tid == 0) s_last = (atomicAdd(&st[3], 1) == TK_NB - 1);
    __syncthreads();
    if (!s_last) return;
    __threadfence();
    if (tid == 0) st[3] = 0;
    if (mode != 0) return;

    // last block: digit search from the top bin down
    for (int i = tid; i < NBINS; i += TK_BD) {
        sh[i] = __ldcg(&hist[i]);
        hist[i] = 0;
    }
    __syncthreads();
    const int kk = PASS == 0 ? k : st[1];
    int local = 0;
    const int top = NBINS - 1 - tid * PER;  // this thread's bins: top, top-1, ..., top-PER+1
    if (tid < ACTIVE) {
#pragma unroll
        for (int q = 0; q < PER; ++q) local += sh[top - q];
    }
    int total;
    const int excl = tk_block_scan(local, s_warp, &total);
    if (PASS == 0 && total <= k) {
        if (tid == 0) { st[2] = 1; st[1] = total; }
        return;
    }
    if (tid == 0 && PASS == 0) st[2] = 0;
    if (tid < ACTIVE && excl < kk && kk <= excl + local) {
        int cum = excl;
        for (int q = 0; q < PER; ++q) {
            const int c = sh[top - q];
            if (kk <= cum + c) {
                st[0] = (int)(prefix | ((uint32_t)(top - q) << SHIFT));
                st[1] = kk - cum;
                break;
            }
            cum += c;
        }
    }
}

__global__ void __launch_bounds__(TK_BD) topk_count_kernel(const float* __restrict__ logits, int W, int64_t stride,
                                                           int* __restrict__ ws)
{
    const int r = blockIdx.y, b = blockIdx.x, tid = threadIdx.x;
    int* st = ws + (size_t)r * TK_WS_ROW + TK_BINS;
    int* cnt = st + 8;
    __shared__ int s_warp[TK_BD / 32 + 1];
    const int mode = st[2];
    const uint32_t K = (uint32_t)st[0];
    const int slice = (W + TK_NB - 1) / TK_NB;
    const int s0 = b * slice, s1 = min(W, s0 + slice);
    const float* row = logits + (size_t)r * stride;
    int gt = 0, eq = 0;
    for (int i = s0 + tid; i < s1; i += TK_BD) {
        uint32_t key;
        if (!tk_key(row[i], key)) continue;
        if (mode == 1 || key > K) ++gt;
        else if (key == K) ++eq;
    }
    int tg, te;
    tk_block_scan(gt, s_warp, &tg);
    tk_block_scan(eq, s_warp, &te);
    if (tid == 0) { cnt[b] = tg; cnt[TK_NB + b] = te; }
}

__global__ void __launch_bounds__(TK_BD) topk_compact_kernel(const float* __restrict__ logits, int W, int64_t stride,
                                                             int k, const int* __restrict__ ws,
                                                             int* __restrict__ out, int64_t out_stride)
{
    const int r = blockIdx.y, b = blockIdx.x, tid = threadIdx.x;
    const int* st = ws + (size_t)r * TK_WS_ROW + TK_BINS;
    const int* cnt = st + 8;
    __shared__ int s_warp[TK_BD / 32 + 1];
    __shared__ int s_off[3];
    const int mode = st[2];
    const uint32_t K = (uint32_t)st[0];
    const int need = mode == 1 ? 0 : st[1];
    if (tid < 32) {
        int og = 0, oe = 0, ag = 0, ae = 0;
        for (int q = tid; q < TK_NB; q += 32) {
            const int g = cnt[q], e = cnt[TK_NB + q];
            ag += g; ae += e;
            if (q < b) { og += g; oe += e; }
        }
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            og += __shfl_xor_sync(0xffffffffu, og, o);
            oe += __shfl_xor_sync(0xffffffffu, oe, o);
            ag += __shfl_xor_sync(0xffffffffu, ag, o);
            ae += __shfl_xor_sync(0xffffffffu, ae, o);
        }
        if (tid == 0) {
            s_off[0] = og + min(oe, need);  // output base of this block
            s_off[1] = oe;                  // equal-to-K elements before this block
            s_off[2] = ag + min(ae, need);  // total selected in the row
        }
    }
    __syncthreads();
    int base = s_off[0], eq_before = s_off[1];
    const int total_sel = s_off[2];
    int* orow = out + (size_t)r * out_stride;
    const int slice = (W + TK_NB - 1) / TK_NB;
    const int s0 = b * slice, s1 = min(W, s0 + slice);
    const float* row = logits + (size_t)r * stride;
    for (int t0 = s0; t0 < s1; t0 += TK_BD) {
        const int i = t0 + tid;
        uint32_t key = 0;
        const bool valid = i < s1 && tk_key(row[i], key);
        const bool is_gt = valid && (mode == 1 || key > K);
        const bool is_eq = valid && mode == 0 && key == K;
        int eq_tot;
        const int eq_rank = eq_before + tk_block_scan(is_eq ? 1 : 0, s_warp, &eq_tot);
        const bool sel = is_gt || (is_eq && eq_rank < need);
        int sel_tot;
        const int pos = base + tk_block_scan(sel ? 1 : 0, s_warp, &sel_tot);
        if (sel) orow[pos] = i;
        base += sel_tot;
        eq_before += eq_tot;
    }
    if (b == 0)
        for (int q = total_sel + tid; q < k; q += TK_BD) orow[q] = -1;
}


// ------------------------------------------------------------------------------------------------------------------
// router_topk: fused MoE router for decode rows. logits[t, e] = bf16(x[t] . W[e]) (fp32 accumulate, bf16 rounding,
// i.e. F.linear in bf16 followed by .to(fp32)); the last block per row (ticket) then applies vLLM's
// single_group_topk semantics: score = 0.5 * tanh(0.5 * l) + 0.5, selection = score + bias (-0 -> 0), top-k by the
// 64-bit key (twiddled selection bits << 32 | 65535 - expert) i.e. ties -> lower expert id, descending order;
// weight = score * (scale / (sum_selected(score) + 1e-20)) when renormalize, else score * scale.

constexpr int RT_ROWS_PER_BLOCK = 4;   // two warps per expert row (each half of H)
constexpr int RT_THREADS = 64 * RT_ROWS_PER_BLOCK;
constexpr int RT_MAX_E = 512;

__device__ __forceinline__ uint64_t rt_key(float v, int idx)
{
    uint32_t b = __float_as_uint(v);
    b = (b & 0x80000000u) ? ~b : (b | 0x80000000u);   // cub TwiddleIn for float
    return ((uint64_t) b << 32) | (uint32_t) (0xFFFF & (65535 - idx));
}

template <typename IT>
__global__ void __launch_bounds__(RT_THREADS) router_topk_kernel(
    const __nv_bfloat16* __restrict__ x, const __nv_bfloat16* __restrict__ W, const float* __restrict__ bias,
    float* __restrict__ logits, int* __restrict__ counters, float* __restrict__ out_w, IT* __restrict__ out_ids,
    int H, int E, int topk, int renorm, float scale)
{
    const int t = blockIdx.y;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int row = warp >> 1, half = warp & 1;
    const int e = blockIdx.x * RT_ROWS_PER_BLOCK + row;
    __shared__ float s_part[RT_ROWS_PER_BLOCK][2];
    __shared__ int s_last;
    if (e < E)
    {
        const int n8 = H / 8, h8 = n8 / 2;   // uint4 per row, per half
        const uint4* xr = reinterpret_cast<const uint4*>(x + (size_t) t * H) + half * h8;
        const uint4* wr = reinterpret_cast<const uint4*>(W + (size_t) e * H) + half * h8;
        float acc = 0.0f;
        constexpr int U = 8;
        int i = lane;
        for (; i + (U - 1) * 32 < h8; i += U * 32)
        {
            uint4 wv[U], xv[U];
            #pragma unroll
            for (int u = 0; u < U; ++u) { wv[u] = __ldg(wr + i + u * 32); xv[u] = __ldg(xr + i + u * 32); }
            #pragma unroll
            for (int u = 0; u < U; ++u)
            {
                const __nv_bfloat162* xa = reinterpret_cast<const __nv_bfloat162*>(&xv[u]);
                const __nv_bfloat162* wa = reinterpret_cast<const __nv_bfloat162*>(&wv[u]);
                #pragma unroll
                for (int q = 0; q < 4; ++q)
                {
                    const float2 xf = __bfloat1622float2(xa[q]), wf = __bfloat1622float2(wa[q]);
                    acc = fmaf(xf.x, wf.x, acc);
                    acc = fmaf(xf.y, wf.y, acc);
                }
            }
        }
        for (; i < h8; i += 32)
        {
            const uint4 xv = __ldg(xr + i), wv = __ldg(wr + i);
            const __nv_bfloat162* xa = reinterpret_cast<const __nv_bfloat162*>(&xv);
            const __nv_bfloat162* wa = reinterpret_cast<const __nv_bfloat162*>(&wv);
            #pragma unroll
            for (int q = 0; q < 4; ++q)
            {
                const float2 xf = __bfloat1622float2(xa[q]), wf = __bfloat1622float2(wa[q]);
                acc = fmaf(xf.x, wf.x, acc);
                acc = fmaf(xf.y, wf.y, acc);
            }
        }
        acc = warp_sum(acc);
        if (lane == 0) s_part[row][half] = acc;
    }
    __syncthreads();
    if (threadIdx.x < RT_ROWS_PER_BLOCK)
    {
        const int er = blockIdx.x * RT_ROWS_PER_BLOCK + threadIdx.x;
        if (er < E)
            logits[(size_t) t * E + er] = __bfloat162float(__float2bfloat16_rn(s_part[threadIdx.x][0] + s_part[threadIdx.x][1]));
    }
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0) s_last = (atomicAdd(&counters[t], 1) == (int) gridDim.x - 1);
    __syncthreads();
    if (!s_last) return;
    __threadfence();
    if (warp != 0) return;
    if (lane == 0) counters[t] = 0;

    // top-k over E experts by warp 0: per lane its experts lane, lane+32, ...; k rounds of warp max
    constexpr int PER = RT_MAX_E / 32;
    uint64_t keys[PER];
    float unb[PER];
    #pragma unroll
    for (int q = 0; q < PER; ++q)
    {
        const int ex = lane + 32 * q;
        keys[q] = 0;
        unb[q] = 0.0f;
        if (ex < E)
        {
            const float l = __ldcg(&logits[(size_t) t * E + ex]);
            const float b = bias ? bias[ex] : 0.0f;
            float sel = -INFINITY;
            float u = 0.0f;
            if (isfinite(l) && isfinite(b))
            {
                u = 0.5f * tanhf(0.5f * l) + 0.5f;
                const float biased = u + b;
                sel = biased == 0.0f ? 0.0f : biased;
            }
            unb[q] = u;
            keys[q] = rt_key(sel, ex);
        }
    }
    float my_unb = 0.0f;
    int my_ex = -1;
    float my_sel = -INFINITY;
    for (int r = 0; r < topk; ++r)
    {
        uint64_t best = 0;
        int bq = -1;
        #pragma unroll
        for (int q = 0; q < PER; ++q) if (keys[q] > best) { best = keys[q]; bq = q; }
        uint64_t wbest = best;
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1)
        {
            const uint64_t ob = __shfl_xor_sync(0xffffffffu, wbest, o);
            wbest = ob > wbest ? ob : wbest;
        }
        const int ex = 65535 - (int) (wbest & 0xFFFF);
        const bool mine = (ex & 31) == lane && bq >= 0 && best == wbest;
        float u = 0.0f;
        #pragma unroll
        for (int q = 0; q < PER; ++q)
            if (mine && q == bq) { u = unb[q]; keys[q] = 0; }
        u = __shfl_sync(0xffffffffu, u, ex & 31);
        uint32_t hb = (uint32_t) (wbest >> 32);
        hb = (hb & 0x80000000u) ? (hb & 0x7fffffffu) : ~hb;   // TwiddleOut
        const float sv = __uint_as_float(hb);
        if (lane == r) { my_unb = u; my_ex = ex; my_sel = sv; }
    }
    const bool finite_sel = lane < topk && my_sel != -INFINITY;
    const float contrib = finite_sel ? my_unb : 0.0f;
    const unsigned finite_mask = __ballot_sync(0xffffffffu, finite_sel);
    const float sum = warp_sum(contrib);
    if (lane < topk)
    {
        float o = 0.0f;
        if (finite_mask == 0) { if (renorm) o = 1.0f / (float) topk; }
        else if (finite_sel)
        {
            float sc = scale;
            if (renorm) sc /= sum + 1e-20f;
            o = contrib * sc;
        }
        out_w[(size_t) t * topk + lane] = o;
        out_ids[(size_t) t * topk + lane] = (IT) my_ex;
    }
}


// ------------------------------------------------------------------------------------------------------------------
// moe_split_tables: split routed-MoE trellis pointer tables into the part computed while the GPU-subset gather is
// in flight (A: everything except freshly gathered experts) and the part computed after it (B: only those).
// Position i is "new" when the LRU admission assigned it a slot this step (miss_slots[i] >= 0). Pointer value 1 =
// silent skip in exl3_moe_gemv. gu tables hold gate (i) and up (P + i) matrices.
__global__ void moe_split_tables_kernel(const int64_t* __restrict__ gu_tr, const int64_t* __restrict__ d_tr,
                                        const int64_t* __restrict__ miss_slots, int64_t* __restrict__ gu_a,
                                        int64_t* __restrict__ gu_b, int64_t* __restrict__ d_a,
                                        int64_t* __restrict__ d_b, int P)
{
    const int i = threadIdx.x;
    if (i >= P) return;
    const bool fresh = miss_slots[i] >= 0;
    const int64_t g = gu_tr[i], u = gu_tr[P + i], d = d_tr[i];
    gu_a[i] = fresh ? 1 : g;  gu_a[P + i] = fresh ? 1 : u;  d_a[i] = fresh ? 1 : d;
    gu_b[i] = fresh ? g : 1;  gu_b[P + i] = fresh ? u : 1;  d_b[i] = fresh ? d : 1;
}

}  // namespace

std::vector<torch::Tensor> mhc_post_pre_norm(
    const torch::Tensor& x, const torch::Tensor& residual, const torch::Tensor& post, const torch::Tensor& comb,
    const torch::Tensor& fn, const torch::Tensor& hc_scale, const torch::Tensor& hc_base,
    const torch::Tensor& norm_w, torch::Tensor& counters, double rms_eps, double pre_eps, double sk_eps,
    double post_mult, int64_t sk_repeat, double norm_eps, int64_t nb)
{
    constexpr int H = 4096;
    TORCH_CHECK(residual.dim() == 3 && residual.size(1) == HC && residual.size(2) == H, "residual must be [T, 4, 4096]");
    const int64_t T = residual.size(0);
    TORCH_CHECK(x.numel() == T * H && x.scalar_type() == at::kBFloat16 && x.is_contiguous(), "x");
    TORCH_CHECK(residual.scalar_type() == at::kBFloat16 && residual.is_contiguous(), "residual");
    TORCH_CHECK(post.numel() == T * HC && post.scalar_type() == at::kFloat && post.is_contiguous(), "post");
    TORCH_CHECK(comb.numel() == T * HC * HC && comb.scalar_type() == at::kFloat && comb.is_contiguous(), "comb");
    TORCH_CHECK(fn.size(0) == NMIX && fn.size(1) == HC * H && fn.scalar_type() == at::kFloat && fn.is_contiguous(), "fn");
    TORCH_CHECK(hc_scale.numel() == 3 && hc_base.numel() == NMIX, "scale/base");
    TORCH_CHECK(norm_w.numel() == H && norm_w.scalar_type() == at::kBFloat16 && norm_w.is_contiguous(), "norm_w");
    TORCH_CHECK(counters.scalar_type() == at::kInt && counters.numel() >= T, "counters");
    const at::cuda::OptionalCUDAGuard guard(x.device());
    auto f32 = x.options().dtype(at::kFloat);
    auto res_out = torch::empty_like(residual);
    auto post_out = torch::empty({T, HC, 1}, f32);
    auto comb_out = torch::empty({T, HC, HC}, f32);
    auto y = torch::empty({T, H}, x.options());
    auto partials = torch::empty({T, nb, NRED}, f32);
    auto stream = at::cuda::getCurrentCUDAStream();
    auto launch = [&](auto nb_c) {
        constexpr int NB = decltype(nb_c)::value;
        mhc_post_pre_norm_kernel<H, NB><<<dim3(NB, T), HC * H / NB + 32, 0, stream>>>(
            (const __nv_bfloat16*)x.data_ptr(), (const __nv_bfloat16*)residual.data_ptr(), post.data_ptr<float>(),
            comb.data_ptr<float>(), fn.data_ptr<float>(), hc_scale.data_ptr<float>(), hc_base.data_ptr<float>(),
            (const __nv_bfloat16*)norm_w.data_ptr(), (__nv_bfloat16*)res_out.data_ptr(), post_out.data_ptr<float>(),
            comb_out.data_ptr<float>(), (__nv_bfloat16*)y.data_ptr(), partials.data_ptr<float>(),
            counters.data_ptr<int>(), (float)rms_eps, (float)pre_eps, (float)sk_eps, (float)post_mult,
            (int)sk_repeat, (float)norm_eps);
    };
    TORCH_CHECK(nb == 64, "nb must be 64 (tuned: 32 and 128 are slower)");
    launch(std::integral_constant<int, 64>{});
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {res_out, post_out, comb_out, y};
}

void topk_sorted(const torch::Tensor& logits, int64_t k, torch::Tensor& out, torch::Tensor& ws)
{
    TORCH_CHECK(logits.dim() == 2 && logits.scalar_type() == at::kFloat && logits.stride(1) == 1, "logits [R, W] f32");
    const int64_t R = logits.size(0), W = logits.size(1);
    TORCH_CHECK(out.dim() == 2 && out.size(0) >= R && out.size(1) >= k && out.scalar_type() == at::kInt &&
                out.stride(1) == 1, "out [R, >=k] int32");
    TORCH_CHECK(ws.scalar_type() == at::kInt && ws.numel() >= R * TK_WS_ROW, "workspace too small");
    TORCH_CHECK(W < (1ll << 31) && k > 0 && R <= 65535, "bad sizes");
    if (R == 0) return;
    const at::cuda::OptionalCUDAGuard guard(logits.device());
    auto stream = at::cuda::getCurrentCUDAStream();
    const dim3 grid(TK_NB, R);
    const float* lp = logits.data_ptr<float>();
    int* wp = ws.data_ptr<int>();
    topk_hist_kernel<0><<<grid, TK_BD, 0, stream>>>(lp, (int)W, logits.stride(0), (int)k, wp);
    topk_hist_kernel<1><<<grid, TK_BD, 0, stream>>>(lp, (int)W, logits.stride(0), (int)k, wp);
    topk_hist_kernel<2><<<grid, TK_BD, 0, stream>>>(lp, (int)W, logits.stride(0), (int)k, wp);
    topk_count_kernel<<<grid, TK_BD, 0, stream>>>(lp, (int)W, logits.stride(0), wp);
    topk_compact_kernel<<<grid, TK_BD, 0, stream>>>(lp, (int)W, logits.stride(0), (int)k, wp, out.data_ptr<int>(),
                                                    out.stride(0));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

int64_t topk_ws_ints_per_row() { return TK_WS_ROW; }

torch::Tensor router_topk(const torch::Tensor& x, const torch::Tensor& W, const c10::optional<torch::Tensor>& bias, int64_t topk,
                 bool renorm, double scale, torch::Tensor& out_w, torch::Tensor& out_ids, torch::Tensor& counters)
{
    TORCH_CHECK(x.dim() == 2 && x.scalar_type() == at::kBFloat16 && x.is_contiguous(), "x [T, H] bf16");
    TORCH_CHECK(W.dim() == 2 && W.scalar_type() == at::kBFloat16 && W.is_contiguous() && W.size(1) == x.size(1), "W [E, H] bf16");
    const int T = (int) x.size(0), H = (int) x.size(1), E = (int) W.size(0);
    TORCH_CHECK(H % 16 == 0 && E <= RT_MAX_E && topk >= 1 && topk <= 32 && topk <= E, "router_topk: sizes");
    TORCH_CHECK(out_w.scalar_type() == at::kFloat && out_w.numel() >= T * topk && out_w.is_contiguous(), "out_w");
    TORCH_CHECK(out_ids.numel() >= T * topk && out_ids.is_contiguous(), "out_ids");
    TORCH_CHECK(counters.scalar_type() == at::kInt && counters.numel() >= T, "counters");
    const float* bp = nullptr;
    if (bias.has_value())
    {
        TORCH_CHECK(bias->scalar_type() == at::kFloat && bias->numel() == E, "bias [E] fp32");
        bp = bias->data_ptr<float>();
    }
    const at::cuda::OptionalCUDAGuard guard(x.device());
    auto logits = torch::empty({T, E}, x.options().dtype(at::kFloat));
    auto stream = at::cuda::getCurrentCUDAStream();
    const dim3 grid((E + RT_ROWS_PER_BLOCK - 1) / RT_ROWS_PER_BLOCK, T);
    if (out_ids.scalar_type() == at::kLong)
        router_topk_kernel<int64_t><<<grid, RT_THREADS, 0, stream>>>(
            (const __nv_bfloat16*) x.data_ptr(), (const __nv_bfloat16*) W.data_ptr(), bp, logits.data_ptr<float>(),
            counters.data_ptr<int>(), out_w.data_ptr<float>(), out_ids.data_ptr<int64_t>(), H, E, (int) topk,
            renorm ? 1 : 0, (float) scale);
    else if (out_ids.scalar_type() == at::kInt)
        router_topk_kernel<int32_t><<<grid, RT_THREADS, 0, stream>>>(
            (const __nv_bfloat16*) x.data_ptr(), (const __nv_bfloat16*) W.data_ptr(), bp, logits.data_ptr<float>(),
            counters.data_ptr<int>(), out_w.data_ptr<float>(), out_ids.data_ptr<int32_t>(), H, E, (int) topk,
            renorm ? 1 : 0, (float) scale);
    else
        TORCH_CHECK(false, "out_ids must be int32 or int64");
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return logits;
}

void moe_split_tables(const torch::Tensor& gu_tr, const torch::Tensor& d_tr, const torch::Tensor& miss_slots,
                      torch::Tensor& gu_a, torch::Tensor& gu_b, torch::Tensor& d_a, torch::Tensor& d_b)
{
    const int P = (int) d_tr.numel();
    TORCH_CHECK(P >= 1 && P <= 1024 && gu_tr.numel() == 2 * P && miss_slots.numel() >= P, "moe_split_tables: sizes");
    TORCH_CHECK(gu_tr.scalar_type() == at::kLong && d_tr.scalar_type() == at::kLong && miss_slots.scalar_type() == at::kLong,
                "moe_split_tables: int64");
    const at::cuda::OptionalCUDAGuard guard(gu_tr.device());
    moe_split_tables_kernel<<<1, 32 * ((P + 31) / 32), 0, at::cuda::getCurrentCUDAStream()>>>(
        gu_tr.data_ptr<int64_t>(), d_tr.data_ptr<int64_t>(), miss_slots.data_ptr<int64_t>(), gu_a.data_ptr<int64_t>(),
        gu_b.data_ptr<int64_t>(), d_a.data_ptr<int64_t>(), d_b.data_ptr<int64_t>(), P);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
