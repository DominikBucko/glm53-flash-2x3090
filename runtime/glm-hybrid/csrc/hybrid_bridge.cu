// Graph-safe GPU <-> CPU bridge for hybrid MoE decode (GLM-5.3 EXL3 experts).
//
// Per process (= per TP rank) one pinned, device-mapped region holds one slot per MoE layer.
// Decode, inside a CUDA graph:
//   hyb_submit   : splits each token's top-k into VRAM-resident experts (stay on the GPU, routing
//                  weight kept) and cold experts (GPU weight -> 0, id/weight/x written to the slot),
//                  then bumps the slot's device-side sequence counter and publishes it (system
//                  fence, release store) to the slot's req_seq in host memory.
//   GPU GEMV     : runs as before over the (masked) top-k.
//   hyb_wait_add : spins until the CPU worker publishes done_seq == the same sequence, then adds the
//                  slot's fp32 partial output into the MoE output (before the TP all-reduce).
// A CPU worker thread (one per region) polls req_seq of every slot and runs ExLlamaV3's native
// CPU MoE kernel (exl3_moe_cpu_forward_raw, moe_mul1.cpp) on the cold selections; resident
// selections arrive as -1 and are skipped by that kernel.
// Everything the GPU does is ordinary kernels on the current stream, so the sequence advances on
// every graph replay without host involvement.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include <atomic>
#include <climits>
#include <cstring>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>
#include <sched.h>
#include <immintrin.h>
#include <time.h>
#include <chrono>

#include "moe_mul1.h"

namespace {

constexpr int HYB_MAX_T = 4;      // decode rows handled per call (max_num_seqs * spec width)
constexpr int HYB_MAX_K = 16;     // top-k capacity per row

inline size_t a64(size_t v) { return (v + 63) & ~size_t(63); }

struct SlotLayout
{
    size_t req = 0, rows = 4, topk = 8, done = 64;
    size_t x, sel, w, out, size;
    explicit SlotLayout(int max_h)
    {
        x = 128;
        sel = a64(x + sizeof(__half) * HYB_MAX_T * max_h);
        w = a64(sel + sizeof(int32_t) * HYB_MAX_T * HYB_MAX_K);
        out = a64(w + sizeof(__half) * HYB_MAX_T * HYB_MAX_K);
        size = (a64(out + sizeof(float) * HYB_MAX_T * max_h) + 4095) & ~size_t(4095);
    }
};

constexpr size_t CTRL_SIZE = 4096;   // u32 abort @0 (GPU timeout), rest reserved

struct Region
{
    int num_slots = 0;
    int max_h = 0;
    SlotLayout lay{0};
    uint8_t* host = nullptr;
    uint8_t* dev = nullptr;
    size_t bytes = 0;
    std::vector<std::atomic<int64_t>> handles;
    std::thread worker;
    std::atomic<bool> quit{false};
    std::atomic<uint64_t> jobs{0};
    std::atomic<uint64_t> compute_ns{0}, gap_ns{0}, last_done_ns{0}, cold_experts{0};
    int log_every = 0;
    int threads = 1;
    std::vector<int> cpus;
    Region(int n, int h) : num_slots(n), max_h(h), lay(h), handles(n) { for (auto& a : handles) a = -1; }
    uint8_t* slot_host(int s) const { return host + CTRL_SIZE + size_t(s) * lay.size; }
    uint8_t* slot_dev(int s) const { return dev + CTRL_SIZE + size_t(s) * lay.size; }
};

std::mutex g_mu;
std::vector<std::unique_ptr<Region>> g_regions;

Region* get_region(int64_t id)
{
    TORCH_CHECK(id >= 0 && id < (int64_t) g_regions.size() && g_regions[id], "bad hybrid region id");
    return g_regions[id].get();
}

inline uint32_t load_acquire(const uint32_t* p) { return __atomic_load_n(p, __ATOMIC_ACQUIRE); }
inline void store_release(uint32_t* p, uint32_t v) { __atomic_store_n(p, v, __ATOMIC_RELEASE); }

void worker_loop(Region* r)
{
    if (!r->cpus.empty())
    {
        cpu_set_t set;
        CPU_ZERO(&set);
        for (int c : r->cpus) CPU_SET(c, &set);
        sched_setaffinity(0, sizeof(set), &set);   // pool workers spawned from here inherit/filter on it
    }
    std::vector<at::Half> px(size_t(HYB_MAX_T) * r->max_h), pw(HYB_MAX_T * HYB_MAX_K);
    std::vector<int32_t> psel(HYB_MAX_T * HYB_MAX_K);
    std::vector<float> pout(size_t(HYB_MAX_T) * r->max_h);
    std::vector<uint32_t> last(r->num_slots, 0);
    for (int s = 0; s < r->num_slots; ++s)
        last[s] = load_acquire(reinterpret_cast<uint32_t*>(r->slot_host(s) + r->lay.done));
    uint64_t idle = 0;
    while (!r->quit.load(std::memory_order_relaxed))
    {
        bool did = false;
        for (int s = 0; s < r->num_slots; ++s)
        {
            uint8_t* sl = r->slot_host(s);
            const uint32_t q = load_acquire(reinterpret_cast<const uint32_t*>(sl + r->lay.req));
            if (q == last[s]) continue;
            const int rows = (int) *reinterpret_cast<volatile uint32_t*>(sl + r->lay.rows);
            const int topk = (int) *reinterpret_cast<volatile uint32_t*>(sl + r->lay.topk);
            float* out = reinterpret_cast<float*>(sl + r->lay.out);
            const uint64_t t_pick = (uint64_t) std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
            const uint64_t ld = r->last_done_ns.load(std::memory_order_relaxed);
            if (ld) r->gap_ns.fetch_add(t_pick - ld, std::memory_order_relaxed);
            const int64_t h = r->handles[s].load(std::memory_order_acquire);
            if (h >= 0 && rows > 0 && rows <= HYB_MAX_T && topk > 0 && topk <= HYB_MAX_K)
            {
                // Work on private (ordinary cacheable heap) copies: the kernel re-reads its input
                // and read-modify-writes its output; doing that directly in the CUDA-mapped region
                // was ~3x slower. Only the final result is written back to the slot.
                const size_t nx = size_t(rows) * r->max_h, nk = size_t(rows) * topk;
                { const int32_t* ss = reinterpret_cast<const int32_t*>(sl + r->lay.sel); uint64_t c = 0;
                  for (size_t i = 0; i < nk; ++i) c += ss[i] >= 0; r->cold_experts.fetch_add(c, std::memory_order_relaxed); }
                std::memcpy(px.data(), sl + r->lay.x, nx * sizeof(at::Half));
                std::memcpy(psel.data(), sl + r->lay.sel, nk * sizeof(int32_t));
                std::memcpy(pw.data(), sl + r->lay.w, nk * sizeof(at::Half));
                exl3_moe_cpu_forward_raw(h, px.data(), psel.data(), pw.data(), pout.data(), rows, topk, r->threads);
                std::memcpy(out, pout.data(), nx * sizeof(float));
            }
            else
                std::memset(out, 0, sizeof(float) * HYB_MAX_T * r->max_h);
            store_release(reinterpret_cast<uint32_t*>(sl + r->lay.done), q);
            const uint64_t t_done = (uint64_t) std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
            r->compute_ns.fetch_add(t_done - t_pick, std::memory_order_relaxed);
            r->last_done_ns.store(t_done, std::memory_order_relaxed);
            last[s] = q;
            const uint64_t nj = r->jobs.fetch_add(1, std::memory_order_relaxed) + 1;
            if (r->log_every > 0 && nj % (uint64_t) r->log_every == 0)
            {
                printf("glm_hybrid: %llu jobs | %.2f cold experts/job | compute %.0f us/job | gap %.0f us/job\n",
                       (unsigned long long) nj, (double) r->cold_experts.load() / nj,
                       r->compute_ns.load() / 1e3 / nj, r->gap_ns.load() / 1e3 / nj);
                fflush(stdout);
            }
            did = true;
        }
        if (did) { idle = 0; continue; }
        if (++idle < 20000) { _mm_pause(); continue; }
        timespec ts{0, 20000};                 // idle > ~ms: back off to 20 us naps (thermals)
        nanosleep(&ts, nullptr);
    }
}

// ---------------------------------------------------------------- device side

__device__ __forceinline__ void st_release_sys(uint32_t* p, uint32_t v)
{
    asm volatile("st.release.sys.global.u32 [%0], %1;" :: "l"(p), "r"(v) : "memory");
}
__device__ __forceinline__ uint32_t ld_acquire_sys(const uint32_t* p)
{
    uint32_t v;
    asm volatile("ld.acquire.sys.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
    return v;
}

template <typename XT> __device__ __forceinline__ float to_f(XT v);
template <> __device__ __forceinline__ float to_f<__nv_bfloat16>(__nv_bfloat16 v) { return __bfloat162float(v); }
template <> __device__ __forceinline__ float to_f<__half>(__half v) { return __half2float(v); }
template <> __device__ __forceinline__ float to_f<float>(float v) { return v; }
template <typename XT> __device__ __forceinline__ XT from_f(float v);
template <> __device__ __forceinline__ __nv_bfloat16 from_f<__nv_bfloat16>(float v) { return __float2bfloat16(v); }
template <> __device__ __forceinline__ __half from_f<__half>(float v) { return __float2half(v); }
template <> __device__ __forceinline__ float from_f<float>(float v) { return v; }

template <typename XT, typename IT>
__global__ void hyb_submit_kernel(uint8_t* slot, size_t o_req, size_t o_rows, size_t o_topk, size_t o_x,
                                  size_t o_sel, size_t o_w, const XT* __restrict__ x, const IT* __restrict__ ids,
                                  const float* __restrict__ w, const int64_t* __restrict__ cache_map,
                                  const int64_t* __restrict__ source_map, const IT* __restrict__ fallback,
                                  int T, int K, int H, uint32_t* __restrict__ dev_seq,
                                  IT* __restrict__ gpu_ids, float* __restrict__ gpu_w,
                                  IT* __restrict__ gpu_admit, float gpu_frac)
{
    __half* hx = reinterpret_cast<__half*>(slot + o_x);
    if ((T * H) % 8 == 0)
    {
        // 16-byte stores: one PCIe write per 8 halves instead of 8 (x is 8 KB per row)
        uint4* hx4 = reinterpret_cast<uint4*>(hx);
        for (int v = threadIdx.x; v < T * H / 8; v += blockDim.x)
        {
            __align__(16) __half tmp[8];
            #pragma unroll
            for (int j = 0; j < 8; ++j) tmp[j] = __float2half(to_f<XT>(x[v * 8 + j]));
            hx4[v] = *reinterpret_cast<const uint4*>(tmp);
        }
    }
    else
        for (int i = threadIdx.x; i < T * H; i += blockDim.x) hx[i] = __float2half(to_f<XT>(x[i]));
    int32_t* hsel = reinterpret_cast<int32_t*>(slot + o_sel);
    __half* hw = reinterpret_cast<__half*>(slot + o_w);
    // one thread per token row: split that row's cold experts, first round(frac * n_cold) -> GPU
    // (admitted + copied before the GEMV), the rest -> CPU. gpu_admit lists every id except the
    // CPU-bound ones (those become the fallback, an already-resident id).
    for (int t = threadIdx.x; t < T; t += blockDim.x)
    {
        int n_cold = 0;
        for (int j = 0; j < K; ++j) n_cold += cache_map[ids[t * K + j]] < 0;
        const int n_gpu = (int) (gpu_frac * (float) n_cold + 0.5f);
        int seen = 0;
        for (int j = 0; j < K; ++j)
        {
            const int i = t * K + j;
            const IT id = ids[i];
            const bool resident = cache_map[id] >= 0;
            const bool to_gpu = resident || (seen++ < n_gpu);
            hsel[i] = to_gpu ? -1 : (int32_t) source_map[id];
            hw[i] = __float2half(w[i]);
            gpu_ids[i] = to_gpu ? id : fallback[0];
            gpu_w[i] = to_gpu ? w[i] : 0.0f;
            gpu_admit[i] = to_gpu ? id : fallback[0];
        }
    }
    __syncthreads();
    if (threadIdx.x == 0)
    {
        __threadfence_system();
        const uint32_t s = dev_seq[0] + 1u;
        dev_seq[0] = s;
        *reinterpret_cast<volatile uint32_t*>(slot + o_rows) = (uint32_t) T;
        *reinterpret_cast<volatile uint32_t*>(slot + o_topk) = (uint32_t) K;
        __threadfence_system();
        st_release_sys(reinterpret_cast<uint32_t*>(slot + o_req), s);
    }
}


// Fused hybrid submit + exact LRU admission of the GPU subset (replaces hyb_submit + _update_exl3_lru_map_kernel).
// Warp 0 splits each row with ballots (same rule as hyb_submit_kernel: residents and the first round(frac*n_cold)
// cold experts in top-k order go to the GPU), thread 0 signals the CPU worker immediately, then warp 0 runs the
// LRU update over gpu_admit with the triton kernel's semantics (protected = slots holding any requested id;
// per request: hit -> age; miss -> victim = argmin(age, protected = 0x7FFFFFFF), lowest index on ties; evict,
// map, protect; miss_local_ids/miss_slots; clock += n; hit/miss counters). Slot arrays live in shared memory.
struct HybLru
{
    const int64_t* source_map;
    int64_t* cache_map;
    int64_t* slot_gid;
    int64_t* slot_age;
    int64_t* clock;
    int64_t* miss_local;
    int64_t* miss_slot;
    int64_t* counters;
    int num_experts;
    int capacity;
};

template <typename XT, typename IT>
__global__ void __launch_bounds__(512) hyb_submit_lru_kernel(uint8_t* slot, size_t o_req, size_t o_rows, size_t o_topk,
                                  size_t o_x, size_t o_sel, size_t o_w, const XT* __restrict__ x, const IT* __restrict__ ids,
                                  const float* __restrict__ w, const IT* __restrict__ fallback,
                                  int T, int K, int H, uint32_t* __restrict__ dev_seq,
                                  IT* __restrict__ gpu_ids, float* __restrict__ gpu_w,
                                  IT* __restrict__ gpu_admit, float gpu_frac, HybLru L)
{
    extern __shared__ int64_t hyb_sm[];
    constexpr int NMAX = HYB_MAX_T * HYB_MAX_K;
    __shared__ int64_t s_req[NMAX], s_local[NMAX], s_cslot[NMAX], s_assigned[NMAX];
    __shared__ int s_miss[NMAX];
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;

    __half* hx = reinterpret_cast<__half*>(slot + o_x);
    if ((T * H) % 8 == 0)
    {
        uint4* hx4 = reinterpret_cast<uint4*>(hx);
        for (int v = tid; v < T * H / 8; v += blockDim.x)
        {
            __align__(16) __half tmp[8];
            #pragma unroll
            for (int j = 0; j < 8; ++j) tmp[j] = __float2half(to_f<XT>(x[v * 8 + j]));
            hx4[v] = *reinterpret_cast<const uint4*>(tmp);
        }
    }
    else
        for (int i = tid; i < T * H; i += blockDim.x) hx[i] = __float2half(to_f<XT>(x[i]));

    int32_t* hsel = reinterpret_cast<int32_t*>(slot + o_sel);
    __half* hw = reinterpret_cast<__half*>(slot + o_w);
    if (warp == 0)
    {
        const IT fb = fallback[0];
        for (int t = 0; t < T; ++t)
        {
            const bool act = lane < K;
            const int i = t * K + lane;
            const IT id = act ? ids[i] : (IT) 0;
            const bool resident = act && L.cache_map[id] >= 0;
            const unsigned cold = __ballot_sync(0xffffffffu, act && !resident);
            const int n_gpu = (int) (gpu_frac * (float) __popc(cold) + 0.5f);
            const int rank = __popc(cold & (lane ? (0xffffffffu >> (32 - lane)) : 0u));
            const bool to_gpu = resident || rank < n_gpu;
            if (act)
            {
                const float wi = w[i];
                hsel[i] = to_gpu ? -1 : (int32_t) L.source_map[id];
                hw[i] = __float2half(wi);
                gpu_ids[i] = to_gpu ? id : fb;
                gpu_w[i] = to_gpu ? wi : 0.0f;
                const IT adm = to_gpu ? id : fb;
                gpu_admit[i] = adm;
                s_req[i] = (int64_t) adm;
            }
        }
    }
    __syncthreads();
    if (tid == 0)
    {
        __threadfence_system();
        const uint32_t sq = dev_seq[0] + 1u;
        dev_seq[0] = sq;
        *reinterpret_cast<volatile uint32_t*>(slot + o_rows) = (uint32_t) T;
        *reinterpret_cast<volatile uint32_t*>(slot + o_topk) = (uint32_t) K;
        __threadfence_system();
        st_release_sys(reinterpret_cast<uint32_t*>(slot + o_req), sq);
    }

    // ---- LRU admission of gpu_admit (all T*K ids, in order)
    const int N = T * K;
    const int cap = L.capacity;
    int64_t* s_gid = hyb_sm;
    int64_t* s_age = hyb_sm + cap;
    uint8_t* s_prot = reinterpret_cast<uint8_t*>(hyb_sm + 2 * cap);
    for (int q = tid; q < cap; q += blockDim.x) { s_gid[q] = L.slot_gid[q]; s_age[q] = L.slot_age[q]; }
    if (tid < N)
    {
        const int64_t gid = s_req[tid];
        const bool valid = gid >= 0 && gid < L.num_experts;
        const int64_t loc = valid ? L.source_map[gid] : -1;
        s_local[tid] = loc;
        s_cslot[tid] = (valid && loc >= 0) ? L.cache_map[gid] : -1;
    }
    __syncthreads();
    for (int q = tid; q < cap; q += blockDim.x)
    {
        bool pr = false;
        for (int r = 0; r < N; ++r) pr |= s_gid[q] == s_req[r];
        s_prot[q] = pr;
    }
    __syncthreads();
    if (warp == 0)
    {
        const int64_t clock = L.clock[0];
        long long hits = 0, misses = 0;
        for (int r = 0; r < N; ++r)
        {
            const int64_t gid = s_req[r];
            const bool valid = gid >= 0 && gid < L.num_experts;
            const int64_t loc = s_local[r];
            const bool is_local = valid && loc >= 0;
            int first = -1;
            for (int j = 0; j < r; ++j) if (s_req[j] == gid) { first = j; break; }
            const bool appeared = first >= 0;
            int64_t cslot = s_cslot[r];
            if (appeared && s_miss[first]) cslot = s_assigned[first];   // sees the earlier admission
            const bool is_hit = is_local && cslot >= 0;
            const bool is_miss = is_local && !is_hit && !appeared;
            int64_t victim = -1;
            if (is_miss)
            {
                int64_t best = LLONG_MAX;
                int bi = INT_MAX;
                for (int q = lane; q < cap; q += 32)
                {
                    const int64_t a = s_prot[q] ? (int64_t) 0x7FFFFFFF : s_age[q];
                    if (a < best) { best = a; bi = q; }
                }
                #pragma unroll
                for (int o = 16; o > 0; o >>= 1)
                {
                    const int64_t ob = __shfl_xor_sync(0xffffffffu, best, o);
                    const int oi = __shfl_xor_sync(0xffffffffu, bi, o);
                    if (ob < best || (ob == best && oi < bi)) { best = ob; bi = oi; }
                }
                victim = bi;
            }
            if (lane == 0)
            {
                if (is_miss)
                {
                    const int64_t old = s_gid[victim];
                    if (old >= 0) L.cache_map[old] = -1;
                    L.cache_map[gid] = victim;
                    s_gid[victim] = gid;
                    s_prot[victim] = 1;
                }
                const int64_t assigned = is_hit ? cslot : victim;
                if (is_hit || is_miss) s_age[assigned] = clock + r + 1;
                L.miss_local[r] = is_miss ? loc : -1;
                L.miss_slot[r] = is_miss ? victim : -1;
                s_miss[r] = is_miss;
                s_assigned[r] = assigned;
                hits += is_hit;
                misses += is_miss;
            }
            __syncwarp();
        }
        if (lane == 0)
        {
            L.clock[0] = clock + N;
            atomicAdd(reinterpret_cast<unsigned long long*>(L.counters), (unsigned long long) hits);
            atomicAdd(reinterpret_cast<unsigned long long*>(L.counters + 1), (unsigned long long) misses);
        }
    }
    __syncthreads();
    for (int q = tid; q < cap; q += blockDim.x) { L.slot_gid[q] = s_gid[q]; L.slot_age[q] = s_age[q]; }
}

template <typename OT>
__global__ void hyb_wait_add_kernel(const uint8_t* slot, size_t o_done, size_t o_out, const uint32_t* dev_seq,
                                    OT* __restrict__ out, int n, uint32_t* abort_flag, long long timeout)
{
    if (threadIdx.x == 0)
    {
        const uint32_t want = dev_seq[0];
        const uint32_t* done = reinterpret_cast<const uint32_t*>(slot + o_done);
        const long long t0 = clock64();
        unsigned ns = 256;
        while (ld_acquire_sys(done) != want)
        {
            if (clock64() - t0 > timeout) { atomicExch(abort_flag, 1u); break; }
            __nanosleep(ns);                 // one poller per launch, gentle on the host fabric
            if (ns < 2048) ns <<= 1;
        }
    }
    __syncthreads();
    const float* cpu = reinterpret_cast<const float*>(slot + o_out);
    if (n % 4 == 0)
    {
        const float4* cpu4 = reinterpret_cast<const float4*>(cpu);
        for (int v = blockIdx.x * blockDim.x + threadIdx.x; v < n / 4; v += gridDim.x * blockDim.x)
        {
            const float4 c = __ldcv(cpu4 + v);
            out[v * 4 + 0] = from_f<OT>(to_f<OT>(out[v * 4 + 0]) + c.x);
            out[v * 4 + 1] = from_f<OT>(to_f<OT>(out[v * 4 + 1]) + c.y);
            out[v * 4 + 2] = from_f<OT>(to_f<OT>(out[v * 4 + 2]) + c.z);
            out[v * 4 + 3] = from_f<OT>(to_f<OT>(out[v * 4 + 3]) + c.w);
        }
    }
    else
        for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)
            out[i] = from_f<OT>(to_f<OT>(out[i]) + __ldcv(cpu + i));
}

}  // namespace

// ---------------------------------------------------------------- host API

int64_t hyb_create(int64_t num_slots, int64_t max_h)
{
    std::lock_guard<std::mutex> lk(g_mu);
    auto r = std::make_unique<Region>((int) num_slots, (int) max_h);
    r->bytes = CTRL_SIZE + size_t(num_slots) * r->lay.size;
    void* host = nullptr;
    C10_CUDA_CHECK(cudaHostAlloc(&host, r->bytes, cudaHostAllocMapped | cudaHostAllocPortable));
    std::memset(host, 0, r->bytes);
    void* dev = nullptr;
    C10_CUDA_CHECK(cudaHostGetDevicePointer(&dev, host, 0));
    r->host = (uint8_t*) host;
    r->dev = (uint8_t*) dev;
    g_regions.push_back(std::move(r));
    return (int64_t) g_regions.size() - 1;
}

void hyb_set_layer(int64_t id, int64_t slot, int64_t handle)
{
    Region* r = get_region(id);
    TORCH_CHECK(slot >= 0 && slot < r->num_slots, "slot out of range");
    r->handles[slot].store(handle, std::memory_order_release);
}

void hyb_start(int64_t id, int64_t threads, std::vector<int64_t> cpus)
{
    Region* r = get_region(id);
    TORCH_CHECK(!r->worker.joinable(), "worker already running");
    r->threads = (int) threads;
    r->cpus.assign(cpus.begin(), cpus.end());
    if (const char* e = getenv("EXL3_HYBRID_LOG_EVERY")) r->log_every = atoi(e);
    r->quit = false;
    r->worker = std::thread(worker_loop, r);
}

void hyb_stop(int64_t id)
{
    Region* r = get_region(id);
    r->quit = true;
    if (r->worker.joinable()) r->worker.join();
}

int64_t hyb_jobs(int64_t id) { return (int64_t) get_region(id)->jobs.load(); }
std::vector<int64_t> hyb_stats(int64_t id, bool reset)
{
    Region* r = get_region(id);
    std::vector<int64_t> v{(int64_t) r->jobs.load(), (int64_t) r->compute_ns.load(), (int64_t) r->gap_ns.load()};
    if (reset) { r->jobs = 0; r->compute_ns = 0; r->gap_ns = 0; r->last_done_ns = 0; }
    return v;
}

// Debug: publish a request from the CPU side (no GPU involved). Copies x/sel/w into the slot,
// bumps req_seq, waits for the worker's done, returns microseconds.
double hyb_cpu_kick(int64_t id, int64_t slot, const torch::Tensor& x, const torch::Tensor& sel, const torch::Tensor& w)
{
    Region* r = get_region(id);
    uint8_t* sl = r->slot_host((int) slot);
    const int T = (int) x.size(0), K = (int) sel.size(1);
    std::memcpy(sl + r->lay.x, x.data_ptr(), size_t(T) * x.size(1) * 2);
    std::memcpy(sl + r->lay.sel, sel.data_ptr(), size_t(T) * K * 4);
    std::memcpy(sl + r->lay.w, w.data_ptr(), size_t(T) * K * 2);
    *reinterpret_cast<volatile uint32_t*>(sl + r->lay.rows) = T;
    *reinterpret_cast<volatile uint32_t*>(sl + r->lay.topk) = K;
    uint32_t* req = reinterpret_cast<uint32_t*>(sl + r->lay.req);
    const uint32_t* done = reinterpret_cast<const uint32_t*>(sl + r->lay.done);
    const uint32_t want = load_acquire(done) + 1;
    const auto t0 = std::chrono::steady_clock::now();
    store_release(req, want);
    while (load_acquire(done) != want) _mm_pause();
    return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count();
}

int64_t hyb_abort_flag(int64_t id) { return (int64_t) *reinterpret_cast<volatile uint32_t*>(get_region(id)->host); }

// CPU tensor aliasing the host memory behind a UVA/mapped CUDA tensor (vLLM's offloaded expert
// masters). Uses the driver's own host pointer instead of assuming host == device address.
torch::Tensor hyb_cpu_view(const torch::Tensor& t)
{
    TORCH_CHECK(t.is_cuda(), "expected a CUDA (UVA) tensor");
    cudaPointerAttributes attr{};
    C10_CUDA_CHECK(cudaPointerGetAttributes(&attr, t.data_ptr()));
    TORCH_CHECK(attr.hostPointer != nullptr, "tensor is not backed by host-mapped memory");
    return torch::from_blob(attr.hostPointer, t.sizes(), t.strides(), t.options().device(torch::kCPU));
}

void hyb_submit(int64_t id, int64_t slot, const torch::Tensor& x, const torch::Tensor& ids,
                const torch::Tensor& w, const torch::Tensor& cache_map, const torch::Tensor& source_map,
                const torch::Tensor& fallback, torch::Tensor& dev_seq, torch::Tensor& gpu_ids, torch::Tensor& gpu_w,
                torch::Tensor& gpu_admit, double gpu_frac)
{
    Region* r = get_region(id);
    const at::cuda::OptionalCUDAGuard guard(x.device());
    const int T = (int) x.size(0), H = (int) x.size(1), K = (int) ids.size(1);
    TORCH_CHECK(T <= HYB_MAX_T && K <= HYB_MAX_K && H <= r->max_h, "hybrid submit: shape exceeds slot");
    TORCH_CHECK(w.scalar_type() == at::kFloat && cache_map.scalar_type() == at::kLong &&
                source_map.scalar_type() == at::kLong, "hybrid submit: dtypes");
    TORCH_CHECK(ids.scalar_type() == fallback.scalar_type() && ids.scalar_type() == gpu_ids.scalar_type(), "id dtypes");
    TORCH_CHECK(x.is_contiguous() && ids.is_contiguous() && w.is_contiguous(), "contiguous inputs");
    uint8_t* s = r->slot_dev((int) slot);
    auto stream = at::cuda::getCurrentCUDAStream();
    uint32_t* seq = reinterpret_cast<uint32_t*>(dev_seq.data_ptr<int32_t>() + slot);
    auto& L = r->lay;
#define HYB_LAUNCH(XT, IT) hyb_submit_kernel<XT, IT><<<1, 512, 0, stream>>>(s, L.req, L.rows, L.topk, L.x, L.sel, L.w, \
        reinterpret_cast<const XT*>(x.data_ptr()), ids.data_ptr<IT>(), w.data_ptr<float>(), cache_map.data_ptr<int64_t>(), \
        source_map.data_ptr<int64_t>(), fallback.data_ptr<IT>(), T, K, H, seq, gpu_ids.data_ptr<IT>(), gpu_w.data_ptr<float>(), \
        gpu_admit.data_ptr<IT>(), (float) gpu_frac)
    if (x.scalar_type() == at::kBFloat16) {
        if (ids.scalar_type() == at::kLong) HYB_LAUNCH(__nv_bfloat16, int64_t); else HYB_LAUNCH(__nv_bfloat16, int32_t);
    } else if (x.scalar_type() == at::kHalf) {
        if (ids.scalar_type() == at::kLong) HYB_LAUNCH(__half, int64_t); else HYB_LAUNCH(__half, int32_t);
    } else TORCH_CHECK(false, "hybrid submit: x must be bf16/fp16");
#undef HYB_LAUNCH
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}


bool hyb_submit_lru(int64_t id, int64_t slot, const torch::Tensor& x, const torch::Tensor& ids,
                    const torch::Tensor& w, const torch::Tensor& fallback, torch::Tensor& dev_seq,
                    torch::Tensor& gpu_ids, torch::Tensor& gpu_w, torch::Tensor& gpu_admit, double gpu_frac,
                    const torch::Tensor& source_map, torch::Tensor& cache_map, torch::Tensor& slot_gid,
                    torch::Tensor& slot_age, torch::Tensor& clock, torch::Tensor& miss_local, torch::Tensor& miss_slot,
                    torch::Tensor& counters, int64_t capacity)
{
    Region* r = get_region(id);
    const at::cuda::OptionalCUDAGuard guard(x.device());
    const int T = (int) x.size(0), H = (int) x.size(1), K = (int) ids.size(1);
    TORCH_CHECK(T <= HYB_MAX_T && K <= HYB_MAX_K && H <= r->max_h, "hybrid submit: shape exceeds slot");
    TORCH_CHECK(K <= 32, "fused submit: K <= 32");
    TORCH_CHECK(w.scalar_type() == at::kFloat && cache_map.scalar_type() == at::kLong &&
                source_map.scalar_type() == at::kLong && slot_gid.scalar_type() == at::kLong &&
                slot_age.scalar_type() == at::kLong && clock.scalar_type() == at::kLong &&
                miss_local.scalar_type() == at::kLong && miss_slot.scalar_type() == at::kLong &&
                counters.scalar_type() == at::kLong, "fused submit: dtypes");
    TORCH_CHECK(ids.scalar_type() == fallback.scalar_type() && ids.scalar_type() == gpu_ids.scalar_type(), "id dtypes");
    TORCH_CHECK(x.is_contiguous() && ids.is_contiguous() && w.is_contiguous(), "contiguous inputs");
    TORCH_CHECK(miss_local.numel() >= T * K && miss_slot.numel() >= T * K, "miss buffers too small");
    const size_t smem = (size_t) capacity * 17 + 64;
    if (capacity < 1 || capacity > slot_gid.numel() || smem > 48 * 1024) return false;
    uint8_t* s = r->slot_dev((int) slot);
    auto stream = at::cuda::getCurrentCUDAStream();
    uint32_t* seq = reinterpret_cast<uint32_t*>(dev_seq.data_ptr<int32_t>() + slot);
    auto& Ly = r->lay;
    HybLru L{source_map.data_ptr<int64_t>(), cache_map.data_ptr<int64_t>(), slot_gid.data_ptr<int64_t>(),
             slot_age.data_ptr<int64_t>(), clock.data_ptr<int64_t>(), miss_local.data_ptr<int64_t>(),
             miss_slot.data_ptr<int64_t>(), counters.data_ptr<int64_t>(), (int) cache_map.numel(), (int) capacity};
#define HYB_LAUNCH(XT, IT) hyb_submit_lru_kernel<XT, IT><<<1, 512, smem, stream>>>(s, Ly.req, Ly.rows, Ly.topk, Ly.x, \
        Ly.sel, Ly.w, reinterpret_cast<const XT*>(x.data_ptr()), ids.data_ptr<IT>(), w.data_ptr<float>(), \
        fallback.data_ptr<IT>(), T, K, H, seq, gpu_ids.data_ptr<IT>(), gpu_w.data_ptr<float>(), \
        gpu_admit.data_ptr<IT>(), (float) gpu_frac, L)
    if (x.scalar_type() == at::kBFloat16) {
        if (ids.scalar_type() == at::kLong) HYB_LAUNCH(__nv_bfloat16, int64_t); else HYB_LAUNCH(__nv_bfloat16, int32_t);
    } else if (x.scalar_type() == at::kHalf) {
        if (ids.scalar_type() == at::kLong) HYB_LAUNCH(__half, int64_t); else HYB_LAUNCH(__half, int32_t);
    } else TORCH_CHECK(false, "hybrid submit: x must be bf16/fp16");
#undef HYB_LAUNCH
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return true;
}


// debug: copy a slot's request area (x, sel, w, rows, topk, req) out of pinned host memory
std::vector<torch::Tensor> hyb_slot_dump(int64_t id, int64_t slot, int64_t T, int64_t K, int64_t H)
{
    Region* r = get_region(id);
    const uint8_t* h = r->slot_host((int) slot);
    auto x = torch::empty({T, H}, torch::kHalf);
    auto sel = torch::empty({T * K}, torch::kInt);
    auto w = torch::empty({T * K}, torch::kHalf);
    auto meta = torch::empty({3}, torch::kInt);
    std::memcpy(x.data_ptr(), h + r->lay.x, sizeof(__half) * T * H);
    std::memcpy(sel.data_ptr(), h + r->lay.sel, sizeof(int32_t) * T * K);
    std::memcpy(w.data_ptr(), h + r->lay.w, sizeof(__half) * T * K);
    int32_t* m = meta.data_ptr<int32_t>();
    m[0] = (int32_t) load_acquire(reinterpret_cast<const uint32_t*>(h + r->lay.rows));
    m[1] = (int32_t) load_acquire(reinterpret_cast<const uint32_t*>(h + r->lay.topk));
    m[2] = (int32_t) load_acquire(reinterpret_cast<const uint32_t*>(h + r->lay.req));
    return {x, sel, w, meta};
}

void hyb_wait_add(int64_t id, int64_t slot, const torch::Tensor& dev_seq, torch::Tensor& out, double timeout_s)
{
    Region* r = get_region(id);
    const at::cuda::OptionalCUDAGuard guard(out.device());
    TORCH_CHECK(out.is_contiguous(), "out must be contiguous");
    const int n = (int) out.numel();
    TORCH_CHECK(n <= HYB_MAX_T * r->max_h, "hybrid wait_add: out too large");
    auto stream = at::cuda::getCurrentCUDAStream();
    const uint8_t* s = r->slot_dev((int) slot);
    const uint32_t* seq = reinterpret_cast<const uint32_t*>(dev_seq.data_ptr<int32_t>() + slot);
    uint32_t* abort_flag = reinterpret_cast<uint32_t*>(r->dev);
    const long long timeout = (long long) (timeout_s * 1.9e9);
    const int blocks = 1;   // single poller; the add is only rows*H elements
    if (out.scalar_type() == at::kBFloat16)
        hyb_wait_add_kernel<__nv_bfloat16><<<blocks, 1024, 0, stream>>>(s, r->lay.done, r->lay.out, seq,
            reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), n, abort_flag, timeout);
    else if (out.scalar_type() == at::kHalf)
        hyb_wait_add_kernel<__half><<<blocks, 512, 0, stream>>>(s, r->lay.done, r->lay.out, seq,
            reinterpret_cast<__half*>(out.data_ptr()), n, abort_flag, timeout);
    else if (out.scalar_type() == at::kFloat)
        hyb_wait_add_kernel<float><<<blocks, 512, 0, stream>>>(s, r->lay.done, r->lay.out, seq,
            out.data_ptr<float>(), n, abort_flag, timeout);
    else TORCH_CHECK(false, "hybrid wait_add: out must be bf16/fp16/fp32");
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
