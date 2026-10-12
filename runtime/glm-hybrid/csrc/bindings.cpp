#include <torch/extension.h>
#include "moe_mul1.h"

int64_t hyb_create(int64_t num_slots, int64_t max_h);
void hyb_set_layer(int64_t id, int64_t slot, int64_t handle);
void hyb_start(int64_t id, int64_t threads, std::vector<int64_t> cpus);
void hyb_stop(int64_t id);
int64_t hyb_jobs(int64_t id);
std::vector<int64_t> hyb_stats(int64_t id, bool reset);
int64_t hyb_abort_flag(int64_t id);
double hyb_cpu_kick(int64_t id, int64_t slot, const torch::Tensor& x, const torch::Tensor& sel, const torch::Tensor& w);
torch::Tensor hyb_cpu_view(const torch::Tensor& t);
void hyb_submit(int64_t id, int64_t slot, const torch::Tensor& x, const torch::Tensor& ids,
                const torch::Tensor& w, const torch::Tensor& cache_map, const torch::Tensor& source_map,
                const torch::Tensor& fallback, torch::Tensor& dev_seq, torch::Tensor& gpu_ids, torch::Tensor& gpu_w,
                torch::Tensor& gpu_admit, double gpu_frac);
void hyb_wait_add(int64_t id, int64_t slot, const torch::Tensor& dev_seq, torch::Tensor& out, double timeout_s);
std::vector<torch::Tensor> hyb_slot_dump(int64_t id, int64_t slot, int64_t T, int64_t K, int64_t H);
bool hyb_submit_lru(int64_t id, int64_t slot, const torch::Tensor& x, const torch::Tensor& ids,
                    const torch::Tensor& w, const torch::Tensor& fallback, torch::Tensor& dev_seq,
                    torch::Tensor& gpu_ids, torch::Tensor& gpu_w, torch::Tensor& gpu_admit, double gpu_frac,
                    const torch::Tensor& source_map, torch::Tensor& cache_map, torch::Tensor& slot_gid,
                    torch::Tensor& slot_age, torch::Tensor& clock, torch::Tensor& miss_local, torch::Tensor& miss_slot,
                    torch::Tensor& counters, int64_t capacity);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    // ExLlamaV3 native CPU MoE kernel (cpu/moe_mul1.cpp, copied unchanged apart from affinity-aware pinning)
    m.def("make_layer", &exl3_moe_cpu_make_layer);
    m.def("free_layer", &exl3_moe_cpu_free_layer);
    m.def("forward", &exl3_moe_cpu_forward, py::call_guard<py::gil_scoped_release>());
    m.def("has_avx2", &exl3_moe_cpu_has_avx2);
    m.def("set_prof", &exl3_moe_cpu_set_prof);
    // bridge
    m.def("create", &hyb_create);
    m.def("set_layer", &hyb_set_layer);
    m.def("start", &hyb_start);
    m.def("stop", &hyb_stop, py::call_guard<py::gil_scoped_release>());
    m.def("jobs", &hyb_jobs);
    m.def("stats", &hyb_stats);
    m.def("abort_flag", &hyb_abort_flag);
    m.def("cpu_kick", &hyb_cpu_kick, py::call_guard<py::gil_scoped_release>());
    m.def("cpu_view", &hyb_cpu_view);
    m.def("submit", &hyb_submit);
    m.def("wait_add", &hyb_wait_add);
    m.def("submit_lru", &hyb_submit_lru);
    m.def("slot_dump", &hyb_slot_dump);
}
