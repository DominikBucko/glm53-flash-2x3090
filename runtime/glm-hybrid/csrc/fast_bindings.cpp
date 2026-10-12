#include <torch/extension.h>

std::vector<torch::Tensor> mhc_post_pre_norm(
    const torch::Tensor& x, const torch::Tensor& residual, const torch::Tensor& post, const torch::Tensor& comb,
    const torch::Tensor& fn, const torch::Tensor& hc_scale, const torch::Tensor& hc_base,
    const torch::Tensor& norm_w, torch::Tensor& counters, double rms_eps, double pre_eps, double sk_eps,
    double post_mult, int64_t sk_repeat, double norm_eps, int64_t nb);

void topk_sorted(const torch::Tensor& logits, int64_t k, torch::Tensor& out, torch::Tensor& ws);
int64_t topk_ws_ints_per_row();

torch::Tensor router_topk(const torch::Tensor& x, const torch::Tensor& W, const c10::optional<torch::Tensor>& bias, int64_t topk,
                 bool renorm, double scale, torch::Tensor& out_w, torch::Tensor& out_ids, torch::Tensor& counters);

void moe_split_tables(const torch::Tensor& gu_tr, const torch::Tensor& d_tr, const torch::Tensor& miss_slots,
                      torch::Tensor& gu_a, torch::Tensor& gu_b, torch::Tensor& d_a, torch::Tensor& d_b);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("mhc_post_pre_norm", &mhc_post_pre_norm);
    m.def("topk_sorted", &topk_sorted);
    m.def("topk_ws_ints_per_row", &topk_ws_ints_per_row);
    m.def("router_topk", &router_topk);
    m.def("moe_split_tables", &moe_split_tables);
}
