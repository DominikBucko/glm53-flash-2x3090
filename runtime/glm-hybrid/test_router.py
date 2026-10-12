#!/usr/bin/env python3
"""glm_fast.router_topk vs the vLLM decode router (F.linear bf16 -> .to(fp32) -> ops.grouped_topk sigmoid/bias/
renorm/scale) on real GLM-5.3 gate weights + correction bias. Reports logit / id / weight agreement over many
random rows (ids may differ only when the bf16 logits differ by an ulp at a near-tie), and graph timing."""
import json, sys
import torch
import torch.nn.functional as F
from safetensors import safe_open
import glm_fast
from vllm import _custom_ops as ops

model = sys.argv[1]
layer = int(sys.argv[2]) if len(sys.argv) > 2 else 10
idx = json.load(open(f"{model}/model.safetensors.index.json"))["weight_map"]
def get(n):
    with safe_open(f"{model}/{idx[n]}", framework="pt", device="cpu") as f:
        return f.get_tensor(n)
dev = torch.device("cuda:0")
pre = f"model.language_model.layers.{layer}.mlp.gate."
W = get(pre + "weight").bfloat16().contiguous().to(dev)
bias_key = [k for k in idx if k.startswith(pre) and "bias" in k]
bias = get(bias_key[0]).float().to(dev) if bias_key else None
print(f"layer {layer}: W {tuple(W.shape)}, bias {bias_key}")
E, H = W.shape
TOPK, SCALE = 8, 2.5
ctr = torch.zeros(64, dtype=torch.int32, device=dev)

def ref(x):
    logits = F.linear(x, W).to(torch.float32)
    w, ids = ops.grouped_topk(logits, 1, 1, TOPK, True, SCALE, bias, 1)
    return logits, w, ids

def fast(x):
    T = x.shape[0]
    w = torch.empty(T, TOPK, dtype=torch.float32, device=dev)
    ids = torch.empty(T, TOPK, dtype=torch.int32, device=dev)
    lg = glm_fast.router_topk(x, W, bias, TOPK, True, SCALE, w, ids, ctr)
    fast.last_logits = lg
    return w, ids

g = torch.Generator(device=dev).manual_seed(0)
n_rows = 0; same_ids = 0; same_w = 0; same_order = 0; same_logits = 0; exact_ref = 0; exact_fast = 0
for trial in range(200):
    T = [1, 1, 2, 4][trial % 4]
    x = (torch.randn(T, H, device=dev, generator=g) * (0.02 + 0.2 * torch.rand(1, device=dev, generator=g))).bfloat16()
    lr, wr, ir = ref(x)
    wf, idf = fast(x)
    exact = (x.double() @ W.double().T).float().bfloat16().float()
    for t in range(T):
        same_logits += torch.equal(lr[t], fast.last_logits[t])
        exact_ref += torch.equal(lr[t], exact[t]); exact_fast += torch.equal(fast.last_logits[t], exact[t])
        if torch.equal(lr[t], fast.last_logits[t]) and not torch.equal(wr[t], wf[t]) and n_rows < 400 and same_logits < 3:
            print("same logits, weights differ:", wr[t].tolist(), wf[t].tolist(), ir[t].tolist(), idf[t].tolist())
        n_rows += 1
        same_order += torch.equal(ir[t].int(), idf[t])
        same_ids += set(ir[t].tolist()) == set(idf[t].tolist())
        same_w += torch.equal(wr[t], wf[t]) and torch.equal(ir[t].int(), idf[t])
print(f"{n_rows} rows: identical ids+order {same_order}, same id set {same_ids}, bit-identical weights {same_w}")
print(f"logit rows identical to vLLM {same_logits}; rows equal to exactly-rounded bf16: vLLM {exact_ref}, fused {exact_fast}")

x = (torch.randn(1, H, device=dev, generator=g) * 0.05).bfloat16()
def bench(fn, reps=50):
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        for _ in range(3): fn(x)
    torch.cuda.synchronize()
    gr = torch.cuda.CUDAGraph()
    with torch.cuda.graph(gr, stream=s):
        for _ in range(42): fn(x)
    gr.replay(); torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    e0.record()
    for _ in range(reps): gr.replay()
    e1.record(); torch.cuda.synchronize()
    return e0.elapsed_time(e1) * 1e3 / (reps * 42)
def ref_ids64(x):
    _, w, ids = ref(x)
    return w, ids.to(torch.long)
print(f"vLLM router (linear + fp32 cast + grouped_topk + int64 ids): {bench(ref_ids64):.1f} us; fused: {bench(fast):.1f} us")
