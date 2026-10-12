#!/usr/bin/env python3
"""glm_fast.mhc_post_pre_norm vs the split decode path (torch reference and the tilelang kernels vLLM runs).

usage: test_mhc.py <model_dir> [layer]
Real hc_attn_fn/scale/base + input_layernorm weight of one layer; synthetic residual streams with GLM-like
tiny magnitudes; post/comb produced by a previous pre block. Reports exactness and graph-replay timing.
"""
import json, sys
import torch
from safetensors import safe_open
import glm_fast

model = sys.argv[1]
layer = int(sys.argv[2]) if len(sys.argv) > 2 else 5
idx = json.load(open(f"{model}/model.safetensors.index.json"))["weight_map"]
pre = f"model.language_model.layers.{layer}."
def get(n):
    with safe_open(f"{model}/{idx[pre + n]}", framework="pt", device="cpu") as f:
        return f.get_tensor(pre + n)
dev = torch.device("cuda:0")
fn = get("hc_attn_fn").float().contiguous().to(dev)
scale = get("hc_attn_scale").float().to(dev)
base = get("hc_attn_base").float().to(dev)
fn2 = get("hc_ffn_fn").float().contiguous().to(dev)
scale2 = get("hc_ffn_scale").float().to(dev)
base2 = get("hc_ffn_base").float().to(dev)
nw = get("input_layernorm.weight").bfloat16().to(dev)
print(f"layer {layer}: fn {tuple(fn.shape)} scale {scale.tolist()} base[:8] {base[:8].tolist()}")
H, HC = 4096, 4
RMS_EPS, HC_EPS, SK, PM, NORM_EPS = 1e-5, 1e-6, 20, 2.0, 1e-5

from vllm.model_executor.kernels.mhc.torch import mhc_pre_torch, mhc_post_torch

def ref(x, res, post, comb):
    res_cur = mhc_post_torch(x, res, post, comb)
    p, c, li = mhc_pre_torch(res_cur, fn, scale, base, RMS_EPS, HC_EPS, HC_EPS, PM, SK)
    lf = li.float()
    y = (lf * torch.rsqrt(lf.pow(2).mean(-1, keepdim=True) + NORM_EPS) * nw.float()).bfloat16()
    return res_cur, p, c, y

tl = None
try:
    from vllm.model_executor.kernels.mhc.tilelang import mhc_pre_tilelang, mhc_post_tilelang
    def tl(x, res, post, comb):
        res_cur = mhc_post_tilelang(x, res, post, comb)
        p, c, li = mhc_pre_tilelang(res_cur, fn, scale, base, RMS_EPS, HC_EPS, HC_EPS, PM, SK, 1, None, 0.0)
        lf = li.float()
        y = (lf * torch.rsqrt(lf.pow(2).mean(-1, keepdim=True) + NORM_EPS) * nw.float()).bfloat16()
        return res_cur, p, c, y
except Exception as e:  # noqa: BLE001
    print(f"tilelang path unavailable: {type(e).__name__}: {e}")

counters = torch.zeros(1024, dtype=torch.int32, device=dev)
NB = 64
def fast(x, res, post, comb):
    return glm_fast.mhc_post_pre_norm(x, res, post.contiguous(), comb.contiguous(), fn, scale, base, nw, counters,
                                      RMS_EPS, HC_EPS, HC_EPS, PM, SK, NORM_EPS, NB)

g = torch.Generator(device=dev).manual_seed(0)
def inputs(T):
    res0 = (torch.randn(T, HC, H, device=dev, generator=g) * 3e-3).bfloat16()
    p0, c0, _ = mhc_pre_torch(res0, fn2, scale2, base2, RMS_EPS, HC_EPS, HC_EPS, PM, SK)
    res = (torch.randn(T, HC, H, device=dev, generator=g) * 3e-3).bfloat16()
    x = (torch.randn(T, H, device=dev, generator=g) * 2e-3).bfloat16()
    return x, res, p0.contiguous(), c0.contiguous()

def ulps(a, b):
    a = a.float(); b = b.float()
    d = (a - b).abs()
    sp = torch.where(b.abs() > 0, b.abs(), torch.ones_like(b)) * 2.0 ** -7
    return (d / sp).max().item(), (a != b).float().mean().item()

ok = True
for T in (1, 2, 4, 8):
    worst = {}
    for trial in range(20):
        x, res, p, c = inputs(T)
        R = ref(x, res, p, c)
        F = fast(x, res, p, c)
        for name, a, b in (("res", F[0], R[0]), ("y", F[3], R[3])):
            u, frac = ulps(a, b)
            w = worst.get(name, (0, 0)); worst[name] = (max(w[0], u), max(w[1], frac))
        for name, a, b in (("post", F[1], R[1]), ("comb", F[2], R[2])):
            rel = ((a - b).abs().max() / b.abs().max()).item()
            worst[name] = max(worst.get(name, 0), rel)
        if counters[:T].abs().sum().item() != 0:
            print("counters not reset!"); ok = False
        if not (worst["res"][0] <= 64 and worst["y"][0] <= 8):
            print("far from torch ref", worst); ok = False
    print(f"T={T} vs torch ref: res max {worst['res'][0]:.2f} ulp ({worst['res'][1]*100:.3f}% differ), "
          f"y max {worst['y'][0]:.2f} ulp ({worst['y'][1]*100:.3f}% differ), post rel {worst['post']:.1e}, comb rel {worst['comb']:.1e}")
    if tl is not None:
        # gate against the production split path: residual bitwise, y within 1.5 bf16 ulp, mixes ~fp32 noise
        for trial in range(10):
            x, res, p, c = inputs(T)
            Tt, F = tl(x, res, p, c), fast(x, res, p, c)
            ur, uy = ulps(F[0], Tt[0]), ulps(F[3], Tt[3])
            dp = ((F[1]-Tt[1]).abs().max()/Tt[1].abs().max()).item(); dc = ((F[2]-Tt[2]).abs().max()/Tt[2].abs().max()).item()
            ok &= ur[0] == 0 and uy[0] <= 2.01 and dp < 2e-5 and dc < 2e-5
            if not (ur[0] == 0 and uy[0] <= 2.01 and dp < 2e-5 and dc < 2e-5): print('FAIL detail', T, ur, uy, dp, dc)
        print(f"      vs tilelang (last of 10): res {ur}, y {uy}, post {dp:.1e}, comb {dc:.1e}")

# timing inside CUDA graphs (decode, T=1), 45 layers x 2 boundaries worth of calls
x, res, p, c = inputs(1)
def bench(fnc, n=90, reps=50):
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        for _ in range(3): fnc(x, res, p, c)
    torch.cuda.synchronize()
    gr = torch.cuda.CUDAGraph()
    with torch.cuda.graph(gr, stream=s):
        for _ in range(n): fnc(x, res, p, c)
    for _ in range(5): gr.replay()
    torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    e0.record()
    for _ in range(reps): gr.replay()
    e1.record(); torch.cuda.synchronize()
    return e0.elapsed_time(e1) * 1e3 / (reps * n)

def tl_norm(x, res, p, c):
    from vllm import _custom_ops as ops
    res_cur = mhc_post_tilelang(x, res, p, c)
    pp, cc, li = mhc_pre_tilelang(res_cur, fn, scale, base, RMS_EPS, HC_EPS, HC_EPS, PM, SK, 1, None, 0.0)
    out = torch.empty_like(li)
    ops.rms_norm(out, li, nw, NORM_EPS)
    return res_cur, pp, cc, out

for NB in (64,):
    print(f"fused kernel NB={NB}: {bench(fast):.2f} us/boundary")
NB = 64
def fast_sk1(x, res, post, comb):
    return glm_fast.mhc_post_pre_norm(x, res, post, comb, fn, scale, base, nw, counters,
                                      RMS_EPS, HC_EPS, HC_EPS, PM, 1, NORM_EPS, 64)
print(f"fused kernel NB=64, sinkhorn 1 iter (timing only): {bench(fast_sk1):.2f} us/boundary")
if tl is not None:
    try:
        print(f"tilelang split + rms_norm: {bench(tl_norm):.2f} us/boundary")
    except Exception as e:  # noqa: BLE001
        print(f"tilelang bench failed: {type(e).__name__}: {e}")
print("OK" if ok else "FAIL")
