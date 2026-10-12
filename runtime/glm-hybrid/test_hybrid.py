#!/usr/bin/env python3
"""glm_hybrid unit test: TP-half CPU math + graph-safe bridge, on real GLM-5.3 EXL3 experts.

1. Layer L experts split like vLLM's TP2 layout (gate/up: output tiles, down: input tiles, matching
   suh/svh halves). CPU(rank0 half) + CPU(rank1 half) must equal CPU(full expert).
2. Bridge: submit (GPU) -> worker (CPU, ExLlamaV3 kernel) -> wait_add (GPU), eager and inside a CUDA
   graph replayed with fresh inputs; each result checked against a direct CPU forward with the same
   cold/resident split. Then 42 slots per replay (one decode token) are timed.
usage: test_hybrid.py <model_dir> <layer> [threads]
"""
import json, sys, time
import torch
from safetensors import safe_open
import glm_hybrid as gh

model, layer = sys.argv[1], int(sys.argv[2])
threads = int(sys.argv[3]) if len(sys.argv) > 3 else 16
idx = json.load(open(f"{model}/model.safetensors.index.json"))["weight_map"]
pre = f"model.language_model.layers.{layer}.mlp.experts"
E, H = 288, 4096
fh = {}
def get(n):
    f = idx[n]
    if f not in fh: fh[f] = safe_open(f"{model}/{f}", framework="pt", device="cpu")
    return fh[f].get_tensor(n)

full = {k: [] for k in "gt gs gv ut us uv dt ds dv".split()}
half = [{k: [] for k in full} for _ in range(2)]
for e in range(E):
    for p, (kt, ks, kv) in (("gate_proj", "gt gs gv".split()), ("up_proj", "ut us uv".split())):
        t, s, v = (get(f"{pre}.{e}.{p}.{n}") for n in ("trellis", "suh", "svh"))
        full[kt].append(t.clone()); full[ks].append(s.half().clone()); full[kv].append(v.half().clone())
        no = t.shape[1] // 2; nv = v.numel() // 2
        for r in range(2):
            half[r][kt].append(t[:, r * no:(r + 1) * no].contiguous()); half[r][ks].append(s.half().clone())
            half[r][kv].append(v.half()[r * nv:(r + 1) * nv].contiguous())
    t, s, v = (get(f"{pre}.{e}.down_proj.{n}") for n in ("trellis", "suh", "svh"))
    full["dt"].append(t.clone()); full["ds"].append(s.half().clone()); full["dv"].append(v.half().clone())
    ni = t.shape[0] // 2; ns = s.numel() // 2
    for r in range(2):
        half[r]["dt"].append(t[r * ni:(r + 1) * ni].contiguous()); half[r]["ds"].append(s.half()[r * ns:(r + 1) * ns].contiguous())
        half[r]["dv"].append(v.half().clone())
print(f"layer {layer}: gate trellis {tuple(full['gt'][0].shape)} -> half {tuple(half[0]['gt'][0].shape)}, "
      f"down {tuple(full['dt'][0].shape)} -> half {tuple(half[0]['dt'][0].shape)}")

def mk(d):
    return gh.make_layer(d["gt"], d["gs"], d["gv"], d["ut"], d["us"], d["uv"], d["dt"], d["ds"], d["dv"], [], [], [], 0, 0.0, 0)
hf, h0, h1 = mk(full), mk(half[0]), mk(half[1])

g = torch.Generator().manual_seed(1)
T, K = 1, 8
x = (torch.randn(T, H, generator=g) * 0.5).half()
sel = torch.randperm(E, generator=g)[:K].view(1, K).int()
w = torch.softmax(torch.randn(T, K, generator=g), -1).half()  # ExLlamaV3 CPU forward reads weights as fp16
def cpu_fwd(h, sel_):
    out = torch.zeros(T, H)
    gh.forward(h, x, sel_, w, out, threads)
    return out
yf, y0, y1 = cpu_fwd(hf, sel), cpu_fwd(h0, sel), cpu_fwd(h1, sel)
# exact-ish GPU reference (ExLlamaV3 non-fused path: had(x*suh) @ reconstruct(W) -> had(.)*svh, fp32 GEMM)
from exllamav3.ext import exllamav3_ext as xe
cu = torch.device("cuda:0")
def lin(xin, tr, su, sv):
    k, n = tr.shape[0] * 16, tr.shape[1] * 16
    xh = torch.empty_like(xin); xe.had_r_128(xin, xh, su.to(cu), None, 1.0)
    W = torch.empty(k, n, dtype=torch.half, device=cu); xe.reconstruct(W, tr.to(cu), tr.shape[2] // 16, False, True)
    y = (xh.float() @ W.float()).half(); xe.had_r_128(y, y, None, sv.to(cu), 1.0)
    return y.float()
ref = torch.zeros(T, H, device=cu)
xg = x.to(cu)
for j in range(K):
    e = int(sel[0, j]); 
    gg = lin(xg, full["gt"][e], full["gs"][e], full["gv"][e]); uu = lin(xg, full["ut"][e], full["us"][e], full["uv"][e])
    a = (torch.nn.functional.silu(gg) * uu).half()
    ref += float(w[0, j]) * lin(a, full["dt"][e], full["ds"][e], full["dv"][e])
ref = ref.cpu()
rf = ((yf - ref).norm() / ref.norm()).item(); rh = ((y0 + y1 - ref).norm() / ref.norm()).item(); rel = ((y0 + y1 - yf).norm() / yf.norm()).item()
print(f"[1] vs GPU fp32 reference: CPU full expert {rf:.2e}, CPU TP halves {rh:.2e}; halves vs full {rel:.2e} (ref rms {ref.pow(2).mean().sqrt():.4f})")
assert rh < 3 * max(rf, 3e-3), "TP-half split is much worse than the full-expert CPU path"

# ---- 2. bridge
dev = torch.device("cuda:0")
NS = 42
reg = gh.create(NS, H)
for s in range(NS):
    gh.set_layer(reg, s, h0)
gh.start(reg, threads, list(range(0, 4)) + list(range(8, 12)) + list(range(16, 20)) + list(range(24, 28)))
cache_map = torch.full((E,), -1, dtype=torch.long, device=dev)
resident = torch.randperm(E, generator=g)[:120].to(dev)
cache_map[resident] = torch.arange(120, device=dev)
source_map = torch.arange(E, dtype=torch.long, device=dev)
fallback = resident[:1].clone()
dev_seq = torch.zeros(NS, dtype=torch.int32, device=dev)
xs = torch.zeros(T, H, dtype=torch.bfloat16, device=dev)
ids_s = torch.zeros(T, K, dtype=torch.long, device=dev)
w_s = torch.zeros(T, K, dtype=torch.float32, device=dev)
gid = torch.zeros_like(ids_s); gw = torch.zeros_like(w_s); gad = torch.zeros_like(ids_s)
outs = [torch.zeros(T, H, dtype=torch.float32, device=dev) for _ in range(NS)]

def step(nslots):
    for s in range(nslots):
        outs[s].zero_()
        gh.submit(reg, s, xs, ids_s, w_s, cache_map, source_map, fallback, dev_seq, gid, gw, gad, 0.0)
        gh.wait_add(reg, s, dev_seq, outs[s], 2.0)

def expected(xc, sc, wc):
    cold = sc.clone()
    rm = cache_map.cpu()[sc.long()] >= 0
    cold[rm] = -1
    out = torch.zeros(T, H)
    gh.forward(h0, xc, cold.int(), wc.half(), out, threads)
    return out, rm

def new_inputs():
    xc = (torch.randn(T, H, generator=g) * 0.5).bfloat16()
    sc = torch.randperm(E, generator=g)[:K].view(1, K)
    wc = torch.softmax(torch.randn(T, K, generator=g), -1)
    xs.copy_(xc); ids_s.copy_(sc); w_s.copy_(wc)
    return xc.half(), sc, wc

xc, sc, wc = new_inputs()
step(1); torch.cuda.synchronize()
ref, rm = expected(xc, sc, wc)
err = ((outs[0].cpu() - ref).norm() / ref.norm().clamp_min(1e-9)).item()
gw_ok = torch.equal(gw.cpu()[rm], wc[rm]) and bool((gw.cpu()[~rm] == 0).all())
print(f"[2a] eager bridge: rel err {err:.2e}, resident {int(rm.sum())}/{K}, gpu weight mask ok={gw_ok}, abort={gh.abort_flag(reg)}")
assert err < 1e-5 and gw_ok

graph = torch.cuda.CUDAGraph()
s0 = torch.cuda.Stream()
with torch.cuda.stream(s0):
    step(1)
torch.cuda.synchronize()
with torch.cuda.graph(graph, stream=s0):
    step(1)
worst = 0.0
for it in range(50):
    xc, sc, wc = new_inputs()
    graph.replay(); torch.cuda.synchronize()
    ref, rm = expected(xc, sc, wc)
    worst = max(worst, ((outs[0].cpu() - ref).norm() / ref.norm().clamp_min(1e-9)).item())
print(f"[2b] graph replay x50: worst rel err {worst:.2e}, abort={gh.abort_flag(reg)}, cpu jobs {gh.jobs(reg)}")
assert worst < 1e-5 and gh.abort_flag(reg) == 0

g42 = torch.cuda.CUDAGraph()
with torch.cuda.stream(s0):
    step(NS)
torch.cuda.synchronize()
with torch.cuda.graph(g42, stream=s0):
    step(NS)
for _ in range(5): g42.replay()
torch.cuda.synchronize()
gh.stats(reg, True)
t = time.perf_counter(); n = 40
for _ in range(n):
    new_inputs(); g42.replay()
torch.cuda.synchronize()
dt = (time.perf_counter() - t) / n
cold = int((cache_map.cpu()[ids_s.cpu().long()] < 0).sum())
print(f"[3] 42-layer token (one graph replay, {cold} cold of {K} per layer, {threads} threads): "
      f"{dt*1e3:.2f} ms/token -> {1/dt:.1f} tok/s CPU-bound ceiling; abort={gh.abort_flag(reg)}")
jobs, cns, gns = gh.stats(reg, False)
print(f"    worker: {jobs} jobs, compute {cns/jobs/1e3:.0f} us/job, gap (done -> next request) {gns/max(jobs-1,1)/1e3:.0f} us/job")
gh.stop(reg)
print("OK")
