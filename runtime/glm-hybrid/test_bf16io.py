#!/usr/bin/env python3
"""BC_LinearEXL3.run_bf16 vs the vLLM fp16 sequence  x.half() -> run_alloc (fp16) -> .to(bf16): bitwise equality on
real GLM-5.3 attention/shared-expert tensors (TP2 halves as vLLM shards them are not needed for the kernel check:
full tensors exercise the same code), plus graph-replay timing of both sequences.
usage: test_bf16io.py <model_dir>"""
import json, sys
import torch
from safetensors import safe_open
from exllamav3.ext import exllamav3_ext as ext

model = sys.argv[1]
idx = json.load(open(f"{model}/model.safetensors.index.json"))["weight_map"]
dev = torch.device("cuda:0")
def get(n):
    with safe_open(f"{model}/{idx[n]}", framework="pt", device="cpu") as f:
        return f.get_tensor(n)

def bc_for(prefix):
    tr = get(prefix + ".trellis").to(dev)
    suh = get(prefix + ".suh").half().to(dev)
    svh = get(prefix + ".svh").half().to(dev)
    K = tr.shape[2] // 16
    mul1 = (prefix + ".mul1") in idx
    mcg = (prefix + ".mcg") in idx
    xh = torch.empty(1, tr.shape[0] * 16, dtype=torch.half, device=dev)
    return ext.BC_LinearEXL3(tr, suh, svh, K, None, mcg, mul1, xh), tr.shape[0] * 16, tr.shape[1] * 16, K, mul1

names = [
    "model.language_model.layers.0.self_attn.qkv_proj",
    "model.language_model.layers.0.self_attn.o_proj",
    "model.language_model.layers.3.self_attn.q_a_proj",
    "model.language_model.layers.3.self_attn.q_b_proj",
    "model.language_model.layers.3.self_attn.o_proj",
    "model.language_model.layers.3.mlp.shared_experts.gate_proj",
    "model.language_model.layers.3.mlp.shared_experts.down_proj",
]
g = torch.Generator(device=dev).manual_seed(0)
ok = True
for n in names:
    if n + ".trellis" not in idx:
        print(f"skip {n}"); continue
    bc, k, nout, K, mul1 = bc_for(n)
    worst = 0
    for trial in range(20):
        scale = [1e-3, 0.05, 1.0, 30.0][trial % 4]
        x = (torch.randn(1, k, device=dev, generator=g) * scale).bfloat16()
        ref = bc.run_alloc(x.half(), nout, False).to(torch.bfloat16)
        y = torch.empty(1, nout, dtype=torch.bfloat16, device=dev)
        handled = bc.run_bf16(x, y)
        if not handled:
            print(f"{n.split('layers.')[1]}: K={K} mul1={mul1} k={k} n={nout} -> not handled (fallback)")
            break
        diff = int((y.view(torch.int16) != ref.view(torch.int16)).sum())
        worst = max(worst, diff)
    else:
        print(f"{n.split('layers.')[1]}: K={K} k={k} n={nout} -> bitwise mismatches (worst of 20) = {worst}")
        ok &= worst == 0
    if handled:
        x = (torch.randn(1, k, device=dev, generator=g) * 0.05).bfloat16()
        y = torch.empty(1, nout, dtype=torch.bfloat16, device=dev)
        def old(): return bc.run_alloc(x.half(), nout, False).to(torch.bfloat16)
        def new(): bc.run_bf16(x, y)
        def bench(f, reps=30):
            s = torch.cuda.Stream()
            with torch.cuda.stream(s):
                for _ in range(3): f()
            torch.cuda.synchronize()
            gr = torch.cuda.CUDAGraph()
            with torch.cuda.graph(gr, stream=s):
                for _ in range(20): f()
            gr.replay(); torch.cuda.synchronize()
            e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            e0.record()
            for _ in range(reps): gr.replay()
            e1.record(); torch.cuda.synchronize()
            return e0.elapsed_time(e1) * 1e3 / (reps * 20)
        print(f"      graph replay: half()+gemv+to(bf16) {bench(old):.1f} us, run_bf16 {bench(new):.1f} us")
print("OK" if ok else "FAIL")
