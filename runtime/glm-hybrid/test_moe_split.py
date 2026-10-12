#!/usr/bin/env python3
"""exl3_moe_gemv split into A (resident) + B (fresh) launches with silent-skip tables vs one full launch: the
per-expert outputs y must be bitwise identical (real GLM-5.3 2-bit experts; some positions weight-0 = null)."""
import json, sys
import torch
from safetensors import safe_open
from exllamav3.ext import exllamav3_ext as ext
import glm_fast

model, layer = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 10
idx = json.load(open(f"{model}/model.safetensors.index.json"))["weight_map"]
dev = torch.device("cuda:0")
fh = {}
def get(n):
    f = idx[n]
    if f not in fh: fh[f] = safe_open(f"{model}/{f}", framework="pt", device="cpu")
    return fh[f].get_tensor(n)
pre = f"model.language_model.layers.{layer}.mlp.experts"
P = 8
keep = []
experts = list(range(P))
mats = {}
for e in experts:
    for p in ("gate_proj", "up_proj", "down_proj"):
        mats[(e, p)] = [get(f"{pre}.{e}.{p}.{n}").to(dev).contiguous() if n == "trellis" else get(f"{pre}.{e}.{p}.{n}").half().to(dev).contiguous() for n in ("trellis", "suh", "svh")]
K = mats[(0, "gate_proj")][0].shape[2] // 16
H = mats[(0, "gate_proj")][0].shape[0] * 16
I = mats[(0, "gate_proj")][0].shape[1] * 16
print(f"layer {layer}: K={K} hidden={H} inter={I}")
def ptr(t): return t.data_ptr()
gu = torch.tensor([[ptr(mats[(e, "gate_proj")][c]) for e in experts] + [ptr(mats[(e, "up_proj")][c]) for e in experts] for c in range(3)], dtype=torch.long, device=dev)
dd = torch.tensor([[ptr(mats[(e, "down_proj")][c]) for e in experts] for c in range(3)], dtype=torch.long, device=dev)
ws = torch.zeros(ext.exl3_moe_gemv_ws_ints(), dtype=torch.int32, device=dev)
x = (torch.randn(1, H, device=dev) * 0.3).bfloat16()

def chain(gtr, dtr, gu_buf, act, y):
    ext.exl3_moe_gemv(x, P, P, gtr, gu[1], gu[2], gu_buf, 2 * P, H, I, K, True, ws)
    ext.exl3_moe_swiglu(gu_buf, act, P, 0.0)
    ext.exl3_moe_gemv(act, P, 1, dtr, dd[1], dd[2], y, P, I, H, K, True, ws)

ok = True
for trial in range(6):
    gtr = gu[0].clone(); dtr = dd[0].clone()
    nulls = [1, 6] if trial % 2 else []          # weight-0 (CPU-side) positions
    for i in nulls: gtr[i] = 0; gtr[P + i] = 0; dtr[i] = 0
    fresh = torch.full((P,), -1, dtype=torch.long, device=dev)
    for i in [[3], [0, 5], [7], [2, 4]][trial % 4]: fresh[i] = 100 + i
    y_full = torch.full((P, H), float("nan"), device=dev); gu_buf = torch.empty(2 * P, I, device=dev); act = torch.empty(P, I, dtype=torch.half, device=dev)
    chain(gtr, dtr, gu_buf, act, y_full)
    ga, gb = torch.empty_like(gtr), torch.empty_like(gtr); da, db = torch.empty_like(dtr), torch.empty_like(dtr)
    glm_fast.moe_split_tables(gtr, dtr, fresh, ga, gb, da, db)
    y_split = torch.full((P, H), float("nan"), device=dev); gu2 = torch.full((2 * P, I), float("nan"), device=dev); act2 = torch.empty(P, I, dtype=torch.half, device=dev)
    chain(ga, da, gu2, act2, y_split)
    chain(gb, db, gu2, act2, y_split)
    torch.cuda.synchronize()
    same = torch.equal(y_full.view(torch.int32), y_split.view(torch.int32))
    print(f"trial {trial}: nulls {nulls} fresh {fresh.ge(0).nonzero().flatten().tolist()} -> y bitwise equal: {same}")
    ok &= same
print("OK" if ok else "FAIL")
