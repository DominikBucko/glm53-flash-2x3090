#!/usr/bin/env python3
"""glm_hybrid.submit_lru (fused) vs glm_hybrid.submit + vLLM's triton _update_exl3_lru_map_kernel: every output must
be identical (GPU split tensors, host slot request area, cache_map, slot ids/ages, clock, miss lists, counters) over
randomized LRU states, T in {1,2,4}, frac in {0, 0.3, 0.5, 1}. Then times both sequences in a CUDA graph."""
import torch, triton
import glm_hybrid as gh
from vllm.model_executor.layers.quantization.exl3 import _update_exl3_lru_map_kernel

dev = torch.device("cuda:0")
E, H, K = 288, 4096, 8
reg = gh.create(4, H)
g = torch.Generator(device="cpu").manual_seed(0)

def make_state(cap):
    perm = torch.randperm(E, generator=g)
    slot_gid = perm[:cap].clone()
    nfree = int(torch.randint(0, 3, (1,), generator=g))
    if nfree: slot_gid[-nfree:] = -1
    cache_map = torch.full((E,), -1, dtype=torch.long)
    for s, e in enumerate(slot_gid.tolist()):
        if e >= 0: cache_map[e] = s
    source_map = torch.arange(E, dtype=torch.long)
    nonlocal_ids = torch.randperm(E, generator=g)[:5]
    source_map[nonlocal_ids] = -1
    cache_map[nonlocal_ids] = -1
    slot_gid[(slot_gid.unsqueeze(1) == nonlocal_ids.unsqueeze(0)).any(1)] = -1
    ages = torch.randperm(10 * cap, generator=g)[:cap].long() + 1
    clock = torch.tensor([20 * cap], dtype=torch.long)
    st = dict(source_map=source_map, cache_map=cache_map, slot_gid=slot_gid, slot_age=ages, clock=clock,
              miss_local=torch.full((64,), -7, dtype=torch.long), miss_slot=torch.full((64,), -7, dtype=torch.long),
              counters=torch.tensor([3, 4], dtype=torch.long))
    return {k: v.to(dev) for k, v in st.items()}

def clone(st): return {k: v.clone() for k, v in st.items()}

def run_old(st, slot, x, ids, w, fb, seq, frac):
    gid, gw, gad = torch.empty_like(ids), torch.empty_like(w), torch.empty_like(ids)
    gh.submit(reg, slot, x, ids, w, st["cache_map"], st["source_map"], fb, seq, gid, gw, gad, frac)
    n = ids.numel(); cap = st["slot_gid"].numel()
    _update_exl3_lru_map_kernel[(1,)](gad.reshape(-1), st["source_map"], st["cache_map"], st["slot_gid"], st["slot_age"],
        st["clock"], st["miss_local"], st["miss_slot"], st["counters"], num_ids=n, global_num_experts=E, capacity=cap,
        id_block=triton.next_power_of_2(n), capacity_block=triton.next_power_of_2(cap), num_warps=4)
    return gid, gw, gad

def run_new(st, slot, x, ids, w, fb, seq, frac):
    gid, gw, gad = torch.empty_like(ids), torch.empty_like(w), torch.empty_like(ids)
    ok = gh.submit_lru(reg, slot, x, ids, w, fb, seq, gid, gw, gad, frac, st["source_map"], st["cache_map"],
                       st["slot_gid"], st["slot_age"], st["clock"], st["miss_local"], st["miss_slot"], st["counters"],
                       st["slot_gid"].numel())
    assert ok
    return gid, gw, gad

bad = 0; cases = 0
for trial in range(300):
    cap = [16, 60, 120, 200][trial % 4]
    T = [1, 1, 2, 4][trial % 4 if trial % 3 else 0]
    frac = [0.0, 0.3, 0.5, 1.0][(trial // 4) % 4]
    st = make_state(cap)
    # routed ids: distinct per row, mixing residents / cold / non-local
    rows = [torch.randperm(E, generator=g)[:K] for _ in range(T)]
    ids = torch.stack(rows).to(dev)
    w = torch.softmax(torch.randn(T, K, generator=g), -1).to(dev)
    x = (torch.randn(T, H, generator=g) * 0.5).bfloat16().to(dev)
    res = (st["cache_map"] >= 0).nonzero().flatten()
    fb = res[:1].clone() if res.numel() else torch.zeros(1, dtype=torch.long, device=dev)
    seq_a = torch.zeros(4, dtype=torch.int32, device=dev); seq_b = seq_a.clone()
    A, B = clone(st), clone(st)
    oa = run_old(A, 0, x, ids, w, fb, seq_a, frac)
    da = gh.slot_dump(reg, 0, T, K, H)
    ob = run_new(B, 1, x, ids, w, fb, seq_b, frac)
    torch.cuda.synchronize()
    db = gh.slot_dump(reg, 1, T, K, H)
    n = T * K
    diffs = []
    for name, a, b in zip(("gpu_ids", "gpu_w", "gpu_admit"), oa, ob):
        if not torch.equal(a, b): diffs.append(name)
    for k in ("cache_map", "slot_gid", "slot_age", "clock", "counters"):
        if not torch.equal(A[k], B[k]): diffs.append(k)
    for k in ("miss_local", "miss_slot"):
        if not torch.equal(A[k][:n], B[k][:n]): diffs.append(k)
    for name, a, b in zip(("x", "sel", "w"), da[:3], db[:3]):
        if not torch.equal(a.view(torch.int16) if a.dtype == torch.half else a, b.view(torch.int16) if b.dtype == torch.half else b): diffs.append("slot_" + name)
    if da[3][:2].tolist() != db[3][:2].tolist() or int(da[3][2]) != int(db[3][2]): diffs.append("slot_meta")
    cases += 1
    if diffs:
        bad += 1
        if bad <= 5: print(f"trial {trial} cap={cap} T={T} frac={frac}: differs in {diffs}")
print(f"{cases} cases, {bad} mismatching")

# timing (T=1, frac 0.3, cap 120) inside a CUDA graph, 42 calls
st = make_state(120)
ids = torch.randperm(E, generator=g)[:K].view(1, K).to(dev)
w = torch.softmax(torch.randn(1, K, generator=g), -1).to(dev)
x = torch.randn(1, H, generator=g).bfloat16().to(dev)
fb = (st["cache_map"] >= 0).nonzero().flatten()[:1].clone()
seq = torch.zeros(4, dtype=torch.int32, device=dev)
def bench(fn):
    S = clone(st)
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        for _ in range(2): fn(S, 2, x, ids, w, fb, seq, 0.3)
    torch.cuda.synchronize()
    gr = torch.cuda.CUDAGraph()
    with torch.cuda.graph(gr, stream=s):
        for _ in range(42): fn(S, 2, x, ids, w, fb, seq, 0.3)
    gr.replay(); torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    e0.record()
    for _ in range(20): gr.replay()
    e1.record(); torch.cuda.synchronize()
    return e0.elapsed_time(e1) * 1e3 / (20 * 42)
print(f"submit + triton LRU: {bench(run_old):.1f} us/layer; fused submit_lru: {bench(run_new):.1f} us/layer")
print("OK" if bad == 0 else "FAIL")
