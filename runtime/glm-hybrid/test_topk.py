#!/usr/bin/env python3
"""glm_fast.topk_sorted vs torch.topk (+ isfinite mask) as used by the GLM-5.3 indexer decode path.

Checks the selected set (ascending, -1 padded) on: random logits with a -inf tail (live context shorter than the
logits width), fully finite rows, heavy ties, fewer finite entries than k, multiple rows. Then times both paths
inside CUDA graphs at the 64K-ctx (W=16384 pools) and 256K-ctx (W=65536 pools) widths, k=512.
"""
import torch
import glm_fast

dev = torch.device("cuda:0")
g = torch.Generator(device=dev).manual_seed(0)
K = 512
ws = torch.zeros(16 * glm_fast.topk_ws_ints_per_row(), dtype=torch.int32, device=dev)

def torch_path(logits, k):
    values, indices = torch.topk(logits, k, dim=1)
    indices = indices.to(torch.int32)
    indices.masked_fill_(~torch.isfinite(values), -1)
    return indices

def fast_path(logits, k):
    out = torch.empty(logits.shape[0], k, dtype=torch.int32, device=dev)
    glm_fast.topk_sorted(logits, k, out, ws)
    return out

def canon(ids):
    big = torch.iinfo(torch.int32).max
    keyed = torch.where(ids < 0, torch.full_like(ids, big), ids)
    keyed = keyed.sort(dim=1).values
    return torch.where(keyed == big, torch.full_like(keyed, -1), keyed)

def check(name, logits, k=K):
    ref = canon(torch_path(logits, k))
    got = fast_path(logits, k)
    sorted_ok = bool((canon(got) == got).all())
    if torch.equal(ref, got):
        print(f"  {name}: identical ({int((got >= 0).sum())} selected)")
        return True
    # ties at the k-th value may legitimately pick different ids: compare selected values instead
    ok = sorted_ok
    for r in range(logits.shape[0]):
        rv = logits[r][ref[r][ref[r] >= 0].long()].sort().values
        gv = logits[r][got[r][got[r] >= 0].long()].sort().values
        ok &= torch.equal(rv, gv) and int((got[r] >= 0).sum()) == int((ref[r] >= 0).sum())
    print(f"  {name}: ids differ only on ties -> values equal={ok}, ascending={sorted_ok}")
    return ok

ok = True
for W in (16384, 65536):
    print(f"W={W}")
    x = torch.randn(1, W, device=dev, generator=g) * 3
    x[:, 9000:] = float("-inf")
    ok &= check("random, -inf tail", x)
    ok &= check("random, all finite", torch.randn(1, W, device=dev, generator=g).abs() * 50 + 100)
    ok &= check("heavy ties", torch.randint(0, 7, (1, W), device=dev, generator=g).float())
    y = torch.full((1, W), float("-inf"), device=dev); y[0, :300] = torch.randn(300, device=dev, generator=g)
    ok &= check("300 finite < k", y)
    z = torch.randn(4, W, device=dev, generator=g); z[1, 5000:] = float("-inf"); z[2] = z[2].round()
    ok &= check("4 rows mixed", z)
    for _ in range(20):
        ok &= check("random repeat", torch.randn(2, W, device=dev, generator=g) * torch.rand(1, device=dev, generator=g) * 10, )

def bench(fnc, x, n=200):
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        for _ in range(3): fnc(x, K)
    torch.cuda.synchronize()
    gr = torch.cuda.CUDAGraph()
    with torch.cuda.graph(gr, stream=s):
        for _ in range(20): fnc(x, K)
    gr.replay(); torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    e0.record()
    for _ in range(n // 20): gr.replay()
    e1.record(); torch.cuda.synchronize()
    return e0.elapsed_time(e1) * 1e3 / n

for W, live in ((16384, 9000), (65536, 9000), (65536, 60000)):
    x = torch.randn(1, W, device=dev, generator=g); x[:, live:] = float("-inf")
    print(f"W={W} live={live}: torch.topk path {bench(torch_path, x):.1f} us, fast {bench(fast_path, x):.1f} us")
print("OK" if ok else "FAIL")
