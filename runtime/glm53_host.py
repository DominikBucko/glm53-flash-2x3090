#!/usr/bin/env python3
"""CPU plan for the hybrid CPU+GPU expert decode: one CPU expert pool per GPU plus the cores for everything else.

During decode each GPU hands the experts that are not cached in its VRAM to a pool of CPU threads. A pool
synchronises with spin barriers, so it must own whole physical cores: any other busy thread on a pool core or on
its SMT sibling stalls the whole pool. The plan therefore:

  * uses one hardware thread per physical core for the pools and leaves the pool cores' SMT siblings idle;
  * spreads each pool over every L3 domain (CCD), because each CCD has its own link to memory and expert
    weights are read at memory bandwidth;
  * splits every L3 domain into two halves: rank 0 takes the first half and rank 1 the second, minus the last
    core of each half, which goes to the main set (vLLM's API server, engine core, GPU workers) together with
    its SMT sibling.

On a Threadripper PRO 5975WX (4 CCDs x 8 cores, siblings n and n + 32) this gives exactly the hand-tuned sets
the published numbers were measured with: rank 0 = 0-2, 8-10, 16-18, 24-26; rank 1 = 4-6, 12-14, 20-22, 28-30;
main = 3, 7, 11, ..., 31 and 35, 39, ..., 63.

  python3 glm53_host.py          # human-readable plan
  python3 glm53_host.py --env    # KEY=VALUE lines for the launcher

Explicitly set EXL3_HYBRID_CPUS_R0 / EXL3_HYBRID_CPUS_R1 / HYBRID_MAIN_CPUS / EXL3_HYBRID_THREADS win.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def parse_list(text: str) -> list[int]:
    out: list[int] = []
    for part in text.strip().split(","):
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-")
            out.extend(range(int(lo), int(hi) + 1))
        else:
            out.append(int(part))
    return out


def fmt(cpus: list[int]) -> str:
    return ",".join(str(c) for c in cpus)


def topology(sys_root: Path, allowed: set[int]) -> list[list[list[int]]]:
    """L3 domains -> physical cores -> hardware threads, restricted to allowed CPUs, sorted."""
    cpu_dir = sys_root / "devices/system/cpu"
    cores: dict[int, list[int]] = {}
    l3_of: dict[int, tuple[int, ...]] = {}
    for path in cpu_dir.glob("cpu[0-9]*"):
        cpu = int(path.name[3:])
        if cpu not in allowed:
            continue
        siblings = [c for c in parse_list((path / "topology/thread_siblings_list").read_text()) if c in allowed]
        first = min(siblings)
        cores[first] = siblings
        l3 = tuple(siblings)
        for index in sorted((path / "cache").glob("index[0-9]*")):
            if (index / "level").read_text().strip() == "3":
                l3 = tuple(parse_list((index / "shared_cpu_list").read_text()))
                break
        l3_of[first] = l3
    domains: dict[tuple[int, ...], list[list[int]]] = {}
    for first, siblings in cores.items():
        domains.setdefault(l3_of[first], []).append(sorted(siblings))
    return [sorted(group) for _, group in sorted(domains.items(), key=lambda kv: min(kv[0]))]


def plan(domains: list[list[list[int]]]) -> dict[str, list[int] | int]:
    r0: list[int] = []
    r1: list[int] = []
    main: list[int] = []
    for cores in domains:
        half = (len(cores) + 1) // 2
        for pool, part in ((r0, cores[:half]), (r1, cores[half:])):
            if len(part) >= 2:
                pool.extend(core[0] for core in part[:-1])
                main.extend(part[-1])
            else:
                main.extend(t for core in part for t in core)
    if not r0 or not r1:
        raise SystemExit("not enough physical cores for two CPU expert pools (need at least 4 per L3 domain "
                         "or 4 in total); set EXL3_HYBRID_CPUS_R0/R1 and HYBRID_MAIN_CPUS by hand")
    return {
        "EXL3_HYBRID_CPUS_R0": sorted(r0),
        "EXL3_HYBRID_CPUS_R1": sorted(r1),
        "HYBRID_MAIN_CPUS": sorted(main),
        "EXL3_HYBRID_THREADS": min(len(r0), len(r1)),
    }


def resolve(sys_root: Path = Path("/sys"), allowed: set[int] | None = None,
            env: dict[str, str] | None = None) -> dict[str, str]:
    env = dict(os.environ) if env is None else env
    allowed = set(os.sched_getaffinity(0)) if allowed is None else allowed
    result = {k: (fmt(v) if isinstance(v, list) else str(v)) for k, v in plan(topology(sys_root, allowed)).items()}
    for key in result:
        if env.get(key):
            result[key] = env[key]
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--env", action="store_true", help="print KEY=VALUE lines")
    ap.add_argument("--sys-root", type=Path, default=Path("/sys"))
    args = ap.parse_args()
    result = resolve(args.sys_root)
    if args.env:
        for key, value in result.items():
            print(f"{key}={value}")
    else:
        print(f"CPU expert pool, GPU 0: {result['EXL3_HYBRID_CPUS_R0']}")
        print(f"CPU expert pool, GPU 1: {result['EXL3_HYBRID_CPUS_R1']}")
        print(f"threads per pool: {result['EXL3_HYBRID_THREADS']}")
        print(f"vLLM processes: {result['HYBRID_MAIN_CPUS']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
