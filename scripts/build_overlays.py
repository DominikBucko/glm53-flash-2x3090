#!/usr/bin/env python3
"""Regenerate runtime/vllm-overlay, runtime/exllamav3-overlay and runtime/glm-hybrid from the research trees.

vllm-overlay      every non-binary file of the research vLLM tree whose content differs from the vendor wheel
                  vllm==0.1.dev20073+g8e685d198 (per the wheel's RECORD), plus files the wheel lacks. The research
                  tree is spele1100/vllm-glm53@f2935ce32 with the vendor's day-0 files and this project's changes;
                  its compiled extensions must be byte-identical to the wheel's (checked here).
exllamav3-overlay files that differ between turboderp-org/exllamav3@<base> and the research branch.
glm-hybrid        sources of the glm_hybrid (CPU experts + CUDA bridge) and glm_fast (fused kernels) extensions.

Each overlay gets a SHA256SUMS.json ({relative path: sha256}) that install_overlay.py and validate_repo.py check.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

SKIP_PARTS = ("__pycache__",)
SKIP_MARKERS = (".before-", ".pre-", ".bak", ".orig", ".swp", ".kate-swp", "~")
BINARY_SUFFIXES = (".so", ".pyd", ".dylib")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def record_digest(hexdigest: str) -> str:
    return base64.urlsafe_b64encode(bytes.fromhex(hexdigest)).rstrip(b"=").decode()


def skipped(rel: str) -> bool:
    return any(p in rel.split("/") for p in SKIP_PARTS) or any(m in rel for m in SKIP_MARKERS) or rel.endswith(".pyc")


def write_manifest(root: Path) -> dict[str, str]:
    sums = {
        p.relative_to(root).as_posix(): sha256_file(p)
        for p in sorted(root.rglob("*"))
        if p.is_file() and p.name != "SHA256SUMS.json"
    }
    (root / "SHA256SUMS.json").write_text(json.dumps(sums, indent=1, sort_keys=True) + "\n")
    return sums


def build_vllm(tree: Path, record: Path, out: Path) -> None:
    wheel = {}
    for line in record.read_text().splitlines():
        path, digest, _size = line.rsplit(",", 2)
        if path.startswith("vllm/") and digest.startswith("sha256="):
            wheel[path[len("vllm/"):]] = digest[len("sha256="):]
    if out.exists():
        shutil.rmtree(out)
    changed = new = same = 0
    for path in sorted(tree.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(tree).as_posix()
        if skipped(rel):
            continue
        digest = record_digest(sha256_file(path))
        if rel.endswith(BINARY_SUFFIXES):
            if wheel.get(rel) != digest:
                raise SystemExit(f"compiled extension differs from the vendor wheel: {rel}")
            continue
        if wheel.get(rel) == digest:
            same += 1
            continue
        if rel in wheel:
            changed += 1
        else:
            new += 1
        dest = out / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dest)
    missing = sorted(r for r in wheel if r.endswith(".py") and not (tree / r).exists())
    if missing:
        raise SystemExit(f"{len(missing)} vendor files missing from the research tree, e.g. {missing[:3]}")
    sums = write_manifest(out)
    print(f"vllm-overlay: {len(sums)} files ({changed} changed, {new} new; {same} identical to the wheel)")


def build_exllamav3(repo: Path, base: str, out: Path) -> None:
    status = subprocess.run(["git", "-C", repo, "status", "--porcelain", "--untracked-files=no"],
                            check=True, capture_output=True, text=True).stdout
    if status.strip():
        raise SystemExit(f"exllamav3 research tree has uncommitted changes:\n{status}")
    diff = subprocess.run(["git", "-C", repo, "diff", "--name-status", "--no-renames", base, "HEAD"],
                          check=True, capture_output=True, text=True).stdout.split("\n")
    if out.exists():
        shutil.rmtree(out)
    deleted = []
    for line in filter(None, diff):
        kind, rel = line.split("\t", 1)
        if kind == "D":
            deleted.append(rel)
            continue
        dest = out / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(subprocess.run(["git", "-C", repo, "show", f"HEAD:{rel}"],
                                        check=True, capture_output=True).stdout)
    if deleted:
        raise SystemExit(f"overlay cannot express deletions yet: {deleted}")
    head = subprocess.run(["git", "-C", repo, "rev-parse", "HEAD"], check=True, capture_output=True,
                          text=True).stdout.strip()
    sums = write_manifest(out)
    print(f"exllamav3-overlay: {len(sums)} files over {base[:9]} (research head {head[:9]})")


def build_glm_hybrid(src: Path, out: Path) -> None:
    if out.exists():
        shutil.rmtree(out)
    keep = ["setup.py", "setup_fast.py", "csrc/bindings.cpp", "csrc/moe_mul1.cpp", "csrc/moe_mul1.h",
            "csrc/hybrid_bridge.cu", "csrc/fast_bindings.cpp", "csrc/glm_fast.cu"]
    keep += sorted(p.name for p in src.glob("test_*.py"))
    for rel in keep:
        dest = out / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src / rel, dest)
    sums = write_manifest(out)
    print(f"glm-hybrid: {len(sums)} files")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vllm-tree", type=Path, required=True, help="research tree's vllm/ package directory")
    ap.add_argument("--vendor-record", type=Path, required=True, help="RECORD of the vendor vllm wheel")
    ap.add_argument("--exllamav3-repo", type=Path, required=True)
    ap.add_argument("--exllamav3-base", required=True)
    ap.add_argument("--glm-hybrid", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[1] / "runtime")
    args = ap.parse_args()
    build_vllm(args.vllm_tree, args.vendor_record, args.out / "vllm-overlay")
    build_exllamav3(args.exllamav3_repo, args.exllamav3_base, args.out / "exllamav3-overlay")
    build_glm_hybrid(args.glm_hybrid, args.out / "glm-hybrid")


if __name__ == "__main__":
    main()
