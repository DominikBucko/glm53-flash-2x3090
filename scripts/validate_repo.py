#!/usr/bin/env python3
"""GPU-free consistency checks: overlay manifests, pins, repository hygiene."""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MAX_GIT_FILE = 50 * 1024 * 1024
OVERLAYS = {"vllm_overlay": "runtime/vllm-overlay", "exllamav3": "runtime/exllamav3-overlay",
            "glm_hybrid": "runtime/glm-hybrid"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_overlay(name: str, relative: str, lock: dict, errors: list[str]) -> int:
    root = ROOT / relative
    manifest = root / "SHA256SUMS.json"
    expected = json.loads(manifest.read_text())
    observed = {p.relative_to(root).as_posix(): sha256(p) for p in root.rglob("*")
                if p.is_file() and p != manifest and "__pycache__" not in p.parts}
    if observed != expected:
        missing = sorted(set(expected) - set(observed))[:5]
        extra = sorted(set(observed) - set(expected))[:5]
        changed = sorted(k for k in expected.keys() & observed.keys() if expected[k] != observed[k])[:5]
        errors.append(f"{relative} manifest mismatch: missing={missing} extra={extra} changed={changed}")
    pinned = lock["runtime"][name]
    if sha256(manifest) != pinned["manifest_sha256"]:
        errors.append(f"{relative}/SHA256SUMS.json differs from repro.lock.json")
    count = pinned.get("files", pinned.get("overlay_files"))
    if count != len(expected):
        errors.append(f"{relative}: {len(expected)} files, repro.lock.json says {count}")
    return len(expected)


def main() -> None:
    errors: list[str] = []
    lock = json.loads((ROOT / "repro.lock.json").read_text())
    files = sum(check_overlay(name, rel, lock, errors) for name, rel in OVERLAYS.items())

    dockerfile = (ROOT / "docker/Dockerfile").read_text()
    runtime = lock["runtime"]
    if f"FROM {runtime['container_image']}" not in dockerfile:
        errors.append("Dockerfile base image differs from repro.lock.json")
    if f"EXLLAMAV3_COMMIT={runtime['exllamav3']['commit']}" not in dockerfile:
        errors.append("Dockerfile ExLlamaV3 commit differs from repro.lock.json")
    for package, version in runtime["python_pins"].items():
        if f"{package}=={version}" not in dockerfile:
            errors.append(f"Dockerfile pin for {package} differs from repro.lock.json")

    ckpt = json.loads((ROOT / lock["checkpoint"]["files_lock"]).read_text())
    if ckpt["revision"] != lock["checkpoint"]["quantized"]["revision"] or \
            ckpt["repo"] != lock["checkpoint"]["quantized"]["repo"]:
        errors.append("runtime/model/checkpoint.lock.json differs from repro.lock.json")
    if ckpt["kpool_aux_index"]["integrated_index_sha256"] != lock["checkpoint"]["index_fix"]["integrated_index_sha256"]:
        errors.append("integrated index hash differs between the two locks")

    profile = (ROOT / "configs/2x3090-256k-agent.env").read_text()
    for key in ("MAX_MODEL_LEN", "MAX_NUM_BATCHED_TOKENS", "EXL3_PINNED_FRAC"):
        if not re.search(rf"^{key}=", profile, re.M):
            errors.append(f"profile lacks {key}")
    if f"MAX_MODEL_LEN={runtime['max_model_len']}" not in profile:
        errors.append("profile MAX_MODEL_LEN differs from repro.lock.json")

    forbidden_suffixes = {".safetensors", ".gguf", ".bin", ".pt", ".pth", ".so"}
    secret = re.compile(r"(?:hf_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|ghp_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9]{32,})")
    host_path = re.compile(r"/home/[a-z][a-z0-9_-]*/|/Users/[A-Za-z]")
    for path in ROOT.rglob("*"):
        if ".git" in path.parts or not path.is_file():
            continue
        relative = path.relative_to(ROOT)
        if path.is_symlink():
            errors.append(f"symlink in source repository: {relative}")
        if path.stat().st_size > MAX_GIT_FILE:
            errors.append(f"file exceeds 50 MiB source limit: {relative}")
        if path.suffix.lower() in forbidden_suffixes:
            errors.append(f"binary or model payload must not be committed: {relative}")
        if "__pycache__" in path.parts or path.suffix == ".pyc":
            errors.append(f"Python cache file present: {relative}")
        if path.stat().st_size < 5 * 1024 * 1024:
            try:
                text = path.read_text()
            except UnicodeDecodeError:
                continue
            if secret.search(text):
                errors.append(f"possible credential in {relative}")
            if relative.parts[0] != "runtime" and relative.name != "validate_repo.py" and host_path.search(text):
                errors.append(f"absolute personal host path in {relative}")

    for relative in ("scripts/serve-container.sh", "scripts/docker_serve.sh", "scripts/preflight.sh",
                     "scripts/download_model.sh", "scripts/install_model.py", "scripts/benchmark_serving.py",
                     "scripts/longctx_gate.py", "runtime/glm53_host.py", "runtime/install_overlay.py"):
        if not os.access(ROOT / relative, os.X_OK):
            errors.append(f"script is not executable: {relative}")

    if errors:
        print("repository validation failed:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        raise SystemExit(1)
    total = sum(p.is_file() and ".git" not in p.parts for p in ROOT.rglob("*"))
    print(f"repository validation passed: {files} overlay files, {total} total files")


if __name__ == "__main__":
    main()
