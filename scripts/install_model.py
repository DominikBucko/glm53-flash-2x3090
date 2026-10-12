#!/usr/bin/env python3
"""Verify a download of the pinned EXL3 checkpoint and link its k-pool auxiliary shard into the index.

turboderp/GLM-5.3-Flash-exl3 (3.05bpw) ships the 22 sparse-indexer tensors of the 11 DSA layers in
kpool_aux.safetensors but does not list them in model.safetensors.index.json, so a loader that discovers tensors
through the index silently runs the indexer without them. This adds the entries, keeps the original index as
model.safetensors.index.pre-kpool-aux-link.json and checks the result against runtime/model/checkpoint.lock.json.
No weight file is modified.

  python3 scripts/install_model.py /models/glm53-flash-exl3-3.05bpw            # verify sizes + integrate
  python3 scripts/install_model.py /models/glm53-flash-exl3-3.05bpw --full     # also SHA-256 every file
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

LOCK = Path(__file__).resolve().parents[1] / "runtime/model/checkpoint.lock.json"
INDEX = "model.safetensors.index.json"
BACKUP = "model.safetensors.index.pre-kpool-aux-link.json"
AUX = "kpool_aux.safetensors"
SPARSE_LAYERS = tuple(range(3, 44, 4))
KPOOL_KEYS = sorted(
    f"model.language_model.layers.{layer}.self_attn.indexer.index_kpool_compress_{kind}"
    for layer in SPARSE_LAYERS
    for kind in ("ape", "gate")
)


def safetensors_header(path: Path) -> dict:
    """Tensor table of a .safetensors file (8-byte little-endian length + JSON), without loading tensors."""
    with path.open("rb") as handle:
        length = int.from_bytes(handle.read(8), "little")
        header = json.loads(handle.read(length))
    header.pop("__metadata__", None)
    return header


def integrated_index(index_text: str, aux_header: dict) -> str:
    """The upstream index with the k-pool tensors linked to the auxiliary shard (indent=2, sorted keys)."""
    data = json.loads(index_text)
    if set(KPOOL_KEYS) & set(data["weight_map"]):
        raise ValueError("upstream index already lists k-pool tensors")
    if sorted(aux_header) != KPOOL_KEYS:
        raise ValueError(f"{AUX} does not hold exactly the 22 expected tensors")
    extra = sum(end - start for start, end in (aux_header[k]["data_offsets"] for k in KPOOL_KEYS))
    data["weight_map"].update({k: AUX for k in KPOOL_KEYS})
    data.setdefault("metadata", {})["total_size"] = int(data.get("metadata", {}).get("total_size", 0)) + extra
    return json.dumps(data, indent=2, sort_keys=True) + "\n"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 24), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv: list[str] | None = None, lock_path: Path = LOCK) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model_dir", type=Path)
    ap.add_argument("--full", action="store_true", help="SHA-256 every file (about 117 GiB of reads)")
    args = ap.parse_args(argv)
    lock = json.loads(lock_path.read_text())
    root = args.model_dir
    integrated = lock["kpool_aux_index"]["integrated_index_sha256"]

    index = root / INDEX
    backup = root / BACKUP
    if backup.exists() and sha256(index) == integrated:
        upstream_index = backup
        state = "already integrated"
    else:
        upstream_index = index
        state = "integrating"

    bad = []
    for name, meta in lock["files"].items():
        path = upstream_index if name == INDEX else root / name
        if not path.is_file():
            bad.append(f"missing: {name}")
        elif path.stat().st_size != meta["size"]:
            bad.append(f"size mismatch: {name}")
    if bad:
        print("\n".join(bad), file=sys.stderr)
        return 1

    to_hash = [n for n in lock["files"] if args.full or n == INDEX or n == AUX or not n.endswith(".safetensors")]

    def check(name: str) -> str | None:
        path = upstream_index if name == INDEX else root / name
        return None if sha256(path) == lock["files"][name]["sha256"] else name

    with ThreadPoolExecutor(8) as pool:
        mismatched = [n for n in pool.map(check, to_hash) if n]
    if mismatched:
        print(f"SHA-256 mismatch against {lock['repo']}@{lock['revision'][:8]}: {mismatched}", file=sys.stderr)
        return 1
    print(f"verified {len(to_hash)} of {len(lock['files'])} files by SHA-256, all sizes match ({state})")

    if state == "integrating":
        try:
            text = integrated_index(index.read_text(), safetensors_header(root / AUX))
        except ValueError as exc:
            print(f"refusing to modify the index: {exc}", file=sys.stderr)
            return 1
        shutil.copy2(index, backup)
        tmp = index.with_suffix(".json.tmp")
        tmp.write_text(text)
        tmp.replace(index)

    got = sha256(index)
    if got != integrated:
        print(f"integrated index sha256 {got} != expected {integrated}", file=sys.stderr)
        return 1
    print(f"index OK: {len(KPOOL_KEYS)} k-pool tensors linked to {AUX} (sha256 {got[:16]}...)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
