#!/usr/bin/env python3
"""Write runtime/model/checkpoint.lock.json: every file of the upstream EXL3 checkpoint at a pinned revision.

Sizes and SHA-256 come from the Hub (LFS sha256; small files are hashed from a local copy that must match the
Hub's git blob id). scripts/install_model.py verifies a download against this lock before patching the index.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from huggingface_hub import HfApi


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="turboderp/GLM-5.3-Flash-exl3")
    ap.add_argument("--revision", required=True, help="full 40-hex commit of the 3.05bpw branch")
    ap.add_argument("--local", type=Path, required=True, help="downloaded copy used to hash non-LFS files")
    ap.add_argument("--integrated-index-sha256", required=True)
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[1] / "runtime/model/checkpoint.lock.json")
    args = ap.parse_args()

    files = {}
    for entry in HfApi().list_repo_tree(args.repo, revision=args.revision, recursive=True):
        if getattr(entry, "size", None) is None:
            continue
        if entry.lfs is not None:
            digest = entry.lfs.sha256
        else:
            data = (args.local / entry.path).read_bytes()
            blob = hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()
            if blob != entry.blob_id:
                raise SystemExit(f"local {entry.path} does not match the Hub blob")
            digest = hashlib.sha256(data).hexdigest()
        files[entry.path] = {"size": entry.size, "sha256": digest}
    lock = {
        "schema_version": 1,
        "repo": args.repo,
        "branch": "3.05bpw",
        "revision": args.revision,
        "files": dict(sorted(files.items())),
        "kpool_aux_index": {
            "note": "The upstream index omits the 22 sparse-indexer tensors stored in kpool_aux.safetensors; "
                    "scripts/install_model.py adds them (json indent=2, sort_keys) and checks the result.",
            "integrated_index_sha256": args.integrated_index_sha256,
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(lock, indent=1) + "\n")
    print(f"{args.out}: {len(files)} files, {sum(f['size'] for f in files.values()) / 2**30:.2f} GiB")


if __name__ == "__main__":
    main()
