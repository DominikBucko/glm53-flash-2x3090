#!/usr/bin/env bash
# Download the pinned EXL3 checkpoint (turboderp/GLM-5.3-Flash-exl3, 3.05bpw branch, about 117 GiB), verify it
# against runtime/model/checkpoint.lock.json and link its k-pool auxiliary shard into the index.
#   scripts/download_model.sh /models/glm53-flash-exl3-3.05bpw
set -euo pipefail

root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
dest=${1:-${MODEL_DIR:?usage: scripts/download_model.sh <directory> (or set MODEL_DIR)}}
lock="$root/runtime/model/checkpoint.lock.json"
read -r repo revision < <(python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); print(d["repo"], d["revision"])' "$lock")

command -v hf >/dev/null || { echo "install the Hugging Face CLI first: pip install -U huggingface_hub" >&2; exit 2; }
mkdir -p "$dest"
# The index is downloaded as published; install_model.py keeps that copy and writes the linked one.
if [[ -f "$dest/model.safetensors.index.pre-kpool-aux-link.json" ]]; then
  hf download "$repo" --revision "$revision" --local-dir "$dest" --exclude model.safetensors.index.json
else
  hf download "$repo" --revision "$revision" --local-dir "$dest"
fi
python3 "$root/scripts/install_model.py" "$dest" ${FULL_VERIFY:+--full}
echo "checkpoint ready: MODEL_DIR=$dest"
