#!/usr/bin/env bash
set -euo pipefail

fail=0
warn() { printf 'WARN: %s\n' "$*" >&2; }
error() { printf 'ERROR: %s\n' "$*" >&2; fail=1; }

[[ "$(uname -s)" == Linux ]] || error "serving requires Linux"
[[ "$(uname -m)" == x86_64 ]] || error "the packaged image was validated on x86_64"
command -v docker >/dev/null || error "docker is not installed"
command -v nvidia-smi >/dev/null || error "nvidia-smi is not installed"

if command -v nvidia-smi >/dev/null; then
  gpu_count=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l | tr -d ' ')
  [[ "$gpu_count" -ge 2 ]] || error "two CUDA GPUs are required by this profile"
  while IFS=, read -r index name memory cc; do
    (( ${memory%% *} >= 24000 )) || error "GPU $index ($name) has ${memory}; 24 GB per GPU is required"
    [[ "${cc// /}" == 8.6 ]] || warn "GPU $index ($name) is compute capability ${cc// /}; the image is built and validated for 8.6 (RTX 3090)"
  done < <(nvidia-smi --query-gpu=index,name,memory.total,compute_cap --format=csv,noheader,nounits)
  nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
fi

if [[ -r /proc/meminfo ]]; then
  mem_kib=$(awk '/^MemTotal:/ {print $2}' /proc/meminfo)
  [[ "$mem_kib" -ge 125000000 ]] || error "this profile needs approximately 128 GiB RAM (about 110 GiB is used while serving)"
  avail_kib=$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)
  (( avail_kib >= 118000000 )) || warn "only $((avail_kib / 1024 / 1024)) GiB RAM is available now; stop other memory users first (loading pins about 110 GiB)"
  if [[ -d /tmp ]] && [[ "$(findmnt -no FSTYPE /tmp 2>/dev/null || true)" == tmpfs ]]; then
    tmp_used_kib=$(df -k --output=used /tmp | tail -1 | tr -d ' ')
    (( tmp_used_kib < 4194304 )) || warn "/tmp is RAM-backed (tmpfs) and holds $((tmp_used_kib / 1024 / 1024)) GiB; that memory is not available to the server"
  fi
fi

physical_cores=$(lscpu -p=CORE,SOCKET 2>/dev/null | grep -v '^#' | sort -u | wc -l | tr -d ' ')
(( physical_cores >= 12 )) || warn "$physical_cores physical cores; the CPU expert pools were validated with 32 (Threadripper PRO 5975WX)"

if command -v docker >/dev/null && ! docker info >/dev/null 2>&1; then
  error "Docker daemon is unavailable to the current user"
fi

(( fail == 0 )) || exit 1
printf 'preflight passed\n'
