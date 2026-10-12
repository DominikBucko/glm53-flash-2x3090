#!/usr/bin/env bash
set -euo pipefail

image=${IMAGE:-glm53-flash-2x3090:locked}
model_dir=${MODEL_DIR:?Set MODEL_DIR to the downloaded checkpoint (scripts/download_model.sh)}
port=${PORT:-8000}

docker_env=(-e "PORT=$port")
# Profile variables set in .env override the image's baked profile (configs/2x3090-256k-agent.env).
while IFS= read -r name; do
  docker_env+=(-e "$name=${!name}")
done < <(
  {
    compgen -e | grep -E '^(GLM53_|EXL3_|VLLM_)[A-Z0-9_]+$' || true
    for name in SERVED_MODEL_NAME MAX_MODEL_LEN MAX_NUM_SEQS MAX_NUM_BATCHED_TOKENS KV_CACHE_MEMORY_BYTES \
                KV_CACHE_DTYPE GPU_MEMORY_UTILIZATION CPU_OFFLOAD_GB CUDA_GRAPH_MODE PREFIX_CACHING ENFORCE_EAGER \
                HYBRID_MAIN_CPUS OMP_NUM_THREADS NCCL_P2P_DISABLE; do
      declare -p "$name" &>/dev/null && echo "$name"
    done
  } | sort -u
)

model_dir=$(cd -- "$model_dir" && pwd -P)
[[ -f "$model_dir/model.safetensors.index.json" ]] || {
  echo "MODEL_DIR is not a model checkpoint: $model_dir" >&2
  exit 2
}

# EXTRA_DOCKER_ARGS (word-split) is appended to docker run, e.g. EXTRA_DOCKER_ARGS="--cpuset-cpus 0-63".
# Optional persistent Triton / TileLang JIT caches: avoids recompiling kernels on every start.
cache_mounts=()
if [[ -n "${JIT_CACHE_DIR:-}" ]]; then
  mkdir -p "$JIT_CACHE_DIR/triton" "$JIT_CACHE_DIR/tilelang"
  jit_cache_dir=$(cd -- "$JIT_CACHE_DIR" && pwd -P)
  cache_mounts=(-v "$jit_cache_dir/triton:/root/.triton" -v "$jit_cache_dir/tilelang:/root/.tilelang")
fi

exec docker run --rm \
  --name glm53-flash \
  --gpus all \
  --ipc host \
  --cap-add SYS_PTRACE \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -p "127.0.0.1:$port:$port" \
  "${docker_env[@]}" \
  ${cache_mounts[@]+"${cache_mounts[@]}"} \
  -v "$model_dir:/model:ro" \
  ${EXTRA_DOCKER_ARGS:-} \
  "$image" /model
