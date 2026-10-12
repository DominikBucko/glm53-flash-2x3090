#!/usr/bin/env bash
# Container entrypoint (/usr/local/bin/glm53-serve): load the baked profile, plan the CPU expert pools, check the
# checkpoint and exec vLLM. Variables already set in the environment override the profile.
set -euo pipefail

profile=${GLM53_PROFILE:-/opt/glm53/configs/2x3090-256k-agent.env}
[[ -f "$profile" ]] || { echo "missing runtime profile: $profile" >&2; exit 2; }
while IFS= read -r line || [[ -n "$line" ]]; do
  [[ "$line" =~ ^([A-Z_][A-Z0-9_]*)=(.*)$ ]] || continue
  key=${BASH_REMATCH[1]}
  [[ -n "${!key+x}" ]] || export "$key=${BASH_REMATCH[2]}"
done < "$profile"

model=${1:-/model}
[[ -f "$model/model.safetensors.index.json" && -f "$model/quantization_config.json" ]] || {
  echo "EXL3 checkpoint not found at $model" >&2
  exit 2
}
grep -q '"kpool_aux.safetensors"' "$model/model.safetensors.index.json" || {
  echo "$model/model.safetensors.index.json does not list kpool_aux.safetensors: run scripts/install_model.py on the" \
       "download first (without it the sparse-attention indexer runs without its k-pool weights)" >&2
  exit 2
}
for file in "$EXL3_LRU_CAPACITY_PLAN" "$EXL3_VMM_HOT_RANKINGS" "$CHAT_TEMPLATE"; do
  [[ -f "$file" ]] || { echo "missing runtime file: $file" >&2; exit 2; }
done

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}
export PYTHONUNBUFFERED=1 MALLOC_ARENA_MAX=4
export NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE:-0} NCCL_IB_DISABLE=1 NCCL_SHM_DISABLE=0
export EXL3_TENSOR_LEDGER="$model/quantization_config.json"
export EXL3_MIXED_VMM_HOT_CACHE=0 EXL3_VMM_DIRECT_ALLOC=0

taskset_prefix=()
if [[ "${EXL3_HYBRID_CPU:-0}" == 1 ]]; then
  while IFS='=' read -r key value; do
    export "$key=$value"
  done < <(python3 /opt/glm53/runtime/glm53_host.py --env)
  python3 /opt/glm53/runtime/glm53_host.py | sed 's/^/[glm53-host] /'
  taskset_prefix=(taskset -c "$HYBRID_MAIN_CPUS")
fi

prefix_args=(--no-enable-prefix-caching)
[[ "${PREFIX_CACHING:-1}" == 1 ]] && prefix_args=(--enable-prefix-caching)
eager_args=()
[[ "${ENFORCE_EAGER:-0}" == 1 ]] && eager_args=(--enforce-eager)
api_key_args=()
[[ -n "${VLLM_API_KEY:-}" ]] && api_key_args=(--api-key "$VLLM_API_KEY")

exec ${taskset_prefix[@]+"${taskset_prefix[@]}"} vllm serve "$model" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --host 0.0.0.0 --port "${PORT:-8000}" \
  --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
  --distributed-executor-backend mp \
  --model-impl vllm \
  --quantization exl3 \
  --dtype bfloat16 \
  --max-model-len "$MAX_MODEL_LEN" \
  --max-num-seqs "$MAX_NUM_SEQS" \
  --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
  --enable-chunked-prefill \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --kv-cache-memory "$KV_CACHE_MEMORY_BYTES" \
  --kv-cache-dtype "${KV_CACHE_DTYPE:-auto}" \
  --offload-backend uva --cpu-offload-gb "$CPU_OFFLOAD_GB" --cpu-offload-params experts \
  ${eager_args[@]+"${eager_args[@]}"} \
  --compilation-config "{\"mode\":0,\"cudagraph_mode\":\"$CUDA_GRAPH_MODE\"}" \
  "${prefix_args[@]}" \
  --chat-template "$CHAT_TEMPLATE" \
  --reasoning-config "$REASONING_CONFIG" \
  --no-enable-flashinfer-autotune \
  --language-model-only \
  --skip-mm-profiling \
  --tool-call-parser glm47 \
  --enable-auto-tool-choice \
  --reasoning-parser glm45 \
  ${api_key_args[@]+"${api_key_args[@]}"}
