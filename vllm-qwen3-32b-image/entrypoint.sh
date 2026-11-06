#!/usr/bin/env bash
set -e

# Default configurations optimized for 1x NVIDIA A100 GPU
MODEL_PATH="${MODEL_PATH:-/model}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-Qwen3-32B}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
DTYPE="${DTYPE:-bfloat16}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.92}"
PORT="${PORT:-8000}"

ARGS=(
  "--model" "$MODEL_PATH"
  "--served-model-name" "$SERVED_MODEL_NAME"
  "--tensor-parallel-size" "$TENSOR_PARALLEL_SIZE"
  "--max-model-len" "$MAX_MODEL_LEN"
  "--dtype" "$DTYPE"
  "--gpu-memory-utilization" "$GPU_MEMORY_UTILIZATION"
  "--trust-remote-code"
  "--port" "$PORT"
)

# Optional quantization (essential if deploying on A100 40GB: e.g. QUANTIZATION=awq or fp8)
if [ -n "$QUANTIZATION" ]; then
  ARGS+=("--quantization" "$QUANTIZATION")
fi

# Optional KV-cache quantization (e.g. KV_CACHE_DTYPE=fp8)
if [ -n "$KV_CACHE_DTYPE" ]; then
  ARGS+=("--kv-cache-dtype" "$KV_CACHE_DTYPE")
fi

# Optional max number of sequences
if [ -n "$MAX_NUM_SEQS" ]; then
  ARGS+=("--max-num-seqs" "$MAX_NUM_SEQS")
fi

# Pass through any additional CLI arguments
if [ "$#" -gt 0 ]; then
  ARGS+=("$@")
fi

echo "============================================================"
echo "🚀 Starting vLLM Engine on 1x NVIDIA A100"
echo "Model Path:           $MODEL_PATH"
echo "Served Model Name:    $SERVED_MODEL_NAME"
echo "Tensor Parallelism:   $TENSOR_PARALLEL_SIZE"
echo "Data Type (Dtype):    $DTYPE"
echo "Max Model Length:     $MAX_MODEL_LEN"
echo "GPU Mem Utilization:  $GPU_MEMORY_UTILIZATION"
if [ -n "$QUANTIZATION" ]; then
  echo "Quantization Mode:    $QUANTIZATION"
fi
if [ -n "$KV_CACHE_DTYPE" ]; then
  echo "KV Cache Dtype:       $KV_CACHE_DTYPE"
fi
echo "============================================================"

exec python3 -m vllm.entrypoints.openai.api_server "${ARGS[@]}"

