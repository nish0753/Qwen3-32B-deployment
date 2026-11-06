FROM vllm/vllm-openai:v0.8.5

# Install curl for container healthchecks
RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*

# Default configuration for 1x NVIDIA A100 GPU
ENV MODEL_PATH=/model \
    SERVED_MODEL_NAME=Qwen3-32B \
    TENSOR_PARALLEL_SIZE=1 \
    DTYPE=bfloat16 \
    MAX_MODEL_LEN=8192 \
    GPU_MEMORY_UTILIZATION=0.92 \
    PORT=8000

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

EXPOSE 8000

ENTRYPOINT ["/entrypoint.sh"]
