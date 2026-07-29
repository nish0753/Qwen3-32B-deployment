FROM vllm/vllm-openai:v0.8.5

CMD ["--model", "/model", \
     "--served-model-name", "./Qwen3-32B", \
     "--tensor-parallel-size", "4", \
     "--max-model-len", "8192", \
     "--dtype", "half", \
     "--disable-custom-all-reduce"]
