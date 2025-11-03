python -m vllm.entrypoints.openai.api_server \
    --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --served-model-name tinyllama \
    --host 127.0.0.1 \
    --port 8080 \
    --gpu-memory-utilization 0.7 \
    --max-model-len 1024
