#!/usr/bin/env bash
set -e

echo "=================================================================="
echo "🔧 STEP 1: Verifying Model Weights Completion (~65 GB)"
echo "=================================================================="
export HF_HUB_ENABLE_HF_TRANSFER=1
python3 -c "
from huggingface_hub import snapshot_download
print('Verifying/resuming Qwen/Qwen3-32B download...')
snapshot_download('Qwen/Qwen3-32B', local_dir='/workspace/Qwen3-32B')
print('Download verified complete!')
"
echo "Model directory size: $(du -sh /workspace/Qwen3-32B | cut -f1)"

echo "=================================================================="
echo "🧹 STEP 2: Stopping Any Stale Processes"
echo "=================================================================="
pkill -9 -f "vllm" || true
pkill -9 -f "serve_ui" || true
pkill -9 -f "cloudflared" || true
sleep 2

echo "=================================================================="
echo "🚀 STEP 3: Starting vLLM Engine on 1x A100 (Port 8000)"
echo "=================================================================="
export VLLM_USE_V1=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn

nohup python3 -m vllm.entrypoints.openai.api_server \
  --model /workspace/Qwen3-32B \
  --served-model-name Qwen3-32B \
  --tensor-parallel-size 1 \
  --dtype bfloat16 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.88 \
  --enforce-eager \
  --trust-remote-code \
  --port 8000 > /workspace/vllm.log 2>&1 &

echo "Waiting for vLLM to initialize weights into A100 VRAM..."
ONLINE=0
for i in {1..60}; do
  if curl -s http://127.0.0.1:8000/health > /dev/null 2>&1; then
    echo ""
    echo "✅ vLLM Engine is ONLINE & healthy on port 8000!"
    ONLINE=1
    break
  fi
  sleep 2
  printf "."
done

if [ $ONLINE -eq 0 ]; then
  echo ""
  echo "❌ vLLM failed to report healthy within 120s. Last 25 log lines:"
  tail -n 25 /workspace/vllm.log
  exit 1
fi

echo "=================================================================="
echo "📊 STEP 4: Starting Observability & Telemetry Proxy (Port 3000)"
echo "=================================================================="
cd /workspace/Qwen3-32B-deployment
nohup python3 serve_ui.py > /workspace/ui.log 2>&1 &
sleep 2

echo "=================================================================="
echo "🌐 STEP 5: Starting Fresh Cloudflare Public Tunnel"
echo "=================================================================="
if [ ! -f /workspace/cloudflared ]; then
  curl -sL https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -o /workspace/cloudflared
  chmod +x /workspace/cloudflared
fi
rm -f /workspace/tunnel.log
nohup /workspace/cloudflared tunnel --url http://localhost:3000 > /workspace/tunnel.log 2>&1 &
sleep 4

PUBLIC_URL=""
for i in {1..10}; do
  PUBLIC_URL=$(grep -a -o 'https://[-a-zA-Z0-9.]*trycloudflare.com' /workspace/tunnel.log | head -n 1 || true)
  if [ -n "$PUBLIC_URL" ]; then
    break
  fi
  sleep 1
done

echo "=================================================================="
echo "🧪 STEP 6: Testing End-to-End Chat Completion via Port 8000"
echo "=================================================================="
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen3-32B",
    "messages": [{"role": "user", "content": "Hello"}],
    "max_tokens": 10
  }' | head -n 5

echo ""
echo "=================================================================="
echo "🎉 QWEN3-32B IS 100% OPERATIONAL ON RUNPOD A100!"
echo "=================================================================="
if [ -n "$PUBLIC_URL" ]; then
  echo "👉 YOUR PUBLIC STUDIO URL: $PUBLIC_URL"
  echo "   (Open this in your browser to chat and benchmark!)"
else
  echo "👉 Check tunnel URL: grep -a -o 'https://.*trycloudflare.com' /workspace/tunnel.log"
fi
echo "=================================================================="

