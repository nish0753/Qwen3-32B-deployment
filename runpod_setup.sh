#!/usr/bin/env bash
# ==============================================================================
# runpod_setup.sh - 1-Click Automated Setup for Qwen3-32B on RunPod (1x A100)
# ==============================================================================
set -e

export PATH="$HOME/.local/bin:/root/.local/bin:$PATH"

echo "=================================================================="
echo "🚀 Initializing Qwen3-32B Single-A100 Deployment on RunPod"
echo "=================================================================="

# 1. Verify NVIDIA Driver & A100 Detection
if ! command -v nvidia-smi &> /dev/null; then
  echo "❌ Error: nvidia-smi not found. Ensure you deployed a GPU pod with NVIDIA drivers."
  exit 1
fi

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)
VRAM_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n 1)
VRAM_GB=$(awk "BEGIN {print int($VRAM_MB/1024)}")

echo "✅ Detected GPU: $GPU_NAME ($VRAM_GB GB VRAM)"

# Determine precision profile
QUANT_ARG=""
if [ "$VRAM_GB" -lt 70 ]; then
  echo "⚠️ VRAM is below 70 GB ($VRAM_GB GB detected). Activating 4-bit AWQ mode for A100 40GB..."
  QUANT_ARG="--quantization awq"
else
  echo "🎯 A100 80GB Profile detected. Using full native bfloat16 precision."
fi

# 2. Update pip & install required packages
echo "📦 Installing vLLM, HuggingFace Hub with fast transfer..."
pip install --upgrade pip
pip install "vllm>=0.8.5" huggingface_hub hf_transfer

# 3. Model Weights Download (High-speed hf_transfer & resume verification)
MODEL_DIR="/workspace/Qwen3-32B"
echo "⬇️ Verifying and downloading complete Qwen3-32B model weights to $MODEL_DIR (~65 GB)..."
export HF_HUB_ENABLE_HF_TRANSFER=1
python3 -c "
import os
from huggingface_hub import snapshot_download
snapshot_download('Qwen/Qwen3-32B', local_dir='$MODEL_DIR', resume_download=True)
"

# Ensure symlink from /root/Qwen3-32B
if [ ! -d "/root/Qwen3-32B" ]; then
  ln -s "$MODEL_DIR" /root/Qwen3-32B || true
fi

# 4. Stop any existing vLLM or serve_ui processes
echo "🧹 Checking for stale processes..."
pkill -f "vllm.entrypoints.openai.api_server" || true
pkill -f "serve_ui.py" || true
sleep 2

# 5. Launch vLLM in the background
echo "🚀 Starting vLLM Engine in background on port 8000..."
nohup python3 -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_DIR" \
  --served-model-name Qwen3-32B \
  --tensor-parallel-size 1 \
  --dtype bfloat16 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.92 \
  --trust-remote-code \
  $QUANT_ARG \
  --port 8000 > /workspace/vllm.log 2>&1 &

echo "⏳ Waiting for vLLM to initialize weights into A100 VRAM (typically 30-45s)..."
for i in {1..60}; do
  if curl -s http://127.0.0.1:8000/health > /dev/null 2>&1; then
    echo "✅ vLLM Engine is ONLINE & healthy!"
    break
  fi
  sleep 2
  printf "."
done

# 6. Start Observability & Security Proxy
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
echo "📊 Launching 'Measure Everything' Observability Proxy on port 3000..."
nohup python3 "$SCRIPT_DIR/serve_ui.py" > /workspace/ui.log 2>&1 &
sleep 2

# 7. Start Cloudflare HTTPS Public Tunnel
echo "🌐 Setting up instant public HTTPS tunnel..."
if ! command -v cloudflared &> /dev/null && [ ! -f /workspace/cloudflared ]; then
  curl -sL https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -o /workspace/cloudflared
  chmod +x /workspace/cloudflared
fi
CF_CMD=$(command -v cloudflared || echo "/workspace/cloudflared")
pkill -f "cloudflared tunnel" || true
nohup "$CF_CMD" tunnel --url http://localhost:3000 > /workspace/tunnel.log 2>&1 &
sleep 4

PUBLIC_URL=""
for i in {1..10}; do
  PUBLIC_URL=$(grep -a -o 'https://[-a-zA-Z0-9.]*trycloudflare.com' /workspace/tunnel.log | head -n 1 || true)
  if [ -n "$PUBLIC_URL" ]; then
    break
  fi
  sleep 1
done

echo ""
echo "=================================================================="
echo "🎉 Qwen3-32B IS READY ON RUNPOD 1x A100!"
echo "=================================================================="
if [ -n "$PUBLIC_URL" ]; then
  echo "👉 YOUR PUBLIC STUDIO URL: $PUBLIC_URL"
  echo "   (Click or open this URL in your browser to access the Studio)"
else
  echo "👉 Check tunnel URL manually: grep -a -o 'https://.*trycloudflare.com' /workspace/tunnel.log"
fi
echo ""
echo "Local endpoints on this pod:"
echo "   UI Studio:       http://localhost:3000"
echo "   vLLM OpenAI API: http://localhost:8000/v1"
echo "   GPU Telemetry:   http://localhost:3000/api/gpu-stats"
echo "   vLLM Metrics:    http://localhost:3000/api/metrics"
echo ""
echo "To monitor logs in real-time:"
echo "   tail -f /workspace/vllm.log"
echo "   tail -f /workspace/ui.log"
echo "=================================================================="

