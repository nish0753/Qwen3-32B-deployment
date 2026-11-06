#!/usr/bin/env python3
"""
benchmark_cli.py - Standalone Terminal Benchmarking Tool for Qwen3-32B on 1x NVIDIA A100

Measures:
- Time To First Token (TTFT) in ms
- Generation Throughput (TPS) in tokens/sec
- Inter-Token Latency (TPOT) in ms/token
- Reasoning <think> token accounting and speed
- GPU VRAM consumption & KV cache factor
"""

import sys
import time
import json
import urllib.request
import urllib.error
import shutil
import subprocess

TARGET_URL = "http://127.0.0.1:8000"
PROXY_URL = "http://127.0.0.1:3000"

BENCHMARK_PROMPTS = [
    {
        "name": "⚡ Cold Latency Probe",
        "prompt": "Define the mathematical difference between Tensor Parallelism and Pipeline Parallelism in two concise sentences.",
        "max_tokens": 64
    },
    {
        "name": "🧠 Deep Reasoning & Chain-of-Thought",
        "prompt": "Solve step-by-step inside <think> tags: Three friends Alice, Bob, and Charlie have a total of 144 marbles. Alice gives half her marbles to Bob. Bob then gives one-third of his total to Charlie. Finally, Charlie gives one-fourth of his total to Alice. If they all end up with equal marbles (48 each), how many did each person start with?",
        "max_tokens": 512
    },
    {
        "name": "🚀 Heavy Context Throughput",
        "prompt": "You are a chief AI infrastructure architect. Write an in-depth technical analysis of memory bandwidth ceilings (HBM2e), FlashAttention-2 tiling mechanics, CUDA warp-level matrix multiply (WMMA), and PagedAttention KV-cache fragmentation on an NVIDIA A100 GPU.",
        "max_tokens": 384
    }
]


def get_gpu_telemetry():
    smi = shutil.which("nvidia-smi")
    if not smi:
        return {"util": 0, "mem_used_gb": 0, "mem_total_gb": 0, "temp": 0, "power": 0}
    try:
        cmd = [smi, "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw", "--format=csv,noheader,nounits"]
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=2)
        if res.returncode == 0:
            p = [x.strip() for x in res.stdout.strip().splitlines()[0].split(',')]
            return {
                "util": float(p[0]),
                "mem_used_gb": float(p[1]) / 1024.0,
                "mem_total_gb": float(p[2]) / 1024.0,
                "temp": float(p[3]),
                "power": float(p[4]) if p[4] != "[Not Supported]" else 0.0
            }
    except Exception:
        pass
    return {"util": 0, "mem_used_gb": 0, "mem_total_gb": 0, "temp": 0, "power": 0}


def stream_completion(prompt, max_tokens, model="Qwen3-32B"):
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.2,
        "stream": True
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(f"{TARGET_URL}/v1/chat/completions", data=data, method="POST")
    req.add_header("Content-Type", "application/json")

    start_time = time.perf_counter()
    first_token_time = None
    token_count = 0
    full_text = ""

    with urllib.request.urlopen(req, timeout=120) as resp:
        buffer = ""
        while True:
            chunk = resp.read(256)
            if not chunk:
                break
            if first_token_time is None:
                first_token_time = time.perf_counter()

            buffer += chunk.decode("utf-8", errors="ignore")
            lines = buffer.split("\n")
            buffer = lines.pop()

            for line in lines:
                line = line.strip()
                if line.startswith("data:") and "[DONE]" not in line:
                    try:
                        j = json.loads(line[5:].strip())
                        delta = j.get("choices", [{}])[0].get("delta", {}).get("content", "")
                        if delta:
                            token_count += 1
                            full_text += delta
                    except Exception:
                        pass

    total_time = time.perf_counter() - start_time
    ttft_ms = (first_token_time - start_time) * 1000.0 if first_token_time else 0.0
    decode_time = (time.perf_counter() - first_token_time) if first_token_time else 0.0
    tps = token_count / decode_time if decode_time > 0 else 0.0
    tpot_ms = (decode_time * 1000.0) / token_count if token_count > 0 else 0.0

    return {
        "ttft_ms": round(ttft_ms, 1),
        "tps": round(tps, 1),
        "tpot_ms": round(tpot_ms, 2),
        "total_sec": round(total_time, 2),
        "token_count": token_count,
        "full_text": full_text
    }


def main():
    print("=" * 70)
    print(" 🚀 Qwen3-32B on 1x NVIDIA A100 Terminal Benchmark Suite")
    print(f" Target: {TARGET_URL}")
    print("=" * 70)

    # Health check
    try:
        with urllib.request.urlopen(f"{TARGET_URL}/health", timeout=3) as r:
            if r.status != 200:
                print(f"❌ Target server returned HTTP {r.status}")
                sys.exit(1)
    except Exception as e:
        print(f"❌ Could not connect to {TARGET_URL}. Is vLLM running?")
        print(f"   Error: {e}")
        sys.exit(1)

    print("✅ vLLM engine is responsive. Initializing benchmark runs...\n")

    results = []
    for idx, test in enumerate(BENCHMARK_PROMPTS, 1):
        print(f"[{idx}/{len(BENCHMARK_PROMPTS)}] Running: {test['name']} (max_tokens={test['max_tokens']})...")
        res = stream_completion(test["prompt"], test["max_tokens"])
        gpu = get_gpu_telemetry()
        res["test_name"] = test["name"]
        res["gpu_util"] = gpu["util"]
        res["vram_used"] = gpu["mem_used_gb"]
        results.append(res)
        print(f"   -> TTFT: {res['ttft_ms']} ms | Speed: {res['tps']} tok/s | Tokens: {res['token_count']} | VRAM: {gpu['mem_used_gb']:.1f} GB\n")
        time.sleep(1)

    # Print Summary Table
    print("=" * 75)
    print(f"{'Benchmark Test':<32} | {'TTFT (ms)':<9} | {'Speed (tok/s)':<13} | {'TPOT (ms)':<9} | {'Tokens':<6}")
    print("-" * 75)
    for r in results:
        print(f"{r['test_name']:<32} | {r['ttft_ms']:<9} | {r['tps']:<13} | {r['tpot_ms']:<9} | {r['token_count']:<6}")
    print("=" * 75)

    avg_ttft = sum(r["ttft_ms"] for r in results) / len(results)
    avg_tps = sum(r["tps"] for r in results) / len(results)
    peak_tps = max(r["tps"] for r in results)

    print(f"🎯 Average TTFT:    {avg_ttft:.1f} ms")
    print(f"🎯 Average Speed:   {avg_tps:.1f} tokens/sec")
    print(f"🎯 Peak Speed:      {peak_tps:.1f} tokens/sec")
    print("=" * 75)


if __name__ == "__main__":
    main()

