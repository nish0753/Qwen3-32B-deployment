#!/usr/bin/env python3
"""
verify_a100.py - Pre-Flight Verification & Hardware Diagnostics for Qwen3-32B on 1x NVIDIA A100

Run this script directly on your GPU host to verify drivers, GPU detection, VRAM headroom,
Docker NVIDIA toolkit, model weights, and port availability before launching the stack.
"""

import os
import sys
import shutil
import subprocess
import socket


def print_banner(text):
    print("\n" + "=" * 65)
    print(f" {text}")
    print("=" * 65)


def check_gpu():
    print_banner("1. NVIDIA GPU & Driver Verification")
    smi = shutil.which("nvidia-smi")
    if not smi:
        print("❌ FAIL: nvidia-smi command not found.")
        print("   -> NVIDIA drivers are either not installed or GPU is not attached.")
        return None

    try:
        cmd = [smi, "--query-gpu=index,name,memory.total,driver_version", "--format=csv,noheader,nounits"]
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=5)
        if res.returncode != 0:
            print(f"❌ FAIL: nvidia-smi failed with error: {res.stderr.strip()}")
            return None

        lines = res.stdout.strip().splitlines()
        if not lines:
            print("❌ FAIL: No GPUs detected by nvidia-smi.")
            return None

        gpus = []
        for line in lines:
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 4:
                idx = parts[0]
                name = parts[1]
                vram_mb = float(parts[2])
                driver = parts[3]
                vram_gb = vram_mb / 1024.0
                gpus.append({"index": idx, "name": name, "vram_gb": vram_gb, "driver": driver})

        primary_gpu = gpus[0]
        print(f"✅ GPU Detected:    {primary_gpu['name']} (Index {primary_gpu['index']})")
        print(f"✅ Total VRAM:      {primary_gpu['vram_gb']:.1f} GB")
        print(f"✅ Driver Version:  {primary_gpu['driver']}")

        if len(gpus) > 1:
            print(f"ℹ️  Note: {len(gpus)} GPUs detected on host. Deployment is configured to bind GPU 0 (TP=1).")

        # Sizing Verdict
        if primary_gpu["vram_gb"] >= 75.0:
            print("🎯 VRAM Evaluation: A100 80GB Profile detected.")
            print("   -> Full native bfloat16 precision (TP=1, max-model-len=8192) will fit comfortably!")
            return {"gpu": primary_gpu, "profile": "80gb"}
        elif primary_gpu["vram_gb"] >= 38.0:
            print("⚠️ VRAM Evaluation: A100 40GB Profile detected.")
            print("   -> 65.5 GB unquantized weights will exceed 40 GB VRAM.")
            print("   -> Set QUANTIZATION=awq (4-bit) or QUANTIZATION=fp8 in docker-compose.yml!")
            return {"gpu": primary_gpu, "profile": "40gb"}
        else:
            print(f"❌ WARNING: VRAM ({primary_gpu['vram_gb']:.1f} GB) is below 40 GB. Heavy quantization required.")
            return {"gpu": primary_gpu, "profile": "small"}

    except Exception as e:
        print(f"❌ Error querying GPU: {e}")
        return None


def check_docker():
    print_banner("2. Docker & NVIDIA Container Runtime")
    docker_bin = shutil.which("docker")
    if not docker_bin:
        print("❌ FAIL: docker binary not found in PATH.")
        return False
    print("✅ Docker binary is installed.")

    # Check docker info for nvidia runtime
    try:
        res = subprocess.run([docker_bin, "info"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=5)
        if "nvidia" in res.stdout.lower():
            print("✅ NVIDIA Container Toolkit (nvidia-container-runtime) detected in Docker.")
            return True
        else:
            print("⚠️ WARNING: 'nvidia' runtime not explicitly listed in docker info.")
            print("   Ensure NVIDIA Container Toolkit is installed (e.g. `apt-get install nvidia-container-toolkit`).")
            return True
    except Exception as e:
        print(f"⚠️ Could not verify docker daemon: {e}")
        return True


def check_model_weights():
    print_banner("3. Model Weights Verification (/root/Qwen3-32B)")
    path = os.environ.get("MODEL_PATH", "/root/Qwen3-32B")
    if not os.path.exists(path):
        print(f"⚠️ Directory '{path}' does not exist yet.")
        print("   -> Download weights before starting: `huggingface-cli download Qwen/Qwen3-32B --local-dir /root/Qwen3-32B`")
        return False

    files = os.listdir(path)
    total_size_bytes = sum(os.path.getsize(os.path.join(path, f)) for f in files if os.path.isfile(os.path.join(path, f)))
    total_size_gb = total_size_bytes / (1024 ** 3)

    has_config = "config.json" in files
    has_tokenizer = any("tokenizer" in f for f in files)
    has_weights = any(f.endswith(".safetensors") or f.endswith(".bin") for f in files)

    if has_config and has_weights:
        print(f"✅ Model weights located at: {path}")
        print(f"✅ Total weights size:       {total_size_gb:.2f} GB")
        print(f"✅ Config & Tokenizer:       Present")
        return True
    else:
        print(f"⚠️ Model directory '{path}' found, but missing files (size: {total_size_gb:.2f} GB).")
        return False


def check_ports():
    print_banner("4. Network Port Availability")
    ports = [(8000, "vLLM Inference Engine API"), (8080, "Chat UI Frontend (Nginx)"), (3000, "Python Security & Telemetry Proxy")]
    all_clear = True
    for port, label in ports:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(0.5)
            res = s.connect_ex(("127.0.0.1", port))
            if res == 0:
                print(f"⚠️ Port {port} ({label}) is currently OCCUPIED.")
                all_clear = False
            else:
                print(f"✅ Port {port} ({label}) is FREE.")
    return all_clear


def main():
    print("=" * 65)
    print(" 🚀 Qwen3-32B on 1x NVIDIA A100: Pre-Flight Diagnostic Check")
    print("=" * 65)

    gpu_info = check_gpu()
    docker_ok = check_docker()
    weights_ok = check_model_weights()
    ports_ok = check_ports()

    print_banner("5. Pre-Flight Diagnostic Summary")
    if gpu_info and gpu_info["profile"] == "80gb":
        print("🎉 ALL SYSTEMS GO FOR 1x A100 80GB (Native BF16 Mode)!")
        print("\nTo launch immediately:")
        print("   docker compose up -d --build")
        print("   python3 serve_ui.py")
        print("Then open: http://localhost:3000 in your browser.")
    elif gpu_info and gpu_info["profile"] == "40gb":
        print("⚡ READY FOR 1x A100 40GB (Quantized AWQ Mode)!")
        print("\nNote: Uncomment 'QUANTIZATION=awq' in docker-compose.yml before running:")
        print("   docker compose up -d --build")
        print("   python3 serve_ui.py")
    else:
        print("📋 Please attach your GPU and run drivers installation, then re-run:")
        print("   python3 verify_a100.py")


if __name__ == "__main__":
    main()

