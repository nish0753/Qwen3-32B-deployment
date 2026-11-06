#!/usr/bin/env python3
"""
serve_ui.py - High-Performance Observability & Security Proxy for Qwen3-32B on 1x NVIDIA A100

Features:
- Proxies /v1/* to vLLM engine with real-time SSE streaming (flush per chunk).
- Proxies /metrics to vLLM Prometheus endpoint.
- Provides /api/metrics endpoint: parses vLLM Prometheus metrics into clean JSON.
- Provides /api/gpu-stats endpoint: queries nvidia-smi for real-time A100 telemetry (VRAM, Util, Temp, Power).
- Serves the 'Measure Everything' Studio UI (chat_ui.html).
- Supports X-Auth-Key security protection.
- Configurable via environment variables (UPSTREAM_URL, PORT, UI_AUTH_KEY).
"""

import http.server
import json
import os
import shutil
import subprocess
import time
import urllib.request
import urllib.error
from urllib.parse import urlparse, parse_qs

PORT = int(os.environ.get('PORT', 3000))
UPSTREAM = os.environ.get('UPSTREAM_URL', os.environ.get('VLLM_HOST', 'http://127.0.0.1:8000')).rstrip('/')
AUTH_KEY = os.environ.get('UI_AUTH_KEY', '')


def get_gpu_stats():
    """Queries nvidia-smi for real-time GPU telemetry."""
    nvidia_smi_path = shutil.which("nvidia-smi")
    if not nvidia_smi_path:
        return {
            "available": False,
            "message": "nvidia-smi not found. GPU not attached or drivers not installed yet.",
            "gpus": []
        }

    try:
        cmd = [
            nvidia_smi_path,
            "--query-gpu=index,name,utilization.gpu,memory.total,memory.used,memory.free,temperature.gpu,power.draw,power.limit,clocks.current.graphics,clocks.current.memory",
            "--format=csv,noheader,nounits"
        ]
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=3)
        if res.returncode != 0:
            return {
                "available": False,
                "message": f"nvidia-smi error: {res.stderr.strip()}",
                "gpus": []
            }

        gpus = []
        for line in res.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(',')]
            if len(parts) >= 11:
                def safe_float(v, default=0.0):
                    try:
                        return float(v)
                    except (ValueError, TypeError):
                        return default

                gpus.append({
                    "index": int(parts[0]),
                    "name": parts[1],
                    "utilization_gpu_pct": safe_float(parts[2]),
                    "memory_total_mb": safe_float(parts[3]),
                    "memory_used_mb": safe_float(parts[4]),
                    "memory_free_mb": safe_float(parts[5]),
                    "temperature_c": safe_float(parts[6]),
                    "power_draw_w": safe_float(parts[7]),
                    "power_limit_w": safe_float(parts[8]),
                    "clock_graphics_mhz": safe_float(parts[9]),
                    "clock_memory_mhz": safe_float(parts[10])
                })

        return {
            "available": len(gpus) > 0,
            "gpus": gpus,
            "timestamp": time.time()
        }
    except Exception as e:
        return {
            "available": False,
            "message": str(e),
            "gpus": []
        }


def parse_prometheus_metrics(raw_text):
    """Parses raw Prometheus text into a clean dictionary of key vLLM metrics."""
    metrics = {}
    for line in raw_text.splitlines():
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        parts = line.split()
        if len(parts) >= 2:
            name_and_labels = parts[0]
            value = parts[1]
            metric_name = name_and_labels.split('{')[0]
            try:
                val = float(value)
                clean_name = metric_name.replace('vllm:', '')
                metrics[clean_name] = val
            except ValueError:
                pass
    return metrics


class ProxyHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header('Cache-Control', 'no-store, no-cache, must-revalidate, max-age=0')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type, X-Auth-Key, Authorization')
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(204)
        self.end_headers()

    def _check_key(self, from_query=False):
        if not AUTH_KEY:
            return True
        if from_query:
            qs = parse_qs(urlparse(self.path).query)
            return qs.get('key', [''])[0] == AUTH_KEY
        return self.headers.get('X-Auth-Key') == AUTH_KEY

    def _deny(self):
        self.send_response(403)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(b'{"error": "Forbidden: missing or invalid X-Auth-Key"}')

    def do_GET(self):
        # 1. GPU Telemetry API endpoint
        if self.path == '/api/gpu-stats':
            if not self._check_key():
                return self._deny()
            stats = get_gpu_stats()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(stats).encode('utf-8'))
            return

        # 2. Parsed vLLM engine metrics API endpoint
        if self.path == '/api/metrics':
            if not self._check_key():
                return self._deny()
            try:
                req = urllib.request.Request(UPSTREAM + '/metrics')
                with urllib.request.urlopen(req, timeout=4) as resp:
                    raw_text = resp.read().decode('utf-8', errors='ignore')
                    parsed = parse_prometheus_metrics(raw_text)
                    self.send_response(200)
                    self.send_header('Content-Type', 'application/json')
                    self.end_headers()
                    self.wfile.write(json.dumps(parsed).encode('utf-8'))
                    return
            except Exception as e:
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(e), "connected": False}).encode('utf-8'))
                return

        # 3. Raw Prometheus metrics proxy
        if self.path == '/metrics':
            if not self._check_key():
                return self._deny()
            return self._proxy('GET')

        # 4. Engine Health proxy
        if self.path == '/health':
            return self._proxy('GET')

        # 5. OpenAI API paths
        if self.path.startswith('/v1/'):
            if not self._check_key():
                return self._deny()
            return self._proxy('GET')

        # 6. Static files (serve chat_ui.html for root)
        clean_path = urlparse(self.path).path
        if clean_path in ('', '/', '/index.html'):
            self.path = '/chat_ui.html'
        return super().do_GET()

    def do_POST(self):
        if self.path.startswith('/v1/'):
            if not self._check_key():
                return self._deny()
            return self._proxy('POST')
        return self._deny()

    def _proxy(self, method):
        length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(length) if length else None
        target_url = UPSTREAM + self.path
        req = urllib.request.Request(target_url, data=body, method=method)
        req.add_header('Content-Type', self.headers.get('Content-Type', 'application/json'))

        # Forward authorization if present
        if 'Authorization' in self.headers:
            req.add_header('Authorization', self.headers['Authorization'])

        try:
            with urllib.request.urlopen(req, timeout=300) as upstream:
                self.send_response(upstream.status)
                self.send_header('Content-Type', upstream.headers.get('Content-Type', 'application/json'))
                self.send_header('Cache-Control', 'no-cache, no-transform')
                self.send_header('X-Accel-Buffering', 'no')
                self.end_headers()
                while True:
                    try:
                        chunk = upstream.read(512)
                        if not chunk:
                            break
                        self.wfile.write(chunk)
                        self.wfile.flush()
                    except (BrokenPipeError, ConnectionResetError):
                        break
        except urllib.error.HTTPError as e:
            self.send_response(e.code)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(e.read())
        except Exception as e:
            self.send_response(502)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            err_msg = json.dumps({"error": f"Upstream vLLM engine unreachable on port 8000: {str(e)}"}).encode('utf-8')
            self.wfile.write(err_msg)


if __name__ == '__main__':
    print("=" * 60)
    print(f"🚀 Qwen3-32B Observability & Security Proxy")
    print(f"Port:       {PORT}")
    print(f"Upstream:   {UPSTREAM}")
    if AUTH_KEY:
        print(f"Auth Key:   {AUTH_KEY}")
    else:
        print("Auth Key:   [Disabled - Open Access]")
    print(f"UI URL:     http://localhost:{PORT}")
    print("=" * 60)
    server = http.server.ThreadingHTTPServer(('0.0.0.0', PORT), ProxyHandler)
    server.serve_forever()
