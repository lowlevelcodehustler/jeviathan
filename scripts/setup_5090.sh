#!/usr/bin/env bash
# Jeviathan — RTX 5090 (32GB) setup: serve Qwen3.8-27B with vLLM.
set -euo pipefail

echo "==> Installing/upgrading vLLM (needs >= 0.17 for Qwen3.8 hybrid attention)"
pip install -U "vllm>=0.17"

MODEL_ALIAS="jevitan-qwen3.8-27b"
PORT=8001

echo "==> Serving Qwen3.8-27B (NVFP4 preferred: ~24.6 GiB, fits one Blackwell GPU)"
# First run downloads weights (~13-29 GB depending on variant).
vllm serve Inferact/Qwen3.8-27B-NVFP4 \
  --served-model-name "$MODEL_ALIAS" \
  --tensor-parallel-size 1 \
  --max-model-len 32768 \
  --kv-cache-dtype fp8 \
  --port "$PORT"

# --- Fallback if NVFP4 has issues on your driver/vLLM combo: official FP8 ---
# vllm serve Qwen/Qwen3.8-27B-FP8 \
#   --served-model-name "$MODEL_ALIAS" \
#   --tensor-parallel-size 1 \
#   --max-model-len 16384 \
#   --kv-cache-dtype fp8 \
#   --port "$PORT"
