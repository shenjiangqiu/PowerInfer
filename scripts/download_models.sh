#!/bin/bash
# 下载 PowerInfer 7B/13B 模型到 $HF_HOME (默认 ./hf_home)
# 用法: bash scripts/download_models.sh
# 或:   HF_HOME=/data/models bash scripts/download_models.sh
set -euo pipefail

DEST="${HF_HOME:-./hf_home}"
mkdir -p "${DEST}"

MODELS=(
    "PowerInfer/ReluLLaMA-7B-PowerInfer-GGUF"
    "PowerInfer/ReluLLaMA-13B-PowerInfer-GGUF"
    "PowerInfer/ProSparse-LLaMA-2-7B-GGUF"
    "PowerInfer/ProSparse-LLaMA-2-13B-GGUF"
    "PowerInfer/Bamboo-base-v0.1-gguf"
    "PowerInfer/Bamboo-DPO-v0.1-gguf"
)

for repo in "${MODELS[@]}"; do
    echo "=== ${repo} ==="
    HF_HUB_OFFLINE= hf download "${repo}"
    echo ""
done

echo "=== 完成 ==="
echo "模型目录: ${DEST}/"
ls -lh "${DEST}/"
