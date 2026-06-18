#!/bin/bash
# ============================================================
# PowerInfer 模型下载脚本 (hf download)
# 用法:
#   bash scripts/download_model.sh                            # 交互式选择
#   bash scripts/download_model.sh ReluLLaMA-7B               # 指定模型
#   MODEL_DIR=./hf_home bash scripts/download_model.sh ReluLLaMA-7B
# ============================================================
set -euo pipefail

# 检查 hf CLI
if ! command -v huggingface-cli &>/dev/null && ! python3 -c "import huggingface_hub" 2>/dev/null; then
    echo "请安装: pip install huggingface_hub"
    exit 1
fi

MODEL_DIR="${MODEL_DIR:-${HF_HOME:-./hf_home}}"
mkdir -p "${MODEL_DIR}"

# ---- 模型列表 ----
declare -A MODELS
MODELS=(
  ["ReluLLaMA-7B"]="SparseLLM/prosparse-llama-2-7b|prosparse-llama-2-7b.powerinfer.gguf|3.8|LLaMA2-7B ReLU ~90%稀疏"
  ["ReluLLaMA-13B"]="SparseLLM/prosparse-llama-2-13b|prosparse-llama-2-13b.powerinfer.gguf|7.3|LLaMA2-13B ReLU ~90%稀疏"
  ["ReluFalcon-40B"]="PowerInfer/ReluFalcon-40B-PowerInfer-GGUF|falcon-40b-relu.powerinfer.gguf|22|Falcon-40B ReLU ~95%稀疏"
  ["Bamboo-7B"]="PowerInfer/Bamboo-base-v0.1-gguf|bamboo-7b-v0.1.gguf|3.9|Bamboo-7B dReLU"
  ["Bamboo-DPO-7B"]="PowerInfer/Bamboo-DPO-v0.1-gguf|bamboo-7b-dpo-v0.1.gguf|3.9|Bamboo-7B DPO"
  ["SmallThinker-4B"]="PowerInfer/SmallThinker-4BA0.6B-Instruct|smallthinker-4b.gguf|2.4|SmallThinker-4B MoE"
  ["OPT-6.7B"]="PowerInfer/opt-6.7b-relu|opt-6.7b-relu.powerinfer.gguf|3.8|OPT-6.7B ReLU ~97%稀疏"
  ["OPT-30B"]="PowerInfer/opt-30b-relu|opt-30b-relu.powerinfer.gguf|16|OPT-30B ReLU ~98%稀疏"
)

# ---- 选择模型 ----
if [ $# -ge 1 ]; then
    MODEL_KEY="$1"
else
    echo "=== 可选模型 ==="
    for key in "${!MODELS[@]}"; do
        IFS='|' read -r repo file size desc <<< "${MODELS[$key]}"
        printf "  %-22s %5s GB   %s\n" "${key}" "${size}" "${desc}"
    done
    echo ""
    read -p "输入模型名: " MODEL_KEY
fi

if [ -z "${MODELS[$MODEL_KEY]:-}" ]; then
    echo "错误: 未知模型 '${MODEL_KEY}'"
    echo "可选: ${!MODELS[*]}"
    exit 1
fi

IFS='|' read -r HF_REPO FILENAME SIZE_GB DESC <<< "${MODELS[$MODEL_KEY]}"

echo "=== 下载 ${MODEL_KEY} ==="
echo "  来源:  https://huggingface.co/${HF_REPO}"
echo "  文件:  ${FILENAME}  (~${SIZE_GB} GB)"
echo "  目标:  ${MODEL_DIR}"

# ---- 下载 ----
if [ -f "${MODEL_DIR}/${FILENAME}" ]; then
    echo "  已存在, 跳过: ${MODEL_DIR}/${FILENAME}"
elif command -v huggingface-cli &>/dev/null; then
    hf download "${HF_REPO}" "${FILENAME}" --local-dir "${MODEL_DIR}" --resume-download
else
    python3 -c "
from huggingface_hub import hf_hub_download
hf_hub_download('${HF_REPO}', '${FILENAME}', local_dir='${MODEL_DIR}', resume_download=True)
"
fi

echo ""
echo "=== 完成 ==="
echo "模型: ${MODEL_DIR}/${FILENAME}"
echo ""
echo "运行:"
echo "  MODEL_PATH=${MODEL_DIR}/${FILENAME} \\"
echo "  POWERINFER_DUMP_SPARSE=sparse_dump.jsonl \\"
echo "  ./build/bin/main -m \${MODEL_PATH} -n 128 -t 8 -p \"hello\" --vram-budget 20"
