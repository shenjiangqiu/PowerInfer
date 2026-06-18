#!/bin/bash
#SBATCH --job-name=powerinfer_sparse_dump
#SBATCH --output=logs/%j_powerinfer.out
#SBATCH --error=logs/%j_powerinfer.err
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --partition=mrigpu

# ============================================================
# PowerInfer 稀疏选择导出
# 用法: sbatch run_powerinfer_sparse_dump.sh
# 环境变量控制:
#   POWERINFER_DUMP_SPARSE  - 导出文件路径 (自动启)
#   LLAMA_SPARSE_PRED_THRESHOLD - predictor 阈值 (默认 0.0)
#   MODEL_PATH             - 模型路径
#   PROMPT_FILE            - prompt 文件路径
#   N_PREDICT              - 生成的 token 数
#   VRAM_BUDGET            - GPU 显存预算 (GB)
#   POWERINFER_DIR         - PowerInfer 仓库路径
# ============================================================

set -euo pipefail

# ---- config ----
POWERINFER_DIR="${POWERINFER_DIR:-$HOME/git/PowerInfer}"
MODEL_PATH="${MODEL_PATH:-/path/to/model.powerinfer.gguf}"
PROMPT_FILE="${PROMPT_FILE:-prompt.txt}"
N_PREDICT="${N_PREDICT:-256}"
VRAM_BUDGET="${VRAM_BUDGET:-20}"
THREADS="${SLURM_CPUS_PER_TASK:-16}"
DUMP_FILE="sparse_dump_${SLURM_JOB_ID}.jsonl"
PRED_THRESHOLD="${LLAMA_SPARSE_PRED_THRESHOLD:-0.0}"

# ---- setup ----
mkdir -p logs

echo "=== Job Info ==="
echo "Job ID:    ${SLURM_JOB_ID}"
echo "Node:      ${SLURMD_NODENAME}"
echo "GPU:       $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo "Model:     ${MODEL_PATH}"
echo "VRAM:      ${VRAM_BUDGET} GB"
echo "Threshold: ${PRED_THRESHOLD}"
echo "Predict:   ${N_PREDICT} tokens"
echo "Dump:      ${DUMP_FILE}"

# ---- build if needed ----
if [ ! -f "${POWERINFER_DIR}/build/bin/main" ]; then
    echo "=== Building PowerInfer ==="
    cd "${POWERINFER_DIR}"
    cmake -S . -B build -DLLAMA_CUBLAS=ON -DCMAKE_BUILD_TYPE=Release
    cmake --build build --config Release -j "${THREADS}"
fi

# ---- prepare prompt file if needed ----
if [ ! -f "${PROMPT_FILE}" ]; then
    echo "Write a long story about a robot learning to paint." > "${PROMPT_FILE}"
    echo "Auto-generated prompt file: ${PROMPT_FILE}"
fi

# ---- run ----
echo "=== Starting PowerInfer ==="

export POWERINFER_DUMP_SPARSE="${DUMP_FILE}"
export LLAMA_SPARSE_PRED_THRESHOLD="${PRED_THRESHOLD}"

"${POWERINFER_DIR}/build/bin/main" \
    -m "${MODEL_PATH}" \
    -f "${PROMPT_FILE}" \
    -n "${N_PREDICT}" \
    -t "${THREADS}" \
    --vram-budget "${VRAM_BUDGET}" \
    --temp 0.8 \
    --repeat-penalty 1.1 \
    -c 2048 \
    2>&1 | tee logs/${SLURM_JOB_ID}_generation.txt

echo "=== Done ==="
echo "Dump file: ${DUMP_FILE} ($(wc -l < "${DUMP_FILE}" 2>/dev/null || echo 0) lines)"

# ---- quick stats ----
if command -v jq &>/dev/null && [ -f "${DUMP_FILE}" ]; then
    echo ""
    echo "=== Sparse Selection Stats ==="
    echo "Total tokens: $(jq -s 'max_by(.token).token + 1' "${DUMP_FILE}")"
    echo "Avg active neurons: $(jq -s '[.[].active] | add / length | floor' "${DUMP_FILE}")"
    echo "Avg sparsity ratio: $(jq -s '[.[].active] | add / ([.[].total] | add)' "${DUMP_FILE}")"
    echo ""
    echo "Per-layer breakdown (first 5 layers):"
    jq -s 'group_by(.layer) | sort_by(.[0].layer) | .[0:5] | map({layer: .[0].layer, n_tokens: length, avg_active: (map(.active) | add/length | floor), total: .[0].total})' "${DUMP_FILE}"
fi
