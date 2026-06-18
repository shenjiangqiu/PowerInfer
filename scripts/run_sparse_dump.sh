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
# 用法:
#   sbatch scripts/run_sparse_dump.sh                            # 默认: ReluLLaMA-7B + C4 prompts
#   sbatch --export=ALL,MODEL=ReluLLaMA-13B,DATASET=wikitext scripts/run_sparse_dump.sh
#
# 环境变量:
#   MODEL         - 要下载/运行的模型 key (见 download_model.sh)
#   MODEL_PATH    - 直接指定模型路径 (覆盖 MODEL 下载)
#   DATASET       - wiki, c4, prompts (ChatGPT prompts), alpaca
#   N_PREDICT     - 每 prompt 生成 token 数 (默认 128)
#   VRAM_BUDGET   - GPU 显存预算 GB (默认 20)
#   THRESHOLD     - predictor 阈值 (默认 0.0)
#   AUTO_DOWNLOAD - 空字符串禁用下载
# ============================================================

set -euo pipefail

# ---- config ----
POWERINFER_DIR="${POWERINFER_DIR:-$HOME/git/PowerInfer}"
MODEL_DIR="${MODEL_DIR:-${HF_HOME:-${POWERINFER_DIR}/hf_home}}"
MODEL="${MODEL:-ReluLLaMA-7B}"
DATASET="${DATASET:-c4}"
N_PREDICT="${N_PREDICT:-128}"
VRAM_BUDGET="${VRAM_BUDGET:-20}"
THREADS="${SLURM_CPUS_PER_TASK:-16}"
THRESHOLD="${THRESHOLD:-0.0}"
AUTO_DOWNLOAD="${AUTO_DOWNLOAD:-1}"

mkdir -p logs "${MODEL_DIR}"

# ---- resolve model path ----
if [ -n "${MODEL_PATH:-}" ]; then
    # 用户直接指定路径
    true
else
    # 根据 MODEL key 推导路径
    case "${MODEL}" in
        ReluLLaMA-7B)    MODEL_PATH="${MODEL_DIR}/prosparse-llama-2-7b.powerinfer.gguf" ;;
        ReluLLaMA-13B)   MODEL_PATH="${MODEL_DIR}/prosparse-llama-2-13b.powerinfer.gguf" ;;
        ReluFalcon-40B)  MODEL_PATH="${MODEL_DIR}/falcon-40b-relu.powerinfer.gguf" ;;
        Bamboo-7B)       MODEL_PATH="${MODEL_DIR}/bamboo-7b-v0.1.gguf" ;;
        Bamboo-DPO-7B)   MODEL_PATH="${MODEL_DIR}/bamboo-7b-dpo-v0.1.gguf" ;;
        SmallThinker-4B) MODEL_PATH="${MODEL_DIR}/smallthinker-4b.gguf" ;;
        OPT-6.7B)        MODEL_PATH="${MODEL_DIR}/opt-6.7b-relu.powerinfer.gguf" ;;
        OPT-30B)         MODEL_PATH="${MODEL_DIR}/opt-30b-relu.powerinfer.gguf" ;;
        *) echo "未知 MODEL=$MODEL"; exit 1 ;;
    esac
fi

DUMP_FILE="sparse_dump_${MODEL}_${DATASET}_${SLURM_JOB_ID}.jsonl"

# ---- download model if needed ----
if [ -n "${AUTO_DOWNLOAD}" ] && [ ! -f "${MODEL_PATH}" ]; then
    echo "=== Downloading model: ${MODEL} ==="
    MODEL_DIR="${MODEL_DIR}" bash "${POWERINFER_DIR}/scripts/download_model.sh" "${MODEL}"
fi

if [ ! -f "${MODEL_PATH}" ]; then
    echo "ERROR: Model not found: ${MODEL_PATH}"
    echo "Set MODEL_PATH=... or AUTO_DOWNLOAD=1 with a valid MODEL key."
    exit 1
fi

# ---- download dataset if needed ----
DATASET_DIR="${POWERINFER_DIR}/datasets"
mkdir -p "${DATASET_DIR}"
PROMPT_FILE="${DATASET_DIR}/${DATASET}_prompts.txt"

if [ ! -f "${PROMPT_FILE}" ]; then
    echo "=== Preparing dataset: ${DATASET} ==="
    case "${DATASET}" in
        wiki)
            # 下载 Wikitext-2 测试集, 每行一个 prompt
            if [ ! -f "${DATASET_DIR}/wikitext-2-test.parquet" ]; then
                python3 -c "
from datasets import load_dataset
ds = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
# 按段落分组, 过滤空行
prompts = []
cur = ''
for t in ds['text']:
    if t.strip():
        cur += ' ' + t.strip()
    elif cur:
        prompts.append(cur.strip()[:2048])
        cur = ''
if cur: prompts.append(cur.strip()[:2048])
with open('${PROMPT_FILE}', 'w') as f:
    for p in prompts[:100]:  # 取前 100 段
        f.write(p + '\n')
print(f'Wikitext: {len(prompts[:100])} prompts saved')
"
            fi
            ;;
        c4)
            python3 -c "
from datasets import load_dataset
ds = load_dataset('allenai/c4', 'en', split='validation', streaming=True)
prompts = []
for i, row in enumerate(ds):
    if i >= 100: break
    prompts.append(row['text'].strip()[:2048])
with open('${PROMPT_FILE}', 'w') as f:
    for p in prompts:
        f.write(p + '\n')
print(f'C4: {len(prompts)} prompts saved')
"
            ;;
        prompts|chatgpt)
            python3 -c "
from datasets import load_dataset
ds = load_dataset('lmsys/chatbot_arena_conversations', split='train', streaming=True)
prompts = []
for i, row in enumerate(ds):
    if i >= 100: break
    q = row['conversation_a'][0]['content'] if row['conversation_a'] else ''
    if q.strip(): prompts.append(q.strip()[:2048])
with open('${PROMPT_FILE}', 'w') as f:
    for p in prompts:
        f.write(p + '\n')
print(f'ChatGPT prompts: {len(prompts)} saved')
"
            ;;
        alpaca)
            python3 -c "
from datasets import load_dataset
ds = load_dataset('tatsu-lab/alpaca', split='train', streaming=True)
prompts = []
for i, row in enumerate(ds):
    if i >= 100: break
    q = row['instruction']
    if row.get('input'): q += ' ' + row['input']
    if q.strip(): prompts.append(q.strip()[:2048])
with open('${PROMPT_FILE}', 'w') as f:
    for p in prompts:
        f.write(p + '\n')
print(f'Alpaca: {len(prompts)} saved')
"
            ;;
        *)
            echo "未知 DATASET=$DATASET, 使用默认 prompt"
            echo "Write a long story about a robot learning to paint." > "${PROMPT_FILE}"
            ;;
    esac
fi

# ---- job info ----
echo "=== Job Info ==="
echo "Job ID:    ${SLURM_JOB_ID}"
echo "Node:      ${SLURMD_NODENAME}"
echo "GPU:       $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo "Model:     ${MODEL}  (${MODEL_PATH})"
echo "Dataset:   ${DATASET}  (${PROMPT_FILE})"
echo "VRAM:      ${VRAM_BUDGET} GB"
echo "Threshold: ${THRESHOLD}"
echo "Predict:   ${N_PREDICT} tokens/prompt"
echo "Dump:      ${DUMP_FILE}"

# ---- build if needed ----
if [ ! -f "${POWERINFER_DIR}/build/bin/main" ]; then
    echo "=== Building PowerInfer ==="
    cd "${POWERINFER_DIR}"
    cmake -S . -B build -DLLAMA_CUBLAS=ON -DCMAKE_BUILD_TYPE=Release
    cmake --build build --config Release -j "${THREADS}"
fi

# ---- run on each prompt ----
echo "=== Starting PowerInfer (${MODEL}, ${DATASET}) ==="

export POWERINFER_DUMP_SPARSE="${DUMP_FILE}"
export LLAMA_SPARSE_PRED_THRESHOLD="${THRESHOLD}"

LINE_NUM=0
while IFS= read -r prompt; do
    [ -z "$prompt" ] && continue
    LINE_NUM=$((LINE_NUM + 1))
    if [ $LINE_NUM -gt 100 ]; then break; fi
    echo "--- Prompt ${LINE_NUM} ---"

    "${POWERINFER_DIR}/build/bin/main" \
        -m "${MODEL_PATH}" \
        -p "${prompt}" \
        -n "${N_PREDICT}" \
        -t "${THREADS}" \
        --vram-budget "${VRAM_BUDGET}" \
        --temp 0.8 --repeat-penalty 1.1 \
        -c 2048 \
        2>&1 | tail -5
done < "${PROMPT_FILE}"

echo "=== Done ==="
DUMP_LINES=$(wc -l < "${DUMP_FILE}" 2>/dev/null || echo 0)
echo "Dump file: ${DUMP_FILE} (${DUMP_LINES} lines)"

# ---- quick stats ----
if command -v jq &>/dev/null && [ "${DUMP_LINES}" -gt 0 ]; then
    echo ""
    echo "=== Sparse Selection Stats ==="
    N_TOKENS=$(jq -s 'max_by(.token).token + 1' "${DUMP_FILE}")
    AVG_ACTIVE=$(jq -s '[.[].active] | add / length | floor' "${DUMP_FILE}")
    SPARSITY=$(jq -s '[.[].active] | add / ([.[].total] | add)' "${DUMP_FILE}")
    echo "Total tokens:      ${N_TOKENS}"
    echo "Avg active neurons: ${AVG_ACTIVE}"
    echo "Avg sparsity ratio: ${SPARSITY}"
    echo ""
    echo "Per-layer (first 8):"
    jq -s 'group_by(.layer) | sort_by(.[0].layer) | .[0:8] | map({L: .[0].layer, tok: length, act: (map(.active)|add/length|floor), tot: .[0].total})' "${DUMP_FILE}"
fi
