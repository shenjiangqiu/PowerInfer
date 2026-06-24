#!/usr/bin/env bash
# Batch run PowerInfer binary dump across models and datasets.
# Usage:
#   bash scripts/run_sparse_dump_bin.sh
#
# Config via env vars:
#   MODELS="ReluLLaMA-7B,OPT-6.7B" DATASETS="wiki,c4" bash scripts/run_sparse_dump_bin.sh
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
POWERINFER_DIR="$(dirname "$SCRIPT_DIR")"

BIN_DIR="${BIN_DIR:-${POWERINFER_DIR}/build_release/bin}"
MAIN_BIN="${MAIN_BIN:-${BIN_DIR}/main}"
DUMPDIR="${DUMPDIR:-${POWERINFER_DIR}/dumpbins}"
THREADS="${THREADS:-20}"
N_PREDICT="${N_PREDICT:-1}"
MODELS="${MODELS:-ReluLLaMA-7B}"
DATASETS="${DATASETS:-wiki}"
MAX_PROMPTS="${MAX_PROMPTS:-100}"

# ---- resolve model path from short key ----
resolve_model() {
    case "$1" in
        ReluLLaMA-7B)   echo "${MODEL_DIR:-$HOME/.cache/huggingface/hub/models--PowerInfer--ReluLLaMA-7B-PowerInfer-GGUF/snapshots/17b5d8a28f0377e05758a74989ce9326f5905860}/llama-7b-relu.powerinfer.gguf" ;;
        ReluLLaMA-13B)  echo "${MODEL_DIR}/prosparse-llama-2-13b.powerinfer.gguf" ;;
        ReluFalcon-40B) echo "${MODEL_DIR}/falcon-40b-relu.powerinfer.gguf" ;;
        OPT-6.7B)       echo "${MODEL_DIR}/opt-6.7b-relu.powerinfer.gguf" ;;
        OPT-30B)        echo "${MODEL_DIR}/opt-30b-relu.powerinfer.gguf" ;;
        Bamboo-7B)      echo "${MODEL_DIR}/bamboo-7b-v0.1.gguf" ;;
        *)
            if [ -f "$1" ]; then echo "$1"; else
                echo "ERROR: unknown model key '$1' and not a file path" >&2
                return 1
            fi
            ;;
    esac
}

# ---- prepare dataset to prompts file ----
ensure_dataset() {
    local ds="$1"
    local PROMPT_FILE="${POWERINFER_DIR}/datasets/${ds}_prompts.txt"
    if [ -f "$PROMPT_FILE" ]; then
        echo "$PROMPT_FILE"
        return
    fi
    mkdir -p "${POWERINFER_DIR}/datasets"
    echo "=== Downloading dataset: $ds ===" >&2
    case "$ds" in
        wiki|c4|alpaca|chatgpt)
            python3 "${SCRIPT_DIR}/download_dataset.py" "$ds" -o "$PROMPT_FILE" --max "$MAX_PROMPTS"
            ;;
        *)
            echo "Unknown dataset $ds, writing default prompt" >&2
            echo "once upon a time" > "$PROMPT_FILE"
            ;;
    esac
    echo "$PROMPT_FILE"
}

# ---- main ----
mkdir -p "$DUMPDIR"

IFS=',' read -ra MODEL_ARR <<< "$MODELS"
IFS=',' read -ra DS_ARR    <<< "$DATASETS"

FAILED=0
TOTAL=0

for model_key in "${MODEL_ARR[@]}"; do
    MODEL_PATH=$(resolve_model "$model_key")
    [ -z "$MODEL_PATH" ] && continue
    if [ ! -f "$MODEL_PATH" ]; then
        echo "SKIP: model not found: $MODEL_PATH" >&2
        continue
    fi

    # short name for filename
    model_short="$(basename "$model_key" .gguf)"

    for ds in "${DS_ARR[@]}"; do
        PROMPT_FILE=$(ensure_dataset "$ds")
        [ -z "$PROMPT_FILE" ] && continue

        echo "=== Model: $model_short  Dataset: $ds ==="

        i=0
        while IFS= read -r prompt; do
            [ -z "$prompt" ] && continue
            i=$((i + 1))
            if [ $i -gt "$MAX_PROMPTS" ]; then break; fi

            DUMPFILE="${DUMPDIR}/${model_short}-${ds}-${i}.bin"

            if [ -f "$DUMPFILE" ] && [ -s "$DUMPFILE" ]; then
                echo "  [$i] SKIP (exists: $DUMPFILE)"
                continue
            fi

            echo "  [$i] $prompt" | head -c 80
            echo ""

            set +e
            POWERINFER_DUMP_BINARY="$DUMPFILE" \
                "$MAIN_BIN" \
                -m "$MODEL_PATH" \
                -p "$prompt" \
                -n "$N_PREDICT" \
                -t "$THREADS" \
                2>&1 | tail -1
            rc=$?
            set -e

            if [ $rc -ne 0 ]; then
                echo "    ** FAILED (rc=$rc) **"
                FAILED=$((FAILED + 1))
                rm -f "$DUMPFILE"
            else
                SIZE=$(wc -c < "$DUMPFILE" 2>/dev/null || echo 0)
                echo "    -> $DUMPFILE  ($SIZE bytes)"
            fi
            TOTAL=$((TOTAL + 1))
        done < "$PROMPT_FILE"
    done
done

echo ""
echo "=== Done ==="
echo "Total runs: $TOTAL  Failed: $FAILED"
echo "Outputs: $DUMPDIR/"
ls -lh "$DUMPDIR/" 2>/dev/null | head -20
