#!/bin/bash
# RLSR evaluation (RLCR protocol + AURC metrics). One checkpoint per invocation.
#
# Usage: ./eval.sh <model_path_or_hf_id> <dataset|all|all-extended> <gpu_ids> <method>
#
#   dataset  one of: hotpot-eval-em hotpot-vanilla-eval-em gsm8k math-500
#            commonsenseqa gpqa simpleqa trivia medqa-eval mbpp
#            "all" = RLCR's official 8; "all-extended" adds medqa-eval and mbpp
#   gpu_ids  comma-separated PHYSICAL device ids, e.g. "0" or "0,1"; the script
#            exports them as CUDA_VISIBLE_DEVICES. If a scheduler already set
#            CUDA_VISIBLE_DEVICES for your job, pass those same ids (e.g.
#            "$CUDA_VISIBLE_DEVICES") — a different value would escape your
#            assignment. All listed GPUs serve both the generator and (for the
#            judge datasets) the Llama-3.1-8B judge, tensor-parallel. Use
#            2x A100-40GB GPUs.
#   method   rlvr | rlcr | rlsr  -> tabc_long prompt (the matched-template runs)
#            rlvr-ta             -> gen prompt (faithful original-RLCR RLVR;
#                                   accuracy-only: it emits no trained confidence)
#            base                -> tac prompt
#
# Environment knobs (defaults are correct for A100-40GB):
#   EVAL_GPU_MEM_UTIL         generator vLLM memory fraction (default 0.9)
#   EVAL_JUDGE_MAX_MODEL_LEN  judge context cap (default 8192; judge prompts
#                             stay under ~2k tokens)
#
# Examples:
#   uv run bash eval.sh /path/to/checkpoint-300 all 0,1 rlsr
#   uv run bash eval.sh Qwen/Qwen2.5-7B hotpot-eval-em 0 base
#
# LoRA checkpoints (adapter_config.json present) are merged automatically into
# <run>-evalout/<checkpoint>-merged before serving; results land under that
# model's eval/ dir (HuggingFace model IDs go to runs/base-models/<id>/eval/).
# Already-evaluated datasets are skipped (cached outputs); delete the dataset's
# outputs/ dir to force a re-run.

TASK_DATASETS=("hotpot-eval-em" "hotpot-vanilla-eval-em" "gsm8k" "math-500" "commonsenseqa" "gpqa" "simpleqa" "trivia")
# Extended OOD (beyond RLCR's official 8): MedQA (MCQ, rule-scored) and MBPP
# (programming; correctness = executing reference unit tests). Build the local
# datasets once first: uv run python data_scripts/medqa.py
# && uv run python data_scripts/mbpp.py
EXTENDED_DATASETS=("medqa-eval" "mbpp")

MODEL_PATH=${1:-""}
DATASET=${2:-"hotpot-eval-em"}
GPU_IDS=${3:-0}
METHOD=${4:-rlsr}

if [ -z "$MODEL_PATH" ]; then
    echo "Usage: ./eval.sh <model_path_or_hf_id> <dataset|all|all-extended> <gpu_ids> <method>"
    exit 1
fi

# Tensor parallelism = number of GPUs given (generator and judge alike)
NUM_GPUS=$(echo "$GPU_IDS" | tr ',' '\n' | wc -l)
export EVAL_TP=$NUM_GPUS
export EVAL_JUDGE_TP=$NUM_GPUS

# Merge LoRA checkpoints (ZeRO-2: the adapter in the checkpoint is complete).
# IMPORTANT: merged models live OUTSIDE the training run directory, in a sibling
# <run>-evalout/ tree. A dir named checkpoint-N-merged inside the run dir would
# match the HF Trainer's checkpoint-rotation glob and get raw checkpoints (and
# itself, eval results included) deleted by save_total_limit.
IS_HF_MODEL=false
if [ -d "$MODEL_PATH" ]; then
    MODEL_PATH=$(realpath "$MODEL_PATH")
    if [ -f "$MODEL_PATH/adapter_config.json" ]; then
        BASE_NAME=$(basename "$MODEL_PATH")
        PARENT_DIR=$(dirname "$MODEL_PATH")
        case "$BASE_NAME" in
            checkpoint-*) MERGED_PATH="${PARENT_DIR}-evalout/${BASE_NAME}-merged" ;;
            *)            MERGED_PATH="${MODEL_PATH}-evalout/final-merged" ;;
        esac
        mkdir -p "$(dirname "$MERGED_PATH")"
        if [ -f "$MERGED_PATH/config.json" ]; then
            echo "Using existing merged model: $MERGED_PATH"
        else
            if ls "$MODEL_PATH"/global_step*/zero_pp_rank_*_model_states.pt 1>/dev/null 2>&1; then
                echo "Error: ZeRO-3 checkpoint detected; reconstruct full weights with the"
                echo "checkpoint's zero_to_fp32.py before merging. The release recipe uses ZeRO-2."
                exit 1
            fi
            echo "Merging LoRA adapter into base model..."
            # Judge success by the artifact, not the exit code: library teardown
            # (deepspeed/triton) can crash the process after a successful save.
            python merge_lora.py "$MODEL_PATH" --output_dir "$MERGED_PATH"
            if [ ! -f "$MERGED_PATH/config.json" ]; then
                echo "Error: LoRA merge failed (no $MERGED_PATH/config.json)"; exit 1
            fi
        fi
        MODEL_PATH="$MERGED_PATH"
    fi
else
    echo "Treating '$MODEL_PATH' as a HuggingFace model ID"
    IS_HF_MODEL=true
fi

# System prompt from method (must match the prompt the model was trained with)
case $METHOD in
    rlcr|rlsr|rlvr) SYS_PROMPT="tabc_long" ;;   # matched-template runs
    rlvr-ta)        SYS_PROMPT="gen" ;;          # original RLCR elicitation
    base)           SYS_PROMPT="tac" ;;
    *) echo "Unknown method '$METHOD' (rlvr|rlcr|rlsr|rlvr-ta|base)"; exit 1 ;;
esac

if [ "$DATASET" == "all" ]; then
    DATASETS_TO_EVAL=("${TASK_DATASETS[@]}")
elif [ "$DATASET" == "all-extended" ]; then
    DATASETS_TO_EVAL=("${TASK_DATASETS[@]}" "${EXTENDED_DATASETS[@]}")
else
    DATASETS_TO_EVAL=("$DATASET")
fi

if [ "$IS_HF_MODEL" = true ]; then
    EVAL_BASE_DIR="$(pwd)/runs/base-models/$(echo "$MODEL_PATH" | sed 's/\//-/g')/eval"
else
    EVAL_BASE_DIR="${MODEL_PATH}/eval"
fi
mkdir -p "$EVAL_BASE_DIR/configs"
EVAL_BASE_DIR=$(realpath "$EVAL_BASE_DIR")

echo "========================================="
echo "Model: $MODEL_PATH"
echo "Method: $METHOD (sys_prompt: $SYS_PROMPT)"
echo "GPUs: $GPU_IDS (TP=$NUM_GPUS, judge TP=$NUM_GPUS)"
echo "Datasets: ${DATASETS_TO_EVAL[@]}"
echo "========================================="

create_eval_config() {
    local dataset_name=$1
    local config_path=$2
    local check_fn="confidence_verifier"
    local hash_key="problem"
    case $dataset_name in
        "hotpot-eval-em")         dataset_hf="mehuldamani/hotpot_qa" ;;
        "hotpot-vanilla-eval-em") dataset_hf="data/hotpot_qa_vanilla" ;;
        "gsm8k")                  dataset_hf="data/gsm8k" ;;
        "math-500")               dataset_hf="HuggingFaceH4/MATH-500" ;;
        "commonsenseqa")          dataset_hf="data/commonsenseqa"; check_fn="llm_confidence_verifier" ;;
        "gpqa")                   dataset_hf="data/gpqa";          check_fn="llm_confidence_verifier" ;;
        "simpleqa")               dataset_hf="basicv8vc/SimpleQA"; check_fn="llm_confidence_verifier" ;;
        "trivia")                 dataset_hf="data/trivia";        check_fn="llm_confidence_verifier"; hash_key="question" ;;
        "medqa-eval")             dataset_hf="data/medqa_4options" ;;
        "mbpp")                   dataset_hf="data/mbpp" ;;
        *) echo "Unknown dataset $dataset_name"; return 1 ;;
    esac
    cat > "$config_path" <<EOF
[
{
    "dataset_name": "$dataset_hf",
    "hash_key": "$hash_key",
    "store_name": "${EVAL_BASE_DIR}/outputs/${dataset_name}",
    "gpu_memory_utilization": ${EVAL_GPU_MEM_UTIL:-0.9},
    "log_path": "${EVAL_BASE_DIR}/results/${dataset_name}"
},
{
    "name": "$(basename $MODEL_PATH)",
    "model": "$MODEL_PATH",
    "check_fn": "$check_fn",
    "sys_prompt_name": "$SYS_PROMPT",
    "vllm_task": ["confidence_at_end","ans_at_end"]
}
]
EOF
}

for dataset in "${DATASETS_TO_EVAL[@]}"; do
    echo ""
    echo "=== Evaluating on: $dataset ==="
    config_path="${EVAL_BASE_DIR}/configs/${dataset}.json"
    create_eval_config "$dataset" "$config_path" || continue
    # Code answers need a larger answer-reprompt budget than the 50-token default
    EXTRA_ENV=""
    case $dataset in mbpp) EXTRA_ENV="ANS_REPROMPT_MAX_TOKENS=${ANS_REPROMPT_MAX_TOKENS:-1024}" ;; esac
    env $EXTRA_ENV CUDA_VISIBLE_DEVICES=$GPU_IDS python evaluation.py --config "$config_path" \
        && echo "OK: $dataset" || echo "FAILED: $dataset"
done

echo ""
echo "Results: ${EVAL_BASE_DIR}/results/<dataset>/{metrics.json,aurc_metrics.json}"
