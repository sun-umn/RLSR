#!/bin/bash
# Train one method on HotPotQA (unified comparison).
#
# Usage: ./launch.sh <rlvr|rlcr|rlsr|rlvr-ta>
#
# Run with 4x A100-40GB GPUs visible (select via CUDA_VISIBLE_DEVICES).
# Outputs default to data/<METHOD>-hotpot (see configs/Qwen-7B/hotpot/).
set -e
cd "$(dirname "$0")"

METHOD=${1:-""}
case $METHOD in
    rlvr)    CONFIG=RLVR;    PORT=29500 ;;
    rlcr)    CONFIG=RLCR;    PORT=29501 ;;
    rlsr)    CONFIG=RLSR;    PORT=29502 ;;
    rlvr-ta) CONFIG=RLVR-TA; PORT=29503 ;;
    *) echo "Usage: ./launch.sh <rlvr|rlcr|rlsr|rlvr-ta>"; exit 1 ;;
esac

export ACCELERATE_CONFIG_FILE="$PWD/deepspeed.yaml"
uv run accelerate launch \
  --num_processes 4 \
  --main_process_port "$PORT" \
  --config_file deepspeed.yaml \
  rl_runner.py \
  --config "configs/Qwen-7B/hotpot/${CONFIG}.yaml"
