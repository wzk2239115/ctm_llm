#!/usr/bin/env bash
# run_baseline_task.sh — Run a single baseline CTM task directly on GPU
#
# Usage:
#   bash scripts/run_baseline_task.sh <task_module> [args...]
#
# Example:
#   bash scripts/run_baseline_task.sh baseline.tasks.parity.train --seed 0 --iterations 75 ...
#
# This is the baseline equivalent of run_train.sh. It sets CUDA device
# and runs the task module directly (no torchrun needed — baseline tasks
# are single-GPU).

set -euo pipefail

if [ $# -lt 1 ]; then
    echo "Usage: $0 <task_module> [args...]"
    exit 1
fi

TASK_MODULE="$1"
shift

# Use CUDA_VISIBLE_DEVICES if set, otherwise default to GPU 0
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    echo "[baseline] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
else
    export CUDA_VISIBLE_DEVICES=0
    echo "[baseline] CUDA_VISIBLE_DEVICES=0 (default)"
fi

echo "[baseline] Task: $TASK_MODULE"
echo "[baseline] Args: $*"
echo "[baseline] Start: $(date)"

python -m "$TASK_MODULE" "$@"

echo "[baseline] Done: $(date)"
