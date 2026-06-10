#!/usr/bin/env bash
# run_via_pool.sh — Cluster pool entry point for baseline CTM tasks
#
# The pool worker chain: train_cluster.sh → run_train.sh → torchrun ... $TRAIN_ENTRY "$@"
# With TRAIN_ENTRY=this script and NPROC_PER_NODE=1, torchrun simply exec's this script.
# "$@" contains the extra_args from pool submit, which is the task module + flags.
#
# Example: "$@" = "baseline.tasks.parity.train --seed 0 --iterations 75 ..."
# This script runs: python -m baseline.tasks.parity.train --seed 0 --iterations 75 ...

set -euo pipefail

if [ $# -lt 1 ]; then
    echo "[run_via_pool] Error: no arguments provided"
    exit 1
fi

# First arg is the module, rest are flags
MODULE="$1"
shift

echo "[run_via_pool] Module: $MODULE"
echo "[run_via_pool] Args: $*"
echo "[run_via_pool] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-all}"
echo "[run_via_pool] Start: $(date)"

exec python -m "$MODULE" "$@"
