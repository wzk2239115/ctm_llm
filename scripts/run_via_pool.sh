#!/usr/bin/env bash
# run_via_pool.sh — Cluster pool entry point for baseline CTM tasks
#
# The pool worker chain: train_cluster.sh -> run_train.sh -> torchrun ... $TRAIN_ENTRY "$@"
# With TRAIN_ENTRY=this script and NPROC_PER_NODE=1, torchrun simply exec's this script.
# "$@" contains the extra_args from pool submit, which is the task module + flags.
#
# Example: "$@" = "baseline.tasks.parity.train --seed 0 --iterations 75 ..."
# This script runs: python -m baseline.tasks.parity.train --seed 0 --iterations 75 ...
#
# Always writes combined stdout+stderr to runs/logs/pool_last_run.log so the
# submit script can cat it on failure.  Also writes .fail.json when
# $CTM_EXPERIMENT_NAME is set.

set -euo pipefail

if [ $# -lt 1 ]; then
    echo "[run_via_pool] Error: no arguments provided"
    exit 1
fi

MODULE="$1"
shift

echo "[run_via_pool] Module: $MODULE"
echo "[run_via_pool] Args: $*"
echo "[run_via_pool] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-all}"
echo "[run_via_pool] Start: $(date)"

LOGDIR="runs/logs"
mkdir -p "$LOGDIR"
LOGPATH="$LOGDIR/pool_last_run.log"

set +o pipefail
set +e
python -m "$MODULE" "$@" 2>&1 | tee "$LOGPATH"
RC=${PIPESTATUS[0]}
set -e
set -o pipefail

if [ $RC -ne 0 ]; then
    echo "[run_via_pool] exit_code=$RC" >> "$LOGPATH"
    if [ -n "${CTM_EXPERIMENT_NAME:-}" ]; then
        METRICS_DIR="${CTM_METRICS_DIR:-runs/metrics}"
        FAIL_PATH="$METRICS_DIR/${CTM_EXPERIMENT_NAME}.fail.json"
        mkdir -p "$METRICS_DIR"
        CTM_EXPERIMENT_NAME="$CTM_EXPERIMENT_NAME" \
        POOL_EXIT_CODE="$RC" \
        POOL_LOG="$LOGPATH" \
        POOL_FAIL_PATH="$FAIL_PATH" \
        python3 -c "
import json, os, sys, time
exp_name = os.environ['CTM_EXPERIMENT_NAME']
try:
    with open(os.environ['POOL_LOG']) as f:
        stderr_text = f.read()[-4000:]
except Exception:
    stderr_text = '<no log captured>'
rc = int(os.environ['POOL_EXIT_CODE'])
status = 'oom' if rc == 137 else 'failed'
payload = {
    'experiment_name': exp_name,
    'status': status,
    'rank': 0,
    'error_type': 'SubprocessError',
    'error': stderr_text,
    'time': time.strftime('%Y-%m-%d %H:%M:%S'),
    'git_commit': 'unknown',
}
path = os.environ['POOL_FAIL_PATH']
os.makedirs(os.path.dirname(path), exist_ok=True)
with open(path, 'w') as f:
    json.dump(payload, f, indent=2, ensure_ascii=False)
print(f'[run_via_pool] wrote failure report: {path}', file=sys.stderr)
"
    fi
fi

exit $RC
