#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/run_train.sh [ENV_FILE] [extra trainer args...]

Examples:
  scripts/run_train.sh infra/envs/smoke_8xh100.env
  scripts/run_train.sh infra/envs/smoke_multinode.env --cross_layer_state 0

Multi-node:
  Run the same command on every node, changing only NODE_RANK.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

ENV_FILE="${1:-infra/envs/smoke_8xh100.env}"
if [[ $# -gt 0 ]]; then
  shift
fi

if [[ ! -f "$ENV_FILE" ]]; then
  echo "Missing env file: $ENV_FILE" >&2
  usage >&2
  exit 1
fi

ENV_KEYS=(
  NNODES NODE_RANK NPROC_PER_NODE MASTER_ADDR MASTER_PORT
  OMP_NUM_THREADS TOKENIZERS_PARALLELISM
  NCCL_DEBUG NCCL_IB_DISABLE NCCL_ASYNC_ERROR_HANDLING NCCL_SOCKET_IFNAME
  TORCH_DISTRIBUTED_DEBUG TRAIN_ARGS DRY_RUN
)

ENV_OVERRIDE_KEYS=()
ENV_OVERRIDE_VALUES=()
for key in "${ENV_KEYS[@]}"; do
  if printenv "$key" >/dev/null 2>&1; then
    ENV_OVERRIDE_KEYS+=("$key")
    ENV_OVERRIDE_VALUES+=("$(printenv "$key")")
  fi
done

set -a
# shellcheck source=/dev/null
source "$ENV_FILE"
set +a

for idx in "${!ENV_OVERRIDE_KEYS[@]}"; do
  key="${ENV_OVERRIDE_KEYS[$idx]}"
  printf -v "$key" '%s' "${ENV_OVERRIDE_VALUES[$idx]}"
  export "$key"
done

: "${NNODES:=1}"
: "${NODE_RANK:=0}"
: "${NPROC_PER_NODE:=8}"
: "${MASTER_ADDR:=127.0.0.1}"
: "${MASTER_PORT:=29500}"
: "${OMP_NUM_THREADS:=1}"
: "${TOKENIZERS_PARALLELISM:=false}"
: "${NCCL_DEBUG:=WARN}"
: "${NCCL_IB_DISABLE:=0}"
: "${NCCL_ASYNC_ERROR_HANDLING:=1}"
: "${TORCH_DISTRIBUTED_DEBUG:=OFF}"
: "${TRAIN_ARGS:=}"
: "${DRY_RUN:=0}"

export OMP_NUM_THREADS
export TOKENIZERS_PARALLELISM
export NCCL_DEBUG
export NCCL_IB_DISABLE
export NCCL_ASYNC_ERROR_HANDLING
export TORCH_DISTRIBUTED_DEBUG

echo "Launch CTM-LLM training"
echo "  env_file       : $ENV_FILE"
echo "  nnodes         : $NNODES"
echo "  node_rank      : $NODE_RANK"
echo "  nproc_per_node : $NPROC_PER_NODE"
echo "  master         : $MASTER_ADDR:$MASTER_PORT"
echo "  train_args     : $TRAIN_ARGS $*"

if [[ "$DRY_RUN" == "1" ]]; then
  echo
  echo "DRY_RUN=1, command not executed:"
  echo "torchrun --nnodes=$NNODES --node_rank=$NODE_RANK --nproc_per_node=$NPROC_PER_NODE --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT trainer/train.py $TRAIN_ARGS $*"
  exit 0
fi

exec torchrun \
  --nnodes="$NNODES" \
  --node_rank="$NODE_RANK" \
  --nproc_per_node="$NPROC_PER_NODE" \
  --master_addr="$MASTER_ADDR" \
  --master_port="$MASTER_PORT" \
  trainer/train.py \
  $TRAIN_ARGS \
  "$@"
