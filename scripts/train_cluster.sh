#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/train_cluster.sh --config infra/clusters/h100_2nodes.env [extra trainer args...]

Run the same command on every node. The script derives NODE_RANK by matching
this machine's IPv4 address against NODE_ADDRS in the cluster config.

Overrides:
  CTM_NODE_ADDR=<ip>   Use this IP for rank matching.
  CTM_POOL_NODE_ADDRS  Optional pool lane specs, e.g. ip:0-1,ip2:0-1.
  CTM_POOL_MASTER_PORT Optional per-task torchrun master port.
  NODE_RANK=<rank>     Force rank if auto-detection is not possible.
  DRY_RUN=1            Print the torchrun command without executing.
EOF
}

CONFIG=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      CONFIG="${2:-}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      break
      ;;
  esac
done

if [[ -z "$CONFIG" || ! -f "$CONFIG" ]]; then
  echo "Missing cluster config." >&2
  usage >&2
  exit 1
fi

# shellcheck source=/dev/null
source "$CONFIG"

parse_pool_node_spec() {
  local spec="$1"
  if [[ "$spec" == *:* ]]; then
    echo "${spec%%:*}"
  else
    echo "$spec"
  fi
}

parse_pool_gpu_spec() {
  local spec="$1"
  if [[ "$spec" == *:* ]]; then
    echo "${spec#*:}"
  fi
}

expand_gpu_spec() {
  local spec="$1"
  local -a out=()
  local part start end idx
  spec="${spec//+/ }"
  spec="${spec//|/ }"
  for part in $spec; do
    if [[ "$part" =~ ^([0-9]+)-([0-9]+)$ ]]; then
      start="${BASH_REMATCH[1]}"
      end="${BASH_REMATCH[2]}"
      for ((idx=start; idx<=end; idx++)); do
        out+=("$idx")
      done
    elif [[ "$part" =~ ^[0-9]+$ ]]; then
      out+=("$part")
    fi
  done
  (IFS=','; echo "${out[*]}")
}

if [[ -n "${CTM_POOL_NODE_ADDRS:-}" ]]; then
  IFS=',' read -r -a CTM_POOL_NODE_SPECS <<< "$CTM_POOL_NODE_ADDRS"
  NODE_ADDRS=()
  CTM_POOL_LOCAL_GPU_SPEC=""
  for spec in "${CTM_POOL_NODE_SPECS[@]}"; do
    addr="$(parse_pool_node_spec "$spec")"
    NODE_ADDRS+=("$addr")
  done
  MASTER_ADDR="${NODE_ADDRS[0]}"
fi

if [[ "${#NODE_ADDRS[@]}" -eq 0 ]]; then
  echo "NODE_ADDRS is empty in $CONFIG" >&2
  exit 1
fi

detect_ipv4_addrs() {
  if [[ -n "${CTM_NODE_ADDR:-}" ]]; then
    echo "$CTM_NODE_ADDR"
    return
  fi

  if command -v hostname >/dev/null 2>&1; then
    hostname -I 2>/dev/null | tr ' ' '\n' | grep -E '^[0-9]+\.' || true
  fi
  if command -v ip >/dev/null 2>&1; then
    ip -o -4 addr show 2>/dev/null | awk '{print $4}' | cut -d/ -f1 || true
  fi
  if command -v ifconfig >/dev/null 2>&1; then
    ifconfig 2>/dev/null | awk '/inet / {print $2}' | grep -E '^[0-9]+\.' || true
  fi
}

NNODES="${#NODE_ADDRS[@]}"
MASTER_ADDR="${MASTER_ADDR:-${NODE_ADDRS[0]}}"
MASTER_PORT="${CTM_POOL_MASTER_PORT:-${MASTER_PORT:-29500}}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
TRAIN_ENV="${TRAIN_ENV:-infra/envs/smoke_multinode.env}"

if [[ -z "${NODE_RANK:-}" ]]; then
  LOCAL_ADDRS="$(detect_ipv4_addrs | sort -u)"
  NODE_RANK=""
  for idx in "${!NODE_ADDRS[@]}"; do
    if echo "$LOCAL_ADDRS" | grep -qx "${NODE_ADDRS[$idx]}"; then
      NODE_RANK="$idx"
      break
    fi
  done

  if [[ -z "$NODE_RANK" ]]; then
    echo "Could not auto-detect NODE_RANK." >&2
    echo "Local IPv4 addresses:" >&2
    echo "$LOCAL_ADDRS" >&2
    echo "Cluster NODE_ADDRS: ${NODE_ADDRS[*]}" >&2
    echo "Set CTM_NODE_ADDR=<one of NODE_ADDRS> or NODE_RANK=<rank>." >&2
    exit 1
  fi
fi

if [[ -n "${CTM_POOL_NODE_SPECS[*]:-}" ]]; then
  local_spec="${CTM_POOL_NODE_SPECS[$NODE_RANK]}"
  local_gpu_spec="$(parse_pool_gpu_spec "$local_spec")"
  if [[ -n "$local_gpu_spec" ]]; then
    CUDA_VISIBLE_DEVICES="$(expand_gpu_spec "$local_gpu_spec")"
    export CUDA_VISIBLE_DEVICES
    IFS=',' read -r -a CTM_VISIBLE_GPU_IDS <<< "$CUDA_VISIBLE_DEVICES"
    NPROC_PER_NODE="${#CTM_VISIBLE_GPU_IDS[@]}"
  fi
fi

export NNODES NODE_RANK MASTER_ADDR MASTER_PORT NPROC_PER_NODE
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"
export NCCL_ASYNC_ERROR_HANDLING="${NCCL_ASYNC_ERROR_HANDLING:-1}"
export TORCH_DISTRIBUTED_DEBUG="${TORCH_DISTRIBUTED_DEBUG:-OFF}"
export TRAIN_ENTRY="${TRAIN_ENTRY:-trainer/train.py}"
export TRAIN_ARGS="${TRAIN_ARGS:-}"
export DRY_RUN="${DRY_RUN:-0}"
if [[ -n "${NCCL_SOCKET_IFNAME:-}" ]]; then
  export NCCL_SOCKET_IFNAME
fi

echo "Cluster config"
echo "  config          : $CONFIG"
echo "  cluster         : ${CLUSTER_NAME:-unnamed}"
echo "  nodes           : ${NODE_ADDRS[*]}"
echo "  resolved rank   : $NODE_RANK/$NNODES"
echo "  visible gpus    : ${CUDA_VISIBLE_DEVICES:-all}"

exec bash scripts/run_train.sh "$TRAIN_ENV" "$@"
