#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/dispatch_cluster.sh --config infra/clusters/h100_2nodes.env [extra trainer args...]

Starts the same cluster job on every node in NODE_ADDRS via SSH:
  cd REPO_DIR && git pull && nohup bash scripts/train_cluster.sh --config CONFIG ...

Environment:
  DRY_RUN=1            Print SSH commands without executing.
  SSH_OPTS="..."       Extra SSH options.
  RUN_LOG_DIR=logs     Remote log directory inside REPO_DIR.
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

if [[ "${#NODE_ADDRS[@]}" -eq 0 ]]; then
  echo "NODE_ADDRS is empty in $CONFIG" >&2
  exit 1
fi

: "${REPO_DIR:=/home/jovyan/h800fast/wangzekai/ctm_llm}"
: "${RUN_LOG_DIR:=logs}"
: "${DRY_RUN:=0}"
: "${SSH_OPTS:=}"

CONFIG_PATH="$CONFIG"
STAMP="$(date +%Y%m%d_%H%M%S)"
EXTRA_ARGS="$*"

echo "Dispatch CTM-LLM cluster job"
echo "  config  : $CONFIG_PATH"
echo "  repo    : $REPO_DIR"
echo "  nodes   : ${NODE_ADDRS[*]}"
echo "  args    : $EXTRA_ARGS"

for addr in "${NODE_ADDRS[@]}"; do
  target="$addr"
  if [[ -n "${SSH_USER:-}" ]]; then
    target="${SSH_USER}@${addr}"
  fi

  log_file="${RUN_LOG_DIR}/${CLUSTER_NAME:-ctm_cluster}_${addr}_${STAMP}.log"
  remote_cmd="cd '$REPO_DIR' && git pull && mkdir -p '$RUN_LOG_DIR' && nohup bash scripts/train_cluster.sh --config '$CONFIG_PATH' $EXTRA_ARGS > '$log_file' 2>&1 & echo started '$log_file'"

  echo
  echo "[$addr] $remote_cmd"
  if [[ "$DRY_RUN" != "1" ]]; then
    # shellcheck disable=SC2086
    ssh $SSH_OPTS "$target" "$remote_cmd"
  fi
done
