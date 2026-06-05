#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-infra/clusters/h100_2nodes.env}"
if [[ $# -gt 0 ]]; then
  shift
fi

exec scripts/ctmctl pool worker "$CONFIG" "$@"
