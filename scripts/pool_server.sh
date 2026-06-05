#!/usr/bin/env bash
set -euo pipefail

exec python scripts/cluster_pool.py server "$@"
