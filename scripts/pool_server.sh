#!/usr/bin/env bash
set -euo pipefail

exec scripts/ctmctl pool server "$@"
