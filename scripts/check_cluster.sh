#!/usr/bin/env bash
set -euo pipefail

ENV_FILE="${1:-infra/envs/smoke_8xh100.env}"
if [[ -f "$ENV_FILE" ]]; then
  ENV_KEYS=(NPROC_PER_NODE MASTER_ADDR MASTER_PORT)
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
fi

: "${NPROC_PER_NODE:=8}"
: "${MASTER_ADDR:=127.0.0.1}"
: "${MASTER_PORT:=29500}"

echo "CTM-LLM cluster check"
echo "  cwd       : $(pwd)"
echo "  host      : $(hostname)"
echo "  master    : $MASTER_ADDR:$MASTER_PORT"
echo "  nproc     : $NPROC_PER_NODE"

echo
echo "[git]"
git rev-parse --short HEAD
git status --short

echo
echo "[gpu]"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
  gpu_count="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)"
  if [[ "$gpu_count" -lt "$NPROC_PER_NODE" ]]; then
    echo "WARNING: visible GPUs ($gpu_count) < NPROC_PER_NODE ($NPROC_PER_NODE)" >&2
  fi
else
  echo "WARNING: nvidia-smi not found" >&2
fi

echo
echo "[paths]"
for path in dataset_data model_tokenizer; do
  if [[ -e "$path" ]]; then
    target="$(readlink "$path" || true)"
    echo "ok: $path ${target:+-> $target}"
  else
    echo "MISSING: $path" >&2
  fi
done

if [[ -f dataset_data/sft_t2a_mini.parquet ]]; then
  echo "ok: dataset_data/sft_t2a_mini.parquet"
else
  echo "MISSING: dataset_data/sft_t2a_mini.parquet" >&2
fi

echo
echo "[python]"
python - <<'PY'
import importlib.util
import torch

print("python ok")
print("torch", torch.__version__)
print("cuda available", torch.cuda.is_available())
print("cuda device count", torch.cuda.device_count())

for name in ["transformers", "pandas", "pyarrow", "tensorboard"]:
    print(name, "ok" if importlib.util.find_spec(name) else "missing")
print("swanlab", "ok" if importlib.util.find_spec("swanlab") else "optional-missing")
PY

echo
echo "[network]"
python - <<PY
import socket
host = "${MASTER_ADDR}"
port = int("${MASTER_PORT}")
try:
    addr = socket.getaddrinfo(host, port, socket.AF_INET, socket.SOCK_STREAM)[0][4]
    print("master ipv4", addr)
except Exception as exc:
    print("WARNING: cannot resolve/connect master address hint:", exc)
PY
