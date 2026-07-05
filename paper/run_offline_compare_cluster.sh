#!/bin/bash
# 多 GPU 并行大规模 GCBC 离线对比 (算力机用)
# 每个 GPU 跑一部分 env, 最后合并 CSV → markdown report
#
# 用法:
#   bash paper/run_offline_compare_cluster.sh
#   N_SEEDS=5 bash paper/run_offline_compare_cluster.sh
#
# 前置: 算力机设代理 + git pull
#   export http_proxy="http://public-proxy.qihoo.net:3128"
#   export https_proxy="http://public-proxy.qihoo.net:3128"

set -e

ENVS=("point-state" "tworoom-state" "reacher" "reacher-partial"
      "mountaincar" "mountaincar-partial" "acrobot" "acrobot-partial"
      "pendulum" "pendulum-partial" "cartpole" "cartpole-partial")

BACKBONES="mlp ctm lstm gru transformer flash flash-shallow flash-deep"
SEEDS="${SEEDS:-0 1 2 3 4}"
COLLECT_EPS="${COLLECT_EPS:-200}"
GCBC_STEPS="${GCBC_STEPS:-2000}"
EVAL_EPS="${EVAL_EPS:-24}"
OUT_DIR="${OUT_DIR:-csv_data}"

N_GPU=$(nvidia-smi -L 2>/dev/null | wc -l)
if [ "$N_GPU" -eq 0 ]; then
    echo "No GPU detected, using CPU (single process)"
    N_GPU=1
fi
N_ENV=${#ENVS[@]}
echo "[cluster] $N_GPU GPUs, $N_ENV envs, backbones=$BACKBONES, seeds=$SEEDS"

mkdir -p "$OUT_DIR" logs

# 分片: 每个 GPU 跑 ceil(N_ENV/N_GPU) 个 env
pids=()
for ((g=0; g<N_GPU; g++)); do
    start=$((g * N_ENV / N_GPU))
    end=$(((g + 1) * N_ENV / N_GPU))
    if [ "$start" -ge "$N_ENV" ]; then break; fi
    shard=("${ENVS[@]:start:end-start}")
    shard_str=$(IFS=,; echo "${shard[*]}")
    echo "[cluster] GPU $g: envs [$start:$end] = ${shard[*]}"

    CUDA_VISIBLE_DEVICES=$g python paper/run_offline_compare.py \
        --envs ${shard[@]} \
        --backbones $BACKBONES \
        --seeds $SEEDS \
        --collect-episodes $COLLECT_EPS \
        --gcbc-steps $GCBC_STEPS \
        --eval-episodes $EVAL_EPS \
        --device cuda \
        --report "$OUT_DIR/offline_compare_gpu${g}.md" \
        > "logs/offline_compare_gpu${g}.log" 2>&1 &
    pids+=($!)
done

echo "[cluster] waiting for ${#pids[@]} GPU workers..."
wait

# 合并 CSV
echo "[cluster] merging CSVs..."
python -c "
import glob, csv, os
rows = []
for f in sorted(glob.glob('$OUT_DIR/offline_compare_gpu*.csv')):
    with open(f) as fh:
        reader = csv.DictReader(fh)
        for r in reader:
            rows.append(r)
if not rows:
    print('No CSV data found!')
    exit(1)
out = '$OUT_DIR/offline_compare_full.csv'
with open(out, 'w') as fh:
    w = csv.DictWriter(fh, fieldnames=rows[0].keys())
    w.writeheader()
    w.writerows(rows)
print(f'Merged {len(rows)} rows -> {out}')
"

echo "[cluster] DONE. CSV: $OUT_DIR/offline_compare_full.csv"
echo "[cluster] Generate markdown report with:"
echo "  python paper/merge_offline_report.py --csv $OUT_DIR/offline_compare_full.csv"
