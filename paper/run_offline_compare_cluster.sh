#!/bin/bash
# 多 GPU + 每卡多进程 并行大规模 GCBC 离线对比 (算力机用)
#
# 每个 GPU 跑 PROCS_PER_GPU 个进程, 每个进程跑一部分 env, 最后合并 CSV
# 80GB H100 + d_model=128: PROCS_PER_GPU=4~6 (GCBC only), 2~3 (带 CEM-WM)
# 估算显存: d_model=128 模型 <0.5GB, CEM-WM(num_samples=128) ~2GB, 余量大
#
# 用法:
#   bash paper/run_offline_compare_cluster.sh
#   PROCS_PER_GPU=4 bash paper/run_offline_compare_cluster.sh   # 每卡4进程
#   PROCS_PER_GPU=6 SEEDS="0 1 2" bash paper/run_offline_compare_cluster.sh
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
PROCS_PER_GPU="${PROCS_PER_GPU:-1}"   # 每卡并行进程数, 调大加速

N_GPU=$(nvidia-smi -L 2>/dev/null | wc -l)
if [ "$N_GPU" -eq 0 ]; then
    echo "No GPU detected, using CPU (single process)"
    N_GPU=1
    PROCS_PER_GPU=1
fi
N_ENV=${#ENVS[@]}
TOTAL_PROCS=$((N_GPU * PROCS_PER_GPU))

# 自动调高 PROCS_PER_GPU 如果进程数 > env 数 (每进程至少 1 个 env)
if [ "$TOTAL_PROCS" -gt "$N_ENV" ]; then
    TOTAL_PROCS=$N_ENV
    PROCS_PER_GPU=$(( (TOTAL_PROCS + N_GPU - 1) / N_GPU ))
    echo "[cluster] capped to TOTAL_PROCS=$TOTAL_PROCS (PROCS_PER_GPU=$PROCS_PER_GPU)"
fi

echo "[cluster] $N_GPU GPUs x $PROCS_PER_GPU procs = $TOTAL_PROCS workers"
echo "[cluster] $N_ENV envs, backbones=$BACKBONES, seeds=$SEEDS"
echo "[cluster] each proc gets ~$((N_ENV / TOTAL_PROCS)) envs"

mkdir -p "$OUT_DIR" logs

# 分片: env 均匀分配到 TOTAL_PROCS 个进程, 每进程绑定到 gpu = proc % N_GPU
pids=()
for ((p=0; p<TOTAL_PROCS; p++)); do
    start=$((p * N_ENV / TOTAL_PROCS))
    end=$(((p + 1) * N_ENV / TOTAL_PROCS))
    if [ "$start" -ge "$N_ENV" ]; then break; fi
    shard=("${ENVS[@]:start:end-start}")
    gpu=$((p % N_GPU))
    echo "[cluster] proc $p -> GPU $gpu: envs [$start:$end] = ${shard[*]}"

    CUDA_VISIBLE_DEVICES=$gpu python paper/run_offline_compare.py \
        --envs ${shard[@]} \
        --backbones $BACKBONES \
        --seeds $SEEDS \
        --collect-episodes $COLLECT_EPS \
        --gcbc-steps $GCBC_STEPS \
        --eval-episodes $EVAL_EPS \
        --with-wm-cem \
        --device cuda \
        --report "$OUT_DIR/offline_compare_proc${p}.md" \
        > "logs/offline_compare_proc${p}.log" 2>&1 &
    pids+=($!)
done

echo "[cluster] waiting for ${#pids[@]} workers..."
wait

# 合并 CSV
echo "[cluster] merging CSVs..."
python -c "
import glob, csv
rows = []
for f in sorted(glob.glob('$OUT_DIR/offline_compare_proc*.csv')):
    with open(f) as fh:
        for r in csv.DictReader(fh):
            rows.append(r)
if not rows:
    print('No CSV data found!'); exit(1)
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
