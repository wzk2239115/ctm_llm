#!/usr/bin/env bash
# Overnight (12h) CTM memory-policy 实验编排. 串行/并行跑 6 批, 填满 5 卡.
#
# 用法 (tmux, 下班前):
#   tmux new -s ovn
#   bash scripts/overnight_memory.sh
#   # Ctrl-B D detach; 明早看 logs/overnight_*.log + csv_data/*_report.md
#
# 批次:
#   A  delay 矩阵 (7 envs 含 delay, 5 backbone x 5 seed)         ~40min  5卡
#   B  belief probe (5 卡各 1 backbone 并行, 5 seed)              ~30min  5卡并行
#   C  核心 pendulum-partial 10 seed 长训练 (坐实统计/排除欠训)   ~40min  5卡
#   D  CTM d_model robustness (64/128/256, 防超参侥幸)            ~1h     5卡
#   E  CTM memory_length robustness (4/8/16)                       ~1h     5卡
#   F  更多 POMDP (reacher-partial + delay 5/8 sweep)             ~30min  5卡
set -u
cd "$(dirname "$0")/.." || exit 1
mkdir -p logs csv_data
GPUS=(0 1 2 3 4)
log() { echo "[$(date '+%m-%d %H:%M')] $*"; }

# ---------- A: delay 矩阵 ----------
log "BATCH A: delay 矩阵 ablation"
python paper/run_memory_policy_ablation.py --procs-per-gpu 10 \
    --envs pendulum pendulum-partial pendulum-delay3 pendulum-partial-delay3 \
           cartpole-partial tworoom-state point-state \
    --seeds 0 1 2 3 4 --total-steps 100000 \
    > logs/overnight_A_delay.log 2>&1
log "A done"

# ---------- B: belief probe (5 卡并行, 每卡 1 backbone) ----------
log "BATCH B: belief probe (5 卡并行)"
i=0
for BB in mlp ctm lstm gru transformer; do
    CUDA_VISIBLE_DEVICES=${GPUS[$((i % 5))]} python paper/probe_belief_encoding.py \
        --backbones "$BB" --seeds 0 1 2 3 4 --total-steps 80000 \
        > "logs/overnight_B_probe_${BB}.log" 2>&1 &
    i=$((i + 1))
done
wait
log "B done (合并: grep 'R²(θdot)' logs/overnight_B_probe_*.log)"

# ---------- C: 核心 pendulum-partial 10 seed 长训练 ----------
log "BATCH C: 核心 pendulum-partial 10 seed x 20万步"
python paper/run_memory_policy_ablation.py --procs-per-gpu 10 \
    --envs pendulum-partial \
    --seeds 0 1 2 3 4 5 6 7 8 9 --total-steps 200000 \
    > logs/overnight_C_core.log 2>&1
log "C done"

# ---------- D: CTM d_model robustness ----------
log "BATCH D: d_model robustness (64/128/256)"
for DM in 64 128 256; do
    python paper/run_memory_policy_ablation.py --procs-per-gpu 10 \
        --envs pendulum-partial --backbones ctm lstm gru \
        --seeds 0 1 2 3 4 --d_model "$DM" --total-steps 100000 \
        > "logs/overnight_D_dm${DM}.log" 2>&1
done
log "D done"

# ---------- E: CTM memory_length robustness ----------
log "BATCH E: memory_length robustness (4/8/16)"
for ML in 4 8 16; do
    python paper/run_memory_policy_ablation.py --procs-per-gpu 10 \
        --envs pendulum-partial --backbones ctm lstm gru \
        --seeds 0 1 2 3 4 --memory_length "$ML" --total-steps 100000 \
        > "logs/overnight_E_ml${ML}.log" 2>&1
done
log "E done"

# ---------- F: 更多 POMDP ----------
log "BATCH F: 更多 POMDP (reacher-partial + delay sweep 5/8)"
python paper/run_memory_policy_ablation.py --procs-per-gpu 10 \
    --envs reacher-partial pendulum-delay5 pendulum-delay8 \
    --seeds 0 1 2 3 4 --total-steps 100000 \
    > logs/overnight_F_more.log 2>&1
log "F done"

log "ALL BATCHES DONE — 看 csv_data/memory_ablation_report.md + logs/overnight_*.log"
