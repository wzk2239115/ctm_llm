#!/bin/bash
set -e

cd "$(dirname "$0")/.."
mkdir -p logs/trunc

# Auto-detect GPUs
NUM_GPUS=$(python -c "import torch; print(torch.cuda.device_count() or 8)")
echo "Detected $NUM_GPUS GPUs"

# ─── Config (env-overridable) ───
# Smoke:  ITERS=5001 TRACK=200 SAVE=5000 bash scripts/run_truncation_bptt.sh
# Full:   bash scripts/run_truncation_bptt.sh        (defaults below)
ITERS=${ITERS:-100001}
TRACK=${TRACK:-1000}
SAVE=${SAVE:-10000}
N=${N:-30}                 # the real test (flat/per_tick_ce collapses here)
SK_ITERS=${SK_ITERS:-5}    # Sinkhorn iterations
SK_TAU0=${SK_TAU0:-2.0}    # start soft
SK_TAU1=${SK_TAU1:-0.1}    # anneal sharp

echo "N=$N  iters=$ITERS  sinkhorn: iters=$SK_ITERS tau $SK_TAU0 -> $SK_TAU1"

# ─── Common args (sort task, per_tick family = truncation-native) ───
# NOTE: per_tick_sinkhorn is per-tick-local → each tick self-supervised →
#       gradient truncation (bp_steps) costs no information. bp_steps=1 is the target.
COMMON="--seed 0 --iterations 50 --memory_length 25 --d_model 512 --d_input 128 \
  --n_synch_out 32 --n_synch_action 32 --synapse_depth 4 --heads 4 --memory_hidden_dims 4 \
  --deep_memory --no-do_normalisation --positional_embedding_type none \
  --neuron_select_type random-pairing --N_to_sort $N --batch_size 32 --batch_size_test 32 \
  --lr 0.001 --training_iterations $ITERS --warmup_steps 5000 --use_scheduler \
  --scheduler_type cosine --weight_decay 0.0 --gradient_clipping -1 \
  --track_every $TRACK --save_every $SAVE --no-reload --device 0"

# Sinkhorn args shared by all per_tick_sinkhorn runs
SK_ARGS="--sort_loss_mode per_tick_sinkhorn --sinkhorn_iters $SK_ITERS \
  --sinkhorn_tau $SK_TAU0 --sinkhorn_tau_min $SK_TAU1 --sinkhorn_anneal linear"

# ─── Experiment matrix: name|extra_args ───
#   per_tick_ce  = control: no permutation structure → expect collapse at N=30
#   per_tick_sinkhorn = fix: + Sinkhorn bijection → should hold across bp_steps
EXPERIMENTS=(
  # control: per_tick_ce (collapses at N=30) at 3 bp_steps
  "ce_bp50|  --sort_loss_mode per_tick_ce --bp_steps 50 --log_dir logs/trunc/N${N}_ce_bp50"
  "ce_bp5|   --sort_loss_mode per_tick_ce --bp_steps 5  --log_dir logs/trunc/N${N}_ce_bp5"
  "ce_bp1|   --sort_loss_mode per_tick_ce --bp_steps 1  --log_dir logs/trunc/N${N}_ce_bp1"
  # the fix: per_tick_sinkhorn — sweep bp_steps = the memory/accuracy Pareto frontier
  "sk_bp50|  $SK_ARGS --bp_steps 50 --log_dir logs/trunc/N${N}_sk_bp50"
  "sk_bp25|  $SK_ARGS --bp_steps 25 --log_dir logs/trunc/N${N}_sk_bp25"
  "sk_bp10|  $SK_ARGS --bp_steps 10 --log_dir logs/trunc/N${N}_sk_bp10"
  "sk_bp5|   $SK_ARGS --bp_steps 5  --log_dir logs/trunc/N${N}_sk_bp5"
  "sk_bp3|   $SK_ARGS --bp_steps 3  --log_dir logs/trunc/N${N}_sk_bp3"
  "sk_bp1|   $SK_ARGS --bp_steps 1  --log_dir logs/trunc/N${N}_sk_bp1"
)

GPU=0
PIDS=()
for entry in "${EXPERIMENTS[@]}"; do
  name="${entry%%|*}"
  args="${entry#*|}"

  if [ $GPU -ge $NUM_GPUS ]; then
    echo "Waiting for a GPU slot..."
    wait -n 2>/dev/null || wait
    GPU=0
  fi

  echo "[GPU $GPU] Launching: $name"
  CUDA_VISIBLE_DEVICES=$GPU python -m baseline.tasks.sort.train \
    $COMMON $args >"logs/trunc/${name}.log" 2>&1 &
  PIDS+=($!)
  GPU=$((GPU + 1))
done

echo "All ${#EXPERIMENTS[@]} experiments launched. Waiting for completion..."
wait
echo "Done. Results in logs/trunc/ (checkpoints hold acc/loss curves + peak_memory_gb)."
echo "Quick summary:  python -c \"import torch,glob; \\
  [print(f'{d.split('/')[-1]:<14} bp={torch.load(d+'/checkpoint.pt',map_location='cpu',weights_only=False)['bp_steps']} best_acc={max(torch.load(d+'/checkpoint.pt',map_location='cpu',weights_only=False)['test_accuracies'])*100:.1f}% mem={torch.load(d+'/checkpoint.pt',map_location='cpu',weights_only=False).get('peak_memory_gb',0):.2f}GB') for d in sorted(glob.glob('logs/trunc/N${N}_*'))]\""
