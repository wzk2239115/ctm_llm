#!/bin/bash
set -e

cd "$(dirname "$0")/.."
mkdir -p logs/msh/msh08 logs/msh/msh01 logs/dtt/dtt01

# Auto-detect GPUs
NUM_GPUS=$(python -c "import torch; print(torch.cuda.device_count() or 8)")
echo "Detected $NUM_GPUS GPUs"

# Experiment definitions: "name|extra_args"
EXPERIMENTS=(
  "dtt01_flat|sort_loss_mode=per_tick_ce --log_dir logs/dtt/dtt01/sort_per_tick_ce_N10"
  "coprime_2x3x5|msh_levels=2,3,5 --msh_mode coprime --log_dir logs/msh/msh08/sort_N10_coprime_2x3x5"
  "coprime_2x3|msh_levels=2,3 --msh_mode coprime --log_dir logs/msh/msh08/sort_N10_coprime_2x3"
  "coprime_3x5x7|msh_levels=3,5,7 --msh_mode coprime --log_dir logs/msh/msh08/sort_N10_coprime_3x5x7"
  "nested_10x5x1|msh_levels=10,5,1 --msh_mode nested --log_dir logs/msh/msh01/sort_N10_nested_10x5x1"
)

COMMON_ARGS="--seed 0 --iterations 50 --memory_length 25 --d_model 512 --d_input 128 --n_synch_out 32 --n_synch_action 32 --synapse_depth 4 --heads 4 --memory_hidden_dims 4 --deep_memory --no-do_normalisation --positional_embedding_type none --neuron_select_type random-pairing --N_to_sort 10 --batch_size 32 --batch_size_test 32 --lr 0.001 --training_iterations 100001 --warmup_steps 5000 --use_scheduler --scheduler_type cosine --weight_decay 0.0 --gradient_clipping -1 --track_every 1000 --save_every 10000 --no-reload --sort_loss_mode per_tick_ce"

GPU=0
for entry in "${EXPERIMENTS[@]}"; do
  name="${entry%%|*}"
  args="${entry#*|}"

  if [ $GPU -ge $NUM_GPUS ]; then
    echo "Waiting for GPU slot..."
    wait -n 2>/dev/null || wait
    GPU=0
  fi

  echo "[GPU $GPU] Launching: $name"
  CUDA_VISIBLE_DEVICES=$GPU python -m baseline.tasks.sort.train \
    $COMMON_ARGS --$args --device 0 \
    > "logs/${name}.log" 2>&1 &

  GPU=$((GPU + 1))
done

echo "All launched. Waiting for completion..."
wait
echo "Done. Check logs/*.log for results."
