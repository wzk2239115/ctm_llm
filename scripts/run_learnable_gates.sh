#!/bin/bash
set -e
cd "$(dirname "$0")/.."
mkdir -p logs/lg

NUM_GPUS=$(python -c "import torch; print(torch.cuda.device_count() or 8)")
echo "Detected $NUM_GPUS GPUs"

COMMON="--seed 0 --iterations 50 --memory_length 25 --d_model 512 --d_input 128 --n_synch_out 32 --n_synch_action 32 --synapse_depth 4 --heads 4 --memory_hidden_dims 4 --deep_memory --no-do_normalisation --positional_embedding_type none --neuron_select_type random-pairing --N_to_sort 10 --batch_size 32 --batch_size_test 32 --lr 0.001 --training_iterations 100001 --warmup_steps 5000 --use_scheduler --scheduler_type cosine --weight_decay 0.0 --gradient_clipping -1 --track_every 1000 --save_every 10000 --no-reload --sort_loss_mode per_tick_ce --device 0"

# name|extra_args
EXPERIMENTS=(
  "lg_coprime_init|--msh_levels 2,3,5 --msh_mode learnable --msh_gate_init coprime --log_dir logs/lg/learnable_coprime_init"
  "lg_random_init|--msh_levels 2,3,5 --msh_mode learnable --msh_gate_init random --log_dir logs/lg/learnable_random_init"
  "lg_uniform_init|--msh_levels 2,3,5 --msh_mode learnable --msh_gate_init uniform --log_dir logs/lg/learnable_uniform_init"
  "lg_coprime_sparsity|--msh_levels 2,3,5 --msh_mode learnable --msh_gate_init coprime --msh_gate_sparsity 0.05 --log_dir logs/lg/learnable_coprime_sparse"
  "lg_fixed_coprime|--msh_levels 2,3,5 --msh_mode coprime --log_dir logs/lg/fixed_coprime_2x3x5"
)

GPU=0
for entry in "${EXPERIMENTS[@]}"; do
  name="${entry%%|*}"
  args="${entry#*|}"
  if [ $GPU -ge $NUM_GPUS ]; then wait -n 2>/dev/null || wait; GPU=0; fi
  echo "[GPU $GPU] $name"
  CUDA_VISIBLE_DEVICES=$GPU python -m baseline.tasks.sort.train $COMMON --$args &
  GPU=$((GPU + 1))
done
wait
echo "Done."
