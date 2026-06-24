#!/bin/bash
# Run MSH coprime + baseline experiments on 8 GPUs.
# Usage: bash scripts/run_msh_experiments.sh
set -e

cd "$(dirname "$0")/.."
mkdir -p logs/msh/msh08 logs/msh/msh01 logs/dtt/dtt01

COMMON="--seed 0 --iterations 50 --memory_length 25 \
  --d_model 512 --d_input 128 --n_synch_out 32 --n_synch_action 32 \
  --synapse_depth 4 --heads 4 --memory_hidden_dims 4 \
  --deep_memory --no-do_normalisation \
  --positional_embedding_type none --neuron_select_type random-pairing \
  --N_to_sort 10 --batch_size 32 --batch_size_test 32 \
  --lr 0.001 --training_iterations 100001 --warmup_steps 5000 \
  --use_scheduler --scheduler_type cosine \
  --sort_loss_mode per_tick_ce --device 0"

echo "Launching 5 experiments on GPU 0-4..."

CUDA_VISIBLE_DEVICES=0 python -m baseline.tasks.sort.train $COMMON \
  --log_dir logs/dtt/dtt01/sort_per_tick_ce_N10 \
  > logs/dtt/dtt01/sort_per_tick_ce_N10.log 2>&1 &
PID0=$!

CUDA_VISIBLE_DEVICES=1 python -m baseline.tasks.sort.train $COMMON \
  --msh_levels 2,3,5 --msh_mode coprime \
  --log_dir logs/msh/msh08/sort_N10_coprime_2x3x5 \
  > logs/msh/msh08/sort_N10_coprime_2x3x5.log 2>&1 &
PID1=$!

CUDA_VISIBLE_DEVICES=2 python -m baseline.tasks.sort.train $COMMON \
  --msh_levels 2,3 --msh_mode coprime \
  --log_dir logs/msh/msh08/sort_N10_coprime_2x3 \
  > logs/msh/msh08/sort_N10_coprime_2x3.log 2>&1 &
PID2=$!

CUDA_VISIBLE_DEVICES=3 python -m baseline.tasks.sort.train $COMMON \
  --msh_levels 3,5,7 --msh_mode coprime \
  --log_dir logs/msh/msh08/sort_N10_coprime_3x5x7 \
  > logs/msh/msh08/sort_N10_coprime_3x5x7.log 2>&1 &
PID3=$!

CUDA_VISIBLE_DEVICES=4 python -m baseline.tasks.sort.train $COMMON \
  --msh_levels 10,5,1 --msh_mode nested \
  --log_dir logs/msh/msh01/sort_N10_nested_10x5x1 \
  > logs/msh/msh01/sort_N10_nested_10x5x1.log 2>&1 &
PID4=$!

echo "GPU 0: flat baseline     PID=$PID0"
echo "GPU 1: coprime [2,3,5]   PID=$PID1"
echo "GPU 2: coprime [2,3]     PID=$PID2"
echo "GPU 3: coprime [3,5,7]   PID=$PID3"
echo "GPU 4: nested [10,5,1]   PID=$PID4"
echo ""
echo "All launched. Monitor with:"
echo "  tail -f logs/dtt/dtt01/sort_per_tick_ce_N10.log"
echo "  tail -f logs/msh/msh08/sort_N10_coprime_2x3x5.log"
echo ""
echo "Waiting for completion..."
wait
echo "All done."
