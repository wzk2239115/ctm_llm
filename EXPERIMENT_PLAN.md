# CTM-LLM Experiment Plan

This plan is designed for pool submissions that can be run serially or in
parallel node lanes. Each experiment gets a stable `experiment_name`,
`swanlab_name`, `save_weight`, and metrics CSV under `runs/metrics/`.

The plan generator has three matrix sizes:

- `core`: compact sanity matrix, currently 33 formal experiments.
- `full`: default sufficient sweep, currently 71 formal experiments.
- `wide`: extra exploratory sweep, currently 82 formal experiments.

Use `--plan_size full` for the normal 40h harvest workflow. Use `core` when
debugging infrastructure and `wide` when the pool has spare time.

## Four-Node Pool

For a 4 x 8 H100/H800 pool, create the cluster config first:

```bash
cp infra/clusters/h100_4nodes.env.example infra/clusters/h100_4nodes.env
```

Edit `infra/clusters/h100_4nodes.env` and replace the two `TODO_NODE_*_IP`
entries. Start one pool server on the first node, then one worker on every node:

```bash
./scripts/ctmctl pool server
./scripts/ctmctl pool worker infra/clusters/h100_4nodes.env
```

Check that all four nodes are online before submitting:

```bash
./scripts/ctmctl pool status --master_addr 11.131.210.78
```

## 40h Harvest Workflow

The recommended flow is:

1. Generate and run short batch probes for every planned experiment.
2. Export a probe CSV that shows which experiment/batch combinations produced
   metrics.
3. Convert successful probes into one recommended batch size per experiment.
4. Export the final execution CSV with tuned batch sizes.
5. Submit the final plan and let the pool run.

### 1. Run Batch Probes

Before the full serial run, probe each experiment with short runs at several
batch sizes:

```bash
python scripts/experiment_plan.py batch-commands --stage all \
  --plan_size full \
  --config infra/clusters/h100_4nodes.env \
  --batch_sizes 2 4 6 8 10 12 \
  --output runs/experiment_plans/batch_tune_plan.csv
```

Run the printed submit commands one by one, or let the helper submit them
serially:

```bash
python scripts/experiment_plan.py run --stage all --batch_tune \
  --plan_size full \
  --config infra/clusters/h100_4nodes.env \
  --master_addr 11.131.210.78 \
  --port 8765 \
  --batch_sizes 2 4 6 8 10 12
```

For faster probing, run four single-node lanes in parallel:

```bash
python scripts/experiment_plan.py run-parallel --stage all --batch_tune \
  --plan_size full \
  --config infra/clusters/h100_4nodes.env \
  --master_addr 11.131.210.78 \
  --port 8765 \
  --batch_sizes 2 4 6 8 10 12 \
  --node_groups 11.131.210.78 11.131.210.3 11.131.209.154 11.131.211.9
```

For medium experiments, you can use two 2-node lanes:

```bash
python scripts/experiment_plan.py run-parallel --stage all --batch_tune \
  --plan_size full \
  --config infra/clusters/h100_4nodes.env \
  --master_addr 11.131.210.78 \
  --port 8765 \
  --batch_sizes 2 4 6 8 10 12 \
  --node_groups 11.131.210.78,11.131.210.3 11.131.209.154,11.131.211.9
```

### 2. Export Probe Status CSV

After probes finish, export every planned `(experiment, batch_size)` probe and
its observed metrics. Rows with `status=missing_metrics` did not produce a CSV
row, which usually means the run failed early, was killed, or has not completed.

```bash
python scripts/experiment_plan.py batch-report --stage all \
  --plan_size full \
  --metrics_dir runs/metrics \
  --batch_sizes 2 4 6 8 10 12 \
  --output runs/metrics/batch_probe_report.csv
```

### 3. Recommend Batches

Then recommend one batch size per experiment:

```bash
python scripts/experiment_plan.py recommend-batches \
  --metrics_dir runs/metrics \
  --target_memory_gb 80 \
  --memory_util 0.90 \
  --output runs/metrics/batch_profile.csv
```

The recommender selects the largest successful batch under
`target_memory_gb * memory_util`. If every successful probe is above the target,
it selects the lowest-memory successful run as a conservative fallback.

### 4. Export Final Plan

```bash
python scripts/experiment_plan.py final-plan --stage all \
  --plan_size full \
  --config infra/clusters/h100_4nodes.env \
  --batch_profile runs/metrics/batch_profile.csv \
  --master_addr 11.131.210.78 \
  --port 8765 \
  --output runs/experiment_plans/final_plan.csv
```

Experiments missing from `batch_profile.csv` keep the plan default
`batch_size=4`. That usually means every probe for that experiment failed
before writing metrics, so rerun a smaller probe such as `--batch_sizes 1 2`.

### 5. Run Final Plan

Serial runner for the final plan:

```bash
python scripts/experiment_plan.py run --stage all \
  --plan_size full \
  --config infra/clusters/h100_4nodes.env \
  --master_addr 11.131.210.78 \
  --port 8765 \
  --batch_profile runs/metrics/batch_profile.csv
```

If the single-node smoke probes show plenty of headroom, you can also run the
formal plan in parallel lanes. Keep node groups non-overlapping:

```bash
python scripts/experiment_plan.py run-parallel --stage all \
  --plan_size full \
  --config infra/clusters/h100_4nodes.env \
  --master_addr 11.131.210.78 \
  --port 8765 \
  --batch_profile runs/metrics/batch_profile.csv \
  --node_groups 11.131.210.78 11.131.210.3 11.131.209.154 11.131.211.9
```

## Phase 0: Smoke

Goal: verify that both model paths and metrics logging work on the cluster.

```bash
python scripts/experiment_plan.py commands --stage smoke \
  --plan_size core \
  --config infra/clusters/h100_2nodes.env \
  --output runs/experiment_plans/smoke_plan.csv
```

Run the printed `./scripts/ctmctl pool submit ...` commands one by one.

## Phase 1: Compass

Goal: establish the first fair comparison between standard Transformer and
current CTM-LLM.

```bash
python scripts/experiment_plan.py commands --stage compass \
  --plan_size full \
  --config infra/clusters/h100_2nodes.env \
  --output runs/experiment_plans/compass_plan.csv
```

Questions:

- Does CTM beat Transformer at similar layer/hidden scale?
- How much speed and memory does tick supervision cost?
- Is `best_tick` actually away from the first tick?
- How do 12/16/24-layer scales move the CTM-vs-Transformer cost curve?

## Phase 2: Tick Sweep

Goal: test whether more internal time helps before adding adaptive halting.

```bash
python scripts/experiment_plan.py commands --stage ticks \
  --plan_size full \
  --config infra/clusters/h100_2nodes.env \
  --output runs/experiment_plans/tick_plan.csv
```

Runs:

- `iterations = 1, 2, 3, 4, 6, 8, 12, 16`: raw tick scaling.
- `tick_loss_mode = mean, last, min_conf`: whether later ticks are trained
  directly, averaged, or selected by confidence.
- `tick_halt_mode=confidence`: soft confidence-weighted tick selection.
- `tick_compute_weight=0.01`: encourages useful earlier ticks.
- `tick_halt_mode=threshold`: hard threshold-style selected tick metric.
- confidence temperature and threshold sweeps for natural early/late compute.

Primary extra columns: `effective_tick`, `losses_per_tick`,
`certainties_per_tick`.

## Phase 3: ELF Horizon

Goal: stop making every tick answer exactly the same next-token task.

```bash
python scripts/experiment_plan.py commands --stage elf \
  --plan_size full \
  --config infra/clusters/h100_2nodes.env \
  --output runs/experiment_plans/elf_plan.csv
```

Runs:

- `elf_horizon_mode=none`: current next-token tick supervision.
- `elf_horizon_mode=linear`: tick `t` predicts farther future tokens.
- `elf_horizon_mode=pow2`: horizon grows as powers of two.
- `iterations=8, elf_max_horizon=8`: gives more internal time to the ELF clock.
- `tick_improve_weight=0.1`: encourages later ticks to improve next-token loss.
- Cross `iterations = 8, 12, 16` with `elf_max_horizon = 4, 8`.
- Sweep `tick_improve_weight = 0.03, 0.1, 0.3`.

Key question: do `best_tick`, `conf_tick`, and eval loss begin to separate
across easy/hard examples instead of collapsing to one tick?

## Phase 4: Cells

Goal: test the "more cells, smaller cell model" hypothesis.

```bash
python scripts/experiment_plan.py commands --stage cells \
  --plan_size full \
  --config infra/clusters/h100_2nodes.env \
  --output runs/experiment_plans/cell_plan.csv
```

This sweeps larger `d_model` with smaller `memory_hidden_dims`,
`memory_length`, and `synapse_depth`. Sparse variants additionally set:

- `cell_sparsity_mode=topk`
- `cell_topk=512`
- extra top-k points such as 256/384/512 on 1024/1536/2048 cells
- `cell_sparsity_rescale=0/1`

This first sparse-cell implementation masks CTM cell activations before
synchronization/output. It gives a clean training signal and metrics through
`active_cell_fraction`; wall-clock acceleration still depends on later replacing
dense projections with grouped or block-sparse kernels.

## Phase 5: Elegant Ablations

Goal: find low-cost simplifications before adding new mechanisms.

```bash
python scripts/experiment_plan.py commands --stage ablations \
  --plan_size full \
  --config infra/clusters/h100_2nodes.env \
  --output runs/experiment_plans/ablation_plan.csv
```

Includes `self_cond`, `cross_layer_state`, shallow synapse, short/long memory,
tiny NLM, `deep_nlms=0`, input width/head-count sensitivity, and combined
low-cost simplifications.

## Optional Sequential Runner

If the pool server is reachable and you want the helper to submit each task and
wait until workers become idle:

```bash
python scripts/experiment_plan.py run --stage smoke \
  --config infra/clusters/h100_2nodes.env \
  --master_addr 11.131.210.78 \
  --port 8765
```

## Summarize Results

After runs finish:

```bash
python scripts/experiment_plan.py summarize \
  --metrics_dir runs/metrics \
  --output runs/metrics/summary.csv
```

Primary columns:

- `loss`
- `tokens_per_sec`
- `peak_memory_mb`
- `best_tick`
- `conf_tick`
- `losses_per_tick`
- `certainties_per_tick`
- `effective_tick`
- `active_cell_fraction`
- `tick_count`
- architecture fields such as `model_type`, `d_model`, `iterations`,
  `memory_hidden_dims`, `synapse_depth`, `elf_horizon_mode`,
  `tick_halt_mode`, and `cell_sparsity_mode`

Use the summary as the first cost/performance compass before implementing
adaptive halting or sparse cell routing.
