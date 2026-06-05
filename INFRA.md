# CTM-LLM Training Infra

This repo uses `torchrun` for both single-node and multi-node training.

## Files

- `scripts/run_train.sh`: launches `trainer/train.py` with an env template.
- `scripts/train_cluster.sh`: launches from a cluster node list and auto-detects node rank.
- `scripts/cluster_pool.py`: no-SSH pool server/worker/task submitter.
- `scripts/pool_server.sh`, `scripts/pool_worker.sh`, `scripts/pool_submit.sh`, `scripts/pool_status.sh`: short wrappers for the pool.
- `scripts/dispatch_cluster.sh`: SSH dispatch from one node to every node in a node list.
- `scripts/check_cluster.sh`: checks GPUs, symlinks, Python packages, and master address.
- `infra/clusters/h100_2nodes.env`: current 2-node cluster table.
- `infra/envs/smoke_8xh100.env`: single-node 8xH100 smoke test.
- `infra/envs/smoke_multinode.env`: multi-node template.
- `infra/envs/ablation_8xh100.env`: single-node ablation template.

## Single Node 8xH100

```bash
cd /home/jovyan/h800fast/wangzekai/ctm_llm
git pull

bash scripts/check_cluster.sh infra/envs/smoke_8xh100.env
DRY_RUN=1 bash scripts/run_train.sh infra/envs/smoke_8xh100.env
bash scripts/run_train.sh infra/envs/smoke_8xh100.env
```

TensorBoard is enabled by default:

```bash
tensorboard --logdir runs --host 0.0.0.0 --port 6006
```

Disable TensorBoard if needed:

```bash
bash scripts/run_train.sh infra/envs/smoke_8xh100.env --no_tensorboard
```

Enable SwanLab only when it is installed and wanted:

```bash
bash scripts/run_train.sh infra/envs/smoke_8xh100.env --use_swanlab
```

## Multi Node With Node List

The preferred workflow is to maintain one cluster table:

```bash
infra/clusters/h100_2nodes.env
```

Current nodes:

```text
rank 0 / master: 11.131.210.78
rank 1         : 11.131.210.3
```

## No-SSH Pool Workflow

Use this when you manually open one terminal on each machine and want the master
to show which machines are online, then submit a task without SSH.

On master `11.131.210.78`, terminal 1:

```bash
cd /home/jovyan/h800fast/wangzekai/ctm_llm
git pull
bash scripts/pool_server.sh
```

The server prints the pool when workers come online and when they acknowledge a
task.

On every node, including master and `11.131.210.3`, open one terminal:

```bash
cd /home/jovyan/h800fast/wangzekai/ctm_llm
git pull
bash scripts/pool_worker.sh infra/clusters/h100_2nodes.env
```

Each worker prints its detected address/rank, then prints a message when it
receives a task.

On master, terminal 2, submit a task:

```bash
bash scripts/pool_submit.sh infra/clusters/h100_2nodes.env
```

The submit command waits for acknowledgements by default and prints which nodes
started the task. Use `--wait 0` if you only want to submit.

Pass trainer overrides after the config:

```bash
bash scripts/pool_submit.sh infra/clusters/h100_2nodes.env \
  --cross_layer_state 0 --iterations 5 --swanlab_name cross0-iter5
```

Check pool state any time:

```bash
bash scripts/pool_status.sh --master_addr 11.131.210.78
```

Run the same command on every node. The script auto-detects `NODE_RANK` from
the machine's IPv4 address:

```bash
cd /home/jovyan/h800fast/wangzekai/ctm_llm
git pull

bash scripts/train_cluster.sh --config infra/clusters/h100_2nodes.env
```

If auto-detection fails because the visible network address differs, force the
matching address once:

```bash
CTM_NODE_ADDR=11.131.210.3 \
  bash scripts/train_cluster.sh --config infra/clusters/h100_2nodes.env
```

Dry run before launching:

```bash
DRY_RUN=1 bash scripts/train_cluster.sh --config infra/clusters/h100_2nodes.env
```

## Dispatch From Master

From the master node, start all nodes via SSH:

```bash
cd /home/jovyan/h800fast/wangzekai/ctm_llm
git pull

DRY_RUN=1 bash scripts/dispatch_cluster.sh --config infra/clusters/h100_2nodes.env
bash scripts/dispatch_cluster.sh --config infra/clusters/h100_2nodes.env
```

The dispatch script runs this remotely on each node:

```bash
cd /home/jovyan/h800fast/wangzekai/ctm_llm
git pull
nohup bash scripts/train_cluster.sh --config infra/clusters/h100_2nodes.env ... &
```

Remote logs are written under `logs/`.

You can pass trainer overrides through dispatch:

```bash
bash scripts/dispatch_cluster.sh --config infra/clusters/h100_2nodes.env \
  --cross_layer_state 0 --iterations 5 --swanlab_name cross0-iter5
```

## Manual Multi Node Fallback

Pick one node as the rendezvous/master node. Use its IPv4 address as `MASTER_ADDR`.

Example for two 8xH100 nodes:

On node 0:

```bash
cd /home/jovyan/h800fast/wangzekai/ctm_llm
git pull

MASTER_ADDR=<node0-ip> NODE_RANK=0 NNODES=2 \
  bash scripts/run_train.sh infra/envs/smoke_multinode.env
```

On node 1:

```bash
cd /home/jovyan/h800fast/wangzekai/ctm_llm
git pull

MASTER_ADDR=<node0-ip> NODE_RANK=1 NNODES=2 \
  bash scripts/run_train.sh infra/envs/smoke_multinode.env
```

For more nodes, keep `MASTER_ADDR` and `NNODES` the same everywhere, and set
`NODE_RANK=0,1,2,...`.

## Parallel Ablations

If you have several independent 8xH100 machines, it is often faster to run one
ablation per machine instead of one huge multi-node job.

```bash
# machine A
bash scripts/run_train.sh infra/envs/ablation_8xh100.env \
  --swanlab_name ctm-cross1-self1-iter5 \
  --cross_layer_state 1 --self_cond 1 --iterations 5

# machine B
bash scripts/run_train.sh infra/envs/ablation_8xh100.env \
  --swanlab_name ctm-cross0-self1-iter5 \
  --cross_layer_state 0 --self_cond 1 --iterations 5

# machine C
bash scripts/run_train.sh infra/envs/ablation_8xh100.env \
  --swanlab_name ctm-cross1-self0-iter5 \
  --cross_layer_state 1 --self_cond 0 --iterations 5
```

If multiple runs share one machine, give each run a different `MASTER_PORT`.

## Common Issues

### `Address family not supported by protocol`

Use IPv4 for the rendezvous address:

```bash
MASTER_ADDR=127.0.0.1 bash scripts/run_train.sh infra/envs/smoke_8xh100.env
```

For multi-node jobs, use the master node's IPv4 address, not `localhost`.

### NCCL chooses the wrong network interface

Set `NCCL_SOCKET_IFNAME` in the env file or at launch time:

```bash
NCCL_SOCKET_IFNAME=eth0 MASTER_ADDR=<node0-ip> NODE_RANK=1 NNODES=2 \
  bash scripts/run_train.sh infra/envs/smoke_multinode.env
```

### Port already in use

Change `MASTER_PORT`:

```bash
MASTER_PORT=29511 bash scripts/run_train.sh infra/envs/smoke_8xh100.env
```

### SwanLab missing

SwanLab is optional. Do not pass `--use_swanlab`, or install it:

```bash
pip install swanlab
```

## Recommended Validation Order

1. `smoke_8xh100.env` on one node.
2. `smoke_multinode.env` across two nodes.
3. Parallel ablations for `cross_layer_state`, `self_cond`, and `iterations`.
4. Larger production configs after the first few thousand steps are stable.
