# ctmctl CLI

`scripts/ctmctl` is the main command-line entrypoint for CTM-LLM training
infra. It wraps the no-SSH pool coordinator, node-list based `torchrun`, env
template launches, and cluster checks.

## Quick Start

On master `11.131.210.78`, terminal 1:

```bash
cd /home/jovyan/h800fast/wangzekai/ctm_llm
git pull
./scripts/ctmctl pool server
```

On every node, including master and `11.131.210.3`, open one worker terminal:

```bash
cd /home/jovyan/h800fast/wangzekai/ctm_llm
git pull
./scripts/ctmctl pool worker infra/clusters/h100_2nodes.env
```

On master, terminal 2, submit a task:

```bash
./scripts/ctmctl pool submit infra/clusters/h100_2nodes.env
```

Submit with trainer overrides:

```bash
./scripts/ctmctl pool submit infra/clusters/h100_2nodes.env \
  --iterations 5 --cross_layer_state 0 --swanlab_name cross0-iter5
```

Check pool state:

```bash
./scripts/ctmctl pool status --master_addr 11.131.210.78
```

## Commands

### `pool server`

Starts the no-SSH pool server. Workers self-register here, and task acks are
printed in this terminal.

```bash
./scripts/ctmctl pool server
./scripts/ctmctl pool server --host 0.0.0.0 --port 8765
```

### `pool worker`

Starts a worker on each training machine. The worker detects its IP, finds its
rank from `NODE_ADDRS`, reports GPU model/memory, then waits for tasks.

```bash
./scripts/ctmctl pool worker infra/clusters/h100_2nodes.env
./scripts/ctmctl pool worker infra/clusters/h100_2nodes.env --port 8765
```

By default, a worker runs `git pull --ff-only` before starting each task. If the
repository changed, the worker restarts itself so the latest pool/training code
is loaded. This means you usually do not need to rebuild the pool after a
`git push`; leave workers online and submit the next task.

Disable this during debugging if needed:

```bash
./scripts/ctmctl pool worker infra/clusters/h100_2nodes.env --no_auto_pull
./scripts/ctmctl pool worker infra/clusters/h100_2nodes.env --no_restart_on_update
```

If IP auto-detection fails:

```bash
CTM_NODE_ADDR=11.131.210.3 \
  ./scripts/ctmctl pool worker infra/clusters/h100_2nodes.env
```

### `pool submit`

Submits a task to all online workers. The submit command waits for worker acks
by default.

```bash
./scripts/ctmctl pool submit infra/clusters/h100_2nodes.env
./scripts/ctmctl pool submit infra/clusters/h100_2nodes.env --iterations 5
./scripts/ctmctl pool submit infra/clusters/h100_2nodes.env --wait 0
```

### `pool status`

Prints raw JSON status, including nodes, GPU inventory, current task, and acks.

```bash
./scripts/ctmctl pool status --master_addr 11.131.210.78
```

### `train cluster`

Runs node-list based `torchrun` directly on the current node. This is useful
when you want to start the same command manually on every node without the pool.

```bash
./scripts/ctmctl train cluster --config infra/clusters/h100_2nodes.env
DRY_RUN=1 ./scripts/ctmctl train cluster --config infra/clusters/h100_2nodes.env
```

### `train env`

Runs `torchrun` from an env template.

```bash
./scripts/ctmctl train env infra/envs/smoke_8xh100.env
DRY_RUN=1 ./scripts/ctmctl train env infra/envs/smoke_multinode.env
```

### `check`

Runs preflight checks for Git status, GPUs, symlinks, data, tokenizer, packages,
and network address hints.

```bash
./scripts/ctmctl check infra/envs/smoke_8xh100.env
```

## Cluster Config

Current two-node config:

```bash
infra/clusters/h100_2nodes.env
```

Important fields:

```bash
NODE_ADDRS=(
  11.131.210.78
  11.131.210.3
)

MASTER_PORT=29500
NPROC_PER_NODE=8
TRAIN_ARGS="..."
```

The first node in `NODE_ADDRS` is used as `MASTER_ADDR` for `torchrun`.

## GPU Reporting

Workers report GPU inventory at startup and in pool status:

```text
8x NVIDIA H800 80GB
  GPUs:
    [0] NVIDIA H800 79.6GB
    ...
```

The pool server/status keeps both a summary and per-GPU details.

## Troubleshooting

### Worker Shows `HTTP Error 403: Forbidden`

Likely causes:

- The pool server is not actually running on `master_addr:port`.
- Another service is listening on that port.
- An HTTP proxy intercepted the request.

The client disables Python HTTP proxy usage, but you can still test directly:

```bash
curl -v http://11.131.210.78:8765/status
```

If this returns 403, switch ports:

```bash
# master
./scripts/ctmctl pool server --port 18765

# workers
./scripts/ctmctl pool worker infra/clusters/h100_2nodes.env --port 18765

# submit/status
./scripts/ctmctl pool submit infra/clusters/h100_2nodes.env --port 18765
./scripts/ctmctl pool status --master_addr 11.131.210.78 --port 18765
```

### Worker Cannot Auto-Detect Rank

Set the address manually:

```bash
CTM_NODE_ADDR=11.131.210.3 \
  ./scripts/ctmctl pool worker infra/clusters/h100_2nodes.env
```

### Submit Times Out Waiting For Ack

Check that every node has an active worker terminal:

```bash
./scripts/ctmctl pool status --master_addr 11.131.210.78
```

### Torchrun Rendezvous Fails

Make sure `MASTER_PORT` in the cluster config is free, or override it:

```bash
MASTER_PORT=29511 ./scripts/ctmctl pool submit infra/clusters/h100_2nodes.env
```

Use IPv4 addresses, not `localhost`, for multi-node jobs.
