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

Generate a sequential experiment command list:

```bash
python scripts/experiment_plan.py commands --stage compass \
  --config infra/clusters/h100_2nodes.env \
  --output runs/experiment_plans/compass_plan.csv
```

Generate short batch-size probes before the full run:

```bash
python scripts/experiment_plan.py batch-commands --stage all \
  --config infra/clusters/h100_4nodes.env \
  --batch_sizes 2 4 6 8 10 12 \
  --output runs/experiment_plans/batch_tune_plan.csv
```

After the probes finish, recommend one batch size per experiment:

```bash
python scripts/experiment_plan.py recommend-batches \
  --metrics_dir runs/metrics \
  --target_memory_gb 80 \
  --memory_util 0.90 \
  --output runs/metrics/batch_profile.csv
```

Generate the full serial plan using the recommended batch profile:

```bash
python scripts/experiment_plan.py commands --stage all \
  --config infra/clusters/h100_4nodes.env \
  --batch_profile runs/metrics/batch_profile.csv \
  --output runs/experiment_plans/full_plan.csv
```

Summarize completed training metrics:

```bash
python scripts/experiment_plan.py summarize \
  --metrics_dir runs/metrics \
  --output runs/metrics/summary.csv
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

By default, a worker updates the repo before starting each task with an explicit
`git fetch <remote> <branch>` followed by `git merge --ff-only <remote>/<branch>`.
If the repository changed, the worker restarts itself so the latest
pool/training code is loaded. This means you usually do not need to rebuild the
pool after a `git push`; leave workers online and submit the next task.

If multiple machines mount the same repository directory, set:

```bash
SHARED_REPO=1
```

With `SHARED_REPO=1`, workers serialize the Git update through
`.ctm_pool/git_update.lock`. This avoids two machines touching the same `.git`
directory at the same time. Every worker still checks whether the shared repo
HEAD changed since that worker started; if it changed, the worker restarts
itself before launching the training task.

The default Git target is the current branch from `origin`. You can pin it in
the cluster config:

```bash
GIT_REMOTE=origin
GIT_BRANCH=master
```

If the cluster needs a proxy for Git, set it in the cluster config:

```bash
GIT_HTTP_PROXY=http://public-proxy.qihoo.net:3128
GIT_HTTPS_PROXY=http://public-proxy.qihoo.net:3128
```

This proxy is used only for Git fetch/merge updates. Pool HTTP traffic to the
master remains direct and does not use proxy settings.

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

Four-node template:

```bash
cp infra/clusters/h100_4nodes.env.example infra/clusters/h100_4nodes.env
```

Edit `infra/clusters/h100_4nodes.env`, replace the two `TODO_NODE_*_IP`
entries, then start workers with that config on all four nodes.

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

### Git Update Fails On A Shared Repo

If workers report a Git update failure while all machines mount the same repo,
make sure the cluster config has:

```bash
SHARED_REPO=1
GIT_REMOTE=origin
GIT_BRANCH=master
```

Then restart the worker terminals once so they load the shared-repo locking
logic. After that, workers can stay online across normal `git push` / task
submit cycles.

### Torchrun Rendezvous Fails

Make sure `MASTER_PORT` in the cluster config is free, or override it:

```bash
MASTER_PORT=29511 ./scripts/ctmctl pool submit infra/clusters/h100_2nodes.env
```

Use IPv4 addresses, not `localhost`, for multi-node jobs.
