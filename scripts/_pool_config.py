"""Central pool configuration — single source of truth for all experiment plans.

All values are read from environment variables with sensible defaults.
Set them once per machine (e.g. in ~/.bashrc or the pool startup script):

    export CTM_POOL_MASTER_ADDR=11.131.211.44
    export CTM_POOL_PORT=8765

If CTM_POOL_MASTER_ADDR is unset, the experiment plan's --master-addr CLI
flag must be used instead.  node_addrs is always empty (any worker claims).
"""

import os

def _master_addr():
    return os.environ.get("CTM_POOL_MASTER_ADDR", "11.131.210.78")

def _port():
    return int(os.environ.get("CTM_POOL_PORT", "8765"))

MASTER_ADDR = _master_addr()
PORT = _port()
# Empty = any registered worker can claim the task (dynamic scheduling).
BASELINE_NODES = ()
POOL_CONFIG = os.environ.get("CTM_POOL_CONFIG", "infra/envs/h100_baseline.env")
