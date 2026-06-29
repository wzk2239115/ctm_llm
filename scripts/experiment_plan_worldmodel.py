#!/usr/bin/env python
"""Pool submission plan for the worldmodel CTM-vs-JEPA comparison.

Each pool task runs ``paper.run_worldmodel`` once with a distinct ``--seed``
(and ``--csv_suffix``), so seeds run in parallel across the pool and write
separate CSVs that are merged afterwards for error bars.

Usage:
    # print the per-seed commands (no submission)
    python scripts/experiment_plan_worldmodel.py plan
    # submit all seeds to the pool (non-blocking)
    python scripts/experiment_plan_worldmodel.py submit --master-addr 11.131.210.78 --port 8765
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request

SEEDS = [0, 1, 2, 3]


def build_command(seed: int, common: str) -> tuple[str, str]:
    name = f"wm_s{seed}"
    extra = f"{common} --seed {seed} --csv_suffix _s{seed}"
    return name, extra


def submit_to_pool(name, extra_args, master_addr, port):
    payload = {
        "config": "paper.run_worldmodel",
        "extra_args": extra_args,
        "env": {"CTM_EXPERIMENT_NAME": name, "CTM_LOG_DIR": "logs/worldmodel"},
    }
    base = f"http://{master_addr}:{port}"
    req = urllib.request.Request(
        f"{base}/submit", data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        resp = opener.open(req, timeout=10)
        return json.loads(resp.read()).get("task")
    except Exception as e:
        print(f"[submit] {name}: error {e}")
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["plan", "submit"])
    ap.add_argument("--master-addr", default="11.131.210.78")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--seeds", type=int, nargs="*", default=SEEDS)
    ap.add_argument(
        "--common", default="--env point-image --episodes 80 --epochs 80 "
        "--var_weight 1.0 --latent_dim 32 --ctm_iterations 8 "
        "--cem_samples 128 --cem_steps 6 --eval_episodes 16",
        help="shared CLI flags forwarded to run_worldmodel.py",
    )
    args = ap.parse_args()

    for s in args.seeds:
        name, extra = build_command(s, args.common)
        if args.mode == "plan":
            print(f"# {name}\npython -m paper.run_worldmodel {extra}\n")
        else:
            t = submit_to_pool(name, extra, args.master_addr, args.port)
            print(f"[submit] {name}: {'ok' if t else 'FAILED'}")
    print(f"\n{len(args.seeds)} seed{'s' if len(args.seeds)!=1 else ''} "
          f"{'listed' if args.mode=='plan' else 'submitted'}.")


if __name__ == "__main__":
    main()
