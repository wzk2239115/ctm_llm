#!/usr/bin/env python3
"""Pool entry point for baseline CTM tasks (replaces run_via_pool.sh)."""

import json
import os
import runpy
import subprocess
import sys
import time


def main():
    if len(sys.argv) < 2:
        print("[run_via_pool] Error: no arguments provided")
        sys.exit(1)

    module = sys.argv[1]
    args = sys.argv[2:]

    print(f"[run_via_pool] Module: {module}")
    print(f"[run_via_pool] Args: {' '.join(args)}")
    print(f"[run_via_pool] CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'all')}")
    print(f"[run_via_pool] Start: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    logdir = "runs/logs"
    os.makedirs(logdir, exist_ok=True)
    logpath = os.path.join(logdir, "pool_last_run.log")

    proc = subprocess.run(
        [sys.executable, "-m", module] + args,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    output = proc.stdout
    with open(logpath, "w") as f:
        f.write(output)
    print(output, end="", flush=True)

    if proc.returncode != 0:
        print(f"[run_via_pool] exit_code={proc.returncode}")
        exp_name = os.environ.get("CTM_EXPERIMENT_NAME")
        if exp_name:
            metrics_dir = os.environ.get("CTM_METRICS_DIR", "runs/metrics")
            fail_path = os.path.join(metrics_dir, f"{exp_name}.fail.json")
            os.makedirs(metrics_dir, exist_ok=True)
            payload = {
                "experiment_name": exp_name,
                "status": "oom" if proc.returncode == 137 else "failed",
                "rank": 0,
                "error_type": "SubprocessError",
                "error": output[-4000:],
                "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "git_commit": "unknown",
            }
            with open(fail_path, "w") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)
            print(f"[run_via_pool] wrote failure report: {fail_path}", file=sys.stderr)

    sys.exit(proc.returncode)


if __name__ == "__main__":
    main()
