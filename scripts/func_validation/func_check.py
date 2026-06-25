#!/usr/bin/env python3
"""Functional validation: prove an idea actually changes model behavior.

This is Step 2 of the mandatory two-step verification (see AGENTS.md):
  Step 1 (smoke)     = it runs without crashing   -> scripts/smoke_baseline.py
  Step 2 (functional) = it actually has an effect  -> this script

For a given task + idea, it runs TWO short configs with the SAME seed but
DIFFERENT values of the idea's key hyperparameter, then compares the test-acc
trajectories:
  - identical curves -> the idea is inert (args not wired onto the model) -> FAIL
  - differing curves -> the idea takes effect                       -> PASS

Runs are launched CONCURRENTLY and packed several-per-GPU (the smoke configs are
tiny), so a full --all sweep finishes in roughly one single-run time.

Usage:
    python scripts/func_validation/func_check.py --task sort   --idea revise
    python scripts/func_validation/func_check.py --task mazes  --idea revise
    python scripts/func_validation/func_check.py --all                 # every task x idea
    python scripts/func_validation/func_check.py --all --devices 0,1,2,3 --pack 6

Options:
    --device N       single CUDA device (default 0; ignored if --devices/--all)
    --devices a,b,c  explicit device list
    --iters N        training iterations per run (default 300)
    --ticks N        CTM thought-ticks / iterations (default: per task)
    --pack K         concurrent jobs per GPU (default 4; models are tiny)
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from smoke_baseline import TASKS  # noqa: E402

ACC_KEYS = ["test_accuracies_full_list", "test_accuracies",
            "val_accuracies", "test_accuracies_most_certain"]

IDEAS = {
    "revise": {
        "enable": dict(draft_mode="revise", draft_block_size=2,
                       draft_revise_weight=0.1),
        "vary": ("draft_corrupt_prob", [0.05, 0.30]),
    },
    "jepa": {
        "enable": dict(cross_tick_jepa_hidden_dim=128,
                       cross_tick_jepa_predictor_depth=2,
                       cross_tick_jepa_dropout=0.0),
        "vary": ("cross_tick_jepa_weight", [0.02, 0.30]),
    },
    "sparsity": {
        "enable": dict(),
        "vary": ("topk_neurons", [0.25, 0.75]),
    },
}

DEFAULT_TICKS = {"sort": 20, "mazes": 20, "cifar10": 15, "parity": 15, "qamnist": 15}


def build_cfg(task, idea, vary_val, device, iters, ticks):
    module, base = TASKS[task]
    cfg = dict(base)
    cfg["iterations"] = ticks
    cfg["training_iterations"] = iters
    cfg["track_every"] = max(25, iters // 6)
    cfg["save_every"] = 10 ** 9
    cfg["reload"] = False
    cfg["device"] = [device]
    cfg.update(IDEAS[idea]["enable"])
    cfg[IDEAS[idea]["vary"][0]] = vary_val
    return module, cfg


def cmd_from_cfg(module, cfg):
    cmd = [sys.executable, "-m", module]
    for k, v in cfg.items():
        if v is None:
            continue
        if isinstance(v, bool):
            cmd.append(f"--{k}" if v else f"--no-{k}")
        elif isinstance(v, (list, tuple)):
            cmd.append(f"--{k}")
            cmd.extend(str(x) for x in v)
        else:
            cmd.append(f"--{k}")
            cmd.append(str(v))
    return cmd


def latest_ckpt(log_dir):
    p = log_dir / "checkpoint.pt"
    if p.exists():
        return p
    cks = sorted(log_dir.glob("checkpoint*.pt"))
    return cks[-1] if cks else None


def run_job(job, device, iters):
    task, idea, vary_val, log_dir, ticks = job
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    module, cfg = build_cfg(task, idea, vary_val, device, iters, ticks)
    cfg["log_dir"] = str(log_dir)
    r = subprocess.run(cmd_from_cfg(module, cfg), capture_output=True, text=True)
    if r.returncode != 0:
        return None, "TRAIN FAILED:\n" + (r.stdout + r.stderr)[-1600:]
    ck = latest_ckpt(log_dir)
    if ck is None:
        return None, "no checkpoint written"
    d = torch.load(ck, map_location="cpu", weights_only=False)
    for k in ACC_KEYS:
        v = d.get(k) if isinstance(d, dict) else None
        if isinstance(v, (list, tuple)) and v:
            return [round(x, 5) for x in v], None
    return None, "no test_accuracies found in checkpoint"


def make_jobs(pairs, iters, ticks_override):
    jobs = {}
    for task, idea in pairs:
        ticks = ticks_override or DEFAULT_TICKS.get(task, 15)
        for val in IDEAS[idea]["vary"][1]:
            log = f"/tmp/func_check/{task}_{idea}_{val}"
            jobs[(task, idea, val)] = (task, idea, val, log, ticks)
    return jobs


def run_concurrent(jobs, devices, iters, pack):
    n_gpu = len(devices)
    max_workers = max(1, n_gpu * pack)
    results = {}
    keys = list(jobs.keys())
    print(f"Launching {len(keys)} runs on {n_gpu} GPU(s) x {pack} pack "
          f"= {max_workers} concurrent workers\n")

    def worker(idx, key):
        dev = devices[idx % n_gpu]
        return key, run_job(jobs[key], dev, iters)

    done = 0
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = [ex.submit(worker, i, k) for i, k in enumerate(keys)]
        for fut in as_completed(futs):
            key, (curve, err) = fut.result()
            results[key] = (curve, err)
            done += 1
            tag = "ok " if err is None else "ERR"
            tail = "" if err is None else " -> " + err.splitlines()[-1][:70]
            print(f"  [{done}/{len(keys)}] {tag} {key[0]}/{key[1]} val={key[2]}{tail}")
    return results


def evaluate(pairs, results):
    out = []
    for task, idea in pairs:
        vary_param, (lo, hi) = IDEAS[idea]["vary"]
        a, ea = results.get((task, idea, lo), (None, "missing"))
        b, eb = results.get((task, idea, hi), (None, "missing"))
        err = ea or eb
        if err:
            out.append((task, idea, False,
                        f"{task}/{idea} {vary_param}: {err}"))
            continue
        n = min(len(a), len(b))
        same = a[:n] == b[:n]
        detail = (f"{task}/{idea}  {vary_param}={lo} vs {hi}\n"
                  f"    {lo}: {a}\n    {hi}: {b}")
        verdict = ("FAIL (curves identical -> idea inert)" if same
                   else "PASS (curves differ -> idea active)")
        out.append((task, idea, not same, f"{verdict}\n{detail}"))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", choices=list(TASKS.keys()))
    ap.add_argument("--idea", choices=list(IDEAS.keys()))
    ap.add_argument("--all", action="store_true", help="check every task x idea")
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--devices", type=str, default=None,
                    help="comma-separated device list, e.g. 0,1,2,3")
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--ticks", type=int, default=None)
    ap.add_argument("--pack", type=int, default=4, help="concurrent jobs per GPU")
    args = ap.parse_args()

    if args.all:
        pairs = [(t, i) for t in TASKS for i in IDEAS]
        if args.devices is not None:
            devices = [int(x) for x in args.devices.split(",") if x.strip() != ""]
        else:
            n = torch.cuda.device_count() or 1
            devices = list(range(n))
    else:
        if not args.task or not args.idea:
            ap.error("provide --task and --idea, or use --all")
        pairs = [(args.task, args.idea)]
        devices = ([int(x) for x in args.devices.split(",") if x.strip() != ""]
                   if args.devices else [args.device])

    jobs = make_jobs(pairs, args.iters, args.ticks)
    results = run_concurrent(jobs, devices, args.iters, args.pack)
    evaluated = evaluate(pairs, results)

    print()
    for task, idea, ok, msg in evaluated:
        print(f"[{task}/{idea}] {'PASS' if ok else 'FAIL'}")
        print("  " + msg.replace("\n", "\n  "))

    print("\n" + "=" * 60)
    npass = sum(1 for _, _, ok, _ in evaluated if ok)
    print(f"FUNC VALIDATION: {npass}/{len(evaluated)} passed")
    for task, idea, ok, _ in evaluated:
        print(f"  {'PASS' if ok else 'FAIL'}  {task}/{idea}")
    print("=" * 60)
    return 0 if npass == len(evaluated) else 1


if __name__ == "__main__":
    sys.exit(main())
