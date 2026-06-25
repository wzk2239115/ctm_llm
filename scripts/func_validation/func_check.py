#!/usr/bin/env python3
"""Functional validation: prove an idea actually changes model behavior.

This is Step 2 of the mandatory two-step verification (see AGENTS.md):
  Step 1 (smoke)   = it runs without crashing   -> scripts/smoke_baseline.py
  Step 2 (functional) = it actually has an effect -> this script

For a given task + idea, it runs TWO short configs with the SAME seed but
DIFFERENT values of the idea's key hyperparameter, then compares the test-acc
trajectories:
  - identical curves -> the idea is inert (args not wired onto the model) -> FAIL
  - differing curves -> the idea takes effect                       -> PASS

The base configs are deterministic (dropout=0), so an inert idea yields
bit-identical curves and a reliable FAIL (unlike long noisy training).

Usage:
    python scripts/func_validation/func_check.py --task sort   --idea revise
    python scripts/func_validation/func_check.py --task mazes  --idea revise
    python scripts/func_validation/func_check.py --task sort   --idea jepa
    python scripts/func_validation/func_check.py --task sort   --idea sparsity
    python scripts/func_validation/func_check.py --all        # every task x idea

Options:
    --device N     CUDA device (default 0)
    --iters N      training iterations per run (default 600)
    --ticks N      CTM thought-ticks / iterations (default: per task)
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))

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

DEFAULT_TICKS = {"sort": 50, "mazes": 50, "cifar10": 30, "parity": 30, "qamnist": 30}


def build_cfg(task, idea, vary_val, device, iters, ticks):
    module, base = TASKS[task]
    cfg = dict(base)
    cfg["iterations"] = ticks
    cfg["training_iterations"] = iters
    cfg["track_every"] = max(50, iters // 6)
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


def run_one(task, idea, vary_val, device, iters, ticks, log_dir):
    import torch
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


def check(task, idea, device, iters, ticks):
    vary_param, (lo, hi) = IDEAS[idea]["vary"]
    a, err = run_one(task, idea, lo, device, iters, ticks,
                     f"/tmp/func_check/{task}_{idea}_lo")
    if err:
        return False, f"{task}/{idea} {vary_param}={lo}: {err}"
    b, err = run_one(task, idea, hi, device, iters, ticks,
                     f"/tmp/func_check/{task}_{idea}_hi")
    if err:
        return False, f"{task}/{idea} {vary_param}={hi}: {err}"
    n = min(len(a), len(b))
    same = a[:n] == b[:n]
    detail = (f"{task}/{idea}  {vary_param}={lo} vs {hi}\n"
              f"    {lo}: {a}\n    {hi}: {b}")
    verdict = "FAIL (curves identical -> idea inert)" if same else "PASS (curves differ -> idea active)"
    return (not same), f"{verdict}\n{detail}"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", choices=list(TASKS.keys()))
    ap.add_argument("--idea", choices=list(IDEAS.keys()))
    ap.add_argument("--all", action="store_true", help="check every task x idea")
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--iters", type=int, default=600)
    ap.add_argument("--ticks", type=int, default=None)
    args = ap.parse_args()

    if args.all:
        pairs = [(t, i) for t in TASKS for i in IDEAS]
    else:
        if not args.task or not args.idea:
            ap.error("provide --task and --idea, or use --all")
        pairs = [(args.task, args.idea)]

    results = []
    for task, idea in pairs:
        ticks = args.ticks or DEFAULT_TICKS.get(task, 30)
        ok, msg = check(task, idea, args.device, args.iters, ticks)
        results.append((task, idea, ok))
        print(f"\n[{task}/{idea}] {'PASS' if ok else 'FAIL'}")
        print("  " + msg.replace("\n", "\n  "))

    print("\n" + "=" * 60)
    npass = sum(1 for _, _, ok in results if ok)
    print(f"FUNC VALIDATION: {npass}/{len(results)} passed")
    for task, idea, ok in results:
        print(f"  {'PASS' if ok else 'FAIL'}  {task}/{idea}")
    print("=" * 60)
    return 0 if npass == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
