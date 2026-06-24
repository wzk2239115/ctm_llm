#!/usr/bin/env python
"""JEPA Deep Study — standalone runner (replaces 02_jepa_deep.ipynb).

Groups:
  main      — w=0.1, 5 seeds × 2 tasks
  sweep     — w ∈ {0.02,0.05,0.1,0.2,0.3,0.5}, 3 seeds × 2 tasks
  ablation  — 7 variants @ w=0.1, 3 seeds × 2 tasks

Run on the compute machine:
    nohup python paper/run_02_jepa.py --gpus 8 > logs/02_jepa.log 2>&1 &
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from exp_runner import Experiment, BASE_CONFIGS, run_all, status

LOG_ROOT = "logs/deep/02_jepa"

JEPA_DEFAULTS = dict(
    cross_tick_jepa_weight=0.1,
    cross_tick_jepa_hidden_dim=128,
    cross_tick_jepa_predictor_depth=2,
    cross_tick_jepa_dropout=0.0,
)

SWEEP_WEIGHTS = [0.02, 0.05, 0.1, 0.2, 0.3, 0.5]

ABLATION_CONFIGS = [
    ("full",        {}),
    ("loss_mse",    {"cross_tick_jepa_loss": "mse"}),
    ("no_stopgrad", {"cross_tick_jepa_target_stop_grad": False}),
    ("depth1",      {"cross_tick_jepa_predictor_depth": 1}),
    ("depth4",      {"cross_tick_jepa_predictor_depth": 4}),
    ("hid64",       {"cross_tick_jepa_hidden_dim": 64}),
    ("hid256",      {"cross_tick_jepa_hidden_dim": 256}),
]

TASKS = ["cifar10", "mazes"]


def _wstr(w):
    return str(w).replace(".", "p")


def make_main(seeds=range(5)):
    exps = []
    for task in TASKS:
        module, base = BASE_CONFIGS[task]
        for s in seeds:
            cfg = {**base, **JEPA_DEFAULTS, "seed": s}
            exps.append(Experiment(
                f"{task}_jepa_w0p1_s{s}", task, module, cfg))
    return exps


def make_sweep(seeds=range(3)):
    exps = []
    for task in TASKS:
        module, base = BASE_CONFIGS[task]
        for w in SWEEP_WEIGHTS:
            for s in seeds:
                cfg = {**base, **JEPA_DEFAULTS,
                       "cross_tick_jepa_weight": w, "seed": s}
                exps.append(Experiment(
                    f"{task}_swp_w{_wstr(w)}_s{s}", task, module, cfg))
    return exps


def make_ablation(seeds=range(3)):
    exps = []
    for task in TASKS:
        module, base = BASE_CONFIGS[task]
        for variant, overrides in ABLATION_CONFIGS:
            for s in seeds:
                cfg = {**base, **JEPA_DEFAULTS, **overrides, "seed": s}
                exps.append(Experiment(
                    f"{task}_abl_{variant}_s{s}", task, module, cfg))
    return exps


def main():
    ap = argparse.ArgumentParser(description="02 JEPA Deep Study runner")
    ap.add_argument("--gpus", type=int, default=8)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--only", choices=["main", "sweep", "ablation"],
                    nargs="*", default=None)
    args = ap.parse_args()

    groups = args.only if args.only else ["main", "sweep", "ablation"]
    builders = {"main": make_main, "sweep": make_sweep, "ablation": make_ablation}

    exps = []
    for g in groups:
        built = builders[g]()
        print(f"  {g}: {len(built)} runs")
        exps += built

    print(f"\nTotal: {len(exps)} experiments\n")
    run_all(exps, gpus=args.gpus, log_root=LOG_ROOT, dry_run=args.dry_run)

    if not args.dry_run:
        print("\n" + "=" * 60)
        status(LOG_ROOT)


if __name__ == "__main__":
    main()
