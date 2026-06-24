#!/usr/bin/env python
"""Winning Combinations — standalone runner (replaces 04_winning_combos.ipynb).

Run on the compute machine:
    nohup python paper/run_04_combos.py --gpus 8 > logs/04_combos.log 2>&1 &

Or via cluster_pool.  Analysis (status/collect/plot) stays in the notebook.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from exp_runner import Experiment, BASE_CONFIGS, run_all, status

LOG_ROOT = "logs/deep/04_combos"


# ── Combo 1: revise + JEPA ──────────────────────────────────────────
def make_revise_jepa():
    exps = []
    for task in ["cifar10", "mazes"]:
        module, base = BASE_CONFIGS[task]
        for s in range(5):
            exps.append(Experiment(
                f"{task}_revise_jepa_s{s}", task, module,
                {**base, "seed": s,
                 "draft_mode": "revise", "draft_revise_weight": 0.1,
                 "draft_corrupt_prob": 0.15, "draft_block_size": 2,
                 "cross_tick_jepa_weight": 0.1,
                 "cross_tick_jepa_hidden_dim": 128,
                 "cross_tick_jepa_predictor_depth": 2,
                 "cross_tick_jepa_dropout": 0.0}))
    return exps


# ── Combo 2: revise + sparsity ──────────────────────────────────────
def make_revise_spar():
    exps = []
    for task in ["sort", "mazes"]:
        module, base = BASE_CONFIGS[task]
        for s in range(5):
            exps.append(Experiment(
                f"{task}_revise_spar_s{s}", task, module,
                {**base, "seed": s,
                 "draft_mode": "revise", "draft_revise_weight": 0.1,
                 "draft_corrupt_prob": 0.15, "draft_block_size": 2,
                 "topk_neurons": 0.5}))
    return exps


# ── Combo 3: JEPA + sparsity ────────────────────────────────────────
def make_jepa_spar():
    exps = []
    for task in ["cifar10", "sort"]:
        module, base = BASE_CONFIGS[task]
        for s in range(5):
            exps.append(Experiment(
                f"{task}_jepa_spar_s{s}", task, module,
                {**base, "seed": s,
                 "cross_tick_jepa_weight": 0.1,
                 "cross_tick_jepa_hidden_dim": 128,
                 "cross_tick_jepa_predictor_depth": 2,
                 "cross_tick_jepa_dropout": 0.0,
                 "topk_neurons": 0.5}))
    return exps


# ── Combo 4: full stack (revise + JEPA + sparsity) ──────────────────
def make_full_stack():
    exps = []
    for task in ["cifar10", "sort"]:
        module, base = BASE_CONFIGS[task]
        for s in range(5):
            exps.append(Experiment(
                f"{task}_full_stack_s{s}", task, module,
                {**base, "seed": s,
                 "draft_mode": "revise", "draft_revise_weight": 0.1,
                 "draft_corrupt_prob": 0.15, "draft_block_size": 2,
                 "cross_tick_jepa_weight": 0.1,
                 "cross_tick_jepa_hidden_dim": 128,
                 "cross_tick_jepa_predictor_depth": 2,
                 "cross_tick_jepa_dropout": 0.0,
                 "topk_neurons": 0.5}))
    return exps


def main():
    ap = argparse.ArgumentParser(description="04 Winning Combinations runner")
    ap.add_argument("--gpus", type=int, default=8)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--only", choices=["revise_jepa", "revise_spar", "jepa_spar", "full_stack"],
                    nargs="*", default=None,
                    help="run only specific combo groups")
    args = ap.parse_args()

    builders = {
        "revise_jepa": make_revise_jepa,
        "revise_spar": make_revise_spar,
        "jepa_spar": make_jepa_spar,
        "full_stack": make_full_stack,
    }
    groups = args.only if args.only else list(builders.keys())

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
