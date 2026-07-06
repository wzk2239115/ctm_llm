#!/usr/bin/env python
"""Paper Reproduction — re-run the 3 finalized CTM-LLM ideas for the paper.

Ideas (work_ideas.md, all finalized):
  1. Cross-Tick JEPA   — coherent-thinking aux loss (final-tick gains on
                         cifar10 +7.5pp, qamnist +17pp; mazes neutral)
  2. Draft-Revise      — draft + noise + revise (parity mc +10pp headline;
                         cifar10 final +9.9pp; mazes neutral)
  3. Sparsity (top-k)  — efficiency Pareto (mazes r=0.1 saves 90% NLM
                         compute for -0.9pp; sort r=0.5 pitfall)

Scope: headline configs only, 5 seeds each, plus 5-seed baselines on every
task so deltas are fair mc-vs-mc / final-vs-final (no stale constants).

Layout (two-level, so extract_ctm_paper_results.py reads it directly):
    paper_repro/logs/<group>/<exp_name>/   (group in {baseline,jepa,revise,sparsity})

== Run on the compute machine (set proxy first) ==
    export http_proxy="http://public-proxy.qihoo.net:3128"
    export https_proxy="http://public-proxy.qihoo.net:3128"
    nohup python paper_repro/run_repro.py --gpus 8 > paper_repro/logs/run.log 2>&1 &

== Smoke (1 seed, one group, fast sanity) ==
    python paper_repro/run_repro.py --gpus 8 --seeds 1 --only jepa --dry-run

== Status (progress snapshot) ==
    python paper_repro/run_repro.py status

== Collect results (computes BOTH most-certain & final-tick acc) ==
    python scripts/extract_ctm_paper_results.py --logs paper_repro/logs \\
        --csv paper_repro/csv_data/repro_summary.csv \\
        --md  paper_repro/csv_data/repro_summary.md --curves
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "paper"))

from exp_runner import (  # noqa: E402
    run_all, status, make_baselines, make_jepa, make_revise, make_sparsity,
)

LOG_ROOT = Path("paper_repro/logs")

# st00 reproduced baselines (mc), for reference / delta bookkeeping.
# Baselines are ALSO re-run here (5 seeds) so deltas are paired, not vs constants.
MC_BASELINE = {"cifar10": 0.8516, "mazes": 0.9117, "parity": 0.8821,
               "qamnist": 0.9953, "sort": 0.8753}
FINAL_BASELINE = {"cifar10": 0.6690, "mazes": 0.9016, "parity": 0.6797,
                  "qamnist": 0.3662, "sort": 0.8753}

ALL_TASKS = ["cifar10", "mazes", "parity", "qamnist", "sort"]


def make_baseline_group(seeds):
    """Plain CTM paper config, no idea flag. Doubles as sparsity r=1.0 reference."""
    return make_baselines(ALL_TASKS, seeds)


def make_jepa_group(seeds):
    """Cross-tick JEPA headline weights (work_ideas section 一)."""
    return (
        make_jepa(["cifar10"], seeds, weights=(0.1,))    # final +7.5pp
        + make_jepa(["mazes"], seeds, weights=(0.1,))    # neutral
        + make_jepa(["qamnist"], seeds, weights=(0.5,))  # final +17pp
    )


def make_revise_group(seeds):
    """Draft-revise headline configs (work_ideas section 二)."""
    return (
        make_revise(["parity"], seeds, w=0.1, cp=0.15)   # mc +10pp (the win)
        + make_revise(["cifar10"], seeds, w=0.2, cp=0.3)  # final +9.9pp
        + make_revise(["mazes"], seeds, w=0.1, cp=0.15)   # neutral
    )


def make_sparsity_group(seeds):
    """Top-k sparsity Pareto sweep (work_ideas section 三).

    mazes r in {0.1,0.25,0.5,0.75}; r=1.0 (dense) is the baseline group.
    sort r=0.5 demonstrates the task-specific pitfall (-12pp).
    """
    return (
        make_sparsity(["mazes"], seeds, ratios=(0.1, 0.25, 0.5, 0.75))
        + make_sparsity(["sort"], seeds, ratios=(0.5,))
    )


GROUPS = {
    "baseline": make_baseline_group,
    "jepa": make_jepa_group,
    "revise": make_revise_group,
    "sparsity": make_sparsity_group,
}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gpus", type=int, default=8)
    ap.add_argument("--seeds", type=int, default=5,
                    help="number of seeds per config (default 5)")
    ap.add_argument("--only", nargs="*",
                    choices=list(GROUPS.keys()), default=None,
                    help="run only these groups (default: all)")
    ap.add_argument("--dry-run", action="store_true",
                    help="print plan only, do not launch")
    ap.add_argument("--mem-util", type=float, default=0.80)
    sub = ap.add_subparsers(dest="cmd")
    sub.add_parser("status", help="progress snapshot of paper_repro/logs")
    args = ap.parse_args()

    if args.cmd == "status":
        for g in GROUPS:
            d = LOG_ROOT / g
            if d.exists():
                print(f"\n=== {g} ===")
                status(str(d))
        return

    seeds = range(args.seeds)
    names = args.only if args.only else list(GROUPS.keys())
    total = 0
    print("=" * 64)
    print(f"Paper Reproduction — {args.seeds} seed(s), groups={names}")
    print("=" * 64)
    for g in names:
        exps = GROUPS[g](seeds)
        total += len(exps)
        print(f"  {g:10s}: {len(exps)} runs")
        for e in exps:
            print(f"    {e.name}")
    print(f"  {'TOTAL':10s}: {total} runs")
    print("=" * 64)

    if args.dry_run:
        print("(dry-run; not launching)")
        return

    for g in names:
        exps = GROUPS[g](seeds)
        gdir = LOG_ROOT / g
        print(f"\n>>> launching {g} ({len(exps)} runs) -> {gdir}")
        run_all(exps, gpus=args.gpus, log_root=str(gdir),
                dry_run=False, mem_util=args.mem_util)

    print("\nAll groups finished. Collect with:")
    print("  python scripts/extract_ctm_paper_results.py --logs paper_repro/logs "
          "--csv paper_repro/csv_data/repro_summary.csv "
          "--md paper_repro/csv_data/repro_summary.md --curves")


if __name__ == "__main__":
    main()
