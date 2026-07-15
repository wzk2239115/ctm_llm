#!/usr/bin/env python
"""Overnight gap-fillers for the three-pillar paper (JEPA / revise / sparsity).

These reuse the exact same configs / runner as `run_repro.py`, so results drop
straight into `paper_repro/logs/<group>/` and are picked up by the same collect
command. Re-run `extract_ctm_paper_results.py --logs paper_repro/logs ...` after.

Streams (pick with --stream):
  sparsity-gap  (DEFAULT) — top-k r-sweep on cifar10 + parity. Fills the
                 explicit VERIFIED_CONCLUSIONS gap ("cifar10/parity 缺完整 r
                 sweep"). r in {0.1,0.25,0.5,0.75}; r=1.0 (dense) is already the
                 baseline group. Writes to logs/sparsity/.
  jepa-robust   — cifar10 JEPA weight sweep w in {0.03,0.05,0.2,0.3} (0.1 already
                 in repro) to locate the sweet spot. Writes to logs/jepa/.
  revise-robust — parity draft-revise corrupt_prob sweep cp in {0.05,0.3,0.5}
                 (0.15 already in repro) to stress the mc +10pp headline.
                 Writes to logs/revise/.

Split across two nodes without collision via --tasks (exp names are task-prefixed):
  node A: python paper_repro/run_overnight.py --tasks cifar10 --gpus 8
  node B: python paper_repro/run_overnight.py --tasks parity   --gpus 8

Smoke first:
  python paper_repro/run_overnight.py --stream sparsity-gap --tasks parity --seeds 1 --dry-run
  python paper_repro/run_overnight.py --stream sparsity-gap --tasks parity --seeds 1 --gpus 1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "paper"))

from exp_runner import (  # noqa: E402
    run_all, make_sparsity, make_jepa, make_revise,
)

LOG_ROOT = Path("paper_repro/logs")


def build(stream, tasks, seeds):
    """Return (experiments, group_dir_name) for the chosen stream."""
    if stream == "sparsity-gap":
        exps = make_sparsity(tasks, seeds, ratios=(0.1, 0.25, 0.5, 0.75))
        return exps, "sparsity"
    if stream == "jepa-robust":
        exps = make_jepa(tasks, seeds, weights=(0.03, 0.05, 0.2, 0.3))
        return exps, "jepa"
    if stream == "revise-robust":
        exps = []
        for cp in (0.05, 0.3, 0.5):  # cp=0.15 (headline) already in repro
            exps += make_revise(tasks, seeds, w=0.1, cp=cp)
        return exps, "revise"
    raise KeyError(stream)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stream", default="sparsity-gap",
                    choices=["sparsity-gap", "jepa-robust", "revise-robust"])
    ap.add_argument("--tasks", nargs="*", default=["cifar10", "parity"],
                    help="subset of tasks (default: cifar10 parity; jepa/revise streams ignore parity graveyard)")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--gpus", type=int, default=8)
    ap.add_argument("--mem-util", type=float, default=0.80)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    seeds = range(args.seeds)
    exps, group = build(args.stream, args.tasks, seeds)

    print("=" * 64)
    print(f"Overnight — stream={args.stream}  tasks={args.tasks}  "
          f"seeds={args.seeds}  gpus={args.gpus}")
    print("=" * 64)
    for e in exps:
        print(f"  {group}/{e.name}")
    print(f"  TOTAL: {len(exps)} runs  ->  {LOG_ROOT}/{group}")
    print("=" * 64)
    if args.dry_run:
        print("(dry-run; not launching)")
        return

    gdir = LOG_ROOT / group
    print(f"\n>>> launching {len(exps)} runs -> {gdir}")
    run_all(exps, gpus=args.gpus, log_root=str(gdir),
            dry_run=False, mem_util=args.mem_util)


if __name__ == "__main__":
    main()
