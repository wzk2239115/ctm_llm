#!/usr/bin/env python
"""CTM Scaling Law — cells (d_model) x iterations (ticks) sweep.

Measures steps-to-target-acc as a function of CTM's two core axes:
  - cells axis: d_model sweep at fixed iterations (capacity scaling)
  - ticks axis: iterations sweep at fixed d_model (depth-in-time scaling)

Both axes use FIXED per-task batch_size and LR (paper baseline), so the only
varying variable is the architectural scale. This isolates sample-efficiency
scaling from optimizer-tuning confounds (muP-style LR scaling is explicitly
out of scope; reported in limitations).

Tasks (visual / algorithmic / reasoning; sort excluded — too fragile, qamnist
— missing data):
  - cifar10  (resnet18 backbone, visual perception)
  - parity   (parity backbone, algorithmic)
  - mazes    (resnet34 backbone, spatial reasoning)

Output curves let us fit:
  steps_to_target = a * (d_model)^b        (cells scaling)
  steps_to_target = a * (iterations)^b     (ticks scaling)
and report whether b ~ 0 (free), b ~ 1 (linear), b > 1 (super-linear).

Layout (two-level, extract_ctm_paper_results.py reads directly):
    paper_scaling/logs/cells/{exp_name}/
    paper_scaling/logs/ticks/{exp_name}/

== Run on compute machine (set proxy first) ==
    export http_proxy="http://public-proxy.qihoo.net:3128"
    export https_proxy="http://public-proxy.qihoo.net:3128"
    nohup python paper_scaling/run_scaling.py --gpus 8 > paper_scaling/logs/run.log 2>&1 &

== Smoke (1 seed, cells only, fast sanity — MUST pass before full run) ==
    python paper_scaling/run_scaling.py --gpus 8 --seeds 1 --only cells --dry-run
    python paper_scaling/run_scaling.py --gpus 8 --seeds 1 --only cells

== Status ==
    python paper_scaling/run_scaling.py status

== Collect results (writes summary + curves) ==
    python scripts/extract_ctm_paper_results.py --logs paper_scaling/logs \\
        --csv paper_scaling/csv_data/scaling_summary.csv \\
        --md  paper_scaling/csv_data/scaling_summary.md --curves

== Analyze (after collect; fits power law, writes PNG) ==
    python paper_scaling/run_scaling.py analyze
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "paper"))

from exp_runner import Experiment, run_all, status, BASE_CONFIGS  # noqa: E402

LOG_ROOT = Path("paper_scaling/logs")
CSV_ROOT = Path("paper_scaling/csv_data")
FIG_ROOT = Path("paper_scaling/figures")

# ─────────────────────────────────────────────────────────────────────────
# SWEEP DESIGN
# ─────────────────────────────────────────────────────────────────────────
# Per-task batch_size and LR are inherited from BASE_CONFIGS (paper baseline)
# and NOT changed across the sweep — this is the controlled-variable design.
# d_model / iterations ranges confirmed runnable by st01 / st02.

# Cells axis: d_model sweep at task-default iterations.
#   cifar10 default ticks=50, parity 75, mazes 75 (from BASE_CONFIGS).
CELLS_SWEEP = {
    "cifar10": [64, 128, 256, 512],     # st01 ran these @batch=512 OK
    "parity":  [256, 512, 1024, 2048],  # st01 ran these @batch=64 OK
    "mazes":   [512, 1024, 2048, 4096], # st01 ran these @batch=64 OK
}

# Ticks axis: iterations sweep at task-default d_model.
#   cifar10 default d_model=256, parity 1024, mazes 2048.
TICKS_SWEEP = {
    "cifar10": [2, 5, 10, 25, 50],   # baseline=50; 1 excluded (degenerate)
    "parity":  [5, 10, 25, 50, 75],  # baseline=75
    "mazes":   [5, 10, 25, 50, 75],  # baseline=75
}

# Extended training to ensure large models actually converge.
# (st01/st02 used cifar10/parity 200k, mazes 100k — large d_model may not
#  have converged, so we extend by 1.5x-2x.)
TRAINING_ITERS = {
    "cifar10": 300_001,
    "parity":  300_001,
    "mazes":   200_001,
}

# Fine-grained curve sampling for accurate steps-to-target interpolation.
TRACK_EVERY = 1_000
SAVE_EVERY = 10_000  # disk-friendly (overrides cifar10 base 2000)

# Targets for steps-to-target analysis (final-tick acc; baseline-final based).
# Multiple targets per task show whether the scaling exponent is target-sensitive.
TARGETS = {
    "cifar10": [0.55, 0.65],
    "parity":  [0.62, 0.68],
    "mazes":   [0.85, 0.88],
}


# ─────────────────────────────────────────────────────────────────────────
# GROUP BUILDERS
# ─────────────────────────────────────────────────────────────────────────
def _common_overrides(task):
    return {
        "training_iterations": TRAINING_ITERS[task],
        "track_every": TRACK_EVERY,
        "save_every": SAVE_EVERY,
    }


def make_cells_group(seeds):
    """d_model sweep at task-default iterations. 3 tasks x 4 points x N seeds."""
    exps = []
    for task, d_models in CELLS_SWEEP.items():
        module, base = BASE_CONFIGS[task]
        for dm in d_models:
            for s in seeds:
                cfg = dict(base)
                cfg["d_model"] = dm
                cfg["seed"] = s
                cfg.update(_common_overrides(task))
                name = f"{task}_cells_d{dm}_s{s}"
                exps.append(Experiment(
                    name=name, task=task, module=module, config=cfg,
                    tags=["cells", task, f"d{dm}", f"s{s}"]))
    return exps


def make_ticks_group(seeds):
    """iterations sweep at task-default d_model. 3 tasks x 5 points x N seeds."""
    exps = []
    for task, ticks in TICKS_SWEEP.items():
        module, base = BASE_CONFIGS[task]
        for t in ticks:
            for s in seeds:
                cfg = dict(base)
                cfg["iterations"] = t
                cfg["seed"] = s
                cfg.update(_common_overrides(task))
                name = f"{task}_ticks_t{t}_s{s}"
                exps.append(Experiment(
                    name=name, task=task, module=module, config=cfg,
                    tags=["ticks", task, f"t{t}", f"s{s}"]))
    return exps


GROUPS = {
    "cells": make_cells_group,
    "ticks": make_ticks_group,
}


# ─────────────────────────────────────────────────────────────────────────
# ANALYZE (run after collect)
# ─────────────────────────────────────────────────────────────────────────
def _steps_to_target(iters, accs, target):
    """First interpolated step where acc >= target. None if never reached."""
    import numpy as np
    iters, accs = np.asarray(iters, float), np.asarray(accs, float)
    if len(accs) < 2:
        return None
    for i in range(1, len(accs)):
        if accs[i - 1] < target <= accs[i]:
            denom = accs[i] - accs[i - 1]
            if denom <= 0:
                return float(iters[i])
            f = (target - accs[i - 1]) / denom
            return float(iters[i - 1] + f * (iters[i] - iters[i - 1]))
    return None


def _fit_power_law(xs, ys):
    """Fit y = a * x^b via log-log linear regression. Returns (a, b, r2)."""
    import numpy as np
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    mask = (xs > 0) & (ys > 0)
    if mask.sum() < 2:
        return None
    lx, ly = np.log(xs[mask]), np.log(ys[mask])
    b, a = np.polyfit(lx, ly, 1)
    pred = a + b * lx
    ss_res = ((ly - pred) ** 2).sum()
    ss_tot = ((ly - ly.mean()) ** 2).sum()
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return float(np.exp(a)), float(b), float(r2)


def analyze():
    """Read collected curves, compute steps-to-target, fit power law, plot."""
    import json
    import glob
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    CSV_ROOT.mkdir(parents=True, exist_ok=True)
    FIG_ROOT.mkdir(parents=True, exist_ok=True)

    curve_files = sorted(glob.glob(str(CSV_ROOT / "*curves*.json")))
    if not curve_files:
        print(f"No curves file in {CSV_ROOT}/. Collect first:")
        print("  python scripts/extract_ctm_paper_results.py --logs paper_scaling/logs "
              "--csv paper_scaling/csv_data/scaling_summary.csv "
              "--md paper_scaling/csv_data/scaling_summary.md --curves")
        return 1

    print(f"Loading curves from {curve_files[-1]}")
    curves = json.load(open(curve_files[-1]))
    print(f"  {len(curves)} runs loaded")

    # Bucket runs by (axis, task, scale_value) -> list[steps_to_target per seed]
    # axis = "cells" | "ticks"; scale_value = d_model or iterations
    bucket = {}  # (axis, task, scale, target) -> [steps, ...]
    for key, v in curves.items():
        meta = v.get("meta", {})
        task = meta.get("task")
        if task not in TARGETS:
            continue
        d_model = meta.get("d_model")
        ticks = meta.get("iterations") or meta.get("n_ticks")
        # identify axis by exp name
        is_cells = "_cells_d" in key
        is_ticks = "_ticks_t" in key
        if is_cells:
            axis, scale = "cells", d_model
        elif is_ticks:
            # iterations may not be in meta; parse from name
            import re
            m = re.search(r"_ticks_t(\d+)_", key)
            axis, scale = "ticks", int(m.group(1)) if m else ticks
        else:
            continue
        if scale is None:
            continue
        iters, accs = v.get("iters", []), v.get("test_acc", [])
        for tgt in TARGETS[task]:
            s = _steps_to_target(iters, accs, tgt)
            if s is not None:
                bucket.setdefault((axis, task, int(scale), tgt), []).append(s)

    # Report + plot per axis
    TASK_COLORS = {"cifar10": "#1f77b4", "parity": "#2ca02c", "mazes": "#ff7f0e"}
    for axis in ("cells", "ticks"):
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=False)
        print(f"\n{'='*72}\n  {axis.upper()} axis: steps-to-target vs scale\n{'='*72}")
        for col, task in enumerate(["cifar10", "parity", "mazes"]):
            ax = axes[col]
            scales = sorted({k[2] for k in bucket if k[0] == axis and k[1] == task})
            for tgt in TARGETS[task]:
                xs, ys, es = [], [], []
                for sc in scales:
                    arr = bucket.get((axis, task, sc, tgt), [])
                    if arr:
                        xs.append(sc)
                        ys.append(float(np.mean(arr)))
                        es.append(float(np.std(arr)))
                if not xs:
                    continue
                xs, ys, es = map(np.array, (xs, ys, es))
                ax.errorbar(xs, ys, yerr=es, fmt="o-", capsize=4,
                            label=f"target={tgt} (n={len([s for s in xs])})")
                fit = _fit_power_law(xs, ys)
                if fit and len(xs) >= 3:
                    a, b, r2 = fit
                    xfit = np.logspace(np.log10(xs.min() * 0.9), np.log10(xs.max() * 1.1), 50)
                    ax.plot(xfit, a * xfit ** b, "--", alpha=0.5,
                            label=f"fit: y={a:.1f}*x^{b:.2f} (R2={r2:.2f})")
                    print(f"  {task:8s} tgt={tgt:.2f}  b={b:+.3f}  R2={r2:.3f}  "
                          f"({len(xs)} pts, scales={xs.tolist()})")
                else:
                    print(f"  {task:8s} tgt={tgt:.2f}  (insufficient pts for fit, "
                          f"{len(xs)} pts)")
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlabel({"cells": "d_model (cells)", "ticks": "iterations (ticks)"}[axis])
            ax.set_ylabel("steps to target")
            ax.set_title(task)
            ax.legend(fontsize=7)
            ax.grid(True, which="both", alpha=0.3)
        fig.suptitle(f"CTM {axis} scaling law — steps-to-target vs {axis}", fontsize=13)
        fig.tight_layout()
        out = FIG_ROOT / f"{axis}_scaling.png"
        fig.savefig(out, dpi=130)
        print(f"  -> {out}")

    print("\nInterpretation:")
    print("  b ~ 0    : scaling is FREE (more capacity/thinking costs no extra steps)")
    print("  b ~ 1    : LINEAR (each 2x scale needs 2x more steps)")
    print("  b > 1    : SUPER-LINEAR (worse than linear)")
    print("  b < 0    : LARGER is FASTER to converge (per-capita sample efficiency up)")
    return 0


# ─────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────
def _print_overview(names, seeds):
    print("=" * 72)
    print(f"CTM Scaling Law — seeds={list(seeds)}, groups={names}")
    print("=" * 72)
    for g in names:
        exps = GROUPS[g](seeds)
        print(f"\n  [{g}] {len(exps)} runs")
        for task in CELLS_SWEEP:
            task_exps = [e for e in exps if e.task == task]
            if task_exps:
                scales = sorted({e.config["d_model"] if g == "cells"
                                 else e.config["iterations"]
                                 for e in task_exps})
                print(f"    {task:8s}: {len(task_exps)} runs, "
                      f"{g}={scales}, training={TRAINING_ITERS[task]//1000}k steps")
    total = sum(len(GROUPS[g](seeds)) for g in names)
    print(f"\n  TOTAL: {total} runs")
    print("=" * 72)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gpus", type=int, default=8)
    ap.add_argument("--seeds", type=int, default=3,
                    help="number of seeds per config (default 3)")
    ap.add_argument("--only", nargs="*", choices=list(GROUPS.keys()), default=None,
                    help="run only these groups (default: all = cells+ticks)")
    ap.add_argument("--dry-run", action="store_true",
                    help="print plan + GPU estimate only, do not launch")
    ap.add_argument("--mem-util", type=float, default=0.80)
    sub = ap.add_subparsers(dest="cmd")
    sub.add_parser("status", help="progress snapshot of paper_scaling/logs")
    sub.add_parser("analyze", help="fit scaling law from collected curves -> PNG")
    args = ap.parse_args()

    if args.cmd == "status":
        for g in GROUPS:
            d = LOG_ROOT / g
            if d.exists():
                print(f"\n=== {g} ===")
                status(str(d))
        return
    if args.cmd == "analyze":
        sys.exit(analyze())

    seeds = range(args.seeds)
    names = args.only if args.only else list(GROUPS.keys())

    _print_overview(names, seeds)

    if args.dry_run:
        print("\n(dry-run: showing GPU packing estimate per group)")
        for g in names:
            exps = GROUPS[g](seeds)
            print(f"\n--- {g} ({len(exps)} runs) ---")
            run_all(exps, gpus=args.gpus, log_root=str(LOG_ROOT / g),
                    dry_run=True, mem_util=args.mem_util)
        print("\n(dry-run; not launching)")
        return

    for g in names:
        exps = GROUPS[g](seeds)
        gdir = LOG_ROOT / g
        print(f"\n>>> launching {g} ({len(exps)} runs) -> {gdir}")
        run_all(exps, gpus=args.gpus, log_root=str(gdir),
                dry_run=False, mem_util=args.mem_util)

    print("\nAll groups finished. Collect with:")
    print("  python scripts/extract_ctm_paper_results.py --logs paper_scaling/logs "
          "--csv paper_scaling/csv_data/scaling_summary.csv "
          "--md paper_scaling/csv_data/scaling_summary.md --curves")
    print("Then analyze:")
    print("  python paper_scaling/run_scaling.py analyze")


if __name__ == "__main__":
    main()
