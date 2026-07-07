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


# Approximate forward FLOPs for common backbones (per sample, FLOPs = 2*MACs).
# Standard resnet refs; "-N" is a width multiplier variant used in CTM configs.
# These are constants — the backbone runs ONCE per forward (perception), so it
# is the same for every (d_model, iterations) at a given task.
BACKBONE_FLOPS = {
    "resnet18-1": 1.12e9,
    "resnet18-2": 2.24e9,
    "resnet34-1": 2.4e9,
    "resnet34-2": 4.8e9,
    "parity_backbone": 1e8,
}


def _estimate_flops_per_step(config):
    """Approximate FLOPs (not MACs) per training step: forward + backward ~ 3x forward.

    CTM anatomy (per sample, per layer):
      - backbone (resnet/parity): runs ONCE (perception), NOT per tick
      - per tick: synapse (dominant on the cells axis, ~ d_model^2)
                  + NLM (~ d_model * mem * mem_hidden)
                  + q_proj + output_proj
    Dominant-term model: precise enough for relative scaling (power-law slope b),
    not for absolute FLOP counts. Backbone constants are approximate; CTM-internal
    is computed from config shapes (SuperLinear / SynapseUNET / Linear).

    Why this matters: on the cells axis, d_model x2 -> per-step FLOPs ~ x4 (synapse
    is quadratic). So "steps-to-target ~ constant" does NOT mean "free" — the model
    just does 4x more work per step. FLOPs-to-target = steps * per-step-FLOPs is the
    fair cost metric. On the ticks axis, iterations x2 -> per-step FLOPs ~ x2 (each
    tick is one pass), so steps and FLOPs scale roughly together.
    """
    d_model = config["d_model"]
    d_input = config.get("d_input", 64)
    iterations = config["iterations"]
    synapse_depth = config.get("synapse_depth", 1)
    memory_length = config.get("memory_length", 10)
    memory_hidden_dims = config.get("memory_hidden_dims", 4)
    batch_size = config["batch_size"]
    self_cond = config.get("self_cond", True)
    n_synch_out = config.get("n_synch_out", 256)
    n_synch_action = config.get("n_synch_action", 512)
    hidden_size = config.get("hidden_size", 768)
    deep_nlms = config.get("deep_nlms", True)
    nst = config.get("neuron_select_type", "random-pairing")

    # Synchronisation pair counts (mirrors model._calc_sizes)
    if nst == "random-pairing":
        synch_action = n_synch_action
        synch_out = n_synch_out
    else:  # random / first-last -> n*(n+1)/2 pairs
        synch_action = n_synch_action * (n_synch_action + 1) // 2
        synch_out = n_synch_out * (n_synch_out + 1) // 2

    backbone = BACKBONE_FLOPS.get(config.get("backbone_type", "resnet18-1"), 1e9)

    # Per-tick CTM-internal (per sample)
    synapse_in = d_input + d_model + (d_model if self_cond else 0)
    if synapse_depth == 1:
        synapse = synapse_in * (2 * d_model) * 2          # Linear(in, 2d) + GLU
    else:
        synapse = synapse_in * d_model * synapse_depth * 2 * 2  # SynapseUNET down+up

    if deep_nlms:
        nlm = (d_model * memory_length * (2 * memory_hidden_dims) * 2
               + d_model * memory_hidden_dims * 2 * 2)
    else:
        nlm = d_model * memory_length * 2 * 2

    # q_proj (sync_action -> d_input); attention itself is small for T=1 CTM inputs;
    # output_proj (sync_out -> hidden_size)
    q_proj = synch_action * d_input * 2
    output_proj = synch_out * hidden_size * 2

    per_tick = synapse + nlm + q_proj + output_proj
    forward_per_sample = backbone + iterations * per_tick
    return batch_size * forward_per_sample * 3.0  # forward + backward ~= 3x


def _config_for_meta(task, meta):
    """Reconstruct full config from curves meta + task BASE (for FLOPs estimate)."""
    _, base = BASE_CONFIGS[task]
    cfg = dict(base)
    cfg["d_model"] = meta.get("d_model", base["d_model"])
    cfg["batch_size"] = meta.get("batch_size", base["batch_size"])
    return cfg


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

    # Bucket runs by (axis, task, scale, target) -> [steps_to_target per seed].
    # Also cache per-step FLOPs per (axis, task, scale): config-driven, identical
    # across seeds at the same scale, so computed once.
    bucket = {}
    flops_per_step = {}
    for key, v in curves.items():
        meta = v.get("meta", {})
        task = meta.get("task")
        if task not in TARGETS:
            continue
        d_model = meta.get("d_model")
        ticks = meta.get("iterations") or meta.get("n_ticks")
        is_cells = "_cells_d" in key
        is_ticks = "_ticks_t" in key
        if is_cells:
            axis, scale = "cells", d_model
        elif is_ticks:
            import re
            m = re.search(r"_ticks_t(\d+)_", key)
            axis, scale = "ticks", int(m.group(1)) if m else ticks
        else:
            continue
        if scale is None:
            continue
        ck = (axis, task, int(scale))
        if ck not in flops_per_step:
            cfg = _config_for_meta(task, meta)
            if axis == "ticks":
                cfg["iterations"] = scale
            flops_per_step[ck] = _estimate_flops_per_step(cfg)
        iters, accs = v.get("iters", []), v.get("test_acc", [])
        for tgt in TARGETS[task]:
            s = _steps_to_target(iters, accs, tgt)
            if s is not None:
                bucket.setdefault((axis, task, int(scale), tgt), []).append(s)

    # Plot per axis: row 0 = steps-to-target, row 1 = FLOPs-to-target.
    # FLOPs-to-target = mean(steps) * per-step-FLOPs; error bar scales the same way.
    for axis in ("cells", "ticks"):
        fig, axes = plt.subplots(2, 3, figsize=(15, 8.5), sharey="row")
        xname = {"cells": "d_model (cells)", "ticks": "iterations (ticks)"}[axis]
        for metric, row in (("steps", 0), ("flops", 1)):
            ylabel = {"steps": "steps to target",
                      "flops": "FLOPs to target"}[metric]
            print(f"\n{'='*72}\n  {axis.upper()} axis / {metric}: {ylabel} vs {axis}\n{'='*72}")
            for col, task in enumerate(["cifar10", "parity", "mazes"]):
                ax = axes[row][col]
                scales = sorted({k[2] for k in bucket
                                 if k[0] == axis and k[1] == task})
                for tgt in TARGETS[task]:
                    xs, ys, es = [], [], []
                    for sc in scales:
                        arr = bucket.get((axis, task, sc, tgt), [])
                        if not arr:
                            continue
                        fps = flops_per_step.get((axis, task, sc), 0.0)
                        m_val = float(np.mean(arr))
                        if metric == "flops":
                            m_val *= fps
                        xs.append(sc)
                        ys.append(m_val)
                        s_val = float(np.std(arr)) * (fps if metric == "flops" else 1.0)
                        es.append(s_val)
                    if not xs:
                        continue
                    xs, ys, es = map(np.array, (xs, ys, es))
                    ax.errorbar(xs, ys, yerr=es, fmt="o-", capsize=4,
                                label=f"target={tgt} (n={len(xs)})")
                    fit = _fit_power_law(xs, ys)
                    if fit and len(xs) >= 3:
                        a, b, r2 = fit
                        xfit = np.logspace(np.log10(xs.min() * 0.9),
                                           np.log10(xs.max() * 1.1), 50)
                        ax.plot(xfit, a * xfit ** b, "--", alpha=0.5,
                                label=f"fit y={a:.1g}*x^{b:.2f} (R2={r2:.2f})")
                        print(f"  {task:8s} tgt={tgt:.2f}  b={b:+.3f}  R2={r2:.3f}  "
                              f"({len(xs)} pts)")
                    else:
                        print(f"  {task:8s} tgt={tgt:.2f}  (insufficient pts for fit, "
                              f"{len(xs)} pts)")
                ax.set_xscale("log")
                ax.set_yscale("log")
                ax.set_xlabel(xname)
                ax.set_ylabel(ylabel)
                if row == 0:
                    ax.set_title(task)
                ax.legend(fontsize=7)
                ax.grid(True, which="both", alpha=0.3)
        fig.suptitle(
            f"CTM {axis} scaling law — steps (top) & FLOPs (bottom) vs {axis}",
            fontsize=13)
        fig.tight_layout()
        out = FIG_ROOT / f"{axis}_scaling.png"
        fig.savefig(out, dpi=130)
        print(f"  -> {out}")

    print("\nInterpretation (b = power-law exponent of [steps|FLOPs] vs scale):")
    print("  STEPS axis:  b~0 free sample-efficiency | b~1 linear | b>1 super-linear")
    print("  FLOPs axis:  compares TOTAL compute; on cells axis, per-step FLOPs ~ d_model^2")
    print("               so a 'free' steps-fit (b~0) becomes b~2 in FLOPs -> NOT free.")
    print("               The fair headline is the FLOPs row.")
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
