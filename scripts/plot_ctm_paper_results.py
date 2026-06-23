#!/usr/bin/env python3
"""Plot CTM paper sweep results from extracted curves.json + summary.csv.

Generates 4 key figures (filtered to drop incomplete seeds):
  1. st02 tick sweep   - best_test_acc vs n_ticks (4 tasks, errorbar)
  2. st01 capacity sweep - bar chart per task (d_model / heads / mh / sd / nst)
  3. most_certain vs final test acc scatter (CTM most-certain tick value)
  4. convergence curves for cifar10 + mazes tick sweep

Usage:
    python scripts/plot_ctm_paper_results.py
    python scripts/plot_ctm_paper_results.py --min-iter 100000
    python scripts/plot_ctm_paper_results.py --data-dir csv_data  # default
"""

import argparse
import json
import math
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "csv_data"
DEFAULT_OUT = ROOT / "runs" / "figures" / "ctm_paper"

# baseline reference accs from st00 (paper config) for horizontal lines
BASELINE_PAPER = {
    "cifar10": 0.6443,
    "mazes": 0.8028,
    "parity": 0.6797,
    "qamnist": 0.2341,
    "sort": 0.7146,
}
# ff (feed-forward) baseline for cifar10
BASELINE_FF = {"cifar10": 0.8407}

TASK_COLORS = {
    "cifar10": "#1f77b4",
    "mazes": "#ff7f0e",
    "parity": "#2ca02c",
    "qamnist": "#d62728",
    "sort": "#9467bd",
}
TASK_ORDER = ["cifar10", "mazes", "parity", "sort"]


def load_data(data_dir):
    curves_path = data_dir / "ctm_paper_curves.json"
    summary_path = data_dir / "ctm_paper_summary.csv"
    with open(curves_path) as f:
        curves = json.load(f)
    df = pd.read_csv(summary_path)
    # coerce numeric cols
    for c in ["final_iter", "best_test_acc", "final_test_acc",
              "best_test_acc_mc", "final_test_acc_mc", "n_points", "seed"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df, curves


def filter_complete(df, min_iter):
    """Drop seeds that didn't reach min_iter (interrupted / OOM half-runs)."""
    return df[df["final_iter"].fillna(0) >= min_iter].copy()


def agg_sweep(df_sub):
    """Group by (stage, task, sweep), return per-group mean/std + seed count."""
    g = df_sub.groupby(["stage", "task", "sweep"])
    out = []
    for (stage, task, sweep), grp in g:
        rec = {
            "stage": stage, "task": task, "sweep": sweep,
            "n_seeds": len(grp),
            "best_mean": grp["best_test_acc"].mean(),
            "best_std": grp["best_test_acc"].std(ddof=1) if len(grp) > 1 else 0.0,
            "final_mean": grp["final_test_acc"].mean(),
            "bmc_mean": grp["best_test_acc_mc"].dropna().mean() if grp["best_test_acc_mc"].notna().any() else None,
        }
        out.append(rec)
    return pd.DataFrame(out)


# ---------------------------- Figure 1: tick sweep ----------------------------

def parse_tick(sweep):
    """tick1 -> 1, tick25 -> 25. qamnist uses 'repeat' instead."""
    m = re.match(r"tick(\d+)$", sweep)
    return int(m.group(1)) if m else None


def fig_tick_sweep(df_complete, out_dir, min_iter):
    """st02 tick sweep: 2x2 grid, errorbar over seeds, baseline lines."""
    sub = df_complete[df_complete["stage"] == "st02"].copy()
    sub["tick"] = sub["sweep"].apply(parse_tick)
    sub = sub.dropna(subset=["tick"])
    if sub.empty:
        print("[fig1] no st02 tick data")
        return
    agg = agg_sweep(sub)
    agg["tick"] = agg["sweep"].apply(parse_tick)

    tasks = [t for t in TASK_ORDER if t in agg["task"].unique()]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    axes = axes.flatten()
    for i, task in enumerate(tasks):
        ax = axes[i]
        a = agg[agg["task"] == task].sort_values("tick")
        x = a["tick"].values
        y = a["best_mean"].values * 100
        yerr = a["best_std"].values * 100
        ax.errorbar(x, y, yerr=yerr, marker="o", capsize=4, linewidth=2,
                    markersize=8, color=TASK_COLORS[task])
        # baseline lines
        if task in BASELINE_PAPER:
            ax.axhline(BASELINE_PAPER[task] * 100, color="gray", linestyle="--",
                       alpha=0.6, label="st00 paper baseline")
        if task in BASELINE_FF:
            ax.axhline(BASELINE_FF[task] * 100, color="red", linestyle=":",
                       alpha=0.5, label="st00 ff baseline")
        # annotate seed count (warn if low)
        for xi, yi, n in zip(x, y, a["n_seeds"].values):
            label = f"n={n}" + ("!" if n < 2 else "")
            ax.annotate(label, (xi, yi), textcoords="offset points",
                        xytext=(0, 8), ha="center", fontsize=7, color="red" if n < 2 else "black")
        ax.set_xscale("log")
        ax.set_xticks(x)
        ax.set_xticklabels([str(int(v)) for v in x])
        ax.set_title(f"{task} — st02 tick sweep (seeds with iter≥{min_iter//1000}k)",
                     fontsize=11)
        ax.set_xlabel("n_ticks (log scale)")
        ax.set_ylabel("best test acc (%)")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="lower right")
    # hide unused
    for j in range(len(tasks), 4):
        axes[j].axis("off")
    fig.suptitle("CTM Paper — st02: How many 'thought ticks' does each task need?",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    path = out_dir / "fig1_tick_sweep.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig1] -> {path}")


# ------------------------ Figure 2: capacity sweep ----------------------------

SWEEP_PREFIXES = [
    ("d_model", "d_model"),
    ("headsh", "heads"),
    ("memory_hidden_dims", "memory_hidden_dims"),
    ("synapse_depth", "synapse_depth"),
    ("nst", "neuron_select_type"),
]
DEFAULTS = {
    "cifar10": dict(d_model=256, heads=16, memory_hidden_dims=64,
                    synapse_depth=5, neuron_select_type="random-pairing"),
    "mazes": dict(d_model=2048, heads=16, memory_hidden_dims=32,
                  synapse_depth=8, neuron_select_type="first-last"),
    "parity": dict(d_model=1024, heads=8, memory_hidden_dims=16,
                   synapse_depth=1, neuron_select_type="random"),
    "sort": dict(d_model=512, heads=4, memory_hidden_dims=4,
                 synapse_depth=4, neuron_select_type="random-pairing"),
}


def parse_capacity_sweep(sweep, task):
    """Return (var_name, value_str) for a st01 sweep name."""
    for prefix, var in SWEEP_PREFIXES:
        if sweep.startswith(prefix):
            val = sweep[len(prefix):].lstrip("s")  # 'headsh16' -> '16', 'synapse_depthsd2' -> 'd2'->'2'
            # special: synapse_depthsd2 -> strip leading 'd', 'sd2' handled by prefix 'synapse_depth' -> 'sd2'->'d2'? fix
            if var == "synapse_depth" and val.startswith("d"):
                val = val[1:]
            if var == "memory_hidden_dims" and val.startswith("mh"):
                val = val[2:]
            if var == "neuron_select_type":
                val = sweep[len("nst"):]
                if not val:
                    val = "default"
            return var, val
    return None, None


def fig_capacity_sweep(df_complete, out_dir, min_iter):
    """st01 capacity sweep: bar chart, normalised to baseline."""
    sub = df_complete[df_complete["stage"] == "st01"].copy()
    if sub.empty:
        print("[fig2] no st01 data")
        return
    agg = agg_sweep(sub)

    tasks = [t for t in TASK_ORDER if t in agg["task"].unique()]
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    axes = axes.flatten()
    for i, task in enumerate(tasks):
        ax = axes[i]
        a = agg[agg["task"] == task].copy()
        # parse (var, val)
        parsed = a["sweep"].apply(lambda s: parse_capacity_sweep(s, task))
        a["var"] = parsed.apply(lambda p: p[0])
        a["val"] = parsed.apply(lambda p: p[1])
        a = a.dropna(subset=["var"])
        # group by var, plot each variant
        a["label"] = a["var"] + "=" + a["val"].astype(str)
        a = a.sort_values(["var", "val"])
        labels = a["label"].tolist()
        means = a["best_mean"].values * 100
        stds = a["best_std"].values * 100
        colors = [plt.cm.tab20(j % 20) for j in range(len(labels))]
        bars = ax.bar(range(len(labels)), means, yerr=stds, capsize=3,
                      color=colors, alpha=0.85)
        # baseline
        if task in BASELINE_PAPER:
            ax.axhline(BASELINE_PAPER[task] * 100, color="gray", linestyle="--",
                       alpha=0.7, label="st00 paper baseline")
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
        ax.set_title(f"{task} — st01 capacity sweep", fontsize=11)
        ax.set_ylabel("best test acc (%)")
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend(fontsize=8, loc="upper right")
    for j in range(len(tasks), 4):
        axes[j].axis("off")
    fig.suptitle("CTM Paper — st01: Capacity / structural sweeps",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    path = out_dir / "fig2_capacity_sweep.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig2] -> {path}")


# ------------------- Figure 3: most_certain vs final --------------------------

def fig_mc_vs_final(df_complete, out_dir):
    """Scatter: best_test_acc_mc (y) vs final_test_acc (x). CTM most-certain tick value."""
    sub = df_complete.dropna(subset=["best_test_acc_mc", "final_test_acc"]).copy()
    sub = sub[(sub["best_test_acc_mc"] > 0) & (sub["final_test_acc"] > 0)]
    # exclude st00 baselines (keep only sweeps)
    sub = sub[sub["stage"].isin([f"st{n:02d}" for n in range(1, 25)])]
    if sub.empty:
        print("[fig3] no mc data")
        return
    fig, ax = plt.subplots(figsize=(9, 8))
    for task, grp in sub.groupby("task"):
        ax.scatter(grp["final_test_acc"] * 100, grp["best_test_acc_mc"] * 100,
                   label=task, color=TASK_COLORS.get(task, "gray"),
                   alpha=0.7, s=55, edgecolors="white", linewidths=0.5)
    # diagonal
    lo, hi = 0, 100
    ax.plot([lo, hi], [lo, hi], "k--", alpha=0.4, label="y=x (no most-certain gain)")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("final test acc (%)  —  using only the LAST tick's prediction")
    ax.set_ylabel("best test acc_mc (%)  —  using the MOST CERTAIN tick")
    ax.set_title("CTM most-certain tick selection: per-sample best tick >> final tick\n"
                 "(points above the diagonal = most-certain tick helps)",
                 fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    path = out_dir / "fig3_mc_vs_final.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig3] -> {path}")


# ----------------------- Figure 4: convergence curves ------------------------

def fig_convergence(df_complete, curves, out_dir, min_iter):
    """cifar10 + mazes tick-sweep convergence curves overlay.

    Uses key name ('st02/cifar10_tick1') + df_complete lookup, so works even
    when curves.json lacks a 'meta' field (older extract versions).
    """
    # index df_complete by (stage, exp) for fast lookup
    df_idx = df_complete.set_index(["stage", "exp"])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    for idx, task in enumerate(["cifar10", "mazes"]):
        ax = axes[idx]
        runs = []
        for key, v in curves.items():
            parts = key.split("/", 1)
            if len(parts) != 2:
                continue
            stage, exp_name = parts
            if stage != "st02":
                continue
            if (stage, exp_name) not in df_idx.index:
                continue
            row = df_idx.loc[(stage, exp_name)]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            if row["task"] != task:
                continue
            fi = row.get("final_iter", 0)
            if pd.isna(fi) or fi < min_iter:
                continue
            tick = parse_tick(row["sweep"])
            if tick is None:
                continue
            iters = v.get("iters", [])
            accs = v.get("test_acc", [])
            if len(iters) != len(accs) or len(iters) < 2:
                continue
            runs.append((tick, iters, accs))
        # average over seeds per tick
        by_tick = {}
        for tick, iters, accs in runs:
            by_tick.setdefault(tick, []).append((iters, accs))
        cmap = plt.cm.viridis
        n_ticks = len(by_tick)
        if n_ticks == 0:
            ax.text(0.5, 0.5, f"no complete {task} tick runs\n(iter≥{min_iter//1000}k)",
                    ha="center", va="center", transform=ax.transAxes, fontsize=12)
            ax.set_title(f"{task} — no data", fontsize=11)
            continue
        for i, tick in enumerate(sorted(by_tick.keys())):
            seed_runs = by_tick[tick]
            max_iter = max(max(it) for it, _ in seed_runs)
            grid = np.linspace(0, max_iter, 100)
            interp_accs = []
            for it, ac in seed_runs:
                if len(it) >= 2:
                    interp = np.interp(grid, it, ac)
                    interp_accs.append(interp)
            if not interp_accs:
                continue
            mean_curve = np.mean(interp_accs, axis=0) * 100
            color = cmap(i / max(n_ticks - 1, 1))
            ax.plot(grid, mean_curve, color=color, linewidth=2,
                    label=f"tick={tick} (n={len(interp_accs)})")
        if task in BASELINE_PAPER:
            ax.axhline(BASELINE_PAPER[task] * 100, color="gray", linestyle="--",
                       alpha=0.6, label="st00 paper baseline")
        ax.set_xlabel("training iteration")
        ax.set_ylabel("test acc (%)")
        ax.set_title(f"{task} — st02 tick sweep convergence (seeds iter≥{min_iter//1000}k)",
                     fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="lower right")
    fig.suptitle("Convergence: more ticks ≠ faster learning on cifar10",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    path = out_dir / "fig4_convergence.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig4] -> {path}")


# ---------------------- Figure 7: compute cost vs n_ticks ----------------------

# default training_iterations budget per task (from experiment_plan_ctm_paper.py)
PLANNED_ITERS = {
    "cifar10": 200001,
    "mazes": 100001,
    "parity": 200001,
    "qamnist": 200001,
    "sort": 100001,
}


def fig_compute_cost(df_complete, out_dir, min_iter):
    """Compute-cost dependency for st02 tick sweep (cifar10 + mazes).

    Three side-by-side panels explain why convergence curves in fig4 have
    uneven lengths:
      1. iterations actually completed vs n_ticks  (shows truncation)
      2. total compute proxy = final_iter * n_ticks vs n_ticks  (fairness check)
      3. best test acc vs total compute  (does compute explain acc gaps?)

    Compute proxy assumes cost scales linearly with n_ticks (each tick adds one
    full recurrent step in the CTM). No wall-clock data is available, so this is
    an approximate but useful proxy.
    """
    sub = df_complete[df_complete["stage"] == "st02"].copy()
    sub["tick"] = sub["sweep"].apply(parse_tick)
    sub = sub.dropna(subset=["tick"])
    sub = sub[sub["task"].isin(["cifar10", "mazes"])]
    if sub.empty:
        print("[fig7] no st02 tick data for cifar10/mazes")
        return

    # aggregate over seeds per (task, tick)
    agg = (
        sub.groupby(["task", "tick"])
        .agg(
            iter_mean=("final_iter", "mean"),
            iter_std=("final_iter", "std"),
            acc_mean=("best_test_acc", "mean"),
            acc_std=("best_test_acc", "std"),
            n_seeds=("exp", "size"),
        )
        .reset_index()
    )
    # fill std=0 when only 1 seed
    agg["iter_std"] = agg["iter_std"].fillna(0)
    agg["acc_std"] = agg["acc_std"].fillna(0)
    # total compute proxy = iterations * ticks
    agg["compute_mean"] = agg["iter_mean"] * agg["tick"]
    # planned total compute = training_iterations * n_ticks
    agg["compute_planned"] = agg["task"].map(PLANNED_ITERS) * agg["tick"]

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5))

    # --- Panel 1: iters completed vs n_ticks ---
    ax = axes[0]
    width = 0.35
    tasks_here = [t for t in ["cifar10", "mazes"] if t in agg["task"].unique()]
    for i, task in enumerate(tasks_here):
        a = agg[agg["task"] == task].sort_values("tick")
        x = a["tick"].values
        # offset bars side by side (log-scale x -> use index positioning)
        xpos = np.arange(len(x)) + (i - 0.5 * (len(tasks_here) - 1)) * width
        ax.bar(
            xpos, a["iter_mean"].values / 1000, width=width,
            yerr=a["iter_std"].values / 1000, capsize=3,
            color=TASK_COLORS[task], alpha=0.8, label=task,
            edgecolor="black", linewidth=0.5,
        )
        # planned-iter line (per task, dashed horizontal markers per tick)
        planned_k = PLANNED_ITERS[task] / 1000
        ax.axhline(planned_k, color=TASK_COLORS[task], linestyle="--",
                   alpha=0.4, linewidth=1)
        # annotate n_seeds
        for xi, yi, n in zip(xpos, a["iter_mean"].values / 1000, a["n_seeds"].values):
            ax.annotate(f"n={int(n)}", (xi, yi), textcoords="offset points",
                        xytext=(0, 6), ha="center", fontsize=7,
                        color="red" if n < 2 else "black")
    # x ticks: union of ticks across tasks, log scale
    all_ticks = sorted(agg["tick"].unique())
    ax.set_xticks(np.arange(len(all_ticks)))
    ax.set_xticklabels([str(int(t)) for t in all_ticks])
    ax.set_xlabel("n_ticks (log scale)")
    ax.set_ylabel("iterations completed (k)")
    ax.set_title("(a) Iterations completed vs n_ticks\n"
                 "(dashed = planned budget; bars below = truncated)",
                 fontsize=10)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize=8, loc="upper right")

    # --- Panel 2: total compute proxy vs n_ticks ---
    ax = axes[1]
    for i, task in enumerate(tasks_here):
        a = agg[agg["task"] == task].sort_values("tick")
        x = a["tick"].values
        xpos = np.arange(len(x)) + (i - 0.5 * (len(tasks_here) - 1)) * width
        # planned (faint outline)
        ax.bar(xpos, a["compute_planned"].values / 1e6, width=width,
               color=TASK_COLORS[task], alpha=0.18, edgecolor=TASK_COLORS[task],
               linewidth=1.2, hatch="//")
        # actual (solid)
        ax.bar(xpos, a["compute_mean"].values / 1e6, width=width,
               color=TASK_COLORS[task], alpha=0.85,
               edgecolor="black", linewidth=0.5,
               label=f"{task} (actual)")
    # legend with hatch
    from matplotlib.patches import Patch
    hatch_handles = [
        Patch(facecolor="gray", alpha=0.18, hatch="//", edgecolor="gray",
              label="planned budget"),
    ]
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles + hatch_handles, fontsize=8, loc="upper left")
    ax.set_xticks(np.arange(len(all_ticks)))
    ax.set_xticklabels([str(int(t)) for t in all_ticks])
    ax.set_xlabel("n_ticks (log scale)")
    ax.set_ylabel("total compute proxy  (iter × n_ticks, units of 1M)")
    ax.set_title("(b) Total compute actually used vs n_ticks\n"
                 "(hatched = planned; solid = actual)",
                 fontsize=10)
    ax.grid(True, axis="y", alpha=0.3)

    # --- Panel 3: best acc vs total compute ---
    ax = axes[2]
    for task in tasks_here:
        a = agg[agg["task"] == task].sort_values("compute_mean")
        ax.errorbar(a["compute_mean"].values / 1e6, a["acc_mean"].values * 100,
                    yerr=a["acc_std"].values * 100, marker="o", capsize=4,
                    linewidth=2, markersize=8, color=TASK_COLORS[task],
                    label=task, alpha=0.85)
        # annotate tick value next to each point
        for x, y, t in zip(a["compute_mean"].values / 1e6,
                            a["acc_mean"].values * 100, a["tick"].values):
            ax.annotate(f"t={int(t)}", (x, y), textcoords="offset points",
                        xytext=(6, 4), fontsize=7, color=TASK_COLORS[task])
        if task in BASELINE_PAPER:
            ax.axhline(BASELINE_PAPER[task] * 100, color=TASK_COLORS[task],
                       linestyle=":", alpha=0.5, linewidth=1)
    ax.set_xscale("log")
    ax.set_xlabel("total compute proxy  (iter × n_ticks, log scale)")
    ax.set_ylabel("best test acc (%)")
    ax.set_title("(c) Acc vs total compute used\n"
                 "(dotted = st00 paper baseline; same compute ≠ same acc)",
                 fontsize=10)
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(fontsize=8, loc="lower right")

    fig.suptitle(
        "Compute-cost dependency in the st02 tick sweep "
        "(explains uneven curve lengths in fig4)",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    path = out_dir / "fig7_compute_cost.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig7] -> {path}")


# -------------------------- Figure 5: jepa weight sweep -----------------------

def parse_jepa_weight(sweep):
    """jepa_w0.1 -> 0.1, jepa_w1.0 -> 1.0."""
    m = re.match(r"jepa_w(\d+\.\d+)$", sweep)
    return float(m.group(1)) if m else None


def fig_jepa_weight(df_complete, out_dir, exclude_tasks=None, filename=None):
    """st04 jepa_weight scan (0.1, 0.5, 1.0): 2x2 bar chart, one panel per task.

    exclude_tasks: list of task names to skip (e.g. ['sort']).
    filename: output filename (default fig5_jepa_weight.png).
    """
    sub = df_complete[df_complete["stage"] == "st04"].copy()
    sub["weight"] = sub["sweep"].apply(parse_jepa_weight)
    sub = sub.dropna(subset=["weight"])
    if sub.empty:
        print("[fig5] no st04 jepa_weight data")
        return

    exclude_tasks = exclude_tasks or []
    tasks = [t for t in ["cifar10", "mazes", "qamnist", "sort", "parity"]
             if t in sub["task"].unique() and t not in exclude_tasks]
    weights_all = [0.1, 0.5, 1.0]
    bar_colors = ["#2ca02c", "#ff7f0e", "#d62728"]

    n_tasks = len(tasks)
    ncols = 2
    nrows = (n_tasks + 1) // 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(11, 4.5 * nrows),
                             squeeze=False)
    axes_flat = axes.flatten()

    for idx, task in enumerate(tasks):
        ax = axes_flat[idx]
        bl = BASELINE_PAPER.get(task, 0) * 100
        grp = sub[sub["task"] == task]
        agg = grp.groupby("weight").agg(
            mean=("best_test_acc", "mean"),
            std=("best_test_acc", "std"),
            n=("best_test_acc", "count"),
        ).reset_index().sort_values("weight")

        vals, errs, present = [], [], []
        for w in weights_all:
            row = agg[agg["weight"] == w]
            if not row.empty:
                vals.append(row["mean"].values[0] * 100)
                errs.append((row["std"].fillna(0).values[0]) * 100)
                present.append(int(row["n"].values[0]))
            else:
                vals.append(0)
                errs.append(0)
                present.append(0)

        x = np.arange(len(weights_all))
        bars = ax.bar(x, vals, 0.55, color=bar_colors, edgecolor="black",
                      linewidth=0.6, alpha=0.85, zorder=3,
                      yerr=errs, capsize=4,
                      error_kw={"elinewidth": 1, "ecolor": "#333"})
        for i, (v, n) in enumerate(zip(vals, present)):
            if n > 0:
                d = v - bl
                sign = "+" if d >= 0 else ""
                ax.text(i, v + errs[i] + 1.2,
                        f"{v:.1f}%\n({sign}{d:.1f}pp)",
                        ha="center", fontsize=8, fontweight="bold",
                        color="#2ca02c" if d >= 0 else "#d62728")

        ax.axhline(bl, color="#333333", linestyle="--", linewidth=1.8,
                   alpha=1.0, zorder=2)
        ax.text(len(weights_all) - 0.5, bl + 0.6,
                f"baseline = {bl:.1f}%", fontsize=9, fontweight="bold",
                color="#333333", ha="right", va="bottom")

        ax.set_xticks(x)
        ax.set_xticklabels([f"w={w}" for w in weights_all], fontsize=10)
        ax.set_ylabel("best test acc (%)", fontsize=10)
        ax.set_title(task, fontsize=12, fontweight="bold",
                     color=TASK_COLORS.get(task, "black"))
        ax.grid(True, axis="y", alpha=0.2)
        top = max(vals + [bl]) + max(errs) + 8
        bottom = max(0, min(vals + [bl]) - 10)
        ax.set_ylim(bottom, top)

    for idx in range(n_tasks, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    if exclude_tasks:
        fig.suptitle("st04: JEPA auxiliary loss weight sweep — visual tasks only"
                     " (bar = mean, errorbar = std)",
                     fontsize=13, fontweight="bold")
    else:
        fig.suptitle("st04: JEPA auxiliary loss weight sweep (bar = mean, errorbar = std)",
                     fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    path = out_dir / (filename or "fig5_jepa_weight.png")
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig5] -> {path}")


# -------------------------- Figure 6: jepa variants ---------------------------

VARIANT_MAP = {
    "jepa_w0.1": "w0.1\n(default)",
    "jepa_mse": "mse loss",
    "jepa_nostopgrad": "no\nstop-grad",
    "jepa_pd1": "predict_\ndelta=1",
    "jepa_pd4": "predict_\ndelta=4",
}
VARIANT_ORDER = ["jepa_w0.1", "jepa_mse", "jepa_nostopgrad", "jepa_pd1", "jepa_pd4"]


def fig_jepa_variants(df_complete, out_dir):
    """st04 jepa variant comparison: bar chart per task (2x2)."""
    sub = df_complete[df_complete["stage"] == "st04"].copy()
    sub = sub[sub["sweep"].isin(VARIANT_MAP.keys())]
    if sub.empty:
        print("[fig6] no st04 jepa variant data")
        return

    tasks = [t for t in ["cifar10", "mazes", "qamnist", "sort"]
             if t in sub["task"].unique()]
    n_tasks = len(tasks)
    nrows = (n_tasks + 1) // 2
    fig, axes = plt.subplots(nrows, 2, figsize=(13, 4.5 * nrows), squeeze=False)
    axes = axes.flatten()
    for i, task in enumerate(tasks):
        ax = axes[i]
        agg = agg_sweep(sub[sub["task"] == task]).set_index("sweep")
        labels, means, stds, colors = [], [], [], []
        present = []
        for j, sw in enumerate(VARIANT_ORDER):
            if sw not in agg.index:
                continue
            row = agg.loc[sw]
            labels.append(VARIANT_MAP[sw])
            means.append(row["best_mean"] * 100)
            stds.append(row["best_std"] * 100)
            present.append(sw)
            colors.append(plt.cm.Set2(j / len(VARIANT_ORDER)))
        x = np.arange(len(labels))
        bars = ax.bar(x, means, yerr=stds, capsize=4, color=colors,
                      alpha=0.85, edgecolor="black", linewidth=0.5)
        # annotate values + n_seeds
        for xi, yi, sw in zip(x, means, present):
            n = int(agg.loc[sw, "n_seeds"])
            ax.annotate(f"{yi:.1f}%\n(n={n})", (xi, yi), textcoords="offset points",
                        xytext=(0, 6), ha="center", fontsize=7.5)
        # baseline
        if task in BASELINE_PAPER:
            ax.axhline(BASELINE_PAPER[task] * 100, color="gray", linestyle="--",
                       alpha=0.7, label="st00 paper baseline")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_title(f"{task} — st04 JEPA variants", fontsize=11)
        ax.set_ylabel("best test acc (%)")
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend(fontsize=8, loc="upper right")
    for j in range(n_tasks, len(axes)):
        axes[j].axis("off")
    fig.suptitle("st04: JEPA auxiliary loss — does each component help?",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    path = out_dir / "fig6_jepa_variants.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig6] -> {path}")


# ------------------- New figures: ideas stages (fig8-12) ---------------------

# Stage-specific marker: annotate figures whose underlying data was affected
# by the log_dir overwrite bug (fixed 2026-06-19). The surviving numbers are
# real runs but with one or more hyperparameter dimensions collapsed; readers
# should treat per-axis trends as trustworthy and collapsed dimensions as
# "one representative variant". See BUGFIX_STAGES in experiment_plan_ctm_paper.py.
_BUGFIX_NOTE = {
    "st06": "note: compute_weight dim collapsed (cw=0/0.001 overwrote)",
    "st09": "note: reflex_weight dim collapsed (w=0.1/0.2 overwrote)",
    "st10": "note: revise_w/corrupt_p dims collapsed (4 variants overwrote)",
    "st12": "note: decays/distill_weight dims collapsed",
    "st13": "note: threshold/compute_weight dims collapsed",
    "st14": "note: sparsity frac dim collapsed",
    "st15": "note: sparsity frac dim collapsed",
}


def fig_baselines(df_complete, out_dir):
    """st00 baseline reproduction: bar chart of all paper / LSTM / FF baselines.

    One bar per (task, baseline_type). Horizontal line per task shows the
    'paper config' reference so subsequent figures can compare against it.
    """
    sub = df_complete[df_complete["stage"] == "st00"].copy()
    if sub.empty:
        print("[fig8] no st00 data")
        return
    # build per-row label
    sub["label"] = sub.apply(
        lambda r: f"{r['task']}\n({r['sweep']})", axis=1)
    sub = sub.sort_values(["task", "sweep"])
    labels = sub["label"].tolist()
    means = sub["best_test_acc"].values * 100
    # color by task
    colors = [TASK_COLORS.get(t, "gray") for t in sub["task"]]
    # hatch the non-paper baselines (lstm/ff) for visual distinction
    hatches = ["" if s == "paper" else "//" for s in sub["sweep"]]

    fig, ax = plt.subplots(figsize=(11, 6))
    bars = ax.bar(range(len(labels)), means, color=colors, alpha=0.85,
                  edgecolor="black", linewidth=0.6)
    for b, h in zip(bars, hatches):
        b.set_hatch(h)
    for xi, yi, n in zip(range(len(labels)), means, sub["n_points"].fillna(1).values):
        ax.annotate(f"{yi:.1f}%", (xi, yi), textcoords="offset points",
                    xytext=(0, 5), ha="center", fontsize=8)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylabel("best test acc (%)")
    ax.set_title("CTM Paper — st00: baseline reproduction\n"
                 "(solid=paper config, // = LSTM / FF baselines)",
                 fontsize=12, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.3)
    # task-color legend
    from matplotlib.patches import Patch
    handles = [Patch(facecolor=TASK_COLORS[t], label=t)
               for t in TASK_ORDER if t in sub["task"].unique()]
    ax.legend(handles=handles, loc="upper right", fontsize=9)
    fig.tight_layout()
    path = out_dir / "fig8_baselines.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig8] -> {path}")


def fig_simplified_core(df_complete, out_dir, min_iter):
    """st03 simplified core: sd1_mh2 / sd2_mh2 vs paper baseline (2x2 grid).

    Compares a stripped-down CTM (few synapses / tiny memory) against the
    paper config. Clean data (not affected by the log_dir bug; tick is the
    only extra dim and it is determined by task).
    """
    sub = df_complete[df_complete["stage"] == "st03"].copy()
    if sub.empty:
        print("[fig9] no st03 data")
        return
    agg = agg_sweep(sub)
    tasks = [t for t in TASK_ORDER if t in agg["task"].unique()]
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes = axes.flatten()
    for i, task in enumerate(tasks):
        ax = axes[i]
        a = agg[agg["task"] == task].sort_values("sweep")
        labels = a["sweep"].tolist()
        means = a["best_mean"].values * 100
        stds = a["best_std"].values * 100
        x = np.arange(len(labels))
        colors = [plt.cm.Set2(j / 3) for j in range(len(labels))]
        ax.bar(x, means, yerr=stds, capsize=4, color=colors, alpha=0.85,
               edgecolor="black", linewidth=0.5)
        for xi, yi, n in zip(x, means, a["n_seeds"].values):
            tag = f"{yi:.1f}%\n(n={int(n)})" + ("!" if n < 2 else "")
            ax.annotate(tag, (xi, yi), textcoords="offset points",
                        xytext=(0, 6), ha="center", fontsize=8)
        if task in BASELINE_PAPER:
            ax.axhline(BASELINE_PAPER[task] * 100, color="gray", linestyle="--",
                       alpha=0.7, label="st00 paper baseline")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_title(f"{task} — st03 simplified core", fontsize=11)
        ax.set_ylabel("best test acc (%)")
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend(fontsize=8, loc="lower right")
    for j in range(len(tasks), 4):
        axes[j].axis("off")
    fig.suptitle("CTM Paper — st03: How small can the 'core' get? "
                 "(synapse_depth + memory_hidden_dims)", fontsize=13,
                 fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    path = out_dir / "fig9_simplified_core.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig9] -> {path}")


def _parse_frac(sweep, prefix):
    """sparsity0.25 -> 0.25, halt0.6 -> 0.6. Returns None if no match."""
    m = re.match(re.escape(prefix) + r"([\d.]+)$", sweep)
    return float(m.group(1)) if m else None


def _fig_numeric_sweep(df_complete, out_dir, stage, prefix, fig_id, title,
                       xlabel, path_name, xscale_log=False,
                       custom_task_order=None):
    """Shared helper for st06/st08-style numeric sweeps: errorbar vs x-axis.

    Annotates with the bugfix warning if the stage was affected by the
    log_dir overwrite bug.
    """
    sub = df_complete[df_complete["stage"] == stage].copy()
    sub["x"] = sub["sweep"].apply(lambda s: _parse_frac(s, prefix))
    sub = sub.dropna(subset=["x"])
    if sub.empty:
        print(f"[{fig_id}] no {stage} data")
        return
    order = custom_task_order or TASK_ORDER
    tasks = [t for t in order if t in sub["task"].unique()]
    fig, ax = plt.subplots(figsize=(9, 6.5))
    for task in tasks:
        grp = sub[sub["task"] == task]
        agg = grp.groupby("x").agg(
            mean=("best_test_acc", "mean"),
            std=("best_test_acc", "std"),
            n=("best_test_acc", "count"),
        ).reset_index().sort_values("x")
        x = agg["x"].values
        y = agg["mean"].values * 100
        yerr = agg["std"].fillna(0).values * 100
        ax.errorbar(x, y, yerr=yerr, marker="o", capsize=5, linewidth=2,
                    markersize=9, color=TASK_COLORS.get(task, "gray"),
                    label=task)
        for xi, yi, n in zip(x, y, agg["n"].values):
            tag = f"n={int(n)}" + ("!" if n < 2 else "")
            ax.annotate(tag, (xi, yi), textcoords="offset points",
                        xytext=(8, -2), fontsize=7,
                        color=TASK_COLORS.get(task, "gray"))
    # baseline lines (light, one per task) for context
    for task in tasks:
        if task in BASELINE_PAPER:
            ax.axhline(BASELINE_PAPER[task] * 100, color=TASK_COLORS[task],
                       linestyle=":", alpha=0.35, linewidth=1)
    if xscale_log:
        ax.set_xscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("best test acc (%)")
    title_str = title
    if stage in _BUGFIX_NOTE:
        title_str += f"\n⚠ {_BUGFIX_NOTE[stage]}"
    ax.set_title(title_str, fontsize=11, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=9)
    fig.tight_layout()
    path = out_dir / path_name
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[{fig_id}] -> {path}")


def fig_tick_halt(df_complete, out_dir):
    """st06 tick-halt threshold sweep (0.3 / 0.6 / 0.9).

    WARNING: compute_weight dimension (cw=0 / 0.001) was collapsed by the
    log_dir bug, so each point is one representative cw variant. Threshold
    trend is still meaningful.
    """
    _fig_numeric_sweep(
        df_complete, out_dir,
        stage="st06", prefix="halt", fig_id="fig11",
        title="st06: tick early-exit / halt threshold sweep",
        xlabel="halt threshold (confidence cutoff)",
        path_name="fig11_tick_halt.png",
    )


def fig_cell_sparsity(df_complete, out_dir):
    """st08 top-k neuron sparsity sweep (0.25 / 0.5 / 0.75). Clean data."""
    _fig_numeric_sweep(
        df_complete, out_dir,
        stage="st08", prefix="sparsity", fig_id="fig10",
        title="st08: top-k cell sparsity sweep",
        xlabel="top-k fraction of neurons activated per tick",
        path_name="fig10_sparsity.png",
    )


def fig_draft_revise(df_complete, out_dir, min_iter):
    """st10 draft-and-revise: single bar per task vs baseline (2x2 grid).

    WARNING: revise_weight (0.1/0.2) and corrupt_prob (0.15/0.30) were
    collapsed by the log_dir bug, so each bar is one representative variant.
    """
    sub = df_complete[df_complete["stage"] == "st10"].copy()
    if sub.empty:
        print("[fig12] no st10 data")
        return
    agg = agg_sweep(sub)
    tasks = [t for t in TASK_ORDER if t in agg["task"].unique()]
    nrows = (len(tasks) + 1) // 2
    fig, axes = plt.subplots(nrows, 2, figsize=(12, 4.5 * nrows), squeeze=False)
    axes = axes.flatten()
    for i, task in enumerate(tasks):
        ax = axes[i]
        a = agg[agg["task"] == task].sort_values("sweep")
        labels = [s.replace("revise", "draft-revise") for s in a["sweep"]]
        means = a["best_mean"].values * 100
        stds = a["best_std"].values * 100
        x = np.arange(len(labels))
        ax.bar(x, means, yerr=stds, capsize=4, color=plt.cm.Set2(0.3),
               alpha=0.85, edgecolor="black", linewidth=0.5)
        for xi, yi, n in zip(x, means, a["n_seeds"].values):
            tag = f"{yi:.1f}%\n(n={int(n)})" + ("!" if n < 2 else "")
            ax.annotate(tag, (xi, yi), textcoords="offset points",
                        xytext=(0, 6), ha="center", fontsize=8)
        if task in BASELINE_PAPER:
            ax.axhline(BASELINE_PAPER[task] * 100, color="gray", linestyle="--",
                       alpha=0.7, label="st00 paper baseline")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_title(f"{task} — st10 draft-revise", fontsize=11)
        ax.set_ylabel("best test acc (%)")
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend(fontsize=8, loc="lower right")
    for j in range(len(tasks), len(axes)):
        axes[j].axis("off")
    suptitle = "CTM Paper — st10: draft-and-revise (draft pass + noisy revision)"
    if "st10" in _BUGFIX_NOTE:
        suptitle += f"\n⚠ {_BUGFIX_NOTE['st10']}"
    fig.suptitle(suptitle, fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    path = out_dir / "fig12_draft_revise.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig12] -> {path}")


# ============================================================================
# New figures: fig13-20
# ============================================================================

# ---- fig13: ideas heatmap (task × idea delta vs baseline) --------------------

# Representative sweep per idea-stage (the "default" or "best" variant)
IDEA_SWEEPS = {
    "JEPA":          ("st04", "jepa_w0.1"),
    "halt":          ("st06", "halt0.6"),
    "sparsity":      ("st08", "sparsity0.5"),
    "reflex":        ("st09", "reflex"),
    "revise":        ("st10", "revise"),
    "EMA":           ("st12", "ema"),
    "JEPA+halt":     ("st13", "jepa_halt"),
}
IDEA_ORDER = list(IDEA_SWEEPS.keys())
ALL_TASKS_HEATMAP = ["cifar10", "mazes", "parity", "qamnist", "sort"]


def _idea_acc_lookup(df_complete, task, stage, sweep):
    """Mean best_test_acc for a given (task, stage, sweep), or None."""
    sub = df_complete[
        (df_complete["task"] == task) &
        (df_complete["stage"] == stage) &
        (df_complete["sweep"] == sweep)
    ]
    if sub.empty:
        return None, 0
    return sub["best_test_acc"].mean(), len(sub)


def fig_ideas_heatmap(df_complete, out_dir):
    """Task × idea heatmap: best_test_acc (%) and delta vs paper baseline."""
    # get baselines
    baselines = {}
    for task in ALL_TASKS_HEATMAP:
        val, _ = _idea_acc_lookup(df_complete, task, "st00", "paper")
        baselines[task] = val

    # build matrices
    acc_mat = np.full((len(ALL_TASKS_HEATMAP), len(IDEA_ORDER) + 1), np.nan)
    n_seeds_mat = np.zeros_like(acc_mat, dtype=int)
    col_labels = ["baseline"] + IDEA_ORDER

    for i, task in enumerate(ALL_TASKS_HEATMAP):
        # baseline column
        if baselines[task] is not None:
            acc_mat[i, 0] = baselines[task]
            n_seeds_mat[i, 0] = 1
        # idea columns
        for j, idea in enumerate(IDEA_ORDER):
            stage, sweep = IDEA_SWEEPS[idea]
            val, n = _idea_acc_lookup(df_complete, task, stage, sweep)
            if val is not None:
                acc_mat[i, j + 1] = val
                n_seeds_mat[i, j + 1] = n

    baselines_array = np.array(
        [baselines.get(t, np.nan) for t in ALL_TASKS_HEATMAP]
    ).reshape(-1, 1)
    delta_mat = acc_mat - baselines_array

    fig, axes = plt.subplots(1, 2, figsize=(18, 5.5),
                             gridspec_kw={"width_ratios": [1, 1]})

    # --- Panel A: absolute acc heatmap ---
    ax = axes[0]
    mat_pct = acc_mat * 100
    im = ax.imshow(mat_pct, cmap="RdYlGn", aspect="auto", vmin=0, vmax=100)
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=35, ha="right", fontsize=9)
    ax.set_yticks(range(len(ALL_TASKS_HEATMAP)))
    ax.set_yticklabels(ALL_TASKS_HEATMAP, fontsize=10)
    for i in range(mat_pct.shape[0]):
        for j in range(mat_pct.shape[1]):
            if np.isnan(mat_pct[i, j]):
                ax.text(j, i, "—", ha="center", va="center", fontsize=9, color="gray")
            else:
                n = n_seeds_mat[i, j]
                color = "white" if mat_pct[i, j] < 40 else "black"
                tag = f"{mat_pct[i, j]:.1f}%"
                if j > 0 and n > 0:
                    tag += f"\n(n={n})"
                ax.text(j, i, tag, ha="center", va="center", fontsize=7.5, color=color)
    ax.set_title("(a) Best test accuracy (%)", fontsize=12, fontweight="bold")
    plt.colorbar(im, ax=ax, shrink=0.8, label="acc (%)")

    # --- Panel B: delta vs baseline ---
    ax = axes[1]
    # mask baseline column (delta = 0)
    delta_display = delta_mat[:, 1:] * 100  # exclude baseline col
    vmax_delta = max(abs(np.nanmin(delta_display)), abs(np.nanmax(delta_display)), 10)
    im2 = ax.imshow(delta_display, cmap="RdBu_r", aspect="auto",
                    vmin=-vmax_delta, vmax=vmax_delta)
    ax.set_xticks(range(len(IDEA_ORDER)))
    ax.set_xticklabels(IDEA_ORDER, rotation=35, ha="right", fontsize=9)
    ax.set_yticks(range(len(ALL_TASKS_HEATMAP)))
    ax.set_yticklabels(ALL_TASKS_HEATMAP, fontsize=10)
    for i in range(delta_display.shape[0]):
        for j in range(delta_display.shape[1]):
            if np.isnan(delta_display[i, j]):
                ax.text(j, i, "—", ha="center", va="center", fontsize=9, color="gray")
            else:
                color = "white" if abs(delta_display[i, j]) > vmax_delta * 0.6 else "black"
                ax.text(j, i, f"{delta_display[i, j]:+.1f}", ha="center", va="center",
                        fontsize=9, color=color)
    ax.set_title("(b) Delta vs paper baseline (pp)", fontsize=12, fontweight="bold")
    plt.colorbar(im2, ax=ax, shrink=0.8, label="delta (pp)")

    fig.suptitle("What helps what? CTM ideas × tasks overview",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    path = out_dir / "fig13_ideas_heatmap.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig13] -> {path}")


# ---- fig14: JEPA × simplified core interaction (st05) -----------------------

def fig_jepa_simplified(df_complete, out_dir):
    """st05: JEPA combined with simplified core (sd2_mh2 / tick2) vs components."""
    sub = df_complete[df_complete["stage"].isin(["st03", "st04", "st05"])].copy()

    # define conditions: (label, stage, sweep)
    conditions = [
        ("baseline\n(st00 paper)", "st00", "paper"),
        ("JEPA\n(st04 w0.1)",     "st04", "jepa_w0.1"),
        ("simplified\n(st03 sd2_mh2)", "st03", "sd2_mh2"),
        ("JEPA+simplified\n(st05 jepa_sd2_mh2)", "st05", "jepa_sd2_mh2"),
        ("JEPA+tick2\n(st05 jepa_tick2)", "st05", "jepa_tick2"),
        ("JEPA+tick2+simplified\n(st05)", "st05", "jepa_tick2_sd2_mh2"),
    ]
    # st00 is not in sub (filter), add separately
    st00 = df_complete[df_complete["stage"] == "st00"]

    tasks = [t for t in ["cifar10", "parity", "sort"]
             if t in sub["task"].unique()]
    n_tasks = len(tasks)
    fig, axes = plt.subplots(1, n_tasks, figsize=(5.5 * n_tasks, 6), squeeze=False)

    for idx, task in enumerate(tasks):
        ax = axes[0][idx]
        labels, means, stds, ns = [], [], [], []
        for label, stage, sweep in conditions:
            src = st00 if stage == "st00" else sub
            grp = src[(src["task"] == task) & (src["stage"] == stage) &
                      (src["sweep"] == sweep)]
            if grp.empty:
                labels.append(label)
                means.append(0)
                stds.append(0)
                ns.append(0)
            else:
                labels.append(label)
                means.append(grp["best_test_acc"].mean() * 100)
                stds.append(float(grp["best_test_acc"].std(ddof=1)) * 100 if len(grp) > 1 else 0)
                ns.append(len(grp))

        x = np.arange(len(labels))
        colors = plt.cm.Set2(np.linspace(0, 0.8, len(labels)))
        bars = ax.bar(x, means, yerr=stds, capsize=4, color=colors,
                      alpha=0.85, edgecolor="black", linewidth=0.5)
        for xi, yi, n in zip(x, means, ns):
            tag = f"{yi:.1f}%" if n > 0 else "N/A"
            if n > 0:
                tag += f"\n(n={n})"
            ax.annotate(tag, (xi, yi), textcoords="offset points",
                        xytext=(0, 6), ha="center", fontsize=7.5)
        # baseline line
        bl = st00[(st00["task"] == task) & (st00["sweep"] == "paper")]
        if not bl.empty:
            ax.axhline(bl["best_test_acc"].values[0] * 100, color="gray",
                       linestyle="--", alpha=0.6, label="paper baseline")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=7, rotation=20)
        ax.set_title(f"{task}", fontsize=12, fontweight="bold")
        ax.set_ylabel("best test acc (%)" if idx == 0 else "")
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend(fontsize=8)

    fig.suptitle("st05: Does JEPA compound with structural simplification?",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    path = out_dir / "fig14_jepa_simplified.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig14] -> {path}")


# ---- fig15: sort failure mode across all stages -----------------------------

def fig_sort_failure(df_complete, out_dir):
    """Sort task accuracy across ALL stages — shows the dramatic collapse pattern."""
    sort_df = df_complete[df_complete["task"] == "sort"].copy()
    if sort_df.empty:
        print("[fig15] no sort data")
        return

    # aggregate per (stage, sweep)
    agg = sort_df.groupby(["stage", "sweep"]).agg(
        mean=("best_test_acc", "mean"),
        std=("best_test_acc", "std"),
        n=("best_test_acc", "count"),
    ).reset_index()

    # stage order
    stage_order = sorted(agg["stage"].unique(),
                         key=lambda s: (int(s[2:])))

    fig, ax = plt.subplots(figsize=(16, 7))

    # jittered strip plot
    np.random.seed(42)
    for i, stage in enumerate(stage_order):
        sub = agg[agg["stage"] == stage].sort_values("sweep")
        for _, row in sub.iterrows():
            x_jitter = i + np.random.uniform(-0.15, 0.15)
            y = row["mean"] * 100
            color = "#2ca02c" if y > 50 else "#d62728"
            ax.scatter(x_jitter, y, s=80, color=color, alpha=0.8,
                       edgecolors="black", linewidths=0.5, zorder=3)
            # label sweep name
            offset_y = 3 if y > 50 else -5
            ax.annotate(f"{row['sweep']}\n({y:.1f}%)", (x_jitter, y),
                        textcoords="offset points", xytext=(0, offset_y),
                        ha="center", fontsize=5.5, color=color, alpha=0.8)

    # baseline line
    bl = sort_df[(sort_df["stage"] == "st00") & (sort_df["sweep"] == "paper")]
    if not bl.empty:
        ax.axhline(bl["best_test_acc"].values[0] * 100, color="gray",
                   linestyle="--", alpha=0.6, linewidth=1.5,
                   label=f"paper baseline ({bl['best_test_acc'].values[0]*100:.1f}%)")

    # collapse zone
    ax.axhspan(0, 5, color="red", alpha=0.06, zorder=0)
    ax.text(len(stage_order) - 0.5, 3, "collapse zone (<5%)",
            fontsize=9, color="red", alpha=0.6, ha="right")

    ax.set_xticks(range(len(stage_order)))
    ax.set_xticklabels(stage_order, rotation=45, fontsize=8)
    ax.set_xlabel("stage", fontsize=11)
    ax.set_ylabel("best test acc (%)", fontsize=11)
    ax.set_title("Sort task: dramatic failure across most CTM ideas\n"
                 "(green = working >50%, red = collapsed <5%)",
                 fontsize=13, fontweight="bold")
    ax.set_ylim(-5, 105)
    ax.grid(True, axis="y", alpha=0.2)
    ax.legend(fontsize=9, loc="upper left")

    fig.tight_layout()
    path = out_dir / "fig15_sort_failure.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig15] -> {path}")


# ---- fig16: reflex + EMA bar charts -----------------------------------------

def fig_reflex_ema(df_complete, out_dir):
    """st09 (reflex) and st12 (EMA): side-by-side bar charts vs baseline."""
    st00 = df_complete[df_complete["stage"] == "st00"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    for panel_idx, (ax, (stage, idea_name)) in enumerate(zip(
            axes, [("st09", "reflex"), ("st12", "EMA")])):
        stage, idea_name = stage, idea_name
        sub = df_complete[df_complete["stage"] == stage]
        if sub.empty:
            ax.text(0.5, 0.5, f"no {stage} data", ha="center", va="center",
                    transform=ax.transAxes)
            continue

        tasks_present = [t for t in TASK_ORDER + ["qamnist"]
                         if t in sub["task"].unique()]
        labels, means, stds, ns, bl_vals = [], [], [], [], []
        for task in tasks_present:
            grp = sub[sub["task"] == task]
            labels.append(task)
            means.append(grp["best_test_acc"].mean() * 100)
            stds.append(float(grp["best_test_acc"].std(ddof=1)) if not pd.isna(grp["best_test_acc"].std(ddof=1)) else 0.0 * 100 if len(grp) > 1 else 0)
            ns.append(len(grp))
            bl = st00[(st00["task"] == task) & (st00["sweep"] == "paper")]
            bl_vals.append(bl["best_test_acc"].values[0] * 100 if not bl.empty else 0)

        x = np.arange(len(labels))
        width = 0.35
        colors = [TASK_COLORS.get(t, "gray") for t in labels]
        bars1 = ax.bar(x - width/2, means, width, yerr=stds, capsize=4,
                       color=colors, alpha=0.85, edgecolor="black", linewidth=0.5,
                       label=idea_name)
        bars2 = ax.bar(x + width/2, bl_vals, width, color="gray", alpha=0.4,
                       edgecolor="black", linewidth=0.5, label="paper baseline")
        for xi, yi, n in zip(x - width/2, means, ns):
            ax.annotate(f"{yi:.1f}%\n(n={n})", (xi, yi),
                        textcoords="offset points", xytext=(0, 5),
                        ha="center", fontsize=7.5)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=10)
        ax.set_title(f"{stage}: {idea_name} vs baseline", fontsize=12,
                     fontweight="bold")
        ax.set_ylabel("best test acc (%)")
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend(fontsize=9)

    fig.suptitle("Auxiliary training mechanisms (thin data — 1-3 seeds)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    path = out_dir / "fig16_reflex_ema.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig16] -> {path}")


# ---- fig17: per-task idea comparison ----------------------------------------

def fig_ideas_per_task(df_complete, out_dir):
    """Per-task horizontal bar: each idea vs paper baseline."""
    st00 = df_complete[df_complete["stage"] == "st00"]
    tasks = [t for t in TASK_ORDER + ["qamnist"]
             if t in df_complete["task"].unique()]
    n_tasks = len(tasks)
    nrows = (n_tasks + 1) // 2
    fig, axes = plt.subplots(nrows, 2, figsize=(14, 4.5 * nrows), squeeze=False)
    axes = axes.flatten()

    for idx, task in enumerate(tasks):
        ax = axes[idx]
        bl = st00[(st00["task"] == task) & (st00["sweep"] == "paper")]
        bl_val = bl["best_test_acc"].values[0] * 100 if not bl.empty else 0

        idea_labels, idea_vals, idea_errs, idea_colors = [], [], [], []
        for idea_name, (stage, sweep) in IDEA_SWEEPS.items():
            grp = df_complete[(df_complete["stage"] == stage) &
                              (df_complete["sweep"] == sweep) &
                              (df_complete["task"] == task)]
            if grp.empty:
                continue
            idea_labels.append(idea_name)
            idea_vals.append(grp["best_test_acc"].mean() * 100)
            idea_errs.append(float(grp["best_test_acc"].std(ddof=1)) if not pd.isna(grp["best_test_acc"].std(ddof=1)) else 0.0 * 100
                            if len(grp) > 1 else 0)
            delta = grp["best_test_acc"].mean() - (bl_val / 100)
            idea_colors.append("#2ca02c" if delta > 0 else "#d62728")

        # add baseline bar
        all_labels = ["paper baseline"] + idea_labels
        all_vals = [bl_val] + idea_vals
        all_errs = [0] + idea_errs
        all_colors = ["gray"] + idea_colors

        y = np.arange(len(all_labels))
        ax.barh(y, all_vals, xerr=all_errs, capsize=3, color=all_colors,
                alpha=0.8, edgecolor="black", linewidth=0.5)
        for yi, vi in zip(y, all_vals):
            ax.annotate(f"{vi:.1f}%", (vi, yi), textcoords="offset points",
                        xytext=(5, 0), va="center", fontsize=8)
        ax.set_yticks(y)
        ax.set_yticklabels(all_labels, fontsize=9)
        ax.set_title(task, fontsize=12, fontweight="bold")
        ax.set_xlabel("best test acc (%)")
        ax.grid(True, axis="x", alpha=0.3)
        ax.invert_yaxis()

    for j in range(n_tasks, len(axes)):
        axes[j].axis("off")

    fig.suptitle("Per-task idea comparison vs paper baseline",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    path = out_dir / "fig17_ideas_per_task.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig17] -> {path}")


# ---- fig18: combo ideas (jepa+halt, halt+sparsity) --------------------------

def fig_combo_ideas(df_complete, out_dir):
    """st13 (jepa+halt) and st15 (halt+sparsity) vs individual components."""
    st00 = df_complete[df_complete["stage"] == "st00"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # --- Panel A: JEPA + halt ---
    ax = axes[0]
    combo_defs_a = [
        ("baseline",   "st00", "paper"),
        ("JEPA",       "st04", "jepa_w0.1"),
        ("halt",       "st06", "halt0.6"),
        ("JEPA+halt",  "st13", "jepa_halt"),
    ]
    tasks_a = [t for t in TASK_ORDER + ["qamnist"]
               if t in df_complete[df_complete["stage"] == "st13"]["task"].unique()]
    x = np.arange(len(tasks_a))
    width = 0.18
    for k, (label, stage, sweep) in enumerate(combo_defs_a):
        vals, errs = [], []
        for task in tasks_a:
            grp = df_complete[(df_complete["stage"] == stage) &
                              (df_complete["sweep"] == sweep) &
                              (df_complete["task"] == task)]
            if grp.empty:
                vals.append(0)
                errs.append(0)
            else:
                vals.append(grp["best_test_acc"].mean() * 100)
                errs.append(float(grp["best_test_acc"].std(ddof=1)) if not pd.isna(grp["best_test_acc"].std(ddof=1)) else 0.0 * 100
                           if len(grp) > 1 else 0)
        ax.bar(x + k * width, vals, width, yerr=errs, capsize=3,
               label=label, alpha=0.85, edgecolor="black", linewidth=0.4)
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(tasks_a, fontsize=10)
    ax.set_title("(a) JEPA + halt (st13) vs components", fontsize=11,
                 fontweight="bold")
    ax.set_ylabel("best test acc (%)")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize=8, loc="upper right")

    # --- Panel B: halt + sparsity ---
    ax = axes[1]
    combo_defs_b = [
        ("baseline",       "st00", "paper"),
        ("halt",           "st06", "halt0.6"),
        ("sparsity",       "st08", "sparsity0.5"),
        ("halt+sparsity",  "st15", "halt_sparsity"),
    ]
    tasks_b = [t for t in TASK_ORDER
               if t in df_complete[df_complete["stage"] == "st15"]["task"].unique()]
    x = np.arange(len(tasks_b))
    for k, (label, stage, sweep) in enumerate(combo_defs_b):
        vals, errs = [], []
        for task in tasks_b:
            grp = df_complete[(df_complete["stage"] == stage) &
                              (df_complete["sweep"] == sweep) &
                              (df_complete["task"] == task)]
            if grp.empty:
                vals.append(0)
                errs.append(0)
            else:
                vals.append(grp["best_test_acc"].mean() * 100)
                errs.append(float(grp["best_test_acc"].std(ddof=1)) if not pd.isna(grp["best_test_acc"].std(ddof=1)) else 0.0 * 100
                           if len(grp) > 1 else 0)
        ax.bar(x + k * width, vals, width, yerr=errs, capsize=3,
               label=label, alpha=0.85, edgecolor="black", linewidth=0.4)
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(tasks_b, fontsize=10)
    ax.set_title("(b) halt + sparsity (st15) vs components", fontsize=11,
                 fontweight="bold")
    ax.set_ylabel("best test acc (%)")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize=8, loc="upper right")

    fig.suptitle("Combining CTM ideas: synergy or interference?",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    path = out_dir / "fig18_combo_ideas.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig18] -> {path}")


# ---- fig19: most-certain tick gain boxplot ----------------------------------

def fig_mc_gain(df_complete, out_dir):
    """Boxplot of (best_test_acc_mc - best_test_acc) per task."""
    sub = df_complete.dropna(subset=["best_test_acc_mc"]).copy()
    sub = sub[(sub["best_test_acc_mc"] > 0) & (sub["best_test_acc"] > 0)]
    sub = sub[sub["stage"] != "st00"]  # exclude baselines
    if sub.empty:
        print("[fig19] no MC data")
        return

    sub["mc_delta"] = (sub["best_test_acc_mc"] - sub["best_test_acc"]) * 100
    tasks = [t for t in TASK_ORDER + ["qamnist"]
             if t in sub["task"].unique()]

    fig, ax = plt.subplots(figsize=(10, 6))
    data = [sub[sub["task"] == t]["mc_delta"].values for t in tasks]
    bp = ax.boxplot(data, labels=tasks, patch_artist=True, widths=0.5,
                    showmeans=True, meanprops=dict(marker="D", markerfacecolor="red",
                                                   markersize=7))
    for patch, task in zip(bp["boxes"], tasks):
        patch.set_facecolor(TASK_COLORS.get(task, "gray"))
        patch.set_alpha(0.6)
    ax.axhline(0, color="black", linestyle="--", alpha=0.5, label="no gain")

    # annotate means
    for i, task in enumerate(tasks):
        vals = sub[sub["task"] == task]["mc_delta"]
        ax.annotate(f"mean={vals.mean():+.1f}%\nn={len(vals)}",
                    (i + 1, vals.mean()),
                    textcoords="offset points", xytext=(10, 0),
                    fontsize=8, color="red")

    ax.set_ylabel("MC gain (pp): most-certain tick acc − last-tick acc",
                  fontsize=11)
    ax.set_title("Most-certain tick selection: large gains on qamnist & cifar10,\n"
                 "minimal on mazes (which already converges at the final tick)",
                 fontsize=12, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    path = out_dir / "fig19_mc_gain.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig19] -> {path}")


# ---- fig20: train vs test gap ------------------------------------------------

def fig_train_test_gap(df_complete, out_dir):
    """Scatter: best_train_acc vs best_test_acc, colored by task."""
    sub = df_complete.dropna(subset=["best_train_acc", "best_test_acc"]).copy()
    sub = sub[(sub["best_train_acc"] > 0) & (sub["best_test_acc"] > 0)]
    if sub.empty:
        print("[fig20] no train/test data")
        return

    tasks = [t for t in TASK_ORDER + ["qamnist"]
             if t in sub["task"].unique()]

    fig, ax = plt.subplots(figsize=(9, 8))
    for task in tasks:
        grp = sub[sub["task"] == task]
        ax.scatter(grp["best_train_acc"] * 100, grp["best_test_acc"] * 100,
                   label=task, color=TASK_COLORS.get(task, "gray"),
                   alpha=0.5, s=35, edgecolors="white", linewidths=0.3)

    # diagonal
    lim = [0, 100]
    ax.plot(lim, lim, "k--", alpha=0.4, label="y=x (no generalisation gap)")
    # gap zones
    ax.fill_between(lim, [v - 20 for v in lim], lim, alpha=0.05, color="red",
                    label="20pp gap zone")

    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("best train acc (%)", fontsize=11)
    ax.set_ylabel("best test acc (%)", fontsize=11)
    ax.set_title("Train–test gap across all experiments\n"
                 "(points far below diagonal = poor generalisation)",
                 fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc="lower right")
    fig.tight_layout()
    path = out_dir / "fig20_train_test_gap.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig20] -> {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default=str(DEFAULT_DATA))
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT))
    ap.add_argument("--min-iter", type=int, default=50000,
                    help="drop seeds with final_iter below this (default 50000)")
    cli = ap.parse_args()

    data_dir = Path(cli.data_dir)
    out_dir = Path(cli.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df, curves = load_data(data_dir)
    print(f"Loaded summary.csv: {len(df)} rows")
    print(f"Loaded curves.json: {len(curves)} runs")

    df_complete = filter_complete(df, cli.min_iter)
    n_dropped = len(df) - len(df_complete)
    print(f"Filter: dropped {n_dropped} seeds with final_iter < {cli.min_iter}")
    print(f"  → {len(df_complete)} complete seeds remain")

    # report stage coverage after filter
    cov = df_complete.groupby("stage").size()
    print("Complete seeds per stage:")
    print(cov.to_string())

    fig_tick_sweep(df_complete, out_dir, cli.min_iter)
    fig_capacity_sweep(df_complete, out_dir, cli.min_iter)
    fig_mc_vs_final(df_complete, out_dir)
    fig_convergence(df_complete, curves, out_dir, cli.min_iter)
    fig_compute_cost(df_complete, out_dir, cli.min_iter)
    fig_jepa_weight(df_complete, out_dir)
    fig_jepa_variants(df_complete, out_dir)
    # new figures: ideas stages (st00/st03/st06/st08/st10)
    fig_baselines(df_complete, out_dir)
    fig_simplified_core(df_complete, out_dir, cli.min_iter)
    fig_cell_sparsity(df_complete, out_dir)
    fig_tick_halt(df_complete, out_dir)
    fig_draft_revise(df_complete, out_dir, cli.min_iter)
    # new figures: fig13-20
    fig_ideas_heatmap(df_complete, out_dir)
    fig_jepa_simplified(df_complete, out_dir)
    fig_sort_failure(df_complete, out_dir)
    fig_reflex_ema(df_complete, out_dir)
    fig_ideas_per_task(df_complete, out_dir)
    fig_combo_ideas(df_complete, out_dir)
    fig_mc_gain(df_complete, out_dir)
    fig_train_test_gap(df_complete, out_dir)

    print(f"\nAll figures saved to: {out_dir}")


if __name__ == "__main__":
    main()
