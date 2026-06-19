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


def fig_jepa_weight(df_complete, out_dir):
    """st04 jepa_weight scan (0.1, 0.5, 1.0): one line per task."""
    sub = df_complete[df_complete["stage"] == "st04"].copy()
    sub["weight"] = sub["sweep"].apply(parse_jepa_weight)
    sub = sub.dropna(subset=["weight"])
    if sub.empty:
        print("[fig5] no st04 jepa_weight data")
        return

    tasks = [t for t in ["cifar10", "mazes", "qamnist", "sort", "parity"]
             if t in sub["task"].unique()]
    fig, ax = plt.subplots(figsize=(9, 6.5))
    for task in tasks:
        grp = sub[sub["task"] == task]
        agg = grp.groupby("weight").agg(
            mean=("best_test_acc", "mean"),
            std=("best_test_acc", "std"),
            n=("best_test_acc", "count"),
        ).reset_index().sort_values("weight")
        x = agg["weight"].values
        y = agg["mean"].values * 100
        yerr = agg["std"].fillna(0).values * 100
        ax.errorbar(x, y, yerr=yerr, marker="o", capsize=5, linewidth=2,
                    markersize=9, color=TASK_COLORS.get(task, "gray"),
                    label=f"{task}")
        for xi, yi, n in zip(x, y, agg["n"].values):
            ax.annotate(f"n={int(n)}", (xi, yi), textcoords="offset points",
                        xytext=(8, -2), fontsize=7,
                        color=TASK_COLORS.get(task, "gray"))
    ax.set_xscale("log")
    ax.set_xticks([0.1, 0.5, 1.0])
    ax.set_xticklabels(["0.1", "0.5", "1.0"])
    ax.set_xlabel("jepa_weight (auxiliary loss weight, log scale)")
    ax.set_ylabel("best test acc (%)")
    ax.set_title("st04: JEPA auxiliary loss weight sweep\n"
                 "(lower weight = less JEPA regularisation)",
                 fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=9)
    fig.tight_layout()
    path = out_dir / "fig5_jepa_weight.png"
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

    print(f"\nAll figures saved to: {out_dir}")


if __name__ == "__main__":
    main()
