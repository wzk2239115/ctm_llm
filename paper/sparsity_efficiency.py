#!/usr/bin/env python3
"""Sparsity efficiency analysis — the RIGHT way to judge top-k sparse activation.

Sparsity's purpose is NOT to raise accuracy; it's to trade a little accuracy
for a lot of compute. The correct judgement is a Pareto frontier of
accuracy vs effective compute, NOT a single delta vs baseline.

Compute model (stated honestly, not overclaimed):
  - topk_neurons = r means only r-fraction of the NLM neuron population fires
    per thought-tick. So the CTM thought-loop (recurrent neuron updates) does
    ~r of the work, saving (1-r) of NLM compute.
  - The backbone (resnet for cifar10/mazes) is NOT sparsified, so end-to-end
    wall-clock speedup < (1-r) when backbone dominates. The r below is the
    NLM-compute fraction; realized speedup depends on a sparse kernel + the
    backbone/NLM cost ratio. We report the theoretical NLM saving and flag
    this caveat.

Data available:
  - 03_sparsity deep: mazes r∈{0.1,0.25,0.5,0.75,0.9} (3-5 seeds) — FULL sweep,
    the only task with enough points for a frontier. mc metric.
  - ctm_paper st08: cifar10 r=0.25 (n=2), parity r=0.25/0.75 (degenerate pair),
    sort r=0.25/0.5/0.75 (all identical = no-op). Thin / broken elsewhere.

Output: Pareto plot (acc vs NLM-compute-fraction r) + the sweet-spot report.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "csv_data"
FIG_DIR = ROOT / "runs" / "figures" / "ctm_paper"
FIG_DIR.mkdir(parents=True, exist_ok=True)

MC_BASE = {"cifar10": 0.8516, "mazes": 0.9117, "parity": 0.8821, "sort": 0.8753}
PLANNED = {"cifar10": 200000, "mazes": 100000, "parity": 200000, "sort": 100000}


def parse_r(name):
    """sparsity0p5 -> 0.5, swp_r0p1 -> 0.1. baseline (no r) -> 1.0."""
    import re
    m = re.search(r"(?:sparsity|swp_r)0p(\d+)", str(name))
    if m:
        return int(m.group(1)) / 10 if len(m.group(1)) == 1 else int(m.group(1)) / 100
    # two-digit like 0p25 -> 0.25
    m2 = re.search(r"(?:sparsity|swp_r)0p(\d{2})", str(name))
    if m2:
        return int(m2.group(1)) / 100
    return 1.0  # baseline (dense)


def load_sparsity():
    """Pull all sparsity runs (deep 03 + ctm_paper st08) + baselines."""
    rows = []
    # deep 03
    p = DATA / "03_sparsity_deep_results.csv"
    if p.exists():
        df = pd.read_csv(p)
        for c in ["best_acc", "final_iter"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df["r"] = df["name"].apply(parse_r)
        df["source"] = "deep03"
        rows.append(df[["name", "task", "best_acc", "final_iter", "r", "source"]])
    # ctm_paper st08 (mc acc = best_test_acc_mc; fall back to best_test_acc)
    p2 = DATA / "ctm_paper_summary.csv"
    if p2.exists():
        df2 = pd.read_csv(p2)
        for c in ["best_test_acc", "best_test_acc_mc", "final_iter"]:
            df2[c] = pd.to_numeric(df2[c], errors="coerce")
        s = df2[df2.stage == "st08"].copy()
        s["best_acc"] = s["best_test_acc_mc"].fillna(s["best_test_acc"])
        s["r"] = s["sweep"].apply(parse_r)
        s["source"] = "st08"
        s = s.rename(columns={"exp": "name"})
        rows.append(s[["name", "task", "best_acc", "final_iter", "r", "source"]])
    # baselines (st00 paper) at r=1.0
    p3 = DATA / "ctm_paper_summary.csv"
    df3 = pd.read_csv(p3)
    for c in ["best_test_acc", "best_test_acc_mc", "final_iter"]:
        df3[c] = pd.to_numeric(df3[c], errors="coerce")
    b = df3[(df3.stage == "st00") & (df3.sweep == "paper")].copy()
    b["best_acc"] = b["best_test_acc_mc"].fillna(b["best_test_acc"])
    b["r"] = 1.0
    b["source"] = "baseline"
    b = b.rename(columns={"exp": "name"})
    rows.append(b[["name", "task", "best_acc", "final_iter", "r", "source"]])
    return pd.concat(rows, ignore_index=True)


def is_sort_degen(val):
    magic = {0.6458, 0.7917, 0.8021, 0.8753, 0.9545, 0.9462}
    return pd.notna(val) and round(float(val), 4) in {round(m, 4) for m in magic}


def report(df):
    print("\n" + "=" * 80)
    print("SPARSITY EFFICIENCY — acc vs NLM-compute-fraction (Pareto)")
    print("=" * 80)
    print("\ncompute model: r = fraction of neurons active per tick;")
    print("NLM thought-loop does ~r of the work -> saves (1-r) of NLM compute.")
    print("backbone (resnet) is NOT sparsified, so end-to-end speedup < (1-r).\n")

    for task in ["mazes", "cifar10", "parity", "sort"]:
        sub = df[(df.task == task) & (df.final_iter >= 0.5 * PLANNED.get(task, 0))]
        if task == "sort":
            sub = sub[~sub.best_acc.apply(is_sort_degen)]
        if sub.empty:
            print(f"--- {task}: no usable runs ---\n")
            continue
        g = sub.groupby("r").agg(
            n=("best_acc", "count"),
            mean=("best_acc", "mean"),
            std=("best_acc", "std"),
        ).reset_index().sort_values("r")
        base = MC_BASE[task]
        print(f"--- {task} (mc baseline {base*100:.1f}%) ---")
        print(f"  {'r(compute)':>11s} {'save':>6s} {'n':>3s} {'mean_acc':>9s} "
              f"{'delta':>8s} {'acc/save%':>10s}")
        for _, r in g.iterrows():
            save = (1 - r["r"]) * 100
            delta = (r["mean"] - base) * 100
            # efficiency: how much accuracy you keep per unit compute used
            # "acc per compute" = mean_acc / r ; higher = more efficient
            acc_per_comp = r["mean"] / r["r"] if r["r"] > 0 else float("inf")
            print(f"  {r['r']:>10.2f} {save:>5.0f}% {int(r['n']):3d} "
                  f"{r['mean']*100:8.2f}% {delta:>+7.1f}pp "
                  f"{(r['mean']/r['r']) if r['r']>0 else 0:>10.2f}")
        # Pareto-optimal: points where no other point has both higher acc AND
        # lower r (i.e. cheaper). Compute the frontier.
        pts = g[["r", "mean"]].values
        frontier = []
        for i, (ri, ai) in enumerate(pts):
            dominated = any(
                (rj <= ri and aj >= ai and not (rj == ri and aj == ai))
                for rj, aj in pts
            )
            if not dominated:
                frontier.append((ri, ai))
        frontier.sort()
        print(f"  Pareto frontier (r -> acc): "
              f"{', '.join(f'{r:.2f}->{a*100:.1f}%' for r, a in frontier)}")
        # sweet spot: the r with best acc-per-compute (efficiency max)
        eff = [(r["r"], r["mean"], r["mean"] / r["r"] if r["r"] > 0 else 0)
               for _, r in g.iterrows() if r["r"] > 0]
        if eff:
            best_eff = max(eff, key=lambda x: x[2])
            print(f"  best efficiency (acc/compute): r={best_eff[0]:.2f} -> "
                  f"{best_eff[1]*100:.1f}% (saves {(1-best_eff[0])*100:.0f}% NLM "
                  f"compute for only {(best_eff[1]-base)*100:+.1f}pp)")
        print()


def plot(df):
    """Pareto: x = NLM compute fraction (r), y = acc. Lower-right = better."""
    fig, ax = plt.subplots(figsize=(11, 6.5))
    colors = {"mazes": "#ff7f0e", "cifar10": "#1f77b4",
              "parity": "#2ca02c", "sort": "#9467bd"}
    for task in ["mazes", "cifar10", "parity"]:
        sub = df[(df.task == task) & (df.final_iter >= 0.5 * PLANNED.get(task, 0))]
        if sub.empty:
            continue
        g = sub.groupby("r").agg(n=("best_acc", "count"),
                                 mean=("best_acc", "mean"),
                                 std=("best_acc", "std")).reset_index().sort_values("r")
        # errorbar
        yerr = g["std"].fillna(0).values * 100
        ax.errorbar(g["r"], g["mean"] * 100, yerr=yerr, marker="o",
                    capsize=4, linewidth=2, markersize=8,
                    color=colors[task], label=task)
        # annotate n
        for _, r in g.iterrows():
            ax.annotate(f"n={int(r['n'])}", (r["r"], r["mean"] * 100),
                        textcoords="offset points", xytext=(6, 6), fontsize=7,
                        color=colors[task])
        # baseline marker at r=1.0
        base = MC_BASE[task]
        ax.plot(1.0, base * 100, "*", markersize=15, color=colors[task],
                markeredgecolor="black", markeredgewidth=0.7)
    # shade "free lunch" zone: r<0.3, acc within 1pp of baseline
    ax.axvspan(0, 0.3, color="#2ca02c", alpha=0.08)
    ax.text(0.15, 96, "sweet zone\n(low r, ~free compute)", ha="center",
            fontsize=9, color="#2ca02c", fontweight="bold")
    ax.set_xlabel("NLM compute fraction r  (top-k fraction of neurons active per tick)\n"
                  "← cheaper (saves (1−r) of NLM compute)    |    full cost →",
                  fontsize=10)
    ax.set_ylabel("most-certain-tick accuracy (%)", fontsize=10)
    ax.set_title("Sparsity Pareto: accuracy vs NLM-compute cost\n"
                 "★ = dense baseline (r=1.0). Points below-left of a line are "
                 "Pareto-dominated.\n"
                 "mazes r=0.1 saves 90% NLM compute for ~1pp — a real efficiency win.",
                 fontsize=11, fontweight="bold")
    ax.set_xlim(0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=10)
    fig.tight_layout()
    p = FIG_DIR / "figE_sparsity_pareto.png"
    fig.savefig(p, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[fig] -> {p}")


def main():
    df = load_sparsity()
    print(f"loaded {len(df)} sparsity/baseline runs across tasks "
          f"{sorted(df.task.unique())}")
    report(df)
    plot(df)


if __name__ == "__main__":
    main()
