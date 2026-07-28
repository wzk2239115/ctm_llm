#!/usr/bin/env python3
"""Repro analysis (0724) — 5-seed three-pillar verification + sparsity Pareto.

Reads paper_repro/csv_data/repro_summary_0724.csv (118 runs: baseline/jepa/revise
5-seed + sparsity 5-seed on mazes/cifar10/parity, sort 1-r).

Two outputs:
  1. console: three-pillar repro verdict (delta vs MATCHED 5-seed baseline,
     mc-caliber primary). Flags partial-trained / inert / low-seed.
  2. figure: sparsity Pareto (acc vs NLM-compute r) for mazes/cifar10/parity.

口径 (caliber) rules (from AGENTS.md, enforced here):
  - mc-vs-mc: idea delta uses best_test_acc_mc vs baseline best_test_acc_mc.
    sort has no mc -> uses best_test_acc (final-tick), flagged.
  - PARTIAL filter: final_iter < 0.5*planned -> excluded (negative delta
    from undertraining is not evidence). cifar10 sparsity0p75 ALL partial
    (~72k/200k) -> dropped. cifar10 sparsity0p5 has 1 partial seed -> that
    seed dropped, 4 kept.
  - sort sparsity: inert-suspected (sort ignores topk) + only 1 r -> flagged,
    not counted as a frontier point.
"""
import sys
from pathlib import Path
import csv
from statistics import mean, stdev

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "paper_repro" / "csv_data" / "repro_summary_0728.csv"
FIG_DIR = ROOT / "runs" / "figures" / "ctm_paper"
FIG_DIR.mkdir(parents=True, exist_ok=True)

PLANNED = {"cifar10": 200000, "mazes": 100000, "parity": 200000,
           "qamnist": 200000, "sort": 100000}


def f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def load():
    rows = []
    for r in csv.DictReader(open(CSV)):
        r["best"] = f(r["best_test_acc"])
        r["mc"] = f(r["best_test_acc_mc"])
        r["iter"] = int(r["final_iter"]) if r["final_iter"] else 0
        r["seed"] = int(r["seed"])
        r["r"] = _parse_r(r["sweep"])
        rows.append(r)
    return rows


def _parse_r(sweep):
    import re
    m = re.search(r"sparsity0p(\d{1,2})", str(sweep))
    if m:
        v = int(m.group(1))
        return v / 10 if v < 10 else v / 100
    return 1.0  # baseline / non-sparsity


def is_partial(row):
    return row["iter"] < 0.5 * PLANNED.get(row["task"], 0)


def stat(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return None, None, 0
    m = mean(vals)
    s = stdev(vals) if len(vals) > 1 else 0.0
    return m, s, len(vals)


def pillar_verdict(rows):
    print("=" * 78)
    print("THREE-PILLAR REPRO VERDICT (5-seed, mc-vs-mc vs matched baseline)")
    print("=" * 78)
    # baselines (full-iter only)
    base_mc, base_best = {}, {}
    for task in PLANNED:
        b = [r for r in rows if r["stage"] == "baseline" and r["task"] == task
             and not is_partial(r)]
        mcm = [r["mc"] for r in b if r["mc"] is not None]
        bm = [r["best"] for r in b if r["best"] is not None]
        if mcm:
            base_mc[task] = mean(mcm)
        if bm:
            base_best[task] = mean(bm)

    pillars = [("jepa", "JEPA"), ("revise", "Draft-Revise"), ("sparsity", "Sparsity")]
    for stage, name in pillars:
        print(f"\n--- {name} ---")
        subs = [r for r in rows if r["stage"] == stage]
        for task in sorted({r["task"] for r in subs}):
            full = [r for r in subs if r["task"] == task and not is_partial(r)]
            npartial = sum(1 for r in subs if r["task"] == task and is_partial(r))
            mcm, mcs, n = stat([r["mc"] for r in full])
            bm, bss, _ = stat([r["best"] for r in full])
            base = base_mc.get(task, base_best.get(task))
            if base is None:
                continue
            if mcm is not None:
                d = (mcm - base) * 100
                verdict = _verdict(d, mcs, n)
                print(f"  {task:8s} mc {mcm*100:6.2f}±{mcs*100:.2f} (n={n})  "
                      f"Δ={d:+.2f}pp vs base {base*100:.2f}  {verdict}")
            else:
                d = (bm - base) * 100
                print(f"  {task:8s} best {bm*100:6.2f}±{bss*100:.2f} (n={n})  "
                      f"Δ={d:+.2f}pp  [no mc — sort]")
            if npartial:
                print(f"          ⚠ {npartial} seed(s) PARTIAL (<50% iter) excluded")


def _verdict(d, sd, n):
    sd = sd * 100 if sd else 0
    if abs(d) < max(1.0, sd):
        return "NEUTRAL (within noise)"
    return "POSITIVE" if d > 0 else "NEGATIVE"


def sparsity_report(rows):
    print("\n" + "=" * 78)
    print("SPARSITY PARETO (mc acc vs NLM-compute fraction r, 5-seed)")
    print("=" * 78)
    print("r = top-k neuron fraction active per tick; saves (1-r) of NLM compute.")
    print("backbone (resnet) NOT sparsified -> wall-clock < (1-r). r=NLM-cost frac.\n")
    sp = [r for r in rows if r["stage"] == "sparsity"]
    for task in ["mazes", "cifar10", "parity", "sort"]:
        sub = [r for r in sp if r["task"] == task and not is_partial(r)]
        if not sub:
            print(f"--- {task}: no full-iter runs ---\n")
            continue
        base = None
        if task != "sort":
            bm = [r["mc"] for r in rows if r["stage"] == "baseline"
                  and r["task"] == task and r["mc"] is not None and not is_partial(r)]
            base = mean(bm) if bm else None
        else:
            bm = [r["best"] for r in rows if r["stage"] == "baseline"
                  and r["task"] == task and r["best"] is not None]
            base = mean(bm) if bm else None
        print(f"--- {task} (baseline {'mc' if task!='sort' else 'best'} "
              f"{base*100:.2f}%) ---")
        print(f"  {'r':>6s} {'save':>6s} {'n':>3s} {'acc%':>10s} {'Δpp':>8s}")
        by_r = {}
        for r in sub:
            by_r.setdefault(r["r"], []).append(r)
        for rv in sorted(by_r):
            rs = by_r[rv]
            mcm, mcs, n = stat([r["mc"] if task != "sort" else r["best"] for r in rs])
            d = (mcm - base) * 100
            print(f"  {rv:>6.2f} {(1-rv)*100:>5.0f}% {n:>3d} "
                  f"{mcm*100:6.2f}±{mcs*100:.2f} {d:>+7.2f}")
        if task == "sort":
            print("  ⚠ sort: only 1 r value + inert-suspected (topk ignored) "
                  "-> NOT a frontier point")
        if task == "cifar10":
            np = sum(1 for r in sp if r["task"] == task and r["r"] == 0.75
                     and is_partial(r))
            if np:
                print(f"  ⚠ cifar10 r=0.75: all {np} seeds PARTIAL (~36% trained) "
                      "-> excluded, trend inferred from r≤0.5")
        print()


def plot_sparsity(rows):
    fig, ax = plt.subplots(figsize=(11, 6.5))
    colors = {"mazes": "#ff7f0e", "cifar10": "#1f77b4", "parity": "#2ca02c"}
    for task in ["mazes", "cifar10", "parity"]:
        sp = [r for r in rows if r["stage"] == "sparsity" and r["task"] == task
              and not is_partial(r)]
        if not sp:
            continue
        by_r = {}
        for r in sp:
            by_r.setdefault(r["r"], []).append(r["mc"])
        rs = sorted(by_r)
        ys = [mean(by_r[k]) * 100 for k in rs]
        es = [stdev(by_r[k]) * 100 if len(by_r[k]) > 1 else 0 for k in rs]
        ns = [len(by_r[k]) for k in rs]
        ax.errorbar(rs, ys, yerr=es, marker="o", capsize=4, linewidth=2,
                    markersize=8, color=colors[task], label=task)
        for ri, yi, ni in zip(rs, ys, ns):
            ax.annotate(f"n={ni}", (ri, yi), textcoords="offset points",
                        xytext=(6, 6), fontsize=7, color=colors[task])
        # baseline star
        bm = [r["mc"] for r in rows if r["stage"] == "baseline"
              and r["task"] == task and r["mc"] is not None and not is_partial(r)]
        if bm:
            ax.plot(1.0, mean(bm) * 100, "*", markersize=15, color=colors[task],
                    markeredgecolor="black", markeredgewidth=0.7)
    # annotate parity r=0.1 dip
    ax.annotate("parity r=0.1 drops\n(parity needs neurons\nfor XOR accumulation)",
                xy=(0.1, 89.98), xytext=(0.18, 84), fontsize=8, color="#2ca02c",
                arrowprops=dict(arrowstyle="->", color="#2ca02c", lw=1.2))
    ax.axvspan(0, 0.3, color="#2ca02c", alpha=0.07)
    ax.text(0.15, 92.5, "low-r zone", ha="center", fontsize=9, color="#2ca02c")
    ax.set_xlabel("NLM compute fraction r  (top-k fraction of neurons active/tick)\n"
                  "← cheaper (saves (1−r) NLM compute) | full cost →", fontsize=10)
    ax.set_ylabel("most-certain-tick accuracy (%)", fontsize=10)
    ax.set_title("Sparsity Pareto (5-seed repro): accuracy vs NLM-compute cost\n"
                 "★ = dense baseline (r=1.0). mazes/cifar10 flat (free); "
                 "parity r=0.1 dips.", fontsize=11, fontweight="bold")
    ax.set_xlim(0, 1.05)
    ax.set_ylim(82, 101)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=10)
    fig.tight_layout()
    p = FIG_DIR / "figE_sparsity_pareto_0728.png"
    fig.savefig(p, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[fig] -> {p}")


def main():
    rows = load()
    print(f"loaded {len(rows)} runs; "
          f"partial(<50% iter)={sum(1 for r in rows if is_partial(r))}")
    pillar_verdict(rows)
    sparsity_report(rows)
    plot_sparsity(rows)


if __name__ == "__main__":
    main()
