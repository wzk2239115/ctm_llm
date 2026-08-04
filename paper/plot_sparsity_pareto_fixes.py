#!/usr/bin/env python3
"""figE (fixes): sparsity Pareto under REAL sparse NLM compute.

Reads paper_repro/csv_data/repro_summary_fixes.csv (detach-fix rerun + real
sparse_nlm_compute sparsity). Plots most-certain-tick accuracy vs NLM-compute
fraction r, per task, with the dense baseline (r=1.0) as a star.

Headline (real sparse, not post-hoc mask):
  - cifar10 / mazes: near-free (Δ within ±0.9pp across r=0.1..0.75).
  - parity: ALL r below baseline (r=0.1 crashes -16pp; r>=0.25 also -1.5..-4pp).
    No sweet spot — the old post-hoc r=0.25 "+3pp" vanished under real compute.
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "paper_repro" / "csv_data" / "repro_summary_fixes.csv"
FIG_DIR = ROOT / "runs" / "figures" / "ctm_paper"
FIG_DIR.mkdir(parents=True, exist_ok=True)
RATIOS = [0.1, 0.25, 0.5, 0.75]
COLORS = {"cifar10": "#1f77b4", "mazes": "#ff7f0e", "parity": "#2ca02c"}


def mc_for(df, prefix):
    r = df[df.exp.str.startswith(prefix, na=False)].dropna(subset=["best_test_acc_mc"])
    return r.best_test_acc_mc.values


def main():
    df = pd.read_csv(CSV)
    fig, ax = plt.subplots(figsize=(10.5, 6.5))
    for task in ["cifar10", "mazes", "parity"]:
        base = mc_for(df, f"{task}_baseline")
        if len(base) == 0:
            continue
        base_m = base.mean() * 100
        xs, ys, es = [1.0], [base_m], [base.std(ddof=0) * 100 / np.sqrt(len(base))]
        for r in RATIOS:
            a = mc_for(df, f"{task}_sparsity{str(r).replace('.', 'p')}")
            if len(a):
                xs.append(r); ys.append(a.mean() * 100)
                es.append(a.std(ddof=0) * 100 / np.sqrt(len(a)))
        order = np.argsort(xs)
        xs, ys, es = np.array(xs)[order], np.array(ys)[order], np.array(es)[order]
        ax.errorbar(xs, ys, yerr=es, marker="o", capsize=4, lw=2, markersize=8,
                    color=COLORS[task], label=f"{task} (baseline {base_m:.1f}%)")
        # mark dense baseline point as star
        ax.plot(1.0, base_m, "*", markersize=15, color=COLORS[task],
                markeredgecolor="black", markeredgewidth=0.7)
    # shade near-free zone (low r)
    ax.axvspan(0.0, 0.3, color="#1f77b4", alpha=0.06)
    ax.text(0.15, 81, "near-free zone\n(perceptual tasks)", ha="center",
            fontsize=9, color="#1f77b4", fontweight="bold")
    ax.set_xlabel("NLM compute fraction r  (real sparse: only k=⌊r·D⌋ neurons computed per tick)\n"
                  "← cheaper   |   full cost (dense) →", fontsize=10)
    ax.set_ylabel("most-certain-tick accuracy (%)", fontsize=10)
    ax.set_title("Sparsity Pareto under REAL sparse NLM compute\n"
                 "★ = dense baseline.  perceptual (cifar10/mazes) near-free;  "
                 "parity all r below baseline (hard boundary, no sweet spot)",
                 fontsize=10.5, fontweight="bold")
    ax.set_xlim(0.0, 1.07)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower left", fontsize=10)
    fig.tight_layout()
    p = FIG_DIR / "figE_sparsity_pareto_fixes.png"
    fig.savefig(p, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] -> {p}")


if __name__ == "__main__":
    main()
