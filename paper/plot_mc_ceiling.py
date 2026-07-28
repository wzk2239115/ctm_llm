#!/usr/bin/env python3
"""mc ceiling figure: ideas don't raise the most-certain-tick ceiling.

Grouped bars per task: baseline vs JEPA vs draft-revise most-certain-tick
accuracy (best_test_acc_mc, 5-seed). Within each task the bars overlap within
noise — the unifying negative result that JEPA/revise shift the per-tick
curve (figS) but do NOT raise the mc ceiling. Missing bars = task killed
(JEPA on parity: 0-iter graveyard) or not run (revise on qamnist).
"""
import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "paper_repro" / "csv_data" / "repro_summary_0728.csv"
FIG_DIR = ROOT / "runs" / "figures" / "ctm_paper"
FIG_DIR.mkdir(parents=True, exist_ok=True)

TASKS = ["cifar10", "mazes", "parity", "qamnist"]
GROUPS = [("baseline", "baseline", "#444444"),
          ("jepa", "JEPA", "#1f77b4"),
          ("revise", "draft-revise", "#2ca02c")]


def load():
    g = defaultdict(list)
    for r in csv.DictReader(open(CSV)):
        if r["best_test_acc_mc"] and r["stage"] in {x[0] for x in GROUPS}:
            g[(r["stage"], r["task"])].append(float(r["best_test_acc_mc"]) * 100)
    return g


def main():
    g = load()
    fig, ax = plt.subplots(figsize=(11, 6))
    x = np.arange(len(TASKS))
    w = 0.26
    for i, (stage, lab, col) in enumerate(GROUPS):
        means, errs, present = [], [], []
        for t in TASKS:
            v = g.get((stage, t), [])
            if v:
                m = mean(v)
                s = stdev(v) if len(v) > 1 else 0
                means.append(m); errs.append(s); present.append(True)
            else:
                means.append(0); errs.append(0); present.append(False)
        offs = (i - 1) * w
        bars = ax.bar(x + offs, means, w, yerr=errs, capsize=4, color=col,
                      alpha=0.85, label=lab)
        # grey out / mark missing
        for j, p in enumerate(present):
            if not p:
                bars[j].set_color("#dddddd")
                ax.text(x[j] + offs, 2, ("退化→随机\n(~50%,图4)" if (stage, TASKS[j])
                        == ("jepa", "parity") else "未跑"),
                        ha="center", va="bottom", fontsize=7, color="#999999")
            else:
                ax.text(x[j] + offs, means[j] + errs[j] + 0.6,
                        f"{means[j]:.1f}", ha="center", va="bottom", fontsize=8,
                        color=col)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{t}\n(baseline mc {mean(g.get(('baseline',t),[0])):.1f})"
                        for t in TASKS])
    ax.set_ylabel("most-certain-tick accuracy (%)", fontsize=10)
    ax.set_ylim(0, 105)
    ax.set_title("mc ceiling unchanged: JEPA / draft-revise do NOT raise the "
                 "most-certain-tick accuracy\n(within baseline noise on every task; "
                 "they shift the per-tick curve instead — see figS)",
                 fontsize=11, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="lower right", fontsize=10)
    fig.tight_layout()
    p = FIG_DIR / "figMC_ceiling_0728.png"
    fig.savefig(p, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] -> {p}")


if __name__ == "__main__":
    main()
