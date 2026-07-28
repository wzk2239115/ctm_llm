#!/usr/bin/env python3
"""mc ceiling figure: who raises the most-certain-tick accuracy?

Grouped bars per task: baseline vs JEPA vs draft-revise vs sparsity(best r).
Findings:
  - JEPA / draft-revise do NOT raise mc (overlap baseline noise on every task).
  - sparsity is flat on perceptual tasks (cifar10/mazes) BUT on parity the
    right ratio (r=0.25) RAISES mc to ~100% (+3pp) — a real ceiling raise
    via sparsity-as-regularization (stabilises parity's bimodal training).
Missing bars = method degenerates/crashes (JEPA on parity → ~50% random) or
not run.
"""
import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "Noto Sans CJK JP", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "paper_repro" / "csv_data" / "repro_summary_0728.csv"
FIG_DIR = ROOT / "runs" / "figures" / "ctm_paper"
FIG_DIR.mkdir(parents=True, exist_ok=True)

TASKS = ["cifar10", "mazes", "parity", "qamnist"]
GROUPS = [("baseline", "baseline", "#444444"),
          ("jepa", "JEPA", "#1f77b4"),
          ("revise", "draft-revise", "#2ca02c"),
          ("sparsity", "sparsity (best r)", "#d62728")]


def load():
    g = defaultdict(list)
    for r in csv.DictReader(open(CSV)):
        if not r["best_test_acc_mc"]:
            continue
        if r["stage"] in {"baseline", "jepa", "revise"}:
            g[(r["stage"], r["task"])].append(float(r["best_test_acc_mc"]) * 100)
        elif r["stage"] == "sparsity":
            # collect per (task, r); best-r chosen downstream
            g[("sparsity", r["task"], r["sweep"])].append(
                float(r["best_test_acc_mc"]) * 100)
    return g


def main():
    g = load()
    fig, ax = plt.subplots(figsize=(12, 6.5))
    x = np.arange(len(TASKS))
    w = 0.20
    for i, (stage, lab, col) in enumerate(GROUPS):
        means, errs, present, notes = [], [], [], []
        for t in TASKS:
            if stage == "sparsity":
                # pick best-r by mean mc
                cand = [(sw, g[("sparsity", t, sw)]) for sw in
                        ["sparsity0p1", "sparsity0p25", "sparsity0p5", "sparsity0p75"]
                        if ("sparsity", t, sw) in g]
                if not cand:
                    means.append(0); errs.append(0); present.append(False); notes.append("")
                    continue
                sw, v = max(cand, key=lambda kv: mean(kv[1]))
                m = mean(v); s = stdev(v) if len(v) > 1 else 0
                rstr = sw.replace("sparsity0p", "r=").replace("p", ".")
                means.append(m); errs.append(s); present.append(True)
                notes.append(rstr)
            else:
                v = g.get((stage, t), [])
                if v:
                    m = mean(v); s = stdev(v) if len(v) > 1 else 0
                    means.append(m); errs.append(s); present.append(True); notes.append("")
                else:
                    means.append(0); errs.append(0); present.append(False); notes.append("")
        offs = (i - 1.5) * w
        bars = ax.bar(x + offs, means, w, yerr=errs, capsize=4, color=col,
                      alpha=0.85, label=lab)
        for j, p in enumerate(present):
            if not p:
                bars[j].set_color("#dddddd")
                txt = "退化→随机\n(~50%,图4)" if (stage, TASKS[j]) == ("jepa", "parity") else "未跑"
                ax.text(x[j] + offs, 2, txt, ha="center", va="bottom",
                        fontsize=7, color="#999999")
            else:
                ax.text(x[j] + offs, means[j] + errs[j] + 0.6,
                        f"{means[j]:.1f}", ha="center", va="bottom",
                        fontsize=7.5, color=col)
                if notes[j]:
                    ax.text(x[j] + offs, 1.5, notes[j], ha="center", va="bottom",
                            fontsize=6.5, color=col, style="italic")
        # highlight parity-sparsity (the ceiling raise)
        if stage == "sparsity":
            j = TASKS.index("parity")
            if present[j]:
                base_v = mean(g.get(("baseline", "parity"), [0]))
                ax.annotate(f"+{means[j]-base_v:.1f}pp\n抬动天花板!",
                            (x[j] + offs, means[j] + errs[j] + 3),
                            ha="center", fontsize=8.5, color="#d62728",
                            fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{t}\n(baseline {mean(g.get(('baseline',t),[0])):.1f})"
                        for t in TASKS])
    ax.set_ylabel("most-certain-tick accuracy (%)", fontsize=10)
    ax.set_ylim(0, 108)
    ax.set_title("mc 天花板: JEPA/draft-revise 不抬动; 但 sparsity 在 parity(r=0.25)"
                 " 反而抬动至 ~100% (+3pp)\n(感知任务 cifar10/mazes 全平; "
                 "parity-sparsity 作正则稳定训练, 消除双峰失败)",
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
