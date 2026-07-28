#!/usr/bin/env python3
"""Historical ctm_paper sweep figures (complement the repro point-estimates).

Three figures from csv_data/ctm_paper_summary.csv (3-seed historical sweeps),
used to show TRENDS within each sweep (internally fair, shared st00 baseline):
  figJW — JEPA weight sweep (acc vs w): sweet-spot on cifar10, parity degenerates
  figJA — JEPA ablation (cifar10): cosine loss essential, stop-grad helps
  figTS — thought-tick sweep (acc vs iterations): parity scales with ticks

NOTE: absolute values differ from the 5-seed repro (different seeds/config);
these figures are for within-sweep trends, not cross-dataset comparison.
"""
import csv
from pathlib import Path
from statistics import mean

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "Noto Sans CJK JP", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "csv_data" / "ctm_paper_summary.csv"
FIG_DIR = ROOT / "runs" / "figures" / "ctm_paper"
FIG_DIR.mkdir(parents=True, exist_ok=True)

TASK_COL = {"cifar10": "#1f77b4", "mazes": "#ff7f0e",
            "parity": "#2ca02c", "qamnist": "#d62728"}


def mc(stage, task, sweep):
    vals = [float(r["best_test_acc_mc"]) for r in csv.DictReader(open(CSV))
            if r["stage"] == stage and r["task"] == task and r["sweep"] == sweep
            and r.get("best_test_acc_mc")]
    return (mean(vals) * 100, len(vals)) if vals else (None, 0)


def fig_jepa_weight():
    """acc vs JEPA weight, per task (baseline at w=0)."""
    ws = [0.1, 0.5, 1.0]
    fig, ax = plt.subplots(figsize=(10, 6))
    for task in ["cifar10", "mazes", "qamnist", "parity"]:
        base, _ = mc("st00", task, "paper")
        ys = [mc("st04", task, f"jepa_w{w}")[0] for w in ws]
        if base is None:
            continue
        xs = [0.0] + ws
        full = [base] + ys
        ax.plot(xs, full, "-o", color=TASK_COL[task], lw=2.2, markersize=8,
                label=f"{task} (base {base:.1f})")
        # mark parity random level
        if task == "parity":
            ax.axhline(50, ls=":", color="gray", alpha=0.5)
            ax.text(1.0, 51, "随机水平 (50%)", fontsize=8, color="gray")
    ax.set_xlabel("JEPA 辅助损失权重 $w$ (0 = 无 JEPA, 即 baseline)", fontsize=11)
    ax.set_ylabel("most-certain-tick accuracy (%)", fontsize=11)
    ax.set_title("图 4. JEPA 权重 sweep: cifar10 有甜点($w{=}0.1$), 高权重掉点; "
                 "parity 在任意权重下退化为随机\n(st04, 3-seed; 趋势用, 非跨数据集比较)",
                 fontsize=11, fontweight="bold")
    ax.set_xticks([0.0, 0.1, 0.5, 1.0])
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower left", fontsize=10)
    ax.set_xlim(-0.05, 1.05)
    fig.tight_layout()
    p = FIG_DIR / "figJW_jepa_weight_0728.png"
    fig.savefig(p, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] -> {p}")


def fig_jepa_ablation():
    """cifar10 ablation: validates cosine loss + stop-grad design."""
    base, _ = mc("st00", "cifar10", "paper")
    items = [
        ("baseline\n(无 JEPA)", base),
        ("JEPA 完整\n(w=0.1)", mc("st04", "cifar10", "jepa_w0.1")[0]),
        ("− stop-grad\n(nostopgrad)", mc("st04", "cifar10", "jepa_nostopgrad")[0]),
        ("cosine→mse\n(均方损失)", mc("st04", "cifar10", "jepa_mse")[0]),
        ("predict_delta=1", mc("st04", "cifar10", "jepa_pd1")[0]),
    ]
    labels = [it[0] for it in items]
    vals = [it[1] if it[1] is not None else 0 for it in items]
    colors = ["#888888", "#1f77b4", "#ff7f0e", "#d62728", "#9467bd"]
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(labels, vals, color=colors, alpha=0.85)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 1, f"{v:.1f}%",
                ha="center", fontsize=10, fontweight="bold")
    ax.axhline(base, ls="--", color="gray", alpha=0.6, label=f"baseline {base:.1f}%")
    ax.set_ylabel("most-certain-tick accuracy (%)", fontsize=11)
    ax.set_title("图 5. JEPA 消融 (cifar10): cosine 损失是关键(mse 崩至 41%), "
                 "stop-grad 有正向贡献\n(st04, 3-seed)", fontsize=11, fontweight="bold")
    ax.set_ylim(0, 100)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize=10)
    fig.tight_layout()
    p = FIG_DIR / "figJA_jepa_ablation_0728.png"
    fig.savefig(p, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] -> {p}")


def fig_tick_sweep():
    """acc vs number of thought-ticks (the other efficiency axis)."""
    ticks = [1, 2, 5, 10, 25, 50]
    fig, ax = plt.subplots(figsize=(10, 6))
    for task in ["cifar10", "mazes", "parity"]:
        ys = [mc("st02", task, f"tick{t}")[0] for t in ticks]
        if all(y is None for y in ys):
            continue
        ax.plot(ticks, ys, "-o", color=TASK_COL[task], lw=2.2, markersize=8,
                label=task)
    ax.set_xscale("log")
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(t) for t in ticks])
    ax.set_xlabel("思考步数 $T$ (内部 tick 数, log 轴)", fontsize=11)
    ax.set_ylabel("most-certain-tick accuracy (%)", fontsize=11)
    ax.set_title("图 6. 思考步数 sweep: parity 强依赖 $T$(XOR 累加需多步, "
                 "tick1→50 从 65→92); cifar10 非单调\n(st02, 3-seed) —— "
                 "与 sparsity(每步激活神经元比例)构成效率的两条轴",
                 fontsize=11, fontweight="bold")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(fontsize=10)
    fig.tight_layout()
    p = FIG_DIR / "figTS_tick_sweep_0728.png"
    fig.savefig(p, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] -> {p}")


def main():
    fig_jepa_weight()
    fig_jepa_ablation()
    fig_tick_sweep()


if __name__ == "__main__":
    main()
