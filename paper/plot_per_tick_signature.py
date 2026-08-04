#!/usr/bin/env python3
"""Signature figure: CTM accuracy vs thought-tick (mc-vs-final caliber + mechanism split).

Reads paper_repro/csv_data/per_tick_fixes.json (per-tick test accuracy from the
detach-fixed rerun). Plots accuracy across CTM's internal thought-ticks, showing:
  - baseline: non-monotonic (rises, peaks mid, final-tick drops), mc★ above curve,
  - JEPA: lifts the LATE tail (late-tick stabilization), mc ~flat,
  - draft-revise (deep supervision, now correctly wired): a sharp EARLY peak at
    the draft boundary (tick ~1, ~85% ≈ mc ceiling) then faster late decay —
    i.e. early commitment via the draft CE, a DISTINCT mechanism from JEPA.

This is the strongest evidence that the two methods act on different parts of the
trajectory (JEPA=late, revise=early) without raising the mc ceiling.
"""
import json
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "paper_repro" / "csv_data" / "per_tick_fixes.json"
FIG_DIR = ROOT / "runs" / "figures" / "ctm_paper"
FIG_DIR.mkdir(parents=True, exist_ok=True)

STYLE = {
    "baseline": dict(color="#444444", ls="-",  label="baseline"),
    "jepa":     dict(color="#1f77b4", ls="--", label="JEPA (w=0.1)"),
    "revise":   dict(color="#2ca02c", ls=":",  label="draft-revise (deep superv.)"),
}


def load_curves(task):
    d = json.load(open(DATA))
    by = defaultdict(list)
    for k, v in d.items():
        grp, name = k.split("/", 1)
        if name.split("_")[0] == task:
            by[grp].append(v)
    return by


def plot_task(ax, task, title):
    by = load_curves(task)
    for grp in ["baseline", "jepa", "revise"]:
        runs = by.get(grp, [])
        if not runs:
            continue
        curves = np.array([r["acc_per_tick"] for r in runs])  # (n_seeds, T)
        m = curves.mean(0) * 100
        s = curves.std(0) * 100 if len(curves) > 1 else np.zeros_like(m)
        ticks = np.arange(len(m))
        st = STYLE[grp]
        ax.plot(ticks, m, color=st["color"], ls=st["ls"], lw=2.2, label=st["label"])
        ax.fill_between(ticks, m - s, m + s, color=st["color"], alpha=0.12)
        mc = np.mean([r["mc_acc"] for r in runs]) * 100
        ax.plot(len(m) - 1 + 1.5, mc, marker="*", color=st["color"],
                markersize=14, markeredgecolor="black", markeredgewidth=0.6,
                clip_on=False)
        ax.annotate(f"{grp}\nmc {mc:.1f}", (len(m) - 1 + 1.5, mc),
                    textcoords="offset points", xytext=(8, 0), fontsize=7.5,
                    color=st["color"], va="center")
    base = by.get("baseline", [])
    if base:
        bm = np.array([r["acc_per_tick"] for r in base]).mean(0) * 100
        ax.annotate(f"baseline final-tick\n{bm[-1]:.1f}%", (len(bm) - 1, bm[-1]),
                    textcoords="offset points", xytext=(-6, -22), fontsize=8,
                    color="#444444", ha="right",
                    arrowprops=dict(arrowstyle="->", color="#444444", lw=1))
    # draft-revise: annotate the sharp early peak (deep supervision at boundary)
    rev = by.get("revise", [])
    if rev and task == "cifar10":
        rm = np.array([r["acc_per_tick"] for r in rev]).mean(0) * 100
        pk = int(rm.argmax())
        ax.annotate(f"deep-supervision\nearly commit\n{rm[pk]:.1f}% @tick {pk}",
                    (pk, rm[pk]), textcoords="offset points", xytext=(10, -34),
                    fontsize=8, color="#2ca02c",
                    arrowprops=dict(arrowstyle="->", color="#2ca02c", lw=1))
    ax.set_title(title, fontsize=10.5, fontweight="bold")
    ax.set_xlabel("thought-tick (internal CTM iteration)", fontsize=10)
    ax.set_ylabel("test accuracy (%)", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-1, (len(m) - 1) + 4)


def main():
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    plot_task(axes[0], "cifar10",
              "cifar10: per-tick signature\n"
              "JEPA lifts the LATE tail; draft-revise (deep superv.) commits EARLY "
              "(tick~1); mc★ ~same for all three")
    plot_task(axes[1], "mazes",
              "mazes: per-tick (saturated — curve ~flat, little room to move)")
    axes[0].legend(loc="lower right", fontsize=9)
    axes[1].legend(loc="lower right", fontsize=9)
    fig.suptitle("CTM thought-tick profile — two distinct mechanisms, same mc ceiling "
                 "(JEPA stabilizes late ticks, draft-revise commits early)",
                 fontsize=11.5, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    p = FIG_DIR / "figS_per_tick_signature_fixes.png"
    fig.savefig(p, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] -> {p}")


if __name__ == "__main__":
    main()
