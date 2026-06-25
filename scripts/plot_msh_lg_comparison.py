#!/usr/bin/env python3
"""Plot MSH-structure & learnable-gates comparison on the sort (N=10) task.

Reads csv_data/msh_results.csv (produced by export_msh_results.py) and draws
4 figures telling the story:

  fig1: MSH structures vs flat    -- structured (coprime/nested) > flat
  fig2: learnable gates vs fixed  -- learning the schedule does NOT help
  fig3: test loss curves (log y)  -- convergence speed
  fig4: summary bars              -- best/final test acc across all N=10 runs

Usage:
    python scripts/plot_msh_lg_comparison.py
    python scripts/plot_msh_lg_comparison.py --data-dir csv_data --out-dir runs/figures/msh_lg
"""

import argparse
import ast
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "csv_data"
DEFAULT_OUT = ROOT / "runs" / "figures" / "msh_lg"


def load_data(data_dir):
    df = pd.read_csv(Path(data_dir) / "msh_results.csv")
    for c in ["best_test_acc", "final_test_acc", "best_test_acc_full",
              "final_test_acc_full", "best_train_acc", "final_train_acc",
              "final_test_loss", "final_train_loss"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    # parse curve columns (stored as python list literals)
    for c in ["train_acc_curve", "test_acc_curve", "test_loss_curve"]:
        df[c] = df[c].apply(lambda s: ast.literal_eval(s) if isinstance(s, str) else [])
    return df


def iters_axis(row):
    n = len(row["test_acc_curve"])
    step = int(row["last_iter"] / max(row["num_track_points"] - 1, 1)) if row["num_track_points"] > 1 else 1000
    return np.arange(n) * step


# name -> (label, color, linestyle, linewidth)
MSH_STYLES = {
    "sort_per_tick_ce_N10":      ("flat (DTT baseline)", "#888888", "--", 2.0),
    "sort_N10_coprime_2x3":      ("coprime 2x3",         "#1f77b4", "-",  1.6),
    "sort_N10_coprime_2x3x5":    ("coprime 2x3x5",       "#2ca02c", "-",  1.6),
    "sort_N10_coprime_3x5x7":    ("coprime 3x5x7",       "#ff7f0e", "-",  1.6),
    "sort_N10_nested_10x5x1":    ("nested 10x5x1",       "#d62728", "-",  2.6),
}

LG_STYLES = {
    "fixed_coprime_2x3x5":       ("fixed coprime 2x3x5 (control)", "#111111", "--", 2.2),
    "learnable_coprime_init":    ("learnable, init=coprime",       "#1f77b4", "-",  1.6),
    "learnable_uniform_init":    ("learnable, init=uniform",       "#2ca02c", "-",  1.6),
    "learnable_random_init":     ("learnable, init=random",        "#ff7f0e", "-",  1.6),
    "learnable_coprime_sparse":  ("learnable, init=coprime, sp=0.05", "#9467bd", "-", 1.6),
}


def _plot_curves(df, names_styles, ax, curve_key, ylabel, title, logy=False):
    for name, (lbl, color, ls, lw) in names_styles.items():
        row = df[df["name"] == name]
        if row.empty:
            continue
        row = row.iloc[0]
        curve = row[curve_key]
        if not curve:
            continue
        x = iters_axis(row)
        ax.plot(x, curve, color=color, linestyle=ls, linewidth=lw,
                label=f"{lbl} ({row['best_test_acc']*100:.2f}%)", alpha=0.92)
    ax.set_xlabel("training iterations", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    if logy:
        ax.set_yscale("log")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8.5, loc="lower right")


# ---------------------------- Figure 1: MSH structures ----------------------------

def fig1_msh_structure(df, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    _plot_curves(df, MSH_STYLES, axes[0], "test_acc_curve",
                 "test accuracy", "(a) Test accuracy: MSH structures vs flat")
    _plot_curves(df, MSH_STYLES, axes[1], "train_acc_curve",
                 "train accuracy", "(b) Train accuracy")

    nested = df[df["name"] == "sort_N10_nested_10x5x1"]
    flat = df[df["name"] == "sort_per_tick_ce_N10"]
    if not nested.empty and not flat.empty:
        gain = (nested.iloc[0]["best_test_acc"] - flat.iloc[0]["best_test_acc"]) * 100
        axes[0].text(0.02, 0.06, f"nested beats flat by {gain:+.2f}pp",
                     transform=axes[0].transAxes, fontsize=10, color="#d62728",
                     fontweight="bold", verticalalignment="bottom")

    fig.suptitle("Structured thinking schedules beat flat backprop on sort (N=10)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    path = out_dir / "fig1_msh_structure.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig1] -> {path}")


# ---------------------------- Figure 2: learnable vs fixed ----------------------------

def fig2_learnable_vs_fixed(df, out_dir):
    fig, ax = plt.subplots(figsize=(9, 6))
    _plot_curves(df, LG_STYLES, ax, "test_acc_curve",
                 "test accuracy", "Learnable gates do not beat fixed coprime (sort N=10)")

    fixed = df[df["name"] == "fixed_coprime_2x3x5"]
    if not fixed.empty:
        ref = fixed.iloc[0]["best_test_acc"] * 100
        ax.axhline(ref, color="#111111", linestyle=":", alpha=0.35, linewidth=1)

    best_lg = df[df["name"].str.startswith("learnable_")]["best_test_acc"].max() * 100
    ax.annotate(f"best learnable {best_lg:.2f}%\nfixed {ref:.2f}%\n=> no gain",
                xy=(90000, ref), xytext=(55000, 0.78),
                fontsize=9.5, color="#444", fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="#444", alpha=0.6))

    fig.tight_layout()
    path = out_dir / "fig2_learnable_vs_fixed.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig2] -> {path}")


# ---------------------------- Figure 3: loss curves ----------------------------

def fig3_loss_curves(df, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    _plot_curves(df, MSH_STYLES, axes[0], "test_loss_curve",
                 "test loss", "(a) Test loss: MSH structures", logy=True)
    _plot_curves(df, LG_STYLES, axes[1], "test_loss_curve",
                 "test loss", "(b) Test loss: learnable vs fixed", logy=True)
    fig.suptitle("Convergence: test loss (log scale)", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    path = out_dir / "fig3_loss_curves.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig3] -> {path}")


# ---------------------------- Figure 4: summary bars ----------------------------

def fig4_summary_bars(df, out_dir):
    order = [
        ("flat",        "sort_per_tick_ce_N10"),
        ("coprime 2x3", "sort_N10_coprime_2x3"),
        ("coprime 2x3x5", "sort_N10_coprime_2x3x5"),
        ("coprime 3x5x7", "sort_N10_coprime_3x5x7"),
        ("nested 10x5x1", "sort_N10_nested_10x5x1"),
        ("LG: fixed coprime", "fixed_coprime_2x3x5"),
        ("LG: init=coprime", "learnable_coprime_init"),
        ("LG: init=uniform", "learnable_uniform_init"),
        ("LG: init=random", "learnable_random_init"),
        ("LG: coprime, sp=0.05", "learnable_coprime_sparse"),
    ]
    labels, bests, finals = [], [], []
    for lbl, name in order:
        row = df[df["name"] == name]
        if row.empty:
            continue
        labels.append(lbl)
        bests.append(row.iloc[0]["best_test_acc"] * 100)
        finals.append(row.iloc[0]["final_test_acc"] * 100)

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(14, 6))
    b1 = ax.bar(x - 0.2, bests, 0.4, color="#2ca02c", alpha=0.8,
                edgecolor="black", linewidth=0.5, label="best test acc")
    b2 = ax.bar(x + 0.2, finals, 0.4, color="#cccccc", alpha=0.9,
                edgecolor="black", linewidth=0.5, label="final test acc")
    for bar in list(b1) + list(b2):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.3, f"{h:.2f}",
                ha="center", fontsize=7.5, fontweight="bold")

    flat_best = df[df["name"] == "sort_per_tick_ce_N10"]["best_test_acc"]
    if not flat_best.empty:
        ax.axhline(float(flat_best.iloc[0]) * 100, color="#888888",
                   linestyle="--", linewidth=1.3, alpha=0.8,
                   label="flat baseline")

    # separator between structure groups and learnable group
    if "nested 10x5x1" in labels:
        sep = labels.index("nested 10x5x1") + 0.5
        ax.axvline(sep, color="#bbb", linestyle=":", linewidth=1)
        ax.text(sep - 0.1, 101, "MSH structures", ha="right", fontsize=9,
                color="#555", fontweight="bold")
        ax.text(sep + 0.1, 101, "learnable gates", ha="left", fontsize=9,
                color="#555", fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("test accuracy (%)", fontsize=11)
    ax.set_title("Sort (N=10) summary: best vs final test accuracy",
                 fontsize=12, fontweight="bold")
    ax.set_ylim(95, 103)
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(fontsize=9, loc="lower right")

    fig.tight_layout()
    path = out_dir / "fig4_summary_bars.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig4] -> {path}")


# ---------------------------- main ----------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default=str(DEFAULT_DATA))
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT))
    cli = ap.parse_args()

    data_dir = Path(cli.data_dir)
    out_dir = Path(cli.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_data(data_dir)
    print(f"Loaded {len(df)} rows from {data_dir / 'msh_results.csv'}")

    fig1_msh_structure(df, out_dir)
    fig2_learnable_vs_fixed(df, out_dir)
    fig3_loss_curves(df, out_dir)
    fig4_summary_bars(df, out_dir)

    print(f"\nAll figures -> {out_dir}")


if __name__ == "__main__":
    main()
