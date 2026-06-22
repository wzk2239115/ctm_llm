#!/usr/bin/env python3
"""Plot HRM-Inspired experiment results.

6 figures telling the HRM story:
  fig1: ideas x tasks heatmap        -- no silver bullet
  fig2: sort task collapse pattern   -- 18/24 configs diverge
  fig3: truncated BPTT task split    -- most counter-intuitive finding
  fig4: combo vs best-single         -- combos systematically underperform
  fig5: most-certain vs test acc     -- CTM essence, big mc gain
  fig6: train-test gap               -- cifar10 baseline shows -13pp regression

All baselines are read dynamically from hr00 rows (not hardcoded).

Usage:
    python scripts/plot_hrm_results.py
    python scripts/plot_hrm_results.py --data-dir csv_data --out-dir runs/figures/hrm_inspired
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "csv_data"
DEFAULT_OUT = ROOT / "runs" / "figures" / "hrm_inspired"

TASKS = ["cifar10", "parity", "sort"]
TASK_COLORS = {
    "cifar10": "#1f77b4",
    "parity": "#2ca02c",
    "sort": "#9467bd",
}

# Below this on sort (~random 1/30 = 3.3%) we call it divergence, not low acc.
FAIL_THRESHOLD = 0.05

# Representative single-idea sweep per stage (used in heatmap + best-single).
IDEA_STAGES = [
    # (label, stage, sweep)
    ("BP\ntrunc",     "hr01", "bp5"),
    ("BP\nwarmup",    "hr02", "warmup0p2"),
    ("state\ndetach", "hr03", "detach10"),
    ("Adam\natan2",   "hr04", "atan2_b95"),
    ("stable\nmax",   "hr05", "stablemax"),
    ("gated\nattn",   "hr06", "gated_attn"),
    ("GRU\ninject",   "hr07", "inj_gru_gate"),
    ("hier\nH2xL25",  "hr08", "h2_l25"),
    ("EMA\n0.9999",   "hr09", "ema0p9999"),
    ("ACT\nhalt",     "hr10", "act_h50_e0p1"),
]
SINGLE_STAGES = [s for _, s, _ in IDEA_STAGES]

# Combos for fig4 (parity + sort only -- these are the tasks with combo data).
COMBOS = [
    ("hr11\nbp5+atan2",     "hr11", "bp5_atan2"),
    ("hr12\nhier+bp3",      "hr12", "h2_l5_bp3"),
    ("hr13\nhier+gated",    "hr13", "h2_l5_gated"),
    ("hr14\nbp+atan2+smax", "hr14", "bp5_atan2_smax"),
    ("hr15\nfull HRM",      "hr15", "full_hrm"),
]


def load_data(data_dir):
    csv_path = Path(data_dir) / "hrm_inspired_summary.csv"
    df = pd.read_csv(csv_path)
    for c in ["final_iter", "best_test_acc", "final_test_acc",
              "best_test_acc_mc", "final_test_acc_mc", "n_points",
              "final_test_loss", "best_train_acc", "final_train_acc"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def get_baseline(df, task):
    """hr00 best_test_acc for a task (single seed in this plan)."""
    sub = df[(df["stage"] == "hr00") & (df["task"] == task)]
    if sub.empty:
        return None
    return float(sub["best_test_acc"].iloc[0])


def is_failed(row):
    """A diverged run: acc below random-chance threshold or NaN."""
    if pd.isna(row["best_test_acc"]):
        return True
    return float(row["best_test_acc"]) < FAIL_THRESHOLD


def lookup(df, task, stage, sweep):
    """Return (best_acc, failed_flag) for (task, stage, sweep)."""
    sub = df[(df["task"] == task) & (df["stage"] == stage) & (df["sweep"] == sweep)]
    if sub.empty:
        return None, False
    row = sub.iloc[0]
    if is_failed(row):
        return None, True
    return float(row["best_test_acc"]), False


def best_single_acc(df, task):
    """Max best_test_acc across all single-idea stages (excluding combos + hr00)."""
    sub = df[(df["task"] == task) & (df["stage"].isin(SINGLE_STAGES))]
    ok = sub[~sub.apply(is_failed, axis=1)]
    if ok.empty:
        return None
    return float(ok["best_test_acc"].max())


# ---------------------------- Figure 1: heatmap ----------------------------

def fig1_hrm_heatmap(df, out_dir):
    baselines = {t: get_baseline(df, t) for t in TASKS}
    n_tasks, n_ideas = len(TASKS), len(IDEA_STAGES)

    acc_mat = np.full((n_tasks, n_ideas), np.nan)
    fail_mat = np.zeros((n_tasks, n_ideas), dtype=bool)
    for i, task in enumerate(TASKS):
        for j, (_, stage, sweep) in enumerate(IDEA_STAGES):
            acc, fail = lookup(df, task, stage, sweep)
            fail_mat[i, j] = fail
            if acc is not None:
                acc_mat[i, j] = acc
    bl_arr = np.array([baselines[t] for t in TASKS]).reshape(-1, 1)
    delta_mat = acc_mat - bl_arr

    fig, axes = plt.subplots(1, 2, figsize=(17, 5),
                             gridspec_kw={"width_ratios": [1, 1]})
    col_labels = [lbl for lbl, _, _ in IDEA_STAGES]

    # Panel A: absolute acc
    ax = axes[0]
    mat_pct = acc_mat * 100
    im = ax.imshow(mat_pct, cmap="RdYlGn", aspect="auto", vmin=0, vmax=100)
    ax.set_xticks(range(n_ideas))
    ax.set_xticklabels(col_labels, fontsize=9)
    ax.set_yticks(range(n_tasks))
    ax.set_yticklabels(TASKS, fontsize=11)
    for i in range(n_tasks):
        for j in range(n_ideas):
            if fail_mat[i, j]:
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                           facecolor="#555555", edgecolor="white"))
                ax.text(j, i, "FAIL", ha="center", va="center",
                        fontsize=8, color="white", fontweight="bold")
            elif np.isnan(mat_pct[i, j]):
                ax.text(j, i, "-", ha="center", va="center",
                        fontsize=11, color="gray")
            else:
                color = "white" if mat_pct[i, j] < 40 else "black"
                ax.text(j, i, f"{mat_pct[i, j]:.1f}", ha="center", va="center",
                        fontsize=9, color=color, fontweight="bold")
    ax.set_title("(a) Best test accuracy (%)", fontsize=12, fontweight="bold")
    plt.colorbar(im, ax=ax, shrink=0.8, label="acc (%)")

    # Panel B: delta vs hr00 baseline
    ax = axes[1]
    delta_pct = delta_mat * 100
    finite = delta_pct[~np.isnan(delta_pct)]
    vmax = max(abs(finite.min()), abs(finite.max()), 10) if len(finite) else 10
    im2 = ax.imshow(delta_pct, cmap="RdBu_r", aspect="auto",
                    vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(n_ideas))
    ax.set_xticklabels(col_labels, fontsize=9)
    ax.set_yticks(range(n_tasks))
    ax.set_yticklabels(TASKS, fontsize=11)
    for i in range(n_tasks):
        for j in range(n_ideas):
            if fail_mat[i, j]:
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                           facecolor="#555555", edgecolor="white"))
                ax.text(j, i, "FAIL", ha="center", va="center",
                        fontsize=8, color="white", fontweight="bold")
            elif np.isnan(delta_pct[i, j]):
                ax.text(j, i, "-", ha="center", va="center",
                        fontsize=11, color="gray")
            else:
                color = "white" if abs(delta_pct[i, j]) > vmax * 0.6 else "black"
                ax.text(j, i, f"{delta_pct[i, j]:+.1f}", ha="center", va="center",
                        fontsize=9, color=color, fontweight="bold")
    ax.set_title("(b) Delta vs hr00 baseline (pp)", fontsize=12, fontweight="bold")
    plt.colorbar(im2, ax=ax, shrink=0.8, label="delta (pp)")

    fig.suptitle("HRM ideas x tasks: no silver bullet (gray = diverged)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    path = out_dir / "fig1_hrm_heatmap.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig1] -> {path}")


# ---------------------------- Figure 2: sort collapse ----------------------------

def fig2_sort_collapse(df, out_dir):
    sort_df = df[df["task"] == "sort"].copy()
    if sort_df.empty:
        print("[fig2] no sort data")
        return
    sort_df["failed"] = sort_df.apply(is_failed, axis=1)
    stage_order = sorted(sort_df["stage"].unique(), key=lambda s: int(s[2:]))

    fig, ax = plt.subplots(figsize=(16, 7))
    np.random.seed(42)

    n_fail = 0
    n_ok = 0
    for i, stage in enumerate(stage_order):
        sub = sort_df[sort_df["stage"] == stage].sort_values("sweep")
        for _, row in sub.iterrows():
            x_j = i + np.random.uniform(-0.15, 0.15)
            failed = bool(row["failed"])
            y = 0.0 if failed else float(row["best_test_acc"]) * 100
            if failed:
                n_fail += 1
                color = "#d62728"
                marker = "X"
                lbl = "diverge"
                offset_y = -7
            else:
                n_ok += 1
                color = "#2ca02c" if y > 50 else "#ff7f0e"
                marker = "o"
                lbl = f"{row['sweep']}\n({y:.1f}%)"
                offset_y = 3 if y > 50 else -7
            ax.scatter(x_j, y, s=90, color=color, alpha=0.85,
                       edgecolors="black", linewidths=0.5, zorder=3, marker=marker)
            ax.annotate(lbl, (x_j, y),
                        textcoords="offset points", xytext=(0, offset_y),
                        ha="center", fontsize=6, color=color, alpha=0.9)

    bl = get_baseline(df, "sort")
    if bl is not None:
        ax.axhline(bl * 100, color="gray", linestyle="--", alpha=0.6,
                   linewidth=1.5, label=f"hr00 baseline ({bl * 100:.1f}%)")

    ax.axhspan(0, 5, color="red", alpha=0.08, zorder=0)
    ax.text(len(stage_order) - 0.5, 3, "collapse zone (<5%)",
            fontsize=9, color="red", alpha=0.7, ha="right")

    ax.set_xticks(range(len(stage_order)))
    ax.set_xticklabels(stage_order, rotation=45, fontsize=9)
    ax.set_xlabel("stage", fontsize=11)
    ax.set_ylabel("best test acc (%)", fontsize=11)
    ax.set_title(f"Sort task: {n_fail} of {n_fail + n_ok} HRM configs diverge "
                 f"(red X = loss diverged, orange = partial)",
                 fontsize=12, fontweight="bold")
    ax.set_ylim(-8, 105)
    ax.grid(True, axis="y", alpha=0.2)
    ax.legend(fontsize=9, loc="upper left")

    fig.tight_layout()
    path = out_dir / "fig2_sort_collapse.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig2] -> {path}")


# ---------------------------- Figure 3: BPTT task split ----------------------------

def fig3_bptt_task_split(df, out_dir):
    """hr01 bp_steps sweep: opposite effects across tasks."""
    bp_steps_list = [1, 2, 5, 10]

    fig, ax = plt.subplots(figsize=(11.5, 7))

    for task in TASKS:
        xs, ys, fails = [0], [], [False]
        bl = get_baseline(df, task)
        ys.append(bl * 100 if bl is not None else None)
        for bp in bp_steps_list:
            acc, fail = lookup(df, task, "hr01", f"bp{bp}")
            xs.append(bp)
            fails.append(fail)
            ys.append(0.0 if fail else (acc * 100 if acc is not None else None))

        # line through non-failed points only
        line_x = [x for x, y, f in zip(xs, ys, fails) if y is not None and not f]
        line_y = [y for y, f in zip(ys, fails) if y is not None and not f]
        ax.plot(line_x, line_y, "-o", color=TASK_COLORS[task],
                label=task, linewidth=2, markersize=8, alpha=0.85, zorder=3)

        # mark failed points with red X
        for x, y, f in zip(xs, ys, fails):
            if f:
                ax.scatter(x, 0.5, marker="X", s=180, color="red",
                           edgecolors="black", linewidths=0.8, zorder=4)

    # baseline reference lines per task
    for task in TASKS:
        bl = get_baseline(df, task)
        if bl is not None:
            ax.axhline(bl * 100, color=TASK_COLORS[task], linestyle=":",
                       alpha=0.35, linewidth=1)

    # annotations
    bl_c = get_baseline(df, "cifar10")
    cifar_bp5, _ = lookup(df, "cifar10", "hr01", "bp5")
    if bl_c and cifar_bp5:
        delta = (cifar_bp5 - bl_c) * 100
        ax.annotate(f"cifar10 +{delta:.1f}pp peak @ bp5",
                    (5, cifar_bp5 * 100),
                    xytext=(30, 25), textcoords="offset points",
                    fontsize=10, color=TASK_COLORS["cifar10"], fontweight="bold",
                    arrowprops=dict(arrowstyle="->",
                                    color=TASK_COLORS["cifar10"], alpha=0.7))
    ax.text(7.5, 8, "sort: all diverged\n(loss 56~75)", fontsize=9, color="red",
            alpha=0.9, ha="left", style="italic")
    ax.text(7.5, 45, "parity: all below baseline\n(-7 to -15pp)",
            fontsize=9, color=TASK_COLORS["parity"], alpha=0.9,
            ha="left", style="italic")

    ax.set_xticks([0, 1, 2, 5, 10])
    ax.set_xticklabels(["full\n(hr00)", "1", "2", "5", "10"], fontsize=10)
    ax.set_xlabel("bp_steps (ticks of gradient backprop)", fontsize=11)
    ax.set_ylabel("best test acc (%)", fontsize=11)
    ax.set_title("Truncated BPTT effects split by task\n"
                 "cifar10 +21pp / parity -15pp / sort diverges",
                 fontsize=12, fontweight="bold")
    ax.set_ylim(-3, 105)
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(fontsize=10, loc="center right")

    fig.tight_layout()
    path = out_dir / "fig3_bptt_task_split.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig3] -> {path}")


# ---------------------------- Figure 4: combo vs single ----------------------------

def fig4_combo_vs_single(df, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    for ax_i, task in enumerate(["parity", "sort"]):
        ax = axes[ax_i]
        bl = get_baseline(df, task)
        bl_pct = (bl * 100) if bl else 0
        best_s = best_single_acc(df, task)
        best_s_pct = (best_s * 100) if best_s else 0

        x = np.arange(len(COMBOS))
        combo_accs, combo_fails = [], []
        for _, stage, sweep in COMBOS:
            acc, fail = lookup(df, task, stage, sweep)
            combo_accs.append((acc * 100) if acc else 0)
            combo_fails.append(fail)

        bars = ax.bar(x, combo_accs, color="#d62728", alpha=0.75,
                      edgecolor="black", linewidth=0.6, label="combo")
        for bar, fail in zip(bars, combo_fails):
            if fail:
                bar.set_color("#777777")
                bar.set_hatch("xx")
            h = bar.get_height()
            lbl = "FAIL" if fail else f"{h:.1f}"
            ax.text(bar.get_x() + bar.get_width() / 2, h + 1.5, lbl,
                    ha="center", fontsize=8, fontweight="bold",
                    color="#555" if fail else "black")

        ax.axhline(best_s_pct, color="#2ca02c", linestyle="--",
                   linewidth=2, alpha=0.85,
                   label=f"best single = {best_s_pct:.1f}%")
        ax.axhline(bl_pct, color="gray", linestyle=":",
                   linewidth=1.2, alpha=0.6,
                   label=f"hr00 baseline = {bl_pct:.1f}%")

        ax.set_xticks(x)
        ax.set_xticklabels([c[0] for c in COMBOS], rotation=15, fontsize=9)
        ax.set_ylabel("best test acc (%)", fontsize=11)
        ax.set_title(f"{task}: combo vs best single idea",
                     fontsize=12, fontweight="bold")
        y_max = max(best_s_pct, max(combo_accs) or 0) * 1.25
        ax.set_ylim(0, max(y_max, 20))
        ax.grid(True, axis="y", alpha=0.2)
        ax.legend(fontsize=9, loc="upper right")

    fig.suptitle("Combos systematically underperform best single idea",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    path = out_dir / "fig4_combo_vs_single.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig4] -> {path}")


# ---------------------------- Figure 5: most-certain gain ----------------------------

def fig5_mc_gain(df, out_dir):
    valid = df[df["best_test_acc_mc"].notna() & df["best_test_acc"].notna()].copy()
    if valid.empty:
        print("[fig5] no mc data")
        return

    fig, ax = plt.subplots(figsize=(8.5, 7.5))
    lo = min(valid["best_test_acc"].min(), valid["best_test_acc_mc"].min()) * 100 - 2
    hi = max(valid["best_test_acc"].max(), valid["best_test_acc_mc"].max()) * 100 + 2
    ax.plot([lo, hi], [lo, hi], "k--", alpha=0.4, linewidth=1,
            label="mc = test (no gain)")

    for task in TASKS:
        sub = valid[valid["task"] == task]
        if sub.empty:
            continue
        ax.scatter(sub["best_test_acc"] * 100, sub["best_test_acc_mc"] * 100,
                   color=TASK_COLORS[task], s=70, alpha=0.7,
                   edgecolors="black", linewidths=0.5, label=task, zorder=3)

    for i, task in enumerate(TASKS):
        sub = valid[valid["task"] == task]
        if sub.empty:
            continue
        gain = (sub["best_test_acc_mc"] - sub["best_test_acc"]).mean() * 100
        ax.text(0.03, 0.95 - i * 0.06,
                f"{task}: +{gain:.1f}pp mean gain",
                transform=ax.transAxes, fontsize=10,
                color=TASK_COLORS[task], fontweight="bold",
                verticalalignment="top")

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("best test acc (%)", fontsize=11)
    ax.set_ylabel("best test acc @ most-certain tick (%)", fontsize=11)
    ax.set_title("CTM essence: most-certain tick >> final-tick prediction",
                 fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.2)
    ax.legend(fontsize=9, loc="lower right")

    fig.tight_layout()
    path = out_dir / "fig5_mc_gain.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig5] -> {path}")


# ---------------------------- Figure 6: train-test gap ----------------------------

def fig6_train_test_gap(df, out_dir):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for ax_i, task in enumerate(TASKS):
        ax = axes[ax_i]
        sub = df[df["task"] == task].copy()
        sub["gap"] = (sub["best_test_acc"] - sub["final_test_acc"]) * 100
        sub["stage_n"] = sub["stage"].str[2:].astype(int)
        sub = sub.sort_values(["stage_n", "sweep"]).reset_index(drop=True)

        x = np.arange(len(sub))
        best_vals = sub["best_test_acc"].values * 100
        final_vals = np.where(np.isnan(sub["final_test_acc"].values),
                              0, sub["final_test_acc"].values * 100)
        gaps = sub["gap"].values

        ax.bar(x - 0.2, best_vals, 0.4, color=TASK_COLORS[task], alpha=0.65,
               edgecolor="black", linewidth=0.4, label="best")
        ax.bar(x + 0.2, final_vals, 0.4, color="#cccccc", alpha=0.85,
               edgecolor="black", linewidth=0.4, label="final")

        for i, gap in enumerate(gaps):
            if not np.isnan(gap) and gap > 10:
                mid = (best_vals[i] + final_vals[i]) / 2
                ax.annotate(f"-{gap:.0f}", (i, mid),
                            ha="center", fontsize=7, color="red", alpha=0.9,
                            fontweight="bold")

        # highlight hr00
        for p in sub.index[sub["stage"] == "hr00"].tolist():
            ax.axvspan(p - 0.5, p + 0.5, color="yellow", alpha=0.18, zorder=0)

        ax.set_xticks(x)
        ax.set_xticklabels([f"{s}\n{w[:9]}" for s, w in
                            zip(sub["stage"], sub["sweep"])],
                           rotation=70, fontsize=6)
        ax.set_ylabel("test acc (%)", fontsize=10)
        ax.set_title(f"{task}: best vs final (red = gap > 10pp)",
                     fontsize=11, fontweight="bold")
        ax.grid(True, axis="y", alpha=0.2)
        ax.legend(fontsize=9, loc="upper right")

    fig.suptitle("Train-test gap: cifar10 hr00 (yellow) shows -13pp regression",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    path = out_dir / "fig6_train_test_gap.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig6] -> {path}")


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
    print(f"Loaded {len(df)} rows from {data_dir / 'hrm_inspired_summary.csv'}")
    print(f"  tasks:  {sorted(df['task'].unique())}")
    print(f"  stages: {sorted(df['stage'].unique())}")

    fig1_hrm_heatmap(df, out_dir)
    fig2_sort_collapse(df, out_dir)
    fig3_bptt_task_split(df, out_dir)
    fig4_combo_vs_single(df, out_dir)
    fig5_mc_gain(df, out_dir)
    fig6_train_test_gap(df, out_dir)

    print(f"\nAll figures -> {out_dir}")


if __name__ == "__main__":
    main()
