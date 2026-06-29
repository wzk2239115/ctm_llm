#!/usr/bin/env python3
"""Deep analysis of the 0629 CTM paper harvest (534 runs, 0 errors).

Goes beyond the canned plot_ctm_paper_results.py figures to surface patterns
that matter for writing the paper but aren't in the standard figs:

  1. parity 0-iter collapse      — which ideas kill parity training entirely
  2. st10 draft-revise 4-variant — w×cp grid per task + partial-run status +
                                   inert-dimension detection (revise_weight
                                   has zero effect on parity/sort)
  3. most-certain tick gain      — per-sample best tick vs final tick, by task
  4. sort fragility              — survive vs collapse, degenerate no-op sweeps
  5. top performers              — best delta vs baseline per task (complete runs)
  6. seed bimodality             — configs with huge std (collapse-or-work runs)

Outputs a text report to stdout + 2 supplementary figures to
runs/figures/ctm_paper/:
  - figA_st10_revise_grid.png  (2x2 w×cp heatmap per task)
  - figB_parity_collapse.png   (which ideas kill parity)

Usage:
  python paper/analyze_0629.py
"""

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "csv_data" / "ctm_paper_summary.csv"
CURVES = ROOT / "csv_data" / "ctm_paper_curves.json"
FIG_DIR = ROOT / "runs" / "figures" / "ctm_paper"
FIG_DIR.mkdir(parents=True, exist_ok=True)

BASELINE = {"cifar10": 0.669044, "mazes": 0.901613, "parity": 0.679713,
            "qamnist": 0.366244, "sort": 0.875347}
PLANNED_ITERS = {"cifar10": 200000, "mazes": 100000, "parity": 200000,
                 "qamnist": 200000, "sort": 100000}
TASK_ORDER = ["cifar10", "mazes", "parity", "qamnist", "sort"]
COLS = ["final_iter", "best_test_acc", "final_test_acc",
        "best_test_acc_mc", "final_test_acc_mc", "n_points", "seed"]


def load():
    df = pd.read_csv(CSV)
    for c in COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def sec(title):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def report_parity_collapse(df):
    sec("1. PARITY 0-ITER COLLAPSE — ideas that kill parity training entirely")
    dead = df[(df.task == "parity") & (df.final_iter.fillna(0) <= 0)]
    alive = df[(df.task == "parity") & (df.final_iter.fillna(0) > 0)]
    npar = len(df[df.task == "parity"])
    print(f"parity runs: {npar} total | dead (iter<=0): {len(dead)} "
          f"| alive: {len(alive)}")
    if dead.empty:
        return
    g = dead.groupby(["stage", "sweep"]).size().reset_index(name="n_dead")
    # how many seeds total per (stage,sweep)?
    tot = df[df.task == "parity"].groupby(["stage", "sweep"]).size().reset_index(name="n_total")
    g = g.merge(tot, on=["stage", "sweep"])
    g["dead_frac"] = (g.n_dead / g.n_total).round(2)
    print("\n(idea) kills ALL its parity seeds (training never progresses):")
    killed = g[g.dead_frac == 1.0].sort_values("stage")
    for _, r in killed.iterrows():
        print(f"  {r.stage} {r.sweep:28s}  {r.n_dead}/{r.n_total} dead")
    print("\n(idea) kills SOME parity seeds (partial/bimodal failure):")
    partial = g[(g.dead_frac > 0) & (g.dead_frac < 1.0)].sort_values("stage")
    for _, r in partial.iterrows():
        print(f"  {r.stage} {r.sweep:28s}  {r.n_dead}/{r.n_total} dead")

    # figure: bar of dead count per stage
    by_stage = dead.groupby("stage").size().sort_index()
    fig, ax = plt.subplots(figsize=(11, 4.5))
    colors = ["#d62728" if n >= 3 else "#ff9f43" for n in by_stage.values]
    ax.bar(range(len(by_stage)), by_stage.values, color=colors, edgecolor="black",
           linewidth=0.5)
    ax.set_xticks(range(len(by_stage)))
    ax.set_xticklabels(by_stage.index, rotation=30, ha="right")
    for i, v in enumerate(by_stage.values):
        ax.text(i, v + 0.2, str(v), ha="center", fontsize=9, fontweight="bold")
    ax.set_ylabel("# parity seeds dead (iter=0)")
    ax.set_title("Which CTM ideas kill parity training entirely?\n"
                 "(final_iter=0 => training never wrote a checkpoint step)",
                 fontsize=12, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    p = FIG_DIR / "figB_parity_collapse.png"
    fig.savefig(p, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[fig] -> {p}")


def report_st10_grid(df):
    sec("2. st10 DRAFT-REVISE — new 4-variant grid (revise_w × corrupt_prob)")
    new = df[(df.stage == "st10") & (df.sweep.str.contains("cp", na=False))]
    if new.empty:
        print("no new st10 variants found")
        return
    # parse w and cp
    new = new.copy()
    new["w"] = new.sweep.str.extract(r"w0p(\d)").astype(float)
    new["cp"] = new.sweep.str.extract(r"cp0p(\d)").astype(float)

    for t in TASK_ORDER:
        sub = new[new.task == t]
        if sub.empty:
            continue
        base = BASELINE.get(t)
        pct = PLANNED_ITERS.get(t, 1)
        print(f"\n--- {t}  (baseline {base*100:.1f}%, planned {pct} iter) ---")
        for _, r in sub.sort_values("sweep").iterrows():
            prog = (r.final_iter / pct * 100) if pct else 0
            d = (r.best_test_acc - base) * 100 if pd.notna(r.best_test_acc) else float("nan")
            tag = "PARTIAL" if prog < 90 else "done"
            print(f"  w0.{int(r.w)} cp0.{int(r.cp)}  best={r.best_test_acc*100:5.1f}% "
                  f"({d:+5.1f}pp)  iter={int(r.final_iter):>6d} [{tag}]")

    # inert-dimension check: does revise_w change the metric?
    print("\ninert-dimension check (revise_weight 0.1 vs 0.2, same corrupt_prob):")
    for t in TASK_ORDER:
        sub = new[new.task == t]
        if sub.empty:
            continue
        for cp in sorted(sub.cp.dropna().unique()):
            vals = sub[sub.cp == cp].sort_values("w").best_test_acc.tolist()
            if len(vals) == 2:
                same = abs(vals[0] - vals[1]) < 1e-6
                tag = "INERT (identical)" if same else "active"
                print(f"  {t:8s} cp0.{int(cp)}: w0.1={vals[0]*100:.2f}% "
                      f"w0.2={vals[1]*100:.2f}%  -> {tag}")

    # heatmap fig: 2x2 per task (rows w, cols cp), but we have 4 tasks -> 2x2 grid of tasks
    tasks_with = [t for t in TASK_ORDER if not new[new.task == t].empty]
    nt = len(tasks_with)
    fig, axes = plt.subplots(1, nt, figsize=(4.2 * nt, 4), squeeze=False)
    for i, t in enumerate(tasks_with):
        ax = axes[0][i]
        sub = new[new.task == t]
        mat = np.full((2, 2), np.nan)
        for _, r in sub.iterrows():
            wi = int(r.w) - 1
            ci = int(r.cp) - 1
            if 0 <= wi < 2 and 0 <= ci < 2:
                mat[wi, ci] = r.best_test_acc * 100
        im = ax.imshow(mat, cmap="RdYlGn", aspect="auto", vmin=0, vmax=100)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["cp0.15", "cp0.3"], fontsize=9)
        ax.set_yticks([0, 1]); ax.set_yticklabels(["w0.1", "w0.2"], fontsize=9)
        for wi in range(2):
            for ci in range(2):
                if not np.isnan(mat[wi, ci]):
                    c = "white" if mat[wi, ci] < 40 else "black"
                    ax.text(ci, wi, f"{mat[wi,ci]:.1f}%", ha="center", va="center",
                            fontsize=10, fontweight="bold", color=c)
        bl = BASELINE.get(t, 0) * 100
        ax.set_title(f"{t}\n(baseline {bl:.1f}%)", fontsize=11, fontweight="bold")
        plt.colorbar(im, ax=ax, shrink=0.8)
    fig.suptitle("st10 draft-revise: best_test_acc (%) by revise_weight × corrupt_prob\n"
                 "(parity/sort show INERT revise_weight — rows identical)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    p = FIG_DIR / "figA_st10_revise_grid.png"
    fig.savefig(p, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[fig] -> {p}")


def report_mc_gain(df):
    sec("3. MOST-CERTAIN TICK GAIN (per-sample best tick vs final tick)")
    print("best_test_acc_mc uses the most-certain tick per sample;\n"
          "best_test_acc uses only the LAST tick. Gap = value of tick selection.\n")
    sub = df[(df.stage == "st00") & (df.sweep == "paper")]
    for t in TASK_ORDER:
        r = sub[sub.task == t]
        if r.empty:
            continue
        bf = r.best_test_acc.mean()
        bm = r.best_test_acc_mc.mean()
        if pd.notna(bm):
            print(f"  {t:8s} final {bf*100:5.1f}%  -> mc {bm*100:5.1f}%  "
                  f"(+{(bm-bf)*100:5.1f}pp)")
        else:
            print(f"  {t:8s} final {bf*100:5.1f}%  -> mc   N/A (not tracked)")
    # also across all stages: how often does mc lift a run above baseline?
    print("\nacross ALL stages, runs where mc crosses the st00-final baseline:")
    for t in TASK_ORDER:
        base = BASELINE[t]
        r = df[(df.task == t) & df.best_test_acc_mc.notna() & (df.best_test_acc_mc > 0)]
        if r.empty:
            continue
        n_lift = (r.best_test_acc_mc > base).sum()
        print(f"  {t:8s} {n_lift}/{len(r)} runs have mc > st00-final-baseline "
              f"({base*100:.1f}%)")


def report_sort_fragility(df):
    sec("4. SORT FRAGILITY — survive vs collapse + degenerate no-op sweeps")
    srt = df[df.task == "sort"]
    surv = srt[srt.best_test_acc > 0.05]
    coll = srt[srt.best_test_acc <= 0.05]
    print(f"sort runs: {len(srt)} | survive (>5%): {len(surv)} | "
          f"collapse (<=5%): {len(coll)}")
    print("\nALL sort combo-idea stages (st16-st24) collapse (<5%):")
    combo = srt[srt.stage.isin([f"st{n}" for n in range(16, 25)])]
    for _, r in combo.sort_values(["stage", "sweep"]).iterrows():
        print(f"  {r.stage} {r.sweep:24s} best={r.best_test_acc*100:5.2f}%")
    # tick cliff
    print("\nsort tick cliff (st02): only tick50 survives")
    tk = srt[srt.stage == "st02"]
    for _, r in tk.sort_values("sweep").iterrows():
        tag = "OK" if r.best_test_acc > 0.05 else "DEAD"
        print(f"  {r.sweep:8s} best={r.best_test_acc*100:5.2f}%  [{tag}]")

    # degenerate: configs with identical stats (no-op sweeps)
    print("\ndegenerate no-op sweeps (different config, identical 0.925±0.0435):")
    g = surv.groupby(["stage", "sweep"]).agg(
        n=("best_test_acc", "count"),
        mean=("best_test_acc", "mean"),
        std=("best_test_acc", "std"),
        mn=("best_test_acc", "min"),
        mx=("best_test_acc", "max"),
    ).reset_index()
    degen = g[(g["mean"].round(4) == 0.9253) & (g["std"].round(4).fillna(0) == 0.0435)]
    for _, r in degen.iterrows():
        print(f"  {r.stage} {r.sweep:28s} mean={r['mean']*100:.1f}% "
              f"std={r['std']*100:.2f} (n={r.n})  <- likely NO-OP (sort ignores this arg)")


def report_top_performers(df):
    sec("5. TOP PERFORMERS per task (delta vs baseline, runs >=50% planned iter)")
    for t in TASK_ORDER:
        base = BASELINE[t]
        pct = PLANNED_ITERS[t]
        sub = df[(df.task == t) & (df.stage != "st00") & (df.final_iter >= pct * 0.5)
                 & df.best_test_acc.notna()]
        if sub.empty:
            print(f"\n--- {t}: no complete runs ---")
            continue
        g = sub.groupby(["stage", "sweep"]).best_test_acc.agg(["mean", "count"]).reset_index()
        g["delta"] = (g["mean"] - base) * 100
        g = g.sort_values("delta", ascending=False).head(5)
        print(f"\n--- {t}  (baseline {base*100:.1f}%) ---")
        for _, r in g.iterrows():
            print(f"  {r.stage} {r.sweep:26s} {r['mean']*100:5.1f}% "
                  f"({r['delta']:+5.1f}pp, n={int(r['count'])})")


def report_bimodality(df):
    sec("6. SEED BIMODALITY — configs with huge std (collapse-or-work)")
    g = df.groupby(["stage", "task", "sweep"]).agg(
        n=("best_test_acc", "count"),
        mean=("best_test_acc", "mean"),
        std=("best_test_acc", "std"),
        mn=("best_test_acc", "min"),
        mx=("best_test_acc", "max"),
    ).reset_index()
    g = g[(g.n >= 2) & g["std"].notna()].sort_values("std", ascending=False)
    g["spread"] = (g.mx - g.mn) * 100
    print("top 12 configs by seed-spread (min..max): bimodal = some seeds collapse")
    print(f"{'stage':6s} {'task':9s} {'sweep':26s} {'n':>2s} {'min':>6s} {'max':>6s} "
          f"{'spread':>7s}")
    for _, r in g.head(12).iterrows():
        print(f"{r.stage:6s} {r.task:9s} {r.sweep:26s} {int(r.n):2d} "
              f"{r.mn*100:5.1f}% {r.mx*100:5.1f}% {r.spread:6.1f}pp")


def main():
    df = load()
    print(f"loaded {len(df)} runs across {df.stage.nunique()} stages, "
          f"{df.task.nunique()} tasks")
    report_parity_collapse(df)
    report_st10_grid(df)
    report_mc_gain(df)
    report_sort_fragility(df)
    report_top_performers(df)
    report_bimodality(df)
    sec("DONE")
    print(f"supplementary figures in {FIG_DIR}")


if __name__ == "__main__":
    main()
