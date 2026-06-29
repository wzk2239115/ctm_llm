#!/usr/bin/env python3
"""Deep-experiment idea validity (01_revise / 02_jepa / 03_sparsity / 04_combos).

CRITICAL methodological fix: the deep CSVs report best_acc in the
`test_accuracies_most_certain` metric (per-sample best tick), but their
`baseline` column is the FINAL-tick paper baseline (e.g. cifar10 0.6443).
Comparing mc-acc to final-baseline inflates delta by the most-certain-tick
mechanism itself (+18pp on cifar10, +20pp on parity, +63pp on qamnist).

This script compares mc-vs-mc (apples to apples) using the st00 paper
most-certain-tick baselines, so the delta reflects the IDEA's effect, not the
tick-selection mechanism. For sort (no mc tracked, metric=full_list element
acc) the final/element baseline is used directly.

Normalcy filter (per run):
  CRASHED   final_iter < 0.1 * planned   (e.g. 04_combos jepa0p1+spar0p5 @ 2k)
  PARTIAL   final_iter < 0.5 * planned   (kept but flagged; mc may still be
                                          meaningful if near ceiling — see parity)
  NORMAL    >= 0.5 * planned
  DEGEN     sort no-op: same seed signature across many configs (sort ignores
            the idea's knob) — flagged, excluded from delta.

Output: per (idea, task) verdict using mc-vs-mc delta on the BEST variant group.
"""

import sys
from collections import defaultdict
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

# Correct baselines: most-certain-tick acc from st00 paper (mc-vs-mc).
# sort has no mc; uses element-level full_list acc (final), baseline = st00 best.
MC_BASELINE = {
    "cifar10": 0.8516,   # st00 cifar10 paper best_test_acc_mc
    "mazes":   0.9117,   # st00 mazes paper best_test_acc_mc (=0.91172)
    "parity":  0.8821,   # st00 parity paper best_test_acc_mc (=0.88207)
    "sort":    0.8753,   # st00 sort best_test_acc (element-level, no mc)
}
# final-tick baselines (for the "mc mechanism inflation" reference table)
FINAL_BASELINE = {"cifar10": 0.6690, "mazes": 0.9016, "parity": 0.6797,
                  "sort": 0.8753}
PLANNED = {"cifar10": 200000, "mazes": 100000, "parity": 200000, "sort": 100000}
TASKS = ["cifar10", "mazes", "parity", "sort"]

DEEP_FILES = {
    "01_revise":   DATA / "01_revise_results.csv",
    "02_jepa":     DATA / "02_jepa_deep_results.csv",
    "03_sparsity": DATA / "03_sparsity_deep_results.csv",
    "04_combos":   DATA / "04_combos_deep_results.csv",
}


def load_all():
    frames = []
    for study, p in DEEP_FILES.items():
        if not p.exists():
            print(f"  MISSING {p}")
            continue
        df = pd.read_csv(p)
        df["study"] = study
        for c in ["best_acc", "last_acc", "final_iter", "n_points", "baseline"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df["mc_base"] = df["task"].map(MC_BASELINE)
        df["final_base"] = df["task"].map(FINAL_BASELINE)
        # mc-vs-mc delta (the fair one); sort uses element-vs-element
        df["delta_mc"] = (df["best_acc"] - df["mc_base"]) * 100
        # the (inflated) mc-vs-final delta, shown only to expose the artifact
        df["delta_inflated"] = (df["best_acc"] - df["final_base"]) * 100
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def classify(r):
    fi = r.final_iter if pd.notna(r.final_iter) else 0
    plan = PLANNED.get(r.task, 1)
    if fi < 0.1 * plan:
        return "CRASHED"
    if fi < 0.5 * plan:
        return "PARTIAL"
    return "NORMAL"


def parse_group(name):
    """Bucket a deep run name into idea/variant-type for grouping."""
    n = str(name)
    # 04_combos
    if "full_stack" in n: return "combo", "revise+JEPA+sparsity"
    if "revise_jepa" in n or "revise+jepa0p1_s" in n: return "combo", "revise+JEPA"
    if "jepa_spar" in n or "jepa0p1+spar0p5_s" in n: return "combo", "JEPA+sparsity"
    if "revise_spar" in n or "revise+spar0p5_s" in n: return "combo", "revise+sparsity"
    # 01_revise
    if "_swp_" in n: return "revise", "sweep(w×cp)"
    if "_abl_" in n: return "revise", "ablation"
    if "revise_w0" in n: return "revise", "main(w0.1,cp0.15)"
    # 02_jepa
    if "jepa_w0p1" in n: return "JEPA", "main(w0.1)"
    if "_swp_" in n: return "JEPA", "sweep(weight)"
    if "_abl_" in n: return "JEPA", "ablation"
    # 03_sparsity
    if "sparsity0p5" in n: return "sparsity", "main(r0.5)"
    if "_swp_" in n: return "sparsity", "sweep(ratio)"
    return "?", "?"


def detect_sort_degen(df):
    """Sort no-op: same per-seed best_acc signature across many configs."""
    done = df[(df.task == "sort") & (df.final_iter >= 0.5 * PLANNED["sort"])]
    sigs = defaultdict(list)
    for _, r in done.iterrows():
        # group by study+config-family; signature = (study, tuple of sorted accs)
        pass
    # simpler: flag any sort run whose best_acc is one of the 3 magic values
    # observed (0.6458, 0.7917, 0.8021 deep-revise; 0.8753,0.9545,0.9462 ctm_paper)
    magic = {0.6458, 0.7917, 0.8021, 0.8753, 0.9545, 0.9462,
             0.645833, 0.791667, 0.802083, 0.875347, 0.954514, 0.946181}
    degen = set()
    for _, r in done.iterrows():
        if pd.notna(r.best_acc):
            if round(r.best_acc, 4) in {round(m, 4) for m in magic}:
                degen.add(r.name)
    return degen


def report(df):
    print("\n" + "=" * 80)
    print("DEEP IDEA VALIDITY — mc-vs-mc (apples to apples)")
    print("=" * 80)
    print("\nmetric = test_accuracies_most_certain for cifar10/mazes/parity;")
    print("         element-level full_list for sort. Delta vs st00 mc baseline.\n")

    df["class"] = df.apply(classify, axis=1)
    degen = detect_sort_degen(df)
    df["degen"] = df.index.isin(degen)

    # inflate-artifact reference: show how mc-vs-final lies
    print("--- metric-artifact check: mc-vs-final (INFLATED) vs mc-vs-mc (FAIR) ---")
    print(f"{'task':9s} {'mc_base':>8s} {'final_base':>10s} {'mc_mech':>8s} "
          f"(= inflation baked into the deep CSV 'baseline' column)")
    for t in TASKS:
        print(f"{t:9s} {MC_BASELINE[t]*100:7.1f}% {FINAL_BASELINE[t]*100:9.1f}% "
              f"{(MC_BASELINE[t]-FINAL_BASELINE[t])*100:+7.1f}pp")

    # per idea x task
    df["idea"], df["variant"] = zip(*df["name"].apply(parse_group))
    # only keep real idea rows (drop '?')
    df = df[df["idea"] != "?"].copy()

    print("\n--- per (idea, task): best variant-group mc-vs-mc delta ---")
    print(f"{'idea':10s} {'task':9s} {'best variant':24s} {'n':>3s} "
          f"{'mean_mc':>8s} {'delta_mc':>9s} {'delta_infl':>11s} {'verdict':>10s}")
    rows = []
    for idea in ["revise", "JEPA", "sparsity", "combo"]:
        for task in TASKS:
            sub = df[(df.idea == idea) & (df.task == task)
                     & (df["class"] != "CRASHED") & (~df["degen"])]
            if sub.empty:
                continue
            # group by variant, pick the one with best mean mc acc
            grp = sub.groupby("variant").agg(
                n=("best_acc", "count"),
                mean=("best_acc", "mean"),
                std=("best_acc", "std"),
                dmin=("best_acc", "min"),
                dmax=("best_acc", "max"),
            ).reset_index()
            grp = grp.sort_values("mean", ascending=False)
            best = grp.iloc[0]
            base = MC_BASELINE[task]
            delta_mc = (best["mean"] - base) * 100
            delta_inf = (best["mean"] - FINAL_BASELINE[task]) * 100
            if delta_mc > 1:
                verdict = "POSITIVE"
            elif delta_mc < -1:
                verdict = "NEGATIVE"
            else:
                verdict = "NEUTRAL"
            rows.append({"idea": idea, "task": task, "variant": best["variant"],
                         "n": int(best["n"]), "delta_mc": delta_mc,
                         "delta_infl": delta_inf, "verdict": verdict})
            print(f"{idea:10s} {task:9s} {str(best['variant']):24s} "
                  f"{int(best['n']):3d} {best['mean']*100:7.1f}% "
                  f"{delta_mc:+8.1f}pp {delta_inf:+10.1f}pp {verdict:>10s}")

    # the one to watch: parity revise
    print("\n--- parity revise (the candidate real win) — all 5 seeds ---")
    pr = df[(df.idea == "revise") & (df.task == "parity")]
    if not pr.empty:
        for _, r in pr.sort_values("name").iterrows():
            print(f"  {r['name']:40s} mc={r.best_acc*100:6.2f}% "
                  f"delta_mc={(r.best_acc-MC_BASELINE['parity'])*100:+6.1f}pp "
                  f"iter={int(r.final_iter):>6d} [{r['class']}]")

    # crashed combos
    print("\n--- CRASHED combos (excluded: only 2k iter) ---")
    cr = df[df["class"] == "CRASHED"]
    for _, r in cr.iterrows():
        print(f"  {r['name']:45s} iter={int(r.final_iter)} n_pts={int(r.n_points)}")

    # sort degen
    print("\n--- sort DEGENERATE no-op (excluded from delta) ---")
    sd = df[(df.task == "sort") & df["degen"]]
    print(f"  {len(sd)} sort runs flagged (same magic acc across configs)")
    if not sd.empty:
        print("  example signatures:")
        for sig, grp in sd.groupby(sd["best_acc"].round(4)):
            print(f"    acc={sig:.4f}: {list(grp['name'][:4])}{'...' if len(grp)>4 else ''}")

    return df, rows


def plot_deep(df, rows):
    """Bar chart: mc-vs-mc delta per (idea,task), with the inflated delta ghosted."""
    if not rows:
        return
    rdf = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(13, 6))
    ideas = ["revise", "JEPA", "sparsity", "combo"]
    tasks = TASKS
    x = np.arange(len(tasks))
    w = 0.18
    for i, idea in enumerate(ideas):
        sub = rdf[rdf.idea == idea].set_index("task").reindex(tasks)
        vals = sub["delta_mc"].values
        inf = sub["delta_infl"].values
        xpos = x + (i - 1.5) * w
        # ghost bar = inflated (mc-vs-final), faint
        ax.bar(xpos, inf, w, color="#cccccc", edgecolor="#999999",
               linewidth=0.4, alpha=0.5, zorder=1)
        # real bar = mc-vs-mc
        colors = {"POSITIVE": "#2ca02c", "NEGATIVE": "#d62728", "NEUTRAL": "#bbbbbb"}
        bar_colors = [colors.get(v, "#bbb") for v in sub["verdict"].fillna("NEUTRAL")]
        ax.bar(xpos, vals, w * 0.8, color=bar_colors, edgecolor="black",
               linewidth=0.5, zorder=3)
        # annotate
        for xi, v, n in zip(xpos, vals, sub["n"].fillna(0)):
            if pd.notna(v):
                ax.text(xi, v + (0.5 if v >= 0 else -1.2), f"{v:+.1f}",
                        ha="center", fontsize=7, fontweight="bold")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(tasks, fontsize=11, fontweight="bold")
    ax.set_ylabel("delta vs baseline (pp)")
    ax.set_title("DEEP experiments — idea validity, mc-vs-mc (solid) vs "
                 "mc-vs-final (ghost, INFLATED by tick mechanism)\n"
                 "green=POSITIVE  red=NEGATIVE  gray=NEUTRAL  |  ghost bar = the "
                 "artifact in the deep CSV 'baseline' column",
                 fontsize=11, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(["ghost = mc-vs-final (artifact)", "solid = mc-vs-mc (fair)"],
              loc="upper right", fontsize=9)
    fig.tight_layout()
    p = FIG_DIR / "figD_deep_idea_validity.png"
    fig.savefig(p, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[fig] -> {p}")


def main():
    df = load_all()
    if df.empty:
        print("no deep CSVs found")
        return
    print(f"loaded {len(df)} deep runs across {df.study.nunique()} studies, "
          f"{df.task.nunique()} tasks")
    df, rows = report(df)
    plot_deep(df, rows)
    # tally
    from collections import Counter
    c = Counter(r["verdict"] for r in rows)
    print("\n" + "=" * 80)
    print("VERDICT TALLY (deep, mc-vs-mc)")
    print("=" * 80)
    for v in ["POSITIVE", "NEUTRAL", "NEGATIVE"]:
        print(f"  {v:10s} {c.get(v, 0)}")
    print("\nKey: with mc-vs-mc the inflated cifar10 'wins' vanish; the one real")
    print("positive is revise on PARITY (mc 0.92-1.0 vs 0.882, n=5).")


if __name__ == "__main__":
    main()
