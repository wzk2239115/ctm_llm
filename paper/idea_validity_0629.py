#!/usr/bin/env python3
"""Idea validity analysis on the 0629 harvest, using ONLY normal data.

Philosophy: a run counts as evidence of an idea's effect only if the idea
actually trained to a meaningful point AND the run is not polluted by a known
plan bug. Everything else is categorised (killed / partial / inert /
bug-polluted) so we report WHY we can't use it, not silently drop it.

Normalcy filter (per run):
  KILLED     final_iter == 0                  -> idea broke training entirely
  PARTIAL    final_iter < 0.5 * planned       -> negative delta unreliable
  BUG        known plan bug pollutes it       -> excluded by name (see BUG_EXCLUDE)
  NORMAL     otherwise                         -> participates in delta judgement
  INERT      normal but acc identical to baseline across seeds (idea not wired)

Known plan bugs excluded by name (documented in fix_bug_st02/st04 scripts):
  - st02 sort tick1/2/5/10/25 : memory_length coupled to n_ticks (sort needs >=25)
  - st04 sort jepa_w0.5/w1.0  : weight confound (sort collapses at weight>=0.5)
  - st04 sort jepa_mse/nostopgrad/pd1/pd4 : same, ran at weight=1.0
  (st04 sort jepa_w0.1 is KEPT — it's the unconfounded point)

Idea -> stage mapping (only single-idea stages; combos st13-24 noted separately):
  JEPA     st04 (jepa_w0.1/0.5/1.0 + mse/nostopgrad/pd1/pd4)
  halt     st06 (halt0.3/0.6/0.9)
  sparsity st08 (sparsity0.25/0.5/0.75)
  reflex   st09
  revise   st10 (legacy 'revise' + new w0.{1,2}_cp0.{15,3} grid)
  EMA      st12

Judgement per (idea, task), using normal runs only:
  KILLED       all seeds KILLED (idea never trains)
  PARTIAL_ONLY only partial runs survived -> delta unverifiable
  INERT        normal runs exist but acc == baseline (idea not wired onto model)
  POSITIVE     mean delta > +1pp
  NEGATIVE     mean delta < -1pp
  NEUTRAL      |delta| <= 1pp
  NO_DATA      no runs at all

Usage:
  python paper/idea_validity_0629.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "csv_data" / "ctm_paper_summary.csv"
FIG_DIR = ROOT / "runs" / "figures" / "ctm_paper"
FIG_DIR.mkdir(parents=True, exist_ok=True)

BASELINE = {"cifar10": 0.669044, "mazes": 0.901613, "parity": 0.679713,
            "qamnist": 0.366244, "sort": 0.875347}
PLANNED = {"cifar10": 200000, "mazes": 100000, "parity": 200000,
           "qamnist": 200000, "sort": 100000}
TASKS = ["cifar10", "mazes", "parity", "qamnist", "sort"]

# (stage, task, sweep) triples polluted by documented plan bugs.
BUG_EXCLUDE = set()
for t in [1, 2, 5, 10, 25]:
    BUG_EXCLUDE.add(("st02", "sort", f"tick{t}"))
for sw in ["jepa_w0.5", "jepa_w1.0", "jepa_mse", "jepa_nostopgrad",
           "jepa_pd1", "jepa_pd4"]:
    BUG_EXCLUDE.add(("st04", "sort", sw))

# Idea -> (stage, list-of-sweeps). Combos handled in a separate block.
IDEAS = {
    "JEPA":     ("st04", ["jepa_w0.1", "jepa_w0.5", "jepa_w1.0",
                          "jepa_mse", "jepa_nostopgrad", "jepa_pd1", "jepa_pd4"]),
    "halt":     ("st06", ["halt0.3", "halt0.6", "halt0.9"]),
    "sparsity": ("st08", ["sparsity0.25", "sparsity0.5", "sparsity0.75"]),
    "reflex":   ("st09", ["reflex"]),
    "revise":   ("st10", ["revise", "revise_w0p1_cp0p15", "revise_w0p1_cp0p3",
                          "revise_w0p2_cp0p15", "revise_w0p2_cp0p3"]),
    "EMA":      ("st12", ["ema"]),
}
# For revise, the legacy 'revise' sweep on sort is a known inert no-op
# (92.53 identical to baseline) — keep it; the INERT detector will flag it.

COMBOS = {
    "JEPA+halt":      ("st13", ["jepa_halt"]),
    "JEPA+sparsity":  ("st14", ["jepa_sparsity"]),
    "halt+sparsity":  ("st15", ["halt_sparsity"]),
}

NUMCOLS = ["final_iter", "best_test_acc", "final_test_acc", "best_test_acc_mc",
           "n_points", "seed"]
INERT_TOL = 1e-4  # acc differences below this = identical (idea not wired)

# Filled by compute_degenerate() — (task, stage, sweep) triples whose per-seed
# acc signature recurs across >=2 different stages = the task ignored the knob
# and reproduced a fixed seed sequence (sort's 0.9253 no-op pattern).
DEGEN = set()


def compute_degenerate(df):
    """Detect no-op sweeps: same per-seed acc signature across >=2 stages.

    Only considers runs that actually trained (final_iter >= 0.5*planned);
    0-iter killed runs all share the random-guess acc (~49.9% on parity) and
    would false-positive as "degenerate".

    Sort's signature pattern: heads/sparsity/nst/tick50/revise all produce the
    exact same 3-seed sequence (0.875, 0.9545, 0.946) because sort's train.py
    ignores those args. This flags them so they count as INERT, not as real
    +5pp gains.
    """
    from collections import defaultdict
    sig_map = defaultdict(list)
    df2 = df.copy()
    df2["final_iter"] = pd.to_numeric(df2["final_iter"], errors="coerce").fillna(0)
    # only completed runs carry a meaningful signature
    done = df2[df2.apply(
        lambda r: r["final_iter"] >= 0.5 * PLANNED.get(r["task"], 0), axis=1)]
    for (task, stage, sweep), grp in done.groupby(["task", "stage", "sweep"]):
        seeds = tuple(sorted(grp.best_test_acc.dropna().round(6).tolist()))
        if len(seeds) >= 2:
            sig_map[(task, seeds)].append((stage, sweep))
    degen = set()
    for (task, seeds), locs in sig_map.items():
        stages = set(s for s, _ in locs)
        if len(stages) >= 2:
            for s, sw in locs:
                degen.add((s, task, sw))  # (stage, task, sweep) — matches classify_run
    return degen


def load():
    df = pd.read_csv(CSV)
    for c in NUMCOLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def classify_run(r):
    """Return one of KILLED / PARTIAL / BUG / DEGEN / NORMAL for a single row.

    DEGEN = no-op (same seed signature recurs across unrelated stages); treated
    as inert (excluded from delta, counted in inert_sweeps).
    """
    key = (r.stage, r.task, r.sweep)
    if key in BUG_EXCLUDE:
        return "BUG"
    if key in DEGEN:
        return "DEGEN"
    fi = r.final_iter if pd.notna(r.final_iter) else 0
    if fi <= 0:
        return "KILLED"
    if fi < 0.5 * PLANNED.get(r.task, 0):
        return "PARTIAL"
    return "NORMAL"


def is_inert_vs_baseline(accs, baseline):
    """A group equals the baseline across seeds (idea has zero effect)."""
    accs = [a for a in accs if pd.notna(a)]
    if not accs:
        return False
    return all(abs(a - baseline) < INERT_TOL for a in accs)


def group_means_identical(means):
    """True if >=2 sweep-groups have identical mean (idea's sweep knob inert).

    Catches the sort no-op degeneracy: heads/sparsity/nst sweeps all produce
    0.9253 exactly because sort ignores those args.
    """
    means = [m for m in means if pd.notna(m)]
    if len(means) < 2:
        return False
    return all(abs(m - means[0]) < INERT_TOL for m in means)


def judge_idea_task(df, idea, task):
    """Aggregate runs for one (idea, task) and return a verdict dict.

    Delta is computed from the BEST active (non-inert) sweep-group mean, not
    the across-all-variants average — averaging confounded/low variants (e.g.
    JEPA w1.0 collapses) would mask a real positive at the stable default.
    Inert sweeps (knob not wired) are detected three ways: a group equal to
    the baseline, >=2 sweep-groups with identical means, or a DEGEN no-op
    signature (same seed sequence recurs across unrelated stages).
    """
    stage, sweeps = IDEAS[idea]
    sub = df[(df.stage == stage) & (df.task == task) & (df.sweep.isin(sweeps))].copy()
    if sub.empty:
        return {"verdict": "NO_DATA", "n_total": 0}

    sub["class"] = sub.apply(classify_run, axis=1)
    n_total = len(sub)
    n_killed = (sub["class"] == "KILLED").sum()
    n_bug = (sub["class"] == "BUG").sum()
    n_partial = (sub["class"] == "PARTIAL").sum()
    n_degen = (sub["class"] == "DEGEN").sum()
    n_normal = (sub["class"] == "NORMAL").sum()

    base = BASELINE[task]

    # All killed -> idea breaks training
    if n_normal == 0 and n_killed > 0 and n_partial == 0:
        return {"verdict": "KILLED", "n_total": n_total, "n_killed": n_killed,
                "n_normal": 0, "delta": None}

    # Only partial / degen -> can't judge a real effect
    if n_normal == 0 and (n_partial > 0 or n_degen > 0):
        if n_degen > 0 and n_partial == 0:
            return {"verdict": "INERT", "n_total": n_total, "n_normal": 0,
                    "n_degen": n_degen, "inert_sweeps": sub.sweep.unique().tolist(),
                    "delta": 0.0}
        return {"verdict": "PARTIAL_ONLY", "n_total": n_total,
                "n_partial": n_partial, "delta": None}

    normal = sub[sub["class"] == "NORMAL"]
    # DEGEN sweeps are inert no-ops — pull them into inert_sweeps too.
    degen_sweeps = sub[sub["class"] == "DEGEN"].sweep.unique().tolist()
    # per-sweep group means (normal only)
    grp_means = (normal.groupby("sweep").best_test_acc
                 .agg(["mean", "count"]).reset_index())

    inert_sweeps = list(degen_sweeps)
    # (a) a sweep-group equal to baseline
    for _, r in grp_means.iterrows():
        if is_inert_vs_baseline(normal[normal.sweep == r["sweep"]].best_test_acc.tolist(),
                                base):
            if r["sweep"] not in inert_sweeps:
                inert_sweeps.append(r["sweep"])
    # (b) degenerate: >=2 sweeps with identical means. Mark all but the
    #     representative as inert.
    active_means_rows = grp_means[~grp_means.sweep.isin(inert_sweeps)]
    if group_means_identical(active_means_rows["mean"].tolist()) and len(active_means_rows) >= 2:
        kept = active_means_rows.iloc[0]["sweep"]
        for sw in active_means_rows["sweep"]:
            if sw != kept and sw not in inert_sweeps:
                inert_sweeps.append(sw)

    active = normal[~normal.sweep.isin(inert_sweeps)]
    if active.empty:
        return {"verdict": "INERT", "n_total": n_total, "n_normal": n_normal,
                "n_degen": n_degen, "inert_sweeps": inert_sweeps,
                "delta": 0.0, "n_active": 0}

    best_row = grp_means[~grp_means.sweep.isin(inert_sweeps)].sort_values("mean").iloc[-1]
    delta = (best_row["mean"] - base) * 100
    n_active = int(active.shape[0])
    best_sweep = best_row["sweep"]
    partial_inert = len(inert_sweeps) > 0

    if delta > 1:
        verdict = "POSITIVE"
    elif delta < -1:
        verdict = "NEGATIVE"
    else:
        verdict = "NEUTRAL"

    return {"verdict": verdict, "n_total": n_total, "n_normal": n_normal,
            "n_active": n_active, "delta": delta,
            "n_killed": n_killed, "n_partial": n_partial, "n_bug": n_bug,
            "n_degen": n_degen,
            "inert_sweeps": inert_sweeps, "partial_inert": partial_inert,
            "best_sweep": best_sweep, "mean_acc": best_row["mean"]}


def judge_combo(df, combo, task):
    stage, sweeps = COMBOS[combo]
    sub = df[(df.stage == stage) & (df.task == task) & (df.sweep.isin(sweeps))].copy()
    if sub.empty:
        return {"verdict": "NO_DATA", "n_total": 0}
    sub["class"] = sub.apply(classify_run, axis=1)
    n_killed = (sub["class"] == "KILLED").sum()
    n_partial = (sub["class"] == "PARTIAL").sum()
    n_normal = (sub["class"] == "NORMAL").sum()
    base = BASELINE[task]
    if n_normal == 0 and n_killed > 0:
        return {"verdict": "KILLED", "n_total": len(sub), "n_killed": n_killed}
    if n_normal == 0:
        return {"verdict": "PARTIAL_ONLY" if n_partial else "NO_DATA",
                "n_total": len(sub), "n_partial": n_partial}
    normal = sub[sub["class"] == "NORMAL"]
    delta = (normal.best_test_acc.mean() - base) * 100
    verdict = ("POSITIVE" if delta > 1 else "NEGATIVE" if delta < -1 else "NEUTRAL")
    return {"verdict": verdict, "n_total": len(sub), "n_normal": n_normal,
            "delta": delta, "mean_acc": normal.best_test_acc.mean()}


# Verdict -> (short code, color)
VCODE = {
    "POSITIVE":     ("+",  "#2ca02c"),
    "NEGATIVE":     ("-",  "#d62728"),
    "NEUTRAL":      ("~",  "#bbbbbb"),
    "INERT":        ("0",  "#7f7f7f"),
    "KILLED":       ("X",  "#000000"),
    "PARTIAL_ONLY": ("?",  "#ff9f43"),
    "NO_DATA":      (".",  "#ffffff"),
}


def report(df):
    print("\n" + "=" * 78)
    print("IDEA VALIDITY — single-idea stages (normal data only)")
    print("=" * 78)
    print("\nNormalcy key:  + POSITIVE  - NEGATIVE  ~ NEUTRAL  "
          "0 INERT(wired-off)  X KILLED(0-iter)  ? PARTIAL_ONLY  . NO_DATA\n")

    # build matrix
    rows = []
    for idea in IDEAS:
        row = {"idea": idea}
        for task in TASKS:
            r = judge_idea_task(df, idea, task)
            row[task] = r
            rows.append({"idea": idea, "task": task, **r})
        # print per-idea line
        cells = []
        for task in TASKS:
            r = row[task]
            code = VCODE[r["verdict"]][0]
            if r["verdict"] in ("POSITIVE", "NEGATIVE", "NEUTRAL") and r.get("delta") is not None:
                bs = r.get("best_sweep", "")
                cells.append(f"{code}{r['delta']:+.1f}pp(n{r.get('n_active', 0)})[{bs}]")
            elif r["verdict"] == "INERT":
                cells.append(f"0 inert(n{r.get('n_normal', 0)})")
            elif r["verdict"] == "KILLED":
                cells.append(f"X killed({r.get('n_killed', 0)}/{r['n_total']})")
            elif r["verdict"] == "PARTIAL_ONLY":
                cells.append(f"? partial({r.get('n_partial', 0)}/{r['n_total']})")
            else:
                cells.append(".")
        print(f"  {idea:10s} " + "  ".join(f"{c:30s}" for c in cells))

    # detailed inert / partial_inert notes
    print("\n--- wiring diagnostics (inert = sweep knob not wired onto model) ---")
    for r in rows:
        if r.get("inert_sweeps"):
            all_sweeps = IDEAS[r["idea"]][1]
            full = (len(r["inert_sweeps"]) >= len(all_sweeps))
            tag = "FULLY INERT (idea not wired)" if full else "some sweeps inert"
            print(f"  {r['idea']:10s} {r['task']:8s} {tag}: {r['inert_sweeps']}")

    print("\n" + "=" * 78)
    print("COMBO ideas (st13-st15) — mostly killed on parity, collapsed on sort")
    print("=" * 78)
    for combo in COMBOS:
        cells = []
        for task in TASKS:
            r = judge_combo(df, combo, task)
            code = VCODE[r["verdict"]][0]
            if r["verdict"] in ("POSITIVE", "NEGATIVE", "NEUTRAL"):
                cells.append(f"{code}{r['delta']:+.1f}pp(n{r['n_normal']})")
            elif r["verdict"] == "KILLED":
                cells.append(f"X({r.get('n_killed', 0)}/{r['n_total']})")
            elif r["verdict"] == "PARTIAL_ONLY":
                cells.append(f"?({r.get('n_partial', 0)}/{r['n_total']})")
            else:
                cells.append(".")
        print(f"  {combo:16s} " + "  ".join(f"{t}={c:14s}" for t, c in zip(TASKS, cells)))

    return rows


def plot_validity(df):
    """Heatmap: rows = ideas, cols = tasks, color = verdict, annotation = delta."""
    ideas = list(IDEAS.keys()) + list(COMBOS.keys())
    mat_code = np.zeros((len(ideas), len(TASKS)))
    mat_delta = np.full((len(ideas), len(TASKS)), np.nan)
    mat_text = [[""] * len(TASKS) for _ in range(len(ideas))]
    for i, idea in enumerate(ideas):
        for j, task in enumerate(TASKS):
            if idea in IDEAS:
                r = judge_idea_task(df, idea, task)
            else:
                r = judge_combo(df, idea, task)
            v = r["verdict"]
            # numeric code for color: -2 kill, -1 neg, 0 inert/neutral, +1 pos, 0.5 partial
            code_map = {"POSITIVE": 2, "NEGATIVE": -2, "NEUTRAL": 0.3,
                        "INERT": 0, "KILLED": -3, "PARTIAL_ONLY": 1, "NO_DATA": 0}
            mat_code[i, j] = code_map.get(v, 0)
            if r.get("delta") is not None:
                mat_delta[i, j] = r["delta"]
            # text
            if v in ("POSITIVE", "NEGATIVE", "NEUTRAL"):
                mat_text[i][j] = f"{VCODE[v][0]}\n{r['delta']:+.1f}pp"
            elif v == "INERT":
                mat_text[i][j] = "0\n(inert)"
            elif v == "KILLED":
                mat_text[i][j] = "X\n(killed)"
            elif v == "PARTIAL_ONLY":
                mat_text[i][j] = "?\n(partial)"
            else:
                mat_text[i][j] = "."

    fig, ax = plt.subplots(figsize=(13, 0.7 * len(ideas) + 2.5))
    # use a diverging cmap; killed = darkest red, partial = orange, pos = green
    from matplotlib.colors import ListedColormap, BoundaryNorm
    cmap = ListedColormap(["#000000", "#d62728", "#bbbbbb", "#7f7f7f",
                           "#ffffff", "#ff9f43", "#2ca02c"])
    bounds = [-3.5, -2.5, -1.5, 0.15, 0.5, 1.5, 2.5, 3.5]
    norm = BoundaryNorm(bounds, cmap.N)
    im = ax.imshow(mat_code, cmap=cmap, norm=norm, aspect="auto")
    ax.set_xticks(range(len(TASKS)))
    ax.set_xticklabels(TASKS, fontsize=11, fontweight="bold")
    ax.set_yticks(range(len(ideas)))
    ax.set_yticklabels(ideas, fontsize=11)
    for i in range(len(ideas)):
        for j in range(len(TASKS)):
            color = "white" if mat_code[i, j] in (-3, -2, 2) else "black"
            ax.text(j, i, mat_text[i][j], ha="center", va="center",
                    fontsize=8.5, color=color, fontweight="bold")
    ax.set_title("Idea validity per task (0629 harvest, normal data only)\n"
                 "+ positive   - negative   ~ neutral   0 inert(not wired)   "
                 "X killed(0-iter)   ? partial-only",
                 fontsize=11, fontweight="bold")
    legend_handles = [
        Patch(facecolor="#2ca02c", label="+ POSITIVE"),
        Patch(facecolor="#d62728", label="- NEGATIVE"),
        Patch(facecolor="#bbbbbb", label="~ NEUTRAL"),
        Patch(facecolor="#7f7f7f", label="0 INERT (idea not wired)"),
        Patch(facecolor="#000000", label="X KILLED (training never started)"),
        Patch(facecolor="#ff9f43", label="? PARTIAL_ONLY"),
    ]
    ax.legend(handles=legend_handles, bbox_to_anchor=(1.02, 1), loc="upper left",
              fontsize=8.5, frameon=False)
    fig.tight_layout(rect=[0, 0, 0.78, 0.93])
    p = FIG_DIR / "figC_idea_validity.png"
    fig.savefig(p, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[fig] -> {p}")


def main():
    global DEGEN
    df = load()
    DEGEN = compute_degenerate(df)
    print(f"loaded {len(df)} runs. DEGEN no-op signatures detected: {len(DEGEN)}")
    if DEGEN:
        for t, s, sw in sorted(DEGEN):
            print(f"  DEGEN  {t:6s} {s:5s} {sw}")
    print()
    rows = report(df)
    plot_validity(df)
    # final tally
    print("\n" + "=" * 78)
    print("TALLY of verdicts (single-idea stages, 6 ideas x 5 tasks = 30 cells)")
    print("=" * 78)
    from collections import Counter
    c = Counter(r["verdict"] for r in rows)
    for v in ["POSITIVE", "NEGATIVE", "NEUTRAL", "INERT", "KILLED",
              "PARTIAL_ONLY", "NO_DATA"]:
        print(f"  {v:14s} {c.get(v, 0):2d}")


if __name__ == "__main__":
    main()
