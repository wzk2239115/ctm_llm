#!/usr/bin/env python3
"""st25 Adaptive-JEPA-weight analysis (mc-vs-mc 口径).

st25 问的问题: st04 发现 cross_tick_jepa_weight=0.1 会把 parity 主任务压制到
chance ("光顾着约束没学数据"). st25 测 3 种自适应权重方案能否解除压制:
  fixed       : 静态 w=0.1 (对照, 应重现压制)
  balance (A) : 损失幅度平衡  eff_w = ratio*L_main/(L_jepa+eps)
  gate    (B) : acc 门控 sigmoid  主任务没学会->JEPA 关
  uncertainty(C): 可学习 sigma (Kendall 2018)

口径:
  - 主对比 = within-st25: fixed 是 control, 其余 3 模式 vs fixed (4 模式共享
    完全相同 base 配置, 只差权重方案 -> 最干净的变量隔离).
  - 次对比 = vs st00 paper baseline (无 JEPA): st25 各 task 配置已核对 == st00
    paper sweep, 公平.
  - cifar10/mazes/parity 用 best_test_acc_mc (mc-vs-mc); sort 无 mc, 用
    best_test_acc (element-level, 同 analyze_deep.py).
  - BROKEN 判定: acc 在 chance 附近 (parity<0.55, sort<0.1, cifar10 final<0.15).

用法:  python paper/analyze_st25.py [--plot]
读 csv_data/st25_0701_summary.csv, 打印表 + 结论; --plot 另出
runs/figures/ctm_paper/figF_st25_adaptive_jepa.png.
"""
from __future__ import annotations
import argparse
import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "csv_data"
ST25 = DATA / "st25_0701_summary.csv"
FIG_DIR = ROOT / "runs" / "figures" / "ctm_paper"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# st00 paper-sweep 复现值 (与 analyze_deep.py / sparsity_efficiency.py 一致)
MC_BASELINE = {
    "cifar10": 0.8516,   # st00 cifar10 paper best_test_acc_mc
    "mazes":   0.9117,   # st00 mazes paper best_test_acc_mc
    "parity":  0.8821,   # st00 parity paper best_test_acc_mc (=0.88207)
    "sort":    0.8753,   # st00 sort best_test_acc (element-level, no mc)
}
TASKS = ["cifar10", "mazes", "parity", "sort"]
MODES = ["fixed", "balance", "gate", "uncertainty"]
MODE_DESC = {
    "fixed": "static w=0.1 (control)",
    "balance": "A: loss-magnitude balance",
    "gate": "B: acc-gated sigmoid",
    "uncertainty": "C: learnable sigma (Kendall)",
}
# chance / broken thresholds
CHANCE = {"cifar10": 0.15, "mazes": 0.0, "parity": 0.55, "sort": 0.10}


def best_acc_col(task: str) -> str:
    return "best_test_acc" if task == "sort" else "best_test_acc_mc"


def agg(df: pd.DataFrame, task: str, mode: str) -> dict:
    col = best_acc_col(task)
    sub = df[(df.task == task) & (df.sweep == mode) & df[col].notna()]
    if len(sub) == 0:
        return dict(n=0, mean=float("nan"), std=float("nan"), accs=[])
    accs = sorted(sub[col].tolist())
    return dict(n=len(sub), mean=sub[col].mean(), std=sub[col].std(ddof=0), accs=accs)


def main():
    df = pd.read_csv(ST25)
    for c in ["best_test_acc", "best_test_acc_mc", "final_test_acc"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    print("=" * 78)
    print("st25 Adaptive-JEPA-weight  (mc-vs-mc; sort=element-level)")
    print("base config == st00 paper sweep for all 4 tasks (verified)")
    print("=" * 78)

    # ── per-task × mode table ──
    rows = []
    for t in TASKS:
        base = MC_BASELINE[t]
        fixed_agg = agg(df, t, "fixed")
        for m in MODES:
            a = agg(df, t, m)
            if a["n"] == 0:
                continue
            d_st00 = (a["mean"] - base) * 100
            d_fixed = (a["mean"] - fixed_agg["mean"]) * 100 if fixed_agg["n"] else float("nan")
            broken = a["mean"] < CHANCE[t]
            rows.append(dict(task=t, mode=m, n=a["n"],
                             mean=a["mean"], std=a["std"],
                             delta_st00=d_st00, delta_fixed=d_fixed,
                             broken=broken, accs=a["accs"]))

    rdf = pd.DataFrame(rows)
    pct = lambda x: f"{x*100:6.2f}"
    dpp = lambda x: ("  --- " if x != x else f"{x:+6.2f}")
    brk = lambda b: ("  BROKEN" if b else "")
    print(f"\n{'task':8s} {'mode':12s} {'n':>2} {'acc%':>7} {'±std':>7}"
          f" {'vsFIXED':>8} {'vsST00':>8}  seeds{brk(False)}")
    print("-" * 78)
    for _, r in rdf.iterrows():
        seeds = "[" + ",".join(f"{a*100:.1f}" for a in r["accs"]) + "]"
        print(f"{r['task']:8s} {r['mode']:12s} {r['n']:>2} {pct(r['mean'])} {pct(r['std']):>7}"
              f" {dpp(r['delta_fixed']):>8} {dpp(r['delta_st00']):>8}  {seeds}{brk(r['broken'])}")

    # ── baseline reference ──
    print("\nst00 paper baseline (mc):")
    for t in TASKS:
        col = best_acc_col(t)
        print(f"  {t:8s} {MC_BASELINE[t]*100:6.2f}%  ({col})")

    # ── conclusions ──
    print("\n" + "=" * 78)
    print("CONCLUSIONS")
    print("=" * 78)
    conclusions = []
    for t in TASKS:
        g = rdf[rdf.task == t]
        if g.empty:
            continue
        best = g.loc[g["mean"].idxmax()]
        worst = g.loc[g["mean"].idxmin()]
        fixed = g[g["mode"] == "fixed"]
        gate = g[g["mode"] == "gate"]
        fb = float(fixed["mean"].iloc[0]) if len(fixed) else float("nan")
        gb = float(gate["mean"].iloc[0]) if len(gate) else float("nan")
        broken_modes = g[g["broken"]]["mode"].tolist()
        parts = [f"[{t}] best={best['mode']}({best['mean']*100:.1f}%)"]
        if broken_modes:
            parts.append(f"BROKEN={broken_modes}")
        if not pd.isna(gb) and not pd.isna(fb):
            parts.append(f"gate-fixed={(gb-fb)*100:+.1f}pp")
        if not pd.isna(gb):
            parts.append(f"gate-vs-st00={(gb-MC_BASELINE[t])*100:+.1f}pp")
        conclusions.append("  " + " | ".join(p for p in parts if p))
    print("\n".join(conclusions))

    # ── headline ──
    print("\nHEADLINE:")
    g = rdf[(rdf["task"] == "parity") & (rdf["mode"] == "gate")]
    f0 = rdf[(rdf["task"] == "parity") & (rdf["mode"] == "fixed")]
    if len(g) and len(f0):
        print(f"  parity (the suppression test case):")
        print(f"    fixed       mc = {f0['mean'].iloc[0]*100:.2f}% "
              f"(seeds {[round(a*100,1) for a in f0['accs'].iloc[0]]})")
        print(f"    gate        mc = {g['mean'].iloc[0]*100:.2f}% "
              f"(seeds {[round(a*100,1) for a in g['accs'].iloc[0]]})")
        print(f"    st00(base)  mc = {MC_BASELINE['parity']*100:.2f}%")
        print(f"    -> gate vs fixed = {(g['mean'].iloc[0]-f0['mean'].iloc[0])*100:+.2f}pp")
        print(f"    -> gate vs st00  = {(g['mean'].iloc[0]-MC_BASELINE['parity'])*100:+.2f}pp")
    s_g = rdf[(rdf["task"] == "sort") & (rdf["mode"] == "gate")]
    s_f = rdf[(rdf["task"] == "sort") & (rdf["mode"] == "fixed")]
    if len(s_g) and len(s_f):
        print(f"  sort (element-level):")
        print(f"    fixed = {s_f['mean'].iloc[0]*100:.2f}%  gate = {s_g['mean'].iloc[0]*100:.2f}%  "
              f"st00 = {MC_BASELINE['sort']*100:.2f}%")
        print(f"    -> gate vs fixed = {(s_g['mean'].iloc[0]-s_f['mean'].iloc[0])*100:+.2f}pp")


def plot(rdf: pd.DataFrame):
    """Grouped bar: 4 tasks × 4 modes, with seed scatter + baseline + BROKEN."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # only plot task×mode combos that exist
    present = [(t, m) for t in TASKS for m in MODES
               if not rdf[(rdf["task"] == t) & (rdf["mode"] == m)].empty]
    tasks_present = [t for t in TASKS if any(tt == t for tt, _ in present)]

    color = {"fixed": "#999999", "balance": "#1f77b4",
             "gate": "#2ca02c", "uncertainty": "#d62728"}
    label = {"fixed": "fixed (control)", "balance": "A: balance",
             "gate": "B: gate", "uncertainty": "C: uncertainty"}

    n_t = len(tasks_present)
    n_m = len(MODES)
    bw = 0.8 / n_m  # bar width per mode
    x = np.arange(n_t)

    fig, ax = plt.subplots(figsize=(12, 6.8))
    rng = np.random.default_rng(7)
    for j, m in enumerate(MODES):
        means, stds, pos = [], [], []
        seeds_all = []
        for i, t in enumerate(tasks_present):
            row = rdf[(rdf["task"] == t) & (rdf["mode"] == m)]
            if row.empty:
                means.append(np.nan); stds.append(0); seeds_all.append([])
                continue
            r = row.iloc[0]
            means.append(r["mean"] * 100); stds.append((r["std"] or 0) * 100)
            seeds_all.append([a * 100 for a in r["accs"]])
            pos.append(i + (j - (n_m - 1) / 2) * bw)
        means = np.array(means, dtype=float)
        pos_full = [i + (j - (n_m - 1) / 2) * bw for i in range(n_t)]
        broken = np.array([rdf[(rdf["task"] == t) & (rdf["mode"] == m)].iloc[0]["broken"]
                           if not rdf[(rdf["task"] == t) & (rdf["mode"] == m)].empty else False
                           for t in tasks_present])
        bars = ax.bar(pos_full, np.nan_to_num(means), width=bw * 0.92,
                      yerr=np.nan_to_num(stds), capsize=3,
                      color=color[m], label=label[m],
                      alpha=0.45 if False else 1.0,
                      edgecolor="black", linewidth=0.6,
                      hatch="xx" if any(broken) else None,
                      zorder=3)
        # dim the broken bars individually via overlay
        for bi, brk in enumerate(broken):
            if brk and not np.isnan(means[bi]):
                ax.bar([pos_full[bi]], [means[bi]], width=bw * 0.92,
                       color=color[m], alpha=0.30, edgecolor="#d62728",
                       linewidth=1.4, hatch="xx", zorder=3.5)
        # seed scatter (jittered dots)
        for i, accs in enumerate(seeds_all):
            if not accs:
                continue
            jit = rng.normal(0, bw * 0.10, len(accs))
            ax.scatter([pos_full[i]] * len(accs), accs,
                       facecolor="white", edgecolor="black",
                       s=26, zorder=5, linewidth=0.7)
            # n seed label
            ax.annotate(f"n={len(accs)}", (pos_full[i], means[i]),
                        textcoords="offset points", xytext=(0, -14),
                        ha="center", fontsize=6.5, color="#333333")
        # value labels (skip broken to reduce clutter — they're hatched red)
        for i, (p, accs) in enumerate(zip(pos_full, seeds_all)):
            if np.isnan(means[i]):
                continue
            brk = rdf[(rdf["task"] == tasks_present[i]) & (rdf["mode"] == m)]
            is_broken = bool(brk.iloc[0]["broken"]) if len(brk) else False
            if is_broken:
                ax.annotate(f"{means[i]:.0f}%", (p, means[i]),
                            textcoords="offset points", xytext=(0, 6),
                            ha="center", fontsize=7, color="#d62728",
                            fontweight="bold")
            else:
                ax.annotate(f"{means[i]:.1f}", (p, means[i]),
                            textcoords="offset points", xytext=(0, 6),
                            ha="center", fontsize=7, color="#222222")

    # baseline (st00) line per task
    for i, t in enumerate(tasks_present):
        base = MC_BASELINE[t] * 100
        ax.hlines(base, x[i] - 0.5, x[i] + 0.5, colors="black",
                  linestyles="--", linewidth=1.8, zorder=6)
        ax.annotate(f"st00\n{base:.1f}%", (x[i] + 0.5, base),
                    textcoords="offset points", xytext=(4, -2),
                    fontsize=7, color="black", va="center")

    ax.set_xticks(x)
    ax.set_xticklabels([f"{t}\n({'mc' if t!='sort' else 'element'})"
                        for t in tasks_present], fontsize=10)
    ax.set_ylabel("accuracy (%)", fontsize=11)
    ax.set_title("st25 — Adaptive JEPA weight schemes vs fixed control\n"
                 "gate (B, acc-gated) is the only scheme that works: "
                 "parity +9.5pp / sort +7.3pp vs fixed;\n"
                 "balance & uncertainty collapse on parity/sort (hatched = BROKEN, "
                 "○ = seed)", fontsize=11.5, fontweight="bold")
    ax.set_ylim(0, 105)
    ax.grid(True, axis="y", alpha=0.3, zorder=0)
    ax.legend(loc="upper right", fontsize=9, ncol=4, framealpha=0.95)
    fig.tight_layout()
    out = FIG_DIR / "figF_st25_adaptive_jepa.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[fig] -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plot", action="store_true", help="also save figure")
    cli = ap.parse_args()
    main()
    if cli.plot:
        # rebuild rdf for plotting (same logic as main's table)
        df = pd.read_csv(ST25)
        for c in ["best_test_acc", "best_test_acc_mc", "final_test_acc"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        rows = []
        for t in TASKS:
            fixed_agg = agg(df, t, "fixed")
            for m in MODES:
                a = agg(df, t, m)
                if a["n"] == 0:
                    continue
                broken = a["mean"] < CHANCE[t]
                rows.append(dict(task=t, mode=m, n=a["n"], mean=a["mean"],
                                 std=a["std"], broken=broken, accs=a["accs"]))
        plot(pd.DataFrame(rows))
