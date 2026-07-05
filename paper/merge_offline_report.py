#!/usr/bin/env python
"""Merge multi-GPU CSV shards into a unified markdown report.

Usage:
  python paper/merge_offline_report.py --csv csv_data/offline_compare_full.csv
"""
import argparse
import csv
from collections import defaultdict

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', required=True)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    rows = []
    with open(args.csv) as f:
        for r in csv.DictReader(f):
            r['success_rate'] = float(r['success_rate'])
            r['expert_success'] = float(r.get('expert_success', 0))
            r['seed'] = int(r['seed'])
            rows.append(r)

    out_path = args.out or args.csv.replace('.csv', '_report.md')
    envs = sorted({r['env'] for r in rows})
    backs = sorted({r['backbone'] for r in rows if r['backbone'] not in ('mlp-random',)})
    by = defaultdict(list)
    expert_by = {}
    for r in rows:
        if r['backbone'] == 'mlp-random':
            continue
        by[(r['env'], r['backbone'])].append(r['success_rate'])
        if r['env'] not in expert_by:
            expert_by[r['env']] = r['expert_success']

    lines = []
    lines.append("# Offline GCBC comparison (merged from multi-GPU shards)\n")
    lines.append(f"Runs: {len(rows)} | envs: {len(envs)} | backbones: {backs}\n")

    # expert quality
    lines.append("\n## Expert data quality\n")
    lines.append("| env | expert_succ |")
    lines.append("|---|---|")
    for e in envs:
        lines.append(f"| {e} | {expert_by.get(e, 0):.1f}% |")

    # success rate
    lines.append("\n## success_rate mean+-std\n")
    lines.append("| env | " + " | ".join(backs) + " |")
    lines.append("|---|" + "|".join(["---"] * len(backs)) + "|")
    for e in envs:
        cells = []
        for b in backs:
            vals = by.get((e, b), [])
            if vals:
                cells.append(f"{np.mean(vals):.1f}+-{np.std(vals):.1f}")
            else:
                cells.append("-")
        lines.append(f"| {e} | " + " | ".join(cells) + " |")

    # Flash synergy
    if 'flash' in backs and 'flash-shallow' in backs and 'flash-deep' in backs:
        lines.append("\n## Flash 混合 vs 单路径\n")
        lines.append("| env | flash | shallow | deep | mix-shallow | mix-deep |")
        lines.append("|---|---|---|---|---|---|")
        for e in envs:
            fm = float(np.mean(by.get((e, 'flash'), [0])))
            sm = float(np.mean(by.get((e, 'flash-shallow'), [0])))
            dm = float(np.mean(by.get((e, 'flash-deep'), [0])))
            lines.append(f"| {e} | {fm:.1f} | {sm:.1f} | {dm:.1f} | "
                         f"{fm-sm:+.1f} | {fm-dm:+.1f} |")

    # CTM vs RNN
    rnn_backs = [b for b in ['lstm', 'gru', 'transformer'] if b in backs]
    if 'ctm' in backs and rnn_backs:
        lines.append("\n## CTM vs RNN 系记忆策略\n")
        lines.append("| env | CTM | RNN 均值 | CTM-RNN | 判定 |")
        lines.append("|---|---|---|---|---|")
        for e in envs:
            ctm = float(np.mean(by.get((e, 'ctm'), [0])))
            rnn_m = float(np.mean([np.mean(by.get((e, b), [0])) for b in rnn_backs]))
            diff = ctm - rnn_m
            verdict = "持平" if abs(diff) < 3 else ("CTM 赢" if diff > 0 else "CTM 输")
            lines.append(f"| {e} | {ctm:.1f} | {rnn_m:.1f} | {diff:+.1f} | {verdict} |")

    # partial vs full comparison
    partial_envs = [e for e in envs if '-partial' in e]
    full_envs = [e.replace('-partial', '') for e in partial_envs
                 if e.replace('-partial', '') in envs]
    if full_envs:
        lines.append("\n## Partial vs Full obs (记忆 backbone 的 POMDP 优势)\n")
        lines.append("| env | mlp_full | mlp_partial | Δmlp | ctm_full | ctm_partial | Δctm |")
        lines.append("|---|---|---|---|---|---|---|")
        for base in full_envs:
            part = base + '-partial'
            mlpf = float(np.mean(by.get((base, 'mlp'), [0])))
            mlpp = float(np.mean(by.get((part, 'mlp'), [0])))
            ctmf = float(np.mean(by.get((base, 'ctm'), [0])))
            ctmp = float(np.mean(by.get((part, 'ctm'), [0])))
            lines.append(f"| {base} | {mlpf:.1f} | {mlpp:.1f} | {mlpp-mlpf:+.1f} | "
                         f"{ctmf:.1f} | {ctmp:.1f} | {ctmp-ctmf:+.1f} |")

    report = "\n".join(lines)
    with open(out_path, 'w') as f:
        f.write(report)
    print(f"Report: {out_path}")
    print(report)


if __name__ == '__main__':
    main()
