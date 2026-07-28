#!/usr/bin/env python
"""Extract per-thought-tick test accuracy from checkpoints (signature figure).

CTM evaluates accuracy at EACH internal tick (0..n_ticks-1) during test; the
training loop stores these per-tick arrays in checkpoint['test_accuracies'].
The last entry = the final model's per-tick accuracy curve. Combined with
checkpoint['test_accuracies_most_certain'][-1] (the most-certain-tick scalar),
this yields the signature plot: accuracy vs thought-tick + the mc ceiling.

No GPU / no forward pass — reads saved arrays directly. Run on the compute
machine (checkpoints live there), then copy the json to dev for plotting.

Storage per task (last dim of test_accuracies[-1] = ticks):
  cifar10/qamnist/parity: (T,)            -> per-tick acc, used directly
  mazes:                  (S, T)          -> mean over route-steps S -> (T,)
  sort:                   scalar          -> skipped (CTC, no per-tick)
"""
import sys
import json
import argparse
from pathlib import Path

import numpy as np
import torch


def extract_one(ckpt_path):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    te = ck.get("test_accuracies", [])
    mc = ck.get("test_accuracies_most_certain", [])
    if not te:
        return None
    last = np.asarray(te[-1], dtype=float)
    if last.ndim == 0:
        return None  # scalar (sort) — no per-tick profile
    while last.ndim > 1:  # mazes (S,T) -> (T,) by averaging over steps
        last = last.mean(axis=0)
    mc_last = float(np.asarray(mc[-1])) if mc else None
    return {
        "iteration": int(ck.get("iteration", 0)),
        "n_ticks": int(last.shape[0]),
        "acc_per_tick": last.tolist(),
        "mc_acc": mc_last,
        "final_acc": float(last[-1]),
        "best_tick_acc": float(last.max()),
        "best_tick": int(last.argmax()),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--logs", default="paper_repro/logs")
    ap.add_argument("--out", default="paper_repro/csv_data/per_tick_0728.json")
    ap.add_argument("--groups", nargs="*", default=["baseline", "jepa", "revise"])
    ap.add_argument("--tasks", nargs="*",
                    default=["cifar10", "qamnist", "parity", "mazes"])
    args = ap.parse_args()

    logs = Path(args.logs)
    out = {}
    n_ok = n_skip = 0
    for group in args.groups:
        gdir = logs / group
        if not gdir.exists():
            continue
        for task in args.tasks:
            for d in sorted(gdir.glob(f"{task}_*_s*")):
                ckpt = d / "checkpoint.pt"
                if not ckpt.exists():
                    continue
                try:
                    r = extract_one(ckpt)
                except Exception as e:
                    print(f"  ERR {group}/{d.name}: {e}")
                    n_skip += 1
                    continue
                if r is None:
                    n_skip += 1
                    continue
                out[f"{group}/{d.name}"] = r
                n_ok += 1

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"\nextracted {n_ok} runs ({n_skip} skipped) -> {args.out}\n")

    # summary: per (group, task) mean curves
    for group in args.groups:
        for task in args.tasks:
            runs = [v for k, v in out.items()
                    if k.startswith(f"{group}/") and f"/{task}_" in k]
            if not runs:
                continue
            curves = np.array([v["acc_per_tick"] for v in runs])
            mcs = [v["mc_acc"] for v in runs if v["mc_acc"] is not None]
            nt = curves.shape[1]
            fin = curves[:, -1].mean() * 100
            best = curves.max(axis=1).mean() * 100
            mcm = np.mean(mcs) * 100 if mcs else float("nan")
            print(f"  {group:8s} {task:8s} n={len(runs)} ticks={nt}  "
                  f"final={fin:5.1f}%  best-tick={best:5.1f}%  mc={mcm:5.1f}%")


if __name__ == "__main__":
    main()
