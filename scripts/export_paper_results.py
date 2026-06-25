#!/usr/bin/env python3
"""Export paper-task experiment results (handles PARTIAL runs) -> CSV.

Reads logs/deep/{LOG_ROOT}/*/checkpoint*.pt and emits a CSV consumable by
exp_runner.collect_csv() on the dev machine.

Handles partial runs: if the final checkpoint.pt is missing, falls back to the
highest-numbered intermediate checkpoint_N.pt and records state + progress so
you can tell which experiments trained enough to be meaningful.

Usage (on the compute machine):
    python scripts/export_paper_results.py --log-root logs/deep/01_revise \\
        --output csv_data/01_revise_results.csv

Then manually copy the CSV to the dev machine's csv_data/ dir and load with:
    from exp_runner import collect_csv
    df = collect_csv('csv_data/01_revise_results.csv')
"""

import argparse
import csv
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "paper"))

from exp_runner import _find_latest_ckpt, BASELINE_ACC  # noqa: E402

ACC_KEYS = [
    "test_accuracies_full_list",
    "test_accuracies",
    "val_accuracies",
    "accuracy",
    "test_accuracies_most_certain",
]


def _task_from_name(name: str) -> str:
    return "cifar10" if name.startswith("cifar10") else name.split("_")[0]


def extract(log_dir: Path):
    final = log_dir / "checkpoint.pt"
    ck = _find_latest_ckpt(log_dir)
    if ck is None:
        return None
    state = "done" if final.exists() else "partial"
    try:
        d = torch.load(ck, map_location="cpu", weights_only=False)
    except Exception as e:
        return {"name": log_dir.name, "task": _task_from_name(log_dir.name),
                "error": str(e)[:120]}

    metric, best_acc, last_acc = "", None, None
    for k in ACC_KEYS:
        v = d.get(k) if isinstance(d, dict) else None
        if isinstance(v, (list, tuple)) and v:
            nums = [x for x in v if isinstance(x, (int, float))]
            if nums:
                metric, best_acc, last_acc = k, max(nums), nums[-1]
                break

    iters = d.get("iters", []) if isinstance(d, dict) else []
    n = len(iters) if iters else 0
    return {
        "name": log_dir.name,
        "task": _task_from_name(log_dir.name),
        "metric": metric,
        "best_acc": best_acc,
        "last_acc": last_acc,
        "n_points": n,
        "final_iter": iters[-1] if iters else 0,
        "state": state,
        "ckpt": ck.name,
        "baseline": BASELINE_ACC.get(_task_from_name(log_dir.name), ""),
        "error": "",
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--log-root", default=str(ROOT / "logs" / "deep" / "01_revise"))
    ap.add_argument("--output", default=str(ROOT / "csv_data" / "01_revise_results.csv"))
    cli = ap.parse_args()

    log_root = Path(cli.log_root)
    if not log_root.is_absolute():
        log_root = ROOT / cli.log_root
    out = Path(cli.output)
    if not out.is_absolute():
        out = ROOT / cli.output
    out.parent.mkdir(parents=True, exist_ok=True)

    if not log_root.exists():
        print(f"No such dir: {log_root}")
        return

    rows = []
    for sub in sorted(log_root.iterdir()):
        if not sub.is_dir():
            continue
        r = extract(sub)
        if r:
            rows.append(r)

    if not rows:
        print(f"No checkpoints found under {log_root}")
        return

    fields = ["name", "task", "metric", "best_acc", "last_acc",
              "n_points", "final_iter", "state", "ckpt", "baseline", "error"]
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)

    done = sum(1 for r in rows if r["state"] == "done")
    partial = sum(1 for r in rows if r["state"] == "partial")
    print(f"\nExported {len(rows)} runs ({done} done, {partial} partial) -> {out}\n")
    print(f"{'name':<42} {'task':<8} {'state':<8} {'final_iter':>10} "
          f"{'best_acc':>9} {'baseline':>9} {'delta':>7}")
    print("-" * 98)
    for r in sorted(rows, key=lambda x: (x["task"], x["name"])):
        acc = r["best_acc"] * 100 if isinstance(r["best_acc"], (int, float)) else float("nan")
        bl = r["baseline"]
        bl_pct = bl * 100 if isinstance(bl, (int, float)) else float("nan")
        delta = (acc - bl_pct) if r["best_acc"] is not None and isinstance(bl, (int, float)) else float("nan")
        print(f"{r['name']:<42} {r['task']:<8} {r['state']:<8} {str(r['final_iter']):>10} "
              f"{acc:>8.2f}% {bl_pct:>8.2f}% {delta:>+6.2f}pp")


if __name__ == "__main__":
    main()
