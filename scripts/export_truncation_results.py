#!/usr/bin/env python3
"""Export truncated-BPTT experiment results → Pareto frontier CSV.

Reads logs/trunc/*/checkpoint.pt and emits csv_data/truncation_results.csv with
the memory/accuracy tradeoff across bp_steps — the core deliverable of the
truncation-as-first-principle line of work.

Usage:
    python scripts/export_truncation_results.py
    python scripts/export_truncation_results.py --log-root logs/trunc --output csv_data/truncation_results.csv
"""

import argparse, csv, glob, os
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]


def extract(log_dir):
    ckpt_path = os.path.join(log_dir, "checkpoint.pt")
    if not os.path.exists(ckpt_path):
        return None
    try:
        c = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    except Exception as e:
        return {"name": os.path.basename(log_dir), "error": str(e)}

    ta = c.get("test_accuracies", [])
    taf = c.get("test_accuracies_full_list", [])
    tl = c.get("test_losses", [])
    iters = c.get("iters", [])
    a = c.get("args", None)

    def g(attr, default=""):
        return getattr(a, attr, default) if a is not None else default

    return {
        "name": os.path.basename(log_dir),
        "sort_loss_mode": c.get("sort_loss_mode", g("sort_loss_mode")),
        "bp_steps": c.get("bp_steps", g("bp_steps", 0)),
        "N_to_sort": g("N_to_sort", ""),
        "best_test_acc": max(ta) if ta else 0,
        "final_test_acc": ta[-1] if ta else 0,
        "best_test_acc_full": max(taf) if taf else 0,
        "final_test_acc_full": taf[-1] if taf else 0,
        "final_test_loss": tl[-1] if tl else 0,
        "peak_memory_gb": c.get("peak_memory_gb", 0),
        "num_track_points": len(ta),
        "last_iter": iters[-1] if iters else 0,
        "test_acc_curve": str([round(x, 4) for x in ta]) if ta else "",
        "test_loss_curve": str([round(x, 4) for x in tl]) if tl else "",
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--log-root", default=str(ROOT / "logs" / "trunc"))
    ap.add_argument("--output", default=str(ROOT / "csv_data" / "truncation_results.csv"))
    cli = ap.parse_args()

    os.makedirs(os.path.dirname(cli.output) or ".", exist_ok=True)
    results = []
    for sub in sorted(glob.glob(os.path.join(cli.log_root, "*"))):
        if os.path.isdir(sub):
            r = extract(sub)
            if r and "error" not in r:
                results.append(r)

    if not results:
        print(f"No checkpoints found under {cli.log_root}")
        return

    results.sort(key=lambda x: (str(x["sort_loss_mode"]), int(x["bp_steps"])))
    fields = list(results[0].keys())
    with open(cli.output, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in results:
            w.writerow(r)

    print(f"\nExported {len(results)} runs -> {cli.output}\n")
    print(f"{'name':<16} {'mode':<18} {'bp':>4} {'best_acc':>9} {'full_acc':>9} {'mem_GB':>7} {'loss':>7}")
    print("-" * 78)
    for r in results:
        print(f"{r['name']:<16} {str(r['sort_loss_mode']):<18} {str(r['bp_steps']):>4} "
              f"{r['best_test_acc']*100:>8.2f}% {r['best_test_acc_full']*100:>8.2f}% "
              f"{r['peak_memory_gb']:>6.2f} {r['final_test_loss']:>7.4f}")

    # Highlight the Pareto frontier (max accuracy at each bp_steps for sinkhorn)
    sk = [r for r in results if r["sort_loss_mode"] == "per_tick_sinkhorn"]
    if sk:
        print("\nPareto (per_tick_sinkhorn): smallest bp_steps holding best acc ≈ full-BPTT acc")
        full = next((r for r in sk if int(r["bp_steps"]) >= 50), sk[-1])
        print(f"  full BPTT (bp={full['bp_steps']}): {full['best_test_acc']*100:.2f}% @ {full['peak_memory_gb']:.2f}GB")
        for r in sk:
            ratio = r["peak_memory_gb"] / full["peak_memory_gb"] if full["peak_memory_gb"] else 0
            hold = "HOLDS" if r["best_test_acc"] >= 0.95 * full["best_test_acc"] else "drops"
            print(f"  bp={str(r['bp_steps']):>3}: {r['best_test_acc']*100:6.2f}% @ {r['peak_memory_gb']:5.2f}GB "
                  f"({ratio*100:4.0f}% mem)  {hold}")


if __name__ == "__main__":
    main()
