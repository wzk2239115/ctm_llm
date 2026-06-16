#!/usr/bin/env python3
"""Collect final test accuracies from baseline task checkpoints into a table.

Usage:
    python scripts/collect_results.py                                    # print table & save CSV
    python scripts/collect_results.py --log-dir logs/ctm_scaling         # custom path
    python scripts/collect_results.py --log-dir logs/ctm_paper --csv only
    python scripts/collect_results.py --log-dir logs/ctm_scaling --fail-only
"""

import argparse, csv, glob, os, re, sys, torch


def find_checkpoints(log_dir):
    for ckpt in sorted(glob.glob(os.path.join(log_dir, "**/checkpoint*.pt"), recursive=True)):
        yield ckpt


def extract_name_and_key(ckpt_path, log_dir):
    rel = os.path.relpath(ckpt_path, log_dir)
    parts = rel.split(os.sep)
    name = parts[-2]  # dir containing the checkpoint
    return name, name


TRACKING_KEYS = [
    "test_accuracies_full_list",
    "test_accuracies",
    "val_accuracies",
    "accuracy",
]


def extract_accuracies(checkpoint):
    d = checkpoint if isinstance(checkpoint, dict) else {}
    for key in TRACKING_KEYS:
        vals = d.get(key, [])
        if vals and isinstance(vals, (list, tuple)) and len(vals) > 0:
            clean = [v for v in vals if isinstance(v, (int, float))]
            if clean:
                return clean[-1], len(clean)
    # nested dicts (parity saves accuracy_means inside)
    for v in d.values():
        if isinstance(v, dict):
            for key2 in TRACKING_KEYS:
                vals = v.get(key2, [])
                if vals and isinstance(vals, (list, tuple)) and len(vals) > 0:
                    clean = [x for x in vals if isinstance(x, (int, float))]
                    if clean:
                        return clean[-1], len(clean)
    return None, 0


def main():
    parser = argparse.ArgumentParser(description="Collect baseline task results")
    parser.add_argument("--log-dir", default="logs/ctm_scaling",
                        help="Root dir containing experiment subdirs with checkpoints")
    parser.add_argument("--csv", default=None,
                        help="Output CSV path (default: runs/metrics/{log_dir_slug}_results.csv)")
    parser.add_argument("--fail-only", action="store_true",
                        help="Only show missing/failed experiments")
    parser.add_argument("--nonzero", action="store_true",
                        help="Only show experiments with accuracy > 0")
    args = parser.parse_args()
    if args.csv is None:
        slug = args.log_dir.strip("/").replace("/", "_")
        args.csv = f"runs/metrics/{slug}_results.csv"

    results = []
    missing = []

    seen = set()
    for ckpt_path in find_checkpoints(args.log_dir):
        exp_name, key = extract_name_and_key(ckpt_path, args.log_dir)
        if key in seen:
            continue
        seen.add(key)

        try:
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        except Exception as e:
            missing.append((exp_name, f"load_error: {e}"))
            continue

        acc, n = extract_accuracies(ckpt)
        if acc is None:
            missing.append((exp_name, "no_accuracy_tracking"))
        else:
            results.append((exp_name, acc, n))

    results.sort(key=lambda x: (x[0].split("_")[0] if "_" in x[0] else "", x[0]))
    missing.sort(key=lambda x: x[0])

    print(f"\n{'='*70}")
    print(f"  Results from: {args.log_dir}")
    print(f"  Completed: {len(results)}  |  Missing/Failed: {len(missing)}")
    print(f"{'='*70}")
    print(f"  {'Experiment':45s} {'Final Acc':>10s}  {'Checkpts':>7s}")
    print(f"  {'-'*45}  {'-'*10}  {'-'*7}")
    for exp_name, acc, n in results:
        if args.nonzero and acc == 0.0:
            continue
        print(f"  {exp_name:45s} {acc:>10.4f}  {n:>7d}")
    if missing and not args.fail_only:
        print(f"\n  {'─'*70}")
        print(f"  Missing / Failed:")
        for exp_name, reason in missing:
            print(f"    {exp_name:45s} {reason}")

    os.makedirs(os.path.dirname(args.csv) or ".", exist_ok=True)
    with open(args.csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["experiment", "final_test_accuracy", "checkpoints_recorded"])
        w.writerows(results)
    print(f"\n  CSV saved: {args.csv}")
    print()


if __name__ == "__main__":
    main()
