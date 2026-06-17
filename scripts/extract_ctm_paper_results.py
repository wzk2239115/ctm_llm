#!/usr/bin/env python3
"""Extract CTM paper experiment results from logs/ctm_paper/ checkpoints.

For each completed experiment (a dir with checkpoint.pt or checkpoint_*.pt),
load training curves + key args, dump to CSV and a per-stage markdown table
aggregating best/final test accuracy across seeds (mean +/- std).

Usage:
    python scripts/extract_ctm_paper_results.py                     # all stages
    python scripts/extract_ctm_paper_results.py --limit 5           # smoke test
    python scripts/extract_ctm_paper_results.py --stages st01,st02  # subset
    python scripts/extract_ctm_paper_results.py --tasks cifar10,sort
    python scripts/extract_ctm_paper_results.py --curves            # also dump full curves
"""

import argparse
import csv
import json
import statistics
import time
from collections import defaultdict
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
LOGS = ROOT / "logs" / "ctm_paper"
OUT_DIR = ROOT / "runs" / "metrics"

# args dumped for each row (sweep-related knobs + key training hyperparams)
KEY_ARGS = [
    "d_model", "heads", "memory_hidden_dims", "synapse_depth", "n_ticks",
    "length", "neuron_select_type", "batch_size", "learning_rate",
    "jepa_weight", "jepa_loss_type", "jepa_predict_delta",
    "cross_tick_jepa_weight", "sparsity",
    "halt_threshold", "halt_confidence_weight", "halt_mode",
    "loss_type", "loss_aggregation", "reflex_weight",
]


def find_ckpt(exp_dir: Path):
    """Prefer checkpoint.pt (full curves); else the highest-iter checkpoint_*.pt."""
    p = exp_dir / "checkpoint.pt"
    if p.exists():
        return p
    cks = []
    for c in exp_dir.glob("checkpoint_*.pt"):
        stem = c.stem.split("_")[-1]
        if stem.isdigit():
            cks.append((int(stem), c))
    if cks:
        cks.sort()
        return cks[-1][1]
    return None


def _to_scalar(x):
    """Coerce x (int/float/tensor/0-d or 1-d array/length-1 list) to float, else None."""
    if x is None:
        return None
    if isinstance(x, (int, float)):
        return float(x)
    try:
        return float(x)  # tensor scalar, numpy 0-d, length-1 array
    except (TypeError, ValueError):
        pass
    try:
        if hasattr(x, "__len__") and len(x) == 1:
            return float(x[0])
    except (TypeError, ValueError):
        pass
    try:
        import numpy as np
        return float(np.asarray(x).mean())
    except Exception:
        return None


def _last(xs):
    if not xs:
        return None
    return _to_scalar(xs[-1])


def _best(xs):
    if not xs:
        return None
    scalars = [_to_scalar(x) for x in xs]
    scalars = [s for s in scalars if s is not None]
    return max(scalars) if scalars else None


def load_summary(ckpt_path: Path):
    """Load relevant fields from a checkpoint. Returns dict (may contain _error)."""
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    except Exception as e:
        return {"_error": f"load: {type(e).__name__}: {e}"}
    if not isinstance(ckpt, dict):
        return {"_error": f"unexpected type {type(ckpt).__name__}"}

    args_ns = ckpt.get("args")
    args_d = vars(args_ns).copy() if args_ns is not None and hasattr(args_ns, "__dict__") else {}

    iters = ckpt.get("iters") or []
    iter_field = ckpt.get("iteration")
    if isinstance(iter_field, (int, float)):
        final_iter = iter_field
    elif iters:
        final_iter = iters[-1]
    else:
        final_iter = None

    test_acc = ckpt.get("test_accuracies") or []
    train_acc = ckpt.get("train_accuracies") or []
    test_acc_mc = ckpt.get("test_accuracies_most_certain") or []
    test_loss = ckpt.get("test_losses") or []

    out = {
        "n_points": len(test_acc),
        "final_iter": final_iter,
        "final_test_acc": _last(test_acc),
        "best_test_acc": _best(test_acc),
        "final_train_acc": _last(train_acc),
        "best_train_acc": _best(train_acc),
        "final_test_acc_mc": _last(test_acc_mc),
        "best_test_acc_mc": _best(test_acc_mc),
        "final_test_loss": _last(test_loss),
        "args": args_d,
    }
    if len(test_acc) > 1:
        out["test_acc_curve"] = [_to_scalar(x) for x in test_acc]
        out["iters_curve"] = [_to_scalar(x) for x in iters] if len(iters) == len(test_acc) else []
    return out


def parse_exp_name(name: str):
    """cifar10_d_model2x_s1 -> (task='cifar10', sweep='d_model2x', seed=1)."""
    parts = name.split("_")
    seed = None
    if parts and parts[-1].startswith("s") and parts[-1][1:].isdigit():
        seed = int(parts[-1][1:])
        parts = parts[:-1]
    task = parts[0] if parts else ""
    sweep = "_".join(parts[1:]) if len(parts) > 1 else ""
    return task, sweep, seed


def fmt_pct(x):
    if x is None:
        return "-"
    return f"{x * 100:.2f}"


def fmt_mean_std(xs):
    if not xs:
        return "-"
    m = statistics.mean(xs)
    if len(xs) > 1:
        sd = statistics.stdev(xs)
        return f"{m * 100:.2f} +/- {sd * 100:.2f}"
    return f"{m * 100:.2f}"


def collect(cli):
    stages = None if cli.stages == "all" else set(cli.stages.split(","))
    tasks = None if cli.tasks == "all" else set(cli.tasks.split(","))

    rows = []
    curves = {}
    t0 = time.time()
    n_ok = n_err = 0

    for stage_dir in sorted(LOGS.iterdir()):
        if not stage_dir.is_dir():
            continue
        stage = stage_dir.name
        if stages is not None and stage not in stages:
            continue
        for exp_dir in sorted(stage_dir.iterdir()):
            if not exp_dir.is_dir():
                continue
            ck = find_ckpt(exp_dir)
            if ck is None:
                continue
            task, sweep, seed = parse_exp_name(exp_dir.name)
            if tasks is not None and task not in tasks:
                continue

            s = load_summary(ck)
            if "_error" in s:
                row = {
                    "stage": stage, "exp": exp_dir.name, "task": task,
                    "sweep": sweep, "seed": seed, "status": "error",
                    "error": s["_error"], "ckpt": ck.name,
                }
                n_err += 1
            else:
                row = {
                    "stage": stage, "exp": exp_dir.name, "task": task,
                    "sweep": sweep, "seed": seed, "status": "ok",
                    "ckpt": ck.name,
                    "n_points": s["n_points"],
                    "final_iter": s["final_iter"],
                    "final_test_acc": s["final_test_acc"],
                    "best_test_acc": s["best_test_acc"],
                    "final_train_acc": s["final_train_acc"],
                    "best_train_acc": s["best_train_acc"],
                    "final_test_acc_mc": s["final_test_acc_mc"],
                    "best_test_acc_mc": s["best_test_acc_mc"],
                    "final_test_loss": s["final_test_loss"],
                }
                if not cli.no_args:
                    a = s["args"]
                    for k in KEY_ARGS:
                        row[k] = a.get(k, "")
                if cli.curves and "test_acc_curve" in s:
                    meta = {
                        "stage": stage,
                        "task": task,
                        "sweep": sweep,
                        "seed": seed,
                        "best_test_acc": s["best_test_acc"],
                        "final_test_acc": s["final_test_acc"],
                        "final_iter": s["final_iter"],
                    }
                    if not cli.no_args:
                        a = s["args"]
                        for k in KEY_ARGS:
                            meta[k] = a.get(k, "")
                    curves[f"{stage}/{exp_dir.name}"] = {
                        "meta": meta,
                        "iters": s.get("iters_curve", []),
                        "test_acc": s["test_acc_curve"],
                    }
                n_ok += 1

            rows.append(row)
            elapsed = time.time() - t0
            ft = row.get("final_test_acc")
            bt = row.get("best_test_acc")
            print(f"[{n_ok + n_err:3d}] {stage}/{exp_dir.name:42s} "
                  f"final={fmt_pct(ft):>7s} best={fmt_pct(bt):>7s}  [{elapsed:.0f}s]",
                  flush=True)

            if cli.limit and n_ok + n_err >= cli.limit:
                break
        if cli.limit and n_ok + n_err >= cli.limit:
            break

    return rows, curves, n_ok, n_err, time.time() - t0


def write_csv(cli, rows):
    keys = []
    for r in rows:
        for k in r.keys():
            if k not in keys:
                keys.append(k)
    with open(cli.csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def write_markdown(cli, rows):
    grouped = defaultdict(list)
    for r in rows:
        if r.get("status") != "ok":
            continue
        grouped[(r["stage"], r["task"], r["sweep"])].append(r)

    lines = ["# CTM Paper Results Summary\n",
             f"- Total experiments scanned: {len(rows)}",
             f"- OK: {sum(1 for r in rows if r.get('status') == 'ok')}",
             f"- Error: {sum(1 for r in rows if r.get('status') == 'error')}\n"]

    cur_stage = None
    for (stage, task, sweep), rs in sorted(grouped.items()):
        if stage != cur_stage:
            lines.append(f"\n## Stage {stage}\n")
            lines.append("| task | sweep | seeds | best_test_acc | final_test_acc | best_test_acc_mc | final_iter |")
            lines.append("|---|---|---|---|---|---|---|")
            cur_stage = stage
        bts = [r["best_test_acc"] for r in rs if r.get("best_test_acc") is not None]
        fts = [r["final_test_acc"] for r in rs if r.get("final_test_acc") is not None]
        bms = [r["best_test_acc_mc"] for r in rs if r.get("best_test_acc_mc") is not None]
        iters = [r.get("final_iter") for r in rs if r.get("final_iter") is not None]
        iter_str = f"{int(statistics.mean(iters))}" if iters else "-"
        lines.append(f"| {task} | {sweep} | {len(rs)} | {fmt_mean_std(bts)} | "
                     f"{fmt_mean_std(fts)} | {fmt_mean_std(bms)} | {iter_str} |")

    with open(cli.md, "w") as f:
        f.write("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default=str(OUT_DIR / "ctm_paper_summary.csv"))
    ap.add_argument("--md", default=str(OUT_DIR / "ctm_paper_summary.md"))
    ap.add_argument("--stages", default="all", help="comma-separated stage ids")
    ap.add_argument("--tasks", default="all", help="comma-separated task names")
    ap.add_argument("--limit", type=int, default=0, help="0 = no limit (smoke test: --limit 5)")
    ap.add_argument("--no-args", action="store_true", help="skip args dump (faster)")
    ap.add_argument("--curves", action="store_true",
                    help="also dump full test_acc curves to runs/metrics/ctm_paper_curves.json")
    cli = ap.parse_args()

    if not LOGS.exists():
        print(f"ERROR: {LOGS} does not exist")
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows, curves, n_ok, n_err, elapsed = collect(cli)

    if not rows:
        print("No experiments found.")
        return

    write_csv(cli, rows)
    print(f"\n=== CSV  -> {cli.csv}  ({len(rows)} rows)")

    write_markdown(cli, rows)
    print(f"=== MD   -> {cli.md}")

    if cli.curves and curves:
        with open(OUT_DIR / "ctm_paper_curves.json", "w") as f:
            json.dump(curves, f)
        print(f"=== curves -> {OUT_DIR / 'ctm_paper_curves.json'} ({len(curves)} runs)")

    print(f"=== ok={n_ok} err={n_err} elapsed={elapsed:.1f}s")


if __name__ == "__main__":
    main()
