#!/usr/bin/env python3
"""Fix for st04 JEPA ablation bug: variant knobs confounded with weight.

BUG: experiment_plan_ctm_paper.py (build_st04_jepa_sweep) runs the loss /
stop-grad / predictor-depth variants at cross_tick_jepa_weight = 1.0, while
the weight sweep and fig6's "default" bar sit at weight = 0.1. JEPA weight is
not a neutral knob (sort: 43% @ w0.1 -> 20% @ w1.0; all weight=1.0 sort
variants collapse to ~0.25%), so each variant changes TWO variables at once
(its target knob AND a 10x weight jump) and the delta cannot be attributed to
loss-type / stop-grad / depth alone.

FIX: this script re-runs the SAME four variants (mse / nostopgrad / pd1 / pd4)
but at weight = 0.1 — the empirically stable default where the base config
trains — so each knob is tested in isolation. Uses stage name 'st04b' to avoid
clashing with in-flight st04 runs and to keep results comparable side-by-side.

parity is deliberately EXCLUDED: it fails (final_iter=0, acc=0.499) whenever
ideas_active=True, in st04 and every other idea stage. That is a separate
runtime bug in baseline/tasks/parity/train.py:314 (ideas_active branch), not a
plan bug — re-running it here would just fail again. Add parity back with
--tasks once that train.py bug is fixed.

The reference baseline for st04b is the existing st04 'jepa_w0.1' run (already
complete with 3 seeds); it is NOT re-run here. Defaults to 3 seeds (0,1,2) so
the variant bars get proper error bars (the original st04 had several n=1
variants: qamnist/jepa_pd1, cifar10/jepa_pd4).

Usage:
    python scripts/fix_bug_st04_jepa_weight_confound.py plan
    python scripts/fix_bug_st04_jepa_weight_confound.py plan --tasks sort
    python scripts/fix_bug_st04_jepa_weight_confound.py submit --no-wait
    python scripts/fix_bug_st04_jepa_weight_confound.py submit --no-wait --tasks sort --seeds 0
    python scripts/fix_bug_st04_jepa_weight_confound.py csv
"""

import argparse
import csv
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from experiment_plan_ctm_paper import (  # noqa: E402
    TASKS, _p, exp, with_seed, submit_to_pool,
    POOL_CONFIG, MASTER_ADDR, PORT,
)

STAGE = "st04b"
# Fixed weight: the stable default. All variants ablate ONE knob vs this.
FIXED_WEIGHT = 0.1
# parity is blocked by a separate ideas_active runtime bug (see module docstring).
AFFECTED_TASKS = ["cifar10", "mazes", "qamnist", "sort"]

# Each variant changes exactly ONE JEPA knob, weight held at FIXED_WEIGHT.
# sweep tag mirrors st04 naming so results can be cross-referenced, with a
# _w0p1 suffix to disambiguate from the (confounded) st04 weight=1.0 variants.
VARIANTS = [
    # (suffix, sweep_tag, question, extra_jepa_cfg)
    ("mse", "jepa_mse_w0p1", "loss=mse",
     {"cross_tick_jepa_loss": "mse"}),
    ("nostopgrad", "jepa_nostopgrad_w0p1", "target_stop_grad=False",
     {"cross_tick_jepa_target_stop_grad": False}),
    ("pd1", "jepa_pd1_w0p1", "predictor_depth=1",
     {"cross_tick_jepa_predictor_depth": 1}),
    ("pd4", "jepa_pd4_w0p1", "predictor_depth=4",
     {"cross_tick_jepa_predictor_depth": 4}),
]

JEPA_DEFAULTS = dict(
    cross_tick_jepa_hidden_dim=128,
    cross_tick_jepa_predictor_depth=2,
    cross_tick_jepa_dropout=0.0,
    # canonical defaults (match st04 weight sweep at w0.1):
    cross_tick_jepa_loss="cosine",
    cross_tick_jepa_target_stop_grad=True,
)


def build_plan(tasks=None, seeds=None):
    """Build the fixed JEPA variant ablation with weight held at FIXED_WEIGHT."""
    if tasks is None:
        tasks = AFFECTED_TASKS
    if seeds is None:
        seeds = [0, 1, 2]
    plan = []
    for task_name in tasks:
        if task_name not in TASKS or task_name == "parity":
            continue  # parity blocked by separate ideas_active runtime bug
        module, base, _ = TASKS[task_name]
        for suffix, sweep_tag, question, extra in VARIANTS:
            for seed in seeds:
                cfg = dict(with_seed(base, seed))
                cfg.update(JEPA_DEFAULTS)
                # KEY FIX: weight held constant at the stable default.
                cfg["cross_tick_jepa_weight"] = FIXED_WEIGHT
                cfg.update(extra)  # apply the single knob under test
                log_dir = f"logs/ctm_paper/{STAGE}/{task_name}_{suffix}"
                name = f"{STAGE}_{task_name}_{suffix}"
                if seed != 0:
                    log_dir += f"_s{seed}"
                    name += f"_s{seed}"
                cfg["log_dir"] = log_dir
                plan.append(exp(
                    name,
                    f"{task_name}: JEPA {question} (weight fixed at {FIXED_WEIGHT})",
                    _p(module, cfg),
                    tags=[task_name, "jepa", sweep_tag, "bugfix", f"seed{seed}"],
                ))
    return plan


def print_plan(plan):
    print(f"\n{'=' * 78}")
    print(f"FIX st04b: JEPA variant ablation with weight held at {FIXED_WEIGHT}")
    print(f"  Bug: experiment_plan_ctm_paper.py build_st04_jepa_sweep ran the")
    print(f"       loss/stopgrad/depth variants at weight=1.0 (10x the w0.1")
    print(f"       default), confounding each knob with a weight jump.")
    print(f"  Reference baseline: existing st04 'jepa_w0.1' (NOT re-run here).")
    print(f"  parity EXCLUDED (separate ideas_active runtime bug in train.py).")
    print(f"  Total experiments: {len(plan)}")
    print(f"{'=' * 78}\n")
    by_task = {}
    for e in plan:
        task = e["tags"][0] if e["tags"] else "?"
        by_task.setdefault(task, []).append(e)
    for task in AFFECTED_TASKS:
        exps = by_task.get(task, [])
        if not exps:
            continue
        print(f"  ── {task} ──")
        for e in exps:
            print(f"     {e['name']}")
        print()
    print(f"TOTAL: {len(plan)} experiments")


def cmd_plan(args):
    plan = build_plan(args.tasks, args.seeds)
    print_plan(plan)


def cmd_submit(args):
    plan = build_plan(args.tasks, args.seeds)
    if not plan:
        print("No experiments to submit.")
        return
    print(f"Submitting {len(plan)} st04b (bugfix) experiments to pool "
          f"at {MASTER_ADDR}:{PORT}")
    for e in plan:
        print(f"  {e['name']}")
    if args.dry_run:
        print("\n[dry-run] no tasks submitted.")
        return
    print()
    for e in plan:
        result = submit_to_pool(e, POOL_CONFIG, MASTER_ADDR, PORT)
        if result is None:
            print(f"  FAILED to submit {e['name']}")
            continue
        task_id = result if isinstance(result, str) else result.get("task_id", "")
        print(f"  {e['name']} -> task_id={task_id}")


def cmd_csv(args):
    plan = build_plan(args.tasks, args.seeds)
    out = Path(args.output or f"runs/experiment_plans/{STAGE}_plan.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["name", "question", "command", "tags"])
        w.writeheader()
        for e in plan:
            w.writerow({
                "name": e["name"],
                "question": e["question"],
                "command": e["command"],
                "tags": ";".join(e["tags"]),
            })
    print(f"Wrote {len(plan)} experiments to {out}")


def parse_seeds(s):
    if isinstance(s, list):
        return [int(x) for x in s]
    return [int(x) for x in str(s).split(",")]


def parse_tasks(s):
    if not s:
        return None
    return [t.strip() for t in str(s).split(",") if t.strip()]


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = ap.add_subparsers(dest="command")

    def add_common(p):
        p.add_argument("--tasks", default=None,
                       help=f"comma-separated task names (default: {','.join(AFFECTED_TASKS)}); "
                            f"parity is blocked by a separate runtime bug")
        p.add_argument("--seeds", default="0,1,2", type=parse_seeds,
                       help="comma-separated seeds (default 0,1,2 for error bars)")

    p_plan = sub.add_parser("plan", help="show the bugfix experiment list")
    add_common(p_plan)

    p_submit = sub.add_parser("submit", help="submit bugfix runs to the pool")
    add_common(p_submit)
    p_submit.add_argument("--dry-run", action="store_true",
                          help="list experiments but do not submit")
    p_submit.add_argument("--master-addr", default=MASTER_ADDR)
    p_submit.add_argument("--port", type=int, default=PORT)

    p_csv = sub.add_parser("csv", help="dump bugfix plan to CSV")
    add_common(p_csv)
    p_csv.add_argument("--output", default=None)

    args = ap.parse_args()
    args.tasks = parse_tasks(args.tasks) if hasattr(args, "tasks") else None
    if args.command == "plan":
        cmd_plan(args)
    elif args.command == "submit":
        cmd_submit(args)
    elif args.command == "csv":
        cmd_csv(args)
    else:
        ap.print_help()
