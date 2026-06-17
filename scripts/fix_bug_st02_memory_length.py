#!/usr/bin/env python3
"""Fix for st02 tick sweep bug: memory_length was coupled to n_ticks.

BUG: experiment_plan_ctm_paper.py:289 (build_st02_tick_sweep) set
    cfg["iterations"] = t                    # n_ticks           (correct - this is the sweep var)
    cfg["memory_length"] = max(2, t // 2)    # NLMS filter history (WRONG - confounds the sweep)

Impact: every st02 tick experiment had TWO variables changed at once, so the
acc vs n_ticks curve cannot be attributed to thinking time alone. Most visible
on sort: tick1-25 collapsed to 0.3% because memory_length dropped to 2-12
(sort needs >=25), while tick50 happened to match the default (memory_length=25)
and worked fine (87%).

FIX: this script re-runs the SAME tick sweep [1, 2, 5, 10, 25, 50] but leaves
memory_length at each task's base default, so n_ticks is the only independent
variable. Uses stage name 'st02b' to avoid clashing with in-flight st02 runs
and to keep results comparable side-by-side.

Usage:
    python scripts/fix_bug_st02_memory_length.py plan
    python scripts/fix_bug_st02_memory_length.py plan --tasks sort
    python scripts/fix_bug_st02_memory_length.py submit --no-wait
    python scripts/fix_bug_st02_memory_length.py submit --no-wait --tasks sort
    python scripts/fix_bug_st02_memory_length.py submit --no-wait --tasks sort --seeds 0,1,2
    python scripts/fix_bug_st02_memory_length.py csv
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

STAGE = "st02b"
TICK_VALUES = [1, 2, 5, 10, 25, 50]
# qamnist uses its own repeat-based sweep in st02 and is NOT affected.
AFFECTED_TASKS = ["cifar10", "mazes", "parity", "sort"]


def build_plan(tasks=None, seeds=None):
    """Build the fixed tick sweep with memory_length held at task default."""
    if tasks is None:
        tasks = AFFECTED_TASKS
    if seeds is None:
        seeds = [0]
    plan = []
    for task_name in tasks:
        if task_name not in TASKS or task_name == "qamnist":
            continue
        module, base, _ = TASKS[task_name]
        default_ml = base.get("memory_length")
        for t in TICK_VALUES:
            for seed in seeds:
                cfg = dict(with_seed(base, seed))
                cfg["iterations"] = t
                # KEY FIX: do NOT touch memory_length; keep base default.
                # (deliberately not setting cfg["memory_length"] so the base value wins)
                log_dir = f"logs/ctm_paper/{STAGE}/{task_name}_tick{t}"
                name = f"{STAGE}_{task_name}_tick{t}"
                if seed != 0:
                    log_dir += f"_s{seed}"
                    name += f"_s{seed}"
                cfg["log_dir"] = log_dir
                plan.append(exp(
                    name,
                    f"{task_name}: {t} ticks "
                    f"(memory_length fixed at default={default_ml})",
                    _p(module, cfg),
                    tags=[task_name, "tick-sweep", "bugfix", f"seed{seed}"],
                ))
    return plan


def print_plan(plan):
    print(f"\n{'=' * 78}")
    print(f"FIX st02b: tick sweep with memory_length held at task default")
    print(f"  Bug: experiment_plan_ctm_paper.py build_st02_tick_sweep coupled")
    print(f"       memory_length = max(2, t//2) with iterations = t")
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
        module, base, _ = TASKS[task]
        print(f"  ── {task}  (default memory_length={base.get('memory_length')}, "
              f"iterations={base.get('iterations')}) ──")
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
    print(f"Submitting {len(plan)} st02b (bugfix) experiments to pool "
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
                       help=f"comma-separated task names (default: {','.join(AFFECTED_TASKS)})")
        p.add_argument("--seeds", default="0", type=parse_seeds,
                       help="comma-separated seeds (default: 0; use '0,1,2' for full repeat)")

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
