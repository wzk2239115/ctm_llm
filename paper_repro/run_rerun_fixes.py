"""Rerun only the experiments affected by the two code fixes (reviewer #1 + #3).

Fixes (committed):
  - draft-revise: removed .detach() on draft_pred -> CE now has gradient (deep
    supervision). Affects cifar10/mazes only (parity CE was shape-guard-skipped,
    = noise injection, unchanged). Baselines unaffected (draft_mode=none).
  - sparsity: real sparse NLM compute (gather/scatter, --sparse_nlm_compute) ->
    saves ~k/N FLOPs. Replaces old post-hoc-mask sparsity runs.

Scope (70 runs total):
  - revise:   cifar10 (w=0.2,cp=0.3) + mazes (w=0.1,cp=0.15), 5 seeds  = 10
  - sparsity: cifar10/mazes/parity x {0.1,0.25,0.5,0.75} x 5 seeds        = 60

NOT rerun (unchanged by fixes): baselines, parity revise, JEPA, qamnist.

Usage (compute machine, after `git pull`):
  export http_proxy="http://public-proxy.qihoo.net:3128"
  export https_proxy="http://public-proxy.qihoo.net:3128"
  python paper_repro/run_rerun_fixes.py --dry-run          # inspect plan
  python paper_repro/run_rerun_fixes.py --gpus 8           # launch
  nohup python paper_repro/run_rerun_fixes.py --gpus 8 > paper_repro/logs/rerun_fixes.log 2>&1 &

After completion, re-extract:
  python scripts/extract_ctm_paper_results.py --logs paper_repro/logs \
      --csv paper_repro/csv_data/repro_summary_fixes.csv \
      --md  paper_repro/csv_data/repro_summary_fixes.md --curves
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "paper"))
from exp_runner import run_all, make_revise, make_sparsity  # noqa: E402

LOG_ROOT = "paper_repro/logs"
RATIOS = (0.1, 0.25, 0.5, 0.75)
SPARSE_TASKS = ["cifar10", "mazes", "parity"]   # figE scope (sort excluded: wiring incomplete)


def build(seeds):
    exps = (
        make_revise(["cifar10"], seeds, w=0.2, cp=0.3)     # detach fix -> CE now active
        + make_revise(["mazes"], seeds, w=0.1, cp=0.15)
        + make_sparsity(SPARSE_TASKS, seeds, ratios=RATIOS, sparse_nlm=True)  # real compute
    )
    return exps


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gpus", type=int, default=8)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--mem-util", type=float, default=0.80)
    args = ap.parse_args()

    seeds = list(range(args.seeds))
    exps = build(seeds)
    print(f"Rerun (fixes): {len(exps)} experiments, {args.seeds} seeds")
    print("  revise:   cifar10 + mazes")
    print(f"  sparsity: {SPARSE_TASKS} x {RATIOS} (sparse_nlm_compute=True)")
    run_all(exps, gpus=args.gpus, log_root=LOG_ROOT, dry_run=args.dry_run,
            mem_util=args.mem_util)


if __name__ == "__main__":
    main()
