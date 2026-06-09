# Overnight Sparse CTM Summary

Source summary: `overnight_sparse_ctm_summary.csv`.

Experiment script: `scripts/experiment_plan_overnight_sparse_ctm.py`.

## Current Status

The exported summary currently contains only the CSV header and no experiment rows.

This means there are no final `og00_` to `og10_` training metrics available in this file yet. It cannot support conclusions about loss, quality, active-cell behavior, or final quality/cost score.

## What This File Should Contain

Once formal overnight runs finish, this summary should contain one final row per experiment with fields such as:

- `experiment_name`;
- `loss`;
- `tokens_per_sec`;
- `peak_memory_mb`;
- `active_cell_fraction`;
- `quality_cost_score`;
- `global_step`;
- `metrics_file`.

Those rows are produced by the `summarize` subcommand, which scans `runs/metrics` and keeps final metrics whose experiment names match the overnight prefixes.

## Interpretation

The empty file likely means one of these is true:

1. Only quick-probe jobs have run so far.
   Quick probes are named like `qp__og...__bs...`, and are intentionally excluded from final summary rows.

2. Formal overnight jobs have not produced final metrics yet.
   In that case, the summary command has no `ogXX_*.csv` metric files to collect.

3. The formal metrics are on the compute machine but not under the `runs/metrics` directory used by the export command.

4. The summary was copied back before the overnight jobs finished.

## What To Do Next

On the compute machine, from `/home/jovyan/h800fast/wangzekai/ctm_llm`, first check whether formal metrics exist:

```bash
ls runs/metrics/og*.csv | head
```

If formal `og*.csv` files exist, rerun:

```bash
python scripts/experiment_plan_overnight_sparse_ctm.py summarize \
  --metrics_dir runs/metrics \
  --output runs/metrics/overnight_sparse_ctm_summary.csv
```

If only `qp__og...` files exist, then the overnight formal training stage has not been run yet. Use `overnight_sparse_ctm_quick_probe_report.csv` only for feasibility and batch planning, not for quality analysis.

## Bottom Line

There is no formal overnight result to analyze yet in `overnight_sparse_ctm_summary.csv`. The next actionable step is to confirm whether formal `og*.csv` metrics exist on the compute machine and rerun the summary export after those jobs finish.
