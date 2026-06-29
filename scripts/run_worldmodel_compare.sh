#!/usr/bin/env bash
# Multi-GPU parallel launcher for paper/run_worldmodel_compare.py.
# Splits the (env x model x seed) grid into NGPU shards (one per GPU), runs them
# concurrently, then merges the per-shard CSVs and prints a mean+-std summary.
#
# Usage on a 4xH100 node:
#   ./scripts/run_worldmodel_compare.sh                    # default: 4 GPUs, seeds 0 1 2
#   NGPU=8 ./scripts/run_worldmodel_compare.sh --seeds 0 1 2 3 4 --epochs 80
#   NGPU=1 ./scripts/run_worldmodel_compare.sh --quick     # serial sanity check
# Extra args after the script are forwarded to run_worldmodel_compare.py.
set -e
NGPU=${NGPU:-4}
PY=${PYTHON:-python}
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
mkdir -p logs

rm -f csv_data/worldmodel_compare_results_gpu*.csv csv_data/worldmodel_compare_results.csv

EXTRA="$@"
echo "[launch] $NGPU shard(s); forwarded args: $EXTRA"
pids=()
for i in $(seq 0 $((NGPU-1))); do
  CUDA_VISIBLE_DEVICES=$i $PY paper/run_worldmodel_compare.py \
    --shard "$i" --nshards "$NGPU" --csv_suffix "_gpu$i" $EXTRA \
    > "logs/wm_compare_gpu$i.log" 2>&1 &
  pids+=($!)
done
echo "[launch] launched ${#pids[@]} workers (PIDs ${pids[*]}); waiting..."
wait

# Merge per-shard CSVs and summarize.
$PY - <<'PYEOF'
import csv, glob, statistics
from collections import defaultdict
shards = sorted(glob.glob("csv_data/worldmodel_compare_results_gpu*.csv"))
if not shards:
    raise SystemExit("[merge] no shard CSVs found — check logs/wm_compare_gpu*.log")
rows, fields = [], None
for p in shards:
    with open(p) as f:
        r = csv.DictReader(f)
        fields = r.fieldnames
        rows.extend(list(r))
rows.sort(key=lambda r: (r["env"], r["model"], int(r["seed"])))
out = "csv_data/worldmodel_compare_results.csv"
with open(out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    w.writerows(rows)
print(f"[merge] {len(rows)} rows from {len(shards)} shard(s) -> {out}\n")
buckets = defaultdict(list)
for r in rows:
    buckets[(r["env"], r["model"])].append(float(r["success_rate"]))
print("[summary] success_rate mean +- std over seeds:")
for (env, model), vals in sorted(buckets.items()):
    m = statistics.mean(vals)
    sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
    print(f"  {env:<12} {model:<12} {m:5.1f} +- {sd:4.1f}  (n={len(vals)})")
# collapse flag per (env, model)
print("\n[collapse] cells with latent_var < 1e-3 (encoder collapsed):")
for r in rows:
    try:
        if float(r["latent_var"]) < 1e-3:
            print(f"  {r['env']:<12} {r['model']:<12} seed{r['seed']} var={r['latent_var']} succ={r['success_rate']}%")
    except (ValueError, KeyError):
        pass
PYEOF
echo "[launch] done. merged CSV: csv_data/worldmodel_compare_results.csv"
