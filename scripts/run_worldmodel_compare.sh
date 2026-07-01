#!/usr/bin/env bash
# Multi-GPU parallel launcher for paper/run_worldmodel_compare.py.
# Splits the (env x model x seed) grid into TOTAL_SHARDS = NGPU*PROCS_PER_GPU
# shards, runs them concurrently (PROCS_PER_GPU processes share each GPU via
# round-robin), then merges the per-shard CSVs and prints a mean+-std summary.
#
# Usage on a 4xH100 node:
#   ./scripts/run_worldmodel_compare.sh                    # 4 GPU x 1 proc = 4 shards
#   PROCS_PER_GPU=4 ./scripts/run_worldmodel_compare.sh --epochs 80   # 4 x 4 = 16 shards (fills GPU)
#   NGPU=8 ./scripts/run_worldmodel_compare.sh --seeds 0 1 2 3 4 --epochs 80
#   NGPU=1 ./scripts/run_worldmodel_compare.sh --quick     # serial sanity check
# Extra args after the script are forwarded to run_worldmodel_compare.py.
# Why PROCS_PER_GPU: world-model tasks are tiny (batch 64, small MLP/CNN), so a
# single process leaves the GPU >80% idle. Running several shards per GPU fills
# the spare capacity without changing any experiment semantics.
# (No `set -e`: a single failed shard should not block the merge/summary, so a
# smoke run surfaces which cell crashed instead of aborting silently.)
NGPU=${NGPU:-4}
PROCS_PER_GPU=${PROCS_PER_GPU:-1}
TOTAL_SHARDS=$((NGPU * PROCS_PER_GPU))
PY=${PYTHON:-python}
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
mkdir -p logs

rm -f csv_data/worldmodel_compare_results_shard*.csv csv_data/worldmodel_compare_results.csv

EXTRA="$@"
echo "[launch] $TOTAL_SHARDS shard(s) on $NGPU GPU(s) ($PROCS_PER_GPU proc/gpu); forwarded args: $EXTRA"
pids=()
for i in $(seq 0 $((TOTAL_SHARDS-1))); do
  CUDA_VISIBLE_DEVICES=$((i % NGPU)) $PY paper/run_worldmodel_compare.py \
    --shard "$i" --nshards "$TOTAL_SHARDS" --csv_suffix "_shard$i" $EXTRA \
    > "logs/wm_compare_shard$i.log" 2>&1 &
  pids+=($!)
done
echo "[launch] launched ${#pids[@]} workers (PIDs ${pids[*]}); waiting..."
wait

# Merge per-shard CSVs and summarize.
$PY - <<'PYEOF'
import csv, glob, statistics
from collections import defaultdict
shards = sorted(glob.glob("csv_data/worldmodel_compare_results_shard*.csv"))
if not shards:
    raise SystemExit("[merge] no shard CSVs found — check logs/wm_compare_shard*.log")
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
