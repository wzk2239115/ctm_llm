#!/usr/bin/env bash
# Full sweep: our world models (jepa-mlp / stream-ctm / ctm-encoder) across the
# stable-worldmodel benchmark env suite, sharded over NGPU GPUs.
#
# REQUIRES: pip install -e stable-worldmodel[all]   (compute machine, with proxy)
#
# Override the grid via env vars:
#   ENVS="swm/PushT-v1 swm/TwoRoom-v1"  MODELS="jepa-mlp stream-ctm"  SEEDS="0 1" \
#   NGPU=4 ./scripts/run_worldmodel_swm.sh
#
# Smoke (one env, fast):  ./scripts/run_worldmodel_swm.sh --quick
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
mkdir -p logs csv_data data/swm_wm

NGPU=${NGPU:-4}
PY=${PYTHON:-python}
ENVS=${ENVS:-"swm/PushT-v1 swm/TwoRoom-v1 swm/OGBCube-v0 swm/OGBScene-v0 swm/CheetahDMControl-v0 swm/WalkerDMControl-v0 swm/ReacherDMControl-v0 swm/FetchReach-v3 swm/FetchPush-v3 swm/FetchSlide-v3 swm/FetchPickAndPlace-v3"}
MODELS=${MODELS:-"jepa-mlp stream-ctm ctm-encoder"}
SEEDS=${SEEDS:-"0 1 2"}
EXTRA="$@"                      # forwarded to the runner (e.g. --quick --epochs 60)
COMMON=${COMMON:-"--episodes 120 --epochs 40 --image_size 64 --latent_dim 64 --var_weight 4.0 --cem_samples 128 --eval_episodes 20"}

echo "[swm-sweep] NGPU=$NGPU"
echo "[swm-sweep] ENVS: $ENVS"
echo "[swm-sweep] MODELS: $MODELS"
echo "[swm-sweep] SEEDS: $SEEDS"
echo "[swm-sweep] COMMON: $COMMON  EXTRA: $EXTRA"

# 1) Prepare datasets once per env (serial; skip envs whose deps are missing).
echo "[swm-sweep] === preparing datasets (serial) ==="
for env in $ENVS; do
  echo "  prepare $env ..."
  $PY paper/run_worldmodel_swm.py --env "$env" --prepare $COMMON \
    > "logs/swm_prepare_$(echo $env | tr '/' '_').log" 2>&1 \
    || echo "  ! prepare $env FAILED (missing dep?) — see log; will skip at sweep"
done

# 2) Build the (env, model, seed) task list and shard it round-robin over NGPU.
$PY - <<PYEOF
import os
envs = "$ENVS".split(); models = "$MODELS".split(); seeds = "$SEEDS".split()
ngpu = $NGPU
tasks = [(e, m, s) for e in envs for m in models for s in seeds]
shards = [[] for _ in range(ngpu)]
for i, t in enumerate(tasks):
    shards[i % ngpu].append(t)
for i, s in enumerate(shards):
    with open(f"logs/swm_shard_{i}.txt", "w") as f:
        for e, m, s in s:
            f.write(f"{e} {m} {s}\n")
print(f"[swm-sweep] {len(tasks)} tasks -> {ngpu} shards (sizes {[len(s) for s in shards]})")
PYEOF

# 3) Launch one worker per GPU; each runs its shard's tasks serially.
echo "[swm-sweep] === launching $NGPU shard workers ==="
rm -f csv_data/worldmodel_swm_results_gpu*.csv
pids=()
for i in $(seq 0 $((NGPU-1))); do
  (
    while read -r env model seed; do
      [ -z "$env" ] && continue
      CUDA_VISIBLE_DEVICES=$i $PY paper/run_worldmodel_swm.py \
        --env "$env" --model "$model" --seed "$seed" \
        --csv "csv_data/worldmodel_swm_results_gpu${i}.csv" \
        $COMMON $EXTRA
    done < "logs/swm_shard_${i}.txt"
  ) > "logs/swm_sweep_gpu${i}.log" 2>&1 &
  pids+=($!)
done
echo "[swm-sweep] launched ${#pids[@]} workers (PIDs ${pids[*]}); waiting..."
wait

# 4) Merge per-shard CSVs + summarize.
$PY - <<'PYEOF'
import csv, glob, statistics
from collections import defaultdict
shards = sorted(glob.glob("csv_data/worldmodel_swm_results_gpu*.csv"))
if not shards:
    raise SystemExit("[merge] no shard CSVs — check logs/swm_sweep_gpu*.log")
rows, fields = [], None
for p in shards:
    with open(p) as f:
        r = csv.DictReader(f); fields = r.fieldnames; rows.extend(list(r))
rows.sort(key=lambda r: (r["env"], r["model"], int(r["seed"])))
out = "csv_data/worldmodel_swm_results.csv"
with open(out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
print(f"\n[merge] {len(rows)} rows from {len(shards)} shard(s) -> {out}\n")
buckets = defaultdict(list)
for r in rows:
    buckets[(r["env"], r["model"])].append(float(r["success_rate"]))
print("[summary] success_rate mean +- std over seeds:")
for (env, model), vals in sorted(buckets.items()):
    m = statistics.mean(vals); sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
    rand = [float(r["random_rate"]) for r in rows if r["env"] == env][:1]
    rstr = f"(rand {rand[0]:.0f}%)" if rand else ""
    print(f"  {env:<26} {model:<12} {m:5.1f} +- {sd:4.1f}  (n={len(vals)}) {rstr}")
print("\n[collapse] cells with latent_var < 1e-3 (encoder collapsed):")
for r in rows:
    try:
        if float(r["latent_var"]) < 1e-3:
            print(f"  {r['env']:<26} {r['model']:<12} seed{r['seed']} var={r['latent_var']} succ={r['success_rate']}%")
    except (ValueError, KeyError):
        pass
PYEOF
echo "[swm-sweep] done. merged CSV: csv_data/worldmodel_swm_results.csv"
