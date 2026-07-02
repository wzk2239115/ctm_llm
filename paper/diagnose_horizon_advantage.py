#!/usr/bin/env python3
"""试金石: CTM 在 world-model 上的长程稳定性优势是否存在.

机制假设: CTM 持续状态每步精炼能纠错; Markov MLP 单步 latent 堆叠, 误差随
horizon 指数累积. 所以 horizon 越长, stream-ctm 相对 jepa-mlp 优势应该越大.
当前 horizon=6 把这个优势架空了 (6步内 Markov 够准).

判定:
  stream 的 success 随 horizon 衰减比 jepa 慢  -> CTM 长程稳定优势确认 (找到了主场)
  两者衰减一样快 / stream 更快                 -> CTM 在 world-model 无结构性优势, 换思路

矩阵: 3 代表 env x horizon[6,12,20,30] x {stream-base, jepa-mlp} x 5 seeds = 120 runs
5 卡 spawn 并行 (~2h), 出 success-vs-horizon 曲线 + 衰减率对比.

跑法 (tmux):
    python paper/diagnose_horizon_advantage.py
"""
import argparse, csv, os, sys, time
from collections import defaultdict
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch

from stream_ctm_deep_eval import build_one, evaluate, run_one, _collect_env, DEEP_FIELDS
import torch.multiprocessing as mp

ENVS = ["pendulum", "pendulum-partial", "tworoom-state"]
HORIZONS = [6, 12, 20, 30]
SEEDS = [0, 1, 2, 3, 4]
# (name, mem, gate, kind)
MODELS = [("jepa-mlp", 8, "none", "jepa"), ("stream-base", 8, "none", "stream")]


def _worker(rank, args, nworkers, all_tasks):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(rank)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    my = all_tasks[rank::nworkers]
    envs_in = sorted({t[0] for t in my})
    bufs = {}
    for e in envs_in:
        bufs[e] = _collect_env(e, args)
        print(f"[gpu{rank}] collect {e}: {len(bufs[e][0].episodes)} eps", flush=True)
    Path("csv_data").mkdir(exist_ok=True)
    out = f"csv_data/horizon_sweep_shard{rank}.csv"
    rows = []
    with open(out, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=DEEP_FIELDS + ["horizon"]); wr.writeheader()
        for (env, mname, mem, gate, kind, seed, hor) in my:
            args.horizon = hor
            try:
                r = run_one(env, mname, mem, gate, kind, seed, args.epochs,
                            bufs[env][0], bufs[env][1], args, device)
            except Exception as exc:
                r = dict(env=env, variant=mname, seed=seed, epochs=args.epochs,
                         memory_length=mem, state_gate=gate, success_rate=-1.0,
                         random_rate=-1.0, dynamics_err=-1.0, latent_var=-1.0, elapsed_s=0.0)
                print(f"[gpu{rank}] ERROR {env}/{mname}/h{hor}/s{seed}: {exc}", flush=True)
            r["horizon"] = hor
            wr.writerow(r); f.flush(); rows.append(r)
            print(f"[gpu{rank}] {env:<16}{mname:<12}h{hor:<3}s{seed} "
                  f"succ={r['success_rate']:5.1f} dyn={r['dynamics_err']} [{r['elapsed_s']:.0f}s]", flush=True)
    print(f"[gpu{rank}] done {len(rows)} -> {out}", flush=True)


def _mean(xs):
    xs = [x for x in xs if x is not None and x == x and x >= 0]
    return float(np.mean(xs)) if xs else float("nan")


def report(rows):
    by = defaultdict(list)  # (env, model, horizon) -> [succ]
    dby = defaultdict(list)
    for r in rows:
        by[(r["env"], r["variant"], r["horizon"])].append(r["success_rate"])
        dby[(r["env"], r["variant"], r["horizon"])].append(r["dynamics_err"])
    L = ["# horizon sweep: CTM 长程稳定性优势试金石\n",
         f"生成: {time.strftime('%Y-%m-%d %H:%M:%S')} | runs: {len(rows)}\n"]
    # success vs horizon 表
    L.append("\n## success_rate vs horizon (mean over seeds)\n")
    L.append("| env | model | h6 | h12 | h20 | h30 | 衰减(h30-h6) |")
    L.append("|---|---|---|---|---|---|---|")
    decay = {}
    for env in ENVS:
        for mname, *_ in MODELS:
            cells = [env, mname]
            succs = []
            for h in HORIZONS:
                m = _mean(by.get((env, mname, h), []))
                succs.append(m)
                cells.append(f"{m:.1f}" if m == m else "-")
            d = (succs[-1] - succs[0]) if all(s == s for s in succs) else float("nan")
            decay[(env, mname)] = d
            cells.append(f"{d:+.1f}")
            L.append("| " + " | ".join(cells) + " |")
    # dyn_err vs horizon
    L.append("\n## dynamics_err vs horizon (长程累积误差, 越低越稳)\n")
    L.append("| env | model | h6 | h12 | h20 | h30 |")
    L.append("|---|---|---|---|---|---|")
    for env in ENVS:
        for mname, *_ in MODELS:
            cells = [env, mname]
            for h in HORIZONS:
                m = _mean(dby.get((env, mname, h), []))
                cells.append(f"{m:.5f}" if m == m else "-")
            L.append("| " + " | ".join(cells) + " |")

    # 判定: stream 衰减是否比 jepa 慢
    L.append("\n## 判定 (stream 衰减比 jepa 慢 = CTM 长程优势确认)\n")
    L.append("| env | jepa 衰减 | stream 衰减 | stream-jepa (越大越说明 CTM 长程更稳) | 判定 |")
    L.append("|---|---|---|---|---|")
    win = []
    for env in ENVS:
        dj = decay.get((env, "jepa-mlp"), float("nan"))
        ds = decay.get((env, "stream-base"), float("nan"))
        diff = ds - dj  # stream 衰减更小(更负) 或衰减更少 => diff 更大
        # success 衰减: h30-h6. stream 衰减少 => ds > dj => diff>0 = stream 更稳
        flag = "CTM 长程更稳 ✓" if (diff == diff and diff > 3) else ("CTM 更差 ✗" if (diff == diff and diff < -3) else "无差异")
        if diff == diff and diff > 3:
            win.append(env)
        L.append(f"| {env} | {dj:+.1f} | {ds:+.1f} | {diff:+.1f} | {flag} |")
    L.append("\n## 结论\n")
    if win:
        L.append(f"CTM 在 {win} 上长程衰减明显比 jepa 慢 —— **找到 world-model 主场**: "
                 "提升 planning horizon 让 CTM 持续状态的长程稳定性成为优势. "
                 "下一步: 在长 horizon (20+) 下深化 (long-range JEPA 目标 / CTM 状态进 cost).")
    else:
        L.append("CTM 长程衰减没有明显优于 jepa —— world-model 框架下 CTM 无结构性优势. "
                 "换思路: 让 CTM 持续状态直接进 CEM cost (不只预测 latent), "
                 "或转向真需要长记忆的任务 (组合 dynamics / 长程因果).")
    txt = "\n".join(L)
    Path("csv_data/horizon_sweep_report.md").write_text(txt)
    print("\n" + "=" * 78)
    print(txt)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--envs", nargs="*", default=ENVS)
    ap.add_argument("--horizons", nargs="*", type=int, default=HORIZONS)
    ap.add_argument("--seeds", nargs="*", type=int, default=SEEDS)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--episodes", type=int, default=60)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--latent_dim", type=int, default=32)
    ap.add_argument("--var_weight", type=float, default=4.0)
    ap.add_argument("--cem_samples", type=int, default=96)
    ap.add_argument("--cem_steps", type=int, default=6)
    ap.add_argument("--eval_episodes", type=int, default=12)
    ap.add_argument("--num_envs", type=int, default=4)
    ap.add_argument("--nworkers", type=int, default=0)
    args = ap.parse_args()

    all_tasks = [(env, m[0], m[1], m[2], m[3], seed, hor)
                 for env in args.envs for m in MODELS
                 for hor in args.horizons for seed in args.seeds]
    n_gpu = torch.cuda.device_count()
    nw = args.nworkers if args.nworkers > 0 else (min(n_gpu, len(all_tasks)) if n_gpu >= 1 else 1)
    for p in Path("csv_data").glob("horizon_sweep_shard*.csv"):
        p.unlink()
    print("=" * 78)
    print("horizon sweep: CTM 长程稳定性试金石")
    print("=" * 78)
    print(f"GPU={n_gpu} workers={nw}  {len(args.envs)} envs x {len(MODELS)} models "
          f"x {len(args.horizons)} horizons {args.horizons} x {len(args.seeds)} seeds = {len(all_tasks)} runs\n")
    t0 = time.time()
    if nw <= 1 or n_gpu < 1:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        bufs = {e: _collect_env(e, args) for e in args.envs}
        rows = []
        for (env, mname, mem, gate, kind, seed, hor) in all_tasks:
            args.horizon = hor
            try:
                r = run_one(env, mname, mem, gate, kind, seed, args.epochs,
                            bufs[env][0], bufs[env][1], args, device)
            except Exception as exc:
                r = dict(env=env, variant=mname, seed=seed, epochs=args.epochs, memory_length=mem,
                         state_gate=gate, success_rate=-1.0, random_rate=-1.0,
                         dynamics_err=-1.0, latent_var=-1.0, elapsed_s=0.0)
                print(f"[serial] ERROR {env}/{mname}/h{hor}: {exc}")
            r["horizon"] = hor
            rows.append(r)
            print(f"[serial] {env:<16}{mname:<12}h{hor} succ={r['success_rate']} [{r['elapsed_s']:.0f}s]")
    else:
        mp.spawn(_worker, args=(args, nw, all_tasks), nprocs=nw, join=True)
        rows = []
        for p in sorted(Path("csv_data").glob("horizon_sweep_shard*.csv")):
            with open(p) as f:
                for r in csv.DictReader(f):
                    for k in ("success_rate", "random_rate", "dynamics_err", "latent_var",
                              "elapsed_s", "epochs", "memory_length", "seed", "horizon"):
                        try:
                            r[k] = float(r[k])
                            if k in ("epochs", "memory_length", "seed", "horizon"):
                                r[k] = int(r[k])
                        except (ValueError, TypeError):
                            pass
                    rows.append(r)
    with open("csv_data/horizon_sweep_results.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=DEEP_FIELDS + ["horizon"]); wr.writeheader()
        for r in rows:
            wr.writerow({k: r.get(k, "") for k in DEEP_FIELDS + ["horizon"]})
    print(f"\n[total] {len(rows)}/{len(all_tasks)} runs, {(time.time()-t0)/60:.0f}min")
    report(rows)


if __name__ == "__main__":
    main()
