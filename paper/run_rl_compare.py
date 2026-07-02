#!/usr/bin/env python3
"""对比三条路线: 谁能让 CTM 在 world-model/控制任务上发挥价值.

  cem-jepa   : CEM + JEPA world-model          (基线: 外挂规划器 + Markov 预测)
  cem-ctm    : CEM + stream-ctm world-model    (CTM 当被动 predictor — 已知吃亏)
  ppo-mlp    : PPO + Markov MLP policy          (端到端基线, 无 world-model)
  ppo-ctm    : PPO + CTM policy (Route 1)       (CTM 直接当大脑, 原版 CTM-RL 路线)
  dreamer    : CTM world-model + 想象 actor-critic (Route 2, CTM 长程稳定服务想象规划)

矩阵: 7 state envs x 5 methods x 5 seeds = 175 runs. 5 卡 spawn (每卡多 worker),
出 mean+-std 对比表, 看 CTM 在哪条路线、哪些任务上真正赢.

跑法 (tmux, 5卡):
    python paper/run_rl_compare.py
    python paper/run_rl_compare.py --procs-per-gpu 8      # 填满 CPU
    python paper/run_rl_compare.py --total-steps 100000   # ppo/dreamer 训练步数
"""
import argparse, csv, os, sys, time
from collections import defaultdict
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch

from stream_ctm_deep_eval import build_one, evaluate, _collect_env, run_one
from worldmodel.envs import make_env
from worldmodel.train import train_world_model
from worldmodel.rl.ppo import PPOTrainer, build_policy
from worldmodel.rl.dreamer import DreamerTrainer

ENVS = ["pendulum", "pendulum-partial", "cartpole", "cartpole-partial",
        "tworoom-state", "point-state", "reacher"]
METHODS = ["cem-jepa", "cem-ctm", "ppo-mlp", "ppo-ctm", "dreamer"]
SEEDS = [0, 1, 2, 3, 4]
FIELDS = ["env", "method", "seed", "success_rate", "dynamics_err", "elapsed_s"]


def run_cem(env, kind, seed, args, device):
    """kind: 'jepa' or 'stream'. 复用 world-model train + CEM eval."""
    buf, env_kw = _collect_env(env, args)
    vname = "jepa-mlp" if kind == "jepa" else "stream-base"
    kk = "jepa" if kind == "jepa" else "stream"
    row = run_one(env, vname, 8, "none", kk, seed, args.cem_epochs, buf, env_kw, args, device)
    return float(row["success_rate"]), float(row.get("dynamics_err", -1))


def run_ppo(env, kind, seed, args, device):
    """kind: 'mlp' or 'ctm'. Route 1: CTM as end-to-end policy."""
    e = make_env(env)
    od = int(np.prod(e.observation_space.shape))
    gd = int(np.prod(e.goal_space.shape))
    ad = int(np.prod(e.action_space.shape))
    pol = build_policy(kind, od, gd, ad, d_model=args.d_model,
                       memory_length=args.memory_length, state_gate=args.state_gate)
    t = PPOTrainer(env, pol, num_envs=args.ppo_envs, num_steps=args.ppo_steps,
                   device=device, lr=args.ppo_lr)
    torch.manual_seed(seed); np.random.seed(seed)
    hist = t.train(args.total_steps, log_iters=10, eval_episodes=args.eval_episodes, seed=seed)
    succ = 0.0
    for h in reversed(hist):
        if "eval_success" in h:
            succ = h["eval_success"]; break
    return succ, -1.0


def run_dreamer(env, seed, args, device):
    """Route 2: CTM world-model + imagination actor-critic."""
    torch.manual_seed(seed); np.random.seed(seed)
    t = DreamerTrainer(env, latent_dim=args.latent_dim, d_model=args.d_model,
                       memory_length=args.memory_length, state_gate=args.state_gate,
                       num_envs=args.ppo_envs, collect_steps=args.ppo_steps,
                       device=device, imagine_horizon=args.imagine_horizon,
                       var_weight=args.var_weight)
    hist = t.train(args.total_steps, log_iters=10, eval_episodes=args.eval_episodes,
                   seed=seed, H_wm=args.cem_horizon)
    succ = 0.0
    for h in reversed(hist):
        if "eval_success" in h:
            succ = h["eval_success"]; break
    return succ, float(hist[-1].get("dyn_err", -1)) if hist else -1.0


def run_task(env, method, seed, args, device):
    t0 = time.time()
    try:
        if method == "cem-jepa":
            s, d = run_cem(env, "jepa", seed, args, device)
        elif method == "cem-ctm":
            s, d = run_cem(env, "stream", seed, args, device)
        elif method == "ppo-mlp":
            s, d = run_ppo(env, "mlp", seed, args, device)
        elif method == "ppo-ctm":
            s, d = run_ppo(env, "ctm", seed, args, device)
        elif method == "dreamer":
            s, d = run_dreamer(env, seed, args, device)
        else:
            raise KeyError(method)
    except Exception as exc:
        print(f"  ERROR {env}/{method}/s{seed}: {exc}", flush=True)
        s, d = -1.0, -1.0
    return {"env": env, "method": method, "seed": seed, "success_rate": round(s, 1),
            "dynamics_err": round(d, 5), "elapsed_s": round(time.time() - t0, 1)}


def _worker(rank, args, nworkers, n_gpu, all_tasks):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(rank % max(n_gpu, 1))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    my = all_tasks[rank::nworkers]
    Path("csv_data").mkdir(exist_ok=True)
    out = f"csv_data/rl_compare_shard{rank}.csv"
    rows = []
    with open(out, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=FIELDS); wr.writeheader()
        for (env, method, seed) in my:
            r = run_task(env, method, seed, args, device)
            wr.writerow(r); f.flush(); rows.append(r)
            print(f"[gpu{rank}] {env:<16}{method:<10}s{seed} succ={r['success_rate']:5.1f} "
                  f"[{r['elapsed_s']:.0f}s]", flush=True)
    print(f"[gpu{rank}] done {len(rows)} -> {out}", flush=True)


def _ms(xs):
    xs = [x for x in xs if x is not None and x == x and x >= 0]
    if not xs:
        return float("nan"), float("nan")
    return float(np.mean(xs)), (float(np.std(xs)) if len(xs) > 1 else 0.0)


def report(rows, envs, path):
    by = defaultdict(list)
    for r in rows:
        by[(r["env"], r["method"])].append(r["success_rate"])
    L = ["# RL 路线对比: 三条路线谁让 CTM 发挥价值\n",
         f"生成: {time.strftime('%Y-%m-%d %H:%M:%S')} | runs: {len(rows)}\n"]
    L.append("\n## success_rate mean+-std\n")
    L.append("| env | " + " | ".join(METHODS) + " |")
    L.append("|" + "---|" * (len(METHODS) + 1))
    stats = {}
    for env in envs:
        cells = [env]
        for m in METHODS:
            mn, sd = _ms(by.get((env, m), []))
            stats[(env, m)] = (mn, sd)
            cells.append(f"{mn:.1f}+-{sd:.1f}" if mn == mn else "-")
        L.append("| " + " | ".join(cells) + " |")
    # 每 env 的赢家
    L.append("\n## 每 env 最佳路线 (CTM 路线是否赢过基线)\n")
    L.append("| env | 最佳 method | succ | cem-jepa(基线) | ppo-ctm(Route1) | dreamer(Route2) |")
    L.append("|---|---|---|---|---|---|")
    for env in envs:
        best_m, best_s = None, -1
        for m in METHODS:
            mn, _ = stats.get((env, m), (float("nan"), 0))
            if mn == mn and mn > best_s:
                best_s, best_m = mn, m
        jepa = stats.get((env, "cem-jepa"), (float("nan"),))[0]
        ppoctm = stats.get((env, "ppo-ctm"), (float("nan"),))[0]
        drm = stats.get((env, "dreamer"), (float("nan"),))[0]
        L.append(f"| {env} | {best_m} | {best_s:.1f} | {jepa:.1f} | {ppoctm:.1f} | {drm:.1f} |")
    # Route 聚合: CTM 两条路线 vs 基线
    L.append("\n## 路线聚合 (CTM 价值总结)\n")
    for route, members in [("cem-ctm(被动predictor)", ["cem-ctm"]),
                           ("ppo-ctm(Route1 直接policy)", ["ppo-ctm"]),
                           ("dreamer(Route2 想象规划)", ["dreamer"])]:
        deltas = []
        for env in envs:
            bm, _ = stats.get((env, members[0]), (float("nan"), 0))
            jm, _ = stats.get((env, "cem-jepa"), (float("nan"), 0))
            if bm == bm and jm == jm:
                deltas.append(bm - jm)
        if deltas:
            L.append(f"- {route}: 平均 delta vs cem-jepa = {np.mean(deltas):+.1f}pp "
                     f"(赢 {sum(1 for d in deltas if d > 0)}/{len(deltas)} envs)")
    txt = "\n".join(L)
    Path(path).write_text(txt)
    print("\n" + "=" * 78)
    print(txt)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--envs", nargs="*", default=ENVS)
    ap.add_argument("--methods", nargs="*", default=METHODS)
    ap.add_argument("--seeds", nargs="*", type=int, default=SEEDS)
    ap.add_argument("--total-steps", type=int, default=80000, help="ppo/dreamer 训练步数")
    ap.add_argument("--cem-epochs", type=int, default=200)
    ap.add_argument("--cem-horizon", type=int, default=6)
    ap.add_argument("--ppo-envs", type=int, default=8)
    ap.add_argument("--ppo-steps", type=int, default=256)
    ap.add_argument("--ppo-lr", type=float, default=3e-4)
    ap.add_argument("--d_model", type=int, default=128)
    ap.add_argument("--memory_length", type=int, default=8)
    ap.add_argument("--state_gate", default="gru")
    ap.add_argument("--latent_dim", type=int, default=32)
    ap.add_argument("--var_weight", type=float, default=4.0)
    ap.add_argument("--imagine_horizon", type=int, default=10)
    ap.add_argument("--episodes", type=int, default=60)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--cem_samples", type=int, default=128)
    ap.add_argument("--cem_steps", type=int, default=6)
    ap.add_argument("--eval_episodes", type=int, default=12)
    ap.add_argument("--num_envs", type=int, default=4)
    ap.add_argument("--horizon", type=int, default=6)
    ap.add_argument("--nworkers", type=int, default=0)
    ap.add_argument("--procs-per-gpu", type=int, default=8)
    ap.add_argument("--report", default="csv_data/rl_compare_report.md")
    args = ap.parse_args()

    all_tasks = [(env, m, seed) for env in args.envs for m in args.methods for seed in args.seeds]
    n_gpu = torch.cuda.device_count()
    ppg = max(1, args.procs_per_gpu)
    nw = args.nworkers if args.nworkers > 0 else (min(n_gpu * ppg, len(all_tasks)) if n_gpu >= 1 else 1)
    for p in Path("csv_data").glob("rl_compare_shard*.csv"):
        p.unlink()
    print("=" * 78)
    print("RL 路线对比: cem-jepa / cem-ctm / ppo-mlp / ppo-ctm / dreamer")
    print("=" * 78)
    print(f"GPU={n_gpu} workers={nw} ({ppg}/gpu)  "
          f"{len(args.envs)} envs x {len(args.methods)} methods x {len(args.seeds)} seeds "
          f"= {len(all_tasks)} runs\n")
    t0 = time.time()
    if nw <= 1 or n_gpu < 1:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        rows = [run_task(e, m, s, args, device) for (e, m, s) in all_tasks]
    else:
        torch.multiprocessing.spawn(_worker, args=(args, nw, n_gpu, all_tasks), nprocs=nw, join=True)
        rows = []
        for p in sorted(Path("csv_data").glob("rl_compare_shard*.csv")):
            with open(p) as f:
                for r in csv.DictReader(f):
                    for k in ("success_rate", "dynamics_err", "elapsed_s", "seed"):
                        try:
                            r[k] = float(r[k])
                            if k == "seed":
                                r[k] = int(r[k])
                        except (ValueError, TypeError):
                            pass
                    rows.append(r)
    with open("csv_data/rl_compare_results.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=FIELDS); wr.writeheader()
        for r in rows:
            wr.writerow({k: r.get(k, "") for k in FIELDS})
    print(f"\n[total] {len(rows)}/{len(all_tasks)} runs, {(time.time()-t0)/60:.0f}min")
    report(rows, args.envs, args.report)


if __name__ == "__main__":
    main()
