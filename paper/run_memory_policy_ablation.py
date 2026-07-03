#!/usr/bin/env python3
"""Memory-policy ablation: CTM vs RNN/Transformer memory policies (same encoder).

GPT 诊断的核心结论: CTM 不该和 world-model 比, 该和 memory-based policy 比
(RNN/LSTM/GRU/Transformer policy), 在 POMDP 上证明 CTM 的持续思考比标准循环
记忆更好/更稳. 本脚本就是这个干净对照:

  encoder (obs+goal -> latent, 端到端 fine-tune)  -- 所有 backbone 共享同一结构
     |
  memory backbone:  mlp | ctm | lstm | gru | transformer   -- 只换这个
     |
  actor (Gaussian) + critic

矩阵: 5 envs (POMDP 为主) x 5 backbones x 5 seeds = 125 runs. 5 卡 spawn.
回答: CTM 记忆是否优于 RNN 系? 记忆 policy 是否优于 Markov(mlp)?

跑法:
    python paper/run_memory_policy_ablation.py
    python paper/run_memory_policy_ablation.py --procs-per-gpu 10
"""
import argparse, csv, os, sys, time
from collections import defaultdict
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch

from worldmodel.envs import make_env
from worldmodel.rl.ppo import PPOTrainer
from worldmodel.rl.memory_policy import build_memory_policy

ENVS = ["pendulum", "pendulum-partial", "pendulum-delay3", "pendulum-partial-delay3",
        "cartpole-partial", "tworoom-state", "point-state"]
BACKBONES = ["mlp", "ctm", "lstm", "gru", "transformer", "flash"]
SEEDS = [0, 1, 2, 3, 4]
FIELDS = ["env", "backbone", "seed", "success_rate", "elapsed_s"]


def run_one(env, backbone, seed, args, device):
    e = make_env(env)
    od = int(np.prod(e.observation_space.shape))
    gd = int(np.prod(e.goal_space.shape))
    ad = int(np.prod(e.action_space.shape))
    pol = build_memory_policy(backbone, od, gd, ad, latent_dim=args.latent_dim,
                              d_model=args.d_model, memory_length=args.memory_length,
                              state_gate="gru")
    t = PPOTrainer(env, pol, num_envs=args.ppo_envs, num_steps=args.ppo_steps,
                   device=device, lr=args.lr)
    torch.manual_seed(seed); np.random.seed(seed)
    t0 = time.time()
    hist = t.train(args.total_steps, log_iters=8, eval_episodes=args.eval_episodes, seed=seed)
    succ = 0.0
    for h in reversed(hist):
        if "eval_success" in h:
            succ = h["eval_success"]; break
    return {"env": env, "backbone": backbone, "seed": seed,
            "success_rate": round(succ, 1), "elapsed_s": round(time.time() - t0, 1)}


def _worker(rank, args, nworkers, n_gpu, all_tasks):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(rank % max(n_gpu, 1))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    my = all_tasks[rank::nworkers]
    Path("csv_data").mkdir(exist_ok=True)
    out = f"csv_data/memory_ablation_shard{rank}.csv"
    rows = []
    with open(out, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=FIELDS); wr.writeheader()
        for (env, bb, seed) in my:
            try:
                r = run_one(env, bb, seed, args, device)
            except Exception as exc:
                r = {"env": env, "backbone": bb, "seed": seed,
                     "success_rate": -1.0, "elapsed_s": 0.0}
                print(f"[gpu{rank}] ERROR {env}/{bb}/s{seed}: {exc}", flush=True)
            wr.writerow(r); f.flush(); rows.append(r)
            print(f"[gpu{rank}] {env:<16}{bb:<12}s{seed} succ={r['success_rate']:5.1f} "
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
        by[(r["env"], r["backbone"])].append(r["success_rate"])
    L = ["# Memory-policy ablation: CTM vs RNN/Transformer memory policies\n",
         f"生成: {time.strftime('%Y-%m-%d %H:%M:%S')} | runs: {len(rows)}\n"]
    L.append("\n## success_rate mean+-std\n")
    L.append("| env | " + " | ".join(BACKBONES) + " |")
    L.append("|" + "---|" * (len(BACKBONES) + 1))
    stats = {}
    for env in envs:
        cells = [env]
        for bb in BACKBONES:
            mn, sd = _ms(by.get((env, bb), []))
            stats[(env, bb)] = mn
            cells.append(f"{mn:.1f}+-{sd:.1f}" if mn == mn else "-")
        L.append("| " + " | ".join(cells) + " |")

    # 每 env 最佳
    L.append("\n## 每 env 最佳 backbone\n")
    L.append("| env | 最佳 | succ | ctm | lstm | gru | transformer |")
    L.append("|---|---|---|---|---|---|---|")
    for env in envs:
        best_bb, best_s = None, -1
        for bb in BACKBONES:
            s = stats.get((env, bb), float("nan"))
            if s == s and s > best_s:
                best_s, best_bb = s, bb
        ctm = stats.get((env, "ctm"), float("nan"))
        lstm = stats.get((env, "lstm"), float("nan"))
        gru = stats.get((env, "gru"), float("nan"))
        tr = stats.get((env, "transformer"), float("nan"))
        L.append(f"| {env} | {best_bb} | {best_s:.1f} | {ctm:.1f} | {lstm:.1f} | {gru:.1f} | {tr:.1f} |")

    # 核心对比: CTM vs RNN 系 (这是 GPT 指出的真正对标)
    L.append("\n## CTM vs RNN 系记忆策略 (核心对标)\n")
    L.append("| env | CTM | RNN 均值(lstm/gru/tr) | CTM-RNN | 判定 |")
    L.append("|---|---|---|---|---|")
    ctm_wins = []
    for env in envs:
        ctm = stats.get((env, "ctm"), float("nan"))
        rnns = [stats.get((env, b), float("nan")) for b in ("lstm", "gru", "transformer")]
        rnns = [x for x in rnns if x == x]
        rnn_mean = float(np.mean(rnns)) if rnns else float("nan")
        delta = ctm - rnn_mean if (ctm == ctm and rnn_mean == rnn_mean) else float("nan")
        flag = "CTM 赢" if delta > 2 else ("CTM 输" if delta < -2 else "持平")
        if delta > 2:
            ctm_wins.append(env)
        L.append(f"| {env} | {ctm:.1f} | {rnn_mean:.1f} | {delta:+.1f} | {flag} |")

    # 记忆 vs Markov (记忆是否有用)
    L.append("\n## 记忆 policy vs Markov(mlp) — 记忆机制是否帮上\n")
    L.append("| env | mlp | 记忆均值 | 记忆-mlp |")
    L.append("|---|---|---|---|")
    for env in envs:
        mlp = stats.get((env, "mlp"), float("nan"))
        mems = [stats.get((env, b), float("nan")) for b in ("ctm", "lstm", "gru", "transformer")]
        mems = [x for x in mems if x == x]
        mem_mean = float(np.mean(mems)) if mems else float("nan")
        delta = mem_mean - mlp if (mem_mean == mem_mean and mlp == mlp) else float("nan")
        L.append(f"| {env} | {mlp:.1f} | {mem_mean:.1f} | {delta:+.1f} |")

    L.append("\n## 结论\n")
    if ctm_wins:
        L.append(f"CTM 在 {ctm_wins} 上显著优于 RNN 系记忆策略 —— CTM 持续思考作为 "
                 "memory policy 有真实价值 (尤其这些任务). 论文可立.")
    else:
        L.append("CTM 未显著优于 RNN 系 —— 说明标准循环记忆已够, CTM 无额外价值 "
                 "(或需调参/换更难记忆任务).")
    txt = "\n".join(L)
    Path(path).write_text(txt)
    print("\n" + "=" * 78)
    print(txt)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--envs", nargs="*", default=ENVS)
    ap.add_argument("--backbones", nargs="*", default=BACKBONES)
    ap.add_argument("--seeds", nargs="*", type=int, default=SEEDS)
    ap.add_argument("--total-steps", type=int, default=100000)
    ap.add_argument("--ppo-envs", type=int, default=8)
    ap.add_argument("--ppo-steps", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--d_model", type=int, default=128)
    ap.add_argument("--latent_dim", type=int, default=64)
    ap.add_argument("--memory_length", type=int, default=8)
    ap.add_argument("--eval_episodes", type=int, default=12)
    ap.add_argument("--nworkers", type=int, default=0)
    ap.add_argument("--procs-per-gpu", type=int, default=8)
    ap.add_argument("--report", default="csv_data/memory_ablation_report.md")
    args = ap.parse_args()

    all_tasks = [(e, b, s) for e in args.envs for b in args.backbones for s in args.seeds]
    n_gpu = torch.cuda.device_count()
    ppg = max(1, args.procs_per_gpu)
    nw = args.nworkers if args.nworkers > 0 else (min(n_gpu * ppg, len(all_tasks)) if n_gpu >= 1 else 1)
    for p in Path("csv_data").glob("memory_ablation_shard*.csv"):
        p.unlink()
    print("=" * 78)
    print("Memory-policy ablation: CTM vs lstm/gru/transformer (同 encoder)")
    print("=" * 78)
    print(f"GPU={n_gpu} workers={nw} ({ppg}/gpu)  "
          f"{len(args.envs)} envs x {len(args.backbones)} backbones x {len(args.seeds)} seeds "
          f"= {len(all_tasks)} runs\n")
    t0 = time.time()
    if nw <= 1 or n_gpu < 1:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        rows = [run_one(e, b, s, args, device) for (e, b, s) in all_tasks]
    else:
        torch.multiprocessing.spawn(_worker, args=(args, nw, n_gpu, all_tasks), nprocs=nw, join=True)
        rows = []
        for p in sorted(Path("csv_data").glob("memory_ablation_shard*.csv")):
            with open(p) as f:
                for r in csv.DictReader(f):
                    for k in ("success_rate", "elapsed_s", "seed"):
                        try:
                            r[k] = float(r[k])
                            if k == "seed":
                                r[k] = int(r[k])
                        except (ValueError, TypeError):
                            pass
                    rows.append(r)
    with open("csv_data/memory_ablation_results.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=FIELDS); wr.writeheader()
        for r in rows:
            wr.writerow({k: r.get(k, "") for k in FIELDS})
    print(f"\n[total] {len(rows)}/{len(all_tasks)} runs, {(time.time()-t0)/60:.0f}min")
    report(rows, args.envs, args.report)


if __name__ == "__main__":
    main()
