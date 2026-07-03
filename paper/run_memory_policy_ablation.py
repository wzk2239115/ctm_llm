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
from stream_ctm_deep_eval import _collect_env

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
    is_image = len(e.observation_space.shape) == 3
    pol = build_memory_policy(backbone, od, gd, ad, latent_dim=args.latent_dim,
                              d_model=args.d_model, memory_length=args.memory_length,
                              state_gate="gru", image=is_image)
    t = PPOTrainer(env, pol, num_envs=args.ppo_envs, num_steps=args.ppo_steps,
                   device=device, lr=args.lr)
    if args.bc_steps > 0:
        buf, _ = _collect_env(env, args)
        t.bc_pretrain(buf, bc_steps=args.bc_steps)
    torch.manual_seed(seed); np.random.seed(seed)
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


def report(rows, envs, path, backbones):
    by = defaultdict(list)
    for r in rows:
        by[(r["env"], r["backbone"])].append(r["success_rate"])
    L = ["# Memory-policy ablation: CTM vs RNN/Transformer memory policies\n",
         f"生成: {time.strftime('%Y-%m-%d %H:%M:%S')} | runs: {len(rows)} | backbones: {backbones}\n"]
    L.append("\n## success_rate mean+-std\n")
    L.append("| env | " + " | ".join(backbones) + " |")
    L.append("|" + "---|" * (len(backbones) + 1))
    stats = {}
    for env in envs:
        cells = [env]
        for bb in backbones:
            mn, sd = _ms(by.get((env, bb), []))
            stats[(env, bb)] = mn
            cells.append(f"{mn:.1f}+-{sd:.1f}" if mn == mn else "-")
        L.append("| " + " | ".join(cells) + " |")

    # Flash 混合 vs 单路径 (flash vs flash-shallow vs flash-deep)
    if "flash" in backbones:
        L.append("\n## Flash 混合 vs 单路径 (混合>单路径 坐实 fig4 修正版)\n")
        L.append("| env | flash(混合) | flash-shallow(z=0) | flash-deep(z=1) | 混合-shallow | 混合-deep |")
        L.append("|---|---|---|---|---|---|")
        for env in envs:
            f = stats.get((env, "flash"), float("nan"))
            fs = stats.get((env, "flash-shallow"), float("nan"))
            fd = stats.get((env, "flash-deep"), float("nan"))
            d_s = (f - fs) if (f == f and fs == fs) else float("nan")
            d_d = (f - fd) if (f == f and fd == fd) else float("nan")
            L.append(f"| {env} | {f:.1f} | {fs if fs!=fs else f'{fs:.1f}'} | "
                     f"{fd if fd!=fd else f'{fd:.1f}'} | {d_s if d_s!=d_s else f'{d_s:+.1f}'} | "
                     f"{d_d if d_d!=d_d else f'{d_d:+.1f}'} |")

    # CTM vs RNN 系 (若有 lstm/gru/transformer)
    rnn_bbs = [b for b in ("lstm", "gru", "transformer") if b in backbones]
    if "ctm" in backbones and rnn_bbs:
        L.append("\n## CTM vs RNN 系记忆策略\n")
        L.append("| env | CTM | RNN 均值 | CTM-RNN | 判定 |")
        L.append("|---|---|---|---|---|")
        ctm_wins = []
        for env in envs:
            ctm = stats.get((env, "ctm"), float("nan"))
            rnns = [x for x in (stats.get((env, b), float("nan")) for b in rnn_bbs) if x == x]
            rnn_mean = float(np.mean(rnns)) if rnns else float("nan")
            delta = ctm - rnn_mean if (ctm == ctm and rnn_mean == rnn_mean) else float("nan")
            flag = "CTM 赢" if delta > 2 else ("CTM 输" if delta < -2 else "持平")
            if delta == delta and delta > 2:
                ctm_wins.append(env)
            L.append(f"| {env} | {ctm:.1f} | {rnn_mean:.1f} | {delta:+.1f} | {flag} |")

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
    ap.add_argument("--bc-steps", type=int, default=0,
                    help="behaviour-cloning warm start on collected data (>0 = fair vs world-model pretrain)")
    ap.add_argument("--episodes", type=int, default=60, help="collect episodes for BC/world-model data")
    ap.add_argument("--nworkers", type=int, default=0)
    ap.add_argument("--procs-per-gpu", type=int, default=8)
    ap.add_argument("--report", default="csv_data/memory_ablation_report.md")
    ap.add_argument("--report-only", action="store_true",
                    help="跳过训练, 从 csv 重新出 report (report 列不全/想换 backbones 显示时用)")
    ap.add_argument("--csv", default="csv_data/memory_ablation_results.csv")
    args = ap.parse_args()

    if args.report_only:
        rows = []
        if Path(args.csv).exists():
            with open(args.csv) as f:
                for r in csv.DictReader(f):
                    for k in ("success_rate", "seed"):
                        try:
                            r[k] = float(r[k])
                            if k == "seed":
                                r[k] = int(r[k])
                        except (ValueError, KeyError, TypeError):
                            pass
                    rows.append(r)
        envs = sorted({r["env"] for r in rows}) or args.envs
        report(rows, envs, args.report, args.backbones)
        return

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
    report(rows, args.envs, args.report, args.backbones)


if __name__ == "__main__":
    main()
