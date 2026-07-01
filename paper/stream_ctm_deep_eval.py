#!/usr/bin/env python3
"""stream-ctm 深度评估 (一晚跑完, 定论 stream-ctm vs jepa + gating/memory/epochs 全维度).

背景: dyn_err bug 修复后发现 (1) stream-ctm 的 dynamics 预测实际比 jepa 更准 (之前
"4-8x 差"是 step-0 初始值的幻觉); (2) 50 epochs 明显欠训 (pendulum 50->200ep,
success 33->66). 所以之前基于假 dyn_err 的 H1/H2 结论要重新用真实数据验证.

本脚本跑两个矩阵 (5 卡自动 spawn 并行):
  [full]  全 vector env x 5 variants x 5 seeds x 200 epochs  (定论对比)
  [sweep] 4 关键 env x stream-base x epochs[50,100,400] x 3 seeds (欠训趋势)

variants:
  jepa-mlp      Markov 对照
  stream-base   mem=8, no gate  (现状)
  stream-gate   mem=8, gru gate (验证 gating 增益是否稳健, 之前单 seed tworoom +16.6)
  stream-mem16  mem=16          (真实 dyn_err 下重验 memory 结论)
  stream-mem32  mem=32          (同上)

输出: csv_data/stream_ctm_deep_results.csv (逐 run) + stream_ctm_deep_report.md (报告)

跑法 (tmux 里前台, 下班 detach):
    tmux new -s deep
    python paper/stream_ctm_deep_eval.py
    # Ctrl-B D detach; 明早 tmux attach -t deep 或看报告
指定卡数 / 减负:
    python paper/stream_ctm_deep_eval.py --nworkers 5 --seeds 0 1 2
"""
import argparse, csv, os, sys, time
from collections import defaultdict
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch

from diagnose_stream_ctm_failures import build_one, evaluate, _collect_env
from worldmodel.envs import make_env
from worldmodel.train import train_world_model

ENVS = ["pendulum", "pendulum-partial", "cartpole", "cartpole-partial",
        "tworoom-state", "point-state", "reacher"]
VARIANTS = [
    ("jepa-mlp",    8,  "none", "jepa"),
    ("stream-base", 8,  "none", "stream"),
    ("stream-gate", 8,  "gru",  "stream"),
    ("stream-mem16",16, "none", "stream"),
    ("stream-mem32",32, "none", "stream"),
]
DEFAULT_SEEDS = [0, 1, 2, 3, 4]
FULL_EPOCHS = 200
SWEEP_EPOCHS = [50, 100, 400]          # 200 在 full 矩阵里
SWEEP_ENVS = ["pendulum", "pendulum-partial", "cartpole-partial", "tworoom-state"]
SWEEP_SEEDS = [0, 1, 2]
DEEP_FIELDS = ["env", "variant", "seed", "epochs", "memory_length", "state_gate",
               "success_rate", "random_rate", "dynamics_err", "latent_var", "elapsed_s"]


def run_one(env_name, vname, mem, gate, kind, seed, epochs, buf, env_kw, args, device):
    obs_key = "pixels" if env_name == "point-image" else "state"
    env = make_env(env_name, **env_kw)
    obs_shape, action_dim = env.observation_space.shape, env.action_space.shape[0]
    t0 = time.time()
    torch.manual_seed(seed); np.random.seed(seed)
    model = build_one(kind, mem, gate, obs_key, obs_shape, action_dim,
                      args.latent_dim, args.var_weight, device)
    hist = train_world_model(model, buf, horizon=args.horizon, epochs=epochs,
                             batch_size=args.batch_size, device=device,
                             log_every=10**9, seed=seed)
    last = hist[-1] if hist else {}
    model.eval()
    succ, rand = evaluate(model, env_name, env_kw, args.num_envs,
                          args.cem_samples, args.cem_steps, args.horizon,
                          args.eval_episodes, device)
    return dict(env=env_name, variant=vname, seed=seed, epochs=epochs,
                memory_length=mem, state_gate=gate,
                success_rate=round(succ, 1), random_rate=round(rand, 1),
                dynamics_err=round(float(last.get("dynamics_err", float("nan"))), 5),
                latent_var=round(float(last.get("latent_var", float("nan"))), 5),
                elapsed_s=round(time.time() - t0, 1))


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
    out = f"csv_data/stream_ctm_deep_shard{rank}.csv"
    rows = []
    with open(out, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=DEEP_FIELDS); wr.writeheader()
        for (env, vname, mem, gate, kind, seed, ep) in my:
            try:
                r = run_one(env, vname, mem, gate, kind, seed, ep,
                            bufs[env][0], bufs[env][1], args, device)
            except Exception as exc:
                r = dict(env=env, variant=vname, seed=seed, epochs=ep, memory_length=mem,
                         state_gate=gate, success_rate=-1.0, random_rate=-1.0,
                         dynamics_err=-1.0, latent_var=-1.0, elapsed_s=0.0)
                print(f"[gpu{rank}] ERROR {env}/{vname}/s{seed}/ep{ep}: {exc}", flush=True)
            wr.writerow(r); f.flush(); rows.append(r)
            print(f"[gpu{rank}] {env:<16}{vname:<13}s{seed} ep{ep:<4} "
                  f"succ={r['success_rate']:5.1f} dyn={r['dynamics_err']} "
                  f"var={r['latent_var']} [{r['elapsed_s']:.0f}s]", flush=True)
    print(f"[gpu{rank}] done {len(rows)} -> {out}", flush=True)


def _mean_std(xs):
    xs = [x for x in xs if x is not None and x == x and x >= 0]  # drop nan/-1
    if not xs:
        return float("nan"), float("nan")
    m = float(np.mean(xs))
    s = float(np.std(xs)) if len(xs) > 1 else 0.0
    return m, s


def _pooled_se(s1, n1, s2, n2):
    if n1 < 1 or n2 < 1:
        return float("nan")
    return float(np.sqrt(s1**2 / max(n1, 1) + s2**2 / max(n2, 1)))


def report(rows, report_path):
    L = []
    L.append("# stream-ctm 深度评估报告\n")
    L.append(f"生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}  |  总 runs: {len(rows)}\n")

    full = [r for r in rows if r["epochs"] == FULL_EPOCHS]
    by = defaultdict(list)   # (env,variant) -> [succ]
    dby = defaultdict(list)  # (env,variant) -> [dyn_err]
    for r in full:
        by[(r["env"], r["variant"])].append(r["success_rate"])
        dby[(r["env"], r["variant"])].append(r["dynamics_err"])
    envs = sorted({r["env"] for r in full})
    variants = [v[0] for v in VARIANTS]

    # Table 1: success_rate mean+-std
    L.append("\n## 1. success_rate mean+-std (200 epochs)\n")
    L.append("| env | " + " | ".join(variants) + " |")
    L.append("|" + "---|" * (len(variants) + 1))
    stats = {}
    for env in envs:
        cells = [env]
        for v in variants:
            m, s = _mean_std(by.get((env, v), []))
            stats[(env, v)] = (m, s, len([x for x in by.get((env, v), []) if x >= 0]))
            cells.append(f"{m:.1f}+-{s:.1f} (n={stats[(env,v)][2]})")
        L.append("| " + " | ".join(cells) + " |")

    # Table 2: dynamics_err mean+-std
    L.append("\n## 2. dynamics_err mean+-std (200 epochs, 越低越好)\n")
    L.append("| env | " + " | ".join(variants) + " |")
    L.append("|" + "---|" * (len(variants) + 1))
    for env in envs:
        cells = [env]
        for v in variants:
            m, s = _mean_std(dby.get((env, v), []))
            cells.append(f"{m:.5f}+-{s:.5f}")
        L.append("| " + " | ".join(cells) + " |")

    # Table 3: gating 增益 (stream-gate - stream-base)
    L.append("\n## 3. gating 增益 (stream-gate - stream-base, 显著性: delta > 2*pooled_se)\n")
    L.append("| env | base succ | gate succ | delta | pooled_se | 判定 |")
    L.append("|---|---|---|---|---|---|")
    for env in envs:
        mb, sb, nb = stats.get((env, "stream-base"), (float("nan"),) * 3 if False else (float("nan"), float("nan"), 0))
        mg, sg, ng = stats.get((env, "stream-gate"), (float("nan"), float("nan"), 0))
        if nb and ng:
            delta = mg - mb
            se = _pooled_se(sb, nb, sg, ng)
            flag = "显著 +" if (delta > 0 and delta > 2 * se) else ("显著 -" if (delta < 0 and -delta > 2 * se) else "不显著")
            L.append(f"| {env} | {mb:.1f} | {mg:.1f} | {delta:+.1f} | {se:.1f} | {flag} |")

    # Table 4: stream vs jepa
    L.append("\n## 4. stream-base vs jepa-mlp (显著性: delta > 2*pooled_se)\n")
    L.append("| env | jepa succ | stream succ | delta | pooled_se | 判定 |")
    L.append("|---|---|---|---|---|---|")
    for env in envs:
        mj, sj, nj = stats.get((env, "jepa-mlp"), (float("nan"), float("nan"), 0))
        ms, ss, ns = stats.get((env, "stream-base"), (float("nan"), float("nan"), 0))
        if nj and ns:
            delta = ms - mj
            se = _pooled_se(sj, nj, ss, ns)
            flag = "stream 显著赢" if (delta > 0 and delta > 2 * se) else ("jepa 显著赢" if (delta < 0 and -delta > 2 * se) else "持平")
            L.append(f"| {env} | {mj:.1f} | {ms:.1f} | {delta:+.1f} | {se:.1f} | {flag} |")

    # Table 5: memory (真实 dyn_err 下重验)
    L.append("\n## 5. memory_length 影响 (stream-mem16/mem32 vs base, 200ep)\n")
    L.append("| env | base succ | mem16 succ | mem32 succ | base dyn | mem16 dyn | mem32 dyn |")
    L.append("|---|---|---|---|---|---|---|")
    for env in envs:
        b = stats.get((env, "stream-base"), (float("nan"), float("nan"), 0))[0]
        m16 = stats.get((env, "stream-mem16"), (float("nan"), float("nan"), 0))[0]
        m32 = stats.get((env, "stream-mem32"), (float("nan"), float("nan"), 0))[0]
        db = _mean_std(dby.get((env, "stream-base"), []))[0]
        d16 = _mean_std(dby.get((env, "stream-mem16"), []))[0]
        d32 = _mean_std(dby.get((env, "stream-mem32"), []))[0]
        L.append(f"| {env} | {b:.1f} | {m16:.1f} | {m32:.1f} | {db:.5f} | {d16:.5f} | {d32:.5f} |")

    # Table 6: epochs sweep (欠训趋势)
    L.append("\n## 6. epochs sweep (stream-base, 看欠训: succ 随 epochs 升 = 之前是欠训)\n")
    sweep = [r for r in rows if r["variant"] == "stream-base"
             and r["env"] in SWEEP_ENVS and r["epochs"] in (SWEEP_EPOCHS + [FULL_EPOCHS])]
    L.append("| env | ep50 | ep100 | ep200 | ep400 |")
    L.append("|---|---|---|---|---|")
    for env in SWEEP_ENVS:
        cells = [env]
        for ep in [50, 100, 200, 400]:
            xs = [r["success_rate"] for r in sweep if r["env"] == env and r["epochs"] == ep]
            m, _ = _mean_std(xs)
            cells.append(f"{m:.1f}" if m == m else "-")
        L.append("| " + " | ".join(cells) + " |")
    L.append("\n判读: 若 ep50->ep400 succ 持续升 => 之前 50ep 严重欠训, 应提默认 epochs; "
             "若 ep200 后平台 => 已收敛, 失败是任务难度不是训练量.")

    # latent_var collapse check
    L.append("\n## 7. encoder 崩塌检查 (latent_var < 1e-3 = 崩塌)\n")
    collapsed = [r for r in full if r["latent_var"] == r["latent_var"] and r["latent_var"] < 1e-3]
    if collapsed:
        for r in collapsed:
            L.append(f"- {r['env']}/{r['variant']}/s{r['seed']} var={r['latent_var']} (CEM 退化)")
    else:
        L.append("- 无崩塌")

    txt = "\n".join(L)
    Path(report_path).write_text(txt)
    print("\n" + "=" * 78)
    print(txt)
    print(f"\n[done] 报告 -> {report_path}")
    print(f"[done] 原始数据 -> csv_data/stream_ctm_deep_results.csv")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--envs", nargs="*", default=ENVS)
    ap.add_argument("--seeds", nargs="*", type=int, default=DEFAULT_SEEDS)
    ap.add_argument("--full-epochs", type=int, default=FULL_EPOCHS)
    ap.add_argument("--episodes", type=int, default=60)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--horizon", type=int, default=6)
    ap.add_argument("--latent_dim", type=int, default=32)
    ap.add_argument("--var_weight", type=float, default=4.0)
    ap.add_argument("--cem_samples", type=int, default=128)
    ap.add_argument("--cem_steps", type=int, default=6)
    ap.add_argument("--eval_episodes", type=int, default=16)
    ap.add_argument("--num_envs", type=int, default=4)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--nworkers", type=int, default=0, help="0=自动=GPU数")
    ap.add_argument("--no-sweep", action="store_true", help="只跑 full 矩阵, 跳过 epochs sweep")
    ap.add_argument("--report", default="csv_data/stream_ctm_deep_report.md")
    args = ap.parse_args()

    # full 矩阵
    full_tasks = [(env, v[0], v[1], v[2], v[3], seed, args.full_epochs)
                  for env in args.envs for v in VARIANTS for seed in args.seeds]
    # epochs sweep (stream-base, 去掉已在 full 的 full_epochs)
    sweep_eps = [e for e in SWEEP_EPOCHS if e != args.full_epochs]
    sweep_tasks = []
    if not args.no_sweep:
        sweep_tasks = [(env, "stream-base", 8, "none", "stream", seed, ep)
                       for env in args.envs if env in SWEEP_ENVS
                       for ep in sweep_eps for seed in SWEEP_SEEDS]
    all_tasks = full_tasks + sweep_tasks

    n_gpu = torch.cuda.device_count()
    nw = args.nworkers if args.nworkers > 0 else (min(n_gpu, len(all_tasks)) if n_gpu >= 1 else 1)
    for p in Path("csv_data").glob("stream_ctm_deep_shard*.csv"):
        p.unlink()

    print("=" * 78)
    print("stream-ctm 深度评估")
    print("=" * 78)
    print(f"GPU={n_gpu}  workers={nw}")
    print(f"[full]  {len(args.envs)} envs x {len(VARIANTS)} variants x {len(args.seeds)} seeds "
          f"x {args.full_epochs} ep = {len(full_tasks)} runs")
    if sweep_tasks:
        print(f"[sweep] {len([e for e in args.envs if e in SWEEP_ENVS])} envs x stream-base "
              f"x {len(sweep_eps)} epochs {sweep_eps} x {len(SWEEP_SEEDS)} seeds = {len(sweep_tasks)} runs")
    print(f"总计 {len(all_tasks)} runs, 预计 ~{len(all_tasks) * 3.5 / max(nw,1) / 60:.1f}h (5卡估)\n")

    t0 = time.time()
    if nw <= 1 or n_gpu < 1:
        device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
        bufs = {e: _collect_env(e, args) for e in args.envs}
        rows = []
        for (env, vname, mem, gate, kind, seed, ep) in all_tasks:
            try:
                r = run_one(env, vname, mem, gate, kind, seed, ep,
                            bufs[env][0], bufs[env][1], args, device)
            except Exception as exc:
                r = dict(env=env, variant=vname, seed=seed, epochs=ep, memory_length=mem,
                         state_gate=gate, success_rate=-1.0, random_rate=-1.0,
                         dynamics_err=-1.0, latent_var=-1.0, elapsed_s=0.0)
                print(f"[serial] ERROR {env}/{vname}/s{seed}/ep{ep}: {exc}")
            rows.append(r)
            print(f"[serial] {env:<16}{vname:<13}s{seed} ep{ep:<4} succ={r['success_rate']:5.1f} [{r['elapsed_s']:.0f}s]")
    else:
        torch.multiprocessing.spawn(_worker, args=(args, nw, all_tasks), nprocs=nw, join=True)
        rows = []
        for p in sorted(Path("csv_data").glob("stream_ctm_deep_shard*.csv")):
            with open(p) as f:
                for r in csv.DictReader(f):
                    for k in ("success_rate", "random_rate", "dynamics_err", "latent_var",
                              "elapsed_s", "epochs", "memory_length", "seed"):
                        try:
                            r[k] = float(r[k])
                            if k in ("epochs", "memory_length", "seed"):
                                r[k] = int(r[k])
                        except (ValueError, TypeError):
                            pass
                    rows.append(r)

    # 写合并 CSV
    Path("csv_data").mkdir(exist_ok=True)
    with open("csv_data/stream_ctm_deep_results.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=DEEP_FIELDS); wr.writeheader()
        for r in rows:
            wr.writerow({k: r.get(k, "") for k in DEEP_FIELDS})
    print(f"\n[total] {len(rows)}/{len(all_tasks)} runs, 耗时 {(time.time()-t0)/60:.0f}min")
    if len(rows) < len(all_tasks):
        print(f"[warn] 缺失 {len(all_tasks)-len(rows)} runs (见 ERROR 行)")
    report(rows, args.report)


if __name__ == "__main__":
    main()
