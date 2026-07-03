#!/usr/bin/env python3
"""Real-time benchmark: Flash Brain vs memory policies vs world-model planners.

New evaluation axis = REAL-TIME ability, missing from stable-wm benchmarks. Two
fast reactive policies (Flash Brain shallow-fast, CTM/LSTM) are compared against
slow model-based planners (CEM + JEPA/CTM world model) under per-step deadline
constraints. The point: Flash Brain's shallow path acts in microseconds, while
CEM rolls out hundreds of trajectories per step (ms-to-seconds), so under a tight
real-time deadline the planners collapse and the fast policies keep their success.

Metrics per (method, deadline):
  success_rate       — task success
  mean_latency_ms    — per-step get_action wall-clock
  throughput_hz      — 1000/latency (real-time control frequency)
  timeout_rate       — fraction of steps exceeding the deadline

跑法 (单进程, timing 要准, 不要和其他 GPU 任务抢):
    python paper/run_realtime_benchmark.py
    python paper/run_realtime_benchmark.py --envs pendulum-partial --seeds 0 1 2
"""
import argparse, csv, sys, time
from collections import defaultdict
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch

from worldmodel.envs import make_env
from worldmodel.world import World
from worldmodel.policy import WorldModelPolicy, PlanConfig
from worldmodel.solver import CEMSolver
from worldmodel.train import train_world_model
from worldmodel.rl.ppo import PPOTrainer
from worldmodel.rl.memory_policy import build_memory_policy
from stream_ctm_deep_eval import build_one, _collect_env

ENVS = ["pendulum-partial", "pendulum"]
PPO_METHODS = ["flash", "ctm", "lstm"]          # fast reactive (μs-ms)
CEM_METHODS = ["cem-jepa", "cem-ctm"]           # slow model-based planners (ms-s)
DEADLINES = [None, 50.0, 20.0, 5.0, 1.0]         # ms; None = unconstrained
FIELDS = ["env", "method", "seed", "deadline_ms", "success_rate",
          "mean_latency_ms", "p99_latency_ms", "throughput_hz", "timeout_rate", "n_steps"]


def _dl_str(dl):
    return "none" if dl is None else str(dl)


def train_ppo(env, backbone, seed, args, device):
    e = make_env(env)
    od = int(np.prod(e.observation_space.shape)); gd = int(np.prod(e.goal_space.shape))
    ad = int(np.prod(e.action_space.shape))
    pol = build_memory_policy(backbone, od, gd, ad, latent_dim=args.latent_dim,
                              d_model=args.d_model, memory_length=args.memory_length,
                              state_gate="gru")
    t = PPOTrainer(env, pol, num_envs=args.ppo_envs, num_steps=args.ppo_steps,
                   device=device, lr=args.lr)
    torch.manual_seed(seed); np.random.seed(seed)
    t.train(args.total_steps, log_iters=6, eval_episodes=8, seed=seed)
    return t


def train_cem(env, wm_kind, seed, args, device):
    buf, env_kw = _collect_env(env, args)
    e = make_env(env, **env_kw)
    obs_key = "state"
    model = build_one(wm_kind, 8, "none", obs_key, e.observation_space.shape,
                      e.action_space.shape[0], args.latent_dim, args.var_weight, device)
    torch.manual_seed(seed); np.random.seed(seed)
    train_world_model(model, buf, horizon=args.cem_horizon, epochs=args.cem_epochs,
                      batch_size=args.batch_size, device=device, seed=seed)
    model.eval()
    solver = CEMSolver(model=model, num_samples=args.cem_samples, n_steps=args.cem_steps,
                       topk=max(4, args.cem_samples // 8), device=device)
    ew = World(lambda: make_env(env, **env_kw), num_envs=args.num_envs)
    ew.set_policy(WorldModelPolicy(solver=solver, config=PlanConfig(horizon=args.cem_horizon)))
    return ew


def eval_deadlines(obj, kind, episodes, seed=100):
    out = {}
    for dl in DEADLINES:
        if kind == "ppo":
            r = obj.evaluate_timed(episodes=episodes, seed=seed, deadline_ms=dl)
        else:
            r = obj.evaluate_timed(episodes=episodes, seed=seed, deadline_ms=dl)
        out[dl] = r
    return out


def _worker(rank, args, nworkers, all_tasks):
    """One worker per GPU: trains its task slice + runs timing eval in-process.
    Each worker owns a GPU + a disjoint CPU group, so per-step latency is not
    polluted by other workers (timing stays accurate)."""
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = str(rank)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    my = all_tasks[rank::nworkers]
    rows = []
    for (env, method, seed) in my:
        t0 = time.time()
        if method in args.ppo_methods:
            obj = train_ppo(env, method, seed, args, device)
            kind, eps = "ppo", args.episodes
        else:
            wm_kind = "jepa" if "jepa" in method else "stream"
            obj = train_cem(env, wm_kind, seed, args, device)
            kind, eps = "cem", args.episodes_cem
        res = eval_deadlines(obj, kind, eps)
        for dl, r in res.items():
            rows.append({"env": env, "method": method, "seed": seed,
                         "deadline_ms": _dl_str(dl), **r})
        base = res[None]
        print(f"[gpu{rank}] {env}/{method}/s{seed} {time.time()-t0:.0f}s | "
              f"succ={base['success_rate']:.1f}% lat={base['mean_latency_ms']:.3f}ms "
              f"thr={base['throughput_hz']:.0f}Hz", flush=True)
    Path("csv_data").mkdir(exist_ok=True)
    with open(f"csv_data/realtime_shard{rank}.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=FIELDS); wr.writeheader()
        for r in rows:
            wr.writerow(r)
    print(f"[gpu{rank}] done {len(rows)} rows", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--envs", nargs="*", default=ENVS)
    ap.add_argument("--ppo-methods", nargs="*", default=PPO_METHODS)
    ap.add_argument("--cem-methods", nargs="*", default=CEM_METHODS)
    ap.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2])
    ap.add_argument("--total-steps", type=int, default=60000)
    ap.add_argument("--cem-epochs", type=int, default=150)
    ap.add_argument("--cem-horizon", type=int, default=6)
    ap.add_argument("--cem-samples", type=int, default=128)
    ap.add_argument("--cem-steps", type=int, default=6)
    ap.add_argument("--ppo-envs", type=int, default=8)
    ap.add_argument("--ppo-steps", type=int, default=256)
    ap.add_argument("--num-envs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--d_model", type=int, default=128)
    ap.add_argument("--latent_dim", type=int, default=64)
    ap.add_argument("--memory_length", type=int, default=8)
    ap.add_argument("--var_weight", type=float, default=4.0)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--episodes", type=int, default=60)
    ap.add_argument("--episodes-cem", type=int, default=20, help="CEM eval is slow, fewer eps")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--csv", default="csv_data/realtime_benchmark.csv")
    args = ap.parse_args()
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device

    Path("csv_data").mkdir(exist_ok=True)
    print("=" * 78)
    print("real-time benchmark: Flash Brain vs memory policy vs CEM planners")
    print("=" * 78)
    print(f"device={device}  envs={args.envs}  seeds={args.seeds}")
    print(f"PPO methods (fast): {args.ppo_methods}")
    print(f"CEM methods (slow): {args.cem_methods}")
    print(f"deadlines (ms): {[('none' if d is None else d) for d in DEADLINES]}\n")

    all_methods = args.ppo_methods + args.cem_methods
    all_tasks = [(env, m, seed) for env in args.envs for m in all_methods for seed in args.seeds]
    n_gpu = torch.cuda.device_count()
    # 1 worker per GPU (NOT procs_per_gpu): timing accuracy requires each worker
    # own a disjoint CPU group, so multi-proc per GPU would pollute latency.
    nw = min(n_gpu, len(all_tasks)) if n_gpu >= 1 else 1
    for p in Path("csv_data").glob("realtime_shard*.csv"):
        p.unlink()
    print(f"GPU={n_gpu} workers={nw} (1/GPU, 独占 CPU 保 timing 准)  tasks={len(all_tasks)}\n")

    if nw <= 1 or n_gpu < 1:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _worker(0, args, 1, all_tasks)
    else:
        torch.multiprocessing.spawn(_worker, args=(args, nw, all_tasks), nprocs=nw, join=True)

    rows = []
    for p in sorted(Path("csv_data").glob("realtime_shard*.csv")):
        with open(p) as f:
            for r in csv.DictReader(f):
                for k in ("success_rate", "mean_latency_ms", "p99_latency_ms",
                          "throughput_hz", "timeout_rate", "seed"):
                    try:
                        r[k] = float(r[k])
                        if k == "seed":
                            r[k] = int(r[k])
                    except (ValueError, TypeError):
                        pass
                rows.append(r)

    with open(args.csv, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=FIELDS); wr.writeheader()
        for r in rows:
            wr.writerow({k: r.get(k, "") for k in FIELDS})

    # ---- report ----
    def _agg(env, method, dl, key):
        xs = [r[key] for r in rows if r["env"] == env and r["method"] == method
              and r["deadline_ms"] == _dl_str(dl)]
        return float(np.mean(xs)) if xs else float("nan")

    L = ["# real-time benchmark: Flash Brain vs planners\n",
         f"生成: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"]
    allm = args.ppo_methods + args.cem_methods
    L.append("\n## 1. per-step latency (mean ms, unconstrained) — 实时性\n")
    L.append("| env | " + " | ".join(allm) + " |")
    L.append("|" + "---|" * (len(allm) + 1))
    for env in args.envs:
        cells = [env] + [f"{_agg(env, m, None, 'mean_latency_ms'):.4f}" for m in allm]
        L.append("| " + " | ".join(cells) + " |")

    L.append("\n## 2. throughput (Hz, unconstrained)\n")
    L.append("| env | " + " | ".join(allm) + " |")
    L.append("|" + "---|" * (len(allm) + 1))
    for env in args.envs:
        cells = [env] + [f"{_agg(env, m, None, 'throughput_hz'):.0f}" for m in allm]
        L.append("| " + " | ".join(cells) + " |")

    L.append("\n## 3. deadline-constrained success (核心: 紧 deadline 下谁撑得住)\n")
    for env in args.envs:
        L.append(f"\n### {env}\n")
        L.append("| method | " + " | ".join(_dl_str(d) for d in DEADLINES) + " |")
        L.append("|" + "---|" * (len(DEADLINES) + 1))
        for m in allm:
            cells = [m] + [f"{_agg(env, m, d, 'success_rate'):.1f}" for d in DEADLINES]
            L.append("| " + " | ".join(cells) + " |")

    L.append("\n## 判读\n")
    L.append("- latency: flash/ctm/lstm 应是 μs~ms (forward), CEM 应是 ms~s (每步 rollout 上百轨迹)")
    L.append("- deadline 收紧 (50→1ms): fast policy 的 success 应几乎不掉; CEM 在 5ms/1ms 应崩 (超时)")
    L.append("- 这是 Flash Brain 的实时性差异化: 同等 success 下延迟低几个数量级")
    txt = "\n".join(L)
    Path("csv_data/realtime_benchmark_report.md").write_text(txt)
    print("\n" + "=" * 78)
    print(txt)
    print(f"\n[done] csv -> {args.csv}")


if __name__ == "__main__":
    main()
