#!/usr/bin/env python3
"""一次性定位 stream-ctm 在 partial/tworoom 失败的根因 (多卡并行版)。

stream-ctm 的"持久状态"有两处可能限制记忆 (worldmodel/wm/streaming.py):
  H1 窗口太短: trace 是 memory_length(=8) 的 FIFO, 8 步前信息被 shift 丢弃
  H2 无状态积累: activated 每步被 NLM 输出【完全覆盖】(无 residual/gating),
                 上一步思考不保留 -> 不是真"持续思考"

本脚本在每个失败场景 + 正对照上扫 (memory_length x state_gate) 网格:
  stream-base        mem=8,  no gate   (现状)
  stream-mem16/32    mem=16/32 no gate  (测 H1)
  stream-gate        mem=8,  gru gate  (测 H2)
  stream-gate-mem32  mem=32, gru gate  (H1+H2 组合)
  jepa-mlp           Markov 对照

判定 (对每个失败 env, vs stream-base):
  mem 加长增益大   -> H1: 窗口太短
  gating 增益大    -> H2: 无状态积累 (实现没到位, 加 GRU 门控残差修)
  组合 >> 单独     -> H1+H2 协同
  全都没用(<5pp)  -> 根因在别处 (dynamics/encoder/CEM, 查 dyn_err/latent_var)

多卡: 自动探测 GPU 数, 任务网格 (env x variant) 均分到各卡, mp.spawn 并行,
      主进程合并 + 判定. 单卡/无 GPU 时自动退化为串行.

算力机前台跑 (4 卡 ~8min, 不重定向, 直接看进度 + 判定):
    python paper/diagnose_stream_ctm_failures.py
快速 (~3min):
    python paper/diagnose_stream_ctm_failures.py --quick
指定卡数:
    python paper/diagnose_stream_ctm_failures.py --nworkers 4
"""
import argparse, csv, os, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

import worldmodel as wm
from worldmodel.envs import make_env
from worldmodel.wm import WorldModel, CNNEncoder, MLPEncoder, StreamingCTMPredictor, build_jepa_wm
from worldmodel.train import train_world_model

DEFAULT_ENVS = ["pendulum", "pendulum-partial", "cartpole-partial", "tworoom-state"]
FAIL_ENVS = {"pendulum-partial", "cartpole-partial", "tworoom-state"}

# (name, memory_length, state_gate, kind)
VARIANTS = [
    ("stream-base",       8,  "none", "stream"),
    ("stream-mem16",      16, "none", "stream"),
    ("stream-mem32",      32, "none", "stream"),
    ("stream-gate",       8,  "gru",  "stream"),
    ("stream-gate-mem32", 32, "gru",  "stream"),
    ("jepa-mlp",          8,  "none", "jepa"),
]
FIELDS = ["env", "variant", "memory_length", "state_gate", "success_rate",
          "random_rate", "dynamics_err", "latent_var", "elapsed_s"]


def build_one(kind, mem, gate, obs_key, obs_shape, action_dim, latent_dim, var_weight, device):
    if kind == "jepa":
        return build_jepa_wm(obs_key, obs_shape, action_dim,
                             latent_dim=latent_dim, var_weight=var_weight).to(device)
    encoder = (CNNEncoder(latent_dim=latent_dim, channels=int(obs_shape[0]))
               if obs_key == "pixels"
               else MLPEncoder(obs_dim=int(obs_shape[0]), latent_dim=latent_dim))
    predictor = StreamingCTMPredictor(
        latent_dim=latent_dim, action_dim=action_dim,
        d_model=max(64, latent_dim * 2), memory_length=mem, nlm_hidden=8,
        state_gate=gate,
    )
    return WorldModel(encoder=encoder, predictor=predictor, obs_key=obs_key,
                      action_dim=action_dim, cost_mode="last",
                      var_weight=var_weight).to(device)


def evaluate(model, env_name, env_kw, num_envs, cem_samples, cem_steps, horizon, eval_episodes, device):
    solver = wm.solver.CEMSolver(model=model, num_samples=cem_samples, n_steps=cem_steps,
                                 topk=max(4, cem_samples // 8), device=device)
    ew = wm.World(lambda: make_env(env_name, **env_kw), num_envs=num_envs)
    ew.set_policy(wm.WorldModelPolicy(solver=solver, config=wm.PlanConfig(horizon=horizon)))
    res = ew.evaluate(episodes=eval_episodes, seed=100)
    rw = wm.World(lambda: make_env(env_name, **env_kw), num_envs=num_envs)
    rw.set_policy(wm.RandomPolicy())
    rres = rw.evaluate(episodes=eval_episodes, seed=100)
    return res["success_rate"], rres["success_rate"]


def run_one(env_name, vname, mem, gate, kind, buf, env_kw, args, device, seed):
    """跑单个 (env, variant): build -> train -> eval. 返回 row dict."""
    obs_key = "pixels" if env_name == "point-image" else "state"
    env = make_env(env_name, **env_kw)
    obs_shape, action_dim = env.observation_space.shape, env.action_space.shape[0]
    t0 = time.time()
    torch.manual_seed(seed); np.random.seed(seed)
    model = build_one(kind, mem, gate, obs_key, obs_shape, action_dim,
                      args.latent_dim, args.var_weight, device)
    hist = train_world_model(model, buf, horizon=args.horizon, epochs=args.epochs,
                             batch_size=args.batch_size, device=device,
                             log_every=10**9, seed=seed)
    last = hist[-1] if hist else {}
    model.eval()
    succ, rand = evaluate(model, env_name, env_kw, args.num_envs,
                          args.cem_samples, args.cem_steps, args.horizon,
                          args.eval_episodes, device)
    return dict(env=env_name, variant=vname, memory_length=mem, state_gate=gate,
                success_rate=round(succ, 1), random_rate=round(rand, 1),
                dynamics_err=round(float(last.get("dynamics_err", float("nan"))), 5),
                latent_var=round(float(last.get("latent_var", float("nan"))), 5),
                elapsed_s=round(time.time() - t0, 1))


def _collect_env(env_name, args):
    env_kw = {"image_size": 32} if env_name == "point-image" else {}
    buf = wm.data.ReplayBuffer()
    cw = wm.World(lambda: make_env(env_name, **env_kw), num_envs=args.num_envs)
    cw.set_policy(wm.RandomPolicy())
    cw.collect(buf, episodes=args.episodes, seed=0)
    return buf, env_kw


def _worker(rank, args, nworkers, all_tasks):
    """mp.spawn 入口: 占 GPU `rank`, 跑分配的任务, 写 shard CSV."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(rank)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    my_tasks = all_tasks[rank::nworkers]
    envs_in = sorted({t[0] for t in my_tasks})
    buffers = {}
    for env_name in envs_in:
        buffers[env_name] = _collect_env(env_name, args)
        print(f"[gpu{rank}] collect {env_name}: "
              f"{len(buffers[env_name][0].episodes)} eps", flush=True)
    Path("csv_data").mkdir(exist_ok=True)
    out_path = f"csv_data/stream_ctm_diag_shard{rank}.csv"
    rows = []
    with open(out_path, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=FIELDS); wr.writeheader()
        for env_name, vname, mem, gate, kind in my_tasks:
            try:
                row = run_one(env_name, vname, mem, gate, kind,
                              *buffers[env_name], args, device, args.seed)
            except Exception as e:
                row = dict(env=env_name, variant=vname, memory_length=mem, state_gate=gate,
                           success_rate=-1.0, random_rate=-1.0, dynamics_err=-1,
                           latent_var=-1, elapsed_s=0.0)
                print(f"[gpu{rank}] ERROR {env_name}/{vname}: {e}", flush=True)
            wr.writerow(row); f.flush(); rows.append(row)
            print(f"[gpu{rank}] {env_name:<18} {vname:<18} succ={row['success_rate']:5.1f}%  "
                  f"dyn_err={row['dynamics_err']} var={row['latent_var']}  [{row['elapsed_s']:.0f}s]",
                  flush=True)
    print(f"[gpu{rank}] done {len(rows)} tasks -> {out_path}", flush=True)


def judge(rows, envs):
    """打印结果表 + 根因判定 + 正对照。rows: list of row dict。"""
    res = {(r["env"], r["variant"]): r for r in rows}
    print("\n" + "=" * 78)
    print("结果表 (success_rate %)")
    print("=" * 78)
    print(f"{'variant':<20}" + "".join(f"{e:<20}" for e in envs))
    for vname, *_ in VARIANTS:
        cells = []
        for env_name in envs:
            r = res.get((env_name, vname))
            cells.append(f"{r['success_rate']:>5.1f}" if r else "  -  ")
        print(f"{vname:<20}" + "".join(f"{c:<20}" for c in cells))

    print("\n" + "=" * 78)
    print("根因判定")
    print("=" * 78)
    for env_name in envs:
        if env_name not in FAIL_ENVS:
            continue
        base = res.get((env_name, "stream-base"), {}).get("success_rate", 0)
        mem_gain = max(res.get((env_name, "stream-mem16"), {}).get("success_rate", 0),
                       res.get((env_name, "stream-mem32"), {}).get("success_rate", 0)) - base
        gate_gain = res.get((env_name, "stream-gate"), {}).get("success_rate", 0) - base
        combo_gain = res.get((env_name, "stream-gate-mem32"), {}).get("success_rate", 0) - base
        jepa = res.get((env_name, "jepa-mlp"), {}).get("success_rate", 0)
        print(f"\n  [{env_name}]  stream-base={base:.1f}%  jepa-mlp(Markov)={jepa:.1f}%")
        print(f"    mem 加长增益:     {mem_gain:+5.1f}   (H1 窗口太短)")
        print(f"    gating 增益:      {gate_gain:+5.1f}   (H2 无状态积累)")
        print(f"    组合(mem32+gate): {combo_gain:+5.1f}   (H1+H2)")
        best = max(mem_gain, gate_gain, combo_gain)
        if best < 5:
            v = ("根因在别处 (非记忆机制): 加 memory/gate 都没用, "
                 "问题在 dynamics 预测/encoder/CEM — 查 dyn_err 和 latent_var")
        elif combo_gain > mem_gain + 3 and combo_gain > gate_gain + 3:
            v = "H1+H2 协同: 窗口和积累都要修 (组合增益远超单独)"
        elif gate_gain >= mem_gain and gate_gain >= 5:
            v = "H2 确认: 无状态积累是主因 -> 给 activated 加 GRU 门控残差 (state_gate='gru')"
        elif mem_gain >= gate_gain and mem_gain >= 5:
            v = "H1 确认: 窗口太短是主因 -> 增大 memory_length"
        else:
            v = "部分改善但均 <5pp, 趋势弱, 补 seed 或加 epochs 再确认"
        print(f"    >>> {v}")

    if "pendulum" in envs:
        print(f"\n  [正对照 pendulum] (stream-ctm 的 win 场景, 改动不应破坏)")
        base_p = res.get(("pendulum", "stream-base"), {}).get("success_rate", 0)
        for vname, *_ in VARIANTS:
            s = res.get(("pendulum", vname), {}).get("success_rate", float("nan"))
            flag = "  <-- 退化>15pp!" if (base_p - s > 15) else ""
            print(f"    {vname:<20} {s:5.1f}%{flag}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--envs", nargs="*", default=DEFAULT_ENVS)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--episodes", type=int, default=60)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--horizon", type=int, default=6)
    ap.add_argument("--latent_dim", type=int, default=32)
    ap.add_argument("--var_weight", type=float, default=4.0)
    ap.add_argument("--cem_samples", type=int, default=128)
    ap.add_argument("--cem_steps", type=int, default=6)
    ap.add_argument("--eval_episodes", type=int, default=12)
    ap.add_argument("--num_envs", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--nworkers", type=int, default=0,
                    help="并行 worker 数 (=GPU 数). 0=自动探测. 1=串行.")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--csv", default="csv_data/stream_ctm_diagnosis.csv")
    args = ap.parse_args()

    if args.quick:
        args.episodes = 20; args.epochs = 15; args.eval_episodes = 8
        args.cem_samples = 64; args.cem_steps = 4; args.horizon = 4

    all_tasks = [(env, v[0], v[1], v[2], v[3]) for env in args.envs for v in VARIANTS]
    n_tasks = len(all_tasks)

    # 探测 GPU 数, 决定 worker 数
    n_gpu = torch.cuda.device_count()
    if args.nworkers > 0:
        nw = args.nworkers
    elif n_gpu >= 1:
        nw = min(n_gpu, n_tasks)
    else:
        nw = 1
    # 清理旧 shard csv
    for p in Path("csv_data").glob("stream_ctm_diag_shard*.csv"):
        p.unlink()

    print("=" * 78)
    print("stream-ctm 失败根因诊断")
    print("=" * 78)
    print(f"GPU 可见数={n_gpu}  workers={nw}  tasks={n_tasks} "
          f"({len(args.envs)} envs x {len(VARIANTS)} variants)")
    print(f"epochs={args.epochs}  cem_samples={args.cem_samples}  seed={args.seed}")
    print(f"envs={args.envs}")
    print("假设: H1=窗口太短(mem加长改善)  H2=无状态积累(gating改善)\n")

    t_start = time.time()
    if nw <= 1 or n_gpu < 1:
        # 串行 (单卡或无 GPU)
        device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
        buffers = {}
        for env_name in args.envs:
            buffers[env_name] = _collect_env(env_name, args)
            print(f"[serial] collect {env_name}: {len(buffers[env_name][0].episodes)} eps")
        rows = []
        for env_name, vname, mem, gate, kind in all_tasks:
            try:
                row = run_one(env_name, vname, mem, gate, kind,
                              *buffers[env_name], args, device, args.seed)
            except Exception as e:
                row = dict(env=env_name, variant=vname, memory_length=mem, state_gate=gate,
                           success_rate=-1.0, random_rate=-1.0, dynamics_err=-1,
                           latent_var=-1, elapsed_s=0.0)
                print(f"[serial] ERROR {env_name}/{vname}: {e}")
            rows.append(row)
            print(f"[serial] {env_name:<18} {vname:<18} succ={row['success_rate']:5.1f}%  "
                  f"[{row['elapsed_s']:.0f}s]")
    else:
        # 多卡并行: mp.spawn, 每 worker 占一卡
        torch.multiprocessing.spawn(
            _worker, args=(args, nw, all_tasks), nprocs=nw, join=True,
        )
        # merge shard csv
        rows = []
        for rank in range(nw):
            p = f"csv_data/stream_ctm_diag_shard{rank}.csv"
            if os.path.exists(p):
                with open(p) as f:
                    rows.extend(csv.DictReader(f))

    print(f"\n[total] {len(rows)}/{n_tasks} runs 完成, 耗时 {time.time()-t_start:.0f}s")
    # 写合并 csv
    Path("csv_data").mkdir(exist_ok=True)
    with open(args.csv, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=FIELDS); wr.writeheader()
        for r in rows:
            wr.writerow({k: r.get(k, "") for k in FIELDS})
    if len(rows) < n_tasks:
        print(f"[warn] 只完成 {len(rows)}/{n_tasks}, 部分任务失败 (见上面 ERROR 行)")
    judge(rows, args.envs)
    print(f"\n[done] 详细数据 -> {args.csv}")


if __name__ == "__main__":
    main()
