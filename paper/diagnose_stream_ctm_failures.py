#!/usr/bin/env python3
"""一次性定位 stream-ctm 在 partial/tworoom 失败的根因。

stream-ctm 的"持久状态"有两处可能限制记忆 (worldmodel/wm/streaming.py):
  H1 窗口太短: trace 是 memory_length(=8) 的 FIFO, 8 步前的信息被 shift 丢弃
  H2 无状态积累: activated 每步被 NLM 输出【完全覆盖】(无 residual/gating),
                 上一步思考不保留 -> 不是真"持续思考"

本脚本在每个失败场景 + 正对照上扫 (memory_length x state_gate) 网格:
  stream-base        mem=8,  no gate   (现状)
  stream-mem16       mem=16, no gate   (测 H1)
  stream-mem32       mem=32, no gate   (测 H1)
  stream-gate        mem=8,  gru gate  (测 H2)
  stream-gate-mem32  mem=32, gru gate  (H1+H2 组合)
  jepa-mlp           Markov 对照 (无记忆)

判定规则 (对每个失败 env, 比较 vs stream-base):
  mem 加长增益大        -> H1: 窗口太短, 加长 memory_length 即修
  gating 增益大         -> H2: 无状态积累, 给 activated 加 GRU 门控残差
  组合 >> 单独          -> H1+H2 协同, 两者都要修
  全都没用 (gain<5)     -> 根因在别处 (dynamics 预测/encoder/CEM, 非记忆机制)

算力机前台跑 (~30min, 不重定向, 直接看进度):
    python paper/diagnose_stream_ctm_failures.py
快速验证 (~8min):
    python paper/diagnose_stream_ctm_failures.py --quick
单 env 调试:
    python paper/diagnose_stream_ctm_failures.py --envs pendulum-partial
"""
import argparse, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

import worldmodel as wm
from worldmodel.envs import make_env
from worldmodel.wm import WorldModel, CNNEncoder, MLPEncoder, StreamingCTMPredictor, build_jepa_wm
from worldmodel.train import train_world_model

# 失败场景 + 正对照 (pendulum 是 stream-ctm 的 win 场景, 改动不该破坏它)
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
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--csv", default="csv_data/stream_ctm_diagnosis.csv")
    args = ap.parse_args()

    if args.quick:
        args.episodes = 20; args.epochs = 15; args.eval_episodes = 8
        args.cem_samples = 64; args.cem_steps = 4; args.horizon = 4

    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    print("=" * 78)
    print("stream-ctm 失败根因诊断")
    print("=" * 78)
    print(f"device={device}  epochs={args.epochs}  cem_samples={args.cem_samples}  seed={args.seed}")
    print(f"envs={args.envs}")
    print(f"variants={[v[0] for v in VARIANTS]}")
    print("假设: H1=窗口太短(mem加长改善)  H2=无状态积累(gating改善)\n")

    Path("csv_data").mkdir(exist_ok=True)
    import csv
    write_header = not Path(args.csv).exists()
    csvfile = open(args.csv, "a", newline="")
    wr = csv.DictWriter(csvfile, fieldnames=FIELDS)
    if write_header:
        wr.writeheader()

    # 1) 每个 env collect 一次数据 (复用)
    buffers = {}
    for env_name in args.envs:
        env_kw = {"image_size": 32} if env_name == "point-image" else {}
        buf = wm.data.ReplayBuffer()
        cw = wm.World(lambda: make_env(env_name, **env_kw), num_envs=args.num_envs)
        cw.set_policy(wm.RandomPolicy())
        cw.collect(buf, episodes=args.episodes, seed=0)
        buffers[env_name] = (buf, env_kw)
        print(f"[collect] {env_name}: {len(buf.episodes)} eps / {buf.total_steps} steps")

    # 2) (env x variant) 网格
    results = {}
    for env_name in args.envs:
        buf, env_kw = buffers[env_name]
        obs_key = "pixels" if env_name == "point-image" else "state"
        env = make_env(env_name, **env_kw)
        obs_shape, action_dim = env.observation_space.shape, env.action_space.shape[0]
        print(f"\n--- {env_name} (obs={obs_key}, act_dim={action_dim}) ---")
        for vname, mem, gate, kind in VARIANTS:
            t0 = time.time()
            torch.manual_seed(args.seed); np.random.seed(args.seed)
            model = build_one(kind, mem, gate, obs_key, obs_shape, action_dim,
                              args.latent_dim, args.var_weight, device)
            hist = train_world_model(model, buf, horizon=args.horizon, epochs=args.epochs,
                                     batch_size=args.batch_size, device=device,
                                     log_every=10**9, seed=args.seed)
            last = hist[-1] if hist else {}
            model.eval()
            succ, rand = evaluate(model, env_name, env_kw, args.num_envs,
                                  args.cem_samples, args.cem_steps, args.horizon,
                                  args.eval_episodes, device)
            row = dict(env=env_name, variant=vname, memory_length=mem, state_gate=gate,
                       success_rate=round(succ, 1), random_rate=round(rand, 1),
                       dynamics_err=round(float(last.get("dynamics_err", float("nan"))), 5),
                       latent_var=round(float(last.get("latent_var", float("nan"))), 5),
                       elapsed_s=round(time.time() - t0, 1))
            wr.writerow(row); csvfile.flush()
            results[(env_name, vname)] = row
            print(f"  {vname:<18} succ={succ:5.1f}% (rand {rand:4.1f})  "
                  f"dyn_err={row['dynamics_err']:.4f}  var={row['latent_var']:.5f}  [{row['elapsed_s']:.0f}s]")
    csvfile.close()

    # 3) 结果总表
    print("\n" + "=" * 78)
    print("结果表 (success_rate %)")
    print("=" * 78)
    print(f"{'variant':<20}" + "".join(f"{e:<20}" for e in args.envs))
    for vname, *_ in VARIANTS:
        cells = []
        for env_name in args.envs:
            r = results.get((env_name, vname))
            cells.append(f"{r['success_rate']:>5.1f}" if r else "  -  ")
        print(f"{vname:<20}" + "".join(f"{c:<20}" for c in cells))

    # 4) 根因判定 (只对失败场景)
    print("\n" + "=" * 78)
    print("根因判定")
    print("=" * 78)
    for env_name in args.envs:
        if env_name not in FAIL_ENVS:
            continue
        base = results.get((env_name, "stream-base"), {}).get("success_rate", 0)
        mem_gain = max(results.get((env_name, "stream-mem16"), {}).get("success_rate", 0),
                       results.get((env_name, "stream-mem32"), {}).get("success_rate", 0)) - base
        gate_gain = results.get((env_name, "stream-gate"), {}).get("success_rate", 0) - base
        combo_gain = results.get((env_name, "stream-gate-mem32"), {}).get("success_rate", 0) - base
        jepa = results.get((env_name, "jepa-mlp"), {}).get("success_rate", 0)
        print(f"\n  [{env_name}]  stream-base={base:.1f}%  jepa-mlp(Markov)={jepa:.1f}%")
        print(f"    mem 加长增益:     {mem_gain:+5.1f}   (H1 窗口太短)")
        print(f"    gating 增益:      {gate_gain:+5.1f}   (H2 无状态积累)")
        print(f"    组合(mem32+gate): {combo_gain:+5.1f}   (H1+H2)")
        best = max(mem_gain, gate_gain, combo_gain)
        if best < 5:
            v = ("根因在别处 (非记忆机制): 加 memory/gate 都没用, "
                 "问题在 dynamics 预测/encoder/CEM 本身 — 查 dyn_err 和 latent_var")
        elif combo_gain > mem_gain + 3 and combo_gain > gate_gain + 3:
            v = "H1+H2 协同: 窗口和积累都要修 (组合增益远超单独)"
        elif gate_gain >= mem_gain and gate_gain >= 5:
            v = "H2 确认: 无状态积累是主因 -> 给 activated 加 GRU 门控残差 (state_gate='gru')"
        elif mem_gain >= gate_gain and mem_gain >= 5:
            v = "H1 确认: 窗口太短是主因 -> 增大 memory_length"
        else:
            v = "部分改善但均 <5pp, 趋势弱, 补 seed 或加 epochs 再确认"
        print(f"    >>> {v}")

    # 5) 正对照: pendulum 改动不该破坏连续控制优势
    if "pendulum" in args.envs:
        print(f"\n  [正对照 pendulum] (stream-ctm 的 win 场景, 改动不应破坏)")
        base_p = results.get(("pendulum", "stream-base"), {}).get("success_rate", 0)
        for vname, *_ in VARIANTS:
            s = results.get(("pendulum", vname), {}).get("success_rate", float("nan"))
            flag = "  <-- 退化!" if (base_p - s > 15) else ""
            print(f"    {vname:<20} {s:5.1f}%{flag}")

    print(f"\n[done] 详细数据 -> {args.csv}")


if __name__ == "__main__":
    main()
