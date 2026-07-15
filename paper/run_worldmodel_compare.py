#!/usr/bin/env python
"""Unified world-model comparison: stream-ctm vs the other models on ALL envs.

Sweep env x model x seed under identical collect / data / CEM / eval, writing
one row per run to ``csv_data/worldmodel_compare_results.csv`` and a printed
mean+-std summary at the end.

Models (the three world-model variants in this framework):
  - ``jepa-mlp``     : (CNN|MLP) encoder + Markov MLPPredictor
  - ``stream-ctm``   : same encoder + StreamingCTMPredictor (persistent NLM state)
  - ``ctm-encoder``  : CTM-as-encoder (image only) + MLPPredictor

Matrix:
  - point-state : jepa-mlp, stream-ctm            (MLP encoder, collapse-free)
  - point-image : jepa-mlp, stream-ctm, ctm-encoder

On point-image the from-scratch CNN encoder can JEPA-collapse (latent_var -> 0,
affects jepa-mlp AND stream-ctm equally — it is an encoder, not a predictor,
issue). Multiple seeds make the bimodal collapse visible; read ``latent_var``.

Compute machine:
    nohup python paper/run_worldmodel_compare.py --seeds 0 1 2 > logs/wm_compare.log 2>&1 &
Quick check:
    python paper/run_worldmodel_compare.py --quick --envs point-state
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

import worldmodel as wm
from worldmodel.envs import make_env
from worldmodel.wm import (
    WorldModel, CNNEncoder, MLPEncoder, MLPPredictor, StreamingCTMPredictor,
    build_jepa_wm, build_ctm_wm,
)
from worldmodel.train import train_world_model

IMAGE_ENVS = {"point-image", "tworoom"}  # ctm-encoder (image-only) applies here


def models_for_env(env_name: str) -> list[str]:
    """Models applicable to an env: image envs get the CTM encoder too."""
    base = env_name.lower().replace("_", "-")
    is_image = base in IMAGE_ENVS or (base.startswith("tworoom") and not base.endswith("state"))
    return ["jepa-mlp", "stream-ctm", "ctm-encoder"] if is_image else ["jepa-mlp", "stream-ctm"]


DEFAULT_ENVS = [
    "cartpole", "cartpole-partial", "pendulum", "pendulum-partial", "reacher",
    "tworoom-state", "tworoom", "point-state", "point-image",
]
FIELDS = [
    "env", "model", "seed", "horizon", "success_rate", "random_rate",
    "final_loss", "dynamics_err", "latent_var", "elapsed_s",
]


def build_model(model_name, obs_key, obs_shape, action_dim, latent_dim, var_weight, device, goal_shape=None):
    # goal_encoder only for state envs: image envs render the goal as an image
    # with the SAME shape as obs, but their goal_space metadata reports the raw
    # 2D target coordinate — trusting it would build a mis-sized goal_encoder
    # that chokes on the image goal. State envs (e.g. reacher obs=4, goal=2)
    # have reliable goal_space metadata, so only they get a separate encoder.
    eff_goal_shape = goal_shape if obs_key == "state" else None
    if model_name == "jepa-mlp":
        m = build_jepa_wm(obs_key, obs_shape, action_dim, latent_dim=latent_dim,
                          var_weight=var_weight, goal_shape=eff_goal_shape)
    elif model_name == "stream-ctm":
        encoder = (CNNEncoder(latent_dim=latent_dim, channels=int(obs_shape[0]))
                   if obs_key == "pixels" else MLPEncoder(obs_dim=int(obs_shape[0]), latent_dim=latent_dim))
        predictor = StreamingCTMPredictor(
            latent_dim=latent_dim, action_dim=action_dim,
            d_model=max(64, latent_dim * 2), memory_length=8, nlm_hidden=8,
        )
        # Separate goal encoder when goal dim != obs dim (e.g. reacher: obs=4, goal=2).
        goal_encoder = None
        if eff_goal_shape is not None and int(np.prod(eff_goal_shape)) != int(np.prod(obs_shape)):
            goal_encoder = MLPEncoder(obs_dim=int(np.prod(eff_goal_shape)), latent_dim=latent_dim)
        m = WorldModel(encoder=encoder, predictor=predictor, obs_key=obs_key,
                       action_dim=action_dim, cost_mode="last", var_weight=var_weight,
                       goal_encoder=goal_encoder)
    elif model_name == "ctm-encoder":
        if obs_key != "pixels":
            raise ValueError("ctm-encoder is image-only")
        m = build_ctm_wm(action_dim=action_dim, iterations=5, d_model=128,
                         n_synch_out=latent_dim, backbone_type="resnet18-1",
                         var_weight=var_weight)
    else:
        raise KeyError(model_name)
    return m.to(device)


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
    ap = argparse.ArgumentParser()
    ap.add_argument("--envs", nargs="*", default=None)
    ap.add_argument("--models", nargs="*", default=None,
                    help="subset of models; default = all applicable per env")
    ap.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2])
    ap.add_argument("--image_size", type=int, default=32)
    ap.add_argument("--episodes", type=int, default=60)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--horizon", type=int, default=6)
    ap.add_argument("--latent_dim", type=int, default=32)
    ap.add_argument("--var_weight", type=float, default=4.0)
    ap.add_argument("--cem_samples", type=int, default=128)
    ap.add_argument("--cem_steps", type=int, default=6)
    ap.add_argument("--eval_episodes", type=int, default=16)
    ap.add_argument("--num_envs", type=int, default=4)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--shard", type=int, default=0, help="this worker's shard id (0..nshards-1)")
    ap.add_argument("--nshards", type=int, default=1, help="total parallel workers")
    ap.add_argument("--csv_suffix", default="", help="append to csv filename (per-shard isolation)")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    if args.quick:
        args.episodes = 16; args.epochs = 10; args.eval_episodes = 8
        args.cem_samples = 48; args.cem_steps = 4; args.horizon = 4
        args.latent_dim = 16; args.seeds = [0]

    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    Path("csv_data").mkdir(exist_ok=True)
    csv_path = f"csv_data/worldmodel_compare_results{args.csv_suffix}.csv"
    write_header = not os.path.exists(csv_path)

    # Build the full task grid, then take this worker's shard.
    envs = args.envs if args.envs else list(DEFAULT_ENVS)
    tasks: list[tuple[str, str, int]] = []
    for env_name in envs:
        models = args.models if args.models else models_for_env(env_name)
        for m in models:
            for seed in args.seeds:
                tasks.append((env_name, m, seed))
    shard_tasks = tasks[args.shard :: args.nshards]
    print(f"[compare] shard {args.shard}/{args.nshards}: {len(shard_tasks)}/{len(tasks)} tasks "
          f"-> {csv_path}  device={device}")

    rows = []
    with open(csv_path, "a", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=FIELDS)
        if write_header:
            wr.writeheader()

        # Group shard tasks by env so each env's dataset is collected only once.
        buffers: dict[str, wm.data.ReplayBuffer] = {}
        for env_name in sorted({t[0] for t in shard_tasks}):
            env_kw = {"image_size": args.image_size} if env_name == "point-image" else {}
            buf = wm.data.ReplayBuffer()
            cw = wm.World(lambda: make_env(env_name, **env_kw), num_envs=args.num_envs)
            cw.set_policy(wm.RandomPolicy())
            cw.collect(buf, episodes=args.episodes, seed=0)
            buffers[env_name] = (buf, env_kw)
            print(f"=== {env_name}: collected {len(buf.episodes)} episodes / {buf.total_steps} steps ===")

        for env_name, model_name, seed in shard_tasks:
            buf, env_kw = buffers[env_name]
            base = env_name.lower().replace("_", "-")
            is_image = base in IMAGE_ENVS or (base.startswith("tworoom") and not base.endswith("state"))
            obs_key = "pixels" if is_image else "state"
            if env_name not in buffers:
                continue
            env = make_env(env_name, **env_kw)
            obs_shape, action_dim = env.observation_space.shape, env.action_space.shape[0]
            gs = getattr(env, "goal_space", None)
            goal_shape = tuple(gs.shape) if gs is not None else None
            t0 = time.time()
            torch.manual_seed(seed); np.random.seed(seed)
            model = build_model(model_name, obs_key, obs_shape, action_dim,
                                args.latent_dim, args.var_weight, device, goal_shape=goal_shape)
            hist = train_world_model(model, buf, horizon=args.horizon, epochs=args.epochs,
                                     batch_size=args.batch_size, device=device,
                                     log_every=10**9, seed=seed)
            last = hist[-1] if hist else {}
            model = model.to(device).eval()
            succ, rand = evaluate(model, env_name, env_kw, args.num_envs,
                                  args.cem_samples, args.cem_steps, args.horizon,
                                  args.eval_episodes, device)
            row = {
                "env": env_name, "model": model_name, "seed": seed, "horizon": args.horizon,
                "success_rate": round(succ, 1), "random_rate": round(rand, 1),
                "final_loss": round(float(last.get("loss", float("nan"))), 5),
                "dynamics_err": round(float(last.get("dynamics_err", float("nan"))), 5),
                "latent_var": round(float(last.get("latent_var", float("nan"))), 5),
                "elapsed_s": round(time.time() - t0, 1),
            }
            wr.writerow(row); f.flush(); rows.append(row)
            print(f"  {env_name:<12} {model_name:<12} seed{seed}: succ={succ:5.1f}% "
                  f"(rand {rand:4.1f}%) loss={row['final_loss']} var={row['latent_var']}")

    print(f"\n[compare] shard {args.shard} wrote {len(rows)} rows -> {csv_path}")
    if args.nshards == 1:
        print("\n[compare] summary (success_rate mean+-std over seeds):")
        for env_name in args.envs:
            models = args.models if args.models else models_for_env(env_name)
            for m in models:
                rs = [r["success_rate"] for r in rows if r["env"] == env_name and r["model"] == m]
                if rs:
                    print(f"  {env_name:<12} {m:<12} {np.mean(rs):5.1f} +- {np.std(rs):4.1f}  (n={len(rs)})")
    else:
        print("[compare] (multi-shard: merge per-shard CSVs and summarize after; "
              "see scripts/run_worldmodel_compare.sh)")


if __name__ == "__main__":
    main()
