#!/usr/bin/env python
"""Experiment: show the streaming-CTM structure works as a world model.

The streaming CTM (:mod:`worldmodel.wm.streaming`) is a *continuous*
ingest / think / emit latent-dynamics module — a persistent CTM recurrence with
no frozen input and no per-sample reset. This script proves it functions inside
the world-model pipeline by swapping it in for the Markov MLP predictor while
keeping the encoder, dataset, and CEM solver identical, then comparing:

  A. ``jepa-mlp``      — :class:`MLPPredictor` (stateless, z+action -> z_next)
  B. ``stream-ctm``    — :class:`StreamingCTMPredictor` (persistent NLM state)

Both are trained AND evaluated at the same horizon, so the comparison is fair.
Outputs ``csv_data/streaming_ctm_results.csv``. A streaming success rate that
beats random and is competitive with the MLP baseline is the evidence that the
always-on structure works as a world model (enables CEM planning).

Run:
    python scripts/experiment_streaming_ctm.py                       # state, reliable
    python scripts/experiment_streaming_ctm.py --env point-image      # image (harder)
    python scripts/experiment_streaming_ctm.py --quick                # tiny, fast check

Note: on ``point-image`` the from-scratch CNN encoder is prone to JEPA collapse
(``latent_var`` -> 0), which affects BOTH predictors and is orthogonal to the
streaming question. ``point-state`` (MLP encoder) is collapse-free and is the
clean demonstration that the streaming structure works as a world model.
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
)
from worldmodel.train import train_world_model

FIELDS = [
    "variant", "env", "predictor", "seed", "horizon",
    "success_rate", "random_rate", "final_loss", "dynamics_err", "latent_var",
    "elapsed_s",
]


def build_model(variant: str, obs_key: str, obs_shape, action_dim, latent_dim, var_weight, device):
    if obs_key == "pixels":
        encoder = CNNEncoder(latent_dim=latent_dim, channels=int(obs_shape[0]))
    else:
        encoder = MLPEncoder(obs_dim=int(obs_shape[0]), latent_dim=latent_dim)
    if variant == "mlp":
        predictor = MLPPredictor(latent_dim=latent_dim, action_dim=action_dim)
    elif variant == "stream":
        predictor = StreamingCTMPredictor(
            latent_dim=latent_dim, action_dim=action_dim,
            d_model=max(64, latent_dim * 2), memory_length=8, nlm_hidden=8,
        )
    else:
        raise KeyError(variant)
    return WorldModel(
        encoder=encoder, predictor=predictor, obs_key=obs_key,
        action_dim=action_dim, cost_mode="last", var_weight=var_weight,
    ).to(device)


def evaluate(model, args, env_kw, device, horizon: int):
    solver = wm.solver.CEMSolver(
        model=model, num_samples=args.cem_samples, n_steps=args.cem_steps,
        topk=max(4, args.cem_samples // 8), device=device,
    )
    ew = wm.World(lambda: make_env(args.env, **env_kw), num_envs=args.num_envs)
    ew.set_policy(wm.WorldModelPolicy(solver=solver, config=wm.PlanConfig(horizon=horizon)))
    res = ew.evaluate(episodes=args.eval_episodes, seed=100)
    rw = wm.World(lambda: make_env(args.env, **env_kw), num_envs=args.num_envs)
    rw.set_policy(wm.RandomPolicy())
    rres = rw.evaluate(episodes=args.eval_episodes, seed=100)
    return res["success_rate"], rres["success_rate"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="point-state", choices=["point-image", "point-state"])
    ap.add_argument("--image_size", type=int, default=32)
    ap.add_argument("--episodes", type=int, default=60)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--horizon", type=int, default=6, help="train + eval horizon H")
    ap.add_argument("--latent_dim", type=int, default=32)
    ap.add_argument("--var_weight", type=float, default=4.0, help="VICReg anti-collapse (image needs >=4)")
    ap.add_argument("--cem_samples", type=int, default=128)
    ap.add_argument("--cem_steps", type=int, default=6)
    ap.add_argument("--eval_episodes", type=int, default=16)
    ap.add_argument("--num_envs", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    if args.quick:
        args.episodes = 16; args.epochs = 10; args.eval_episodes = 8
        args.cem_samples = 48; args.cem_steps = 4
        args.horizon = 4; args.latent_dim = 16

    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    env_kw = {"image_size": args.image_size} if args.env == "point-image" else {}
    obs_key = "pixels" if args.env == "point-image" else "state"
    env = make_env(args.env, **env_kw)
    obs_shape = env.observation_space.shape
    action_dim = env.action_space.shape[0]
    print(f"[stream-exp] env={args.env} obs={obs_key}{obs_shape} act={action_dim} "
          f"device={device} H={args.horizon}")

    # Shared dataset collected once.
    buffer = wm.data.ReplayBuffer()
    cw = wm.World(lambda: make_env(args.env, **env_kw), num_envs=args.num_envs)
    cw.set_policy(wm.RandomPolicy())
    cw.collect(buffer, episodes=args.episodes, seed=0)
    print(f"[stream-exp] collected {len(buffer.episodes)} episodes / {buffer.total_steps} steps\n")

    Path("csv_data").mkdir(exist_ok=True)
    csv_path = "csv_data/streaming_ctm_results.csv"
    write_header = not os.path.exists(csv_path)
    rows = []
    with open(csv_path, "a", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=FIELDS)
        if write_header:
            wr.writeheader()
        for variant in ["mlp", "stream"]:
            t0 = time.time()
            torch.manual_seed(args.seed); np.random.seed(args.seed)
            model = build_model(variant, obs_key, obs_shape, action_dim, args.latent_dim, args.var_weight, device)
            hist = train_world_model(
                model, buffer, horizon=args.horizon, epochs=args.epochs,
                batch_size=args.batch_size, device=device, log_every=10**9, seed=args.seed,
            )
            last = hist[-1] if hist else {}
            model = model.to(device).eval()
            name = {"mlp": "jepa-mlp", "stream": "stream-ctm"}[variant]
            succ, rand = evaluate(model, args, env_kw, device, args.horizon)
            row = {
                "variant": name, "env": args.env, "predictor": variant, "seed": args.seed,
                "horizon": args.horizon,
                "success_rate": round(succ, 1), "random_rate": round(rand, 1),
                "final_loss": round(float(last.get("loss", float("nan"))), 5),
                "dynamics_err": round(float(last.get("dynamics_err", float("nan"))), 5),
                "latent_var": round(float(last.get("latent_var", float("nan"))), 5),
                "elapsed_s": round(time.time() - t0, 1),
            }
            wr.writerow(row); f.flush(); rows.append(row)
            print(f"  [{name} @ H={args.horizon}] succ={succ:.1f}% (rand {rand:.1f}%) "
                  f"loss={row['final_loss']} var={row['latent_var']} ({row['elapsed_s']}s)")
            print()

    print(f"[stream-exp] wrote {len(rows)} rows -> {csv_path}")
    mlp = [r for r in rows if r["predictor"] == "mlp"]
    stm = [r for r in rows if r["predictor"] == "stream"]
    print("\n[stream-exp] summary (success_rate % at H={}):".format(args.horizon))
    if mlp: print(f"  jepa-mlp   : {mlp[0]['success_rate']}%   (rand {mlp[0]['random_rate']}%)")
    if stm: print(f"  stream-ctm : {stm[0]['success_rate']}%   (rand {stm[0]['random_rate']}%)")
    if stm:
        verdict = "WORKS" if stm[0]["success_rate"] > stm[0]["random_rate"] else "no signal"
        print(f"  -> stream-ctm as a world model: {verdict}")


if __name__ == "__main__":
    main()
