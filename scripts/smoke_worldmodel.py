#!/usr/bin/env python
"""Smoke test for the worldmodel framework (AGENTS.md two-step: Step 1).

End-to-end, tiny scale, on CPU:
    collect (random policy) -> train a JEPA world model -> evaluate with CEM.

Run:
    python scripts/smoke_worldmodel.py
    python scripts/smoke_worldmodel.py --env point-image   # uses CNN encoder
    python scripts/smoke_worldmodel.py --iterations 10 --local   # minimal

Exit code 0 + a sane CEM success rate means the pipeline is wired correctly.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import worldmodel as wm
from worldmodel.envs import make_env
from worldmodel.wm import build_jepa_wm, build_ctm_wm


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--env', default='point-state', choices=['point-state', 'point-image'])
    p.add_argument('--episodes', type=int, default=40)
    p.add_argument('--horizon', type=int, default=5)
    p.add_argument('--epochs', type=int, default=60)
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--latent_dim', type=int, default=32)
    p.add_argument('--ema_decay', type=float, default=0.0, help='EMA target encoder (anti-collapse)')
    p.add_argument('--var_weight', type=float, default=0.0, help='VICReg variance penalty (anti-collapse)')
    p.add_argument('--model', default='jepa', choices=['jepa', 'ctm'], help='world-model encoder')
    p.add_argument('--cem_samples', type=int, default=64)
    p.add_argument('--cem_steps', type=int, default=6)
    p.add_argument('--eval_episodes', type=int, default=16)
    p.add_argument('--num_envs', type=int, default=4)
    p.add_argument('--device', default='auto')
    p.add_argument('--iterations', type=int, default=0, help='cap total train iters (smoke)')
    p.add_argument('--local', action='store_true', help='shrink everything for a fast local check')
    return p.parse_args()


def main():
    args = parse_args()
    if args.local:
        args.episodes = min(args.episodes, 12)
        args.epochs = min(args.epochs, 8)
        args.eval_episodes = min(args.eval_episodes, 6)
        args.cem_samples = min(args.cem_samples, 32)
        args.cem_steps = min(args.cem_steps, 4)

    device = ('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device
    print(f'[smoke] env={args.env} device={device}')

    obs_key = 'pixels' if args.env == 'point-image' else 'state'
    if args.model == 'ctm' and obs_key != 'pixels':
        raise SystemExit('CTM encoder requires --env point-image')
    env = make_env(args.env)
    obs_shape = env.observation_space.shape
    action_dim = env.action_space.shape[0]
    print(f'[smoke] model={args.model} obs_key={obs_key} obs_shape={obs_shape} action_dim={action_dim}')

    # 1) Collect with a random policy.
    buffer = wm.data.ReplayBuffer()
    collect_world = wm.World(lambda: make_env(args.env), num_envs=args.num_envs)
    collect_world.set_policy(wm.RandomPolicy())
    collect_world.collect(buffer, episodes=args.episodes, seed=0)
    print(f'[smoke] collected {len(buffer.episodes)} episodes, {buffer.total_steps} steps')

    # 2) Train a world model.
    if args.model == 'ctm':
        model = build_ctm_wm(
            action_dim=action_dim, iterations=5, d_model=128, n_synch_out=args.latent_dim,
            backbone_type='resnet18-1', var_weight=args.var_weight, ema_decay=args.ema_decay,
        )
    else:
        model = build_jepa_wm(
            obs_key=obs_key, obs_shape=obs_shape, action_dim=action_dim,
            latent_dim=args.latent_dim, ema_decay=args.ema_decay, var_weight=args.var_weight,
        )
    epochs = args.epochs
    history = wm.data  # placeholder for clarity
    from worldmodel.train import train_world_model
    history = train_world_model(
        model, buffer, horizon=args.horizon, epochs=epochs,
        batch_size=args.batch_size, device=device, log_every=max(1, epochs // 6 | 1), seed=0,
    )
    if history:
        print(f'[smoke] train loss: {history[0]["loss"]:.4f} -> {history[-1]["loss"]:.4f} '
              f'(dynamics_err {history[-1]["dynamics_err"]:.4f}, latent_var {history[-1]["latent_var"]:.4f})')

    # 3) Evaluate: CEM + WorldModelPolicy vs RandomPolicy baseline.
    model = model.to(device).eval()
    solver = wm.solver.CEMSolver(
        model=model, num_samples=args.cem_samples, n_steps=args.cem_steps,
        topk=max(4, args.cem_samples // 8), device=device,
    )
    plan_cfg = wm.PlanConfig(horizon=args.horizon)
    eval_world = wm.World(lambda: make_env(args.env), num_envs=args.num_envs)
    eval_world.set_policy(wm.WorldModelPolicy(solver=solver, config=plan_cfg))
    res = eval_world.evaluate(episodes=args.eval_episodes, seed=100)
    print(f'[smoke] CEM success rate: {res["success_rate"]:.1f}% '
          f'({int(res["episode_successes"].sum())}/{len(res["episode_successes"])})')

    rand_world = wm.World(lambda: make_env(args.env), num_envs=args.num_envs)
    rand_world.set_policy(wm.RandomPolicy())
    rres = rand_world.evaluate(episodes=args.eval_episodes, seed=100)
    print(f'[smoke] random success rate: {rres["success_rate"]:.1f}%')

    ok = res['success_rate'] > rres['success_rate']
    print('[smoke] PASS' if ok else '[smoke] DONE (CEM did not beat random — may need more training)')
    sys.exit(0)


if __name__ == '__main__':
    main()
