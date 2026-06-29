#!/usr/bin/env python
"""CTM-vs-JEPA world-model comparison (replaces a deep notebook).

Self-contained: collect ONE shared dataset (random policy), then sweep world
models — CTM encoder vs plain JEPA encoder — train each, evaluate with CEM, and
log a row per run to ``csv_data/worldmodel_results.csv``. This is both the
result harvest and the two-step *functional validation* (AGENTS.md): the sweep
intentionally varies the encoder's defining hyperparameter (CTM ``iterations``,
JEPA ``latent_dim``) so that, if the encoders are actually wired in, the
success rates must differ across the sweep.

Run on the compute machine:
    nohup python paper/run_worldmodel.py --env point-image --gpus 1 \
        > logs/worldmodel.log 2>&1 &

Quick local check:
    python paper/run_worldmodel.py --quick
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
from worldmodel.wm import build_ctm_wm, build_jepa_wm
from worldmodel.train import train_world_model

FIELDS = [
    "exp_name", "env", "model", "seed", "iterations", "latent_dim",
    "var_weight", "horizon", "epochs", "success_rate", "random_rate",
    "final_loss", "dynamics_err", "latent_var", "elapsed_s",
]


def _build_model(args, model_kind: str, overrides: dict):
    action_dim = args.action_dim
    if model_kind == "ctm":
        return build_ctm_wm(
            action_dim=action_dim,
            iterations=overrides.get("iterations", args.ctm_iterations),
            d_model=args.ctm_d_model,
            n_synch_out=overrides.get("latent_dim", args.latent_dim),
            backbone_type="resnet18-1",
            var_weight=args.var_weight,
            ema_decay=args.ema_decay,
        )
    return build_jepa_wm(
        obs_key="pixels", obs_shape=(3, args.image_size, args.image_size),
        action_dim=action_dim,
        latent_dim=overrides.get("latent_dim", args.latent_dim),
        var_weight=args.var_weight, ema_decay=args.ema_decay,
    )


def _run_one(args, name, model_kind, overrides, buffer, device):
    t0 = time.time()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    model = _build_model(args, model_kind, overrides).to(device)
    history = train_world_model(
        model, buffer, horizon=args.horizon, epochs=args.epochs,
        batch_size=args.batch_size, device=device, log_every=10**9, seed=args.seed,
    )
    last = history[-1] if history else {}
    model = model.to(device).eval()

    solver = wm.solver.CEMSolver(
        model=model, num_samples=args.cem_samples, n_steps=args.cem_steps,
        topk=max(4, args.cem_samples // 8), device=device,
    )
    ew = wm.World(lambda: make_env(args.env), num_envs=args.num_envs)
    ew.set_policy(wm.WorldModelPolicy(solver=solver, config=wm.PlanConfig(horizon=args.horizon)))
    res = ew.evaluate(episodes=args.eval_episodes, seed=100)

    rw = wm.World(lambda: make_env(args.env), num_envs=args.num_envs)
    rw.set_policy(wm.RandomPolicy())
    rres = rw.evaluate(episodes=args.eval_episodes, seed=100)

    row = {
        "exp_name": name, "env": args.env, "model": model_kind, "seed": args.seed,
        "iterations": overrides.get("iterations", args.ctm_iterations),
        "latent_dim": overrides.get("latent_dim", args.latent_dim),
        "var_weight": args.var_weight, "horizon": args.horizon, "epochs": args.epochs,
        "success_rate": round(res["success_rate"], 2),
        "random_rate": round(rres["success_rate"], 2),
        "final_loss": round(float(last.get("loss", float("nan"))), 5),
        "dynamics_err": round(float(last.get("dynamics_err", float("nan"))), 5),
        "latent_var": round(float(last.get("latent_var", float("nan"))), 5),
        "elapsed_s": round(time.time() - t0, 1),
    }
    print(f"  [{name}] {model_kind} succ={row['success_rate']}% "
          f"(rand {row['random_rate']}%) loss={row['final_loss']} "
          f"var={row['latent_var']} ({row['elapsed_s']}s)")
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="point-image", choices=["point-image"])
    ap.add_argument("--image_size", type=int, default=32)
    ap.add_argument("--episodes", type=int, default=80)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--horizon", type=int, default=5)
    ap.add_argument("--latent_dim", type=int, default=32)
    ap.add_argument("--ctm_iterations", type=int, default=8)
    ap.add_argument("--ctm_d_model", type=int, default=128)
    ap.add_argument("--var_weight", type=float, default=1.0)
    ap.add_argument("--ema_decay", type=float, default=0.0)
    ap.add_argument("--cem_samples", type=int, default=128)
    ap.add_argument("--cem_steps", type=int, default=6)
    ap.add_argument("--eval_episodes", type=int, default=16)
    ap.add_argument("--num_envs", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--csv_suffix", default="", help="append to csv filename (e.g. _s1)")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    if args.quick:
        args.episodes = 20; args.epochs = 12; args.eval_episodes = 8
        args.cem_samples = 48; args.cem_steps = 4
        args.ctm_iterations = 5; args.latent_dim = 16

    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    env = make_env(args.env, image_size=args.image_size)
    args.action_dim = env.action_space.shape[0]
    csv_path = f"csv_data/worldmodel_results{args.csv_suffix}.csv"
    print(f"[wm] env={args.env} device={device} action_dim={args.action_dim} csv={csv_path}")

    # Shared dataset collected once.
    buffer = wm.data.ReplayBuffer()
    cw = wm.World(lambda: make_env(args.env, image_size=args.image_size), num_envs=args.num_envs)
    cw.set_policy(wm.RandomPolicy())
    cw.collect(buffer, episodes=args.episodes, seed=0)
    print(f"[wm] collected {len(buffer.episodes)} episodes / {buffer.total_steps} steps\n")

    # Sweep: CTM iterations and JEPA latent_dim (functional validation knobs),
    # plus the matched-latent head-to-head.
    plan = []
    for it in sorted(set([args.ctm_iterations, 3, 12])):
        plan.append((f"ctm_it{it}", "ctm", {"iterations": it}))
    for ld in sorted(set([args.latent_dim, 16, 64])):
        plan.append((f"jepa_ld{ld}", "jepa", {"latent_dim": ld}))
    plan.append((f"ctm_it{args.ctm_iterations}_ld{args.latent_dim}_MATCH", "ctm",
                 {"iterations": args.ctm_iterations, "latent_dim": args.latent_dim}))

    Path("csv_data").mkdir(exist_ok=True)
    write_header = not os.path.exists(csv_path)
    rows = []
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if write_header:
            w.writeheader()
        for name, kind, ov in plan:
            try:
                row = _run_one(args, name, kind, ov, buffer, device)
                w.writerow(row); f.flush()
                rows.append(row)
            except Exception as e:
                print(f"  [{name}] FAILED: {e}")

    print(f"\n[wm] wrote {len(rows)} rows -> {csv_path}")
    if rows:
        ctm = [r for r in rows if r["model"] == "ctm"]
        jepa = [r for r in rows if r["model"] == "jepa"]
        if ctm:
            print(f"[wm] CTM  mean success: {np.mean([r['success_rate'] for r in ctm]):.1f}%")
        if jepa:
            print(f"[wm] JEPA mean success: {np.mean([r['success_rate'] for r in jepa]):.1f}%")


if __name__ == "__main__":
    main()
