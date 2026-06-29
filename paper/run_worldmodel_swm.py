#!/usr/bin/env python
"""Bridge: run our world models (jepa-mlp / stream-ctm / ctm-encoder) on ANY
stable-worldmodel environment, using swm's World / CEMSolver / WorldModelPolicy
for collection + evaluation. Our model only has to be a swm Costable (it is).

One invocation = one (env, model, seed) run. The launcher
(scripts/run_worldmodel_swm.sh) enumerates the full grid and shards it across
GPUs. A `--prepare` mode collects an env's dataset once (reused across models /
seeds / shards) so we don't re-roll the env every run.

REQUIRES stable-worldmodel + its env deps installed (compute machine only):
    pip install -e stable-worldmodel[all]        # or: pip install stable-worldmodel[all]

Smoke (on compute, after install):
    python paper/run_worldmodel_swm.py --env swm/PushT-v1 --model jepa-mlp --episodes 20 --epochs 5 --quick
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
from worldmodel.wm import (
    WorldModel, CNNEncoder, MLPPredictor, StreamingCTMPredictor, build_jepa_wm,
)

import stable_worldmodel as swm
from stable_worldmodel.policy import WorldModelPolicy as SwmWM_Policy  # noqa
from stable_worldmodel.solver import CEMSolver as SwmCEM

FIELDS = [
    "env", "model", "seed", "horizon", "image_size", "success_rate",
    "random_rate", "final_loss", "dynamics_err", "latent_var", "elapsed_s",
]


def build_model(model_name, channels, action_dim, latent_dim, var_weight, device):
    obs_shape = (channels, 0, 0)  # encoder is shape-agnostic (AdaptiveAvgPool)
    if model_name == "jepa-mlp":
        m = build_jepa_wm("pixels", obs_shape, action_dim, latent_dim=latent_dim, var_weight=var_weight)
    elif model_name == "stream-ctm":
        encoder = CNNEncoder(latent_dim=latent_dim, channels=channels)
        predictor = StreamingCTMPredictor(
            latent_dim=latent_dim, action_dim=action_dim,
            d_model=max(64, latent_dim * 2), memory_length=8, nlm_hidden=8,
        )
        m = WorldModel(encoder=encoder, predictor=predictor, obs_key="pixels",
                       action_dim=action_dim, cost_mode="last", var_weight=var_weight)
    elif model_name == "ctm-encoder":
        from worldmodel.wm import build_ctm_wm
        m = build_ctm_wm(action_dim=action_dim, iterations=5, d_model=128,
                         n_synch_out=latent_dim, backbone_type="resnet18-1",
                         var_weight=var_weight)
    else:
        raise KeyError(model_name)
    return m.to(device)


class _SWMChunkDataset(torch.utils.data.Dataset):
    """Adapt an swm dataset to our JEPA training format.

    swm yields chunks of `num_steps` frames; we emit (H+1) frames + H actions,
    pixels normalized to [0,1].
    """

    def __init__(self, ds, horizon: int):
        self.ds = ds
        self.horizon = horizon

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        item = self.ds[i]
        px = item["pixels"]                       # (T, C, H, W)
        if not torch.is_tensor(px):
            px = torch.as_tensor(np.asarray(px))
        px = px.float()
        if px.max() > 1.5:
            px = px / 255.0
        act = item["action"]                      # (T, A) or (T, ...) -> (T, A)
        if not torch.is_tensor(act):
            act = torch.as_tensor(np.asarray(act))
        act = act.float().reshape(act.shape[0], -1)
        # need H+1 frames and H actions
        H = self.horizon
        if px.shape[0] < H + 1:
            raise ValueError(f"chunk has {px.shape[0]} frames, need {H+1}")
        return {"pixels": px[: H + 1], "action": act[: H]}


def _collate(batch):
    return {k: torch.stack([b[k] for b in batch], 0) for k in batch[0]}


def prepare_data(env_name, data_path, episodes, num_envs, image_size, seed=0):
    world = swm.World(env_name, num_envs=num_envs, image_shape=(image_size, image_size))
    world.set_policy(swm.policy.RandomPolicy())
    Path(data_path).parent.mkdir(parents=True, exist_ok=True)
    world.collect(data_path, episodes=episodes, seed=seed)
    world.close()
    print(f"[prepare] {env_name}: collected {episodes} episodes -> {data_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", required=True, help="swm env id, e.g. swm/PushT-v1")
    ap.add_argument("--model", default="jepa-mlp", choices=["jepa-mlp", "stream-ctm", "ctm-encoder"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--episodes", type=int, default=120)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--horizon", type=int, default=6)
    ap.add_argument("--image_size", type=int, default=64)
    ap.add_argument("--latent_dim", type=int, default=64)
    ap.add_argument("--var_weight", type=float, default=4.0)
    ap.add_argument("--cem_samples", type=int, default=128)
    ap.add_argument("--cem_steps", type=int, default=6)
    ap.add_argument("--eval_episodes", type=int, default=20)
    ap.add_argument("--num_envs", type=int, default=8)
    ap.add_argument("--data_root", default="data/swm_wm")
    ap.add_argument("--csv", default="csv_data/worldmodel_swm_results.csv")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--prepare", action="store_true", help="collect dataset only, then exit")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    if args.quick:
        args.episodes = 16; args.epochs = 4; args.eval_episodes = 6
        args.cem_samples = 32; args.cem_steps = 3

    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    env_tag = args.env.replace("/", "_")
    data_path = os.path.join(args.data_root, f"{env_tag}_{args.image_size}_{args.episodes}.lance")

    if args.prepare:
        prepare_data(args.env, data_path, args.episodes, args.num_envs, args.image_size)
        return

    # 1) Data (collect on first need).
    if not os.path.exists(data_path):
        print(f"[run] dataset missing, collecting -> {data_path}")
        prepare_data(args.env, data_path, args.episodes, args.num_envs, args.image_size)
    ds = swm.data.load_dataset(data_path, num_steps=args.horizon + 1)
    train_ds = _SWMChunkDataset(ds, args.horizon)
    loader = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True,
        collate_fn=_collate, num_workers=0,
    )
    # action dim from data
    sample = train_ds[0]
    channels, action_dim = sample["pixels"].shape[0], sample["action"].shape[1]
    print(f"[run] env={args.env} model={args.model} seed={args.seed} "
          f"img=({channels},{args.image_size},{args.image_size}) act={action_dim} device={device}")

    # 2) Train.
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    model = build_model(args.model, channels, action_dim, args.latent_dim, args.var_weight, device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    last = {}
    t0 = time.time()
    model.train()
    for ep in range(args.epochs):
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            loss, metrics = model.jepa_loss(batch)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            model._update_ema()
            last = metrics
    model = model.to(device).eval()

    # 3) Evaluate with swm's CEM + WorldModelPolicy (drives the real env).
    solver = SwmCEM(model=model, num_samples=args.cem_samples, n_steps=args.cem_steps,
                    topk=max(4, args.cem_samples // 8), device=device)
    plan_cfg = swm.policy.PlanConfig(horizon=args.horizon, receding_horizon=1)
    eworld = swm.World(args.env, num_envs=args.num_envs, image_shape=(args.image_size, args.image_size))
    eworld.set_policy(swm.policy.WorldModelPolicy(solver=solver, config=plan_cfg))
    res = eworld.evaluate(episodes=args.eval_episodes, seed=100)
    # random baseline
    rworld = swm.World(args.env, num_envs=args.num_envs, image_shape=(args.image_size, args.image_size))
    rworld.set_policy(swm.policy.RandomPolicy())
    rres = rworld.evaluate(episodes=args.eval_episodes, seed=100)
    eworld.close(); rworld.close()

    row = {
        "env": args.env, "model": args.model, "seed": args.seed, "horizon": args.horizon,
        "image_size": args.image_size, "success_rate": round(float(res["success_rate"]), 2),
        "random_rate": round(float(rres["success_rate"]), 2),
        "final_loss": round(float(last.get("loss", float("nan"))), 5),
        "dynamics_err": round(float(last.get("dynamics_err", float("nan"))), 5),
        "latent_var": round(float(last.get("latent_var", float("nan"))), 5),
        "elapsed_s": round(time.time() - t0, 1),
    }
    Path("csv_data").mkdir(exist_ok=True)
    write_header = not os.path.exists(args.csv)
    with open(args.csv, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if write_header:
            w.writeheader()
        w.writerow(row)
    print(f"[run] {args.env} {args.model} seed{args.seed}: succ={row['success_rate']}% "
          f"(rand {row['random_rate']}%) loss={row['final_loss']} var={row['latent_var']} ({row['elapsed_s']}s)")


if __name__ == "__main__":
    main()
