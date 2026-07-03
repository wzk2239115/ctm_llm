#!/usr/bin/env python3
"""Probe gate adaptation: does Flash Brain's GRUGate learn to OPEN on POMDP
(deep CTM engages to infer occluded state) and CLOSE on fully-observed tasks
(shallow reflex dominates, since memory is a burden there)?

gate z: 0 = shallow path only, 1 = deep CTM fully engaged.
If the multi-timescale claim holds, z should be LOW on fully-observed pendulum
and HIGH on pendulum-partial — i.e. the gate self-adapts to task demands.

跑法: python paper/probe_gate_adaptation.py   ->  figures/fig4_gate_adaptation.png
"""
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from worldmodel.envs import make_env
from worldmodel.rl.ppo import PPOTrainer
from worldmodel.rl.memory_policy import build_memory_policy

ENVS = ["pendulum", "pendulum-partial"]


def _obs_goal(obs, device):
    s = np.asarray(obs["state"]); g = np.asarray(obs["goal"])
    return torch.as_tensor(np.concatenate([s, g])[None], dtype=torch.float32, device=device)


@torch.no_grad()
def collect_gate(policy, env_name, n_steps, device, seed=0):
    env = make_env(env_name)
    obs = env.reset(seed=seed)
    policy.init_state(1, device)
    gates = []
    for _ in range(n_steps):
        x = _obs_goal(obs, device)
        _, dist, _ = policy.forward_feat(x)
        a = dist.mean.cpu().numpy()[0]
        gates.append(policy.backbone.last_gate_z)
        obs, r, term, trunc, info = env.step(np.clip(a, env.action_space.low, env.action_space.high))
        done = bool(term or trunc)
        policy.mask_reset(torch.tensor([done], device=device))
        if done:
            env = make_env(env_name); obs = env.reset(seed=seed + len(gates))
            policy.init_state(1, device)
    return np.asarray(gates)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--total-steps", type=int, default=60000)
    ap.add_argument("--probe-steps", type=int, default=3000)
    ap.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2])
    ap.add_argument("--d_model", type=int, default=128)
    ap.add_argument("--latent_dim", type=int, default=64)
    ap.add_argument("--memory_length", type=int, default=8)
    ap.add_argument("--ppo-envs", type=int, default=8)
    ap.add_argument("--ppo-steps", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    Path("figures").mkdir(exist_ok=True)

    print(f"=== gate adaptation probe: flash on {ENVS} ===\n")
    all_gates = {e: [] for e in ENVS}
    for env in ENVS:
        for seed in args.seeds:
            e0 = make_env(env)
            od = int(np.prod(e0.observation_space.shape)); gd = int(np.prod(e0.goal_space.shape))
            ad = int(np.prod(e0.action_space.shape))
            pol = build_memory_policy("flash", od, gd, ad, latent_dim=args.latent_dim,
                                      d_model=args.d_model, memory_length=args.memory_length)
            t = PPOTrainer(env, pol, num_envs=args.ppo_envs, num_steps=args.ppo_steps,
                           device=device, lr=args.lr)
            torch.manual_seed(seed); np.random.seed(seed)
            t.train(args.total_steps, log_iters=5, eval_episodes=8, seed=seed)
            g = collect_gate(pol, env, args.probe_steps, device, seed=seed * 7 + 3)
            all_gates[env].append(g)
            print(f"  {env:20s} seed{seed}: gate z mean={g.mean():.3f} std={g.std():.3f}  "
                  f"(0=shallow, 1=deep)", flush=True)

    print("\n" + "=" * 60)
    print("gate adaptation: mean gate opening z (0=shallow reflex, 1=deep CTM)")
    print("=" * 60)
    for env in ENVS:
        allg = np.concatenate(all_gates[env])
        print(f"  {env:20s}: z = {allg.mean():.3f} +- {allg.std():.3f}")
    fo = np.concatenate(all_gates["pendulum"])
    fp = np.concatenate(all_gates["pendulum-partial"])
    delta = fp.mean() - fo.mean()
    print(f"\n  partial - full = {delta:+.3f}")
    if delta > 0.05:
        print("  >>> 自适应成立: POMDP gate 更开 (deep 介入), 全观测更关 (shallow 主导)")
    elif delta < -0.05:
        print("  >>> 反向 (全观测更开) — 和预期相反, 需排查")
    else:
        print("  >>> 自适应不明显")

    # ---- fig4 ----
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.5))
    colors = {"pendulum": "#1f77b4", "pendulum-partial": "#d62728"}
    labels = {"pendulum": "fully-observed (memory is burden)",
              "pendulum-partial": "POMDP (need memory)"}
    for env in ENVS:
        g = all_gates[env][0][:300]
        a1.plot(g, color=colors[env], lw=1.5, alpha=0.85, label=labels[env])
    a1.set_xlabel("env step"); a1.set_ylabel("gate opening z")
    a1.set_title("(a) Gate opens on POMDP, closes when fully observed")
    a1.legend(frameon=False, fontsize=9); a1.set_ylim(0, 1)
    for env in ENVS:
        allg = np.concatenate(all_gates[env])
        a2.hist(allg, bins=40, alpha=0.55, color=colors[env], label=labels[env], density=True)
    a2.set_xlabel("gate opening z (0=shallow, 1=deep)"); a2.set_ylabel("density")
    a2.set_title("(b) Gate distribution shifts with task")
    a2.legend(frameon=False, fontsize=9)
    plt.tight_layout()
    plt.savefig("figures/fig4_gate_adaptation.png", bbox_inches="tight"); plt.close()
    print("\n[fig4] -> figures/fig4_gate_adaptation.png")


if __name__ == "__main__":
    main()
