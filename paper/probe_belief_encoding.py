#!/usr/bin/env python3
"""Probe: does CTM's hidden state encode the occluded belief (angular velocity)?

pendulum-partial hides angular velocity (obs = cos/sin only). A memory policy
that "solves" it must internally infer the hidden θdot from the position
history. We test this directly: train each backbone, then fit a linear probe
(hidden feat -> true θdot) and report R². High R² = the backbone encoded the
belief; low R² = it did not (so its success, if any, is not from belief inference).

This is the mechanistic evidence for "why CTM wins": CTM should show high R² on
θdot (the occluded variable), while LSTM/GRU (which failed on pendulum-partial)
should show low R².

跑法:
    python paper/probe_belief_encoding.py
    python paper/probe_belief_encoding.py --total-steps 80000 --probe-steps 4000
"""
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from worldmodel.envs import make_env
from worldmodel.rl.ppo import PPOTrainer
from worldmodel.rl.memory_policy import build_memory_policy

BACKBONES = ["mlp", "ctm", "lstm", "gru", "transformer"]
ENV = "pendulum-partial"


def _obs_goal(obs_dict, device):
    s = np.asarray(obs_dict["state"]); g = np.asarray(obs_dict["goal"])
    return torch.as_tensor(np.concatenate([s, g])[None], dtype=torch.float32, device=device)


@torch.no_grad()
def collect(policy, env_name, n_steps, device, seed=0):
    """Roll out the trained policy; record (feat, true θdot, true θ) per step."""
    env = make_env(env_name)
    obs = env.reset(seed=seed)
    policy.init_state(1, device)
    feats, thds, ths = [], [], []
    for _ in range(n_steps):
        x = _obs_goal(obs, device)
        feat, dist, _ = policy.forward_feat(x)
        a = dist.mean.cpu().numpy()[0]
        feats.append(feat.squeeze(0).cpu().numpy())
        thds.append(float(env._thd))   # the OCCLUDED variable (belief target)
        ths.append(float(env._th))     # angle (observable via cos/sin, sanity check)
        obs, r, term, trunc, info = env.step(np.clip(a, env.action_space.low, env.action_space.high))
        done = bool(term or trunc)
        policy.mask_reset(torch.tensor([done], device=device))
        if done:
            env = make_env(env_name)
            obs = env.reset(seed=seed + len(feats))
            policy.init_state(1, device)
    return np.array(feats), np.array(thds), np.array(ths)


def linear_probe_r2(X, y, split=0.8):
    """Fit least-squares linear probe on train split, report R² on test split."""
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).ravel()
    n = len(y)
    if n < 20:
        return float("nan")
    idx = np.random.permutation(n)
    ntr = int(n * split)
    tr, te = idx[:ntr], idx[ntr:]
    # augment with bias
    Xtr = np.concatenate([X[tr], np.ones((len(tr), 1))], axis=1)
    Xte = np.concatenate([X[te], np.ones((len(te), 1))], axis=1)
    w, _, _, _ = np.linalg.lstsq(Xtr, y[tr], rcond=None)
    yp = Xte @ w
    ss_res = float(((y[te] - yp) ** 2).sum())
    ss_tot = float(((y[te] - y[te].mean()) ** 2).sum())
    return 1.0 - ss_res / ss_tot if ss_tot > 1e-9 else 0.0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--total-steps", type=int, default=60000, help="PPO steps to train each backbone")
    ap.add_argument("--probe-steps", type=int, default=3000, help="rollout steps for probe data")
    ap.add_argument("--d_model", type=int, default=128)
    ap.add_argument("--latent_dim", type=int, default=64)
    ap.add_argument("--memory_length", type=int, default=8)
    ap.add_argument("--ppo-envs", type=int, default=8)
    ap.add_argument("--ppo-steps", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--eval_episodes", type=int, default=12)
    ap.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2])
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device

    e = make_env(ENV)
    od = int(np.prod(e.observation_space.shape))
    gd = int(np.prod(e.goal_space.shape))
    ad = int(np.prod(e.action_space.shape))
    print(f"=== belief probe on {ENV} (occluded var = angular velocity θdot) ===")
    print(f"device={device}  train={args.total_steps} steps  probe={args.probe_steps} steps  seeds={args.seeds}\n")

    results = {bb: {"thdot": [], "theta": [], "succ": []} for bb in BACKBONES}
    for seed in args.seeds:
        print(f"--- seed {seed} ---")
        for bb in BACKBONES:
            torch.manual_seed(seed); np.random.seed(seed)
            pol = build_memory_policy(bb, od, gd, ad, latent_dim=args.latent_dim,
                                      d_model=args.d_model, memory_length=args.memory_length,
                                      state_gate="gru")
            t = PPOTrainer(ENV, pol, num_envs=args.ppo_envs, num_steps=args.ppo_steps,
                           device=device, lr=args.lr)
            hist = t.train(args.total_steps, log_iters=5, eval_episodes=args.eval_episodes, seed=seed)
            succ = 0.0
            for h in reversed(hist):
                if "eval_success" in h:
                    succ = h["eval_success"]; break
            np.random.seed(seed * 7 + 1)
            feats, thds, ths = collect(pol, ENV, args.probe_steps, device, seed=seed * 13 + 5)
            r2_dot = linear_probe_r2(feats, thds)
            r2_th = linear_probe_r2(feats, ths)
            results[bb]["thdot"].append(r2_dot)
            results[bb]["theta"].append(r2_th)
            results[bb]["succ"].append(succ)
            print(f"  {bb:12s} succ={succ:5.1f}%  probe R²(θdot)={r2_dot:+.3f}  R²(θ)={r2_th:+.3f}")

    print("\n" + "=" * 70)
    print(f"belief probe 结果 ({ENV}, mean over seeds)")
    print("=" * 70)
    print(f"{'backbone':<14}{'succ%':<10}{'R²(θdot 遮挡)':<18}{'R²(θ 可观测)':<16}")
    print("-" * 58)
    for bb in BACKBONES:
        s = np.nanmean(results[bb]["succ"])
        rd = np.nanmean(results[bb]["thdot"])
        rt = np.nanmean(results[bb]["theta"])
        print(f"{bb:<14}{s:<10.1f}{rd:<+18.3f}{rt:<+16.3f}")

    print("\n判读:")
    print("  R²(θdot) 高 = backbone 内部推断出了被遮挡的角速度 (belief encoding)")
    print("  R²(θ)    高 = sanity (θ 从 cos/sin 可算, 所有 backbone 都该高)")
    print("  CTM 应: R²(θdot) 高 (解释它 pendulum-partial 81.7% success)")
    print("  LSTM/GRU 应: R²(θdot) 低 (解释它们失败 ~12%)")


if __name__ == "__main__":
    main()
