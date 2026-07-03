#!/usr/bin/env python3
"""Flash Brain stream 机制深挖: shallow-fast/deep-slow/gate 在全部 env 上如何分工.

研究落点 = flash 的 stream 模式 (持续思考/指令流水). 本脚本在多类 env 上诊断:
  1. gate opening z (0=shallow 反射, 1=deep 思考): 不同 env 上 stream 怎么调节?
  2. shallow/deep output norm: 两条路径的相对活跃度 (谁主导 feat)
  3. deep state 持续性: CTM activated 跨步变化率 (stream 记忆是否持续)

如果 stream 模式有效: 不同任务类型(POMDP vs 全观测, locomotion vs manipulation)
gate/norm 应有差异化调节 — 即 Flash Brain 的 stream 按任务自适应分工.

跑法: python paper/probe_flash_stream.py   (单进程, 各 env 训 flash + collect 诊断)
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

# 代表性 env (各类型 x full/partial)
ENVS = [
    "pendulum", "pendulum-partial",           # 经典控制 + POMDP
    "mountaincar", "mountaincar-partial",
    "swimmer", "swimmer-partial",              # locomotion
    "pusht", "pusht-partial",                  # manipulation/benchmark
    "cube", "cube-partial",                    # ogbench
    "fetch-push", "fetch-push-partial",
]


def _obs_goal(obs, device):
    s = np.asarray(obs["state"]); g = np.asarray(obs["goal"])
    return torch.as_tensor(np.concatenate([s, g])[None], dtype=torch.float32, device=device)


@torch.no_grad()
def collect_stream(policy, env_name, n_steps, device, seed=0):
    """Collect per-step stream diagnostics: gate z, shallow/deep norm, deep activated delta."""
    env = make_env(env_name)
    obs = env.reset(seed=seed)
    policy.init_state(1, device)
    gates, snorms, dnorms, d_deltas = [], [], [], []
    prev_activated = None
    for _ in range(n_steps):
        x = _obs_goal(obs, device)
        _, dist, _ = policy.forward_feat(x)
        a = dist.mean.cpu().numpy()[0]
        bb = policy.backbone
        gates.append(bb.last_gate_z)
        snorms.append(bb.last_shallow_norm)
        dnorms.append(bb.last_deep_norm)
        # deep activated continuity (CTM state): how much it changed this step
        cur_act = bb.deep._activated if hasattr(bb, 'deep') and hasattr(bb.deep, '_activated') and bb.deep._activated is not None else None
        if cur_act is not None and prev_activated is not None:
            d_deltas.append(float((cur_act - prev_activated).norm().item() / max(prev_activated.norm().item(), 1e-6)))
        if cur_act is not None:
            prev_activated = cur_act.detach().clone()
        obs, r, term, trunc, info = env.step(np.clip(a, env.action_space.low, env.action_space.high))
        done = bool(term or trunc)
        policy.mask_reset(torch.tensor([done], device=device))
        if done:
            env = make_env(env_name); obs = env.reset(seed=seed + len(gates))
            policy.init_state(1, device); prev_activated = None
    return (np.asarray(gates), np.asarray(snorms), np.asarray(dnorms),
            np.asarray(d_deltas) if d_deltas else np.asarray([0.0]))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--total-steps", type=int, default=50000)
    ap.add_argument("--probe-steps", type=int, default=2000)
    ap.add_argument("--seeds", nargs="*", type=int, default=[0, 1])
    ap.add_argument("--d_model", type=int, default=128)
    ap.add_argument("--latent_dim", type=int, default=64)
    ap.add_argument("--memory_length", type=int, default=8)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    Path("figures").mkdir(exist_ok=True)

    print(f"=== flash stream 诊断: {len(ENVS)} envs ===\n")
    results = {}  # env -> {gate, snorm, dnorm, ddelta, succ}
    for env in ENVS:
        gs, ss, ds, dd, sucs = [], [], [], [], []
        for seed in args.seeds:
            e0 = make_env(env)
            od = int(np.prod(e0.observation_space.shape)); gd = int(np.prod(e0.goal_space.shape))
            ad = int(np.prod(e0.action_space.shape))
            pol = build_memory_policy("flash", od, gd, ad, latent_dim=args.latent_dim,
                                      d_model=args.d_model, memory_length=args.memory_length)
            t = PPOTrainer(env, pol, num_envs=8, num_steps=256, device=device, lr=3e-4)
            torch.manual_seed(seed); np.random.seed(seed)
            hist = t.train(args.total_steps, log_iters=4, eval_episodes=8, seed=seed)
            succ = 0.0
            for h in reversed(hist):
                if "eval_success" in h:
                    succ = h["eval_success"]; break
            g, s, d, delta = collect_stream(pol, env, args.probe_steps, device, seed=seed * 7 + 3)
            gs.append(g.mean()); ss.append(s.mean()); ds.append(d.mean()); dd.append(delta.mean()); sucs.append(succ)
            print(f"  {env:22s} s{seed}: succ={succ:5.1f} gate={g.mean():.3f} "
                  f"shallow_norm={s.mean():.2f} deep_norm={d.mean():.2f} deep_delta={delta.mean():.3f}", flush=True)
        results[env] = {"gate": np.mean(gs), "snorm": np.mean(ss), "dnorm": np.mean(ds),
                        "ddelta": np.mean(dd), "succ": np.mean(sucs)}

    # ---- report ----
    print("\n" + "=" * 80)
    print("flash stream 诊断 (mean over seeds)")
    print("=" * 80)
    print(f"{'env':<24}{'succ%':>8}{'gate_z':>9}{'shallow':>10}{'deep':>10}{'deep_Δ':>9}{'deep/shallow':>13}")
    print("-" * 83)
    for env in ENVS:
        r = results[env]
        ratio = r["dnorm"] / max(r["snorm"], 1e-6)
        print(f"{env:<24}{r['succ']:>8.1f}{r['gate']:>9.3f}{r['snorm']:>10.2f}"
              f"{r['dnorm']:>10.2f}{r['ddelta']:>9.3f}{ratio:>13.2f}")

    # ---- fig: stream 调节 across envs ----
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    envs = ENVS
    x = np.arange(len(envs))
    # (a) gate z per env
    gates = [results[e]["gate"] for e in envs]
    colors = ["#d62728" if "-partial" in e else "#1f77b4" for e in envs]
    axes[0].bar(x, gates, color=colors); axes[0].set_xticks(x)
    axes[0].set_xticklabels([e.replace("-", "\n") for e in envs], fontsize=7, rotation=0)
    axes[0].set_ylabel("gate opening z (0=shallow, 1=deep)")
    axes[0].set_title("(a) Stream 调节: gate per env (红=partial/POMDP)")
    axes[0].axhline(0.23, color="gray", ls="--", alpha=0.5, label="~0.23 baseline")
    axes[0].legend(fontsize=8)
    # (b) shallow vs deep norm
    sn = [results[e]["snorm"] for e in envs]; dn = [results[e]["dnorm"] for e in envs]
    w = 0.35
    axes[1].bar(x - w/2, sn, w, label="shallow", color="#2ca02c")
    axes[1].bar(x + w/2, dn, w, label="deep", color="#ff7f0e")
    axes[1].set_xticks(x); axes[1].set_xticklabels([e[:8] for e in envs], fontsize=6, rotation=45)
    axes[1].set_ylabel("output norm"); axes[1].legend(fontsize=8)
    axes[1].set_title("(b) shallow vs deep 活跃度")
    # (c) deep state continuity (delta) — stream 持续性
    dd = [results[e]["ddelta"] for e in envs]
    axes[2].bar(x, dd, color=colors); axes[2].set_xticks(x)
    axes[2].set_xticklabels([e.replace("-", "\n") for e in envs], fontsize=7, rotation=0)
    axes[2].set_ylabel("deep activated relative Δ per step")
    axes[2].set_title("(c) stream 持续性: deep state 跨步变化率")
    plt.tight_layout()
    plt.savefig("figures/fig5_flash_stream.png", bbox_inches="tight"); plt.close()
    print("\n[fig5] -> figures/fig5_flash_stream.png")

    # 判读
    print("\n判读:")
    full_gates = [results[e]["gate"] for e in ENVS if "-partial" not in e]
    partial_gates = [results[e]["gate"] for e in ENVS if "-partial" in e]
    if np.mean(partial_gates) - np.mean(full_gates) > 0.03:
        print("  -> POMDP(partial) gate 更高: stream 在需要记忆时让 deep 更多介入 (自适应分工)")
    else:
        print("  -> gate 在 full/partial 间差异不大: stream 调节不显著 (但 shallow/deep norm 可能有分工)")
    high_deep = [e for e in ENVS if results[e]["dnorm"] / max(results[e]["snorm"], 1e-6) > 1.0]
    if high_deep:
        print(f"  -> deep 主导的 env (deep_norm>shallow): {high_deep} — 这些任务 deep 思考更重要")


if __name__ == "__main__":
    main()
