#!/usr/bin/env python
"""Offline comparison aligned with stable-wm: expert data -> GCBC (policy).

Paradigm (mirrors stable-worldmodel):
  1. COLLECT: ExpertPolicy collects goal-reaching trajectories (ONCE per env)
  2. TRAIN:   GCBC trains memory policies on the SAME expert buffer
  3. EVAL:    direct rollout (no further training / online interaction)

This replaces the old PPO-from-scratch + random-BC approach which was both
unfair (PPO collects its own data != world-model data) and harmful (random
actions are noise). Expert data has goal-reaching signal, so GCBC works.

All memory backbones (mlp/ctm/lstm/gru/transformer/flash/flash-shallow/
flash-deep) share the same GCBC trainer + expert data, so the comparison is
purely about the memory mechanism.

Usage (local smoke):
  python paper/run_offline_compare.py --envs point-state --backbones flash mlp --seeds 0 --local

Full sweep:
  python paper/run_offline_compare.py \
      --envs point-state tworoom-state cartpole cartpole-partial pendulum pendulum-partial \
             reacher reacher-partial \
      --backbones mlp ctm lstm gru transformer flash flash-shallow flash-deep \
      --seeds 0 1 2 --report csv_data/offline_compare.md
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from collections import defaultdict

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from worldmodel.envs import make_env
from worldmodel.data import ReplayBuffer
from worldmodel.world import World
from worldmodel.policy import ExpertPolicy, RandomPolicy
from worldmodel.rl.memory_policy import build_memory_policy
from worldmodel.rl.gcbc import GCBCTrainer


def collect_expert_buffer(env_name, episodes, seed=0, num_envs=8, noise=0.1):
    """Collect expert trajectories + measure expert success rate (data quality)."""
    # evaluate expert quality first
    ew = World(lambda: make_env(env_name), num_envs=min(num_envs, 4))
    ew.set_policy(ExpertPolicy(seed=seed + 777, noise=noise))
    eres = ew.evaluate(episodes=20, seed=999)
    # collect
    cw = World(lambda: make_env(env_name), num_envs=num_envs)
    cw.set_policy(ExpertPolicy(seed=seed, noise=noise))
    buf = ReplayBuffer()
    cw.collect(buf, episodes=episodes, seed=seed)
    print(f"  [collect] {env_name}: {len(buf.episodes)} eps / {buf.total_steps} steps, "
          f"expert success={eres['success_rate']:.1f}% (noise={noise})", flush=True)
    return buf, eres['success_rate']


def collect_random_buffer(env_name, episodes, seed=0, num_envs=8):
    """Collect random-policy data (baseline data quality for comparison)."""
    cw = World(lambda: make_env(env_name), num_envs=num_envs)
    cw.set_policy(RandomPolicy(seed=seed))
    buf = ReplayBuffer()
    cw.collect(buf, episodes=episodes, seed=seed)
    return buf


def get_dims(env_name):
    env = make_env(env_name)
    obs_shape = env.observation_space.shape
    goal_shape = env.goal_space.shape
    action_dim = env.action_space.shape[0]
    image = len(obs_shape) == 3
    return int(np.prod(obs_shape)), int(np.prod(goal_shape)), action_dim, image


def run_one(env_name, backbone, seed, buf, expert_succ, args, device='cpu'):
    """GCBC train + eval on an already-collected expert buffer."""
    t0 = time.time()
    torch.manual_seed(seed)
    np.random.seed(seed)

    obs_dim, goal_dim, action_dim, image = get_dims(env_name)
    policy = build_memory_policy(
        backbone, obs_dim, goal_dim, action_dim,
        latent_dim=args.latent_dim, d_model=args.d_model,
        image=image, memory_length=args.memory_length,
    ).to(device)

    trainer = GCBCTrainer(
        policy, env_name, device=device, lr=args.lr, max_grad_norm=0.5)
    trainer.train(buf, args.gcbc_steps, batch_eps=args.batch_eps)

    succ = trainer.evaluate(episodes=args.eval_episodes, seed=1000 + seed)

    elapsed = time.time() - t0
    result = {
        'env': env_name, 'backbone': backbone, 'seed': seed,
        'success_rate': succ, 'expert_success': expert_succ,
        'n_episodes': len(buf.episodes), 'n_steps': buf.total_steps,
        'elapsed_s': round(elapsed, 1), 'device': device,
    }
    print(f"  [done] {env_name}/{backbone}/seed{seed}: "
          f"succ={succ:.1f}% (expert={expert_succ:.1f}%) "
          f"{elapsed:.0f}s", flush=True)
    return result


def run_wm_cem(env_name, buf, seed, args, device='cpu'):
    """World-model + CEM baseline: same expert data -> learn dynamics -> plan.

    Mirrors stable-wm: JEPA world model on expert buffer, then CEM zero-shot
    planning. This is the 'model-based planning' baseline that GCBC policies
    compete against — on the SAME data.
    """
    from worldmodel.wm import build_jepa_wm
    from worldmodel.train import train_world_model
    from worldmodel.solver import CEMSolver
    from worldmodel.policy import WorldModelPolicy, PlanConfig

    t0 = time.time()
    torch.manual_seed(seed)
    np.random.seed(seed)

    env = make_env(env_name)
    obs_shape = env.observation_space.shape
    action_dim = env.action_space.shape[0]
    obs_key = 'pixels' if len(obs_shape) == 3 else 'state'
    goal_dim = int(np.prod(env.goal_space.shape))
    obs_dim = int(np.prod(obs_shape))

    # CEM-WM requires goal_encoder to be trained via JEPA loss, but jepa_loss
    # only encodes obs frames (not goals). When goal_dim == obs_dim, we reuse
    # the obs encoder (trained). When goal_dim != obs_dim, goal_encoder is
    # randomly initialised -> cost is meaningless -> CEM fails. Skip those envs.
    if goal_dim != obs_dim:
        print(f"  [cem-wm] skip {env_name}: goal_dim={goal_dim} != obs_dim={obs_dim} "
              f"(goal_encoder untrained; only same-dim goals supported)", flush=True)
        return None

    model = build_jepa_wm(
        obs_key=obs_key, obs_shape=obs_shape, action_dim=action_dim,
        latent_dim=args.wm_latent_dim, var_weight=1.0,
        goal_shape=env.goal_space.shape,
    )
    train_world_model(
        model, buf, horizon=args.wm_horizon, epochs=args.wm_epochs,
        batch_size=64, device=device,
        log_every=max(1, args.wm_epochs // 4), seed=seed)

    model = model.to(device).eval()
    solver = CEMSolver(
        model=model, num_samples=args.cem_samples, n_steps=args.cem_steps,
        topk=max(4, args.cem_samples // 8), device=device)
    eval_world = World(lambda: make_env(env_name), num_envs=4)
    eval_world.set_policy(WorldModelPolicy(solver=solver,
                                           config=PlanConfig(horizon=args.wm_horizon)))
    res = eval_world.evaluate(episodes=args.eval_episodes, seed=1000 + seed)
    elapsed = time.time() - t0
    print(f"  [cem-wm] {env_name}/seed{seed}: succ={res['success_rate']:.1f}% "
          f"{elapsed:.0f}s", flush=True)
    return {
        'env': env_name, 'backbone': 'cem-wm', 'seed': seed,
        'success_rate': res['success_rate'], 'expert_success': 0,
        'n_episodes': len(buf.episodes), 'n_steps': buf.total_steps,
        'elapsed_s': round(elapsed, 1), 'device': device,
    }


def build_report(results, args):
    """Generate markdown report grouped by env x backbone."""
    by = defaultdict(list)
    for r in results:
        by[(r['env'], r['backbone'])].append(r['success_rate'])

    envs = sorted({r['env'] for r in results})
    backs = args.backbones
    lines = []
    lines.append(f"# Offline comparison: GCBC on expert data vs world-model+CEM\n")
    lines.append(f"生成: {time.strftime('%Y-%m-%d %H:%M:%S')} | "
                 f"runs: {len(results)} | backbones: {backs}\n")
    lines.append(f"范式: ExpertPolicy 收集 → GCBC 训练 (对齐 stable-wm)\n")
    lines.append(f"配置: collect={args.collect_episodes}eps, gcbc={args.gcbc_steps}steps, "
                 f"noise={args.expert_noise}, seeds={args.seeds}\n\n")

    # expert success per env (data quality)
    lines.append("## Expert data quality (goal-reaching success rate)\n")
    lines.append("| env | expert_succ | n_episodes | n_steps |")
    lines.append("|---|---|---|---|")
    seen_env = {}
    for r in results:
        if r['env'] not in seen_env:
            seen_env[r['env']] = r
            lines.append(f"| {r['env']} | {r['expert_success']:.1f}% | "
                         f"{r['n_episodes']} | {r['n_steps']} |")
    lines.append("")

    # success_rate mean+-std
    lines.append("## success_rate mean+-std (GCBC on expert data)\n")
    header = "| env | " + " | ".join(backs) + " |"
    sep = "|---|" + "|".join(["---"] * len(backs)) + "|"
    lines.append(header)
    lines.append(sep)
    for env in envs:
        cells = []
        for b in backs:
            vals = by.get((env, b), [])
            if vals:
                m, s = float(np.mean(vals)), float(np.std(vals))
                cells.append(f"{m:.1f}+-{s:.1f}")
            else:
                cells.append("-")
        lines.append(f"| {env} | " + " | ".join(cells) + " |")
    lines.append("")

    # Flash synergy: mix vs single-path
    if 'flash' in backs and 'flash-shallow' in backs and 'flash-deep' in backs:
        lines.append("## Flash 混合 vs 单路径\n")
        lines.append("| env | flash(混合) | flash-shallow | flash-deep | 混合-shallow | 混合-deep |")
        lines.append("|---|---|---|---|---|---|")
        for env in envs:
            fm = float(np.mean(by.get((env, 'flash'), [0])))
            sm = float(np.mean(by.get((env, 'flash-shallow'), [0])))
            dm = float(np.mean(by.get((env, 'flash-deep'), [0])))
            lines.append(f"| {env} | {fm:.1f} | {sm:.1f} | {dm:.1f} | "
                         f"{fm-sm:+.1f} | {fm-dm:+.1f} |")
        lines.append("")

    # CTM vs RNN
    rnn_backs = [b for b in ['lstm', 'gru', 'transformer'] if b in backs]
    if 'ctm' in backs and rnn_backs:
        lines.append("## CTM vs RNN 系记忆策略\n")
        lines.append("| env | CTM | RNN 均值 | CTM-RNN | 判定 |")
        lines.append("|---|---|---|---|---|")
        for env in envs:
            ctm = float(np.mean(by.get((env, 'ctm'), [0])))
            rnn_vals = [float(np.mean(by.get((env, b), [0]))) for b in rnn_backs]
            rnn_m = float(np.mean(rnn_vals)) if rnn_vals else 0.0
            diff = ctm - rnn_m
            if abs(diff) < 3:
                verdict = "持平"
            elif diff > 0:
                verdict = "CTM 赢"
            else:
                verdict = "CTM 输"
            lines.append(f"| {env} | {ctm:.1f} | {rnn_m:.1f} | {diff:+.1f} | {verdict} |")
        lines.append("")

    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--envs', nargs='+', default=['point-state'])
    ap.add_argument('--backbones', nargs='+',
                    default=['mlp', 'ctm', 'lstm', 'gru', 'transformer',
                             'flash', 'flash-shallow', 'flash-deep'])
    ap.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2])
    ap.add_argument('--collect-episodes', type=int, default=200)
    ap.add_argument('--gcbc-steps', type=int, default=3000)
    ap.add_argument('--eval-episodes', type=int, default=20)
    ap.add_argument('--batch-eps', type=int, default=4)
    ap.add_argument('--expert-noise', type=float, default=0.1)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--d_model', type=int, default=128)
    ap.add_argument('--latent_dim', type=int, default=64)
    ap.add_argument('--memory_length', type=int, default=8)
    ap.add_argument('--num_envs', type=int, default=8)
    ap.add_argument('--device', default='auto')
    ap.add_argument('--report', default='csv_data/offline_compare.md')
    ap.add_argument('--with-random-baseline', action='store_true',
                    help='also run GCBC on random data (should fail -> proves expert data matters)')
    ap.add_argument('--with-wm-cem', action='store_true',
                    help='also run world-model + CEM planning on same expert data')
    ap.add_argument('--wm-horizon', type=int, default=5)
    ap.add_argument('--wm-epochs', type=int, default=50)
    ap.add_argument('--wm-latent-dim', type=int, default=64)
    ap.add_argument('--cem-samples', type=int, default=128)
    ap.add_argument('--cem-steps', type=int, default=8)
    ap.add_argument('--local', action='store_true', help='shrink for fast local check')
    args = ap.parse_args()

    if args.local:
        args.collect_episodes = min(args.collect_episodes, 40)
        args.gcbc_steps = min(args.gcbc_steps, 300)
        args.eval_episodes = min(args.eval_episodes, 6)
        args.seeds = [0]
        args.num_envs = 4

    device = ('cuda' if torch.cuda.is_available() else 'cpu') \
        if args.device == 'auto' else args.device
    print(f"[offline-compare] device={device} envs={args.envs} "
          f"backbones={args.backbones} seeds={args.seeds}")

    results = []
    for env_name in args.envs:
        for seed in args.seeds:
            # collect expert buffer ONCE, shared by all backbones + WM-CEM
            try:
                buf, expert_succ = collect_expert_buffer(
                    env_name, args.collect_episodes, seed=seed,
                    num_envs=args.num_envs, noise=args.expert_noise)
            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"  [FAIL] collect {env_name}/seed{seed}: {e}", flush=True)
                continue

            for backbone in args.backbones:
                try:
                    r = run_one(env_name, backbone, seed, buf, expert_succ, args, device=device)
                    results.append(r)
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    print(f"  [FAIL] {env_name}/{backbone}/seed{seed}: {e}", flush=True)
                    results.append({
                        'env': env_name, 'backbone': backbone, 'seed': seed,
                        'success_rate': -1.0, 'expert_success': expert_succ,
                        'n_episodes': len(buf.episodes), 'n_steps': buf.total_steps,
                        'elapsed_s': 0,
                    })

            if args.with_wm_cem:
                try:
                    r = run_wm_cem(env_name, buf, seed, args, device=device)
                    results.append(r)
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    print(f"  [FAIL] cem-wm {env_name}/seed{seed}: {e}", flush=True)

            if args.with_random_baseline:
                try:
                    rbuf = collect_random_buffer(
                        env_name, args.collect_episodes, seed=seed,
                        num_envs=args.num_envs)
                    r = run_one(env_name, 'mlp-random', seed, rbuf, 0.0, args, device=device)
                    results.append(r)
                except Exception as e:
                    print(f"  [FAIL] random baseline {env_name}: {e}", flush=True)

    # write report
    os.makedirs(os.path.dirname(args.report) or '.', exist_ok=True)
    report = build_report(results, args)
    with open(args.report, 'w') as f:
        f.write(report)

    # write CSV (for multi-GPU merge)
    csv_path = args.report.replace('.md', '.csv')
    with open(csv_path, 'w') as f:
        f.write('env,backbone,seed,success_rate,expert_success,n_episodes,n_steps,elapsed_s\n')
        for r in results:
            f.write(f"{r['env']},{r['backbone']},{r['seed']},"
                    f"{r['success_rate']},{r.get('expert_success',0)},"
                    f"{r.get('n_episodes',0)},{r.get('n_steps',0)},"
                    f"{r.get('elapsed_s',0)}\n")

    print(f"\n[total] {len(results)} runs -> {args.report} + {csv_path}")
    print(report)


if __name__ == '__main__':
    main()
