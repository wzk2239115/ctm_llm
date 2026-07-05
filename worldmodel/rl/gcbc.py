"""Goal-Conditioned Behavioural Cloning (GCBC) — offline policy training on the
SAME expert dataset that trains the world model.

Mirrors stable-wm's GCBC baseline: pure supervised learning, no env interaction.
Episode-level BPTT through a memory backbone, minimising -log_prob(a_true) at
each timestep. All memory backbones (mlp/ctm/lstm/gru/transformer/flash) share
this trainer, so the comparison is purely about the memory mechanism.

This replaces the previous PPO-from-scratch + random-BC approach which was both
unfair (on-policy data != world-model data) and harmful (random actions are
noise, BC learns noise). Expert data contains goal-reaching actions, so GCBC
has a real signal — exactly like stable-wm.
"""
from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn as nn

from worldmodel.envs import make_env


def _obs_goal_tensor(infos, device):
    """Convert obs dict -> concat tensor (same as ppo._obs_goal_tensor)."""
    if "pixels" in infos:
        p = torch.as_tensor(np.asarray(infos["pixels"]), dtype=torch.float32, device=device)
        g = torch.as_tensor(np.asarray(infos["goal"]), dtype=torch.float32, device=device)
        return torch.cat([p, g], dim=1)
    state = torch.as_tensor(np.asarray(infos["state"]), dtype=torch.float32, device=device)
    goal = torch.as_tensor(np.asarray(infos["goal"]), dtype=torch.float32, device=device)
    return torch.cat([state, goal], dim=-1)


class GCBCTrainer:
    """Goal-conditioned BC on expert episodes.

    Episode-level BPTT: init backbone state, unfold the whole episode (state
    carries across timesteps), BC loss at each step. For image envs the encoder
    handles pixels; for state envs it handles vectors. The critic head is
    unused (pure BC).
    """

    def __init__(self, policy, env_name, device='cpu', lr=1e-3,
                 max_grad_norm=0.5, env_kw=None):
        self.policy = policy
        self.env_name = env_name
        self.env_kw = env_kw or {}
        self.device = device
        self.max_grad_norm = max_grad_norm
        self.opt = torch.optim.AdamW(policy.parameters(), lr=lr)

    def train(self, buffer, steps, batch_eps=4, max_seq_len=32, log_every=200):
        """TBPTT training: forward entire episode (state carries full history),
        backward every ``max_seq_len`` steps with gradient truncation.

        This fixes the long-episode problem (e.g. mountaincar 121 steps): the
        old code only learned the first ``max_seq_len`` steps, missing the
        crucial swing-up phase at the end. Now state accumulates the full
        episode context while gradients stay bounded.
        """
        episodes = buffer.episodes
        obs_key = 'pixels' if 'pixels' in episodes[0] else 'state'
        losses = []
        for step in range(steps):
            batch = [episodes[i] for i in np.random.randint(0, len(episodes), batch_eps)]
            T = min(len(ep['action']) for ep in batch)
            self.policy.init_state(batch_eps, self.device)
            self.opt.zero_grad()
            total_loss_val = 0.0
            chunk_loss = torch.tensor(0.0, device=self.device)
            for t in range(T):
                obs = np.stack([b[obs_key][t] for b in batch])
                goal = np.stack([b['goal'][t] for b in batch])
                x = _obs_goal_tensor({obs_key: obs, 'goal': goal}, self.device)
                dist, _ = self.policy(x)
                a_true = torch.as_tensor(
                    np.stack([b['action'][t] for b in batch]),
                    dtype=torch.float32, device=self.device)
                loss = -dist.log_prob(a_true).sum(-1).mean() / T
                chunk_loss = chunk_loss + loss
                total_loss_val += float(loss.item())
                if (t + 1) % max_seq_len == 0 or t == T - 1:
                    chunk_loss.backward()
                    nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                    self.opt.step()
                    self.opt.zero_grad()
                    self.policy.detach_state()  # TBPTT: truncate grad, keep state
                    chunk_loss = torch.tensor(0.0, device=self.device)
            losses.append(total_loss_val)
            if step % log_every == 0 or step == steps - 1:
                print(f"  [gcbc] step {step}/{steps} loss {np.mean(losses[-log_every:]):.4f}",
                      flush=True)
        return losses

    @torch.no_grad()
    def evaluate(self, episodes=20, seed=100):
        self.policy.eval()
        env = make_env(self.env_name, **self.env_kw)
        succ = 0
        for k in range(episodes):
            obs = env.reset(seed=seed + k)
            self.policy.init_state(1, self.device)
            done = False
            while not done:
                infos = {kk: np.asarray(v)[None] for kk, v in obs.items()}
                x = _obs_goal_tensor(infos, self.device)
                dist, _ = self.policy(x)
                a = dist.mean.cpu().numpy()[0]
                obs, r, term, trunc, info = env.step(
                    np.clip(a, env.action_space.low, env.action_space.high))
                done = bool(term or trunc)
                if term:
                    succ += 1
        self.policy.train()
        return 100.0 * succ / episodes

    @torch.no_grad()
    def evaluate_timed(self, episodes=12, seed=100, deadline_ms=None):
        """GCBC eval + per-step latency (for real-time comparison vs CEM)."""
        import time as _time
        self.policy.eval()
        env = make_env(self.env_name, **self.env_kw)
        succ = 0
        lats = []
        timeouts = 0
        total = 0
        for k in range(episodes):
            obs = env.reset(seed=seed + k)
            self.policy.init_state(1, self.device)
            done = False
            while not done:
                infos = {kk: np.asarray(v)[None] for kk, v in obs.items()}
                x = _obs_goal_tensor(infos, self.device)
                t0 = _time.perf_counter()
                dist, _ = self.policy(x)
                lat = (_time.perf_counter() - t0) * 1000.0
                lats.append(lat)
                total += 1
                a = dist.mean.cpu().numpy()[0]
                if deadline_ms is not None and lat > deadline_ms:
                    timeouts += 1
                    a = np.zeros_like(a)
                obs, r, term, trunc, info = env.step(
                    np.clip(a, env.action_space.low, env.action_space.high))
                done = bool(term or trunc)
                if term:
                    succ += 1
        self.policy.train()
        la = np.asarray(lats) if lats else np.asarray([1e-6])
        ml = float(la.mean())
        return {
            'success_rate': 100.0 * succ / episodes,
            'mean_latency_ms': ml,
            'p99_latency_ms': float(np.percentile(la, 99)),
            'throughput_hz': float(1000.0 / max(ml, 1e-6)),
            'timeout_rate': float(timeouts / max(total, 1)),
        }
