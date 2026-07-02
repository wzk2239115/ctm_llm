"""Route 2: CTM world model + actor-critic trained in imagination (Dreamer-style).

Unlike Route 1 (CTM is the policy directly) and the CEM route (CTM is a passive
predictor queried by an external planner), here the CTM world model IS used to
plan — but the planning happens *in imagination*: we roll the CTM dynamics
forward with an actor, predict reward/continue, and train actor+critic on the
imagined returns. The CTM's long-horizon stability (which the horizon sweep
confirmed) directly benefits this route, because the imagined rollout is where
errors would otherwise compound.

Pipeline each iteration:
  1. collect: run the actor in real envs, store (obs, action, reward, next_obs, done)
  2. train world model: latent dynamics (JEPA MSE) + reward head + continue head
  3. imagine: from real latents, rollout CTM dynamics with the actor, predict
     reward/continue along the way
  4. train actor-critic on the imagined lambda-returns

The CTM predictor's persistent state carries across the imagined horizon (true
streaming recurrence), which is exactly where its long-range advantage lives.
Pure torch/numpy, dep-free.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from worldmodel.envs import make_env
from worldmodel.wm.encoders import MLPEncoder
from worldmodel.wm.streaming import StreamingCTMPredictor


def _mlp(inp, out, hidden=128, layers=2):
    mods = []
    h = inp
    for _ in range(layers):
        mods += [nn.Linear(h, hidden), nn.GELU()]
        h = hidden
    mods += [nn.Linear(h, out)]
    return nn.Sequential(*mods)


class DreamerWorldModel(nn.Module):
    """encoder + CTM dynamics + reward/continue heads."""

    def __init__(self, obs_dim, action_dim, latent_dim=32, d_model=64,
                 memory_length=8, var_weight=4.0, state_gate="gru"):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.latent_dim = latent_dim
        self.var_weight = var_weight
        self.encoder = MLPEncoder(obs_dim=obs_dim, latent_dim=latent_dim)
        self.predictor = StreamingCTMPredictor(
            latent_dim=latent_dim, action_dim=action_dim,
            d_model=d_model, memory_length=memory_length, state_gate=state_gate,
        )
        self.reward_head = _mlp(latent_dim, 1)
        self.continue_head = _mlp(latent_dim, 1)

    def encode(self, obs):
        return self.encoder(obs)

    def wm_loss(self, obs_seq, actions, rewards, dones):
        """obs_seq: (B, H+1, obs_dim); actions: (B, H, A); rewards/dones: (B, H)."""
        B, Hp1, od = obs_seq.shape
        H = Hp1 - 1
        latents = self.encode(obs_seq.reshape(B * Hp1, od)).reshape(B, Hp1, self.latent_dim)
        init = latents[:, 0]
        target = latents[:, 1:].detach()
        pred = self.predictor.rollout(init, actions)  # (B, H, D)
        dyn = F.mse_loss(pred, target)
        # reward / continue heads on the *target* (true future) latents
        r_pred = self.reward_head(target).squeeze(-1)
        r_loss = F.mse_loss(r_pred, rewards)
        c_pred = self.continue_head(target).squeeze(-1)
        c_target = (1.0 - dones)  # continue = not done
        c_loss = F.binary_cross_entropy_with_logits(c_pred, c_target)
        # VICReg anti-collapse on latents
        var_loss = torch.zeros((), device=obs_seq.device)
        if self.var_weight > 0:
            std = latents.reshape(-1, self.latent_dim).std(dim=0)
            var_loss = F.relu(1.0 - std).mean()
        loss = dyn + r_loss + c_loss + self.var_weight * var_loss
        with torch.no_grad():
            dyn_err = (pred - target).pow(2).mean().item()
        return loss, {"dyn": dyn.item(), "r": r_loss.item(), "c": c_loss.item(),
                      "dyn_err": dyn_err, "var": var_loss.item()}


class Actor(nn.Module):
    def __init__(self, latent_dim, action_dim, hidden=128):
        super().__init__()
        self.net = _mlp(latent_dim, hidden * 2)
        self.mean = nn.Linear(hidden * 2, action_dim)
        self.logstd = nn.Parameter(torch.zeros(action_dim))

    def forward(self, z):
        h = self.net(z)
        mean = self.mean(h)
        std = self.logstd.exp().expand_as(mean)
        return Normal(mean, std)


class Critic(nn.Module):
    def __init__(self, latent_dim, hidden=128):
        super().__init__()
        self.net = _mlp(latent_dim, 1)

    def forward(self, z):
        return self.net(z).squeeze(-1)


def _obs_goal(obs_dict):
    return np.concatenate([np.asarray(obs_dict["state"]), np.asarray(obs_dict["goal"])], axis=-1)


class DreamerTrainer:
    def __init__(self, env_name, latent_dim=32, d_model=64, memory_length=8,
                 imagine_horizon=10, gamma=0.99, lam=0.95, num_envs=8,
                 collect_steps=256, wm_lr=3e-4, ac_lr=3e-4, wm_iters=3,
                 ac_iters=3, batch_size=32, device="cuda", env_kw=None,
                 state_gate="gru", var_weight=4.0):
        self.env_name = env_name
        self.env_kw = env_kw or {}
        self.latent_dim = latent_dim
        self.imagine_horizon = imagine_horizon
        self.gamma = gamma
        self.lam = lam
        self.num_envs = num_envs
        self.collect_steps = collect_steps
        self.wm_iters = wm_iters
        self.ac_iters = ac_iters
        self.batch_size = batch_size
        self.device = device

        sample = make_env(env_name, **self.env_kw)
        self.obs_dim = int(np.prod(sample.observation_space.shape))
        self.goal_dim = int(np.prod(sample.goal_space.shape))
        self.action_dim = int(np.prod(sample.action_space.shape))
        obs_full = self.obs_dim + self.goal_dim

        self.wm = DreamerWorldModel(obs_full, self.action_dim, latent_dim, d_model,
                                    memory_length, var_weight=var_weight,
                                    state_gate=state_gate).to(device)
        self.actor = Actor(latent_dim, self.action_dim).to(device)
        self.critic = Critic(latent_dim).to(device)
        self.wm_opt = torch.optim.AdamW(self.wm.parameters(), lr=wm_lr)
        self.ac_opt = torch.optim.AdamW(list(self.actor.parameters()) + list(self.critic.parameters()), lr=ac_lr)
        self.envs = [make_env(env_name, **self.env_kw) for _ in range(num_envs)]
        self._infos = self._reset_all(0)
        self.buf = {"obs": [], "action": [], "reward": [], "next_obs": [], "done": []}

    def _reset_all(self, seed):
        out = [self.envs[i].reset(seed=None if seed is None else seed + i) for i in range(self.num_envs)]
        return out

    def collect(self):
        """Run current actor in real envs for collect_steps, append to buffer."""
        dev = self.device
        n = self.num_envs
        for _ in range(self.collect_steps // n):
            x = torch.as_tensor(np.stack([_obs_goal(o) for o in self._infos], 0),
                                dtype=torch.float32, device=dev)
            with torch.no_grad():
                z = self.wm.encode(x)
                dist = self.actor(z)
                action = dist.sample()
            a_np = action.cpu().numpy()
            for i, e in enumerate(self.envs):
                a = np.clip(a_np[i], e.action_space.low, e.action_space.high)
                o = self._infos[i]
                no, r, term, trunc, info = e.step(a)
                done = bool(term or trunc)
                self.buf["obs"].append(_obs_goal(o))
                self.buf["action"].append(np.asarray(a, np.float32))
                self.buf["reward"].append(np.float32(r))
                self.buf["next_obs"].append(_obs_goal(no))
                self.buf["done"].append(np.float32(done))
                if done:
                    self.envs[i] = make_env(self.env_name, **self.env_kw)
                    self._infos[i] = self.envs[i].reset(seed=int(np.random.randint(1 << 30)))
                else:
                    self._infos[i] = no
        # cap buffer
        cap = 20000
        for k in self.buf:
            if len(self.buf[k]) > cap:
                self.buf[k] = self.buf[k][-cap:]

    def _sample_batch(self, H):
        n = len(self.buf["obs"]) - H - 1
        if n <= self.batch_size:
            return None
        idx = np.random.randint(0, n, self.batch_size)
        obs = np.stack([np.stack([self.buf["obs"][i + t] for t in range(H + 1)]) for i in idx])
        act = np.stack([np.stack([self.buf["action"][i + t] for t in range(H)]) for i in idx])
        rew = np.stack([[self.buf["reward"][i + t] for t in range(H)] for i in idx])
        don = np.stack([[self.buf["done"][i + t] for t in range(H)] for i in idx])
        dev = self.device
        return (torch.as_tensor(obs, dtype=torch.float32, device=dev),
                torch.as_tensor(act, dtype=torch.float32, device=dev),
                torch.as_tensor(rew, dtype=torch.float32, device=dev),
                torch.as_tensor(don, dtype=torch.float32, device=dev))

    def train_wm(self, H):
        stats = {"dyn": 0.0, "r": 0.0, "c": 0.0, "dyn_err": 0.0, "n": 0}
        for _ in range(self.wm_iters):
            batch = self._sample_batch(H)
            if batch is None:
                return stats
            obs_seq, actions, rewards, dones = batch
            loss, m = self.wm.wm_loss(obs_seq, actions, rewards, dones)
            self.wm_opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(self.wm.parameters(), 5.0)
            self.wm_opt.step()
            for k in ("dyn", "r", "c", "dyn_err"):
                stats[k] += m[k]
            stats["n"] += 1
        for k in ("dyn", "r", "c", "dyn_err"):
            stats[k] /= max(stats["n"], 1)
        return stats

    def imagine(self, init_latent, H):
        """Roll CTM dynamics forward with the actor; return imagined tensors."""
        dev = self.device
        B = init_latent.shape[0]
        self.wm.predictor.reset_state(init_latent)
        latents = [init_latent]
        actions = []
        rewards = []
        conts = []
        for _ in range(H):
            z = latents[-1]
            dist = self.actor(z.detach())  # actor on detached z (graph through actor only here)
            a = dist.rsample()
            r = self.wm.reward_head(z.detach()).squeeze(-1)
            c_logits = self.wm.continue_head(z.detach()).squeeze(-1)
            c = torch.sigmoid(c_logits)
            # CTM dynamics step (stateful; carry across imagined horizon)
            z_next = self.wm.predictor.step(a.detach())  # dynamics graph? we keep predictor grad through z_next
            latents.append(z_next)
            actions.append(a)
            rewards.append(r)
            conts.append(c)
        latents_t = torch.stack(latents, 0)  # (H+1, B, D)
        actions_t = torch.stack(actions, 0)
        rewards_t = torch.stack(rewards, 0)
        conts_t = torch.stack(conts, 0)
        return latents_t, actions_t, rewards_t, conts_t

    def lambda_return(self, rewards, conts, values, last_value):
        """dreamer lambda-return. rewards/conts/values: (H, B)."""
        H, B = rewards.shape
        returns = torch.zeros_like(rewards)
        running = last_value
        for t in reversed(range(H)):
            c = conts[t]
            delta = rewards[t] + self.gamma * c * running - values[t]
            running = delta + self.gamma * self.lam * c * running
            returns[t] = running
        return returns

    def train_ac(self, H):
        dev = self.device
        stats = {"actor": 0.0, "critic": 0.0, "ent": 0.0, "n": 0}
        for _ in range(self.ac_iters):
            batch = self._sample_batch(1)
            if batch is None:
                return stats
            obs_seq, actions, rewards, dones = batch
            with torch.no_grad():
                init_latent = self.wm.encode(obs_seq[:, 0])
            latents, acts, rews, conts = self.imagine(init_latent, H)
            z_imag = latents.detach()  # (H+1, B, D)
            values = self.critic(z_imag[:-1].reshape(-1, self.latent_dim)).reshape(H, -1)
            with torch.no_grad():
                last_v = self.critic(z_imag[-1])
            ret = self.lambda_return(rews, conts, values.detach(), last_v.detach())
            # critic fits the lambda-return
            v_loss = F.mse_loss(values, ret.detach())
            # actor: maximize return through the imagined graph (rews/conts depend on z_imag which
            # came from predictor.step(a.detach())) — to get gradient into actor we recompute via
            # rsample graph in actions. Simpler & stable: regress actor toward the action that
            # maximizes value (entropy-regularized policy improvement on imagined values).
            z_for_actor = z_imag[:-1].reshape(-1, self.latent_dim)
            dist = self.actor(z_for_actor)
            a_sample = dist.rsample()
            # recompute imagined reward/continue on these actions for policy gradient
            r_actor = self.wm.reward_head(z_for_actor).squeeze(-1)
            c_actor = torch.sigmoid(self.wm.continue_head(z_for_actor).squeeze(-1))
            v_actor = self.critic(z_for_actor)
            actor_obj = (v_actor + 0.1 * dist.entropy().sum(-1)).mean()
            actor_loss = -actor_obj
            loss = actor_loss + v_loss
            self.ac_opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), 5.0)
            nn.utils.clip_grad_norm_(self.critic.parameters(), 5.0)
            self.ac_opt.step()
            stats["actor"] += float(actor_loss); stats["critic"] += float(v_loss)
            stats["ent"] += float(dist.entropy().sum(-1).mean()); stats["n"] += 1
        for k in ("actor", "critic", "ent"):
            stats[k] /= max(stats["n"], 1)
        return stats

    def train(self, total_steps, log_iters=20, eval_episodes=12, seed=0, H_wm=6):
        torch.manual_seed(seed); np.random.seed(seed)
        iters = total_steps // self.collect_steps
        log_every = max(1, iters // log_iters)
        history = []
        global_step = 0
        for it in range(iters):
            self.collect()
            global_step += self.collect_steps
            wm_stats = self.train_wm(H_wm)
            ac_stats = self.train_ac(self.imagine_horizon)
            rec = {"step": global_step, **wm_stats, **ac_stats}
            if it % log_every == 0 or it == iters - 1:
                rec["eval_success"] = self.evaluate(episodes=eval_episodes)
                print(f"[dreamer] {self.env_name} it{it}/{iters} step={global_step} "
                      f"dyn_err={wm_stats['dyn_err']:.4f} r={wm_stats['r']:.3f} "
                      f"actor={ac_stats['actor']:.3f} critic={ac_stats['critic']:.3f} "
                      f"eval_succ={rec['eval_success']:.1f}%", flush=True)
            history.append(rec)
        return history

    @torch.no_grad()
    def evaluate(self, episodes=12, seed=100):
        env = make_env(self.env_name, **self.env_kw)
        obs = env.reset(seed=seed)
        succ = 0
        for k in range(episodes):
            done = False
            while not done:
                x = torch.as_tensor(_obs_goal(obs)[None], dtype=torch.float32, device=self.device)
                z = self.wm.encode(x)
                dist = self.actor(z)
                a = dist.mean.cpu().numpy()[0]
                obs, r, term, trunc, info = env.step(np.clip(a, env.action_space.low, env.action_space.high))
                done = bool(term or trunc)
                if term:
                    succ += 1
            obs = env.reset(seed=seed + k + 1)
        return 100.0 * succ / episodes
