"""Route 1: CTM as an end-to-end policy (PPO), like the original CTM-RL.

The CTM backbone maintains a persistent ``(activated, trace)`` state across env
steps (reset on episode done) — the "continuous_state_trace" mechanism the
original CTM-RL used. Each step: ingest obs -> synapse+NLM -> feature -> actor
(Gaussian) / critic. Trained with PPO.

Recurrent-PPO correctness: during collect we snapshot the backbone state at
every step; during update we re-evaluate the policy from the stored state
(stateless) so the surrogate ratio is right and no minibatch pollutes the
carry. Pure torch/numpy, dep-free.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal

from worldmodel.envs import make_env
from worldmodel.wm.streaming import _NLM


class CTMPolicyBackbone(nn.Module):
    """Persistent NLM state backbone. step_stateless is a pure fn (no side
    effects) so update can re-evaluate from a stored state snapshot."""

    def __init__(self, input_dim, d_model=128, memory_length=8, nlm_hidden=8,
                 state_gate="none", emb=32):
        super().__init__()
        self.d_model = d_model
        self.memory_length = memory_length
        self.state_gate = state_gate
        self.input_embed = nn.Sequential(nn.Linear(input_dim, emb), nn.GELU(),
                                         nn.Linear(emb, emb))
        self.synapse = nn.Sequential(nn.Linear(emb + d_model, d_model * 2),
                                     nn.GLU(), nn.LayerNorm(d_model))
        self.trace_processor = _NLM(d_model, memory_length, nlm_hidden)
        self.feat = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model), nn.GELU())
        if state_gate == "gru":
            self.gate_linear = nn.Linear(d_model, d_model)
        self._activated = None
        self._trace = None

    def init_state(self, batch, device):
        z = torch.zeros(batch, self.d_model, device=device)
        self._activated = z
        self._trace = z.unsqueeze(-1).expand(-1, self.d_model, self.memory_length).clone()

    def detach_state(self):
        if self._activated is not None:
            self._activated = self._activated.detach()
            self._trace = self._trace.detach()

    def mask_reset(self, done_mask):
        if self._activated is None or done_mask is None:
            return
        m = done_mask.to(self._activated).unsqueeze(-1)
        self._activated = self._activated * (1 - m)
        self._trace = self._trace * (1 - m.unsqueeze(1))

    def step_stateless(self, x, activated, trace):
        """Pure: (x, activated, trace) -> (feat, new_activated, new_trace)."""
        lead = activated.shape[:-1]
        e = self.input_embed(x)
        inp = torch.cat([e, activated], dim=-1)
        new = self.synapse(inp).unsqueeze(-1)
        trace2 = torch.cat([trace[..., 1:], new], dim=-1)
        trace_flat = trace2.reshape(-1, self.d_model, self.memory_length)
        act_flat = self.trace_processor(trace_flat)
        if self.state_gate == "gru":
            old = activated.reshape(-1, self.d_model)
            g = torch.sigmoid(self.gate_linear(old))
            merged = g * act_flat + (1 - g) * old
            new_act = merged.reshape(*lead, self.d_model)
        else:
            new_act = act_flat.reshape(*lead, self.d_model)
        return self.feat(new_act), new_act, trace2

    def step(self, x):
        feat, new_act, new_trace = self.step_stateless(x, self._activated, self._trace)
        self._activated = new_act
        self._trace = new_trace
        return feat

    def snapshot(self):
        return (self._activated.detach(), self._trace.detach())


class CTMPolicyNetwork(nn.Module):
    def __init__(self, obs_dim, goal_dim, action_dim, d_model=128, memory_length=8,
                 state_gate="gru"):
        super().__init__()
        self.backbone = CTMPolicyBackbone(obs_dim + goal_dim, d_model=d_model,
                                          memory_length=memory_length, state_gate=state_gate)
        self.actor_mean = nn.Linear(d_model, action_dim)
        self.actor_logstd = nn.Parameter(torch.zeros(action_dim))
        self.critic = nn.Linear(d_model, 1)

    def init_state(self, batch, device):
        self.backbone.init_state(batch, device)

    def detach_state(self):
        self.backbone.detach_state()

    def mask_reset(self, done_mask):
        self.backbone.mask_reset(done_mask)

    def snapshot(self):
        return self.backbone.snapshot()

    def forward(self, x):
        feat = self.backbone.step(x)
        return self._heads(feat)

    def recompute(self, x, state):
        feat, _, _ = self.backbone.step_stateless(x, state[0], state[1])
        return self._heads(feat)

    def _heads(self, feat):
        mean = self.actor_mean(feat)
        std = self.actor_logstd.exp().expand_as(mean)
        return Normal(mean, std), self.critic(feat)


class MLPPolicyNetwork(nn.Module):
    """Markov MLP actor-critic (no carry). Same interface as CTM."""

    def __init__(self, obs_dim, goal_dim, action_dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(obs_dim + goal_dim, hidden), nn.Tanh(),
                                 nn.Linear(hidden, hidden), nn.Tanh())
        self.actor_mean = nn.Linear(hidden, action_dim)
        self.actor_logstd = nn.Parameter(torch.zeros(action_dim))
        self.critic = nn.Linear(hidden, 1)

    def init_state(self, batch, device):
        pass

    def detach_state(self):
        pass

    def mask_reset(self, done_mask):
        pass

    def snapshot(self):
        return None

    def forward(self, x):
        return self._heads(self.net(x))

    def recompute(self, x, state=None):
        return self._heads(self.net(x))

    def _heads(self, h):
        mean = self.actor_mean(h)
        std = self.actor_logstd.exp().expand_as(mean)
        return Normal(mean, std), self.critic(h)


def build_policy(kind, obs_dim, goal_dim, action_dim, d_model=128,
                 memory_length=8, state_gate="gru"):
    if kind == "ctm":
        return CTMPolicyNetwork(obs_dim, goal_dim, action_dim, d_model=d_model,
                                memory_length=memory_length, state_gate=state_gate)
    if kind == "mlp":
        return MLPPolicyNetwork(obs_dim, goal_dim, action_dim)
    raise KeyError(kind)


def _obs_goal_tensor(infos, device):
    if "pixels" in infos:
        p = torch.as_tensor(np.asarray(infos["pixels"]), dtype=torch.float32, device=device)
        g = torch.as_tensor(np.asarray(infos["goal"]), dtype=torch.float32, device=device)
        return torch.cat([p, g], dim=1)  # (B, 6, H, W): stacked obs+goal frames
    state = torch.as_tensor(np.asarray(infos["state"]), dtype=torch.float32, device=device)
    goal = torch.as_tensor(np.asarray(infos["goal"]), dtype=torch.float32, device=device)
    return torch.cat([state, goal], dim=-1)


class PPOTrainer:
    def __init__(self, env_name, policy, num_envs=8, num_steps=256,
                 gamma=0.99, gae_lambda=0.95, clip_coef=0.1, ent_coef=0.1,
                 vf_coef=0.5, update_epochs=10, num_minibatches=4, lr=3e-4,
                 max_grad_norm=0.5, device="cuda", env_kw=None):
        self.env_name = env_name
        self.env_kw = env_kw or {}
        self.num_envs = num_envs
        self.num_steps = num_steps
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_coef = clip_coef
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.update_epochs = update_epochs
        self.num_minibatches = num_minibatches
        self.device = device
        self.policy = policy.to(device)
        self.opt = torch.optim.AdamW(policy.parameters(), lr=lr)
        self.max_grad_norm = max_grad_norm

        sample = make_env(env_name, **self.env_kw)
        self.obs_dim = int(np.prod(sample.observation_space.shape))
        self.goal_dim = int(np.prod(sample.goal_space.shape))
        self.action_dim = int(np.prod(sample.action_space.shape))
        self.envs = [make_env(env_name, **self.env_kw) for _ in range(num_envs)]
        self._infos = self._reset_all(0)

    def _reset_all(self, seed):
        out = [self.envs[i].reset(seed=None if seed is None else seed + i)
               for i in range(self.num_envs)]
        return {k: np.stack([o[k] for o in out], 0) for k in out[0]}

    def _step_envs(self, actions):
        obs_list, rewards, dones = [], [], []
        for i, e in enumerate(self.envs):
            a = np.clip(np.asarray(actions[i]), e.action_space.low, e.action_space.high)
            o, r, term, trunc, info = e.step(a)
            done = bool(term or trunc)
            if done:
                rewards.append(float(r)); dones.append(True)
                self.envs[i] = make_env(self.env_name, **self.env_kw)
                obs_list.append(self.envs[i].reset(seed=int(np.random.randint(1 << 30))))
            else:
                rewards.append(float(r)); dones.append(False); obs_list.append(o)
        self._infos = {k: np.stack([o[k] for o in obs_list], 0) for k in obs_list[0]}
        return self._infos, np.asarray(rewards, np.float32), np.asarray(dones, bool)

    def collect_rollout(self):
        n, s, dev = self.num_envs, self.num_steps, self.device
        self.policy.init_state(n, dev)
        x_probe = _obs_goal_tensor(self._infos, dev)  # probe shape: image (n,6,H,W), state (n,ob_dim)
        buf = {
            "x": torch.zeros((s,) + x_probe.shape, device=dev),
            "action": torch.zeros(s, n, self.action_dim, device=dev),
            "logprob": torch.zeros(s, n, device=dev),
            "reward": torch.zeros(s, n, device=dev),
            "done": torch.zeros(s, n, device=dev),
            "value": torch.zeros(s, n, device=dev),
            "snap": [None] * s,   # backbone state snapshot per step (for recompute)
        }
        infos = self._infos
        for t in range(s):
            x = _obs_goal_tensor(infos, dev)
            buf["snap"][t] = self.policy.snapshot()
            with torch.no_grad():
                dist, value = self.policy(x)
                action = dist.sample()
                logprob = dist.log_prob(action).sum(-1)
            buf["x"][t] = x
            buf["action"][t] = action
            buf["logprob"][t] = logprob
            buf["value"][t] = value.squeeze(-1)
            infos, reward, done = self._step_envs(action.cpu().numpy())
            buf["reward"][t] = torch.as_tensor(reward, device=dev)
            buf["done"][t] = torch.as_tensor(done.astype(np.float32), device=dev)
            self.policy.mask_reset(torch.as_tensor(done, device=dev))
        with torch.no_grad():
            _, last_value = self.policy(_obs_goal_tensor(infos, dev))
            last_value = last_value.squeeze(-1)
        adv = torch.zeros_like(buf["reward"])
        lastgaelam = 0
        for t in reversed(range(s)):
            nextnt = 1.0 - buf["done"][t]
            nextv = last_value if t == s - 1 else buf["value"][t + 1]
            delta = buf["reward"][t] + self.gamma * nextv * nextnt - buf["value"][t]
            lastgaelam = delta + self.gamma * self.gae_lambda * nextnt * lastgaelam
            adv[t] = lastgaelam
        returns = adv + buf["value"]
        self.policy.detach_state()
        return buf, adv, returns

    def update(self, buf, adv, returns):
        b, n = self.num_steps, self.num_envs
        x = buf["x"]
        action = buf["action"]
        logp_old = buf["logprob"]
        snaps = buf["snap"]
        adv_flat = (adv - adv.mean()) / (adv.std() + 1e-8)
        ret_flat = returns
        flat = b * n
        idx = np.arange(flat)
        mb = max(1, flat // self.num_minibatches)
        stats = {"pg": 0.0, "v": 0.0, "ent": 0.0, "n": 0}
        for _ in range(self.update_epochs):
            np.random.shuffle(idx)
            for i0 in range(0, flat, mb):
                mbi = idx[i0:i0 + mb]
                ts = torch.as_tensor(mbi, device=self.device, dtype=torch.long)
                ti = ts // n          # timestep index
                ei = ts % n           # env index
                xb = x[ti, ei]
                ab = action[ti, ei]
                lpb = logp_old[ti, ei]
                advb = adv_flat[ti, ei]
                retb = ret_flat[ti, ei]
                # recompute from the snapshot taken at collect time (recurrent-correct).
                # state is a tuple of tensors (CTM:(act,trace), LSTM:(h,c), MLP:()).
                snap0 = snaps[0]
                snaps_b = []
                if snap0 is not None and len(snap0) > 0:
                    for (tt, ee) in zip(ti.tolist(), ei.tolist()):
                        full = snaps[int(tt)]
                        snaps_b.append(tuple(comp[int(ee)] for comp in full))
                dist, value = self._recompute_batch(xb, snaps_b)
                logp = dist.log_prob(ab).sum(-1)
                ent = dist.entropy().sum(-1).mean()
                ratio = (logp - lpb).exp()
                pg1 = -advb * ratio
                pg2 = -advb * ratio.clamp(1 - self.clip_coef, 1 + self.clip_coef)
                pg_loss = torch.max(pg1, pg2).mean()
                v_loss = ((value.squeeze(-1) - retb) ** 2).mean()
                loss = pg_loss + self.vf_coef * v_loss - self.ent_coef * ent
                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.opt.step()
                stats["pg"] += float(pg_loss); stats["v"] += float(v_loss)
                stats["ent"] += float(ent); stats["n"] += 1
        return {k: stats[k] / max(stats["n"], 1) for k in ("pg", "v", "ent")}

    def _recompute_batch(self, xb, snaps_b):
        # Generic: state is a tuple of tensors. Stack each position across the
        # minibatch. MLP (empty tuple) -> recompute with no state.
        if not snaps_b:
            return self.policy.recompute(xb, None)
        n = len(snaps_b[0])
        if n == 0:
            return self.policy.recompute(xb, ())
        state = tuple(torch.stack([s[i] for s in snaps_b], 0) for i in range(n))
        return self.policy.recompute(xb, state)

    def train(self, total_steps, log_iters=20, eval_episodes=12, seed=0):
        torch.manual_seed(seed); np.random.seed(seed)
        iters = total_steps // (self.num_envs * self.num_steps)
        self.policy.init_state(self.num_envs, self.device)
        self._infos = self._reset_all(seed)
        history = []
        global_step = 0
        log_every = max(1, iters // log_iters)
        for it in range(iters):
            buf, adv, returns = self.collect_rollout()
            stats = self.update(buf, adv, returns)
            global_step += self.num_envs * self.num_steps
            ep_ret = float(buf["reward"].sum().item())
            rec = {"step": global_step, "pg": stats["pg"], "v": stats["v"],
                   "ent": stats["ent"], "ep_return": ep_ret}
            if it % log_every == 0 or it == iters - 1:
                rec["eval_success"] = self.evaluate(episodes=eval_episodes)
                print(f"[ppo] {self.env_name} it{it}/{iters} step={global_step} "
                      f"ep_ret={ep_ret:.1f} pg={stats['pg']:.3f} v={stats['v']:.3f} "
                      f"ent={stats['ent']:.3f} eval_succ={rec['eval_success']:.1f}%", flush=True)
            history.append(rec)
        return history

    @torch.no_grad()
    def evaluate(self, episodes=12, seed=100):
        self.policy.init_state(1, self.device)
        env = make_env(self.env_name, **self.env_kw)
        obs = env.reset(seed=seed)
        succ = 0
        for k in range(episodes):
            done = False
            while not done:
                infos = {k2: np.asarray(v)[None] for k2, v in obs.items()}
                x = _obs_goal_tensor(infos, self.device)
                dist, _ = self.policy(x)
                a = dist.mean.cpu().numpy()[0]
                obs, r, term, trunc, info = env.step(np.clip(a, env.action_space.low, env.action_space.high))
                done = bool(term or trunc)
                self.policy.mask_reset(torch.tensor([done], device=self.device))
                if term:
                    succ += 1
            obs = env.reset(seed=seed + k + 1)
        return 100.0 * succ / episodes

    @torch.no_grad()
    def evaluate_timed(self, episodes=12, seed=100, deadline_ms=None):
        """evaluate + per-step latency/throughput/deadline-constrained success.
        Same real-time metrics as World.evaluate_timed, for the nn.Module policy
        (Flash Brain / RNN). If deadline_ms is set, a step whose forward exceeds
        it is a timeout (zero action used), so success drops under tight budgets."""
        import time as _time
        self.policy.init_state(1, self.device)
        env = make_env(self.env_name, **self.env_kw)
        obs = env.reset(seed=seed)
        succ = 0
        lats = []; timeouts = 0; total = 0
        for k in range(episodes):
            done = False
            while not done:
                infos = {k2: np.asarray(v)[None] for k2, v in obs.items()}
                x = _obs_goal_tensor(infos, self.device)
                t0 = _time.perf_counter()
                dist, _ = self.policy(x)
                a = dist.mean.cpu().numpy()[0]
                lat = (_time.perf_counter() - t0) * 1000.0
                lats.append(lat); total += 1
                if deadline_ms is not None and lat > deadline_ms:
                    timeouts += 1
                    a = np.zeros_like(a)
                obs, r, term, trunc, info = env.step(np.clip(a, env.action_space.low, env.action_space.high))
                done = bool(term or trunc)
                self.policy.mask_reset(torch.tensor([done], device=self.device))
                if term:
                    succ += 1
            obs = env.reset(seed=seed + k + 1)
        la = np.asarray(lats) if lats else np.asarray([1e-6])
        ml = float(la.mean())
        return {
            'success_rate': 100.0 * succ / episodes,
            'mean_latency_ms': ml,
            'p99_latency_ms': float(np.percentile(la, 99)),
            'throughput_hz': float(1000.0 / max(ml, 1e-6)),
            'timeout_rate': float(timeouts / max(total, 1)),
        }
