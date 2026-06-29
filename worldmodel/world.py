"""World — drives a policy through a pool of (vectorized) envs.

Mirrors the collect/train/evaluate shape of stable-worldmodel's World but with a
dependency-free Env interface (see :mod:`worldmodel.envs`). ``num_envs`` envs
are stepped together each tick (a plain Python loop; fine for the small/short
episodes used here). ``infos`` is a dict of stacked arrays keyed like the env
observation (``'state'`` / ``'pixels'`` plus ``'goal'``), shape
``(num_envs, ...)``.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Callable

import numpy as np

from worldmodel.data import ReplayBuffer


def _stack_obs(obs_list: list[dict]) -> dict:
    out: dict[str, np.ndarray] = {}
    for k in obs_list[0]:
        out[k] = np.stack([o[k] for o in obs_list], axis=0)
    return out


class World:
    def __init__(self, env_fn: Callable, num_envs: int = 1):
        self.envs = [env_fn() for _ in range(num_envs)]
        self._policy = None
        self.infos: dict = {}
        self.rewards = None
        self.terminateds = None
        self.truncateds = None

    @property
    def num_envs(self) -> int:
        return len(self.envs)

    @property
    def observation_space(self):
        return self.envs[0].observation_space

    @property
    def action_space(self):
        return self.envs[0].action_space

    @property
    def goal_space(self):
        return self.envs[0].goal_space

    def close(self) -> None:
        pass

    def set_policy(self, policy) -> None:
        self._policy = policy
        policy.set_env(self)

    def reset(self, seed: int | None = None) -> dict:
        obs_list = []
        for i, env in enumerate(self.envs):
            s = None if seed is None else seed + i
            obs_list.append(env.reset(seed=s))
        self.infos = _stack_obs(obs_list)
        self.terminateds = np.zeros(self.num_envs, dtype=bool)
        self.truncateds = np.zeros(self.num_envs, dtype=bool)
        return self.infos

    def step(self, actions: np.ndarray):
        obs_list, rewards, terms, truncs, infos = [], [], [], [], []
        for i, env in enumerate(self.envs):
            a = np.asarray(actions[i])
            # Frozen (already-done) envs hold their last obs until reset.
            if self.terminateds[i] or self.truncateds[i]:
                obs_list.append(self._frozen_obs(i))
                rewards.append(0.0); terms.append(True); truncs.append(True)
                infos.append({}); continue
            o, r, term, trunc, info = env.step(a)
            obs_list.append(o); rewards.append(float(r))
            terms.append(bool(term)); truncs.append(bool(trunc)); infos.append(info)
        self.infos = _stack_obs(obs_list)
        self.rewards = np.asarray(rewards, dtype=np.float32)
        self.terminateds = np.asarray(terms, dtype=bool)
        self.truncateds = np.asarray(truncs, dtype=bool)
        return self.infos, self.rewards, self.terminateds, self.truncateds

    def _frozen_obs(self, i: int) -> dict:
        return {k: v[i] for k, v in self.infos.items()}

    def _reset_env(self, i: int, seed: int | None) -> dict:
        return self.envs[i].reset(seed=seed)

    # -- public workflows --
    def collect(
        self,
        buffer: ReplayBuffer,
        episodes: int,
        seed: int | None = None,
    ) -> ReplayBuffer:
        if self._policy is None:
            raise RuntimeError('Call set_policy(...) before collect().')
        self.reset(seed=seed)
        acc = [defaultdict(list) for _ in range(self.num_envs)]
        # Seed each env's episode with its initial (reset) frame.
        for i in range(self.num_envs):
            for k, v in self.infos.items():
                acc[i][k].append(np.asarray(v[i]))
        done_count = 0
        next_seed = (seed + self.num_envs) if seed is not None else None

        def _seed_acc(i, s):
            obs = self._reset_env(i, s)
            for k, v in obs.items():
                self.infos[k][i] = v
            self.terminateds[i] = False
            self.truncateds[i] = False
            new_acc = defaultdict(list)
            for k, v in obs.items():
                new_acc[k].append(np.asarray(v))
            acc[i] = new_acc

        while done_count < episodes:
            actions = self._policy.get_action(self.infos)
            self.step(actions)
            for i in range(self.num_envs):
                # Append the action and the resulting frame to env i's episode.
                acc[i]['action'].append(np.asarray(actions[i]))
                for k, v in self.infos.items():
                    acc[i][k].append(np.asarray(v[i]))
                if self.terminateds[i] or self.truncateds[i]:
                    ep = {k: np.stack(v, axis=0) for k, v in acc[i].items()}
                    buffer.add_episode(ep)
                    done_count += 1
                    if done_count >= episodes:
                        return buffer
                    _seed_acc(i, None if next_seed is None else next_seed + done_count)
        return buffer

    def evaluate(self, episodes: int, seed: int | None = None) -> dict:
        if self._policy is None:
            raise RuntimeError('Call set_policy(...) before evaluate().')
        self.reset(seed=seed)
        successes = np.zeros(episodes, dtype=bool)
        seeds_used = np.zeros(episodes, dtype=np.int64)
        per_env_success = np.zeros(self.num_envs, dtype=bool)
        ep_idx = 0
        next_seed = (seed + self.num_envs) if seed is not None else None
        while ep_idx < episodes:
            actions = self._policy.get_action(self.infos)
            self.step(actions)
            for i in range(self.num_envs):
                if self.terminateds[i]:
                    per_env_success[i] = True
                if self.terminateds[i] or self.truncateds[i]:
                    if ep_idx < episodes:
                        successes[ep_idx] = per_env_success[i]
                        seeds_used[ep_idx] = int(
                            seed if seed is None else seed + i
                        )
                        ep_idx += 1
                    per_env_success[i] = False
                    s = None if next_seed is None else next_seed + ep_idx
                    obs = self._reset_env(i, s)
                    for k, v in obs.items():
                        self.infos[k][i] = v
                    self.terminateds[i] = False
                    self.truncateds[i] = False
        return {
            'success_rate': float(successes[:ep_idx].mean()) * 100.0,
            'episode_successes': successes[:ep_idx],
            'seeds': seeds_used[:ep_idx],
        }
