"""Policies: random, and model-based MPC via a solver.

:func:`WorldModelPolicy` is the model-based planner: every env step it asks the
solver (e.g. CEM) for an action sequence that minimises the world model's cost
to the goal, and executes the first action. (Re-planning every step is the
simplest correct MPC; action reuse / receding-horizon buffering is left as a
future optimisation.)

Keeping the planner this thin means the *same* policy/solver can drive any
:class:`~worldmodel.protocols.Costable` world model — CTM or JEPA — for a clean
head-to-head comparison.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch

from worldmodel.solver import Solver


@dataclass(frozen=True)
class PlanConfig:
    """MPC planning configuration.

    Attributes:
        horizon: how many steps ahead the solver plans.
        receding_horizon: env steps to execute before re-planning (1 = replan
            every step). Currently only ``1`` is exercised by WorldModelPolicy.
    """

    horizon: int
    receding_horizon: int = 1

    @property
    def plan_len(self) -> int:
        return self.horizon


class BasePolicy:
    def __init__(self, seed: int | None = None):
        self.env = None
        self.seed = seed

    def set_env(self, env) -> None:
        self.env = env

    def get_action(self, info_dict: dict) -> np.ndarray:
        raise NotImplementedError


class RandomPolicy(BasePolicy):
    """Samples an action uniformly from the action space, per env."""

    def get_action(self, info_dict: dict) -> np.ndarray:
        n = self.env.num_envs
        return np.stack([self.env.action_space.sample() for _ in range(n)])


class WorldModelPolicy(BasePolicy):
    """Plans an action with a solver + world model, every step."""

    def __init__(
        self,
        solver: Solver,
        config: PlanConfig,
        transform: dict[str, Callable] | None = None,
    ):
        super().__init__()
        self.solver = solver
        self.cfg = config
        self.transform = transform or {}

    def set_env(self, env) -> None:
        self.env = env
        self.solver.configure(
            action_space=env.action_space,
            n_envs=env.num_envs,
            config=self.cfg,
        )

    def _to_tensor(self, info_dict: dict) -> dict:
        out = {}
        dev = self.solver.device
        for k, v in info_dict.items():
            t = torch.as_tensor(np.asarray(v))
            if t.dtype == torch.float64:
                t = t.float()
            out[k] = t.to(dev)
        return out

    def get_action(self, info_dict: dict) -> np.ndarray:
        info = self._to_tensor(info_dict)
        out = self.solver(info)
        actions = out['actions']  # (num_envs, horizon, action_dim)
        first = actions[:, 0, :].numpy().astype(np.float32)
        return first


class ExpertPolicy(BasePolicy):
    """Per-env heuristic oracle for offline data collection.

    Mirrors stable-wm's approach: each env gets a scripted/oracle policy that
    produces goal-reaching trajectories (TwoRoom=scripted navigation, PushT=
    heuristic push, DMC=pretrained SAC). This is the data source for BOTH
    world-model training AND policy baselines (GCBC / GCIQL).

    Action noise (default 0.1) adds diversity so the dataset is not a single
    deterministic trajectory — same as stable-wm ExpertPolicy(action_noise=...).

    Accesses env internal state via ``self.env.envs[i]`` — the expert is a
    teacher with full state access, NOT a policy being evaluated.
    """

    def __init__(self, seed: int | None = None, noise: float = 0.1):
        super().__init__(seed=seed)
        self.noise = float(noise)
        self._rng = np.random.default_rng(seed)

    def get_action(self, infos: dict) -> np.ndarray:
        n = self.env.num_envs
        acts = [self._expert_action(self.env.envs[i]) for i in range(n)]
        a = np.stack(acts).astype(np.float32)
        if self.noise > 0:
            a = a + self._rng.normal(0, self.noise, a.shape).astype(np.float32)
        return np.clip(a, -1.0, 1.0)

    def _expert_action(self, env) -> np.ndarray:
        name = type(env).__name__
        fn = getattr(self, '_act_' + name, None)
        if fn is None:
            return env.action_space.sample().astype(np.float32)
        return np.asarray(fn(env), dtype=np.float32)

    # --- point reach ---
    def _act_PointStateReach(self, env):
        return self._go_to(env._agent, env._goal, gain=5.0)

    def _act_PointImageReach(self, env):
        return self._go_to(env._agent, env._goal, gain=5.0)

    # --- two-room: navigate via door ---
    def _act_TwoRoomNav(self, env):
        same_room = (env._agent[0] < env.wall_x) == (env._goal[0] < env.wall_x)
        if same_room:
            target = env._goal
        else:
            door = np.array([env.wall_x, env._door_y])
            at_door = abs(env._agent[0] - env.wall_x) < 0.1 and \
                abs(env._agent[1] - env._door_y) < env._door_h
            target = env._goal if at_door else door
        return self._go_to(env._agent, target, gain=8.0)

    # --- cartpole: balance pole + move to goal_x ---
    def _act_CartPole(self, env):
        x, xd, th, thd = env._s
        u = 0.6 * (env._goal_x - x) - 1.0 * xd + 10.0 * th + 2.5 * thd
        return np.array([np.clip(u, -1.0, 1.0)], dtype=np.float32)

    # --- pendulum: energy-based swing-up + PD stabilize ---
    def _act_Pendulum(self, env):
        th, thd, gth = env._th, env._thd, env._goal_th
        d = (th - gth + np.pi) % (2 * np.pi) - np.pi
        if abs(d) < 1.0 and abs(thd) < 3.0:
            u = -8.0 * d - 2.0 * thd
        else:
            # energy-based pump: drive energy toward goal energy
            E = 0.5 * thd ** 2 + env.g * (np.cos(th) + 1.0)
            E_goal = env.g * (np.cos(gth) + 1.0)
            u = 2.0 * np.sign(thd) if E < E_goal else -2.0 * np.sign(thd)
        return np.array([np.clip(u / 2.0, -1.0, 1.0)], dtype=np.float32)

    # --- reacher: rotate joints toward goal direction ---
    def _act_Reacher(self, env):
        ee = env._ee(env._a1, env._a2)
        diff = env._goal - ee
        angle_to_goal = np.arctan2(diff[1], diff[0])
        base_err = angle_to_goal - env._a1
        u1 = 1.5 * np.sin(base_err)
        u2 = 1.0 * np.sin(base_err - env._a2)
        return np.clip([u1, u2], -1.0, 1.0).astype(np.float32)

    # --- mountaincar: energy pumping ---
    def _act_MountainCar(self, env):
        pos, vel = env._s
        u = 1.0 if vel >= 0 else -1.0
        return np.array([u], dtype=np.float32)

    # --- acrobot: energy pumping (coupled joint velocity) ---
    def _act_Acrobot(self, env):
        th1, th2, dth1, dth2 = env._s
        u = float(np.sign(dth2 + 0.5 * dth1)) if abs(dth2 + 0.5 * dth1) > 0.01 else 1.0
        return np.array([u], dtype=np.float32)

    # --- swimmer: turn toward goal + bang-bang thrust (keep angles in sin>0 zone) ---
    def _act_Swimmer(self, env):
        to_goal = env._goal - env._pos
        goal_angle = np.arctan2(to_goal[1], to_goal[0])
        d_head = (goal_angle - env._angles[0] + np.pi) % (2 * np.pi) - np.pi
        a0 = float(np.clip(2.5 * d_head / np.pi, -1.0, 1.0))
        acts = [a0]
        for j in range(1, env.n_links):
            ang = float(env._angles[j])
            if ang < 0.5:
                acts.append(1.0)
            elif ang > 2.5:
                acts.append(-1.0)
            else:
                acts.append(0.7)
        return np.clip(acts, -1.0, 1.0).astype(np.float32)

    # --- pusht: agent behind block, push toward goal ---
    def _act_PushT(self, env):
        return self._push_heuristic(env._agent, env._t[:2], env._goal[:2], behind=0.05)

    # --- cube: same push heuristic ---
    def _act_CubePush(self, env):
        return self._push_heuristic(env._agent, env._block[:2], env._goal[:2], behind=0.06)

    def _push_heuristic(self, agent, block, goal, behind=0.05):
        diff_bg = block - goal
        dist_bg = float(np.linalg.norm(diff_bg))
        if dist_bg < 0.02:
            return np.zeros(2, dtype=np.float32)
        push_dir = diff_bg / dist_bg
        push_point = block + push_dir * behind
        return self._go_to(agent, push_point, gain=5.0)

    # --- fetchpush: ee behind block, push toward goal ---
    def _act_FetchPush(self, env):
        ee = env._ee(env._a1, env._a2)
        block, goal = env._block, env._goal
        diff_bg = block - goal
        dist_bg = float(np.linalg.norm(diff_bg))
        if dist_bg < 0.05:
            target = block
        else:
            target = block + (diff_bg / dist_bg) * 0.08
        diff_ee = target - ee
        u1 = 2.5 * diff_ee[0]
        u2 = 2.5 * diff_ee[1]
        return np.clip([u1, u2], -1.0, 1.0).astype(np.float32)

    @staticmethod
    def _go_to(agent, goal, gain=5.0):
        return np.clip((np.asarray(goal) - np.asarray(agent)) * gain, -1.0, 1.0).astype(np.float32)
